"""Advance only the registered second-half CPU OT cache, without a model.

The exact existing HybridStream._sample/_mapping path is used.  Nothing is
written for the first4000 samples, and no scientific parameter is changed.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'native_t1/runs/chair_capped500_gradbudget_v1'
FIRST, END = 4000, 8000


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):h.update(b)
    return h.hexdigest()


def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,p)


def append(p, value):
    value=dict(value,pid=os.getpid(),time_unix=time.time())
    with Path(p).open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def resources():
    commands={
        'cpu_count':['nproc'],
        'memory_bytes':['free','-b'],
        'processes':['ps','-eo','pid,ppid,pcpu,pmem,rss,comm','--sort=-pcpu'],
        'gpu':['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],
        'gpu_processes':['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],
    }
    result={}
    for name,command in commands.items():
        p=subprocess.run(command,capture_output=True,text=True,check=False)
        result[name]=dict(command=command,returncode=p.returncode,
            stdout='\n'.join(p.stdout.splitlines()[:15]) if name=='processes' else p.stdout,stderr=p.stderr)
    result.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_workers=1,threads_per_worker=2,
        GPU_workers_launched=0,decision='One CPU-only worker; disjoint registered sample indices4000..7999; abundant measured RAM/cores; existing GPU worker is untouched.')
    return result


def run(out=OUT):
    out=Path(out).resolve()
    if out!=OUT.resolve():raise ValueError('Only isolated authorized output')
    folder=out/'registration/future_ot_precompute';folder.mkdir(parents=True,exist_ok=True)
    receipt_path=folder/'receipt.json'
    if receipt_path.exists():
        r=read(receipt_path)
        if r['status']!='PASS':raise RuntimeError('Prior failure is retained; no automatic retry')
        assert r['source_sha256']==sha(__file__)
        for item in r['cache_files']:
            assert sha(item['path'])==item['sha256']
        return dict(status='CACHE_VERIFIED',prior_receipt_sha256=sha(receipt_path),new_OT_calls=0,model_calls=0)
    if (folder/'failure.json').exists() or (folder/'running.json').exists():
        raise RuntimeError('Prior unfinished CPU job requires explicit engineering review; no retry')
    write(folder/'resources_before.json',resources())
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    from .stream import HybridStream,SEED
    source=ROOT/'native_t1/runs/control_mvp48_v1'
    chair=ROOT/'native_t1/runs/chair_domain_control_v1'
    protocol=read(out/'protocol.json')
    assert protocol['training']['seed']==SEED
    for p,h in protocol['scientific_sources'].items():assert sha(p)==h
    plan_path=out/'streams/repeat/train_plan.json'
    plan=read(plan_path);registration=read(out/'registration/training_plan.json')
    assert sha(plan_path)==registration['sha256'] and len(plan['records'])==END
    selected=plan['records'][FIRST:END]
    assert [x['sample_index'] for x in selected]==list(range(FIRST,END))
    assert {x['pilot_step'] for x in selected}==set(range(501,1001))
    # The caller explicitly dispatches while GPU workers are in the first half.
    # Check journal evidence rather than assuming current worker timing.
    for arm in ('T','L','G'):
        p=out/'training'/(arm+'_events.jsonl')
        if p.exists():
            for line in p.read_text().splitlines():
                row=json.loads(line)
                if row.get('event')=='INPUT_BEGIN' and row['pilot_step']>=501:
                    raise RuntimeError('Future range already dispatched on GPU: no concurrent cache writer')
    stream=HybridStream(source,chair,out/'streams/repeat','T',seed=SEED)
    cache=stream.cache_dir
    for index in range(FIRST,END):
        for attempt in cache.glob(f'{index:04d}_*.attempt.json'):
            target=attempt.with_name(attempt.name[:-len('.attempt.json')]+'.npz')
            if not target.exists():
                raise RuntimeError('Unfinished registered OT attempt; stop without reOT: '+str(attempt))
    identity=dict(sample_start=FIRST,sample_end_exclusive=END,steps=[501,1000],samples=4000,
        plan_sha256=sha(plan_path),stream_source_sha256=sha(Path(__file__).with_name('stream.py')),
        protocol_sha256=sha(out/'protocol.json'),seed=SEED,source_sha256=sha(__file__),
        cache_directory=str(cache),GPU_lock_acquired=False,CPU_workers=1,threads=2,
        math='Unmodified HybridStream._sample and _mapping including original matching costs and exact source/corner maps')
    write(folder/'running.json',identity)
    tick=time.perf_counter();cache_files=[];input_rows=folder/'registered_inputs.jsonl'
    try:
        for record in selected:
            index=record['sample_index']
            append(folder/'events.jsonl',dict(event='REGISTERED_SAMPLE_BEGIN',sample_index=index,pilot_step=record['pilot_step']))
            sample=stream._sample(record)
            cp=cache/(f'{index:04d}_'+sample['ot_cache_key']+'.npz')
            item=dict(sample_index=index,pilot_step=record['pilot_step'],uid=record['uid'],task_id=record['task_id'],
                path=str(cp),sha256=sha(cp),cache_status=sample['ot_cache_status'],
                ot_cache_key=sample['ot_cache_key'],shared_input_sha256=sample['shared_input_sha256'],
                coupled_input_sha256=sample['coupled_input_sha256'],free_coordinate_count=sample['free_coordinate_count'])
            append(input_rows,item);cache_files.append(item)
            append(folder/'events.jsonl',dict(event='REGISTERED_SAMPLE_RETURN',sample_index=index,
                pilot_step=record['pilot_step'],cache_status=sample['ot_cache_status'],counts=dict(stream.counts)))
            if (index+1)%100==0:
                print('FUTURE_OT',index+1-FIRST,'/4000',dict(stream.counts),'seconds',round(time.perf_counter()-tick,3),flush=True)
        assert stream.counts['actual_OT_attempts']==stream.counts['actual_OT_returns']
        assert stream.counts['actual_OT_returns']+stream.counts['OT_cache_hits']==4000
        assert not torch.cuda.is_initialized()
        receipt=dict(status='PASS',**identity,counts=dict(stream.counts),cache_files=cache_files,
            registered_inputs_sha256=sha(input_rows),events_sha256=sha(folder/'events.jsonl'),
            wall_seconds=time.perf_counter()-tick,model_calls=0,model_loads=0,model_generated_rollouts=0,
            new_feedback_matching=0,new_scientific_samples=0,
            noise_scope='Only already-registered primary sample Z/epsilon are materialized; no new evaluation/feedback noise.',
            first_half_cache_writes=0,training_cursor_modified=False,CUDA_initialized=False)
        write(receipt_path,receipt);write(folder/'resources_after.json',resources())
        return {k:v for k,v in receipt.items() if k!='cache_files'}
    except BaseException as exc:
        write(folder/'failure.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),
            completed=len(cache_files),counts=dict(stream.counts),wall_seconds=time.perf_counter()-tick,
            source_sha256=sha(__file__),model_calls=0,no_retry=True))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));args=p.parse_args()
    print(json.dumps(run(args.out),sort_keys=True,indent=2))
