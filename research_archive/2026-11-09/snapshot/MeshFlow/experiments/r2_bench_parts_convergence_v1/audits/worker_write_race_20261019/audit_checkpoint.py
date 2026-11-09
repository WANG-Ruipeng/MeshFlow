"""Minimal CPU checkpoint/tail audit and immutable incident snapshots."""
import collections
import hashlib
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import random
from pathlib import Path
import shutil
import time

AUDIT=Path(__file__).resolve().parent
ROOT=AUDIT.parents[1]
REPO=ROOT.parents[1]
OUT=ROOT/'training/COMMON'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,value):
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def resolved(p):return Path(str(p).replace('E:\\MeshFlow','/mnt/e/MeshFlow').replace('E:/MeshFlow','/mnt/e/MeshFlow'))

def main():
    start=time.monotonic();snapshots={}
    for name in ('status.json','failure.json','rolling.json','ledger.jsonl','exposure.jsonl','training_log.jsonl'):
        source=OUT/name;target=AUDIT/('interrupted.'+name)
        if not target.exists():shutil.copy2(source,target)
        assert sha(source)==sha(target)
        snapshots[str(source)]=dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size)
    for name in ('status.json','failure.json','identity.json'):
        source=ROOT/'dispatch'/name;target=AUDIT/('dispatch.'+name)
        if not target.exists():shutil.copy2(source,target)
        assert sha(source)==sha(target)
        snapshots[str(source)]=dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size)
    source=ROOT/'registration.json';target=AUDIT/'registration.json'
    if not target.exists():shutil.copy2(source,target)
    assert sha(source)==sha(target)
    snapshots[str(source)]=dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size)
    identity=read(AUDIT/'dispatch.identity.json')
    for text,expected in identity['code_hashes'].items():
        p=resolved(text)
        if p.name=='run_all.py':
            candidate=p
            if sha(candidate)!=expected:
                matches=list((ROOT/'audits').rglob(p.name+'.after.'+expected))
                assert len(matches)==1, 'Cannot locate exact pre-incident dispatcher'
                candidate=matches[0]
            target=AUDIT/(p.name+'.before')
            if not target.exists():shutil.copy2(candidate,target)
            assert sha(target)==expected
    receipt=read(OUT/'rolling.json');assert receipt['step']==8750
    expected='2c37e0cd7c0c7054f1bea215e9e6634e96be1f52838f8cf9ab11cad1f33a936c'
    assert receipt['sha256']==expected and sha(OUT/'rolling.pt')==expected
    archive=AUDIT/'interrupted.rolling.pt'
    if not archive.exists():shutil.copy2(OUT/'rolling.pt',archive)
    assert sha(archive)==expected
    snapshots[str(OUT/'rolling.pt')]=dict(path=str(archive),sha256=expected,bytes=archive.stat().st_size)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    p=torch.load(archive,map_location='cpu',weights_only=True,mmap=True)
    assert p['schema']=='bench_parts_training_v1' and p['arm']=='COMMON' and p['EMA'] is False
    assert p['step']==p['global_step']==p['stage_step']==8750
    assert p['head_state'] is None and p['head_optimizer'] is None and p['head_step']==0
    h=hashlib.sha256()
    for key,tensor in sorted(p['generator_state'].items()):
        tensor=tensor.detach().cpu().contiguous()
        h.update(key.encode());h.update(str((tuple(tensor.shape),tensor.dtype)).encode());h.update(tensor.view(torch.uint8).numpy().tobytes())
    assert h.hexdigest()==receipt['generator_state_sha256']==p['generator_state_sha256']
    assert len(p['optimizer']['state'])==194
    assert {float(s['step']) for s in p['optimizer']['state'].values()}=={8750.}
    assert all('exp_avg' in s and 'exp_avg_sq' in s for s in p['optimizer']['state'].values())
    rng=p['rng'];random.Random().setstate(rng['python']);n=rng['numpy']
    np.random.RandomState().set_state((n['kind'],np.asarray(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached']))
    torch.Generator(device='cpu').set_state(rng['torch_cpu'])
    assert len(rng['torch_cuda'])==1 and rng['torch_cuda'][0].numel()>0 and rng['torch_cuda'][0].dtype==torch.uint8
    reg=read(ROOT/'registration.json');assert p['registration_sha256']==sha(ROOT/'registration.json')
    for text,expected_hash in reg['training_code_hashes'].items():assert sha(resolved(text))==expected_hash
    schedule_path=resolved(reg['input_schedule']);assert sha(schedule_path)==reg['input_schedule_sha256']
    schedule=read(schedule_path)['steps'];rows={}
    for name in ('training_log.jsonl','ledger.jsonl','exposure.jsonl'):
        rows[name]=[json.loads(line) for line in (AUDIT/('interrupted.'+name)).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    metrics=rows['training_log.jsonl'];ledger=rows['ledger.jsonl'];exposure=rows['exposure.jsonl']
    assert [r['step'] for r in metrics]==list(range(1,8887))
    counts=collections.Counter(r['event'] for r in ledger if r.get('step',0)>8750)
    expected_counts=dict(UPDATE_STARTED=136,FORWARD_STARTED=1088,FORWARD_COMPLETE=1088,BACKWARD_STARTED=1088,
        BACKWARD_COMPLETE=1088,OPTIMIZER_STARTED=136,OPTIMIZER_COMPLETE=136,UPDATE_COMPLETE=136)
    assert dict(counts)==expected_counts,counts
    for event in ('UPDATE_STARTED','UPDATE_COMPLETE','OPTIMIZER_STARTED','OPTIMIZER_COMPLETE'):
        assert [r['step'] for r in ledger if r.get('step',0)>8750 and r['event']==event]==list(range(8751,8887))
    tail=[r for r in exposure if r['step']>8750]
    assert len(tail)==1088 and len({(r['step'],r['micro']) for r in exposure})==len(exposure)==8886*8
    for row in tail:
        entry=schedule[row['step']-1];assert entry['step']==row['step']
        item=entry['samples'][row['micro']]
        for key,value in item.items():
            if key in row:assert row[key]==value,(row['step'],key)
    assert read(OUT/'status.json')['step']==8886 and read(OUT/'status.json')['status']=='BLOCKED'
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_STEP8750_PAYLOAD_AND_TAIL_ACCOUNTING',checkpoint=str(OUT/'rolling.pt'),
        checkpoint_archive=str(archive),checkpoint_file_sha256=receipt['sha256'],generator_state_sha256=h.hexdigest(),
        checkpoint_step=8750,generator_parameters=sum(t.numel() for t in p['generator_state'].values()),
        optimizer_state_count=194,optimizer_step=8750,saved_RNG_CPU_states_loadable=True,saved_CUDA_RNG_present=True,
        CUDA_initialized=False,model_forwards=0,registration_sha256=sha(ROOT/'registration.json'),
        training_code_sha256=sha(REPO/'r2/pilots/bench_parts_convergence_v1/train.py'),
        input_schedule_sha256=sha(schedule_path),all1088_tail_inputs_match_schedule=True,cases_reaudited=0,
        tail=dict(completed_updates=136,first_step=8751,last_step=8886,partial_updates=0,forward_completed=1088,
            backward_completed=1088,optimizer_updates=136,head_updates=0,new_fixed_probe_forwards=0,exposure_rows=1088,event_counts=dict(counts)),
        snapshots=snapshots,starts_training=False,production_files_modified=False,seconds=time.monotonic()-start)
    write(AUDIT/'checkpoint_audit.json',result)
    print(json.dumps(dict(status=result['status'],tail=result['tail'],receipt=str(AUDIT/'checkpoint_audit.json'),seconds=result['seconds'])))

if __name__=='__main__':main()
