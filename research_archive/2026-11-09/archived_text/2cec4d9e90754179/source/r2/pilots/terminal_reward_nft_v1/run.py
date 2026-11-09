"""Recoverable bounded entrance. Missing actual review stops at a review gate."""
from __future__ import annotations
from . import common as W
from . import data as D
import argparse, json, os, subprocess, sys, time, math

OUT=D.OUT

def source_verify():
    lock=OUT/'execution_source_lock.json'
    if not lock.exists():raise RuntimeError('Freeze execution source lock before dispatch')
    for name,digest in D.read(lock)['hashes'].items():
        if D.sha(name)!=digest:raise RuntimeError('Execution source changed: '+name)

def gpu_values(snapshot):
    row=snapshot['gpu']['stdout'].splitlines()[0].split(',')
    return dict(total=float(row[-4]),used=float(row[-3]),free=float(row[-2]),util=float(row[-1]))

def launch(name,module,args):
    source_verify();root=OUT/'dispatch'/name
    if root.exists():raise RuntimeError('Prior dispatch exists; completed science stages must be reused; failed attempts need explicit recovery: '+name)
    snap=W.resource_snapshot();g=gpu_values(snap)
    is_gpu=module.endswith(('sample_worker','train_worker','denoise_worker')) and '--join' not in args
    if is_gpu and g['free']<max(2048,.1*g['total']):raise RuntimeError('Insufficient memory headroom for dispatch')
    D.write(root/'admission.json',dict(resources=snap,command=[module]+args,reason='Authorized bounded stage; no unrelated process termination'))
    log=(root/'stdout.log').open('x',encoding='utf8')
    cmd=[sys.executable,'-u','-B','-m',module]+args
    proc=subprocess.Popen(cmd,cwd=D.ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
    print('LAUNCH',name,'PID',proc.pid,flush=True)
    return dict(name=name,proc=proc,log=log,path=root,initial=snap,started=time.perf_counter())

def wait(job):
    code=job['proc'].wait();job['log'].close()
    D.write(job['path']/'exit.json',dict(returncode=code,seconds=time.perf_counter()-job['started'],resources=W.resource_snapshot()))
    print('EXIT',job['name'],code,flush=True)
    if code:raise RuntimeError('Worker failed, no silent retry: '+job['name']+' see '+str(job['path']/'stdout.log'))

def single(name,module,args):wait(launch(name,module,args))

def wait_group(jobs):
    errors=[]
    for job in jobs:
        try:wait(job)
        except BaseException as exc:errors.append(repr(exc))
    if errors:raise RuntimeError('Workers settled; no further dispatch: '+str(errors))

def completed_checkpoint(ri,arm):
    path=OUT/f'checkpoints/{arm}/round{ri:02d}.pt'
    if not path.exists():return False
    receipt=path.with_suffix('.json');phase=OUT/f'training/round_{ri:02d}/train_{arm}/completion.json'
    if not phase.exists():
        recovered=OUT/f'training/round_{ri:02d}/train_{arm}/completion_recovery.json'
        alternate=OUT/f'training/round_{ri:02d}/train_{arm}_recovery01/completion.json'
        phase=recovered if recovered.exists() else alternate
    if not receipt.exists() or not phase.exists():raise RuntimeError('Incomplete checkpoint transaction; no reuse/retrain: '+str(path))
    r=D.read(receipt)
    if D.sha(path)!=r['sha256'] or r['step']!=(ri+1)*16 or r['round_index']!=ri or D.read(phase)['status']!='COMPLETE':raise RuntimeError('Checkpoint receipt/clock mismatch')
    return True

def complete_raw_index(path):
    if not path.exists():return False
    v=D.read(path)
    if v['status']!='COMPLETE':raise RuntimeError('Incomplete raw index')
    for r in v['receipts']:
        if r['status']!='COMPLETE' or D.sha(r['raw_path'])!=r['raw_sha256']:raise RuntimeError('RAW cache changed')
    return True

def sample(ri):
    stage=OUT/f'training/round_{ri:02d}'
    if complete_raw_index(stage/'raw_index.json'):return
    # Round0 was the measured serial task. Following rounds use registered shards.
    if ri==0:
        single('round00_sample','r2.pilots.terminal_reward_nft_v1.sample_worker',['--round','0']);return
    jobs=[]
    first=launch(f'round{ri:02d}_sample0','r2.pilots.terminal_reward_nft_v1.sample_worker',['--round',str(ri),'--workers','2','--worker','0']);jobs.append(first)
    # Wait for a real rollout receipt: includes model load and representative execution.
    while first['proc'].poll() is None:
        if len(list((stage/'raw').glob('*/receipt.json')))>=1:break
        time.sleep(1)
    if first['proc'].poll() is not None and first['proc'].returncode!=0:wait(first)
    snap=W.resource_snapshot();g=gpu_values(snap);initial=gpu_values(first['initial'])
    # Prior serial process increment was about1GiB; current observed delta retains context.
    peak=max(1100.,g['used']-initial['used']);slots=math.floor(max(0,g['free']-max(2048,.1*g['total']))/(1.2*peak))
    parallel=slots>=1 and g['util']<85
    D.write(stage/'sampling_dispatch.json',dict(workers_requested=2,overlap_admitted=parallel,resources=snap,observed_increment_MiB=peak,incremental_slots=slots,reason='Below85pct compute and measured memory headroom' if parallel else 'Compute saturation or memory headroom: shards run sequentially'))
    if not parallel:wait(first);jobs=[]
    jobs.append(launch(f'round{ri:02d}_sample1','r2.pilots.terminal_reward_nft_v1.sample_worker',['--round',str(ri),'--workers','2','--worker','1']))
    wait_group(jobs)
    single(f'round{ri:02d}_samplejoin','r2.pilots.terminal_reward_nft_v1.sample_worker',['--round',str(ri),'--workers','2','--join'])

def prepare(ri):
    stage=OUT/f'training/round_{ri:02d}';sample(ri)
    if not (stage/'proposal_index.json').exists():single(f'round{ri:02d}_D0','r2.pilots.terminal_reward_nft_v1.denoise_worker',['--round',str(ri)])
    if not (stage/'gallery/manifest.json').exists():single(f'round{ri:02d}_CPU','r2.pilots.terminal_reward_nft_v1.cpu_worker',['--round',str(ri),'--stage','all','--workers','2'])
    if not (stage/'reviews_actual.json').exists():
        print('WAIT_ACTUAL_REVIEW',ri,str(stage/'gallery/index.html'),flush=True);return False
    if not (stage/'training_stream.json').exists():single(f'round{ri:02d}_reward','r2.pilots.terminal_reward_nft_v1.prepare_rewards',['--round',str(ri)])
    gate=D.read(stage/'reward_gate.json')
    if gate['status']!='PASS':print('STOP_REWARD_BLOCKED',flush=True);return False
    return True

def learn(ri):
    stage=OUT/f'training/round_{ri:02d}'
    seal=D.read(stage/'learning_input_seal.json')
    if seal['status']!='PASS' or D.read(stage/'reward_gate.json')['status']!='PASS':raise RuntimeError('Actual reward gate not passed')
    for path,digest in seal['hashes'].items():
        if D.sha(path)!=digest:raise RuntimeError('Learning input seal changed: '+path)
    needed=[a for a in ('NFT','SELF_FM') if not completed_checkpoint(ri,a)]
    if not needed:return
    first_arm=needed[0];first=launch(f'round{ri:02d}_train_{first_arm}','r2.pilots.terminal_reward_nft_v1.train_worker',['--round',str(ri),'--arm',first_arm]);jobs=[first]
    if len(needed)>1:
        events=stage/f'train_{first_arm}/events.jsonl'
        while first['proc'].poll() is None:
            if events.exists() and 'UPDATE_COMPLETE' in events.read_text():break
            time.sleep(1)
        if first['proc'].poll() is not None and first['proc'].returncode!=0:wait(first)
        observed=events.exists() and 'UPDATE_COMPLETE' in events.read_text()
        if not observed:raise RuntimeError('No completed authorized first update; stop dispatch')
        snap=W.resource_snapshot();g=gpu_values(snap);before=gpu_values(first['initial'])
        peak=max(1500.,g['used']-before['used']);slots=math.floor(max(0,g['free']-max(2048,.1*g['total']))/(1.2*peak));parallel=slots>=1 and g['util']<85
        D.write(stage/'training_dispatch.json',dict(workers_requested=2,overlap_admitted=parallel,resources=snap,first_actual_update_observed=observed,observed_increment_MiB=peak,incremental_slots=slots,reason='Independent arms, shared frozen data, separate optimizer/RNG; compute/resource gate applied',timing='Concurrent wall time is not isolated benchmark'))
        if not parallel:wait(first);jobs=[]
        second=needed[1];jobs.append(launch(f'round{ri:02d}_train_{second}','r2.pilots.terminal_reward_nft_v1.train_worker',['--round',str(ri),'--arm',second]))
    wait_group(jobs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pipeline','prepare','learn','status'],default='pipeline');p.add_argument('--round',type=int,default=0);a=p.parse_args()
    if a.stage=='status':
        for ri in range(16):
            stage=OUT/f'training/round_{ri:02d}';print(ri,{'RAW':(stage/'raw_index.json').exists(),'POST':(stage/'proposal_index.json').exists(),'GALLERY':(stage/'gallery/manifest.json').exists(),'REVIEW':(stage/'reviews_actual.json').exists(),'NFT':(OUT/f'checkpoints/NFT/round{ri:02d}.pt').exists(),'SELF':(OUT/f'checkpoints/SELF_FM/round{ri:02d}.pt').exists()})
        return
    if a.stage=='prepare':prepare(a.round);return
    if a.stage=='learn':learn(a.round);return
    for ri in range(a.round,16):
        if not prepare(ri):return
        learn(ri)
    print('TRAINING_COMPLETE_FINAL_EVALUATION_REQUIRED',flush=True)

if __name__=='__main__':main()
