"""Read-only CPU verification of the fixed final RL endpoint and actual ledger."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, collections, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'r2/src'))
from r2.pilots.bench_three_mechanisms_v1 import common as C


def audit_calls(rows,forwards,backwards,updates):
    counts=collections.Counter(row['event'] for row in rows)
    for name,expected in (('FORWARD',forwards),('BACKWARD',backwards),('OPTIMIZER_STEP',updates)):
        if counts[name+'_START']!=expected or counts[name+'_RETURN']!=expected:
            raise ValueError('Incomplete/mismatched '+name+' actual ledger')
    if counts['BACKWARD_START']>40 or counts['OPTIMIZER_STEP_START']>1:
        raise ValueError('Per-opportunity budget exceeded')
    return dict(counts)


def check_identity():
    r=C.read(C.OUT/'rl/registration.json')
    if r['root_registration_sha256']!=C.sha(C.REGISTRATION) or r['schedule_sha256']!=C.sha(C.OUT/'data/rl_schedule.json'):
        raise ValueError('Registered input changed')
    for path,digest in r['source_code_sha256'].items():
        if C.sha(ROOT/path)!=digest:raise ValueError('Frozen code changed: '+path)
    return r


def verify(write=False):
    reg=check_identity();schedule=C.read(C.OUT/'data/rl_schedule.json')['opportunities']
    work=C.OUT/'rl/training';done=C.read(work/'complete.json')
    if (done['status']!='COMPLETE' or done['version']!=reg['schema'] or done['opportunities']!=16 or
        done['policy_trajectories']!=64 or done['reference_trajectories']!=64 or done['D0_forwards']!=0 or
        done['registration_sha256']!=reg['root_registration_sha256'] or not done['base_unchanged'] or not done['no_best_selection']):
        raise ValueError('Fixed final completion contract failed')
    import numpy as np
    import torch
    from meshflow_control.checkpoints import state_hash
    from r2.pilots.bench_three_mechanisms_v1 import reward as Q
    if torch.cuda.is_initialized():raise RuntimeError('Verification must remain CPU')
    previous=None;updates=forwards=backwards=0;per=[];files={}
    for i,item in enumerate(schedule):
        folder=work/f'opportunity_{i:03d}';r=C.read(folder/'complete.json');update=r['update']
        if (r['status']!='COMPLETE' or r['opportunity']!=i or r['parent_id']!=item['case']['parent_id'] or
            r['condition_id']!=item['case']['condition_id'] or r['N']!=item['case']['N'] or r['K']!=item['case']['K']):
            raise ValueError('Opportunity identity mismatch')
        rows=[json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines() if line.strip()]
        calls=audit_calls(rows,r['generator_forwards'],r['backwards'],update['optimizer_steps'])
        if any(row['opportunity']!=i for row in rows):raise ValueError('Call opportunity mismatch')
        rewards=[];sample_forwards=0;active_steps=0
        for spec in item['trajectories']:
            td=folder/f"trajectory_{spec['index']}";s=C.read(td/'complete.json')
            if s['noise_seed']!=spec['noise_seed'] or s['policy_noise_seed']!=spec['policy_noise_seed']:
                raise ValueError('Random identity changed')
            n,k=r['N'],r['K'];noise=C.arrays(td/'initial_noise.npz')['z']
            expected=np.random.RandomState(spec['noise_seed']).standard_normal((n-k,3,3)).astype(np.float32)
            if not np.array_equal(noise,expected):raise ValueError('Initial Gaussian differs')
            ref=C.read(td/'reference.json');mask=np.asarray(ref['edit_mask'],bool)
            if mask[:k].any() or mask.sum()>8:raise ValueError('Local mask invalid')
            bundle=C.arrays(td/'transition_noise.npz');eps=Q.draw_policy_noise(spec['policy_noise_seed'],n,mask)
            if not np.array_equal(bundle['epsilon'],eps) or not np.array_equal(bundle['edit_mask'],mask):raise ValueError('Transition array changed')
            base=C.arrays(td/'base_raw.npz')['raw'];raw=C.arrays(td/'policy_raw.npz')['raw']
            b=C.arrays(td/'buffer.npz');reward=C.read(td/'reward.json')
            if ref['reference_RAW_sha256']!=Q.array_hash(base) or reward['current_RAW_sha256']!=Q.array_hash(raw):raise ValueError('RAW/reward identity mismatch')
            if reward['reference_RAW_sha256']!=Q.array_hash(base) or raw[:k].tobytes()!=base[:k].tobytes():raise ValueError('Reference/C mismatch')
            if reward['reward']!=s['reward']:raise ValueError('Saved reward differs')
            rewards.append(float(reward['reward']))
            steps=list(range(30,40)) if mask.any() else []
            if b['steps'].tolist()!=steps or not np.array_equal(b['edit_mask'],mask):raise ValueError('Actual transition buffer mask/window')
            active_steps+=len(steps)
            for j in range(len(steps)):
                mu=b['old_mean'][j].reshape(n,3,3);act=b['state_after'][j].reshape(n,3,3)
                expected_action=np.where(mask[:,None,None],mu+np.float32(.005)*eps[j],mu)
                if not np.array_equal(act,expected_action):raise ValueError('Saved action is not registered Gaussian transition')
                if not np.array_equal(mu[~mask],b['base_mean'][j].reshape(n,3,3)[~mask]):raise ValueError('Outside mean not frozen base')
            for name in ('base_trajectory.npz','policy_trajectory.npz'):
                trajectory=C.arrays(td/name)['state']
                if trajectory.shape!=(51,n,3,3) or any(x[:k].tobytes()!=base[:k].tobytes() for x in trajectory):raise ValueError('C trajectory contract')
            ba,pa=s['base_audit'],s['policy_audit']
            if ba['generator_forwards']!=50 or pa['generator_forwards']!=50+len(steps):raise ValueError('Rollout call count')
            if pa['transition_noise_sha256']!=hashlib.sha256(eps.tobytes()).hexdigest():raise ValueError('Transition audit hash')
            sample_forwards+=ba['generator_forwards']+pa['generator_forwards']
            for f in td.iterdir():
                if f.is_file():files[str(f)]=C.sha(f)
        values=np.asarray(rewards,np.float64)
        advantage=values-(values.sum()-values)/3
        if not np.array_equal(advantage,np.asarray(update['advantages'])) or update['rewards']!=rewards:
            raise ValueError('RLOO advantage or reward ledger differs')
        active=any(a!=0 and C.read(folder/f'trajectory_{j}/complete.json')['mask_faces']>0 for j,a in enumerate(advantage))
        if update['optimizer_steps']!=int(active) or update['skipped']==active:raise ValueError('Update skip rule mismatch')
        if update['backwards']!=(active_steps if active else 0) or update['generator_forwards']!=(2*active_steps if active else 0):raise ValueError('Update/diagnostic calls differ')
        if r['generator_forwards']!=sample_forwards+update['generator_forwards'] or r['backwards']!=update['backwards']:
            raise ValueError('Opportunity computation total differs')
        if update['old_logp_error']!=0:raise ValueError('On-policy replay failed')
        path=folder/'checkpoint.pt'
        if C.sha(path)!=r['checkpoint_sha256']:raise ValueError('Checkpoint file hash mismatch')
        ck=torch.load(path,map_location='cpu',weights_only=True)
        updates+=update['optimizer_steps'];forwards+=r['generator_forwards'];backwards+=r['backwards']
        if (ck['schema']!=reg['schema'] or ck['completed_opportunities']!=i+1 or ck['update_in_progress'] or
            ck['optimizer_updates']!=updates or ck['base_file_sha256']!=reg['source_SHA256'] or
            ck['registration_sha256']!=reg['root_registration_sha256'] or ck['schedule_sha256']!=reg['schedule_sha256']):
            raise ValueError('Checkpoint lineage/cursor mismatch')
        ah=state_hash(ck['adapters'])
        if ah!=ck['adapters_sha256'] or ah!=update['adapter_after'] or (previous is not None and previous!=update['adapter_before']):
            raise ValueError('Adapter state/continuation hash mismatch')
        if not active and update['adapter_before']!=ah:raise ValueError('Skipped optimizer changed adapter')
        for state in ck['optimizer_state']['state'].values():
            if float(state['step'])!=updates:raise ValueError('AdamW update clock differs')
        previous=ah
        per.append(dict(opportunity=i,parent_id=r['parent_id'],N=r['N'],K=r['K'],rewards=rewards,
            optimizer_updates=update['optimizer_steps'],forwards=r['generator_forwards'],backwards=r['backwards'],
            calls=calls,checkpoint_sha256=r['checkpoint_sha256']))
        for f in (folder/'complete.json',folder/'calls.jsonl',folder/'checkpoint.pt'):files[str(f)]=C.sha(f)
        del ck
    final=work/'opportunity_015/checkpoint.pt'
    if (updates!=done['optimizer_updates'] or done['skipped_opportunities']!=16-updates or
        forwards!=done['generator_forwards'] or backwards!=done['backwards'] or forwards>8320 or backwards>640 or
        C.resolve(done['final_checkpoint'])!=final or C.sha(final)!=done['final_checkpoint_sha256']):raise ValueError('Final ledger totals/hash differ')
    if torch.cuda.is_initialized():raise RuntimeError('Unexpected CUDA in CPU verification')
    files[str(work/'complete.json')]=C.sha(work/'complete.json')
    result=dict(status='PASS',opportunities=16,reference_trajectories=64,policy_trajectories=64,
        generator_forwards=forwards,backwards=backwards,optimizer_updates=updates,skipped_opportunities=16-updates,
        D0_forwards=0,model_forwards_in_verifier=0,CUDA_initialized=False,
        final_checkpoint_sha256=C.sha(final),root_registration_sha256=reg['root_registration_sha256'],
        rl_registration_sha256=C.sha(C.OUT/'rl/registration.json'),schedule_sha256=reg['schedule_sha256'],
        opportunities_audit=per,files=files)
    if write:C.write(C.OUT/'rl/verification.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--write',action='store_true')
    args=p.parse_args()
    if args.check:
        check_identity()
        audit_calls([dict(event=e) for e in ('FORWARD_START','FORWARD_RETURN','BACKWARD_START','BACKWARD_RETURN','OPTIMIZER_STEP_START','OPTIMIZER_STEP_RETURN')],1,1,1)
        try:audit_calls([dict(event='FORWARD_START')],1,0,0)
        except ValueError:pass
        else:raise AssertionError('Incomplete call accepted')
        result=dict(status='PASS',scope='CPU interface and ledger primitive only; final run not yet verified',model_calls=0)
    else:result=verify(args.write)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','opportunities_audit')}))


if __name__=='__main__':main()
