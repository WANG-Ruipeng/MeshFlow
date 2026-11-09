"""Owned policy process; all requested samples are preregistered."""
from __future__ import annotations
from . import common as W
from . import data as D
import argparse, sys, time
sys.path.insert(0,str(D.ROOT/'r2/src'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--round',type=int);p.add_argument('--evaluation',choices=['BASE','NFT','SELF_FM'])
    a=p.parse_args()
    if (a.round is None)==(a.evaluation is None):raise ValueError('Exactly one registered stage')
    m=D.read(D.OUT/'inputs/manifest.json')
    if a.round is not None:
        if not 0<=a.round<16:raise ValueError('Round budget')
        qs=[q for q in m['training_requests'] if q['round_index']==a.round]
        stage=D.OUT/f'training/round_{a.round:02d}';label='NFT'
        checkpoint=None if a.round==0 else D.OUT/f'checkpoints/NFT/round{a.round-1:02d}.pt'
    else:
        label=a.evaluation;qs=m['evaluation_requests'];stage=D.OUT/'evaluation'/label
        if label=='BASE':qs=[q for q in qs if q['evaluation_noise_partition']=='FRESH']
        if len(qs)!=(12 if label=='BASE' else 36):raise RuntimeError('Evaluation budget mismatch')
        checkpoint=None if label=='BASE' else D.OUT/f'checkpoints/{label}/round15.pt'
    if checkpoint is not None and not checkpoint.exists():raise FileNotFoundError(checkpoint)
    with W.Phase(stage/'sampling_worker',dict(stage='sampling',label=label,round=a.round,requests=len(qs))) as phase:
        import torch
        torch.set_num_threads(2)
        from .model import load_policy
        from .learning import load_checkpoint
        from .sample import sample_request
        from meshflow_control.checkpoints import state_hash
        b=m['source_binding']
        policy=load_policy(D.resolve(b['path']),expected_sha256=b['sha256'],device='cuda')
        if checkpoint is not None:load_checkpoint(checkpoint,policy,restore_random_state=False)
        identity=dict(base_sha256=b['sha256'],adapter_sha256=state_hash(policy.adapter_state_dict()),round_index=a.round,
                      checkpoint_path=str(checkpoint) if checkpoint else None,checkpoint_sha256=D.sha(checkpoint) if checkpoint else None)
        budget=W.Budget(stage/'sampling_events.jsonl',len(qs),policy if a.round==0 else None)
        results=[]
        with policy.activate('current'):
            for i,q in enumerate(qs):
                rec=sample_request(policy,q,budget,stage/'raw'/q['request_id'],label=label,model_identity=identity)
                results.append(rec)
                print(f'SAMPLE {i+1}/{len(qs)} {q["parent_id"][:6]} n{q["candidate"]} {rec["sampling"]["seconds"]:.2f}s',flush=True)
        D.write(stage/'raw_index.json',dict(status='COMPLETE',label=label,round=a.round,model_identity=identity,receipts=results,
            actual_new_rollouts=budget.rollouts,actual_sampling_forwards=budget.forwards))

if __name__=='__main__':main()
