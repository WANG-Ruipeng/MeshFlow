"""Owned sampling shards; root resource admission and CPU-only join are explicit.

Each shard owns a separate model, Phase and ledger. Frozen request seeds and
Euler50 arithmetic are unchanged. Root measures and incrementally launches.
"""
from __future__ import annotations
from . import common as W
from . import data as D
import argparse, sys, json
sys.path.insert(0,str(D.ROOT/'r2/src'))


def select_stage(m,round_index=None,evaluation=None):
    if (round_index is None)==(evaluation is None):raise ValueError('Exactly one registered stage')
    if round_index is not None:
        if not 0<=round_index<16:raise ValueError('Round budget')
        qs=[q for q in m['training_requests'] if q['round_index']==round_index]
        if len(qs)!=32:raise RuntimeError('Training sampling budget mismatch')
        stage=D.OUT/f'training/round_{round_index:02d}';label='NFT'
        checkpoint=None if round_index==0 else D.OUT/f'checkpoints/NFT/round{round_index-1:02d}.pt'
    else:
        if evaluation not in ('BASE','NFT','SELF_FM'):raise ValueError('Evaluation label')
        label=evaluation;qs=m['evaluation_requests'];stage=D.OUT/'evaluation'/label
        if label=='BASE':qs=[q for q in qs if q['evaluation_noise_partition']=='FRESH']
        if len(qs)!=(12 if label=='BASE' else 36):raise RuntimeError('Evaluation budget mismatch')
        checkpoint=None if label=='BASE' else D.OUT/f'checkpoints/{label}/round15.pt'
    return list(qs),stage,label,checkpoint


def shard_requests(qs,worker,workers):
    if workers not in (1,2) or not 0<=worker<workers:
        raise ValueError('Only resource-admitted 1 or 2 workers, valid worker index')
    return [q for i,q in enumerate(qs) if i%workers==worker]


def shard_paths(stage,worker,workers):
    if workers==1:
        return stage/'sampling_worker',stage/'sampling_events.jsonl',stage/'raw_index.json'
    suffix=f'{worker:02d}_of_{workers:02d}'
    return (stage/('sampling_worker_'+suffix),stage/('sampling_events_'+suffix+'.jsonl'),
            stage/('raw_index_shard_'+suffix+'.json'))


def join_stage(m,*,round_index=None,evaluation=None,workers=2):
    """Authenticate complete disjoint shards in frozen order, without torch."""
    if workers!=2:raise ValueError('Explicit join is only needed for two shards')
    if round_index==0:raise ValueError('Round0 is the completed one-worker validation round')
    qs,stage,label,checkpoint=select_stage(m,round_index,evaluation)
    from .sample import verify_receipt
    all_receipts={};identity=None;rollouts=0;forwards=0;shards=[]
    for worker in range(workers):
        phase_dir,_,index_path=shard_paths(stage,worker,workers)
        completion=D.read(phase_dir/'completion.json')
        if completion.get('status')!='COMPLETE':raise RuntimeError('All shard workers must complete before join')
        index=D.read(index_path)
        assigned=shard_requests(qs,worker,workers)
        if (index.get('status'),index.get('label'),index.get('round'),index.get('worker'),index.get('workers'))!=('COMPLETE',label,round_index,worker,workers):
            raise ValueError('Shard stage/identity differs')
        if [r['request_id'] for r in index['receipts']]!=[q['request_id'] for q in assigned]:
            raise ValueError('Shard requests differ from frozen deterministic assignment')
        if identity is None:identity=index['model_identity']
        elif index['model_identity']!=identity:raise ValueError('Shards used different policies')
        if index['actual_new_rollouts']!=len(assigned) or index['actual_sampling_forwards']!=50*len(assigned):
            raise ValueError('Shard did not execute exactly its assigned budget')
        for q,receipt in zip(assigned,index['receipts']):
            if q['request_id'] in all_receipts:raise ValueError('Duplicate shard request')
            verify_receipt(q,receipt,model_identity=identity)
            all_receipts[q['request_id']]=receipt
        rollouts+=index['actual_new_rollouts'];forwards+=index['actual_sampling_forwards']
        shards.append(dict(worker=worker,index_path=str(index_path),index_sha256=D.sha(index_path),
                           completion_path=str(phase_dir/'completion.json'),completion_sha256=D.sha(phase_dir/'completion.json')))
    if set(all_receipts)!={q['request_id'] for q in qs}:raise ValueError('Incomplete aggregate')
    result=dict(status='COMPLETE',label=label,round=round_index,model_identity=identity,
        receipts=[all_receipts[q['request_id']] for q in qs],actual_new_rollouts=rollouts,
        actual_sampling_forwards=forwards,workers=workers,shards=shards,
        aggregate_order='FROZEN_MANIFEST_ORDER',seed_policy='UNCHANGED_FROZEN_REQUEST_SEEDS')
    D.write(stage/'raw_index.json',result)
    return result


def run_worker(m,*,round_index=None,evaluation=None,worker=0,workers=1):
    qs,stage,label,checkpoint=select_stage(m,round_index,evaluation)
    if round_index==0 and workers!=1:raise ValueError('Only one authorized round0 validation forward')
    all_qs=qs;qs=shard_requests(qs,worker,workers)
    phase_dir,event_path,index_path=shard_paths(stage,worker,workers)
    if workers>1 and (stage/'raw_index.json').exists():raise RuntimeError('Complete aggregate already exists')
    if workers>1 and (stage/'sampling_worker').exists():raise RuntimeError('Cannot change ownership of a started serial stage')
    if workers==1 and any(stage.glob('sampling_worker_*_of_*')):raise RuntimeError('Cannot change ownership of a started sharded stage')
    if any((stage/'raw'/q['request_id']).exists() for q in qs):
        raise RuntimeError('Assigned RAW cell already exists; no implicit shard replay')
    if checkpoint is not None and not checkpoint.exists():raise FileNotFoundError(checkpoint)
    with W.Phase(phase_dir,dict(stage='sampling',label=label,round=round_index,requests=len(qs),
                              total_requests=len(all_qs),worker=worker,workers=workers)):
        import torch
        torch.set_num_threads(2)
        from .model import load_policy
        from .learning import load_checkpoint
        from .sample import sample_request
        from meshflow_control.checkpoints import state_hash
        b=m['source_binding']
        policy=load_policy(D.resolve(b['path']),expected_sha256=b['sha256'],device='cuda')
        if checkpoint is not None:load_checkpoint(checkpoint,policy,restore_random_state=False)
        identity=dict(base_sha256=b['sha256'],adapter_sha256=state_hash(policy.adapter_state_dict()),round_index=round_index,
                      checkpoint_path=str(checkpoint) if checkpoint else None,checkpoint_sha256=D.sha(checkpoint) if checkpoint else None)
        budget=W.Budget(event_path,len(qs),policy if round_index==0 else None)
        results=[]
        with policy.activate('current'):
            for i,q in enumerate(qs):
                rec=sample_request(policy,q,budget,stage/'raw'/q['request_id'],label=label,model_identity=identity)
                results.append(rec)
                print(f'SAMPLE {i+1}/{len(qs)} {q["parent_id"][:6]} n{q["candidate"]} {rec["sampling"]["seconds"]:.2f}s',flush=True)
        if budget.rollouts!=len(qs) or budget.forwards!=50*len(qs):raise RuntimeError('Incomplete shard call budget')
        result=dict(status='COMPLETE',label=label,round=round_index,model_identity=identity,receipts=results,
            actual_new_rollouts=budget.rollouts,actual_sampling_forwards=budget.forwards)
        if workers>1:result.update(worker=worker,workers=workers,assignment='FROZEN_MANIFEST_INDEX_MOD_WORKERS')
        D.write(index_path,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--round',type=int);p.add_argument('--evaluation',choices=['BASE','NFT','SELF_FM'])
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,choices=[1,2],default=1)
    p.add_argument('--join',action='store_true')
    a=p.parse_args();m=D.read(D.OUT/'inputs/manifest.json')
    if a.join:
        result=join_stage(m,round_index=a.round,evaluation=a.evaluation,workers=a.workers)
    else:
        result=run_worker(m,round_index=a.round,evaluation=a.evaluation,worker=a.worker,workers=a.workers)
    print(json.dumps(dict(status=result['status'],requests=len(result['receipts']),
                          actual_sampling_forwards=result['actual_sampling_forwards']),sort_keys=True))


if __name__=='__main__':main()
