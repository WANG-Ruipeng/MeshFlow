"""Frozen-policy sampling; old sampler and runtime unchanged, zero updates."""
from .bootstrap import bind, OLD
D, E = bind()
from r2.pilots.terminal_reward_nft_v1 import sample_worker as S

def select_stage(m, round_index=None, evaluation=None):
    if round_index is not None or evaluation not in ('BASE','NFT','SELF_FM'):
        raise ValueError('Only frozen evaluation is authorized')
    qs = m['evaluation_requests']
    if len(qs) != 32 or len({q['condition_id'] for q in qs}) != 8:
        raise ValueError('Require eight unchanged conditions times four new noises')
    checkpoint = None if evaluation == 'BASE' else OLD / f'checkpoints/{evaluation}/round15.pt'
    return list(qs), D.OUT/'evaluation'/evaluation, evaluation, checkpoint

S.select_stage = select_stage

if __name__ == '__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True,choices=['BASE','NFT','SELF_FM'])
    a=p.parse_args()
    from .run import source_verify, require_frozen_endpoints
    source_verify();require_frozen_endpoints()
    result=S.run_worker(D.read(D.OUT/'inputs/manifest.json'),evaluation=a.evaluation)
    print(json.dumps({'status':result['status'],'rollouts':result['actual_new_rollouts'],'forwards':result['actual_sampling_forwards']}),flush=True)
