"""Reuse authenticated D0 job creation for the new32-per-model frozen panel."""
from .bootstrap import bind
D,E=bind()
from r2.pilots.terminal_reward_nft_v1 import denoise_worker as P
OFFICIAL_D0_SHA256=P.OFFICIAL_D0_SHA256
retained_d0_binding=P.retained_d0_binding

def stage_requests(manifest, *, round_index=None, evaluation=None):
    if round_index is not None or evaluation not in ('BASE','NFT','SELF_FM'):
        raise ValueError('Frozen evaluation only')
    qs=manifest['evaluation_requests']
    if len(qs)!=32 or len({q['request_id'] for q in qs})!=32:
        raise ValueError('32 frozen requests required')
    return D.OUT/'evaluation'/evaluation,evaluation,list(qs)

P.stage_requests=stage_requests
prepare_jobs=P.prepare_jobs
