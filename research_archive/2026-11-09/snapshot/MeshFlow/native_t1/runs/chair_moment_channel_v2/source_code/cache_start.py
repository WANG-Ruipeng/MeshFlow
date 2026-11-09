"""Verify the actual HYBRID5000 endpoint and reference its 138 saved outputs."""
from pathlib import Path
import argparse
import numpy as np
from experiments.mf_h1_local0.chair_surface_attention_v1 import scoring as source
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.common import atomic_json,array_hash

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'native_t1/runs/chair_moment_channel_v2'
CHAIR=source.CHAIR
HYBRID=source.HYBRID
START_SHA=source.START_SHA


def run(out=OUT):
    out=Path(out).resolve()
    if out in (CHAIR.resolve(),HYBRID.resolve()):raise ValueError('V2 output must be isolated')
    manifest=source.read_json(HYBRID/'checkpoint_manifest.json')
    endpoint=next(r for r in manifest['endpoints'] if r['metadata']['arm']=='OT_HYBRID')
    meta=endpoint['metadata'];checkpoint=Path(endpoint['path'])
    if endpoint['checkpoint_sha256']!=START_SHA or source.sha(checkpoint)!=START_SHA:
        raise ValueError('Wrong actual HYBRID checkpoint')
    if (meta['pilot_step'],meta['conditional_total_step'],meta['lambda_variance'])!=(1000,5000,.25):
        raise ValueError('Wrong HYBRID lineage')
    if not meta['full_recovery_state']:raise ValueError('Full original recovery state required')
    origin_path=HYBRID/'evaluation/OT_HYBRID/generated_rows.json'
    origin=source.read_json(origin_path);lookup={r['case_id']:r for r in origin['rows']}
    registered=source.records();rows=[]
    if len(registered)!=144 or len(lookup)!=144:raise ValueError('Original panel changed')
    for rec in registered:
        old=lookup[rec['case_id']]
        row=dict(rec,label='START',checkpoint_label='START',checkpoint_sha256=START_SHA,model_calls=0)
        if rec['status']!='READY':
            if old['status']!='NOT_APPLICABLE':raise ValueError('Original applicability differs')
            rows.append(dict(row,status='NOT_APPLICABLE'));continue
        if old['status']!='SAMPLED' or old['checkpoint_sha256']!=START_SHA or old['input_sha256']!=rec['input_sha256']:
            raise ValueError('Old generated identity differs')
        m=old['metadata'];runtime=m['runtime_before']
        if not (m['solver']=='original_Euler50' and m['nfe']==50 and m['known_all51_bitwise'] and
                m['forward_dtype']=='bfloat16' and m['integration_dtype']=='float32' and m['GT_model_inputs'] is False and
                runtime['sdpa_math_enabled'] and not runtime['sdpa_flash_enabled'] and not runtime['tf32_matmul'] and
                runtime['deterministic_algorithms'] and not runtime['deterministic_warn_only']):
            raise ValueError('Original sampler contract differs')
        input_path=source.resolve(CHAIR,rec['input_path']);raw_path=Path(old['raw_path'])
        if source.sha(input_path)!=rec['input_sha256'] or source.sha(raw_path)!=old['raw_sha256']:
            raise ValueError('Original input/raw file changed')
        bank=source.arrays(input_path);raw=source.arrays(raw_path)
        for a,b in (('C','C'),('noise','z'),('known_mask','known_mask')):
            if raw[a].shape!=bank[b].shape or raw[a].dtype!=bank[b].dtype or raw[a].tobytes()!=bank[b].tobytes():
                raise ValueError('Original C/noise/mask identity differs')
        if raw['raw'].shape!=(rec['N'],9) or not np.isfinite(raw['raw']).all():raise ValueError('Invalid original raw')
        if raw['raw'][:rec['K']].tobytes()!=bank['C'].reshape(-1,9).tobytes():raise ValueError('Original C not preserved')
        if array_hash(raw['raw'])!=old['raw_array_sha256']:raise ValueError('Original raw array identity differs')
        score=ROOT/'native_t1/runs/chair_surface_attention_v1/evaluation/START/scoring/scores'/(rec['case_id']+'.json')
        row.update(status='CACHED',raw_path=str(raw_path),raw_sha256=old['raw_sha256'],raw_array_sha256=old['raw_array_sha256'],
            historical_checkpoint_label='OT_HYBRID',historical_metadata=m,
            metadata=dict(status='CACHED',cache_identity_verified=True,new_model_calls=0,original_nfe=50,original_known_all51_bitwise=True),
            score_path=str(score) if score.exists() else None,score_sha256=source.sha(score) if score.exists() else None)
        rows.append(row)
    if sum(r['status']=='CACHED' for r in rows)!=138 or sum(r['status']=='NOT_APPLICABLE' for r in rows)!=6:
        raise ValueError('Original 138+6 panel identity differs')
    value=dict(label='START',checkpoint_label='START',status='COMPLETE',requested=144,cached=138,sampled=0,not_applicable=6,
        model_calls=0,rows=rows,source_generated_rows_sha256=source.sha(origin_path),source_checkpoint_sha256=START_SHA)
    target=out/'evaluation/START/generated_rows.json'
    if target.exists():
        if source.read_json(target)!=value:raise ValueError('Existing V2 START registration differs')
    else:atomic_json(target,value)
    atomic_json(out/'stages/cache_START.json',dict(status='DONE',requested=144,cached=138,not_applicable=6,new_model_calls=0))
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT)
    print(run(p.parse_args().out)['status'])
