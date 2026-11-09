"""Verify and reference all old HYBRID endpoint raw; never run the model."""
from pathlib import Path
import numpy as np
from .common import *
from .ledger import Budget


def run(out=OUT):
    out=Path(out);where=out/'evaluation/START';target=where/'generated_rows.json'
    if target.exists():
        prior=read_json(target)
        assert prior['status']=='COMPLETE'
        for row in prior['rows']:
            if row['status']=='CACHED':assert digest(row['raw_path'])==row['raw_sha256']
        return prior
    origin=read_json(HYBRID/'evaluation/OT_HYBRID/generated_rows.json')
    lookup={r['case_id']:r for r in origin['rows']};rows=[]
    for rec in records():
        old=lookup[rec['case_id']];row=dict(rec,label='START',checkpoint_sha256=START_SHA,model_calls=0)
        if rec['status']!='READY':
            assert old['status']=='NOT_APPLICABLE'
            rows.append(dict(row,status='NOT_APPLICABLE'));continue
        assert old['checkpoint_sha256']==START_SHA and old['status']=='SAMPLED'
        assert old['input_sha256']==rec['input_sha256']
        meta=old['metadata']
        assert meta['solver']=='original_Euler50' and meta['nfe']==50 and meta['known_all51_bitwise']
        assert meta['forward_dtype']=='bfloat16' and meta['integration_dtype']=='float32' and meta['GT_model_inputs'] is False
        runtime=meta['runtime_before']
        assert runtime['sdpa_math_enabled'] and not runtime['sdpa_flash_enabled'] and not runtime['tf32_matmul']
        assert runtime['deterministic_algorithms'] and not runtime['deterministic_warn_only']
        input_path=resolve(rec['input_path']);raw_path=Path(old['raw_path'])
        assert digest(input_path)==rec['input_sha256'] and digest(raw_path)==old['raw_sha256']
        with np.load(input_path,allow_pickle=False) as z:bank={k:z[k].copy() for k in z.files}
        with np.load(raw_path,allow_pickle=False) as z:raw={k:z[k].copy() for k in ('raw','C','noise','known_mask')}
        for before,after in (('C','C'),('noise','z'),('known_mask','known_mask')):
            a,b=raw[before],bank[after]
            assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
        assert raw['raw'].shape==(rec['N'],9) and np.isfinite(raw['raw']).all()
        assert raw['raw'][:rec['K']].tobytes()==bank['C'].reshape(-1,9).tobytes()
        assert array_hash(raw['raw'])==old['raw_array_sha256']
        row.update(status='CACHED',raw_path=str(raw_path),raw_sha256=old['raw_sha256'],raw_array_sha256=old['raw_array_sha256'],
            historical_raw_path=str(raw_path),historical_checkpoint_label='OT_HYBRID',historical_metadata=meta,
            metadata=dict(status='CACHED',new_model_calls=0,cache_identity_verified=True,
                original_solver='original_Euler50',original_nfe=50,original_known_all51_bitwise=True),
            cache_scope='Existing full raw tensors, all identities checked; no new trajectory or I/P replay')
        rows.append(row)
    value=dict(label='START',rows=rows,status='COMPLETE',requested=144,sampled=0,cached=138,not_applicable=6,
        source_generated_rows_sha256=digest(HYBRID/'evaluation/OT_HYBRID/generated_rows.json'),model_calls=0)
    atomic_json(target,value);Budget(out,'cache_start').snapshot()
    atomic_json(out/'stages/cache_START.json',dict(status='DONE',requested=144,cached=138,not_applicable=6,new_model_calls=0))
    print('START_CACHE_COMPLETE 138 raw references, 6 NOT_APPLICABLE',flush=True)
    return value


if __name__=='__main__':run()
