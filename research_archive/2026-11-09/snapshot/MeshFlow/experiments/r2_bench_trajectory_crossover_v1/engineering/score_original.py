import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import argparse, concurrent.futures, multiprocessing, resource
import numpy as np

def one(job):
    combo,q=job; rid=q['request_id']; dest=score_path(combo,rid)
    if (dest/'complete.json').exists(): verify(read(dest/'complete.json')['files']); return str(dest/'row.json')
    if dest.exists():raise RuntimeError('Partial score preserved: explicit recovery needed '+str(dest))
    receipt=read(raw_path(combo,rid).parent/'receipt.json');verify(receipt['files'])
    assert receipt['generator_forwards']==25 and receipt['cached_prefix_steps']==25
    from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
    assert 'torch' not in sys.modules
    tick=time.perf_counter();save(dest/'attempt.json',dict(combo=combo,request_id=rid,raw_sha256=sha(raw_path(combo,rid))))
    mesh=arrays(raw_path(combo,rid));mesh['index_domain']=np.asarray('ACTUAL_RAW_C_SEAM_SHARED_INDEX')
    CPU.score_current(q,mesh,dest,'RAW',True)
    assert 'torch' not in sys.modules
    save(dest/'complete.json',dict(status='COMPLETE',combo=combo,seconds=time.perf_counter()-tick,
        model_calls=0,new_queries=0,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        files={str(p):sha(p) for p in dest.iterdir() if p.is_file()}))
    print('SCORED',combo,rid,flush=True);return str(dest/'row.json')

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--first-only',action='store_true');a=p.parse_args()
    reg=registration();verify(reg['source_hashes'])
    save(OUT/'resources'/('cpu_first.json' if a.first_only else 'cpu_full.json'),resources())
    jobs=[(combo,q) for combo in reg['new_combinations'] for q in reg['requests']]
    if a.first_only:jobs=jobs[:1]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(one,jobs))
    save(OUT/('scores_first.json' if a.first_only else 'scores_complete.json'),dict(status='COMPLETE',rows=len(results),workers=a.workers,files={p:sha(p) for p in results}))

if __name__=='__main__':main()
