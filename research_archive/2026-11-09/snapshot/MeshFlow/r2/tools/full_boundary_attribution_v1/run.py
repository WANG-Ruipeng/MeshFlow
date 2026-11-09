"""Recoverable CPU-only dispatcher. Every cell is registered; no neural imports."""
import os
for _k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[_k]='2'
import argparse,concurrent.futures as cf,multiprocessing as mp,resource,time,traceback
from . import common as C
from . import geometry as G

def cell_run(cell,registration_sha):
    dest=C.OUT/'results'/cell['cell_id'];done=dest/'completion.json'
    if done.exists():
        old=C.read(done);assert old['registration_sha256']==registration_sha
        for p,h in old['files'].items():assert C.sha(p)==h,p
        return dict(old,reused=True)
    if (dest/'started.json').exists():raise RuntimeError('Partial cell requires explicit engineering review: '+cell['cell_id'])
    begin=time.perf_counter();C.write(dest/'started.json',dict(utc=C.utc(),cell_id=cell['cell_id'],registration_sha256=registration_sha,pid=os.getpid(),threads=2))
    try:
        r=C.read(C.OUT/'registration.json');a=C.load_geometry(cell)
        out=G.analyze(a,r['geometry_config']);paths=[]
        for key,value in out.items():
            if key in ['vectors','normal_line_hits','coplanar_intervals']:
                p=dest/(key+'.npz');C.frozen_npz(p,value)
            elif key=='summary':p=dest/(key+'.json');C.write(p,value)
            else:p=dest/(key+'.csv');C.csvwrite(p,value)
            paths.append(p)
        receipt=dict(status='COMPLETE',cell_id=cell['cell_id'],registration_sha256=registration_sha,
          files={str(p):C.sha(p) for p in paths},elapsed_seconds=time.perf_counter()-begin,
          peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=2,
          new_model_calls=0,new_noise=0,new_quality_scans=0,geometry_edits=0,reused=False,utc=C.utc())
        C.write(done,receipt);return receipt
    except Exception as e:
        C.write(dest/'failure.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',utc=C.utc(),error=repr(e),traceback=traceback.format_exc()))
        raise

def main():
    start=time.perf_counter();reg=C.read(C.OUT/'registration.json');rh=C.sha(C.OUT/'registration.json')
    for p,h in reg['scientific_code_hashes'].items():assert C.sha(p)==h,p
    complete=C.OUT/'analysis_completion.json'
    if complete.exists():
        old=C.read(complete)
        for c in reg['cells']:cell_run(c,rh)
        print('COMPLETE cached analysis verified:44 cells');return
    C.event('CPU_DISPATCH_BEGIN',cpu_threads_per_worker=2,requested_workers=2,model_calls=0)
    first=cell_run(reg['cells'][0],rh);C.event('FIRST_AUTHORIZED_CELL_MEASURED',**{k:first[k] for k in ['cell_id','peak_RSS_KiB','elapsed_seconds','reused']})
    res=C.resources(C.OUT/'resources_after_first_cell.json')
    available=int(next(x.split()[1] for x in res['memory'].splitlines() if x.startswith('MemAvailable:')))
    workers=2 if available>max(2*1024**2,4*first['peak_RSS_KiB']) and (res['loadavg'] or [0])[0]<8 else 1
    C.event('CPU_CONCURRENCY_SELECTED',workers=workers,available_memory_KiB=available,reason='Two independent registered CPU cells with measured RSS and low CPU load; zero GPU workers')
    receipts=[first]
    with cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(cell_run,c,rh):c for c in reg['cells'][1:]}
        for f in cf.as_completed(futures):
            item=f.result();receipts.append(item)
            print(item['cell_id'],f"{len(receipts)}/44",f"{item['elapsed_seconds']:.2f}s",flush=True)
    order={c['cell_id']:i for i,c in enumerate(reg['cells'])};receipts.sort(key=lambda x:order[x['cell_id']])
    C.write(complete,dict(status='COMPLETE',registration_sha256=rh,cells=44,receipts=receipts,workers=workers,
      measured_first_cell_peak_RSS_KiB=first['peak_RSS_KiB'],peak_worker_RSS_KiB=max(x['peak_RSS_KiB'] for x in receipts),
      wall_seconds=time.perf_counter()-start,model_calls=0,new_noise=0,new_quality_scans=0,geometry_edits=0,
      timing_note='Includes first actual cell serial then concurrent registered cells; not an isolated performance benchmark.'))
    C.resources(C.OUT/'resources_after_analysis.json');C.event('CPU_ANALYSIS_COMPLETE',cells=44,workers=workers,elapsed_seconds=time.perf_counter()-start)
if __name__=='__main__':main()