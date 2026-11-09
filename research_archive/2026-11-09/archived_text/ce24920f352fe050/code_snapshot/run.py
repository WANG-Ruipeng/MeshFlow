"""Recoverable CPU-only entrypoint: prepare / freeze / geometry / render / aggregate / audit."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, concurrent.futures, resource, time, traceback, sys
from . import common as c

def worker(cell,config):
    from .geometry import process_cell
    start=time.monotonic()
    result=process_cell(cell,config,c.OUT/'diagnostics'/cell['cell_id'])
    assert not any(x=='torch' or x.startswith('torch.') for x in sys.modules),'No model framework permitted'
    return dict(cell_id=cell['cell_id'],status=result['status'],seconds=time.monotonic()-start,
        peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pid=os.getpid(),model_framework_loaded=False)

def freeze():
    from .geometry import DEFAULTS
    import unittest
    from . import test_geometry
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_geometry)
    start=time.monotonic();res=unittest.TextTestRunner(verbosity=2).run(suite)
    assert res.wasSuccessful()
    c.write(c.OUT/'analytic_tests.json',dict(tests=res.testsRun,failures=len(res.failures),errors=len(res.errors),status='PASS'))
    files=[c.CODE/x for x in ('__init__.py','common.py','prepare.py','run.py','geometry.py','test_geometry.py')]
    hashes={str(p):c.sha(p) for p in files}
    c.write(c.OUT/'science_freeze.json',dict(status='FROZEN_BEFORE_REAL_GEOMETRY_DIAGNOSTICS',
        registration_sha256=c.sha(c.OUT/'registration.json'),config=DEFAULTS,code_hashes=hashes,
        scientific_budget=c.read(c.OUT/'registration.json')['budgets'],no_models=True,no_old_scoring=True))
    c.event('SCIENCE_FROZEN',tests=res.testsRun,code_hashes=hashes)

def geometry():
    freeze=c.read(c.OUT/'science_freeze.json');reg=c.read(c.OUT/'registration.json')
    assert c.sha(c.OUT/'registration.json')==freeze['registration_sha256']
    for p,h in freeze['code_hashes'].items():assert c.sha(p)==h,p
    for p,h in c.read(c.OUT/'protected_hashes.json').items():assert c.sha(p)==h,p
    jobs=[]
    for cell in reg['cells']+reg['references']:
        receipt=c.OUT/'receipts'/f"{cell['cell_id']}.json"
        if receipt.exists():
            r=c.read(receipt)
            for p,h in r['files'].items():assert c.sha(p)==h,p
            continue
        jobs.append(cell)
    start=time.monotonic();before=c.resources();c.write(c.OUT/'resources_dispatch.json',before)
    c.event('CPU_DISPATCH',worker_count=2,threads_per_worker=1,remaining=len(jobs),gpu_workers=0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(worker,cell,freeze['config']):cell for cell in jobs}
        for fut in concurrent.futures.as_completed(futures):
            cell=futures[fut]
            try:
                value=fut.result();dest=c.OUT/'diagnostics'/cell['cell_id']
                value['files']={str(p):c.sha(p) for p in sorted(dest.iterdir()) if p.is_file()}
                c.write(c.OUT/'receipts'/f"{cell['cell_id']}.json",value)
                c.event('CELL_COMPLETE',**{k:v for k,v in value.items() if k!='files'})
                print(cell['cell_id'],value['status'],round(value['seconds'],3),flush=True)
            except BaseException:
                c.event('CELL_FAILURE',cell_id=cell['cell_id'],traceback=traceback.format_exc())
                raise
    c.write(c.OUT/'geometry_completion.json',dict(status='COMPLETE',registered_cells=44,source_references=6,
        actual_receipts=len(list((c.OUT/'receipts').glob('*.json'))),wall_seconds=time.monotonic()-start,
        cpu_workers=2,threads_per_worker=1,resources_after=c.resources(),model_calls=0,old_scorer_calls=0))
    c.event('GEOMETRY_STAGE_COMPLETE',wall_seconds=time.monotonic()-start)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['prepare','freeze','geometry','render','aggregate','audit']);a=p.parse_args()
    if a.stage=='prepare':
        from .prepare import build;build()
    elif a.stage=='freeze':freeze()
    elif a.stage=='geometry':geometry()
    elif a.stage=='render':
        from .render import render;render()
    elif a.stage=='aggregate':
        from .aggregate import main as aggregate;aggregate()
    elif a.stage=='audit':
        import runpy;runpy.run_path(str(c.OUT/'independent_audit/audit.py'),run_name='__main__')
if __name__=='__main__':main()