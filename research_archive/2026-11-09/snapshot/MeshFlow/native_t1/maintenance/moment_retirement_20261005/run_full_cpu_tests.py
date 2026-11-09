"""One authorized full CPU test run; guards prevent real experiment execution."""
from contextlib import ExitStack
from pathlib import Path
from unittest import mock
import hashlib
import json
import os
import sys
import time
import unittest

ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'native_t1/maintenance/moment_retirement_20261005'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
import torch
from models.equidit import DiT
from native_t1.model import NativeInpaintingModel
from native_t1.chair_model import ChairModel

torch.set_num_threads(2)
attempts={}
synthetic_loads=[]
original_load=torch.load
def forbidden(name):
    def blocked(*args,**kwargs):
        attempts[name]=attempts.get(name,0)+1
        raise AssertionError('CPU suite attempted forbidden operation: '+name)
    return blocked

def temporary_checkpoint_load(path,*args,**kwargs):
    resolved=Path(path).resolve()
    if Path('/tmp') not in resolved.parents:
        return forbidden('non_temporary_checkpoint_read')(path)
    synthetic_loads.append(str(resolved))
    return original_load(path,*args,**kwargs)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

before={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'native_t1/tests').glob('test_*.py'))}
OUT.mkdir(parents=True,exist_ok=True)
assert not torch.cuda.is_initialized()
start=time.monotonic()
with ExitStack() as stack:
    for owner,name,label in ((torch.cuda,'_lazy_init','CUDA_initialization'),
            (DiT,'__init__','real_DiT_construction'),(DiT,'forward','real_DiT_forward'),
            (NativeInpaintingModel,'forward','real_Native_forward'),
            (ChairModel,'forward','real_Chair_forward'),
            (torch.optim.AdamW,'step','actual_AdamW_update')):
        stack.enter_context(mock.patch.object(owner,name,new=forbidden(label)))
    stack.enter_context(mock.patch.object(torch,'load',new=temporary_checkpoint_load))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'native_t1/tests'),pattern='test_*.py')
    with (OUT/'full_cpu_tests.log').open('w',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
after={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'native_t1/tests').glob('test_*.py'))}
counts={name:getattr(module,'COUNTS') for name,module in sys.modules.copy().items()
        if name.startswith('test_') and hasattr(module,'COUNTS')}
receipt=dict(status='PASS' if result.wasSuccessful() and not attempts and before==after and not torch.cuda.is_initialized() else 'BLOCKED',
    tests_run=result.testsRun,failures=[(str(t),s) for t,s in result.failures],errors=[(str(t),s) for t,s in result.errors],
    skipped=[(str(t),s) for t,s in result.skipped],seconds=time.monotonic()-start,
    command='python -B native_t1/maintenance/moment_retirement_20261005/run_full_cpu_tests.py',
    scope='Complete native_t1/tests discovery; isolated synthetic CPU modules/coordinate autograd/OT and temporary checkpoint fixtures permitted; full trained backbone/CUDA/AdamW step forbidden.',
    CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],cpu_threads=torch.get_num_threads(),
    cuda_initialized=torch.cuda.is_initialized(),forbidden_attempts=attempts,
    real_model_forwards=0,real_training_updates=0,real_generations=0,
    synthetic_checkpoint_load_count=len(synthetic_loads),synthetic_test_counters=counts,
    test_source_hashes_before=before,test_source_hashes_after=after,test_sources_unchanged=before==after,
    log_sha256=sha(OUT/'full_cpu_tests.log'),runner_sha256=sha(__file__))
(OUT/'full_cpu_tests.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ('status','tests_run','failures','errors','skipped','seconds','forbidden_attempts','cuda_initialized','test_sources_unchanged')}))
raise SystemExit(0 if receipt['status']=='PASS' else 1)
