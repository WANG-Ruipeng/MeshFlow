"""CPU IO-guard and routing fixtures; no real trainer/model execution."""
import errno
import hashlib
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
import time
from unittest.mock import patch
from r2.pilots.bench_parts_convergence_v1 import worker_entry as W

BASE=Path('/mnt/e/MeshFlow/experiments/r2_bench_parts_convergence_v1')
OUT=BASE/('worker_entry_cpu_test_20261019_'+str(time.time_ns()));OUT.mkdir()
root=OUT/'fixture';root.mkdir();checks=[]
real_replace=os.replace
class Clock:
    def __init__(self):self.value=0.
    def now(self):return self.value
    def sleep(self,n):self.value+=n
def guard(fn,timeout=.07):
    c=Clock();g=W.ScopedIO(root,fn,log_path=OUT/'io_events.jsonl',replace_timeout=timeout,metadata_timeout=.07,clock=c.now,sleep=c.sleep)
    return g,c
src=root/'checkpoint.pt.partial';dst=root/'checkpoint.pt';src.write_bytes(b'exact prepared bytes')
attempts=[]
def transient(a,b):
    attempts.append((a,b))
    if len(attempts)<3:raise PermissionError(errno.EACCES,'fixture sharing violation')
    return real_replace(a,b)
g,c=guard(transient);g.replace(src,dst)
assert dst.read_bytes()==b'exact prepared bytes' and not src.exists() and len(attempts)==3
checks.append('transient_EACCES_retries_exact_prepared_bytes_then_succeeds')
src.write_bytes(b'preserved source');prior=dst.read_bytes();attempts=[]
def persistent(a,b):attempts.append(1);raise PermissionError(errno.EPERM,'fixture persistent sharing')
g,c=guard(persistent)
try:g.replace(src,dst)
except PermissionError:pass
else:raise AssertionError('Persistent replacement falsely succeeded')
assert c.value<=.070000001 and src.read_bytes()==b'preserved source' and dst.read_bytes()==prior
checks.append('persistent_EPERM_bounded_preserves_source_and_old_destination')
g,c=guard(real_replace)
try:g.replace(root/'missing.partial',dst)
except FileNotFoundError:pass
else:raise AssertionError('Missing source accepted')
assert c.value==0
checks.append('missing_source_no_retry_or_fake_success')
g,c=guard(persistent);attempts=[]
try:g.replace(OUT/'outside.partial',dst)
except PermissionError:pass
else:raise AssertionError('Outside root accepted')
assert len(attempts)==1 and c.value==0
checks.append('outside_exact_root_bypasses_retry')
meta=root/'status.json';meta.write_text('{"step":8750}')
calls=[]
def metadata_transient(p):
    calls.append(1)
    if len(calls)==1:raise FileNotFoundError(errno.ENOENT,'fixture atomic read gap')
    return json.loads(Path(p).read_text())
g,c=guard(real_replace)
assert g.metadata_reader(metadata_transient)(meta)=={'step':8750} and len(calls)==2
checks.append('metadata_transient_absence_retries_without_training_replay')
g,c=guard(real_replace)
def jsonread(p):return json.loads(Path(p).read_text())
try:g.metadata_reader(jsonread)(root/'absent.json')
except FileNotFoundError:pass
else:raise AssertionError('Persistent absent metadata accepted')
assert c.value<=.070000001
checks.append('persistent_metadata_missing_bounded_failure')
meta.write_text('{bad')
g,c=guard(real_replace)
try:g.metadata_reader(jsonread)(meta)
except json.JSONDecodeError:pass
else:raise AssertionError('Malformed metadata accepted')
assert c.value==0
checks.append('malformed_JSON_not_hidden_by_IO_retry')

from r2.pilots.bench_parts_convergence_v1 import train as T, resume_train as R
from r2.pilots.bench_control_transfer_v1 import train as OLD
train_hash=T.sha(T.__file__);reg_hash=T.sha(BASE/'registration.json')
original_T_read=T.read;original_OLD_read=OLD.read
routes=[]
def normal():
    assert os.replace is not real_replace and T.os.replace is os.replace and OLD.os.replace is os.replace
    assert T.read is not original_T_read and OLD.read is not original_OLD_read
    routes.append('train')
def offgrid():
    assert os.replace is not real_replace;routes.append('resume_train')
cases=[('fresh',[],None),('fork',['--fork',str(root/'training/COMMON/step12000.pt')],None),
       ('on_grid',['--resume',str(root/'training/COMMON/rolling.pt')],8000),
       ('off_grid',['--resume',str(root/'training/COMMON/rolling.pt')],8750)]
with patch.object(T,'DEFAULT_ROOT',root),patch.object(T,'main',side_effect=normal),patch.object(R,'main',side_effect=offgrid):
    for name,extra,cursor in cases:
        folder=root/'training/COMMON';folder.mkdir(parents=True,exist_ok=True)
        if cursor is not None:(folder/'rolling.json').write_text(json.dumps(dict(step=cursor)))
        with patch.object(sys,'argv',['worker_entry','--root',str(root),'--arm','COMMON','--target','10000',*extra]):W.main()
        assert os.replace is real_replace and T.read is original_T_read and OLD.read is original_OLD_read
assert routes==['train','train','train','resume_train']
checks.append('fresh_fork_1000_resume_delegate_train_offgrid_resume_delegate_resume_train')
checks.append('same_scoped_os_replace_covers_original_checkpoint_inference_and_metadata_writers')
# Finally-restoration also holds if the delegated worker fails.
with patch.object(T,'DEFAULT_ROOT',root),patch.object(T,'main',side_effect=RuntimeError('fixture worker failure')):
    try:
        with patch.object(sys,'argv',['worker_entry','--root',str(root),'--arm','COMMON','--target','10000']):W.main()
    except RuntimeError as e:assert str(e)=='fixture worker failure'
    else:raise AssertionError('Delegated failure swallowed')
assert os.replace is real_replace and T.read is original_T_read and OLD.read is original_OLD_read
checks.append('delegated_failure_propagates_and_IO_hooks_restore')
assert T.sha(T.__file__)==train_hash and T.sha(BASE/'registration.json')==reg_hash
assert not T.torch.cuda.is_initialized()
report=dict(status='PASS_CPU_IO_ONLY_WORKER_ENTRY',checks=checks,worker_entry_sha256=T.sha(W.__file__),
    train_sha256=train_hash,registration_sha256=reg_hash,protected_sources_unchanged=True,
    actual_model_forwards=0,actual_training_calls=0,CUDA_initialized=False,
    source='Actual wrapper IO paths and CLI executed; only original train/resume entrypoints substituted',
    test_retry_clock='Deterministic fake clock verifies exhaustion; real filesystem source/destination bytes checked',
    directory=str(OUT))
T.write(OUT/'receipt.json',report);print(json.dumps(report))
