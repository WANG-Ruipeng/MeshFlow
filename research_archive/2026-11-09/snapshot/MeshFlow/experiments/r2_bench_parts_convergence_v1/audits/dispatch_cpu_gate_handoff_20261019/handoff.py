"""Explicit coordinator-only handoff; leave the authorized training child running."""
import difflib, hashlib, json, os, signal, time
from pathlib import Path
import psutil

root=Path('/mnt/e/MeshFlow/experiments/r2_bench_parts_convergence_v1')
audit=Path(__file__).parent
code=Path('/mnt/e/MeshFlow/r2/pilots/bench_parts_convergence_v1/run_all.py')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
status=read(root/'dispatch/status.json')
assert status['pid']==339 and status['phase']=='TRAIN_4000'
parent=psutil.Process(339)
assert 'r2.pilots.bench_parts_convergence_v1.run_all' in parent.cmdline()
child=psutil.Process(665)
assert child.ppid()==339 and 'r2.pilots.bench_parts_convergence_v1.train' in child.cmdline()
training_before=read(root/'training/COMMON/status.json')
assert training_before['pid']==665 and training_before['target']==4000
old=read(audit/'identity.before.json')
new=read(root/'dispatch/identity.json')
assert new==old
for p,h in old['code_hashes'].items():
    if Path(p)!=code:assert sha(Path(p))==h,p
assert sha(root/'registration.json')==old['registration_sha256']
write(audit/'handoff.before.json',dict(controller=status,training=training_before,controller_cmd=parent.cmdline(),training_cmd=child.cmdline(),unix=time.time()))
# SIGTERM only the coordinator. Training has its own stdout file and continues.
os.kill(339,signal.SIGTERM)
time.sleep(2)
assert not psutil.pid_exists(339) or psutil.Process(339).status()==psutil.STATUS_ZOMBIE
assert child.is_running() and child.status()!=psutil.STATUS_ZOMBIE
training_after=read(root/'training/COMMON/status.json')
assert training_after['pid']==665 and training_after['step']>=training_before['step']
new['code_hashes'][str(code)]=sha(code)
write(root/'dispatch/identity.json',new)
write(audit/'identity.after.json',new)
(audit/'code.diff').write_text(''.join(difflib.unified_diff((audit/'run_all.before.py').read_text(encoding='utf-8').splitlines(True),code.read_text(encoding='utf-8').splitlines(True),fromfile='run_all.before.py',tofile='run_all.py')),encoding='utf-8')
receipt=dict(status='PASS',action='COORDINATOR_ONLY_EXPLICIT_HANDOFF',terminated_controller_pid=339,
    protected_live_training_pid=665,training_before=training_before,training_after=training_after,
    registration_sha256=new['registration_sha256'],run_all_before=old['code_hashes'][str(code)],run_all_after=new['code_hashes'][str(code)],
    changes=['GPU peak accounting excludes CPU jobs','five-column nvidia telemetry accepted','gallery cache requires completion receipt'],
    replayed_updates=0,new_model_forwards=0,training_signal_sent=False,
    child_job_completion='Original controller cannot write completion for train_COMMON_4000; explicit reattach will observe the same child and saved checkpoint. Reconcile after it exits.',unix=time.time())
write(audit/'receipt.json',receipt)
print(json.dumps(receipt))
