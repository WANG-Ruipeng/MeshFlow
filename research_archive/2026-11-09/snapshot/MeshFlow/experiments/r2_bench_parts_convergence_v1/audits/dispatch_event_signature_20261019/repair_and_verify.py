"""CPU-only logging regression and explicit dispatcher identity migration."""
import ast, difflib, json, sys
from pathlib import Path
from r2.pilots.bench_parts_convergence_v1 import run_all as R

root=R.DEFAULT_ROOT
audit=Path(__file__).resolve().parent
before=audit/'run_all.before.py';after=Path(R.__file__)
old=R.read(audit/'identity.before.json')
assert old['code_hashes'][str(after)]==R.sha(before)
assert R.sha(root/'registration.json')==old['registration_sha256']
for path,h in old['code_hashes'].items():
    if path!=str(after):assert R.sha(path)==h,(path,'unexpected code change')
ast.parse(after.read_text())
driver=object.__new__(R.Driver)
driver.dest=audit/'cpu_regression';driver.dest.mkdir(exist_ok=True)
driver.measurements={};driver.force_serial={}
driver.snapshot=lambda: dict(gpu_total_mib=16303,gpu_free_mib=12000,gpu_used_mib=4303,gpu_utilization=10,cpu_logical=24,memory_available_mib=60000,cpu_utilization=5)
plan=driver.gate('sampling',1,0)
assert plan['additional_workers']==1
R.event(driver.dest/'events.jsonl','CHILD_STARTED',key='fixture',pid=0,kind='sampling',command=['NO_EXECUTION'])
assert R.read(root/'training/COMMON/status.json')['step']==2000
cp=R.read(root/'training/COMMON/rolling.json')
assert R.sha(root/'training/COMMON/rolling.pt')==cp['sha256']
assert not list((root/'dispatch/jobs').glob('*'))
assert not list((root/'evaluation').glob('NATIVE_S2000/*/raw/receipt.json'))
diff=''.join(difflib.unified_diff(before.read_text().splitlines(True),after.read_text().splitlines(True),fromfile='run_all.before.py',tofile='run_all.py'))
(audit/'code.diff').write_text(diff,encoding='utf-8')
identity=json.loads(json.dumps(old));identity['code_hashes'][str(after)]=R.sha(after)
R.write(audit/'identity.after.json',identity)
R.write(root/'dispatch/identity.json',identity)
receipt=dict(status='PASS',root_cause='event second argument named kind collided with job kind metadata',
    change='Rename logger positional argument to event_name; payload and algorithms unchanged',
    before_sha256=R.sha(before),after_sha256=R.sha(after),registration_unchanged=True,
    completed_updates_preserved=2000,checkpoint_sha256=cp['sha256'],scientific_updates_replayed=0,
    generator_forwards=0,D0_forwards=0,new_child_jobs=0,
    regression=['Real gate with sampling kind metadata','Real event logger with job kind metadata'],
    resume='Explicit controller reattach; first sampling job never launched in failed dispatcher')
R.write(audit/'receipt.json',receipt)
print(json.dumps(receipt))
