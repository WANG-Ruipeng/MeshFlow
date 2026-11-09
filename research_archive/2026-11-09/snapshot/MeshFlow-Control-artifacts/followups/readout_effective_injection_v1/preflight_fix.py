from pathlib import Path
import json,hashlib,time
r=Path('/mnt/e/MeshFlow-Control-artifacts/followups/readout_effective_injection_v1')
p=r/'plan.json';src=r/'run_check.py'
assert not (r/'budget/calls.jsonl').exists()
old=p.read_bytes();before=src.read_bytes();s=before.decode('utf-8')
assert 'def event(kind,**data):' in s
s=s.replace('def event(kind,**data):','def event(event_type,**data):').replace('row=dict(event=kind,time_unix=time.time(),**data)','row=dict(event=event_type,time_unix=time.time(),**data)')
compile(s,str(src),'exec')
(r/'plan.initial.json').write_bytes(old)
src.write_bytes(s.encode('utf-8'))
plan=json.loads(old);oldhash=plan['runner_sha256'];plan['runner_sha256']=hashlib.sha256(src.read_bytes()).hexdigest()
plan['preflight_revision']='Resolved diagnostic ledger parameter-name collision before any model call; scientific plan unchanged.'
p.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n',encoding='utf-8')
record=dict(time_unix=time.time(),actual_model_calls_before_revision=0,reason=plan['preflight_revision'],old_runner_sha256=oldhash,new_runner_sha256=plan['runner_sha256'],old_plan_sha256=hashlib.sha256(old).hexdigest(),new_plan_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(r/'preflight_revision.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))