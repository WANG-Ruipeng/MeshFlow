"""One user-authorized checkpoint recovery; original attempts remain archived."""
import collections, difflib, hashlib, json, os, time
from pathlib import Path
import psutil
root=Path('/mnt/e/MeshFlow/experiments/r2_bench_parts_convergence_v1')
audit=Path(__file__).parent
code=Path('/mnt/e/MeshFlow/r2/pilots/bench_parts_convergence_v1/run_all.py')
out=root/'training/COMMON'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    tmp=p.with_name(p.name+'.recovery_partial')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
for pid in (339,665):
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE
assert not (audit/'authorized_recovery_receipt.json').exists()
checkpoint=read(out/'rolling.json')
assert checkpoint['step']==2000 and sha(out/'rolling.pt')==checkpoint['sha256']
assert (root/'dispatch/jobs/train_COMMON_4000__interrupted_attempt0/attempt.json').exists()
assert not (root/'dispatch/jobs/train_COMMON_4000').exists()
old_identity=read(audit/'identity.before.json')
assert read(root/'dispatch/identity.json')==old_identity
assert sha(root/'registration.json')==old_identity['registration_sha256']
for path,digest in old_identity['code_hashes'].items():
    if Path(path)!=code:assert sha(Path(path))==digest,path
accounting={};files={}
for name in ('training_log.jsonl','exposure.jsonl','ledger.jsonl'):
    p=out/name;original=audit/('interrupted.'+name)
    assert sha(p)==sha(original)
    lines=p.read_text(encoding='utf-8-sig').splitlines(True)
    keep=[];lost=[]
    for line in lines:
        if not line.strip():continue
        r=json.loads(line)
        (keep if int(r.get('step',0))<=2000 else lost).append(line)
    (audit/('uncommitted.'+name)).write_text(''.join(lost),encoding='utf-8')
    tmp=p.with_name(p.name+'.recovery_partial');tmp.write_text(''.join(keep),encoding='utf-8');os.replace(tmp,p)
    files[name]=dict(original_sha256=sha(original),committed_prefix_sha256=sha(p),discarded_from_active_rows=len(lost),archived_original=str(original))
    if name=='ledger.jsonl':accounting=dict(collections.Counter(json.loads(s)['event'] for s in lost))
assert accounting['UPDATE_COMPLETE']==77
last=json.loads((out/'training_log.jsonl').read_text().splitlines()[-1]);assert last['step']==2000
old_status=read(out/'status.json');assert old_status['step']==2077
write(out/'status.json',dict(status='READY',arm='COMMON',step=2000,pid=0,target=4000,last=last,
    checkpoint=str(out/'rolling.pt'),engineering_recovery='EXPLICIT_USER_AUTHORIZED_RECOMPUTATION_FROM_COMMITTED_CHECKPOINT',
    recovery_record=str(audit/'authorized_recovery_receipt.json'),interrupted_completed_cursor=2077,interrupted_partial_cursor=2078))
new_identity=json.loads(json.dumps(old_identity));new_identity['code_hashes'][str(code)]=sha(code)
write(root/'dispatch/identity.json',new_identity);write(audit/'identity.after.json',new_identity)
(audit/'code.diff').write_text(''.join(difflib.unified_diff((audit/'run_all.before.py').read_text().splitlines(True),code.read_text().splitlines(True),fromfile='run_all.before.py',tofile='run_all.py')),encoding='utf-8')
receipt=dict(status='READY_FOR_AUTHORIZED_RECOVERY',authorization='User explicitly allowed one recovery recomputation and continued monitoring to completion in current conversation',
    reason='Coordinator SIGTERM unexpectedly ended WSL child lifecycle; no successful safe checkpoint was produced',
    source_checkpoint=checkpoint,restored_cursor=2000,lost_completed_updates=77,partial_update=2078,
    original_attempt_counts=accounting,original_files=files,scientific_definition_changed=False,
    original_registration_sha256=old_identity['registration_sha256'],
    changed_code='Coordinator only; CPU GPU-peak guard, five-column telemetry, gallery completion guard',
    train_code_sha256=old_identity['code_hashes'][str(code.with_name('train.py'))],
    prior_exact_replay_verified=False,replay_verification='Compare all available scalar losses, SSE/counts, gradient norms and exposures for2001..2077 after resume; report differences honestly',
    extra_compute_budget_authorized='77 completed optimizer updates plus partial2078 beyond the original unique-update schedule; no endpoint extension',
    model_forwards_in_this_recovery=0,unix=time.time())
write(audit/'authorized_recovery_receipt.json',receipt)
with (root/'dispatch/events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='EXPLICIT_USER_AUTHORIZED_ENGINEERING_RECOVERY',receipt=str(audit/'authorized_recovery_receipt.json'),unix=time.time()))+'\n')
print(json.dumps(receipt))
