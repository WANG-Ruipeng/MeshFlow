"""Validate the second recovery by default; never starts a worker.

Production edits require BOTH --apply and a matching explicit user approval
JSON passed by --authorization-record. Directory archival is deliberately not
implemented: root must perform the verified native PowerShell move first.
"""
from __future__ import annotations
import argparse
import collections
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import psutil

AUDIT=Path(__file__).resolve().parent
ROOT=AUDIT.parents[1]
REPO=ROOT.parents[1]
OUT=ROOT/'training/COMMON'
CODE=REPO/'r2/pilots/bench_parts_convergence_v1'
INCIDENT='metadata_read_shutdown_20261019'
CHECKPOINT_SHA='c322eea82d0691a15a04c55e126f3e5820af7d0ad34893a62194cd69fedae377'
STATE_SHA='3503cbc9207c77b5dce5379a33bb4cab8b4c1e286d6c2fcb55d4deb507eec660'
REG_SHA='b477f2dd77f0e1b216a80a86982086d045394d646faf3ce1950400856c1c7677'
PAYLOAD_AUDIT=ROOT/'recovery2_cpu_audit_20261019_1792345993997925922/receipt.json'
JOB=ROOT/'dispatch/jobs/train_COMMON_6000'
JOB_ARCHIVE=ROOT/'dispatch/jobs/train_COMMON_6000__interrupted_attempt0'
FINAL=AUDIT/'authorized_recovery_receipt.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def require(condition,message):
    if not condition:raise RuntimeError(message)
def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
def create_once(path,raw):
    if path.exists():require(path.read_bytes()==raw,'Existing immutable audit differs: '+str(path));return
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def resolved(text):
    value=str(text).replace('E:\\MeshFlow','/mnt/e/MeshFlow').replace('E:/MeshFlow','/mnt/e/MeshFlow') if os.name!='nt' else str(text)
    return Path(value).resolve()


def process_evidence():
    """Identify actual experiment commands; a reused numeric PID is not proof."""
    workers=[];reused=[];uncertain=[]
    modules={'r2.pilots.bench_parts_convergence_v1.'+m for m in ('run_all','train','resume_train','sample','postprocess')}
    scripts={str((CODE/(m+'.py')).resolve()) for m in ('run_all','train','resume_train','sample','postprocess')}
    for process in psutil.process_iter(['pid','name','cmdline','status','create_time']):
        try:
            info=process.info;cmd=info.get('cmdline') or []
            target=any(arg in modules or arg in scripts for arg in cmd)
            same_root=('--root' not in cmd or (cmd.index('--root')+1<len(cmd) and resolved(cmd[cmd.index('--root')+1])==ROOT))
            if target and same_root and info.get('status')!=psutil.STATUS_ZOMBIE:
                workers.append(info)
            if info['pid'] in (336,3398):
                reused.append(dict(info,actual_target_command=bool(target and same_root)))
        except psutil.NoSuchProcess:continue
        except psutil.AccessDenied:
            if process.pid in (336,3398):uncertain.append(dict(pid=process.pid,reason='Cannot inspect recorded target PID command line'))
    require(not uncertain,'Target process ownership cannot be inspected: '+str(uncertain))
    require(not workers,'Live experiment command still owns outputs; do not recover: '+str(workers))
    return dict(active_target_workers=workers,recorded_PID_observations=reused,
                check='Actual module/script plus exact --root command identity; PID reuse alone never blocks or permits recovery',
                scope='WSL process table for WSL-owned experiment workers',unix=time.time())


def authorization(path):
    require(path is not None,'--apply requires --authorization-record FILE')
    path=Path(path).resolve();value=read(path)
    expected=dict(approved=True,actor='user',incident_id=INCIDENT,checkpoint_step=5250,
        checkpoint_sha256=CHECKPOINT_SHA,max_replayed_optimizer_updates=112,
        max_replayed_training_forwards=896,scientific_endpoint_extension=0)
    for key,want in expected.items():require(type(value.get(key))==type(want) and value[key]==want,'Authorization mismatch: '+key)
    for key in ('authorization_text','authorization_source'):
        require(isinstance(value.get(key),str) and bool(value[key].strip()),'Missing explicit user evidence: '+key)
    return dict(path=str(path),sha256=sha(path),record=value)


def validate(apply=False,authorization_file=None):
    require(not FINAL.exists(),'Recovery already prepared; never silently replay preparation')
    require(not (AUDIT/'apply_intent.json').exists(),'Prior apply intent requires explicit completion/rollback audit')
    approval=authorization(authorization_file) if apply else (authorization(authorization_file) if authorization_file else None)
    processes=process_evidence()
    proof=read(PAYLOAD_AUDIT)
    require(proof['status']=='PASS_STEP5250_PAYLOAD_AND_TAIL_ACCOUNTING','Missing CPU payload PASS')
    require(proof['checkpoint_file_sha256']==CHECKPOINT_SHA and proof['generator_state_sha256']==STATE_SHA,'Payload audit identity differs')
    sidecar=read(OUT/'rolling.json')
    require(sidecar['step']==5250 and sidecar['sha256']==CHECKPOINT_SHA and sidecar['generator_state_sha256']==STATE_SHA,'Wrong checkpoint receipt')
    require(sha(OUT/'rolling.pt')==CHECKPOINT_SHA,'Actual rolling checkpoint changed')
    protected={str(OUT/'rolling.pt'):CHECKPOINT_SHA}
    # Bind both independently archived copies and the still-unmodified working
    # evidence to the hashes recorded by the actual CPU payload/tail audit.
    for name in ('status.json','rolling.json','training_log.jsonl','exposure.jsonl','ledger.jsonl'):
        source=OUT/name;archive=AUDIT/('interrupted.'+name)
        expected=proof['snapshots'][str(source)]['sha256']
        require(sha(source)==sha(archive)==expected,'Interrupted source/archive hash differs: '+name)
        protected[str(source)]=expected;protected[str(archive)]=expected
    old_status=read(OUT/'status.json')
    require(old_status['arm']=='COMMON' and old_status['step']==5362 and old_status['status']=='TRAINING','Unexpected interrupted worker status')
    old_identity=read(AUDIT/'dispatch.identity.json')
    require(read(ROOT/'dispatch/identity.json')==old_identity,'Live dispatch identity differs from incident archive')
    require(old_identity['registration_sha256']==REG_SHA and sha(ROOT/'registration.json')==REG_SHA,'Frozen registration changed')
    protected[str(ROOT/'registration.json')]=REG_SHA
    reg=read(ROOT/'registration.json')
    for field in ('manifest','input_schedule','probe_manifest','evaluation_manifest'):
        path=resolved(reg[field]);expected=reg[field+'_sha256']
        require(sha(path)==expected,'Protected data/stream/input manifest changed: '+field);protected[str(path)]=expected
    for path,expected in reg['training_code_hashes'].items():
        path=resolved(path);require(sha(path)==expected,'Protected scientific source changed: '+str(path));protected[str(path)]=expected
    changed={CODE/'run_all.py',CODE/'controller.py'}
    new_identity=copy.deepcopy(old_identity);changes=[]
    for path,expected in old_identity['code_hashes'].items():
        path=resolved(path)
        if path in changed:
            before=AUDIT/(path.name+'.before');require(sha(before)==expected,'Before-source copy differs: '+str(path))
            new=sha(path);new_identity['code_hashes'][str(path)]=new
            changes.append(dict(path=str(path),before=str(before),old_sha256=expected,new_sha256=new))
        else:require(sha(path)==expected,'Unapproved dispatch source changed: '+str(path));protected[str(path)]=expected
    wrapper=CODE/'resume_train.py';require(wrapper.exists(),'Off-grid wrapper missing')
    new_identity['code_hashes'][str(wrapper)]=sha(wrapper)
    # Importing the dispatcher is CPU-only; compare its exact identity shape.
    sys.path.insert(0,str(REPO))
    from r2.pilots.bench_parts_convergence_v1 import run_all as driver
    expected_sources=[CODE/'run_all.py',CODE/'controller.py',*[CODE/(n+'.py') for n in ('train','resume_train','sample','postprocess','evaluation','curves','convergence')]]
    expected_identity=dict(schema=driver.VERSION,registration_sha256=REG_SHA,
        code_hashes={str(p):sha(p) for p in expected_sources},plan=driver.plan(ROOT),no_automatic_scientific_retry=True)
    require(new_identity==expected_identity,'Migrated identity does not exactly match new driver; scientific plan must remain unchanged')
    require(read(ROOT/'dispatch/status.json')==read(AUDIT/'dispatch.status.json'),'Dispatch status changed after failure')
    require(sha(ROOT/'dispatch/failure.json')==proof['snapshots'][str(ROOT/'dispatch/failure.json')]['sha256'],'Failure evidence changed')
    for p in (ROOT/'dispatch/status.json',ROOT/'dispatch/identity.json'):
        protected[str(p)]=sha(p)
    # Original attempt can be present during dry-run. Apply requires the root's
    # separate native PowerShell archival, never a Python directory move.
    require(not (JOB.exists() and JOB_ARCHIVE.exists()),'Both live/archive job directories exist; ambiguous ownership')
    original_attempt=(JOB if JOB.exists() else JOB_ARCHIVE)/'attempt.json'
    require(original_attempt.exists(),'Missing original interrupted child attempt')
    attempt=read(original_attempt)
    require(attempt['key']=='train_COMMON_6000' and attempt['arm']=='COMMON' and attempt['start_step']==4000 and attempt['target']==6000,'Wrong interrupted job identity')
    frozen_attempt=AUDIT/'train_COMMON_6000.attempt.before.json'
    if frozen_attempt.exists():require(sha(frozen_attempt)==sha(original_attempt),'Job archival content differs')
    if apply:require(not JOB.exists() and JOB_ARCHIVE.exists(),'Native directory archival must be completed before --apply')
    rawlogs={};prefix={};tail={};rows={};log_records={}
    for name in ('training_log.jsonl','exposure.jsonl','ledger.jsonl'):
        raw=(OUT/name).read_bytes();rawlogs[name]=raw;keep=[];drop=[];parsed=[]
        for line in raw.splitlines(keepends=True):
            if not line.strip():continue
            row=json.loads(line.decode('utf-8-sig'));parsed.append(row)
            (keep if int(row.get('step',0))<=5250 else drop).append(line)
        prefix[name]=b''.join(keep);tail[name]=b''.join(drop);rows[name]=parsed
        log_records[name]=dict(original_sha256=digest(raw),prefix_sha256=digest(prefix[name]),tail_sha256=digest(tail[name]),
            prefix_lines=len(keep),tail_lines=len(drop),archived_original=str(AUDIT/('interrupted.'+name)))
    finished=list(range(5251,5363));metrics=rows['training_log.jsonl'];ledger=rows['ledger.jsonl'];exposure=rows['exposure.jsonl']
    require([r['step'] for r in metrics]==list(range(1,5363)),'Canonical metric cursors are not unique1..5362')
    counts=collections.Counter(r['event'] for r in ledger if r.get('step',0)>5250)
    expected_counts=dict(UPDATE_STARTED=112,FORWARD_STARTED=896,FORWARD_COMPLETE=896,BACKWARD_STARTED=896,
        BACKWARD_COMPLETE=896,OPTIMIZER_STARTED=112,OPTIMIZER_COMPLETE=112,UPDATE_COMPLETE=112)
    require(dict(counts)==expected_counts,'Tail call accounting differs or includes unknown events')
    for event in ('UPDATE_STARTED','UPDATE_COMPLETE','OPTIMIZER_STARTED','OPTIMIZER_COMPLETE'):
        require([r['step'] for r in ledger if r.get('step',0)>5250 and r['event']==event]==finished,'Incomplete/duplicate tail: '+event)
    require(len([r for r in exposure if r['step']>5250])==896,'Tail exposure count differs')
    require(len({(r['step'],r['micro']) for r in exposure})==len(exposure)==5362*8,'Exposure duplication or missing rows')
    last=next(r for r in metrics if r['step']==5250);last={k:v for k,v in last.items() if k not in ('event','unix')}
    new_status=dict(status='READY',arm='COMMON',step=5250,pid=0,target=6000,last=last,checkpoint=str(OUT/'rolling.pt'),
        engineering_recovery='SECOND_EXPLICIT_USER_AUTHORIZED_RECOVERY_FROM_COMMITTED5250',recovery_record=str(FINAL),
        interrupted_completed_cursor=5362,interrupted_partial_cursor=None)
    new_dispatch=copy.deepcopy(read(ROOT/'dispatch/status.json'))
    new_dispatch.update(pid=0,owned_active=[],previous_coordinator_pid=new_dispatch.get('pid'),
        engineering_recovery='PREPARED_AFTER_EXPLICIT_SECOND_USER_AUTHORIZATION',recovery_record=str(FINAL))
    mutations={**{OUT/name:raw for name,raw in prefix.items()},OUT/'status.json':encoded(new_status),
        ROOT/'dispatch/status.json':encoded(new_dispatch),ROOT/'dispatch/identity.json':encoded(new_identity)}
    for path in mutations:require(not path.with_name(path.name+'.recovery2_partial').exists(),'Existing staged recovery file requires audit: '+str(path))
    report=dict(status='DRY_RUN_PASS_NO_PRODUCTION_WRITES' if not apply else 'ALL_PRECONDITIONS_PASS',incident_id=INCIDENT,
        checkpoint_sha256=CHECKPOINT_SHA,restored_cursor=5250,lost_completed_updates=112,partial_updates=0,
        tail_counts=dict(counts),log_files=log_records,changed_dispatch_sources=changes,
        new_wrapper=dict(path=str(wrapper),sha256=sha(wrapper)),registration_sha256=REG_SHA,
        native_directory_archive_complete=not JOB.exists() and JOB_ARCHIVE.exists(),
        required_before_apply=['Explicit second user authorization record','Root native PowerShell directory archival'],
        authorization=approval,process_evidence=processes,protected_hashes=protected,
        planned_new_identity=new_identity,planned_mutations={str(p):digest(b) for p,b in mutations.items()},
        no_GPU_or_model_calls=True,starts_training=False,unix=time.time())
    return report,mutations,tail,original_attempt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');p.add_argument('--authorization-record',type=Path)
    args=p.parse_args();report,mutations,tail,attempt=validate(args.apply,args.authorization_record)
    # All audit-only artifacts remain immutable, including dry-run source diffs.
    create_once(AUDIT/'train_COMMON_6000.attempt.before.json',attempt.read_bytes())
    for change in report['changed_dispatch_sources']:
        source=Path(change['path']);before=Path(change['before']);newhash=change['new_sha256']
        create_once(AUDIT/(source.name+'.after.'+newhash),source.read_bytes())
        diff=''.join(difflib.unified_diff(before.read_text(encoding='utf-8').splitlines(True),source.read_text(encoding='utf-8').splitlines(True),fromfile=before.name,tofile=source.name))
        create_once(AUDIT/(source.name+'.'+newhash+'.diff'),diff.encode('utf-8'))
    if not args.apply:
        path=AUDIT/('dry_run_receipt_'+str(time.time_ns())+'.json');create_once(path,encoded(report))
        print(json.dumps(dict(status=report['status'],receipt=str(path),native_directory_archive_complete=report['native_directory_archive_complete'],tail_counts=report['tail_counts'])));return
    # Recheck live ownership and hashes after potentially slow checkpoint work,
    # before creating any staging file or changing production content.
    report['pre_apply_process_recheck']=process_evidence()
    for text,expected in report['protected_hashes'].items():
        if text==str(OUT/'rolling.pt'):continue # Full hash already checked; no live target writer exists.
        require(sha(Path(text))==expected,'Pre-apply source changed: '+text)
    for text,expected in report['planned_new_identity']['code_hashes'].items():require(sha(Path(text))==expected,'Code changed since validation: '+text)
    for name,raw in tail.items():create_once(AUDIT/('uncommitted.'+name),raw)
    create_once(AUDIT/'identity.after.json',encoded(report['planned_new_identity']))
    create_once(AUDIT/'apply_intent.json',encoded(report))
    staged=[]
    try:
        for path,raw in mutations.items():
            temporary=path.with_name(path.name+'.recovery2_partial')
            with temporary.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            require(sha(temporary)==digest(raw),'Staged content mismatch');staged.append((path,temporary))
        for path,temporary in staged:os.replace(temporary,path)
        for path,raw in mutations.items():require(sha(path)==digest(raw),'Applied content differs: '+str(path))
        result=dict(report,status='READY_FOR_SECOND_AUTHORIZED_RECOVERY',scientific_definition_changed=False,
            offgrid_entrypoint='r2.pilots.bench_parts_convergence_v1.resume_train',extra_offgrid_probe_forwards=0,
            replay_verification='PENDING: compare112 updates and896 exposures after explicit worker resume',
            actual_model_calls=0,applied_unix=time.time())
        create_once(FINAL,encoded(result));print(json.dumps(dict(status=result['status'],receipt=str(FINAL))))
    except BaseException as exc:
        create_once(AUDIT/'apply_failure.json',encoded(dict(error=repr(exc),intent=str(AUDIT/'apply_intent.json'),
            status='PARTIAL_PREPARATION_NEEDS_EXPLICIT_AUDIT_NO_WORKER_START',staged=[dict(path=str(p),temporary=str(t)) for p,t in staged],unix=time.time())))
        raise


if __name__=='__main__':main()
