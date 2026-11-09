"""CPU file/sidecar inventory. Never import torch or deserialize checkpoints.

Live mode lists sidecar-declared identities without hashing large weights.
Final mode, after a strict final accounting audit, hashes every local checkpoint
file once and compares sidecars. Original START is referenced, never copied;
its identity can be reused from that final accounting audit with explicit scope.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'native_t1/runs/chair_control_feedback_auto_v1'
START=ROOT/'native_t1/runs/chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt'
START_SHA='d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'


def resolve(value):
    text=str(value).replace('\\','/')
    for prefix in ('E:/MeshFlow/','e:/MeshFlow/','/mnt/e/MeshFlow/'):
        if text.startswith(prefix):return ROOT/text[len(prefix):]
    path=Path(text)
    return path if path.is_absolute() else ROOT/path


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def stat(path):
    value=Path(path).stat()
    return dict(bytes=value.st_size,mtime_ns=value.st_mtime_ns,inode=value.st_ino)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)


def classify(path,metadata):
    step=metadata.get('feedback_step')
    if path.name.endswith('_latest.pt'):
        return 'ROLLING_RECOVERY',False,'Rolling recovery cursor; latest is a filename, not the highest-step endpoint. It may remain at step 900 after a separately saved step1000 endpoint.'
    if step in (500,1000) and path.name.endswith('_'+str(step)+'.pt'):
        return 'FROZEN_ENDPOINT',True,'Actual named checkpoint at the metadata feedback_step; conditional_total_step=5000+feedback_step.'
    return 'UNCLASSIFIED_OR_INCOMPLETE',False,'Not a registered named500/1000 endpoint or rolling recovery file; not certified usable by this inventory.'


def _final_audit(out):
    path=out/'derived/audit/final_latest.json'
    if not path.exists():raise RuntimeError('Strict final accounting audit required before large-checkpoint final hashing')
    value=read(path)
    if value.get('mode')!='FINAL' or value.get('status')!='PASS' or value['calls']['pending'] or value['GPU_occupancy']['open_workers']:
        raise RuntimeError('Strict final accounting is not complete/PASS')
    registered={resolve(r['path']):r['sha256'] for r in value['calls']['sources']}
    current=set((out/'ledger').glob('calls_*.jsonl'))
    if set(registered)!=current or any(sha(p)!=expected for p,expected in registered.items()):
        raise RuntimeError('Calls changed after final audit; do not inventory an active experiment as final')
    occupancy=value['GPU_occupancy']['source']
    if sha(resolve(occupancy['path']))!=occupancy['sha256']:raise RuntimeError('GPU ownership changed after final audit')
    return path,value


def run(out=OUT,*,live=False):
    out=resolve(out);started=time.perf_counter();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    auditpath,audit=(None,None) if live else _final_audit(out)
    directory=out/'checkpoints';rows=[];issues=[];hashed_bytes=0;hashed_files=0
    files=set(directory.glob('*.pt'))|set(directory.glob('*.pt.tmp'))
    sidecars={p.with_suffix('.pt'):p for p in directory.glob('*.json') if 'metadata' in read(p)}
    for path in sorted(files|set(sidecars)):
        side=sidecars.get(path);payload=read(side) if side else {};metadata=payload.get('metadata',{})
        kind,endpoint,description=classify(path,metadata);exists=path.exists();before=stat(path) if exists else None
        expected=payload.get('checkpoint_sha256');actual=None
        row=dict(path=str(path),filename=path.name,exists=exists,role=kind,is_endpoint=endpoint,
            phase=metadata.get('phase'),arm=metadata.get('arm'),feedback_step=metadata.get('feedback_step'),
            conditional_total_step=metadata.get('conditional_total_step'),file_stat=before,
            bytes=before['bytes'] if before else None,sidecar_path=str(side) if side else None,
            sidecar_sha256=sha(side) if side else None,sidecar_declared_checkpoint_sha256=expected,
            actual_checkpoint_sha256=None,checkpoint_payload_loaded=False,hash_recomputed=False,
            metadata=metadata,description=description,
            tensor_validation_scope='No tensor/AdamW loading by this inventory. Existing checkpoint.save/load validation verifies full model/optimizer/RNG/stream hashes and AdamW step5000+feedback_step during saving/loading. A sidecar binds declared hashes; file readback occurs after sidecar creation but before save() returns. This inventory does not repeat tensor validation.')
        metadata_ok=bool(side and exists and resolve(payload.get('path',''))==path and
            isinstance(metadata.get('feedback_step'),int) and metadata.get('conditional_total_step')==5000+metadata['feedback_step'] and
            metadata.get('source_checkpoint_sha256')==START_SHA and metadata.get('full_recovery_state') is True and
            metadata.get('optimizer_reset') is False and all(metadata.get(k) for k in
                ('model_state_sha256','optimizer_state_sha256','rng_sha256','stream_state_sha256')))
        row['metadata_consistent']=metadata_ok
        if not live and exists:
            actual=sha(path);hashed_bytes+=before['bytes'];hashed_files+=1
            row.update(actual_checkpoint_sha256=actual,hash_recomputed=True)
            if stat(path)!=before:issues.append(dict(kind='CHECKPOINT_CHANGED_DURING_HASH',path=str(path)))
        if live:
            row['status']='LIVE_METADATA_ONLY_NOT_REHASHED' if metadata_ok and kind!='UNCLASSIFIED_OR_INCOMPLETE' else 'LIVE_INCOMPLETE_OR_WRITE_IN_PROGRESS'
        else:
            row['status']='PASS' if metadata_ok and expected==actual and kind!='UNCLASSIFIED_OR_INCOMPLETE' else 'FAIL'
            if row['status']!='PASS':issues.append(dict(kind='CHECKPOINT_IDENTITY_OR_METADATA_FAILURE',path=str(path),metadata_consistent=metadata_ok,
                expected_sha256=expected,actual_sha256=actual,role=kind))
        if side and sha(side)!=row['sidecar_sha256']:
            issues.append(dict(kind='SIDECAR_CHANGED_DURING_INVENTORY',path=str(side),allowed_in_live=live))
        evidence=[]
        if metadata.get('phase') and metadata.get('arm'):
            pointer=out/'training'/(metadata['phase']+'_'+metadata['arm']+'_pointer.json')
            if pointer.exists():
                saved_pointer=read(pointer)
                if saved_pointer.get('checkpoint_sha256')==expected and saved_pointer.get('metadata')==metadata:
                    evidence.append(dict(kind='POINTER_WRITTEN_AFTER_SAVE_READBACK_RETURN',path=str(pointer),sha256=sha(pointer)))
        row['save_completion_evidence']=evidence
        rows.append(row)
    start=dict(path=str(START),role='PROTECTED_START_REFERENCE',is_endpoint=False,in_archive=False,copied=False,
        feedback_step=0,conditional_total_step=5000,original_hybrid_pilot_step=1000,
        expected_sha256=START_SHA,exists=START.exists(),file_stat=stat(START) if START.exists() else None,
        checkpoint_payload_loaded=False,hash_recomputed=False,bytes=START.stat().st_size if START.exists() else None)
    if live:start.update(status='LIVE_REFERENCE_NOT_REHASHED',actual_checkpoint_sha256=None,
        identity_evidence='Pinned original START identity; no current byte-hash claim in live mode.')
    else:
        evidence=audit['protected_sources']['START']
        if resolve(evidence['path'])==START and evidence.get('status')=='PASS' and evidence.get('actual_sha256')==START_SHA:
            start.update(status='PASS_REFERENCED_FINAL_AUDIT',actual_checkpoint_sha256=START_SHA,
                identity_evidence='Original START full bytes were hashed by strict final accounting; reused explicitly rather than rereading the large weight.',
                audit_path=str(auditpath),audit_sha256=sha(auditpath))
        else:
            actual=sha(START);hashed_bytes+=START.stat().st_size;hashed_files+=1
            start.update(status='PASS' if actual==START_SHA else 'FAIL',actual_checkpoint_sha256=actual,hash_recomputed=True)
            if actual!=START_SHA:issues.append(dict(kind='PROTECTED_START_HASH_FAILURE'))
    endpoints=[r for r in rows if r['is_endpoint']];rolling=[r for r in rows if r['role']=='ROLLING_RECOVERY']
    byarm=[]
    for phase,arm in sorted({(r['phase'],r['arm']) for r in rows if r['phase'] and r['arm']}):
        group=[r for r in rows if r['phase']==phase and r['arm']==arm]
        steps=sorted(r['feedback_step'] for r in group if r['is_endpoint'])
        recovery=[r['feedback_step'] for r in group if r['role']=='ROLLING_RECOVERY']
        byarm.append(dict(phase=phase,arm=arm,actual_endpoint_steps=steps,rolling_recovery_steps=recovery,
            highest_saved_feedback_step=max([r['feedback_step'] for r in group]),
            latest_filename_is_not_terminal=bool(recovery and steps and max(recovery)<max(steps))))
    errors=[x for x in issues if not (live and x.get('allowed_in_live'))]
    report=dict(version='chair_feedback_checkpoint_inventory_v1',mode='LIVE' if live else 'FINAL',
        status='BLOCKED' if errors else 'LIVE_INVENTORY_ONLY' if live else 'PASS',
        snapshot_utc=stamp,seconds=time.perf_counter()-started,START=start,checkpoints=rows,by_arm=byarm,
        actual_checkpoint_files=sum(r['exists'] for r in rows),named_endpoints=len(endpoints),rolling_recovery_files=len(rolling),
        actual_weight_hashes_recomputed=hashed_files,actual_weight_bytes_hashed=hashed_bytes,
        torch_imported=False,model_calls=0,checkpoint_tensors_loaded=False,issues=issues,
        source_sha256=sha(__file__),meaning='Inventory and file identity only. It does not certify model quality, select a checkpoint, or make rolling recovery files into experiment endpoints.')
    dest=out/'derived/checkpoints';mode='live' if live else 'final'
    write(dest/(mode+'_'+stamp+'.json'),report);write(dest/'checkpoint_manifest.json',report)
    flat=[]
    for r in [start]+rows:
        flat.append({k:r.get(k) for k in ('path','role','phase','arm','feedback_step','conditional_total_step','is_endpoint','status','bytes',
            'sidecar_path','sidecar_sha256','sidecar_declared_checkpoint_sha256','actual_checkpoint_sha256','hash_recomputed','metadata_consistent','description')})
    with (dest/'checkpoint_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps({k:report[k] for k in ('status','mode','actual_checkpoint_files','named_endpoints','rolling_recovery_files',
        'actual_weight_hashes_recomputed','actual_weight_bytes_hashed','by_arm')}))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(OUT));parser.add_argument('--live',action='store_true')
    args=parser.parse_args();result=run(args.out,live=args.live)
    raise SystemExit(2 if result['status']=='BLOCKED' else 0)
