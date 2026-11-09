"""One small final review ZIP, with explicit local-only raw/checkpoint manifests.

--check only writes a delivery plan. Final mode requires a completed report, a
fresh strict accounting audit, no open real calls/GPU ownership and no running
experiment worker. No model, mesh scoring, rendering or checkpoint loading.
"""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile
import zlib
from . import assets
from . import accounting_audit
from . import bundle_layout

OUT=assets.OUT
CODE=Path(__file__).resolve().parent
ARCHIVE_NAME='CONTROL_FEEDBACK_AUTONOMOUS_REVIEW.zip'
REPORT_NAME='CONTROL_FEEDBACK_AUTONOMOUS_REPORT.md'
MAX_BYTES=9_900_000


def _json(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')


def _sha(data):return hashlib.sha256(data).hexdigest()


def _active_workers():
    """Read process command lines, ignoring unrelated GPU/desktop processes."""
    if os.name=='nt':
        command=['powershell','-NoProfile','-Command',
            'Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress']
        p=subprocess.run(command,capture_output=True,text=True,timeout=20)
        if p.returncode:return dict(status='UNKNOWN',reason=p.stderr)
        raw=json.loads(p.stdout);rows=raw if isinstance(raw,list) else [raw]
        found=[dict(pid=r['ProcessId'],command=r.get('CommandLine')) for r in rows
               if 'chair_control_feedback_auto_v1.worker' in (r.get('CommandLine') or '')]
    else:
        p=subprocess.run(['ps','-eo','pid=,args='],capture_output=True,text=True,timeout=10)
        if p.returncode:return dict(status='UNKNOWN',reason=p.stderr)
        found=[]
        for line in p.stdout.splitlines():
            parts=line.strip().split(None,1)
            if len(parts)==2 and 'chair_control_feedback_auto_v1.worker' in parts[1]:
                found.append(dict(pid=int(parts[0]),command=parts[1]))
    return dict(status='ACTIVE' if found else 'CLEAR',workers=found)


def prerequisites(out,report):
    problems=[];facts={}
    if not report.exists() or not report.read_text(encoding='utf-8-sig').strip():problems.append('Final research report missing or empty')
    auditpath=out/'derived/audit/final_latest.json'
    final=assets.read_json(auditpath) if auditpath.exists() else None
    if not final or final.get('mode')!='FINAL' or final.get('status')!='PASS':
        problems.append('Strict final accounting audit is missing or not PASS')
    else:
        facts['final_audit_sha256']=assets.digest(auditpath)
        for source in final['calls']['sources']:
            p=assets.resolve(source['path'])
            if not p.exists() or assets.digest(p)!=source['sha256']:
                problems.append('Call ledger changed after final audit: '+str(p))
        source=final['GPU_occupancy']['source'];p=assets.resolve(source['path'])
        if not p.exists() or assets.digest(p)!=source['sha256']:problems.append('GPU occupancy ledger changed after final audit')
        if final['calls']['pending'] or final['GPU_occupancy']['open_workers']:problems.append('Final audit still contains pending calls/GPU worker')
        if final['protected_sources']['START'].get('status')!='PASS':problems.append('Final START hash is not verified')
        if final['protected_sources']['source_file_count']!=167 or final['protected_sources']['unchanged_source_file_count']!=167:
            problems.append('Protected 167-file source inventory not intact')
        # The just-written report may be later than the model audit. However no
        # output/metadata, which that audit validated, may change in between.
        for r in final['outputs']['rows']:
            p=assets.resolve(r['result_path'])
            if not p.exists() or assets.digest(p)!=r['result_sha256']:problems.append('Output result changed after audit: '+str(p))
    # Independently re-read calls, even if the saved audit happened to be stale.
    issues=[]
    try:
        calls,*_=accounting_audit.audit_calls(out,False,issues)
        occupancy=accounting_audit.audit_occupancy(out,False,issues,time.time())
        if final and calls['total']!=final['calls']['total']:problems.append('Actual call counts changed after strict final audit')
        if final and {s['path']:s['sha256'] for s in calls['sources']}!={s['path']:s['sha256'] for s in final['calls']['sources']}:
            problems.append('Call ledger set or content changed after strict final audit')
        if final and {str(p) for p in (out/'runtime').glob('*/*/result.json')}!={r['result_path'] for r in final['outputs']['rows']}:
            problems.append('Runtime output set changed after strict final audit')
        if calls['pending']:problems.append('Real call is pending now')
        if occupancy['open_workers']:problems.append('GPU owner has not released the worker lock')
        if issues:problems.append('Current strict ledger checks contain issues')
        facts['current_ledger_issues']=issues
    except Exception as exc:problems.append('Current ledger parsing failed: '+repr(exc))
    inventory=out/'derived/checkpoints/checkpoint_manifest.json'
    if not inventory.exists():problems.append('Final checkpoint inventory missing')
    else:
        inventory_data=assets.read_json(inventory)
        if inventory_data.get('mode')!='FINAL' or inventory_data.get('status')!='PASS':
            problems.append('Checkpoint inventory is live/unverified rather than final PASS')
        else:
            for item in inventory_data['checkpoints']:
                p=assets.resolve(item['path']);before=item.get('file_stat',{})
                if not p.exists() or p.stat().st_size!=before.get('bytes') or p.stat().st_mtime_ns!=before.get('mtime_ns'):
                    problems.append('Checkpoint changed after final inventory: '+str(p))
                side=item.get('sidecar_path')
                if side and assets.digest(assets.resolve(side))!=item.get('sidecar_sha256'):
                    problems.append('Checkpoint sidecar changed after final inventory: '+side)
    processes=_active_workers();facts['process_check']=processes
    if processes['status']!='CLEAR':problems.append('Experiment GPU worker active or process inspection unavailable')
    return problems,facts,final


def _add_file(entries,source,name,category):
    source=Path(source)
    if not source.is_file():return
    if source.suffix.lower() in ('.pt','.pth','.npz','.npy','.ply','.obj','.stl','.glb'):
        raise ValueError('Large/raw geometry or checkpoint not allowed in archive: '+str(source))
    data=source.read_bytes()
    if name in entries and entries[name]['sha256']!=_sha(data):raise ValueError('Archive name collision: '+name)
    entries[name]=dict(data=data,sha256=_sha(data),source_path=str(source),category=category)


def collect(out,report,final):
    """Collect required small evidence; optional images considered separately."""
    entries={};missing=[]
    for name in (report.name,'protocol.json','hypotheses.json','hypothesis_register.md','task_contract.txt','accounting.json','REPRODUCE.md'):
        p=report if name==report.name else out/name
        if not p.exists():missing.append(str(p))
        else:_add_file(entries,p,'review/'+name,'report_protocol')
    # These are the authoritative complete tables. No row filtering or truncation.
    for folder in ('evaluation','comparisons'):
        root=out/folder
        for p in sorted(root.rglob('*.csv')):_add_file(entries,p,'review/'+p.relative_to(out).as_posix(),'complete_CSV')
        for pattern in ('**/per_output.json','**/summary.json','**/gate.json'):
            for p in sorted(root.glob(pattern)):_add_file(entries,p,'review/'+p.relative_to(out).as_posix(),'numeric_summary')
    attribution=out/'derived/attribution/DEV/latest.json'
    if attribution.exists():
        index=assets.read_json(attribution)
        _add_file(entries,attribution,'review/derived/attribution/DEV/latest.json','descriptive_attribution')
        p=assets.resolve(index['path'])
        _add_file(entries,p,'review/'+p.relative_to(out).as_posix(),'descriptive_attribution')
        for item in index['tables'].values():
            p=assets.resolve(item['path'])
            if assets.digest(p)!=item['sha256']:raise ValueError('Attribution table changed')
            _add_file(entries,p,'review/'+p.relative_to(out).as_posix(),'complete_CSV')
    transfer=out/'derived/transfer_attribution/latest.json'
    if transfer.exists():
        index=assets.read_json(transfer)
        for p in [transfer,assets.resolve(index['path']),assets.resolve(index['memo_path'])]+[assets.resolve(x['path']) for x in index['tables'].values()]:
            _add_file(entries,p,'review/'+p.relative_to(out).as_posix(),'complete_CSV' if p.suffix=='.csv' else 'transfer_attribution')
    digest=out/'derived/attribution/DEV/distribution_digest_latest.json'
    if digest.exists():
        index=assets.read_json(digest)
        for p in [digest,assets.resolve(index['path']),assets.resolve(index['memo_path'])]+[assets.resolve(x['path']) for x in index['tables'].values()]:
            _add_file(entries,p,'review/'+p.relative_to(out).as_posix(),'complete_CSV' if p.suffix=='.csv' else 'parent_distribution_digest')
    for name in ('derived/attribution/DEV/matched_map.json','derived/attribution/DEV/matched_map_receipt.json','derived/attribution/DEV/coefficient_ablation_map.json'):
        _add_file(entries,out/name,'review/'+name,'frozen_matched_mapping')
    for folder in ('registration','manifests','decisions','status'):
        for p in sorted((out/folder).rglob('*')):
            if p.suffix.lower() in ('.json','.jsonl','.md'):
                _add_file(entries,p,'review/'+p.relative_to(out).as_posix(),'registration_decisions')
    for name in ('derived/checkpoints/checkpoint_manifest.json','derived/checkpoints/checkpoint_manifest.csv'):
        _add_file(entries,out/name,'review/'+name,'complete_checkpoint_inventory')
    # Conditional future evidence only: no dormant H3 recipe or proof is invented.
    for relative in ('training_diagnostics/final','cap_control_equivalence/final','decisions/H3_control_equivalence_freeze','decisions/H3_evidence_freeze','report_tables'):
        for p in sorted((out/relative).rglob('*')):
            if p.is_file() and p.suffix.lower() in ('.json','.csv','.md','.py','.jsonl'):
                _add_file(entries,p,'review/'+p.relative_to(out).as_posix(),'final_training_or_H3_evidence')
    for p in sorted((out/'checkpoints').glob('*.json')):
        _add_file(entries,p,'review/'+p.relative_to(out).as_posix(),'checkpoint_metadata')
    for p in sorted((out/'audit').glob('*.json')):_add_file(entries,p,'review/'+p.relative_to(out).as_posix(),'audits')
    if final:
        _add_file(entries,out/'derived/audit/final_latest.json','review/derived/audit/final_latest.json','final_accounting')
    for name in ('references/definition_and_eligibility.json','tests/evaluation_cpu_results.json',
                 'feedback/pool_v1/manifest.json','feedback/pool_v1/summary.json'):
        _add_file(entries,out/name,'review/'+name,'frozen_reference_or_tests')
    # Necessary repair receipts/source differences; avoid duplicate old numerical
    # outputs in engineering/originals because those remain explicitly local.
    for p in sorted((out/'engineering').rglob('*')):
        if any(part in ('originals','evaluation_fixture') for part in p.parts):continue
        if p.is_file() and p.suffix in ('.py','.diff','.md','.csv') or p.is_file() and p.name in ('receipt.json','source_identity.json','actual_start_generation.json'):
            _add_file(entries,p,'review/'+p.relative_to(out).as_posix(),'engineering_evidence')
    _add_file(entries,out/'delivery/cpu_tests/layout_receipt.json','review/delivery/cpu_tests/layout_receipt.json','delivery_semantic_tests')
    source_manifest={};diff=[]
    for p in sorted(CODE.rglob('*.py')):
        if '__pycache__' in p.parts:continue
        rel=p.relative_to(assets.ROOT).as_posix();data=p.read_bytes()
        _add_file(entries,p,'source_code/current/'+rel,'new_current_source')
        source_manifest[rel]=dict(sha256=_sha(data),bytes=len(data),local_path=str(p),role='Final current file; executed versions separately archived')
        diff.extend(difflib.unified_diff([],data.decode('utf-8-sig').splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+rel))
    # These worker snapshots are the actually loaded versions, including earlier
    # engineering revisions. Do not mislabel final source as every arm's source.
    for p in sorted((out/'code_snapshots').rglob('*')):
        if p.suffix in ('.py','.json'):_add_file(entries,p,'source_code/executed/'+p.relative_to(out/'code_snapshots').as_posix(),'actual_executed_source')
    sourcefiles={}
    definition=out/'references/definition_and_eligibility.json'
    if definition.exists():
        sources=assets.read_json(definition)['sources']
        for group in ('control','legacy'):sourcefiles.update(sources[group])
        sourcefiles['experiments/mf_h1_local0/chair_control_usability_quick_v1/scoring.py']=sources['quality']
    for rel,expected in sourcefiles.items():
        p=assets.resolve(rel)
        if assets.digest(p)!=expected:raise ValueError('Original scorer source hash changed: '+str(p))
        _add_file(entries,p,'source_scoring/'+Path(rel).as_posix(),'original_actual_scoring_source')
    entries['code.diff']=dict(data=''.join(diff).encode('utf-8'),source_path=None,category='all_new_files_diff',sha256=_sha(''.join(diff).encode('utf-8')))
    for name,value in [('source_code/SOURCE_HASHES.json',source_manifest),('source_scoring/SOURCE_HASHES.json',sourcefiles)]:
        data=_json(value);entries[name]=dict(data=data,sha256=_sha(data),source_path=None,category='source_identity')
    local=[];checkpoints=[]
    if final:
        for r in final['outputs']['rows']:
            result=assets.read_json(assets.resolve(r['result_path']));raw=assets.resolve(r['result_path']).with_name('raw.npz')
            for role,p,sha in [('RAW_WITH_51_STATES',raw,r.get('raw_sha256')),
                ('LOSSLESS_EXPORTED_NPZ',assets.resolve(result['export']['npz']),result['export']['npz_sha256']),
                ('LOSSLESS_BINARY_PLY',assets.resolve(result['export']['ply']),result['export']['ply_sha256'])]:
                local.append(dict(label=r['label'],split=r['split'],request_id=r['request_id'],role=role,
                    path=str(p),sha256=sha,bytes=p.stat().st_size,in_archive=False,
                    identity_evidence='Verified by strict final accounting audit; bytes not duplicated in ZIP'))
        for r in final['checkpoints']['rows']:
            p=assets.resolve(r['path']);record=dict(path=str(p),sha256=r['checkpoint_sha256'],bytes=p.stat().st_size,
                metadata=r['metadata'],in_archive=False,identity_evidence=r['tensor_inspection'])
            checkpoints.append(record)
        start=final['protected_sources']['START'];p=assets.resolve(start['path'])
        checkpoints.append(dict(path=str(p),sha256=start['actual_sha256'],bytes=p.stat().st_size,
            role='Protected original START',in_archive=False,identity_evidence='Final byte hash verified'))
        for s in final['calls']['sources']:
            local.append(dict(role='FULL_REAL_CALL_LEDGER',path=s['path'],sha256=s['sha256'],bytes=s['bytes'],in_archive=False))
        for stage in final['phases']:
            if stage['kind']=='training':
                s=stage['source'];local.append(dict(role='FULL_TRAINING_METRICS',path=s['path'],sha256=s['sha256'],bytes=s['bytes'],in_archive=False))
    visuals=[]
    for p in sorted((out/'visuals').glob('*/manifest.json')):
        r=assets.read_json(p);_add_file(entries,p,'review/'+p.relative_to(out).as_posix(),'full_gallery_manifest')
        contact=r.get('contact_sheet')
        if contact:
            image=assets.resolve(contact)
            if image.exists():
                if r.get('contact_sheet_sha256') and assets.digest(image)!=r['contact_sheet_sha256']:raise ValueError('Contact sheet changed')
                visuals.append(dict(path=image,labels=len(r.get('labels',[])),split=r.get('split'),
                    bytes=image.stat().st_size,manifest=str(p),requests=r.get('requests')))
        local.append(dict(role='FULL_LOCAL_GALLERY',path=str(p.parent),manifest_path=str(p),manifest_sha256=assets.digest(p),
            in_archive=False,note='Only selected whole contact sheets may be included. Individual full-resolution panels remain local.'))
    inventories=dict(raw_exports_and_large_logs=local,checkpoints=checkpoints,
        omitted_materials_policy='No raw NPZ, model/optimizer weights, full call ledgers, full training logs, or complete image galleries are in this ZIP. They are retained locally at the stated paths and hashes.',
        actual_raw_count=len(final['outputs']['rows']) if final else None)
    inventory_path=out/'derived/checkpoints/checkpoint_manifest.json'
    checkpoint_evidence=assets.read_json(inventory_path) if inventory_path.exists() else checkpoints
    for name,value in [('LOCAL_MATERIALS_MANIFEST.json',inventories),('CHECKPOINT_MANIFEST.json',checkpoint_evidence)]:
        data=_json(value);entries[name]=dict(data=data,sha256=_sha(data),source_path=None,category='local_only_inventory')
    return entries,visuals,missing


def _estimate(entries):
    # Conservative enough for planning; actual archive byte size is authoritative.
    return sum(len(zlib.compress(v['data'],9))+250+2*len(k.encode()) for k,v in entries.items())+22


def _encode(entries,path):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9,allowZip64=True) as archive:
        for name,value in sorted(entries.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,6,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16
            archive.writestr(info,value['data'],compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)


def run(out=OUT,*,report=None,check=False):
    out=assets.resolve(out);delivery=out/'delivery';delivery.mkdir(parents=True,exist_ok=True)
    report=assets.resolve(report) if report else out/REPORT_NAME
    problems,facts,final=prerequisites(out,report)
    entries,images,missing=collect(out,report,final)
    logical_entries=dict(entries)
    entries,layout=bundle_layout.compact(entries)
    problems.extend('Missing required evidence: '+x for x in missing)
    required_bytes=_estimate(entries)
    plan=dict(mode='CHECK_ONLY' if check else 'FINAL',status='NOT_READY' if problems else 'READY',
        prerequisite_problems=problems,checks=facts,required_file_count=len(entries),
        required_compressed_estimate_bytes=required_bytes,strict_max_bytes=MAX_BYTES,
        logical_required_file_count=len(logical_entries),exact_duplicate_alias_count=len(layout['index']['exact_duplicate_aliases']),
        lossless_layout_deflate_saved_bytes=layout['deflate_bytes_saved'],unique_evidence_omitted=0,
        optional_contact_sheets=[dict(path=str(x['path']),labels=x['labels'],split=x['split'],bytes=x['bytes']) for x in images],
        raw_or_weight_payloads_included=False,complete_CSV_tables=True,model_calls=0)
    if required_bytes>MAX_BYTES:plan['status']='SIZE_BLOCKED';problems.append('Mandatory evidence exceeds small-package target; CSV rows will not be removed')
    _write=accounting_audit._write
    _write(delivery/'bundle_plan.json',plan)
    _write(delivery/'bundle_layout_plan.json',dict(mode='CHECK_ONLY' if check else 'FINAL',status=plan['status'],layout=layout,
        policy='Preserve every unique source, gate, proof and complete CSV. Exact duplicates have a reconstructable alias map. No report_notes or live diagnostic trees are selected.',
        required_compressed_estimate_bytes=required_bytes,strict_max_bytes=MAX_BYTES))
    if check:
        print(json.dumps({k:plan[k] for k in ('mode','status','required_file_count','required_compressed_estimate_bytes','prerequisite_problems')}))
        return plan
    if problems:raise RuntimeError('Final archive refused: '+'; '.join(problems))
    destination=delivery/ARCHIVE_NAME
    if destination.exists():raise FileExistsError('A review ZIP already exists; refuse an untracked replacement: '+str(destination))
    # Prefer whole comparative contact sheets, never candidate-picked panels.
    selected=[]
    for item in sorted(images,key=lambda x:(-x['labels'],x['split']!='DEV',str(x['path']))):
        if len(selected)>=2:break
        name='review/'+item['path'].relative_to(out).as_posix();data=item['path'].read_bytes()
        entry=dict(data=data,source_path=str(item['path']),sha256=_sha(data),category='optional_whole_contact_sheet')
        trial=dict(entries);trial[name]=entry
        if _estimate(trial)<MAX_BYTES-60_000:entries=trial;selected.append(name)
    statement=(
        '# CONTROL feedback autonomous review\n\n'
        'This is a compact review package, not a complete experiment backup. It contains the final report, complete registered CSV tables, '
        'comparison gates, accounting audit, protocols/manifests, final and actually executed new source, original scoring source, and source hashes.\n\n'
        'RAW outputs with trajectories, checkpoints including optimizer state, full logs, and the full galleries remain locally available. '
        'Their paths and hashes are listed in LOCAL_MATERIALS_MANIFEST.json and CHECKPOINT_MANIFEST.json. No weight or raw-geometry payload is in this ZIP.\n\n'
        'code.diff contains every final new experiment Python file as an addition against /dev/null. Executed worker snapshots preserve versions used before engineering fixes. '
        'The final current source is not presented as the source of all earlier calls.\n\n'
        'Exact byte-identical duplicate source/proof/metadata copies share one physical member. ALIAS_AND_OMISSION_INDEX.json records every logical path, local source path, SHA-256 and canonical member. '
        'Run python RESTORE_ALIASES.py in the extracted folder to restore these copies if needed. Every unique selected review byte sequence and every complete evaluation/parent/noise CSV is retained; no unique selected evidence is dropped.\n')
    data=statement.encode();entries['README.md']=dict(data=data,sha256=_sha(data),source_path=None,category='scope')
    index={name:dict(sha256=v['sha256'],bytes=len(v['data']),category=v['category'],source_path=v['source_path']) for name,v in sorted(entries.items())}
    manifest=dict(status='FINAL_REVIEW_PACKAGE',report_sha256=assets.digest(report),strict_audit_sha256=assets.digest(out/'derived/audit/final_latest.json'),
        package_limit_bytes=MAX_BYTES,all_required_CSV_rows_preserved=True,selected_whole_contact_sheets=selected,files=index,
        logical_required_file_count=len(logical_entries),exact_duplicate_alias_count=len(layout['index']['exact_duplicate_aliases']),
        alias_index='ALIAS_AND_OMISSION_INDEX.json',unique_evidence_omitted=0,
        no_RAW_or_weights=True,source_capture_time_utc=datetime_now(),build_script_sha256=assets.digest(__file__))
    data=_json(manifest);entries['BUNDLE_MANIFEST.json']=dict(data=data,sha256=_sha(data),source_path=None,category='archive_index')
    temp=delivery/(ARCHIVE_NAME+'.inprogress')
    if temp.exists():raise FileExistsError('Unfinished archive exists; inspect it before retrying')
    _encode(entries,temp)
    if temp.stat().st_size>=MAX_BYTES:
        size=temp.stat().st_size;temp.unlink()
        _write(delivery/'bundle_size_blocked.json',dict(status='SIZE_BLOCKED',actual_bytes=size,limit=MAX_BYTES,rows_removed=0))
        raise RuntimeError('Archive exceeds strict size target; preserved all local evidence, final ZIP not published')
    with zipfile.ZipFile(temp) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist())==set(entries)
        for name,v in entries.items():assert _sha(archive.read(name))==v['sha256']
    # Recheck live safety and immutable source bytes immediately before publishing.
    last_problems,_,_=prerequisites(out,report)
    for name,v in {**logical_entries,**entries}.items():
        if v['source_path'] and assets.digest(Path(v['source_path']))!=v['sha256']:last_problems.append('Source changed during packaging: '+name)
    if last_problems:temp.unlink();raise RuntimeError('Final publication refused: '+'; '.join(last_problems))
    (delivery/'code.diff').write_bytes(entries['code.diff']['data'])
    for name,v in logical_entries.items():
        if name.startswith(('source_code/','source_scoring/')):
            p=delivery/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v['data'])
    _write(delivery/'BUNDLE_MANIFEST.json',manifest)
    temp.replace(destination)
    receipt=dict(status='PASS',archive=str(destination),bytes=destination.stat().st_size,sha256=assets.digest(destination),
        maximum_bytes=MAX_BYTES,archive_files=len(entries),selected_whole_contact_sheets=selected,
        required_CSV_files=sum(v['category']=='complete_CSV' for v in logical_entries.values()),all_CSV_rows_preserved=True,
        original_CSV_files=layout['index']['original_CSV_files'],protected_physical_CSV_files=layout['index']['protected_physical_CSV_files'],
        exact_duplicate_aliases=len(layout['index']['exact_duplicate_aliases']),unique_evidence_omitted=0,
        raw_payload_count=0,checkpoint_payload_count=0,model_calls=0)
    _write(delivery/'bundle_receipt.json',receipt);print(json.dumps(receipt));return receipt


def datetime_now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(OUT));parser.add_argument('--report');parser.add_argument('--check',action='store_true')
    args=parser.parse_args();run(args.out,report=args.report,check=args.check)
