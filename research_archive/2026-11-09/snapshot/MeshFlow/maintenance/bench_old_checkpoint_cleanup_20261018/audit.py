"""Read-only checkpoint/dependency inventory; never deletes or loads a model."""
import csv
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = ROOT/'experiments/r2_bench_control_transfer_v1'
NEW = ROOT/'experiments/r2_bench_parts_convergence_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(path):
    s=str(path).replace('\\','/')
    return s.replace('E:/MeshFlow','/mnt/e/MeshFlow')


def windows(path):
    return canonical(path).replace('/mnt/e/MeshFlow','E:/MeshFlow').replace('/','\\')


def resolve(path):
    return Path(canonical(path)) if os.name!='nt' else Path(windows(path))


def leaves(obj, trail=''):
    if isinstance(obj,dict):
        for key,value in obj.items():
            yield trail+'/'+str(key),str(key)
            yield from leaves(value,trail+'/'+str(key))
    elif isinstance(obj,list):
        for i,value in enumerate(obj):yield from leaves(value,trail+'/'+str(i))
    elif isinstance(obj,str):yield trail,obj


def write(path,value):
    if path.exists():raise FileExistsError('Audit evidence already exists: '+str(path))
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    started=time.time()
    candidates=[]
    for path in sorted((OLD/'training').rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ('.pt','.ckpt','.pth','.safetensors'):continue
        resolved=path.resolve()
        if not resolved.is_relative_to((OLD/'training').resolve()):raise ValueError('Target escapes old training directory')
        side=path.with_suffix('.json');meta=read(side)
        if canonical(meta['path'])!=canonical(resolved):raise ValueError('Sidecar target differs')
        st=path.stat()
        candidates.append(dict(path=windows(resolved),wsl_path=canonical(resolved),bytes=st.st_size,
            mtime_ns=st.st_mtime_ns,device=st.st_dev,inode=st.st_ino,hardlink_count=st.st_nlink,
            sha256=meta['sha256'],sha256_origin='Previously recorded and protected through final rolling retirement; NOT recomputed over large weight in this read-only audit',
            hash_sidecar=windows(side),hash_sidecar_sha256=sha(side),arm=path.parent.name,step=meta['step'],
            action='PROPOSE_DELETE_EXACT_FILE_ONLY',old_capability_retired='Old Bench model inference and exact optimizer resume at this checkpoint',
            new_experiment_weight_dependency=False))
    if len(candidates)!=4 or {(r['arm'],r['step']) for r in candidates}!={('PLAIN',1000),('PLAIN',3000),('SEAM',1000),('SEAM',3000)}:
        raise ValueError('Inventory differs from reviewed four old checkpoints; no automatic expanded selection')
    wanted={r['wsl_path'] for r in candidates}
    source_docs=[NEW/'registration.json',NEW/'data/manifest.json']
    for path in (NEW/'panel.json',NEW/'panel_manifest.json',NEW/'evaluation_manifest.json',NEW/'fixed_probe_manifest.json'):
        if path.exists():source_docs.append(path)
    matches=[];protected_sources=set();doc_records=[]
    for path in source_docs:
        obj=read(path);strings=list(leaves(obj));doc_records.append(dict(path=windows(path),sha256=sha(path),string_entries=len(strings)))
        for key,value in strings:
            norm=canonical(value)
            if any(weight in norm for weight in wanted):matches.append(dict(document=windows(path),key=key,value=value))
            if '/r2_bench_control_transfer_v1/data/cases/' in norm and norm.endswith('.npz'):
                protected_sources.add(norm)
    code_matches=[]
    for path in sorted((ROOT/'r2/pilots/bench_parts_convergence_v1').glob('*.py')):
        for number,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
            if any(weight in canonical(line) for weight in wanted):code_matches.append(dict(path=windows(path),line=number,text=line))
    process_rows=[]
    if os.name!='nt':
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
            try:
                args=(proc/'cmdline').read_bytes().split(b'\0')
                args=[a.decode('utf-8','replace') for a in args if a]
            except (OSError,PermissionError):continue
            if not args or 'python' not in Path(args[0]).name.lower():continue
            text=' '.join(args)
            if 'MeshFlow' not in text and 'r2.pilots.' not in text:continue
            process_rows.append(dict(pid=int(proc.name),argv=args,
                candidate_checkpoint_references=[p for p in wanted if p in canonical(text)],
                old_bench_scientific_worker=('r2.pilots.bench_control_transfer_v1.' in text)))
    current_reg=read(NEW/'registration.json')
    protected=[dict(path=windows(resolve(current_reg['official'])),expected_sha256=current_reg['official_sha256'],kind='OFFICIAL_BENCH_GENERATOR'),
               dict(path=windows(ROOT/'checkpoints/official_bench_denoiser/0021000.pt'),expected_sha256=current_reg['evaluation']['D0_sha256'],kind='OFFICIAL_BENCH_D0')]
    protected.extend(dict(path=windows(resolve(path)),sha256=sha(resolve(path)),bytes=resolve(path).stat().st_size,kind='OLD_SOURCE_CASE_REQUIRED_BY_NEW_DATA') for path in sorted(protected_sources))
    receipt_source=OLD/'cleanup/final_rolling_retirement/cleanup_receipt.json'
    retirement=read(receipt_source)
    # The prior receipt retains the before/after protection checks. Preserve
    # its identity rather than claim a new large-checkpoint byte scan here.
    summary=dict(schema='bench_old_checkpoint_cleanup_readonly_audit_v1',status='READY_FOR_ROOT_EXACT_FILE_CLEANUP_REVIEW',
        planned_files=len(candidates),planned_logical_bytes=sum(r['bytes'] for r in candidates),
        planned_GiB=sum(r['bytes'] for r in candidates)/(1024**3),deleted_files=0,deleted_bytes=0,
        new_manifest_checkpoint_references=matches,new_sourcecode_checkpoint_references=code_matches,
        process_snapshot=process_rows,process_scope='All readable WSL Python scientific-worker argv; Windows host command lines not audited here',
        old_Bench_processes_running=any(r['old_bench_scientific_worker'] for r in process_rows),
        current_candidate_process_references=[r for r in process_rows if r['candidate_checkpoint_references']],
        new_official_initialization=current_reg['initialization'],new_fork=current_reg['fork'],
        new_data_requires_old_source_cases=len(protected_sources),dependency_documents=doc_records,
        protected_scope=['All official Bench/Chair weights','All retained Chair positive checkpoints','All old data/raw/post/reports/scores/code and sidecars','All new experiment assets'],
        previous_rolling_retirement=dict(path=windows(receipt_source),sha256=sha(receipt_source),status=retirement.get('status')),
        note='Only four old training checkpoint files are candidates. Old source data and imported old code remain live dependencies. Removing these checkpoints retires old model replay, not saved-output analysis.',
        model_calls=0,GPU_calls=0,wall_seconds=time.time()-started)
    if matches or code_matches or summary['old_Bench_processes_running'] or summary['current_candidate_process_references']:
        summary['status']='BLOCKED_DEPENDENCY_REVIEW'
    write(OUT/'deletion_manifest.json',dict(status='PROPOSED_NOT_EXECUTED',files=candidates,expected_count=4,
        expected_logical_bytes=sum(r['bytes'] for r in candidates),allowed_root=windows(OLD/'training'),recursive_delete=False))
    write(OUT/'protected_dependencies.json',dict(files=protected,official_large_hashes_recomputed=False,source_case_hashes_recomputed=True))
    write(OUT/'audit_summary.json',summary)
    write(OUT/'cleanup_receipt_DRAFT.json',dict(status='DRAFT_NOT_EXECUTED',authorized_scope='User requested removal of unused old checkpoints, especially old Bench pilot; no dataset deletion',
        files=candidates,actual_deleted_files=0,actual_deleted_bytes=0,planned_logical_bytes=sum(r['bytes'] for r in candidates),
        required_execution_checks=['Resolve each exact absolute target within E:\\MeshFlow\\experiments\\r2_bench_control_transfer_v1\\training',
            'Recheck path/bytes and selected SHA before per-file Remove-Item -LiteralPath; no recursive delete',
            'Recheck no active old worker or new dependency changed',
            'Verify all protected source-case hashes and official identities remain',
            'Record actual deletion, free-space measurements and explicit old replay retirement without modifying old reports']))
    with (OUT/'deletion_manifest.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        cols=('path','bytes','sha256','arm','step','hash_sidecar','action')
        writer=csv.DictWriter(stream,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(candidates)
    print(json.dumps({k:summary[k] for k in ('status','planned_files','planned_logical_bytes','planned_GiB','new_data_requires_old_source_cases','old_Bench_processes_running','deleted_files')},indent=2))


if __name__=='__main__':main()
