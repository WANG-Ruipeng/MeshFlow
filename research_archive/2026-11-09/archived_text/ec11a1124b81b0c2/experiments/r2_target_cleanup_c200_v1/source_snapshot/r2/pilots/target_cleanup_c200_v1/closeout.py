"""Package small evidence and verify completed execution without model calls."""
import difflib, json, shutil, zipfile
from pathlib import Path
from . import common as C

def main():
    destination=C.OUT
    assert (destination/'REPORT.md').is_file()
    r=C.read(destination/'registration.json');e=C.read(destination/'evaluation_freeze.json')
    protected={**r['protected'],**r['training_code_hashes'],**e['evaluation_code_hashes']}
    for p,h in protected.items():assert C.sha(p)==h,p
    weights=[]
    for arm in C.ARMS:
        s=C.read(destination/'training'/arm/'status.json');assert s['status']=='COMPLETE' and s['local_step']==200
        assert not s['calls']['failed'] and not s['calls']['pending_call_ids']
        for kind in ('checkpoint','inference_export'):
            rec=s[kind];assert C.sha(rec['path'])==rec['sha256'];weights.append({**rec,'arm':arm,'kind':kind})
    C.write(destination/'WEIGHTS.json',dict(status='RETAINED',weights=weights,no_intermediate_weights=True))
    execution=[]
    for arm in C.ARMS:
        raw=[C.read(destination/'evaluation/raw'/arm/q['request_id']/'receipt.json') for q in r['requests']]
        post=[C.read(destination/'evaluation/post'/arm/q['request_id']/'receipt.json') for q in r['requests']]
        assert len(raw)==len(post)==24
        assert sum(x['new_rollouts'] for x in raw)==24 and sum(x['new_sampling_F'] for x in raw)==1200
        assert sum(x['new_D0_F'] for x in post)==24 and sum(x['GEO_calls'] for x in post)==24
        for x in raw+post:
            assert x['status']=='COMPLETE'
            for field in ('raw','post'):
                if field+'_path' in x:assert C.sha(x[field+'_path'])==x[field+'_sha256']
        execution.append(dict(arm=arm,RAW=24,POST=24,sampling_forwards=1200,D0_forwards=24,GEO_calls=24))
    C.write(destination/'evaluation_execution_audit.json',dict(status='PASS',rows=execution,additional_model_calls=0))
    sourcepaths=set(r['training_code_hashes'])|set(e['evaluation_code_hashes'])
    for folder in (C.CODE,C.ROOT/'r2/tools/target_cleanup_v1',C.ROOT/'r2/tools/target_cleanup_training_audit_v1',C.ROOT/'r2/tools/d0_restore_quality_v1'):
        sourcepaths.update(str(p) for p in folder.rglob('*.py'))
    sources=[];diff=[]
    for raw in sorted(sourcepaths):
        p=C.resolve(raw)
        rel=p.relative_to(C.ROOT) if p.is_relative_to(C.ROOT) else Path('external')/p.relative_to('/mnt/e')
        target=C.checked(destination/'source_snapshot'/rel);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(p,target)
        assert C.sha(target)==C.sha(p)
        sources.append(dict(path=str(p),sha256=C.sha(p),snapshot=str(target)))
        if p.is_relative_to(C.CODE):diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='b/'+str(rel)))
    C.csvwrite(destination/'source_hashes.csv',sources)
    C._write_bytes(destination/'code.diff',''.join(diff).encode('utf-8'))
    C.write(destination/'protection_audit.json',dict(status='PASS',protected_files=len(protected),all_unchanged=True,model_calls=0))
    folders=[destination,C.ROOT/'experiments/r2_target_cleanup_v1',C.ROOT/'experiments/r2_d0_restore_quality_v1',C.ROOT/'experiments/r2_quality_followup_readiness_v1']
    allowed={'.md','.csv','.py','.diff','.html'};chosen=[]
    for folder in folders:
        for p in sorted(folder.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:continue
            if p.suffix in allowed:chosen.append(p)
    # Include all newly rendered paired figures and scientific learning curves.
    chosen.extend(sorted((destination/'evaluation/gallery').glob('*.jpg')))
    chosen.extend(sorted((destination/'training_audit').rglob('*.png')))
    for p in (destination/'registration.json',destination/'evaluation_freeze.json',destination/'WEIGHTS.json',destination/'protection_audit.json',destination/'evaluation_execution_audit.json',destination/'packaging_recovery.json'):
        chosen.append(p)
    for p in (folders[1]/'data_manifest.json',folders[1]/'datasets/ORIGINAL/train_manifest.json',folders[1]/'datasets/DEDUP/train_manifest.json',
              folders[2]/'completion.json',folders[3]/'readiness.json',destination/'training_audit/completion.json'):
        if p.is_file():chosen.append(p)
    for arm in C.ARMS:
        chosen.extend([destination/'training'/arm/'status.json',destination/'training'/arm/'training_log.jsonl',destination/'training'/arm/'exposure.jsonl'])
    chosen=sorted(set(chosen));archive=C.checked(destination/'R2_QUALITY_TARGET_CLEANUP_C200_REVIEW.zip')
    if archive.exists():raise FileExistsError('Evidence archive already exists; verify rather than overwrite')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in chosen:z.write(p,str(p.relative_to(C.ROOT)))
    C.write(destination/'archive_manifest.json',dict(path=str(archive),sha256=C.sha(archive),bytes=archive.stat().st_size,
        files=[dict(path=str(p.relative_to(C.ROOT)),sha256=C.sha(p),bytes=p.stat().st_size) for p in chosen],
        exclusions='No weights or raw NPZ in small review package. Full data/geometry/weights remain at recorded local paths. D0 diagnostic full gallery remains local.'))
    final=[destination/'REPORT.md',destination/'WEIGHTS.json',destination/'analysis/summary.csv',destination/'evaluation/tables/per_output.csv',
           destination/'evaluation/gallery/actual_visual_review.csv',destination/'training_audit/completion.json',archive]
    final.extend(C.resolve(x['path']) for x in weights)
    assert all(p.is_file() for p in final)
    C.write(destination/'completion.json',dict(status='COMPLETE',deliverable_hashes={str(p):C.sha(p) for p in final},
        additional_training_updates=400,new_train_forwards=3200,new_train_backwards=3200,new_sampling_forwards=2400,new_D0_forwards=48,
        third_item='PREPARATION_ONLY',no_default_switch=True,no_commit_or_push=True))
    print(json.dumps(dict(status='COMPLETE',archive=str(archive),bytes=archive.stat().st_size)))

if __name__=='__main__':main()
