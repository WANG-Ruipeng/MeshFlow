from pathlib import Path
import hashlib,json,shutil,difflib
root=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_early_alignment_schedule40_v1')
repo=Path('/mnt/e/MeshFlow-Control')
hash_file=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=['verify_final_checkpoints.py','summarize_fixed_panel.py','capture_final_analysis_sources.py']
manifest={}
for name in paths:
    p=root/'audit'/name
    destination=root/'source_code'/'experiment_audit'/name
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and hash_file(destination)!=hash_file(p): raise ValueError('Existing source copy differs '+name)
    if not destination.exists(): shutil.copyfile(p,destination)
    manifest['experiment_audit/'+name]={'live_path':str(p),'sha256':hash_file(p),'snapshot_path':str(destination)}
q=root/'audit'/'postfreeze_analysis_hashes.json'
if q.exists() and json.loads(q.read_text())!=manifest: raise ValueError('Existing analysis identities differ')
q.write_text(json.dumps(manifest,indent=2)+'\n')
old=root/'audit'/'engineering'/'code_diff_before_final_analysis.diff'
old.parent.mkdir(parents=True,exist_ok=True)
if not old.exists(): shutil.copyfile(root/'code.diff',old)
new_sources=json.loads((root/'audit'/'new_source_hashes.json').read_text())
parts=[]
for relative,expected in new_sources.items():
    p=repo/relative
    if hash_file(p)!=expected: raise ValueError('Frozen new source differs '+relative)
    content=p.read_text().splitlines(keepends=True)
    parts.extend(difflib.unified_diff([],content,fromfile='/dev/null',tofile='b/'+relative))
for relative,record in manifest.items():
    content=Path(record['live_path']).read_text().splitlines(keepends=True)
    parts.extend(difflib.unified_diff([],content,fromfile='/dev/null',tofile='b/'+relative))
(root/'code.diff').write_text(''.join(parts))
print(json.dumps({'analysis_files':len(manifest),'new_repo_files':len(new_sources),'diff_bytes':(root/'code.diff').stat().st_size,'model_calls':0}))
