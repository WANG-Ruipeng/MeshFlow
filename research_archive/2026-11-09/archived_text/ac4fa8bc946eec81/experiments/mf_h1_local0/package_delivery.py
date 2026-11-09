"""Package completed evidence without importing a model or running an experiment."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def run():
    project=Path(__file__).resolve().parent
    repo=project.parents[1]
    out=project/'outputs/run_20260928_01'
    archive=out/'MF_H1_Phase0_2_delivery_v2.zip'
    if archive.exists():raise FileExistsError(archive)
    sources=sorted(p for p in project.rglob('*') if p.is_file()
        and not any(x in {'assets','outputs','__pycache__','results'} for x in p.relative_to(project).parts)
        and p.suffix in {'.py','.md','.yaml','.yml'})
    patch=out/'code.patch'
    chunks=[]
    for p in sources:
        r=subprocess.run(['git','-c','core.autocrlf=false','diff','--no-index','--binary','--','/dev/null',str(p.relative_to(repo))],cwd=repo,capture_output=True)
        if r.returncode not in (0,1):raise RuntimeError(r.stderr.decode())
        chunks.append(r.stdout)
    patch.write_bytes(b''.join(chunks))
    check=subprocess.run(['git','-c','core.autocrlf=false','apply','--reverse','--check','--whitespace=nowarn',str(patch)],cwd=repo,capture_output=True,text=True)
    if check.returncode:raise RuntimeError(check.stderr)
    payload=set(sources)
    payload.update(p for p in (project/'tests/results').rglob('*') if p.is_file())
    for dirname in ['asset_audit_20260928_02','asset_extraction_20260928']:
        payload.update(p for p in (project/'outputs'/dirname).rglob('*') if p.is_file())
    omitted=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file():continue
        relative=p.relative_to(out)
        if p.suffix=='.zip' or p.name.startswith(('artifact_manifest','delivery_verification')):continue
        if p.suffix=='.pt' and p.name!='adapter_step0200.pt':omitted.append(str(p.relative_to(project)));continue
        if any(part in {'phase2_contacts','phase2_contacts_v2'} for part in relative.parts):omitted.append(str(p.relative_to(project)));continue
        payload.add(p)
    raw_npz=list((out/'raw').glob('*.npz'));raw_obj=list((out/'raw').glob('*.obj'))
    assert len(raw_npz)==len(raw_obj)==115
    counts=json.loads((out/'execution_counts.json').read_text())
    assert counts['phase2_optimizer_updates']==400 and counts['official_backbone_forward_calls']==7479
    assert counts['complete_rollouts']==115 and counts['phase3_optimizer_updates']==0
    manifest=dict(schema_version=1,scope='Completed Phase 0-2 evidence and source; no model execution by this packager',
        paths_relative_to='experiments/mf_h1_local0',
        omitted_categories=['Official 2GB checkpoint and original/extracted data assets are not packaged; see report paths and hashes','Earlier render layout drafts and intermediate/disposable probe checkpoints remain locally available'],
        omitted_run_files=omitted,source_files=len(sources),raw_npz_count=len(raw_npz),raw_obj_count=len(raw_obj),
        files=[dict(path=str(p.relative_to(project)),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(payload)])
    mf=out/'artifact_manifest_v2.json';mf.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    payload.add(mf)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(payload):z.write(p,str(p.relative_to(repo)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP CRC failed')
        for row in manifest['files']:
            name='experiments/mf_h1_local0/'+row['path']
            if hashlib.sha256(z.read(name)).hexdigest()!=row['sha256']:raise RuntimeError('ZIP hash mismatch: '+name)
    tracked=subprocess.run(['git','-c','core.autocrlf=true','diff','--exit-code','--'],cwd=repo,capture_output=True,text=True)
    if tracked.returncode:raise RuntimeError('Tracked original source differs')
    verification=dict(status='PASS',patch_reverse_check='PASS_CHECK_ONLY_NO_FILES_CHANGED',patch_sha256=sha256(patch),
        archive=str(archive.relative_to(repo)),archive_bytes=archive.stat().st_size,archive_sha256=sha256(archive),
        zipped_files=len(payload),zip_crc='PASS',every_payload_sha256='PASS',tracked_original_files_unchanged=True,
        actual_counts=counts,packaging_model_forwards=0,packaging_optimizer_updates=0,
        archive_note='artifact_manifest_v2.json included; this final archive hash verification is a sidecar to avoid a self-referential hash')
    (out/'delivery_verification_v2.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification,ensure_ascii=False))


if __name__=='__main__':run()
