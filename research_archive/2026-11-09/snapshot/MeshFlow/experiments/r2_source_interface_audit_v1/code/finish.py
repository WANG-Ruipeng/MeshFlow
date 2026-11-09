"""Finalize only this audit; include text evidence, no model assets or mesh arrays."""
from pathlib import Path
import csv,json,hashlib,zipfile,difflib,subprocess,sys,time
OUT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def csvinfo(p):
    with p.open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f);x=list(r)
        return len(x),r.fieldnames
def main():
    done=read(OUT/'completion.json');ind=read(OUT/'protocol/independent_validation.json')
    assert done['pass_count']==258 and done['status']=='COMPLETE'
    # Independent status schema is preserved, not rewritten.
    print('Independent status:',ind.get('status'))
    assert 'PASS' in str(ind.get('status',''))
    reg=read(OUT/'protocol/registration.json')
    assert sha(OUT/'code/audit.py')==reg['entry_sha256']
    for p,h in reg['protected'].items():assert sha(p)==h,p
    run=subprocess.run([sys.executable,'-B',str(OUT/'code/resume.py')],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    receipt=json.loads(run.stdout);assert receipt['status']=='ALREADY_COMPLETE_VERIFIED'
    (OUT/'protocol/resume_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    # Exact additions; the frozen original scoring snapshots have no changes.
    scripts=[OUT/'code/audit.py',OUT/'code/readout.py',OUT/'code/resume.py',Path(__file__)]
    patch=''
    for p in scripts:
        patch+=''.join(difflib.unified_diff([],p.read_text('utf-8').splitlines(keepends=True),
            fromfile='/dev/null',tofile=str(p.relative_to(OUT)),lineterm='\n'))
    (OUT/'code.diff').write_text(patch,encoding='utf-8')
    (OUT/'EXECUTION.json').write_text(json.dumps(dict(status='COMPLETE',source_conditions=258,source_parents=99,
        new_boundary_nearest_jobs=258,new_valid_graph_calls=1032,new_fixed_C_orientation_diagnostics=516,
        topology_domains=2,source_meshes_modified=0,new_triangle_pair_tests=0,models=0,training=0,D0=0,GEO=0,render=0,
        CPU_workers=done['workers'],threads_per_worker=1,GPU_workers=0,CPU_run_wall_seconds=done['wall_seconds'],
        maximum_worker_process_peak_RSS_bytes=max(r['peak_RSS'] for r in done['receipts']),
        protected_files_verified=len(reg['protected']),independent_validation=ind.get('status'),
        independent_geometry_or_model_calls=0,raw_POST_generated_scores_inherited_only=True,
        commit=False,push=False,default_model_changed=False),ensure_ascii=False,indent=2)+'\n')
    forbidden={'.pt','.pth','.npz','.npy','.ply','.obj','.png','.jpg','.html','.pyc'}
    files=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name in {'MANIFEST.json','FILE_INDEX.csv'}:continue
        assert p.suffix not in forbidden,p
        rel=p.relative_to(OUT).as_posix()
        n,fields=csvinfo(p) if p.suffix=='.csv' else (None,None)
        origin=('NEW_SOURCE_INTERFACE_MEASUREMENT_WITH_INHERITED_PAIR_LEGALITY' if rel.startswith('conditions/') or p.name in {'source_interfaces.csv','AVMR_PMR_all8.csv','boundary_vertices.csv','C_Y_edge_details.csv'} else
                'FROZEN_ORIGINAL_SCORING_SOURCE' if rel.startswith('code/frozen/') else
                'INHERITED_USER_FEEDBACK' if p.name=='user_feedback.txt' else 'DERIVED_OR_EXECUTION_METADATA')
        files.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p),rows=n,fields=fields,kind=origin,
            source_reference='protocol/registration.json protected sources and per-condition identity; statistics and exact_count_audit retain their own source manifests'))
    manifest=dict(task='R2_SOURCE_INTERFACE_AUDIT_V1',created_utc=time.time(),files=files,
        measurements=dict(conditions=258,source_interface_rows=1032,threshold_rows=2064,boundary_vertex_rows=3055,C_Y_edge_rows=5970),
        no_original_results_overwritten=True,scientific_definition='Frozen selected functions and detector identity in protocol/registration.json',
        exclusions=['weights','noise/OT tensors','source/generated meshes','images','old full archives'],
        classification_reuse='Existing complete independently verified source pair scans; same vertices/permutations and detector hash checked',
        source_coordinate_domain='Source model-scale mesh, not D0/GEO processing of source completion',
        self_hash_policy='External ZIP receipt hashes the manifest and file index together with complete payload')
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (OUT/'FILE_INDEX.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,['path','bytes','rows','sha256','kind']);w.writeheader()
        w.writerows({k:r[k] for k in w.fieldnames} for r in files)
    archive=OUT.parents[1]/'deliverables/R2_SOURCE_INTERFACE_AUDIT_V1.zip'
    if archive.exists():raise FileExistsError(archive)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for r in files:z.write(OUT/r['path'],r['path'])
        for name in ['MANIFEST.json','FILE_INDEX.csv']:z.write(OUT/name,name)
    assert archive.stat().st_size<10*1024**2
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for r in files:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    receipt=dict(status='COMPLETE',archive=str(archive),bytes=archive.stat().st_size,
        MiB=archive.stat().st_size/1024**2,sha256=sha(archive),file_count=len(files)+2,
        CRC_pass=True,payload_hashes_pass=True,protected_files_unchanged=len(reg['protected']),
        independent_validation=ind.get('status'),resume_no_new_work_verified=True)
    dest=archive.with_name('R2_SOURCE_INTERFACE_AUDIT_V1_DELIVERY.json')
    dest.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
