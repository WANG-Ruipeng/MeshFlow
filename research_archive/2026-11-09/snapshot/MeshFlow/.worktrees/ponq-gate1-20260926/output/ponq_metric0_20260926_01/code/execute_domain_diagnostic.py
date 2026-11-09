"""Freeze all inputs and source hashes before the bounded corner-domain diagnostic."""
import os,sys,json,time,shutil,subprocess,datetime,argparse
from pathlib import Path
import numpy as np
from common import sha,dump,load_mesh,sample
from summarize_clay import method_failures
from safe_extract_domain import corner_domain


def read(path):return json.loads(Path(path).read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--fm-parent',type=Path,required=True);ap.add_argument('--factorial-parent',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve();fm=args.fm_parent.resolve();factorial=args.factorial_parent.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for name in ['runs','audit','logs','references']:(out/name).mkdir()
    module=Path(__file__).resolve().parents[1]
    for name in ['code','vendor']:shutil.copytree(module/name,out/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Core implementation/metrics must match both immutable parent studies.
    shared=['code/safe_extract.py','code/common.py','code/exact_audit.py','code/run_clay.py']
    shared += [str(p.relative_to(module)) for p in sorted((module/'vendor').rglob('*')) if p.is_file() and '__pycache__' not in str(p) and p.suffix!='.pyc']
    for parent in [fm,factorial]:
        for rel in shared:assert sha(module/rel)==sha(parent/rel),('changed_shared_source',str(parent),rel)
    rows=read(fm/'LEDGER.json');assert len(rows)==64 and {(r['arm'],r['sample_id']) for r in rows}=={(a,i) for a in ['direct','transport_residual'] for i in range(32)}
    data=read(fm/'DATA_CHECK.json');assert data['passed'] and len(data['records'])==24
    with np.load(fm/'data.npz') as z:names=[str(x) for x in z['object_names']]
    assert len(names)==24 and len(set(names))==24
    references={}
    def reference(key,mesh,seed=None,cached=None):
        dest=out/'references'/key;dest.mkdir();shutil.copy2(mesh,dest/'reference.obj')
        if cached is not None:shutil.copy2(cached,dest/'eval_samples.npz')
        else:
            v,f=load_mesh(mesh);points,normals,_=sample(v,f,8192,seed);np.savez_compressed(dest/'eval_samples.npz',points=points,normals=normals)
        references[key]=dict(source_mesh=str(mesh),source_mesh_sha256=sha(mesh),mesh_sha256=sha(dest/'reference.obj'),
            sample_sha256=sha(dest/'eval_samples.npz'),sampling_seed=seed,cached_samples=None if cached is None else str(cached))
    for j,name in enumerate(names):reference(name,fm/'targets'/name/'mesh.obj',seed=29000+j)
    qsource=factorial/'runs/thin_u_frame06__q_only';qold=next(r for r in read(factorial/'LEDGER.json') if r['object']=='thin_u_frame06' and r['state']=='q_only')
    assert not method_failures(qold),'Selected q_only target is not valid'
    reference('single_q_only',factorial/'inputs/thin_u_frame06/reference.obj',cached=factorial/'inputs/thin_u_frame06/reference_eval_samples.npz')
    slots=[]
    def slot(identifier,group,domain,source,target,reference_id,metric_seed,baseline,arm=None,sample_id=None):
        item=dict(id=identifier,group=group,domain=domain,parameters=str(source/'parameters.npz'),parameters_sha256=sha(source/'parameters.npz'),
            fixed_mesh_npz=str(source/'mesh.npz'),fixed_mesh_npz_sha256=sha(source/'mesh.npz'),
            fixed_mesh=str(source/'mesh.obj'),fixed_mesh_sha256=sha(source/'mesh.obj'),target=target,reference_id=reference_id,
            metric_seed=metric_seed,baseline_fixed=baseline,arm=arm,sample_id=sample_id)
        if (source/'extractor_trace.npz').exists():item.update(fixed_trace=str(source/'extractor_trace.npz'),fixed_trace_sha256=sha(source/'extractor_trace.npz'))
        slots.append(item)
    qbase=dict(valid=True,status=qold['status'],faces=qold['faces'],rms_pct_bboxdiag=qold['rms_pct_bboxdiag'])
    slot('identity__thin_u_frame06_q_only__fixed','fixed_identity','fixed',qsource,'thin_u_frame06_q_only','single_q_only',52206,qbase)
    slot('single__thin_u_frame06_q_only__enclosing','single_target','enclosing',qsource,'thin_u_frame06_q_only','single_q_only',52206,qbase)
    for report in data['records']:
        name=report['object'];source=fm/'targets'/name
        baseline=dict(valid=bool(report['valid']),status='EXTRACTED',faces=report['geometry']['faces'],rms_pct_bboxdiag=0.,rms_note='Canonical target compared with its own fixed surface')
        slot('oracle__'+name+'__enclosing','oracle24','enclosing',source,name,name,29000+names.index(name),baseline)
    for row in sorted(rows,key=lambda r:(r['arm'],r['sample_id'])):
        arm=row['arm'];i=row['sample_id'];name=row['nearest_target_by_site_cd'];source=fm/'samples'/f'{arm}_{i:03d}'
        baseline=dict(valid=bool(row['valid']),status=row['status'],faces=row.get('faces'),rms_pct_bboxdiag=row.get('geometry_vs_site_selected_target',{}).get('rms_pct_bboxdiag'),
            old_input_sites_outside_corner_cube=row.get('input_sites_outside_corner_cube'),legacy_threshold_abs_coordinate=.5)
        slot(f'generated__{arm}__{i:03d}__enclosing','generated','enclosing',source,name,name,30000+i,baseline,arm,i)
    assert len(slots)==90
    source_files={}
    for item in slots:
        for key in ['parameters','fixed_mesh','fixed_mesh_npz','fixed_trace']:
            if item.get(key):source_files[item[key]]=sha(item[key])
    for parent in [fm,factorial]:
        for file in ['MANIFEST.json','LEDGER.json']:
            path=parent/file;source_files[str(path)]=sha(path)
    snapshot=[dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size) for folder in ['code','vendor'] for p in sorted((out/folder).rglob('*')) if p.is_file()]
    dump(out/'SOURCE_SNAPSHOT.json',dict(files=snapshot))
    _,fixed=corner_domain(np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]]),'fixed')
    protocol=dict(version='PONQ_CORNER_DOMAIN_DIAGNOSTIC_V1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        fm_parent=str(fm),factorial_parent=str(factorial),requested_new_extractions=90,reused_fixed_generated_rows=64,
        enclosing_generated=64,enclosing_float64_oracles=24,enclosing_single_target=1,fixed_identity_replays=1,
        grid_scale=32,margin_rule='0.10*extent per nonzero axis; zero axis gets max_extent*1e-6; all-zero extent fails',
        fixed_corners=fixed['anchor_coordinates'],generated_fields_changed=[],anchor_attributes_changed=['positions_only'],
        parameters='Saved physical v,n,Q,eigs loaded byte-identically; p/metadata retained but not consumed; no clipping, scaling, normal normalization or Q refit',
        reference_policy='No reference contributes to domain selection. Generated RMS reuses old site-selected target and identical sampling seeds; oracle24 uses canonical fixed mesh; single q_only uses archived PL reference/cached samples.',
        geometry_contract='closed oriented simple vertex-link 2-manifold, Betti[1,0,1], no duplicate/degenerate/intersecting faces; preserve actual faces',
        interpretation='Extraction-domain intervention, not additional training or generated-site correction; domain may change Delaunay and labels.',
        references=references,slots=slots,source_files=[dict(path=p,sha256=h) for p,h in sorted(source_files.items())],
        source_snapshot_sha256=sha(out/'SOURCE_SNAPSHOT.json'))
    dump(out/'FROZEN_PROTOCOL.json',protocol);dump(out/'FROZEN_PROTOCOL_SHA256.json',dict(sha256=sha(out/'FROZEN_PROTOCOL.json')))
    dump(out/'audit/PARENT_INPUT_CHECKS.json',dict(passed=True,shared_source=shared,verified_sources=len(source_files)))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-B',str(out/'code/run_domain_diagnostic.py')];start=time.perf_counter()
    with (out/'logs/diagnostic.log').open('w') as log:r=subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
    dump(out/'EXECUTION.json',dict(status='COMPLETE' if r.returncode==0 else 'FAILED',returncode=r.returncode,command=command,seconds=time.perf_counter()-start))
    dump(out/'MANIFEST.json',dict(files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']))
    print('DOMAIN_EXECUTION',r.returncode,out,flush=True)
    raise SystemExit(r.returncode)
if __name__=='__main__':main()