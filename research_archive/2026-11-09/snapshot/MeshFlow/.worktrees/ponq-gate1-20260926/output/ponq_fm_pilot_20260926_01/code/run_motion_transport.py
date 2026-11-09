"""Position-only local motion Jacobians for full-Q transport: 48 frozen slots."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,json,time,datetime,traceback,copy,shutil
from pathlib import Path
import numpy as np
from common import ROOT,dump,sha,save_mesh,load_mesh,mesh_audit,metrics
from run_clay import array_hash,sorted_simplex,strict_links,centered,world
from safe_extract import extract
from motion_jacobian import prepare_neighborhood,estimate_motion
ARMS=['lsq16','ridge16'];OID={'sphere':0,'thin_u':2,'bottle_local_1':6}
def read(p):return json.loads(p.read_text())
def loadparams(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
    parent=Path(sys.argv[1]).resolve();factorial=Path(sys.argv[2]).resolve();old=read(parent/'FROZEN_PROTOCOL.json')
    validation=parent.parent/'ponq_motion_jacobian_validation_20260926_01'/'VALIDATION.json'
    vd=read(validation);assert vd['passed'] and vd['tests_run']==7
    for item in vd['source_files']:
        assert sha(ROOT/'code'/Path(item['path']).name)==item['sha256'],('tested_estimator_source_changed',item['path'])
    shutil.copy2(validation,ROOT/'audit/ESTIMATOR_UNIT_TESTS.json')
    verified=[]
    for directory in [parent,factorial]:
        manifest=read(directory/'MANIFEST.json')
        for e in manifest['files']:assert sha(directory/e['path'])==e['sha256'],str(directory/e['path'])
        verified.append(dict(path=str(directory),files=len(manifest['files']),manifest_sha256=sha(directory/'MANIFEST.json')))
    shared=['code/common.py','code/safe_extract.py','code/run_clay.py','code/exact_audit.py']+[str(p.relative_to(parent)) for p in sorted((parent/'vendor').rglob('*')) if p.is_file()]
    for rel in shared:assert sha(ROOT/rel)==sha(parent/rel),('shared_backend_changed',rel)
    files=[p/d for p in [parent,factorial] for d in ['MANIFEST.json','LEDGER.json','FROZEN_PROTOCOL.json']]
    files.append(validation)
    for source in OID:
        files.extend(parent/'path_inputs'/source/f for f in ['base.npz','path.json'])
    for obj in old['objects']:
        for directory,arm in [(parent,'freeze'),(factorial,'q_only')]:
            files.extend(directory/'runs'/f'{obj}__{arm}'/f for f in ['parameters.npz','mesh.obj','extractor_trace.npz','extractor_audit.json'])
        files.extend(p for p in sorted((parent/'inputs'/obj).iterdir()) if p.is_file())
    config=copy.deepcopy(old['config']);config.update(version='PONQ_POSITION_ONLY_JACOBIAN_V1',parent_run=str(parent),factorial_run=str(factorial),run_states=ARMS,
        new_extractions=48,reused_control_rows=48,combined_rows=96,reused_controls=['freeze','q_only'],
        source_files=[dict(path=str(p),sha256=sha(p)) for p in files],
        motion_estimator=dict(k=16,neighbors='source-space Euclidean, excluding self, frozen across time; kNN ties not permutation-certified; zero kth-neighbor scale is explicit input rejection',weights='exp(-(distance/h)^2), normalized; h kth distance',
            design='X=(x0_neighbor-x0_center)/h; D=(xt_neighbor-xt_center-x0_neighbor+x0_center)/h; J=I+B.T, D=X B',
            lsq16='weighted SVD, relative singular threshold 1e-6; rank<3 uses identity fallback',
            ridge16='lambda=1e-3*trace(X.T W X)/3; identity prior; rank deficiency recorded, not rejected solely for deficiency',
            guards='nonfinite J, determinant<=1e-8, condition>1e4 use identity fallback; no oracle replacement or clipping',
            information='only initial and current matched input element coordinates; no target surface, current normals, analytical J, or mesh-quality feedback'),
        transport='initial full local C=A/g/c, apply diag(J_used^-1,1) congruence and recenter at current v; recompute eigs; n/p/v exactly parent freeze',
        initial_frame='estimate and record J, then reuse canonical initial Q/eigs to avoid floating-point tie changes',
        known_limitations=['3 existing objects with correlated frames','paired persistent point identities and prescribed motion, not learned/noise-to-shape generation','regularized rank-deficient directions use a prior, not recovered information','fixed 16-neighbor model, no hyperparameter search','independent mapping from initial state, not cumulative motion integration','same archived PL target, bottle reference sensitivity unresolved','no training or matched-cost speed claim'])
    dump(ROOT/'FROZEN_PROTOCOL.json',dict(config=config,objects=old['objects'],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    dump(ROOT/'FROZEN_PROTOCOL_SHA256.json',dict(sha256=sha(ROOT/'FROZEN_PROTOCOL.json')))
    dump(ROOT/'audit/PARENT_INTEGRITY.json',dict(passed=True,parents=verified,shared_code_hash_checks=shared))
    cache={}
    for source in OID:
        base=loadparams(parent/'path_inputs'/source/'base.npz');neighborhood=prepare_neighborhood(base['v'],k=16)
        dest=ROOT/'motion_inputs'/source;dest.mkdir(parents=True)
        shutil.copy2(parent/'path_inputs'/source/'base.npz',dest/'base.npz')
        np.savez_compressed(dest/'neighborhood.npz',**neighborhood['arrays']);dump(dest/'stats.json',neighborhood['stats'])
        cache[source]=(base,neighborhood,centered(base['Q'],base['v']))
    rows=[];previous={};refreuse=[]
    for obj,spec in old['objects'].items():
        source=spec['source_object'];fi=spec['frame'];base,neighborhood,C0=cache[source]
        freeze=loadparams(parent/'runs'/f'{obj}__freeze'/'parameters.npz')
        dest=ROOT/'inputs'/obj;shutil.copytree(parent/'inputs'/obj,dest)
        assert read(dest/'reference_exact_audit.json')['source_sha256']==sha(dest/'reference.obj')
        refreuse.append(dict(object=obj,reference_sha256=sha(dest/'reference.obj'),reused_from=str(parent/'inputs'/obj)))
        # Target assets are only consumed by the evaluation below, never by estimate_motion.
        gv,gf=load_mesh(dest/'reference.obj')
        with np.load(dest/'reference_eval_samples.npz') as z:gt=z['points'].copy();gn=z['normals'].copy()
        for arm in ARMS:
            print('EXTRACT_START',obj,arm,flush=True)
            rd=ROOT/'runs'/f'{obj}__{arm}';rd.mkdir();start=time.perf_counter()
            row=dict(object=obj,source_object=source,path_name=spec['path_name'],frame=fi,t=spec['t'],state=arm,budget=256,
                extraction_attempts=1,reference_betti=[1,0,1],reference_assisted=False,new_extraction=True,input_vertex_sha256=array_hash(freeze['v']))
            try:
                est=estimate_motion(base['v'],freeze['v'],neighborhood,mode=arm)
                np.savez_compressed(rd/'estimated_motion.npz',**dict(est['arrays'],J_raw=est['J_raw'],J_used=est['J_used']))
                dump(rd/'estimated_motion_stats.json',est['stats']);row['motion_estimate_stats']=est['stats']
                inv=np.linalg.inv(est['J_used']);H=np.broadcast_to(np.eye(4),(len(inv),4,4)).copy();H[:,:3,:3]=inv
                C=H.transpose(0,2,1)@C0@H;Q=world(C,freeze['v'])
                p={k:v.copy() for k,v in freeze.items()};p['Q']=Q;p['eigs']=np.linalg.svd(Q[:,:3,:3],compute_uv=False)
                if fi==0:p['Q']=freeze['Q'].copy();p['eigs']=freeze['eigs'].copy()
                row['parameter_update_seconds']=time.perf_counter()-start
                np.savez_compressed(rd/'parameters.npz',**p)
                et=time.perf_counter();v,f,ids,trace,audit=extract(p);row['extraction_seconds']=time.perf_counter()-et
                np.savez_compressed(rd/'extractor_trace.npz',**trace);dump(rd/'extractor_audit.json',audit);row.update(audit)
                save_mesh(rd/'mesh.obj',v,f);lv,lf=load_mesh(rd/'mesh.obj');assert np.array_equal(lv,v) and np.array_equal(lf,f)
                np.savez_compressed(rd/'mesh.npz',vertices=v,faces=f,input_vertex_ids=ids)
                sf=sorted_simplex(ids[f]);fs=set(map(tuple,sf));row['surface_input_faces_sha256']=array_hash(sf);row['output_input_vertex_sha256']=array_hash(np.sort(ids))
                key=(source,arm)
                if key in previous:
                    pf,ps=previous[key];row.update(previous_successful_frame=pf,previous_comparison_is_adjacent=pf==fi-1,surface_faces_added_from_previous=len(fs-ps),surface_faces_removed_from_previous=len(ps-fs))
                previous[key]=(fi,fs)
                row.update(mesh_audit(v,f));row.update(strict_links(f))
                mt=time.perf_counter();row.update(metrics(v,f,gv,gf,gt,gn,52000+100*OID[source]+fi,n=8192));row['metric_seconds']=time.perf_counter()-mt
                diag=float(np.linalg.norm(np.ptp(gv,axis=0)));row.update(reference_bbox_diagonal=diag,rms_absolute=row['rms_pct_bboxdiag']*diag/100,p95_absolute=row['p95_pct_bboxdiag']*diag/100)
                row['betti_match']=[row['beta'+str(i)] for i in range(3)]==[1,0,1];row['mesh_sha256']=sha(rd/'mesh.obj');row['status']='EXTRACTED'
                if fi==0:assert sha(rd/'mesh.obj')==sha(parent/'runs'/f'{obj}__freeze'/'mesh.obj'),('t0_replay_changed',obj,arm)
            except Exception as e:
                row.update(status='EXTRACTION_OR_EVALUATION_FAILED',error=repr(e),traceback=traceback.format_exc())
                if hasattr(e,'audit'):dump(rd/'extractor_audit.json',e.audit);row['failure_extractor_audit']=e.audit
                if hasattr(e,'trace_arrays'):np.savez_compressed(rd/'extractor_trace.npz',**e.trace_arrays)
            row['arm_wall_seconds']=time.perf_counter()-start;dump(rd/'result.json',row);rows.append(row);dump(ROOT/'LEDGER_PARTIAL.json',rows)
            print('EXTRACT_END',obj,arm,row['status'],row.get('rms_pct_bboxdiag'),flush=True)
    assert len(rows)==48;dump(ROOT/'LEDGER_RAW.json',rows)
    dump(ROOT/'audit/REFERENCE_AUDIT_REUSE.json',dict(passed=True,reference_count=24,records=refreuse));print('ALL_48_SLOTS_RECORDED',flush=True)
if __name__=='__main__':main()