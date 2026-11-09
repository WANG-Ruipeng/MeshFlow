"""CLAY_Q_ACCUM0: 24 forward and 21 re-estimated return extractions."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys, time, datetime, traceback, copy, shutil, json
from pathlib import Path
import numpy as np
from common import ROOT,dump,sha,save_mesh,load_mesh,mesh_audit,metrics
from run_clay import array_hash,sorted_simplex,strict_links
from safe_extract import extract
from incremental_transport import step_transport

SOURCES={'sphere':0,'thin_u':2,'bottle_local_1':6}
ARM='incremental_lsq16'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def main():
    parent,factorial,motion,validation=map(lambda s:Path(s).resolve(),sys.argv[1:5])
    old=read(parent/'FROZEN_PROTOCOL.json'); verified=[]
    for directory in [parent,factorial,motion]:
        manifest=read(directory/'MANIFEST.json')
        for e in manifest['files']:
            assert sha(directory/e['path'])==e['sha256'],str(directory/e['path'])
        verified.append(dict(path=str(directory),files=len(manifest['files']),manifest_sha256=sha(directory/'MANIFEST.json')))
    shared=['code/common.py','code/safe_extract.py','code/run_clay.py','code/exact_audit.py','code/motion_jacobian.py']
    shared += [str(p.relative_to(motion)) for p in sorted((motion/'vendor').rglob('*')) if p.is_file()]
    for rel in shared:assert sha(ROOT/rel)==sha(motion/rel),('shared_backend_changed',rel)
    vd=read(validation); assert vd['passed'],vd
    for item in vd['source_files']:
        assert sha(ROOT/'code'/Path(item['path']).name)==item['sha256'],('tested_source_changed',item['path'])
    shutil.copy2(validation,ROOT/'audit/TRANSPORT_UNIT_TESTS.json')
    objects={}
    for source in SOURCES:
        for direction,frames in [('forward',range(8)),('reverse',range(6,-1,-1))]:
            for fi in frames:
                ref=f'{source}_frame{fi:02d}'
                objects[f'{source}_{direction}_frame{fi:02d}']=dict(old['objects'][ref],direction=direction,source_frame_object=ref)
    control_rows=read(motion/'COMBINED_LEDGER.json')
    end={r['state']:r for r in control_rows if r['source_object']=='thin_u' and r['frame']==7}
    rf=end['freeze']['rms_pct_bboxdiag'];rd=end['lsq16']['rms_pct_bboxdiag']
    config=copy.deepcopy(old['config'])
    config.update(version='CLAY_Q_ACCUM0',parent_run=str(parent),factorial_run=str(factorial),motion_run=str(motion),
        run_states=[ARM],new_extractions=45,forward_extractions=24,reverse_extractions=21,reused_control_rows=72,
        source_files=[dict(path=str(p),sha256=sha(p)) for p in [validation]+[d/f for d in [parent,factorial,motion] for f in ['MANIFEST.json','LEDGER.json','FROZEN_PROTOCOL.json']]],
        estimator=dict(k=16,rank_rtol=1e-6,determinant_floor=1e-8,condition_limit=1e4,
            frozen='initial neighbor IDs, weights, h; no per-frame bbox renormalization',
            current='design X and weighted SVD/rank/G recomputed from each step source coordinates',
            fallback='J=I: fixed centered local full quadric carried by pure translation, not frozen world Q'),
        transport='world H=[[J,x_next-J*x_current],[0,1]]; Q_next=H^-T Q_previous H^-1; unnormalized raw state recurs',
        reverse='actual forward terminal Q; re-estimate from reverse paired positions, never reset or reuse inverse forward J',
        extraction='n/p/v from same-frame frozen inputs; Q/eigs on fresh copy; no Q refit, relocation or mesh feedback',
        gates=dict(algebra_relative_tolerance=1e-10,thin_u_gain_retention_minimum=.90,
            thin_u_freeze_endpoint_rms=rf,thin_u_direct_endpoint_rms=rd,thin_u_incremental_endpoint_rms_max=rf-.90*(rf-rd)),
        known_limitations=['3 known objects with correlated states; no independent object validation',
            'prescribed paired motion, real initial quadrics and frozen normals; not noise-to-shape FM',
            'one finite forward/return cycle; no infinite stability or timestep convergence claim',
            'bottle reference PL sensitivity unresolved; not a promotion criterion',
            'no network training, parameter sweep, or extraction feedback'])
    dump(ROOT/'FROZEN_PROTOCOL.json',dict(config=config,objects=objects,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    dump(ROOT/'FROZEN_PROTOCOL_SHA256.json',dict(sha256=sha(ROOT/'FROZEN_PROTOCOL.json')))
    dump(ROOT/'audit/PARENT_INTEGRITY.json',dict(passed=True,parents=verified,shared_code_hash_checks=shared))
    for name in old['objects']:shutil.copytree(parent/'inputs'/name,ROOT/'inputs'/name)
    rows=[]; controls=[]
    for source in SOURCES:
        dest=ROOT/'motion_inputs'/source
        shutil.copytree(motion/'motion_inputs'/source,dest)
        hood=dict(arrays=arrays(dest/'neighborhood.npz'),stats=read(dest/'stats.json'))
        base=arrays(dest/'base.npz')
        initial=arrays(parent/'runs'/f'{source}_frame00__freeze'/'parameters.npz')
        assert np.array_equal(base['v'],initial['v']) and np.array_equal(base['Q'],initial['Q'])
        Q0=initial['Q'].copy();x0=initial['v'].copy();Q=Q0.copy();x=x0.copy()
        cumulative=np.broadcast_to(np.eye(4),Q.shape).copy();forward_H=[];previous_key='';chain_error=None
        tasks=[(d,f) for d,frames in [('forward',range(8)),('reverse',range(6,-1,-1))] for f in frames]
        for step,(direction,fi) in enumerate(tasks):
            obj=f'{source}_{direction}_frame{fi:02d}'; spec=objects[obj];ref=spec['source_frame_object']
            rd=ROOT/'runs'/f'{obj}__{ARM}';rd.mkdir()
            start=time.perf_counter()
            row=dict(object=obj,state=ARM,**spec,budget=256,reference_betti=[1,0,1],reference_assisted=False,
                new_extraction=True,step_index=step,previous_run_key=previous_key,extraction_attempts=0)
            print('EXTRACT_START',obj,flush=True)
            freeze=arrays(parent/'runs'/f'{ref}__freeze'/'parameters.npz')
            row['input_vertex_sha256']=array_hash(freeze['v'])
            try:
                if chain_error:raise RuntimeError('Previous state failed; no reset allowed: '+chain_error)
                Qprev=Q.copy();xprev=x.copy();Cprev=cumulative.copy()
                Qnext,H,est,current_nb=step_transport(Qprev,xprev,freeze['v'],hood)
                if step==0:
                    # No update at t0: replay the canonical state exactly.
                    Qnext=Q0.copy()
                cumulative=H@Cprev;Q=Qnext.copy();x=freeze['v'].copy()
                if direction=='forward' and fi>0:forward_H.append(H.copy())
                np.savez_compressed(rd/'raw_state.npz',Q=Q,H=H,cumulative_H=cumulative,Q_previous=Qprev,
                    x_previous=xprev,x_current=x,initial_Q=Q0,initial_x=x0,cumulative_H_previous=Cprev,
                    previous_run_key=np.array(previous_key),step_index=np.array(step))
                np.savez_compressed(rd/'current_neighborhood.npz',**current_nb['arrays'])
                np.savez_compressed(rd/'estimated_motion.npz',**est['arrays'])
                dump(rd/'estimated_motion_stats.json',est['stats']);row['motion_estimate_stats']=est['stats']
                invC=np.linalg.inv(cumulative);one=invC.transpose(0,2,1)@Q0@invC
                row['recursive_vs_composed_relative']=float(np.linalg.norm(Q-one)/np.linalg.norm(one))
                p={k:v.copy() for k,v in freeze.items()};p['Q']=Q.copy();p['eigs']=np.linalg.svd(Q[:,:3,:3],compute_uv=False)
                if step==0:p['eigs']=initial['eigs'].copy()
                np.savez_compressed(rd/'parameters.npz',**p)
                row['parameter_update_seconds']=time.perf_counter()-start
            except Exception as e:
                chain_error=repr(e);row.update(status='STATE_UPDATE_FAILED',error=repr(e),traceback=traceback.format_exc())
            if not chain_error:
                try:
                    row['extraction_attempts']=1;et=time.perf_counter()
                    v,f,ids,trace,audit=extract(p)
                    assert np.array_equal(p['Q'],Q),'Extractor mutated raw state'
                    row['extraction_seconds']=time.perf_counter()-et
                    np.savez_compressed(rd/'extractor_trace.npz',**trace);dump(rd/'extractor_audit.json',audit);row.update(audit)
                    save_mesh(rd/'mesh.obj',v,f);lv,lf=load_mesh(rd/'mesh.obj')
                    assert np.array_equal(lv,v) and np.array_equal(lf,f)
                    np.savez_compressed(rd/'mesh.npz',vertices=v,faces=f,input_vertex_ids=ids)
                    row['surface_input_faces_sha256']=array_hash(sorted_simplex(ids[f]))
                    row['output_input_vertex_sha256']=array_hash(np.sort(ids))
                    row.update(mesh_audit(v,f));row.update(strict_links(f))
                    refdir=ROOT/'inputs'/ref;gv,gf=load_mesh(refdir/'reference.obj')
                    z=arrays(refdir/'reference_eval_samples.npz');mt=time.perf_counter()
                    row.update(metrics(v,f,gv,gf,z['points'],z['normals'],52000+100*SOURCES[source]+fi,n=8192))
                    row['metric_seconds']=time.perf_counter()-mt
                    diag=float(np.linalg.norm(np.ptp(gv,axis=0)))
                    row.update(reference_bbox_diagonal=diag,rms_absolute=row['rms_pct_bboxdiag']*diag/100,p95_absolute=row['p95_pct_bboxdiag']*diag/100)
                    row['betti_match']=[row['beta'+str(i)] for i in range(3)]==[1,0,1]
                    row['mesh_sha256']=sha(rd/'mesh.obj');row['status']='EXTRACTED'
                    if step==0:assert row['mesh_sha256']==sha(parent/'runs'/f'{ref}__freeze'/'mesh.obj')
                except Exception as e:
                    row.update(status='EXTRACTION_OR_EVALUATION_FAILED',error=repr(e),traceback=traceback.format_exc())
                    if hasattr(e,'audit'):dump(rd/'extractor_audit.json',e.audit);row['failure_extractor_audit']=e.audit
                    if hasattr(e,'trace_arrays'):np.savez_compressed(rd/'extractor_trace.npz',**e.trace_arrays)
            if direction=='forward' and fi==7 and not chain_error:
                undo=Q.copy()
                for saved_H in reversed(forward_H):undo=saved_H.transpose(0,2,1)@undo@saved_H
                controls.append(dict(source_object=source,forward_inverse_undo_relative=float(np.linalg.norm(undo-Q0)/np.linalg.norm(Q0))))
                np.savez_compressed(dest/'inverse_control.npz',Q_undone=undo,Q_initial=Q0,forward_H=np.stack(forward_H))
            row['arm_wall_seconds']=time.perf_counter()-start
            dump(rd/'result.json',row);rows.append(row);dump(ROOT/'LEDGER_PARTIAL.json',rows)
            previous_key=f'{obj}__{ARM}'
            print('EXTRACT_END',obj,row['status'],row.get('rms_pct_bboxdiag'),flush=True)
    assert len(rows)==45
    dump(ROOT/'LEDGER_RAW.json',rows);dump(ROOT/'audit/ALGEBRA_CONTROL.json',dict(controls=controls))
    print('ALL_45_SLOTS_RECORDED',flush=True)
if __name__=='__main__':main()
