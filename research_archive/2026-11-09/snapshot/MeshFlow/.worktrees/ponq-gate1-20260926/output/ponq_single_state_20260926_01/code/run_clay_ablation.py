"""48 new extractions completing the frozen Q x normal transport factorial."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,json,time,datetime,traceback,shutil,copy
from pathlib import Path
import numpy as np
from common import ROOT,dump,sha,save_mesh,load_mesh,mesh_audit,metrics
from run_clay import array_hash,sorted_simplex,strict_links
from safe_extract import extract
ARMS=['q_only','n_only']
OID={'sphere':0,'thin_u':2,'bottle_local_1':6}
def read(p):return json.loads(p.read_text())
def params(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
    parent=Path(sys.argv[1]).resolve()
    oldmanifest=read(parent/'MANIFEST.json')
    for e in oldmanifest['files']:assert sha(parent/e['path'])==e['sha256'],e['path']
    oldprotocol=read(parent/'FROZEN_PROTOCOL.json');oldrows=read(parent/'LEDGER.json')
    by={(r['object'],r['state']):r for r in oldrows}
    rels=['code/common.py','code/safe_extract.py','code/run_clay.py','code/exact_audit.py']
    rels += [str(p.relative_to(parent)) for p in sorted((parent/'vendor').rglob('*')) if p.is_file()]
    for rel in rels:assert sha(ROOT/rel)==sha(parent/rel),('backend_or_metric_changed',rel)
    source_paths=[parent/'MANIFEST.json',parent/'LEDGER.json',parent/'FROZEN_PROTOCOL.json']
    for obj in oldprotocol['objects']:
        for arm in ['freeze','transport','refit']:
            for file in ['parameters.npz','mesh.obj','extractor_trace.npz','extractor_audit.json']:
                source_paths.append(parent/'runs'/f'{obj}__{arm}'/file)
        for p in sorted((parent/'inputs'/obj).iterdir()):
            if p.is_file():source_paths.append(p)
    config=copy.deepcopy(oldprotocol['config']);config.update(
        version='PONQ_CLAY_Q_NORMAL_FACTORIAL_V1',run_states=ARMS,parent_run=str(parent),
        source_files=[dict(path=str(p),sha256=sha(p)) for p in source_paths],
        new_extractions=48,reused_parent_rows=72,combined_rows=120,
        q_only='Q and eigs copied exactly from parent transport; n and other fields from parent freeze',
        n_only='n copied exactly from parent transport; Q, eigs and other fields from parent freeze',
        reused_controls=['freeze','transport','refit'],primary_diagnostic='thin_u frame06 nonmanifold vertex plus all 24 frame pairs',
        known_limitations=['3 existing development objects and correlated frames','Q denotes full A/g/c with its normalization eigenvalues','not an A-only or capacity-only intervention','normal update can affect seeds and cut scores','same archived PL reference; bottle refinement sensitivity unresolved','old controls are manifest-bound reused results, not new runs','no training, no process feedback, no matched-cost speed comparison'])
    dump(ROOT/'FROZEN_PROTOCOL.json',dict(config=config,objects=oldprotocol['objects'],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    dump(ROOT/'FROZEN_PROTOCOL_SHA256.json',dict(sha256=sha(ROOT/'FROZEN_PROTOCOL.json')))
    dump(ROOT/'audit/PARENT_INTEGRITY.json',dict(passed=True,parent=str(parent),manifest_sha256=sha(parent/'MANIFEST.json'),verified_files=len(oldmanifest['files']),shared_code_hash_checks=rels))
    rows=[];refreuse=[];previous={}
    for obj,spec in oldprotocol['objects'].items():
        source=spec['source_object'];fi=spec['frame'];dest=ROOT/'inputs'/obj
        shutil.copytree(parent/'inputs'/obj,dest)
        ex=read(dest/'reference_exact_audit.json');assert ex['source_sha256']==sha(dest/'reference.obj')
        refreuse.append(dict(object=obj,reference_sha256=sha(dest/'reference.obj'),audit_sha256=sha(dest/'reference_exact_audit.json'),reused_from=str(parent/'inputs'/obj)))
        gv,gf=load_mesh(dest/'reference.obj')
        with np.load(dest/'reference_eval_samples.npz') as z:gt=z['points'].copy();gn=z['normals'].copy()
        freeze=params(parent/'runs'/f'{obj}__freeze'/'parameters.npz');transport=params(parent/'runs'/f'{obj}__transport'/'parameters.npz')
        assert np.array_equal(freeze['v'],transport['v']) and np.array_equal(freeze['p'],transport['p'])
        for arm in ARMS:
            print('EXTRACT_START',obj,arm,flush=True)
            rd=ROOT/'runs'/f'{obj}__{arm}';rd.mkdir();start=time.perf_counter()
            row=dict(object=obj,source_object=source,path_name=spec['path_name'],frame=fi,t=spec['t'],state=arm,budget=256,
                extraction_attempts=1,reference_betti=[1,0,1],reference_assisted=False,new_extraction=True,
                input_vertex_sha256=array_hash(freeze['v']),parent_run=str(parent),
                parameter_donors=dict(Q='transport' if arm=='q_only' else 'freeze',eigs='transport' if arm=='q_only' else 'freeze',n='transport' if arm=='n_only' else 'freeze',v='freeze',p='freeze'))
            try:
                p={k:v.copy() for k,v in freeze.items()}
                for k in (['Q','eigs'] if arm=='q_only' else ['n']):p[k]=transport[k].copy()
                row['parameter_update_seconds']=time.perf_counter()-start
                np.savez_compressed(rd/'parameters.npz',**p)
                et=time.perf_counter();v,f,ids,trace,audit=extract(p);row['extraction_seconds']=time.perf_counter()-et
                np.savez_compressed(rd/'extractor_trace.npz',**trace);dump(rd/'extractor_audit.json',audit);row.update(audit)
                save_mesh(rd/'mesh.obj',v,f);lv,lf=load_mesh(rd/'mesh.obj');assert np.array_equal(lv,v) and np.array_equal(lf,f)
                np.savez_compressed(rd/'mesh.npz',vertices=v,faces=f,input_vertex_ids=ids)
                sf=sorted_simplex(ids[f]);fs=set(map(tuple,sf));row['surface_input_faces_sha256']=array_hash(sf)
                row['output_input_vertex_sha256']=array_hash(np.sort(ids))
                key=(source,arm)
                if key in previous:
                    pf,ps=previous[key];row.update(previous_successful_frame=pf,previous_comparison_is_adjacent=pf==fi-1,surface_faces_added_from_previous=len(fs-ps),surface_faces_removed_from_previous=len(ps-fs))
                previous[key]=(fi,fs)
                row.update(mesh_audit(v,f));row.update(strict_links(f))
                mt=time.perf_counter();row.update(metrics(v,f,gv,gf,gt,gn,52000+100*OID[source]+fi,n=8192));row['metric_seconds']=time.perf_counter()-mt
                diag=float(np.linalg.norm(np.ptp(gv,axis=0)));row.update(reference_bbox_diagonal=diag,rms_absolute=row['rms_pct_bboxdiag']*diag/100,p95_absolute=row['p95_pct_bboxdiag']*diag/100)
                row['betti_match']=[row['beta'+str(i)] for i in range(3)]==[1,0,1]
                row['mesh_sha256']=sha(rd/'mesh.obj');row['status']='EXTRACTED'
                if fi==0:assert row['mesh_sha256']==by[obj,'freeze']['mesh_sha256'],('t0_replay_changed',obj,arm)
            except Exception as e:
                row.update(status='EXTRACTION_OR_EVALUATION_FAILED',error=repr(e),traceback=traceback.format_exc())
                if hasattr(e,'audit'):dump(rd/'extractor_audit.json',e.audit);row['failure_extractor_audit']=e.audit
                if hasattr(e,'trace_arrays'):np.savez_compressed(rd/'extractor_trace.npz',**e.trace_arrays)
            row['arm_wall_seconds']=time.perf_counter()-start
            dump(rd/'result.json',row);rows.append(row);dump(ROOT/'LEDGER_PARTIAL.json',rows)
            print('EXTRACT_END',obj,arm,row['status'],row.get('rms_pct_bboxdiag'),flush=True)
    assert len(rows)==48;dump(ROOT/'LEDGER_RAW.json',rows)
    dump(ROOT/'audit/REFERENCE_AUDIT_REUSE.json',dict(passed=True,reference_count=24,records=refreuse))
    print('ALL_48_NEW_SLOTS_RECORDED',flush=True)
if __name__=='__main__':main()