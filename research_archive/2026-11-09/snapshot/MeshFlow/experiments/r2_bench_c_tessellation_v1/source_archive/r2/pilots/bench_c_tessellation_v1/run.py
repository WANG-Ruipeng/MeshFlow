"""Recoverable, bounded frozen-model experiment. No training or D0 calls."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='1'
import argparse, copy, json, sys, time, traceback
from pathlib import Path
from .common import *
sys.path.insert(0,str(ROOT/'r2/src'))

def prepare():
    if (OUT/'registration.json').exists():
        r=read(OUT/'registration.json'); verify(r['protected']); verify(r['inputs']); return r
    from .geometry import midpoint4,audit_subdivision
    from .evaluate import prepare_query
    from meshflow_control.source_winding import capture_source,validate_source
    from r2.pilots.bench_four_routes_v1.resources import snapshot
    write(OUT/'resources/preparation.json',snapshot())
    if sha(WEIGHT)!=WEIGHT_SHA: raise ValueError('Wrong EXPOSURE weight')
    manifest=read(OLD/'data/eval_manifest.json')
    qs=[q for q in manifest['requests'] if any(q['parent_id'].startswith(p) for p in PREFIXES)]
    qs.sort(key=lambda q:(PREFIXES.index(next(p for p in PREFIXES if q['parent_id'].startswith(p))),q['noise_index']))
    assert len(qs)==6 and all(q['noise_index'] in (0,1) for q in qs)
    protected={str(WEIGHT):WEIGHT_SHA,str(OLD/'data/eval_manifest.json'):sha(OLD/'data/eval_manifest.json')}
    rows=[];cells=[];jobs=[];refined_cache={}
    for q0 in qs:
        q=copy.deepcopy(q0); rid=q['request_id']; cid=q['condition_id']
        for key in ('inference_npz','metadata_npz','source_npz','noise_npz'):
            protected[str(resolve(q[key]))]=sha(q[key])
        protected[str(resolve(q['query_descriptor']['path']))]=sha(q['query_descriptor']['path'])
        baseline=OLD/'evaluation/EXPOSURE'/rid/'raw/raw.npz'
        receipt=read(baseline.parent/'receipt.json')
        assert receipt['raw_sha256']==sha(baseline) and receipt['noise_sha256']==sha(q['noise_npz'])
        protected[str(baseline)]=sha(baseline)
        base_score=OLD/'evaluation/scores/EXPOSURE'/rid/'RAW'
        for name in ('row.json','score.json','complete.json'): protected[str(base_score/name)]=sha(base_score/name)
        inp=arrays(q['inference_npz']); k,n=int(q['K']),int(q['N']); f=n-k
        assert arrays(q['noise_npz'])['z'].shape==(f,3,3)
        if cid not in refined_cache:
            ref=midpoint4(inp['C_vertices'],inp['C_faces_original'])
            audit=audit_subdivision(inp,ref); assert audit['status']=='PASS'
            d=OUT/'inputs'/cid; newn=f+len(ref['C_faces_original'])
            new_input={key:ref[key] for key in ('C_vertices','C_faces_original','C_faces_model')}
            new_input.update(N=np.asarray(newn),K=np.asarray(4*k))
            npz(d/'condition.npz',new_input);npz(d/'subdivision.npz',ref)
            oldmeta=arrays(q['metadata_npz']); ids=np.asarray(oldmeta['C_source_vertex_ids'])
            assert ids.shape==(len(inp['C_vertices']),)
            newids=np.r_[ids,np.arange(int(ids.max())+1,int(ids.max())+1+len(ref['C_vertices'])-len(ids),dtype=np.int64)]
            npz(d/'metadata.npz',dict(C_source_vertex_ids=newids,
                vertex_id_domain=np.asarray('original_C_IDs_plus_unique_derived_midpoints_not_full_GT'),
                child_parent_face_ids=ref['child_parent_face_ids']))
            src=capture_source(ref['C_vertices'][ref['C_faces_original']],
                source_vertex_ids=newids[ref['C_faces_original']],corner_permutation=ref['corner_permutations'])
            validate_source(src,ref['C_vertices'][ref['C_faces_model']]);npz(d/'source_condition.npz',src)
            query=prepare_query(q,ref['C_vertices'],ref['C_faces_original'],d/'queries',new_C_faces_model=ref['C_faces_model'])
            write(d/'geometry_audit.json',audit)
            refined_cache[cid]=(new_input,query,d)
        new_input,query,d=refined_cache[cid]; nk,nn=int(new_input['K']),int(new_input['N'])
        qref=copy.deepcopy(q);qref.update(N=nn,K=nk,inference_npz=str(d/'condition.npz'),
            metadata_npz=str(d/'metadata.npz'),source_npz=str(d/'source_condition.npz'),
            boundary_edges=int(q['boundary_edges'])*2,query_descriptor=query['descriptor'])
        for key,hkey in (('inference_npz','inference_npz_sha256'),('metadata_npz','metadata_sha256'),('source_npz','source_sha256')):
            qref[hkey]=sha(qref[key])
        rows.append(dict(parent_id=q['parent_id'],noise_index=q['noise_index'],group=q['report_group'],
            original_K=k,refined_K=nk,free_faces=f,original_N=n,refined_N=nn,
            original_count_bin=n//20,refined_count_bin=nn//20,same_free_noise_sha=sha(q['noise_npz'])))
        cells.append(dict(cell_id='ORIGINAL__'+rid,label='ORIGINAL',q=q,original_q=q,
            raw_path=str(baseline),score_path=str(base_score),origin='REUSED_CACHE',count_bin=n//20))
        variants=[('REFINED_NATIVE',qref,nn//20)]
        if q['parent_id'].startswith('459'):
            variants += [('ORIGINAL_NEW_BIN',q,nn//20),('REFINED_OLD_BIN',qref,n//20)]
        for label,qq,count_bin in variants:
            cell_id=label+'__'+rid
            cell=dict(cell_id=cell_id,label=label,q=qq,original_q=q,count_bin=count_bin,
                diagnostic_count_intervention=count_bin!=int(qq['N'])//20,
                raw_path=str(OUT/'outputs'/cell_id/'raw.npz'),origin='NEW')
            cells.append(cell);jobs.append(cell)
    assert len(cells)==16 and len(jobs)==10
    replay=dict(cell_id='BASELINE_REPLAY',label='BASELINE_REPLAY',q=qs[0],original_q=qs[0],
        count_bin=qs[0]['N']//20,diagnostic_count_intervention=False,
        reference=str(OLD/'evaluation/EXPOSURE'/qs[0]['request_id']/'raw/raw.npz'),
        raw_path=str(OUT/'outputs/BASELINE_REPLAY/raw.npz'),origin='REPLAY_VALIDATION')
    jobs=[replay]+jobs
    for i,j in enumerate(jobs): j['job_index']=i
    write(OUT/'inputs/cells.json',cells);write(OUT/'inputs/jobs.json',jobs);csvwrite(OUT/'inputs/panel.csv',rows)
    inputs={str(p):sha(p) for p in (OUT/'inputs').rglob('*') if p.is_file()}
    r=dict(schema='bench_c_tessellation_frozen_v1',status='FROZEN',weight=str(WEIGHT),weight_sha256=WEIGHT_SHA,
        parents=list(PREFIXES),requests=6,new_rollouts=10,baseline_validation_rollouts=1,generator_forward_budget=550,
        new_D0_forwards=0,training_updates=0,new_noise_arrays=0,output_stage='RAW_ONLY',
        new_original_metric_rows=10,reused_metric_rows=6,common_physical_C_extra_CF_I_rows=8,
        primary='Same physical C, fixed free count and fixed saved Gaussian; midpoint4 representation sensitivity',
        count_diagnostic='459 2x2 C tessellation x count-bin lookup only; true y always equals valid count; off-diagonal NOT production',
        not_identified='Pure attention dilution versus changed centers/edge/area Geo features; no generalization claim from 3 development parents',
        failure_contract='No silent retry of any started model call; completed jobs resume by immutable receipt only.',
        retention='Old inputs, weights, RAW, reports and scorers read only; no commit/push/default change',
        closest_routes=['R16 direct context readout','R28 contour readout','cloud synthetic count-readout toy'],
        difference='Frozen real generator representation intervention, not new readout training or loss',
        protected=protected,inputs=inputs)
    write(OUT/'registration.json',r);return r

def sample(indices,worker):
    import torch
    from meshflow_control import runtime
    from meshflow_control.checkpoints import state_hash
    from r2.pilots.bench_natural_n_recipe_v1.runtime import build_model,inference_layout
    from r2.pilots.bench_control_transfer_v1.sample import rollout
    from r2.pilots.seam_slot_c200_v1.generate import raw_arrays,dispatch_guard
    from r2.pilots.bench_four_routes_v1.resources import snapshot
    from r2.pilots.user_condition_finetune_v1.trainer import GPUObserver
    from meshflow_control.geo_io import write_indexed_ply
    r=read(OUT/'registration.json');verify(r['protected']);verify(r['inputs'])
    jobs=read(OUT/'inputs/jobs.json'); chosen=[jobs[i] for i in indices]
    if 0 not in indices and not (OUT/'outputs/BASELINE_REPLAY/complete.json').exists():
        raise RuntimeError('Baseline identity gate must pass first')
    stage=OUT/'workers'/worker
    if stage.exists(): raise RuntimeError('Worker already exists; use a new worker name after checking per-job receipts')
    stage.mkdir(parents=True);snap=snapshot();write(stage/'resources_before.json',snap)
    dispatch_guard(snap,2800)
    codefiles=[Path(__file__),ROOT/'r2/pilots/bench_natural_n_recipe_v1/runtime.py',
        ROOT/'r2/pilots/bench_control_transfer_v1/sample.py',ROOT/'r2/src/meshflow_control/models/native.py',
        ROOT/'r2/src/meshflow_control/models/geometry.py',ROOT/'r2/src/meshflow_control/models/backbone/attention.py']
    write(stage/'code_hashes.json',{str(p):sha(p) for p in codefiles})
    p=torch.load(WEIGHT,map_location='cpu',weights_only=True,mmap=True)
    assert p['schema']=='bench_four_routes_exposure_generator_v1' and p['global_step']==25500
    assert state_hash(p['generator_state'])==p['generator_state_sha256']
    runtime.configure_stable_runtime()
    model=build_model(copy.deepcopy(p['model_config']),readout_mode='none',trainable=False)
    model.load_state_dict(p['generator_state'],strict=True);model.to('cuda').set_trainable(False).eval()
    write(stage/'resources_loaded.json',snapshot())
    observer=GPUObserver('frozen_tessellation_rollouts_only',interval=1.);observer.__enter__()
    completed=[]
    try:
        for job in chosen:
            dest=resolve(job['raw_path']).parent
            if (dest/'complete.json').exists(): verify(read(dest/'complete.json')['files']);continue
            if dest.exists(): raise RuntimeError('Partial rollout forbids automatic replay: '+str(dest))
            dest.mkdir(parents=True);q=job['q'];inp=arrays(q['inference_npz']);source=arrays(q['source_npz'])
            core={k:inp[k] for k in ('C_vertices','C_faces_original','C_faces_model')};core['total_N']=q['N']
            lay=inference_layout(**core);noise=arrays(q['noise_npz'])['z'];hook=None;count_calls=[]
            def count_hook(module,args):
                count_calls.append(args[0].detach().cpu().tolist())
                expected=int(q['N'])//20
                if not bool((args[0]==expected).all()): raise ValueError('Unexpected count bucket input')
                return (torch.full_like(args[0],job['count_bin']),*args[1:])
            if job['diagnostic_count_intervention']: hook=model.backbone.y_embedder.register_forward_pre_hook(count_hook)
            write(dest/'attempt.json',dict(job=job,model_forward_budget=50,worker=worker,weight_sha=WEIGHT_SHA))
            try: raw,traj,audit=rollout(model,lay,noise,'PLAIN',dest/'calls.jsonl')
            finally:
                if hook is not None: hook.remove()
            if job['diagnostic_count_intervention'] and len(count_calls)!=50: raise ValueError('Count hook call budget')
            mesh=raw_arrays(raw,lay,np.full(len(lay['occupied']),-1,np.float32),source)
            npz(dest/'raw.npz',mesh);npz(dest/'trajectory.npz',dict(state=traj))
            write_indexed_ply(dest/'raw.ply',mesh['vertices'],mesh['faces'])
            if job['label']=='BASELINE_REPLAY':
                ref=arrays(job['reference']);keys=('raw','output','C','free','vertices','faces','raw_anchor_ids','source_corner_ids')
                diffs={k:bool(mesh[k].dtype==ref[k].dtype and np.array_equal(mesh[k],ref[k])) for k in keys}
                write(dest/'baseline_identity.json',dict(status='PASS' if all(diffs.values()) else 'FAIL',checks=diffs))
                if not all(diffs.values()): raise RuntimeError('Baseline replay differs; do not dispatch new experiment outputs')
            write(dest/'complete.json',dict(status='COMPLETE',job_index=job['job_index'],job_id=job['cell_id'],
                generator_forwards=50,training_updates=0,D0_forwards=0,new_noise=0,audit=audit,
                count_lookup_calls=len(count_calls),count_bin=job['count_bin'],actual_N=q['N'],
                files={str(x):sha(x) for x in dest.iterdir() if x.is_file()}))
            completed.append(job['job_index'])
            write(stage/('resources_after_'+str(job['job_index'])+'.json'),snapshot())
            write(stage/('peak_after_'+str(job['job_index'])+'.json'),dict(summary=observer.summary(),samples=observer.samples,audit=audit))
            print(json.dumps(dict(job_index=job['job_index'],id=job['cell_id'],seconds=audit['seconds'],worker=worker)),flush=True)
        assert state_hash(model.state_dict())==p['generator_state_sha256']
        write(stage/'complete.json',dict(status='COMPLETE',completed=completed,new_forward_calls=50*len(completed)))
    except BaseException as e:
        write(stage/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=completed,no_silent_retry=True));raise
    finally:
        observer.__exit__(None,None,None)
        write(stage/'resource_observation.json',dict(summary=observer.summary(),samples=observer.samples))
        write(stage/'resources_finished.json',snapshot())

def status():
    jobs=read(OUT/'inputs/jobs.json') if (OUT/'inputs/jobs.json').exists() else []
    return dict(jobs=[dict(index=j['job_index'],id=j['cell_id'],complete=(resolve(j['raw_path']).parent/'complete.json').exists()) for j in jobs])

def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','sample','status'])
    a.add_argument('--jobs',type=int,nargs='+');a.add_argument('--worker',default='worker0');v=a.parse_args()
    if v.action=='prepare': result=prepare();print(json.dumps({k:x for k,x in result.items() if k not in ('inputs','protected')},ensure_ascii=False))
    elif v.action=='sample': sample(v.jobs,v.worker)
    else: print(json.dumps(status(),indent=2))
if __name__=='__main__': main()
