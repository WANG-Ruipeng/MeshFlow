"""Preparation and frozen Euler50 sampling; old sources/results are read only."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,json,sys,traceback
from pathlib import Path
import numpy as np
from .common import *
sys.path.insert(0,str(ROOT/'r2/src'))


def prepare():
    if (OUT/'registration.json').exists():
        r=read(OUT/'registration.json');verify(r['protected']);verify(r['inputs']);return r
    from .geometry import flip_one
    from .evaluate import prepare_query_any
    from r2.pilots.bench_c_tessellation_v1.geometry import canonical_faces
    from meshflow_control.source_winding import capture_source,validate_source
    old=read(PREVIOUS/'registration.json');verify(old['protected']);verify(old['inputs'])
    baseline=read(PREVIOUS/'outputs/BASELINE_REPLAY/complete.json');verify(baseline['files'])
    assert read(PREVIOUS/'outputs/BASELINE_REPLAY/baseline_identity.json')['status']=='PASS'
    protected=dict(old['protected'])
    for p in [PREVIOUS/'inputs/cells.json',PREVIOUS/'outputs/BASELINE_REPLAY/complete.json',PREVIOUS/'workers/w0/code_hashes.json']:
        protected[str(p)]=sha(p)
    verify(read(PREVIOUS/'workers/w0/code_hashes.json'))
    originals=[c for c in read(PREVIOUS/'inputs/cells.json') if c['label']=='ORIGINAL']
    cells=[];jobs=[];panel=[];cache={};blocked=[]
    for oldcell in originals:
        q=copy.deepcopy(oldcell['q']);rid=q['request_id'];cid=q['condition_id'];inp=arrays(q['inference_npz'])
        cells.append(oldcell)
        if cid not in cache:
            f,audit=flip_one(inp['C_vertices'],inp['C_faces_original'])
            if audit['status']!='PASS':
                write(OUT/'inputs'/cid/'geometry_audit.json',audit)
                cache[cid]=None
                blocked.append(dict(condition_id=cid,parent_id=q['parent_id'],status='BLOCKED',reason=audit['reason']))
                continue
            model,p=canonical_faces(inp['C_vertices'],f)
            d=OUT/'inputs'/cid
            ni=dict(inp);ni.update(C_faces_original=f,C_faces_model=model)
            npz(d/'condition.npz',ni);write(d/'geometry_audit.json',audit)
            meta=arrays(q['metadata_npz']);npz(d/'metadata.npz',meta)
            src=capture_source(inp['C_vertices'][f],source_vertex_ids=meta['C_source_vertex_ids'][f],corner_permutation=p)
            validate_source(src,inp['C_vertices'][model]);npz(d/'source_condition.npz',src)
            query=prepare_query_any(q,inp['C_vertices'],f,d/'queries',new_C_faces_model=model)
            cache[cid]=(d,query,audit)
        if cache[cid] is None:continue
        d,query,audit=cache[cid]
        qq=copy.deepcopy(q);qq.update(inference_npz=str(d/'condition.npz'),metadata_npz=str(d/'metadata.npz'),
            source_npz=str(d/'source_condition.npz'),query_descriptor=query['descriptor'])
        for field,hfield in [('inference_npz','inference_npz_sha256'),('metadata_npz','metadata_sha256'),('source_npz','source_sha256')]:qq[hfield]=sha(qq[field])
        cell=dict(cell_id='EDGE_FLIP__'+rid,label='EDGE_FLIP',q=qq,original_q=q,
            raw_path=str(OUT/'outputs'/('EDGE_FLIP__'+rid)/'raw.npz'),origin='NEW',count_bin=int(q['N'])//20)
        cells.append(cell);jobs.append(cell)
        panel.append(dict(request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],K=q['K'],N=q['N'],
            free_faces=q['N']-q['K'],count_bin=q['N']//20,noise_sha256=sha(q['noise_npz']),
            original_condition_sha=sha(q['inference_npz']),flipped_condition_sha=sha(qq['inference_npz']),geometry_status=audit['status']))
    assert len(cells)==10 and len(jobs)==4 and len(blocked)==1
    for i,j in enumerate(jobs):j['job_index']=i
    write(OUT/'inputs/cells.json',cells);write(OUT/'inputs/jobs.json',jobs);csvwrite(OUT/'inputs/panel.csv',panel)
    r=dict(schema='bench_C_internal_edgeflip_frozen_v1',status='FROZEN',
        requests=6,new_rollouts=4,reused_RAW=6,generator_forward_budget=200,new_D0_forwards=0,training_updates=0,
        expected_cells=10,expected_new_raw=4,expected_pairs=4,blocked_inputs=blocked,
        new_noise_arrays=0,new_original_metric_rows=4,common_physical_C_extra_CF_I_rows=4,
        weight=str(WEIGHT),weight_sha256=WEIGHT_SHA,output_stage='RAW_ONLY',
        preserved='K,N,count bin,free slots/noise,original vertices,oriented external edges,physical surface',
        changed='One deterministic legal coplanar internal diagonal per C; changes centers/edge geometry and token contents, not token count',
        baseline_gate='Prior exact baseline replay reused and protected; identical generator/runtime/rollout source hashes',
        previous_routes='Frozen midpoint4 probe; R16/R28 readout training are distinct. This isolates triangulation from K/N/boundary-edge changes.',
        fragment_scope='RAW skinny/tiny triangles and actual image review; soup exact-index islands are not visual detached-fragment count. Old FINAL is background only, not EDGE_FLIP quality.',
        failure_contract='No silent retry of any begun model forward; no unregistered new noise or checkpoint selection',
        protected=protected,inputs={str(p):sha(p) for p in (OUT/'inputs').rglob('*') if p.is_file()})
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
    jobs=read(OUT/'inputs/jobs.json');chosen=[jobs[i] for i in indices]
    if all((resolve(j['raw_path']).parent/'complete.json').exists() for j in chosen):
        for j in chosen:verify(read(resolve(j['raw_path']).parent/'complete.json')['files'])
        print('COMPLETE_RECEIPTS_VERIFIED_NO_MODEL_LOAD');return
    stage=OUT/'workers'/worker
    if stage.exists():raise RuntimeError('Worker directory exists; inspect receipts, no silent replay')
    stage.mkdir(parents=True);snap=snapshot();write(stage/'resources_before.json',snap)
    dispatch_guard(snap,2800)
    codefiles=[Path(__file__),ROOT/'r2/pilots/bench_natural_n_recipe_v1/runtime.py',ROOT/'r2/pilots/bench_control_transfer_v1/sample.py',
        ROOT/'r2/src/meshflow_control/models/native.py',ROOT/'r2/src/meshflow_control/models/geometry.py',ROOT/'r2/src/meshflow_control/models/backbone/attention.py']
    write(stage/'code_hashes.json',{str(p):sha(p) for p in codefiles})
    p=torch.load(WEIGHT,map_location='cpu',weights_only=True,mmap=True)
    assert p['schema']=='bench_four_routes_exposure_generator_v1' and p['global_step']==25500
    assert state_hash(p['generator_state'])==p['generator_state_sha256']
    runtime.configure_stable_runtime()
    model=build_model(copy.deepcopy(p['model_config']),readout_mode='none',trainable=False)
    model.load_state_dict(p['generator_state'],strict=True);model.to('cuda').set_trainable(False).eval()
    write(stage/'resources_loaded.json',snapshot())
    observer=GPUObserver('frozen_edgeflip_rollouts_only',interval=1.);observer.__enter__();completed=[]
    try:
        for job in chosen:
            dest=resolve(job['raw_path']).parent
            if (dest/'complete.json').exists():verify(read(dest/'complete.json')['files']);continue
            if dest.exists():raise RuntimeError('Partial rollout forbids automatic replay: '+str(dest))
            dest.mkdir(parents=True);q=job['q'];inp=arrays(q['inference_npz']);source=arrays(q['source_npz'])
            lay=inference_layout(inp['C_vertices'],inp['C_faces_original'],inp['C_faces_model'],q['N'])
            noise=arrays(q['noise_npz'])['z']
            write(dest/'attempt.json',dict(job=job,model_forward_budget=50,worker=worker,weight_sha=WEIGHT_SHA))
            raw,traj,audit=rollout(model,lay,noise,'PLAIN',dest/'calls.jsonl')
            mesh=raw_arrays(raw,lay,np.full(len(lay['occupied']),-1,np.float32),source)
            npz(dest/'raw.npz',mesh);npz(dest/'trajectory.npz',dict(state=traj))
            write_indexed_ply(dest/'raw.ply',mesh['vertices'],mesh['faces'])
            write(dest/'complete.json',dict(status='COMPLETE',job_index=job['job_index'],job_id=job['cell_id'],
                generator_forwards=50,training_updates=0,D0_forwards=0,new_noise=0,audit=audit,
                actual_N=q['N'],count_bin=q['N']//20,files={str(x):sha(x) for x in dest.iterdir() if x.is_file()}))
            completed.append(job['job_index']);write(stage/('resources_after_'+str(job['job_index'])+'.json'),snapshot())
            write(stage/('peak_after_'+str(job['job_index'])+'.json'),dict(summary=observer.summary(),samples=observer.samples,audit=audit))
            print(json.dumps(dict(job_index=job['job_index'],id=job['cell_id'],seconds=audit['seconds'],worker=worker)),flush=True)
        after=state_hash(model.state_dict());assert after==p['generator_state_sha256']
        write(stage/'complete.json',dict(status='COMPLETE',completed=completed,new_forward_calls=50*len(completed),model_state_sha256_after=after))
    except BaseException as e:
        write(stage/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=completed,no_silent_retry=True));raise
    finally:
        observer.__exit__(None,None,None);write(stage/'resource_observation.json',dict(summary=observer.summary(),samples=observer.samples))
        write(stage/'resources_finished.json',snapshot())


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','sample','status'));p.add_argument('--jobs',nargs='+',type=int);p.add_argument('--worker',default='w0');a=p.parse_args()
    if a.action=='prepare':
        r=prepare();print(json.dumps({k:v for k,v in r.items() if k not in ('protected','inputs')}))
    elif a.action=='sample':sample(a.jobs,a.worker)
    else:
        rows=read(OUT/'inputs/jobs.json');print(json.dumps([dict(job=j['cell_id'],complete=(resolve(j['raw_path']).parent/'complete.json').exists()) for j in rows]))

if __name__=='__main__':main()
