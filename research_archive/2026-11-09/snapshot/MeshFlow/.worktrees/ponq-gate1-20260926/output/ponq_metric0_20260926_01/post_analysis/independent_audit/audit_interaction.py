"""Independent saved-artifact audit of SINGLE_STATE_INTERACTION0; no extraction."""
from pathlib import Path
import argparse,collections,hashlib,itertools,json,shutil,sys,time
import numpy as np
import torch
from audit_fm_outputs import independent_topology,valid,failure_tags
from audit_single_state import decode,assign,semantic,near_dict,canon

ARMS=['B0','B1','B2'];IDS=[0,4,8,12,16,20,24,28];COMBOS=['POO','OPO','OOP']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def equal(a,b):return np.array_equal(a,b,equal_nan=True)
def deep_equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(deep_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(deep_equal(x,y) for x,y in zip(a,b))
    return a==b
def noise_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def summary(vals):
    a=np.asarray(vals,float)
    return dict(count=len(a),minimum=float(a.min()),median=float(np.median(a)),maximum=float(a.max()),mean=float(a.mean())) if len(a) else dict(count=0)


def contains(saved,want):
    return all(k in saved and ((v is None and saved[k] is None) or (v is not None and saved[k] is not None and np.isclose(v,saved[k],rtol=2e-8,atol=2e-10))) for k,v in want.items())
def extra_semantic(state,target,diag):
    P=state[:,:3];T=target[:,:3];D=np.sqrt(((P[:,None,:]-T[None,:,:])**2).sum(-1));a=D.min(1);b=D.min(0);out={}
    for frac,label in [(.001,'0p1pct'),(.005,'0p5pct'),(.01,'1p0pct')]:
        for dist,prefix in [(b,'P_target_coverage_'),(a,'P_pred_target_proximity_')]:
            count=int((dist<=frac*diag).sum());out[prefix+label+'_count']=count;out[prefix+label+'_fraction']=count/256
    dd=np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(-1));np.fill_diagonal(dd,np.inf);close=dd<=.001*diag
    out.update(P_near_duplicate_pair_count=int(np.triu(close,1).sum()),P_near_duplicate_point_count=int(close.any(0).sum()),P_near_duplicate_point_fraction=float(close.any(0).mean()),P_exact_duplicate_excess_count=len(P)-len(np.unique(P,axis=0)),P_nearest_other_distance_min=float(dd.min()))
    out['P_symmetric_nn_rms_pct_bbox']=float(100*np.sqrt(.5*(np.mean(a*a)+np.mean(b*b)))/diag)
    n=state[:,3:6];nt=target[:,3:6];ln=np.linalg.norm(n,axis=1);lt=np.linalg.norm(nt,axis=1);good=(ln>1e-8)&(lt>1e-8)
    out.update(n_nearzero_count=int((ln<=1e-8).sum()),n_target_nearzero_count=int((lt<=1e-8).sum()),n_direction_usable_count=int(good.sum()),n_length_mae=float(np.mean(abs(ln-lt))),n_length_signed_mean_error=float(np.mean(ln-lt)))
    if good.any():
        cos=np.clip(np.sum(n[good]*nt[good],axis=1)/(ln[good]*lt[good]),-1,1);angles=np.degrees(np.arccos(cos));rev=int((cos<0).sum())
        out.update(n_reverse_count=rev,n_reverse_rate=rev/int(good.sum()),n_oriented_angle_mean_deg_above_nearzero=float(angles.mean()),n_oriented_angle_p95_deg_above_nearzero=float(np.quantile(angles,.95)))
    else:out.update(n_reverse_count=0,n_reverse_rate=None,n_oriented_angle_mean_deg_above_nearzero=None,n_oriented_angle_p95_deg_above_nearzero=None)
    return out


def main(root):
    started=time.perf_counter();torch.set_num_threads(1);root=Path(root).resolve();out=root/'post_analysis'/'independent_audit'
    if out.exists():raise RuntimeError('Refuse existing independent audit')
    checks=[]
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    protocol=read(root/'RUN_PROTOCOL.json');execution=read(root/'EXECUTION.json');previous=Path(protocol['previous_run'])
    check('execution_complete',execution['status']=='COMPLETE')
    preflight=read(root/'preflight/RESULT.json');check('archived_preflight44',preflight['passed'] and preflight['check_count']==44 and all(c['passed'] for c in preflight['checks']) and all(sha(r['path'])==r['sha256'] for r in preflight['source_files']))
    manifest=read(root/'MANIFEST.json');bad=[]
    for r in manifest['files']:
        p=root/r['path']
        if not p.is_file() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:bad.append(r['path'])
    check('original_manifest',not bad,dict(count=len(manifest['files']),bad=bad))
    check('protocol_sha',sha(root/'RUN_PROTOCOL.json')==read(root/'RUN_PROTOCOL_SHA256.json')['sha256'])
    check('frozen_data_hashes',all(sha(root/k)==v for k,v in protocol['hashes'].items()))
    check('source_snapshot',all(sha(root/r['path'])==r['sha256'] for r in read(root/'SOURCE_SNAPSHOT.json')['files']))
    check('parent_sources',all(sha(r['path'])==r['sha256'] for r in protocol['parent_sources']))
    parent=torch.load(root/'parent/final.pt',map_location='cpu',weights_only=False);oldnames=list(parent['model_state_dict']);oldopt=parent['optimizer_state_dict'];oldids=oldopt['param_groups'][0]['params'];old_by_name=dict(zip(oldnames,oldids))
    check('parent_checkpoint_bytes',sha(root/'parent/final.pt')==sha(previous/'training/direct/final.pt')==protocol['parent_checkpoint_sha256'])
    check('unchanged_data_chart',sha(root/'data.npz')==sha(previous/'data.npz') and sha(root/'probe_noise.npz')==sha(previous/'probe_noise.npz'))
    data=arrays(root/'data.npz');target=data['states'][0];diag=float(np.linalg.norm(np.ptp(target[:,:3],axis=0)))
    samples=arrays(root/'sample_noise.npz');noise=samples['noise'];oldnoise=arrays(root/'sample_noise_old.npz')['noise'];newnoise=arrays(root/'sample_noise_new.npz')['noise'];probe=arrays(root/'probe_noise.npz')
    check('sample64_frozen',noise.shape==(64,256,22) and equal(noise[:32],oldnoise) and equal(noise[32:],newnoise) and sha(root/'sample_noise_old.npz')==sha(previous/'sample_noise.npz'))
    check('new32_seed_replay',equal(newnoise,np.random.default_rng(260930).standard_normal((32,256,22)).astype(np.float32)))
    check('noise_split_labels',samples['split'].tolist()==['old']*32+['new']*32 and equal(samples['split_id'],np.tile(np.arange(32),2)))
    forbidden={noise_hash(x) for x in noise}|{noise_hash(x) for x in probe['noise']}
    check('new_noise_not_old_or_probe',not ({noise_hash(x) for x in newnoise}&({noise_hash(x) for x in oldnoise}|{noise_hash(x) for x in probe['noise']})))
    # This is an audit of the actual archived class, not the mutable module path.
    sys.path.insert(0,str(root/'code'))
    for key in ['fm_model','interaction_model']:
        sys.modules.pop(key,None)
    from fm_model import DeepSetsVelocity
    from interaction_model import InteractionVelocity
    legacy=DeepSetsVelocity(128);legacy.load_state_dict(parent['model_state_dict']);legacy.eval()
    z=torch.from_numpy(probe['noise'][:2]);tt=torch.tensor([.25,.9]);perm=np.random.default_rng(781).permutation(256)
    with torch.no_grad():expected_velocity=legacy(z,tt)
    training={};traces={};initials={};finals={};function_differences={}
    for arm in ARMS:
        d=root/'training'/arm;r=read(d/'RESULT.json');p=read(d/'PROTOCOL.json');initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);final=torch.load(d/'final.pt',map_location='cpu',weights_only=False)
        trace=[json.loads(line) for line in (d/'TRAIN_TRACE.jsonl').read_text().splitlines()];traces[arm]=trace;initials[arm]=initial;finals[arm]=final
        check(arm+'_fixed_updates',r['status']=='COMPLETED' and r['completed_steps']==9000 and r['new_updates']==6000 and r['new_attempted_batches']==6000 and [x['step'] for x in trace]==list(range(3001,9001)) and [x['new_update'] for x in trace]==list(range(1,6001)))
        bindings={'final.pt':'final_checkpoint_sha256','initial.pt':'initial_checkpoint_sha256','initial_model.pt':'initial_model_sha256','OPTIMIZER_MAPPING.json':'optimizer_mapping_sha256','PROTOCOL.json':'protocol_sha256'}
        check(arm+'_hash_bindings',all(sha(d/f)==r[k] for f,k in bindings.items()) and final['protocol_sha256']==sha(d/'PROTOCOL.json') and initial['parent_checkpoint_sha256']==sha(root/'parent/final.pt')==final['parent_checkpoint_sha256'])
        check(arm+'_data_chart_binding',p['data_sha256']==sha(root/'data.npz') and deep_equal(initial['mean'],parent['mean']) and deep_equal(initial['scale'],parent['scale']) and deep_equal(final['mean'],parent['mean']) and deep_equal(final['scale'],parent['scale']))
        names=initial['optimizer_param_names'];io=initial['optimizer_state_dict'];fo=final['optimizer_state_dict'];initial_ids=io['param_groups'][0]['params'];final_ids=fo['param_groups'][0]['params'];imap=dict(zip(names,initial_ids));fmap=dict(zip(final['optimizer_param_names'],final_ids))
        check(arm+'_name_mapping',len(names)==len(initial_ids) and set(names)==set(initial['model_state_dict']) and names[:len(oldnames)]==oldnames)
        check(arm+'_finite_final_model',all(torch.isfinite(v).all() for v in final['model_state_dict'].values()))
        check(arm+'_old_model_tensors_exact',all(deep_equal(initial['model_state_dict'][k],parent['model_state_dict'][k]) for k in oldnames))
        check(arm+'_old_optimizer_exact',all(deep_equal(io['state'][imap[k]],oldopt['state'][old_by_name[k]]) for k in oldnames))
        check(arm+'_optimizer_options_exact',{k:v for k,v in io['param_groups'][0].items() if k!='params'}=={k:v for k,v in oldopt['param_groups'][0].items() if k!='params'})
        added=[k for k in names if k not in oldnames]
        check(arm+'_new_optimizer_zero',all(all(torch.count_nonzero(io['state'][imap[k]][part]).item()==0 for part in ['step','exp_avg','exp_avg_sq']) for k in added))
        initial_model=torch.load(d/'initial_model.pt',map_location='cpu',weights_only=True);check(arm+'_initial_model_file_binding',deep_equal(initial_model,initial['model_state_dict']))
        zero_keys=[] if arm=='B0' else [f'interaction.linear2.{k}' for k in ['weight','bias']] if arm=='B1' else [f'interaction.attention.out_proj.{k}' for k in ['weight','bias']]
        check(arm+'_zero_residual_projection',all(torch.count_nonzero(initial['model_state_dict'][k]).item()==0 for k in zero_keys))
        check(arm+'_final_optimizer_steps',all(float(fo['state'][fmap[k]]['step'])==(9000 if k in oldnames else 6000) for k in names))
        check(arm+'_initial_rng_exact',deep_equal(initial['rng'],parent['rng']))
        check(arm+'_initial_progress_exact',initial['completed_steps']==3000 and initial['new_updates']==0 and initial['exposure_counts']==[24000] and initial['new_exposure_counts']==[0] and initial['attempted_stream_sha256']==parent['attempted_stream_sha256'])
        check(arm+'_final_progress_exact',final['completed_steps']==9000 and final['new_updates']==6000 and final['exposure_counts']==r['exposure_counts']==[72000] and final['new_exposure_counts']==r['new_exposure_counts']==[48000])
        new_digest=bytes(32);cumulative=bytes.fromhex(parent['attempted_stream_sha256'])
        for row in trace:
            bh=bytes.fromhex(row['batch_sha256']);new_digest=hashlib.sha256(new_digest+bh).digest();cumulative=hashlib.sha256(cumulative+bh).digest()
        check(arm+'_stream_hashes',new_digest.hex()==r['new_attempted_stream_sha256']==final['new_attempted_stream_sha256'] and cumulative.hex()==r['attempted_stream_sha256']==final['attempted_stream_sha256'])
        check(arm+'_block_loss_identity',all(np.isclose(row['loss'],(3*row['x_mse']+3*row['n_mse']+16*row['F_mse'])/22,rtol=2e-6,atol=2e-7) for row in trace))
        model=InteractionVelocity(arm,128);model.load_state_dict(initial['model_state_dict']);model.eval()
        with torch.no_grad():velocity=model(z,tt);reordered=model(z[:,perm],tt)
        function_differences[arm]=dict(initial_max_abs=float((velocity-expected_velocity).abs().max()),permutation_max_abs=float((reordered-velocity[:,perm]).abs().max()))
        check(arm+'_same_initial_function',torch.equal(velocity,expected_velocity),function_differences[arm])
        check(arm+'_permutation_equivariance',torch.allclose(reordered,velocity[:,perm],rtol=3e-5,atol=3e-5))
        counts=model.parameter_counts();check(arm+'_parameter_counts',counts==r['parameter_counts'] and counts['added']==dict(B0=0,B1=66176,B2=66304)[arm])
        training[arm]=dict(counts=counts,completed_steps=r['completed_steps'],new_updates=r['new_updates'],training_seconds=r['training_elapsed_seconds'],total_seconds=r['total_wall_seconds_including_initialization_and_save'],final_loss=trace[-1]['loss'],last100_loss=float(np.mean([t['loss'] for t in trace[-100:]])),checkpoint_sha256=sha(d/'final.pt'))
    check('three_paired6000_batch_hashes',all([r['batch_sha256'] for r in traces[arm]]==[r['batch_sha256'] for r in traces['B0']] for arm in ARMS))
    generator=torch.Generator();generator.set_state(parent['rng']['batch_generator_cpu'].cpu());batch_replay=[];noise_collisions=[]
    for step in range(6000):
        n=torch.randn(8,256,22,generator=generator);ts=torch.rand(8,1,1,generator=generator)
        for bi,x in enumerate(n.numpy()):
            if noise_hash(x) in forbidden:noise_collisions.append([step+3001,bi])
        if step in [0,999,2999,5999]:
            sx=n.numpy()[...,:3]*data['scale'][:3]+data['mean'][:3]
            targets=np.stack([((target[assign(x,target[:,:3])]-data['mean'])/data['scale']).astype(np.float32) for x in sx])
            h=hashlib.sha256(n.numpy().tobytes()+ts.numpy().tobytes()+targets.tobytes()+np.zeros(8,np.int64).tobytes()).hexdigest();batch_replay.append(dict(step=step+3001,matched=h==traces['B0'][step]['batch_sha256'],sha256=h))
    check('independent_continued_batch_replay',all(x['matched'] for x in batch_replay),batch_replay)
    check('no_exact_noise_reuse_new48000',not noise_collisions,noise_collisions)
    check('final_batch_rng_matches_replay',all(torch.equal(c['rng']['batch_generator_cpu'],generator.get_state()) for c in finals.values()))
    # One-target balanced schedule consumes exactly eight cycles per update.
    check('final_balanced_schedule',all(c['rng']['balanced_schedule']['cycles']==72000 and c['rng']['balanced_schedule']['cursor']==1 and c['rng']['balanced_schedule']['order']==[0] for c in finals.values()))
    geometry_records=[];results={};signset=set(itertools.product([-1.,1.],repeat=3));canonical=arrays(root/'target/canonical_mesh.npz');canonical_faces=canon(canonical['input_vertex_ids'][canonical['faces']])
    def geometry_check(row,directory,expected,permutation):
        label=str(directory.relative_to(root));check(label+'_state',equal(arrays(directory/'state.npz')['physical_state'],expected));check(label+'_row',read(directory/'result.json')==row)
        pp=directory/'parameters.npz'
        if pp.is_file():
            p=arrays(pp);want=decode(expected);check(label+'_decode',all(equal(p[k],want[k]) for k in want) and sha(pp)==row['parameters_sha256'])
            corners=arrays(directory/'corners.npz')['corners'];check(label+'_fixed_corners',corners.shape==(8,3) and set(map(tuple,corners))==signset and row['domain']=='fixed')
            tracep=directory/'extractor_trace.npz'
            if tracep.is_file():
                trace=arrays(tracep)
                if 'delaunay_points' in trace:check(label+'_trace_sites',equal(trace['delaunay_points'][:256],p['v']) and equal(trace['delaunay_points'][256:],corners))
        check(label+'_valid',valid(row)==row['valid'])
        obs=dict(location=label,arm=row['arm'],sample_id=row['sample_id'],split=row['split'],combination=row['combination'],valid=valid(row),status=row['status'],faces=row.get('faces'),rms_pct_bboxdiag=row.get('rms_pct_bboxdiag'),failure_tags=failure_tags(row))
        if (directory/'mesh.npz').is_file():
            m=arrays(directory/'mesh.npz');top=independent_topology(m['vertices'],m['faces']);obs['independent_topology']=top
            check(label+'_topology',top['faces']==row['faces'] and len(top['bad_vertex_links'])==row['strict_noncycle_vertex_count'] and top['nonmanifold_edges']==row['nonmanifold_edges'] and top['boundary_edges']==row['boundary_edges'] and (top['betti_if_closed_oriented'] is None or top['betti_if_closed_oriented']==[row['beta'+str(k)] for k in range(3)]))
            check(label+'_mesh_hash',sha(directory/'mesh.obj')==row['mesh_sha256'])
            if np.all(permutation>=0):
                mapping=np.r_[permutation,np.arange(256,264)];obs['same_target_id_surface']=equal(canon(mapping[m['input_vertex_ids'][m['faces']]]),canonical_faces)
        geometry_records.append(obs)
    for arm in ARMS:
        d=root/'evaluation'/arm;ep=arrays(d/'endpoints.npz');roll=read(d/'ROLLOUT.json');free=read(d/'FREE_LEDGER.json');field=read(d/'FIELD_LEDGER.json');stored=read(d/'SUMMARY.json')
        check(arm+'_64_endpoints',ep['states'].shape==(64,256,22) and ep['target_permutations'].shape==(64,256) and len(roll['rows'])==64)
        check(arm+'_rollout_binding',roll['noise_sha256']==sha(root/'sample_noise.npz') and roll['checkpoint_sha256']==training[arm]['checkpoint_sha256'])
        check(arm+'_64_free_slots',len(free)==64 and {r['sample_id'] for r in free}==set(range(64)))
        check(arm+'_24_field_slots',len(field)==24 and {(r['sample_id'],r['combination']) for r in field}==set(itertools.product(IDS,COMBOS)))
        for r in roll['rows']:
            i=r['sample_id'];pred=ep['states'][i];perm=ep['target_permutations'][i]
            check(f'{arm}_{i}_split',r['split']==('old' if i<32 else 'new') and r['split_id']==i%32 and r['checkpoint_sha256']==training[arm]['checkpoint_sha256'])
            if r['status']=='COMPLETE':
                check(f'{arm}_{i}_Hungarian',equal(perm,assign(pred[:,:3],target[:,:3])) and r['nfe']==64)
                basic=semantic(pred,target[perm],diag);saved=r['endpoint_semantic'];check(f'{arm}_{i}_semantic_basic',all(k in saved and ((v is None and saved[k] is None) or (v is not None and saved[k] is not None and np.isclose(v,saved[k],rtol=2e-8,atol=2e-10))) for k,v in basic.items()))
                check(f'{arm}_{i}_semantic_extended',contains(saved,extra_semantic(pred,target[perm],diag)))
        for row in free+field:
            i=row['sample_id'];perm=ep['target_permutations'][i];pred=ep['states'][i];c=row['combination'];available=np.all(perm>=0) and len(np.unique(perm))==256
            expected=pred if c=='PPP' else np.concatenate([pred[:,a:b] if mark=='P' else target[perm,a:b] for mark,(a,b) in zip(c,[(0,3),(3,6),(6,22)])],axis=1) if available else np.full((256,22),np.nan)
            check(f'{arm}_{i}_{c}_metadata',row['arm']==arm and row['nfe']==64 and row['domain']=='fixed' and row['generated']==(c=='PPP') and row['oracle_assisted']==(c!='PPP') and row['split']==('old' if i<32 else 'new') and row['split_id']==i%32 and equal(row['target_permutation'],perm) and row['checkpoint_sha256']==training[arm]['checkpoint_sha256'] and row['noise_sha256']==sha(root/'sample_noise.npz'))
            geometry_check(row,d/f'{i:03d}_{c}_fixed',expected,perm)
            if c=='PPP':check(f'{arm}_{i}_quality_gate',row['quality_pass']==bool(valid(row) and row.get('rms_pct_bboxdiag',float('inf'))<=.20))
        results[arm]=dict(splits={},field_valid={c:sum(valid(r) for r in field if r['combination']==c) for c in COMBOS},integration_seconds=roll['integration_seconds'],posthoc_seconds=roll['posthoc_seconds'],geometry_evaluation_seconds=stored['extraction_evaluation_seconds'])
        for split in ['old','new']:
            rr=[r for r in free if r['split']==split];results[arm]['splits'][split]=dict(count=len(rr),valid=sum(valid(r) for r in rr),quality_pass=sum(valid(r) and r.get('rms_pct_bboxdiag',float('inf'))<=.20 for r in rr),all_output_rms=summary([r['rms_pct_bboxdiag'] for r in rr if 'rms_pct_bboxdiag' in r]),valid_output_rms=summary([r['rms_pct_bboxdiag'] for r in rr if valid(r)]),faces=summary([r['faces'] for r in rr if 'faces' in r]),failure_tags=dict(collections.Counter(t for r in rr for t in failure_tags(r))))
            check(arm+'_'+split+'_summary',results[arm]['splits'][split]['valid']==stored['splits'][split]['valid'] and results[arm]['splits'][split]['quality_pass']==stored['splits'][split]['quality_pass'])
        results[arm]['advance']=results[arm]['splits']['new']['quality_pass']>=24;check(arm+'_advance_gate',results[arm]['advance']==stored['advance'])
        check(arm+'_field_summary',results[arm]['field_valid']==stored['field_valid'])
        check(arm+'_sampling_timing_sum',np.isclose(sum(r['integration_seconds'] for r in roll['rows']),roll['integration_seconds']) and np.isclose(sum(r['posthoc_seconds'] for r in roll['rows']),roll['posthoc_seconds']))
    check('total192_free72_field',len(geometry_records)==264 and sum(r['combination']=='PPP' for r in geometry_records)==192)
    probe_results={}
    for arm in ARMS:
        d=root/'probes'/arm;rows=read(d/'PROBE_ROWS.json');grads=read(d/'GRADIENT_ROWS.json');states=arrays(d/'probe_states.npz');sr=read(d/'SUMMARY.json');times=probe['times']
        check(arm+'_192_probe_slots',len(rows)==192 and {(r['probe_id'],r['time']) for r in rows}==set(itertools.product(range(32),map(float,times))))
        check(arm+'_24_gradient_slots',len(grads)==24 and {(r['probe_id'],r['time']) for r in grads}==set(itertools.product([0,8,16,24],map(float,times))))
        check(arm+'_probe_bindings',sr['checkpoint_sha256']==training[arm]['checkpoint_sha256'] and sr['probe_noise_sha256']==sha(root/'probe_noise.npz') and sr['data_sha256']==sha(root/'data.npz'))
        check(arm+'_probe_no_updates',sr['optimizer_updates']==0 and sr['geometry_extractions']==0 and sr['model_weights_unchanged'] and sr['model_tensor_digest_before']==sr['model_tensor_digest_after'] and sr['parameter_grad_buffers_remain_none'])
        for row in rows:
            i=row['probe_id'];ti=int(np.where(times==row['time'])[0][0]);t=float(times[ti]);perm=assign(probe['noise'][i,:,:3].astype(np.float64)*data['scale'][:3]+data['mean'][:3],target[:,:3]);aligned=target[perm]
            check(f'{arm}_probe_{i}_{ti}_matching',equal(perm,states['target_permutations'][i]))
            zt=((aligned-data['mean'])/data['scale']).astype(np.float32);z0=probe['noise'][i];path=(((1-t)*z0+t*zt)*data['scale'].astype(np.float32)+data['mean'].astype(np.float32)).astype(np.float64)
            check(f'{arm}_probe_{i}_{ti}_path',np.allclose(states['path_states'][i,ti],path,rtol=2e-6,atol=2e-7))
            baseline=states['path_states'][i,ti]
            check(f'{arm}_probe_{i}_{ti}_zero_baseline',contains(row['zero_velocity_endpoint'],semantic(baseline,aligned,diag)) and contains(row['zero_velocity_endpoint'],extra_semantic(baseline,aligned,diag)))
            if row['status']=='COMPLETE':
                state=states['endpoint_predictions'][i,ti];check(f'{arm}_probe_{i}_{ti}_endpoint',contains(row['endpoint'],semantic(state,aligned,diag)) and contains(row['endpoint'],extra_semantic(state,aligned,diag)))
                check(f'{arm}_probe_{i}_{ti}_loss_weights',np.isclose(row['velocity_mse'],(3*row['velocity_x_mse']+3*row['velocity_n_mse']+16*row['velocity_F_mse'])/22,rtol=2e-6,atol=2e-7))
        for g in grads:
            weights={'x':3/22,'n':3/22,'F':16/22};tag=f'{arm}_gradient_{g["probe_id"]}_{g["time"]}'
            check(tag+'_weighted_norms',all(np.isclose(g['gradient_'+k+'_weighted_norm'],w*g['gradient_'+k+'_unweighted_norm'],rtol=1e-12,atol=1e-15) for k,w in weights.items()))
            squared=sum(g['gradient_'+k+'_weighted_norm']**2 for k in weights)
            for a,b in [('x','n'),('x','F'),('n','F')]:
                cosine=g['gradient_cosine_'+a+'_'+b]
                if cosine is not None:squared+=2*g['gradient_'+a+'_weighted_norm']*g['gradient_'+b+'_weighted_norm']*cosine
            check(tag+'_vector_sum_identity',np.isclose(g['gradient_weighted_sum_norm']**2,squared,rtol=3e-5,atol=1e-10))
            check(tag+'_total_gradient_identity',g['all_gradients_finite'] and g['shared_hidden_shape']==[1,256,128] and g['gradient_weighted_sum_difference_norm']<=1e-6*g['gradient_total_norm']+1e-7)
        probe_results[arm]=dict(semantic_records=len(rows),completed=sum(r['status']=='COMPLETE' for r in rows),gradient_records=len(grads),seconds=sr['seconds'],no_updates=sr['optimizer_updates']==0)
    result=dict(passed=all(c['passed'] for c in checks),check_count=len(checks),failed_checks=[c for c in checks if not c['passed']],checks=checks,training=training,initial_function_differences=function_differences,independent_batch_replay=batch_replay,results=results,geometry_records=geometry_records,execution=execution,audit_seconds=time.perf_counter()-started,source_manifest_sha256=sha(root/'MANIFEST.json'),scope='Saved-artifact audit: complete optimizer/RNG restoration, CPU initial-function and permutation tests, paired/replayed batch streams, selected chart/decode/domain/topology and full denominators. No optimizer updates, extraction, training, sampling, or exact intersection rerun.',limitations=['New noise is same-object confirmation, not new-parent generalization.','A matched added parameter count does not match FLOPs or optimization geometry.','B1/B2 initial functions match, but their first clipped updates need not match B0 because new-parameter gradients contribute to global clipping.','Only PPP free rows count toward progression; field interventions are privileged diagnostics.','Geometry valid and valid-plus-0.20%-RMS remain separate outcomes.'])
    result['probes']=probe_results
    out.mkdir(parents=True);write(out/'INTERACTION_AUDIT.json',result)
    for name in ['audit_interaction.py','audit_single_state.py','audit_fm_outputs.py']:shutil.copy2(Path(__file__).with_name(name),out/name)
    write(out/'SUPPLEMENTAL_MANIFEST.json',dict(original_manifest_sha256=sha(root/'MANIFEST.json'),files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]))
    print(json.dumps({k:result[k] for k in ['passed','check_count','failed_checks','training','initial_function_differences','results','audit_seconds']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);main(p.parse_args().root)
