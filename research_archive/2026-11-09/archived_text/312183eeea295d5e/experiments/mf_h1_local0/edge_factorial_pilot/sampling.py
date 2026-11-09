"""Four frozen endpoints, registered contexts and original Gaussian Euler paths."""
from .common import *
from .checkpoint import load_endpoint
from native_t1.sampling import make_noise,clamped_sample,new_counts

def validate_arrays(raw,states,z,C):
 if raw.dtype!=np.float32 or raw.shape!=(112,9) or states.dtype!=np.float32 or states.shape!=(51,112,9):raise ValueError('Output shape/dtype')
 if not np.isfinite(raw).all() or not np.isfinite(states).all():raise ValueError('Nonfinite output')
 if states[:,:12].tobytes()!=np.broadcast_to(C.reshape(1,12,9),(51,12,9)).tobytes():raise ValueError('C changed')
 if states[0,12:].tobytes()!=z[0,12:].tobytes() or states[-1].tobytes()!=raw.tobytes():raise ValueError('Noise/endpoint changed')

def sample_one(r,out,model,name,unit,seed,z,C,reference=None):
 c=r['counts']['sampling'];replay=reference is not None;key='replay_output_attempts' if replay else 'formal_output_attempts'
 if c.get(key,0)>=(2 if replay else r['protocol']['evaluation']['expected_outputs']):raise RuntimeError('Complete generation cap')
 inc(c,key);local=new_counts();start=time.perf_counter();save_seconds=0.;native_seconds=0.
 identity=dict(model=name,parent=unit['parent'],patch=unit['patch'],seed=seed,K=12,phase='pilot',patch_group=unit['patch_group'])
 attempt=dict(identity,replay=replay,status='RUNNING',local_counts=local);r.setdefault('sampling_attempts',[]).append(attempt)
 try:
  raw,states,audit=clamped_sample(model,torch.from_numpy(z.copy()).cuda(),torch.from_numpy(C.copy()).cuda(),local)
  native_seconds=audit['seconds'];validate_arrays(raw,states,z,C)
  if local['model_forward_returns']!=50 or local['complete_rollouts']!=1 or any(local[k] for k in ('model_backward_calls','optimizer_updates','geometry_gradient_calls','postprocessing_repairs')):raise RuntimeError('Sampler accounting differs')
  row=dict(identity,sampling=audit,C_all51_bitwise=True,known_all51_bitwise=True,total_faces=112,free_faces=100,
   output_sha256=array_hash(raw),path_sha256=array_hash(states),initial_gaussian_sha256=array_hash(z),
   geometry_loss_calls=0,geometry_gradient_calls=0,OT_calls=0,postprocessing_repairs=0,reused=False)
  start_save=time.perf_counter()
  if replay:
   if file_sha256(reference['raw_path'])!=reference['raw_sha256']:raise RuntimeError('Replay reference changed')
   with np.load(reference['raw_path'],allow_pickle=False) as previous:
    if any(previous[k].tobytes()!=v.tobytes() for k,v in [('output',raw),('path',states),('initial_gaussian',z)]):raise RuntimeError('Strict reload trajectory differs')
   row.update(reference_path=reference['raw_path'],all51_bitwise_equal=True);r['replays'].append(row);inc(c,'replay_outputs_verified')
  else:
   path=Path(out)/'raw'/f'{name}_P{unit["parent"]}_patch{unit["patch"]}_seed{seed}.npz';path.parent.mkdir(exist_ok=True)
   with path.open('xb') as f:np.savez_compressed(f,output=raw,path=states,initial_gaussian=z)
   row.update(raw_path=str(path.resolve()),raw_sha256=file_sha256(path));r['generated'].append(row);inc(c,'formal_outputs_saved')
  save_seconds=time.perf_counter()-start_save;attempt['status']='PASS';event('edge_sample',**identity,replay=replay,saved=c.get('formal_outputs_saved',0))
 finally:
  for k,v in local.items():c[k]=c.get(k,0)+v
  c['trajectory_save_seconds']=c.get('trajectory_save_seconds',0.)+save_seconds
  c['native_sampling_seconds']=c.get('native_sampling_seconds',0.)+native_seconds
  c['generation_wall_seconds']=c.get('generation_wall_seconds',0.)+time.perf_counter()-start
  attempt.update(wall_seconds=time.perf_counter()-start,trajectory_save_seconds=save_seconds);write(r,out)

def sample_all(r,out=OUT):
 if r['training']['status']!='PASS' or r['generated'] or 'sampling_phases' in r:raise RuntimeError('Require fixed endpoint, no automatic retry')
 source_guard(r);ev=r['protocol']['evaluation'];units=ev['units'];seeds=ev['seeds'];root=Path(out)
 asset=r['assets']['evaluation_inputs']
 if file_sha256(asset['path'])!=asset['sha256']:raise RuntimeError('Frozen contexts changed')
 with np.load(asset['path'],allow_pickle=False) as f:
  conditions={(u['parent'],u['patch']):f[f'parent{u["parent"]}_patch{u["patch"]}_constraints'].copy() for u in units}
 noises={seed:make_noise(seed).numpy().copy() for seed in seeds}
 path=root/'fixed_noise.npz'
 with path.open('xb') as f:np.savez_compressed(f,**{f'noise_seed{s}':v for s,v in noises.items()})
 r['assets']['fixed_noise']=record(path);r['noise_audit']=dict(private_CPU_generator=True,OT_calls=0,same_across_all_models_contexts=True,seeds=seeds)
 r['counts']['sampling']['noise_generation_calls']=len(seeds)
 block=dict(status='RUNNING',models_audit={});r['sampling_phases']={'pilot':block};write(r,out)
 for name in MODELS:
  model=None;entry=dict(status='RUNNING');block['models_audit'][name]=entry
  try:
   model,entry['load']=load_endpoint(r,name)
   for u in units:
    for seed in seeds:sample_one(r,out,model,name,u,seed,noises[seed],conditions[u['parent'],u['patch']])
   entry['state_after']=state_sha256(model)
   if entry['state_after']!=entry['load']['state_sha256']:raise RuntimeError('Sampling mutated endpoint')
   entry['status']='PASS'
  finally:model=None;release();write(r,out)
 if len(r['generated'])!=ev['expected_outputs']:raise RuntimeError('Formal matrix incomplete')
 # One prespecified fresh-load replay, not selected by performance.
 model=None
 try:
  spec=ev['registered_replay'];u=next(u for u in units if (u['parent'],u['patch'])==(spec['parent'],spec['patch']))
  ref=next(v for v in r['generated'] if all(v[k]==spec[k] for k in ('model','parent','patch','seed')))
  model,loaded=load_endpoint(r,spec['model'])
  sample_one(r,out,model,spec['model'],u,spec['seed'],noises[spec['seed']],conditions[u['parent'],u['patch']],reference=ref)
  if state_sha256(model)!=loaded['state_sha256']:raise RuntimeError('Replay mutated model')
  block['replay']=dict(status='PASS',load=loaded,all51_bitwise_equal=True);block['status']='PASS'
 finally:model=None;release();write(r,out)

