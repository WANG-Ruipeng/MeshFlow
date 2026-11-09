"""Read-only CPU final execution audit. Default --status never waits or dispatches.

Only this audit's receipts are written, below audits/final. --run is explicit.
A completed phase can resume only after every captured input is re-verified.
No model construction, forward, training, noise generation, OT solve or scoring.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import argparse,ast,csv,hashlib,json,math,sys,time,traceback
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/e/MeshFlow');OUT=ROOT/'experiments/r2_full_seat_learning_curve_v1';AUD=OUT/'audits/final';CODE=ROOT/'r2/pilots/full_seat_learning_curve_v1'
LABELS=('FULL_S0000','FULL_S1000','FULL_S3000','FULL_S6000','SMALL_C200','CREF_C200');STEPS=(0,1000,3000,6000)
HASHES={};HASH_CACHE={};SOURCE_HASH=None;REG=None;REG_HASH=None

def utc():return datetime.now(timezone.utc).isoformat()
def checked(path):
 p=Path(path).resolve()
 if not p.is_relative_to(AUD.resolve()):raise ValueError('Audit writes must stay under audits/final')
 return p

def digest(path,prefix=None):
 p=Path(path);st=p.stat();cache=(str(p),st.st_size,st.st_mtime_ns,prefix)
 if cache in HASH_CACHE:return HASH_CACHE[cache]
 h=hashlib.sha256();remaining=prefix
 with p.open('rb') as f:
  while remaining is None or remaining>0:
   a=f.read(4*1024*1024 if remaining is None else min(4*1024*1024,remaining))
   if not a:break
   h.update(a)
   if remaining is not None:remaining-=len(a)
 if remaining not in (None,0):raise ValueError('Captured source prefix shortened: '+str(p))
 value=h.hexdigest();HASH_CACHE[cache]=value;return value

def verify(path,expected):
 key=str(path)
 if '@captured_complete_bytes=' in key:
  file,count=key.rsplit('@captured_complete_bytes=',1);actual=digest(file,int(count))
 else:actual=digest(key)
 if actual!=expected:raise ValueError('Changed input: '+key)
 HASHES[key]=actual;return actual

def read(path):
 p=Path(path);b=p.read_bytes();HASHES[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b)
def rows(path):
 p=Path(path);b=p.read_bytes();end=b.rfind(b'\n')+1
 if end!=len(b):raise ValueError('Final ledger has incomplete last row: '+str(p))
 HASHES[str(p)+'@captured_complete_bytes='+str(end)]=hashlib.sha256(b).hexdigest();return [json.loads(x) for x in b.splitlines()]
def arrays(path):
 HASHES[str(path)]=digest(path)
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def ah(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str((a.dtype,a.shape)).encode()+a.tobytes()).hexdigest()
def ch(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def near(a,b):return math.isclose(float(a),float(b),rel_tol=2e-6,abs_tol=1e-8)
def write(path,value):
 path=checked(path);path.parent.mkdir(parents=True,exist_ok=True);blob=(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 if path.exists():
  if path.read_bytes()!=blob:raise ValueError('Existing final audit receipt differs: '+str(path))
  return
 temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 with temp.open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)

def status():
 missing=[];trainpath=OUT/'training/FULL_SEAT/status.json';s=json.loads(trainpath.read_text()) if trainpath.exists() else {}
 if s.get('status')!='COMPLETE' or s.get('local_step')!=6000 or s.get('global_step')!=12000:missing.append('completed training local6000/global12000')
 for step in STEPS:
  label=f'FULL_S{step:04d}'
  if not (OUT/'evaluation/bindings'/f'{label}.json').exists():missing.append('binding '+label)
  if step and not (OUT/'checkpoints'/f'FULL_SEAT_step{step}.pt').exists():missing.append('checkpoint '+str(step))
 for label in LABELS:
  for sub in ('score_completion','boundary_completion'):
   p=OUT/'evaluation'/sub/f'{label}.json'
   if not p.exists():missing.append(sub+'/'+label)
   else:
    q=json.loads(p.read_text())
    if q.get('status')!='COMPLETE' or len(q.get('receipts',[]))!=48:missing.append('all48 '+sub+'/'+label)
  for name in ('raw_index.json','post_index.json','complete.json'):
   if not (OUT/'evaluation/indices'/label/name).exists():missing.append(label+'/'+name)
 for path,expected in [(OUT/'training_analysis/latest.json','COMPLETE'),(OUT/'evaluation/comparison/latest.json','COMPLETE'),(OUT/'evaluation/gallery/manifest.json','COMPLETE'),(OUT/'evaluation/sections_gallery/manifest.json','COMPLETE_RENDERING_NOT_REVIEW')]:
  if not path.exists() or json.loads(path.read_text()).get('status')!=expected:missing.append(str(path.relative_to(OUT)))
 p=OUT/'orchestration/controller_state.json';controller=json.loads(p.read_text()) if p.exists() else {}
 if controller.get('status')!='COMPLETE_PENDING_ACTUAL_IMAGE_REVIEW_AND_REPORT' or controller.get('active_jobs'):missing.append('controller safely finished registered stages')
 return dict(status='PENDING' if missing else 'READY_FOR_EXPLICIT_CPU_AUDIT',training_status=s.get('status'),local_step=s.get('local_step'),missing=missing,model_calls=0,dispatches=0,waits=0,torch_imported='torch' in sys.modules)

def phase(name,fn):
 global HASHES
 p=AUD/'phases'/f'{name}.json'
 if p.exists():
  old=json.loads(p.read_text());assert old['registration_sha256']==REG_HASH and old['audit_source_sha256']==SOURCE_HASH
  for path,h in old['input_hashes'].items():verify(path,h)
  print(json.dumps(dict(phase=name,status='VERIFIED_CACHE',inputs=len(old['input_hashes']))),flush=True);return old['result']
 HASHES={};tick=time.perf_counter();value=fn();receipt=dict(status='PASS',phase=name,utc=utc(),seconds=time.perf_counter()-tick,registration_sha256=REG_HASH,audit_source_sha256=SOURCE_HASH,result=value,input_hashes=dict(HASHES))
 write(p,receipt);print(json.dumps(dict(phase=name,status='PASS',seconds=receipt['seconds'])),flush=True);return value

def contracts_and_prefix():
 for path,h in {**REG['training_code_hashes'],**REG['evaluation_code_hashes']}.items():verify(path,h)
 prior=read(OUT/'audits/interim_execution_review.json');assert prior['status']=='PASS_WITH_NONSCIENTIFIC_METADATA_CAVEAT' and not prior['blocking_issues'];verify(OUT/'audits/interim_execution_review.py',prior['audit_source_sha256'])
 for path,h in prior['readback_hashes'].items():verify(path,h)
 side=read(OUT/'audits/metadata_identity_sidecar.json');verify(OUT/'audits/metadata_identity_audit.csv',side['audit_csv_sha256']);assert side['affected_requests']==8 and side['affected_conditions']==2 and not side['stale_fields_present_in_per_output_columns']
 assert REG['budget']==dict(additional_updates=6000,checkpoints_global=[6000,7000,9000,12000],checkpoints_local=[0,1000,3000,6000],evaluation_requests_per_checkpoint=24,logical_POST=96,logical_RAW=96,new_D0_forward=88,new_rollouts=88,new_sampling_forward=4400,optimizer_update=6000,reference_cached_POST=48,reference_cached_RAW=48,train_backward=48000,train_forward=48000)
 assert REG['training_contract']['no_extra_forward'] is True and REG['training_contract']['no_new_loss'] is True
 assert len(REG['train_task_ids'])==47 and len(REG['requests'])==24 and len({q['condition_id'] for q in REG['requests']})==6
 return dict(interim_audit_sha256=digest(OUT/'audits/interim_execution_review.json'),reverified_interim_inputs=len(prior['readback_hashes']),reusable_mask_sample_prefix=8008,metadata_sidecar=side,all_registered_code_hashes_match=True)

def checkpoint_chain():
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 def tree(v):
  h=hashlib.sha256()
  def add(a):
   if isinstance(a,torch.Tensor):
    x=a.detach().cpu().contiguous();h.update(str(('tensor',str(x.dtype),tuple(x.shape))).encode());h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
   elif isinstance(a,dict):
    h.update(b'dict')
    for k in sorted(a,key=lambda k:(type(k).__name__,repr(k))):add(k);add(a[k])
   elif isinstance(a,(list,tuple)):
    h.update(str((type(a).__name__,len(a))).encode())
    for b in a:add(b)
   else:h.update(str((type(a).__name__,repr(a))).encode())
  add(v);return h.hexdigest()
 def state(d):
  h=hashlib.sha256()
  for k,v in sorted(d.items()):h.update(k.encode());h.update(str((tuple(v.shape),v.dtype)).encode());h.update(v.detach().contiguous().view(torch.uint8).numpy().tobytes())
  return h.hexdigest()
 initpaths=sorted((OUT/'training/FULL_SEAT/attempts').glob('*/initialization.json'));inits=[read(x) for x in initpaths];assert sorted(x['restored_global_step'] for x in inits)==[6000,7000,9000]
 plan=read(OUT/'plans/FULL_SEAT.json');out=[]
 for step in STEPS:
  pth=Path(REG['source_checkpoint']) if step==0 else OUT/'checkpoints'/f'FULL_SEAT_step{step}.pt';receipt=read(pth.with_suffix('.json')) if step else None
  verify(pth,receipt['sha256'] if step else REG['source_checkpoint_sha256']);p=torch.load(pth,map_location='cpu',weights_only=True,mmap=True)
  assert p['global_step']==6000+step and p['update_in_progress'] is False and not any(k in p for k in ('ema','ema_state','generator_ema'))
  gh=state(p['generator_state']);oh=tree(p['optimizer']);rh=tree(p['rng']);assert gh==p['generator_state_sha256'] and oh==p['optimizer_state_sha256'] and rh==p['rng_sha256']
  opt=p['optimizer'];assert len(opt['state'])==194==len(p['generator_state']);assert {int(x['step'].item()) for x in opt['state'].values()}=={6000+step};assert all(x.dtype==torch.float32 and x.device.type=='cpu' for x in p['generator_state'].values())
  group=opt['param_groups'][0];assert group['lr']==1e-5 and tuple(group['betas'])==(.9,.95) and group['weight_decay']==0 and group['eps']==1e-8
  if step:
   assert p['EMA'] is False and p['protocol']==REG and p['local_step']==step and p['plan_sha256']==digest(OUT/'plans/FULL_SEAT.json');stream=p['stream_state'];assert stream['batch_index']==step and stream['next_sample_index']==step*8 and stream['parent_exposure_counts']==dict(Counter(x['uid'] for x in plan['records'][:step*8]))
  else:assert p['protocol']['EMA'] is False and gh==REG['source_generator_sha256'] and oh==REG['source_optimizer_sha256'] and rh==REG['source_rng_sha256']
  if step<6000:
   init=next(x for x in inits if x['restored_global_step']==6000+step);assert init['loaded_generator_sha256']==gh and init['loaded_optimizer_sha256']==oh and init['loaded_rng_sha256']==rh and init['optimizer_restored'] and init['rng_restored'] and init['EMA'] is False
  binding=read(OUT/'evaluation/bindings'/f'FULL_S{step:04d}.json');verify(binding['path'],binding['sha256']);export=torch.load(binding['path'],map_location='cpu',weights_only=True,mmap=True)
  assert state(export['generator_state'])==gh==binding['generator_state_sha256'] and binding['global_step']==6000+step and binding['local_step']==step and binding['EMA'] is False and not any(k in export for k in ('optimizer','rng','ema'))
  out.append(dict(local_step=step,global_step=6000+step,checkpoint_path=str(pth),checkpoint_sha256=digest(pth),generator_state_sha256=gh,optimizer_state_sha256=oh,rng_sha256=rh,inference_path=binding['path'],inference_sha256=binding['sha256'],optimizer_tensor_steps=6000+step,optimizer_tensors=194,EMA=False,CPU_tensor_hashes_recomputed=True));del p,export,opt
 assert not torch.cuda.is_initialized();return dict(chain=out,CUDA_initialized=False,model_constructions=0)

def training():
 plan=read(OUT/'plans/FULL_SEAT.json');rec=plan['records'];assert len(rec)==48000 and [x['sample_index'] for x in rec]==list(range(48000))
 logs=rows(OUT/'training/FULL_SEAT/training_log.jsonl');ex=rows(OUT/'training/FULL_SEAT/exposure.jsonl');di=rows(OUT/'training/FULL_SEAT/interface_diagnostics.jsonl');calls=rows(OUT/'training/FULL_SEAT/calls.jsonl')
 assert [x['local_step'] for x in logs]==list(range(1,6001));assert [x['sample_index'] for x in ex]==list(range(48000));assert [x['sample_index'] for x in di]==list(range(48000))
 begun={};done={};counts=Counter();types=Counter();attempts=Counter()
 for c in calls:
  target=begun if c['event']=='CALL_BEGIN' else done;assert c['event'] in ('CALL_BEGIN','CALL_DONE') and c['call_id'] not in target;target[c['call_id']]=c
  assert c['context']['global_step']==6000+c['context']['local_step']
  if c['event']=='CALL_DONE':counts[c['kind']]+=1;attempts[c['attempt_id']]+=1
 assert len(calls)==204000 and set(begun)==set(done) and counts==Counter(train_forward=48000,train_backward=48000,optimizer_update=6000)
 assert sorted(begun)==list(range(1,102001))
 for cid,b in begun.items():
  d=done[cid];ctx=b['context'];assert ctx==d['context'] and b['kind']==d['kind'] and b['time_unix']<=d['time_unix'];types[ctx['local_step'],b['kind']]+=1
  if b['kind']!='optimizer_update':
   e=ex[ctx['sample_index']];assert e['local_step']==ctx['local_step'] and e['input_bytehash']==ctx['input_bytehash'] and e['label_bytehash']==ctx['label_bytehash'] and ctx['microbatch']==ctx['sample_index']%8
 for step in range(1,6001):assert [types[step,k] for k in ('train_forward','train_backward','optimizer_update')]==[8,8,1]
 maxerr=0.;bins=REG['diagnostic_time_bin_edges']
 for i,(e,d,p) in enumerate(zip(ex,di,rec)):
  for k in ('uid','task_id','N','K','alpha','t','global_step','condition_type'):assert e[k]==p[k]
  assert e['local_step']==p['step']==i//8+1 and e['K']>=2 and e['condition_type']=='SEAT_FULL' and e['free_coordinate_count']==9*(e['N']-e['K'])
  assert d['interface_mask_sha256']==e['interface_mask_sha256'] and d['interface_scalar_count']==3*e['interface_corner_count'];assert d['interface_scalar_count']+d['other_scalar_count']==e['free_coordinate_count'] and d['t']==e['t']
  assert d['time_bin']==min(4,int(np.searchsorted(bins,d['t'],side='right')-1));factor=float(np.float32(1)-np.float32(d['t']))**2;assert d['endpoint_error_factor']==factor
  for prefix in ('interface','other'):
   s=d[prefix+'_velocity_sse'];n=d[prefix+'_scalar_count'];assert n>0 and near(d[prefix+'_velocity_mse'],s/n) and near(d[prefix+'_endpoint_scaled_sse'],factor*s) and near(d[prefix+'_endpoint_scaled_mse'],factor*s/n)
  assert near(d['all_free_velocity_sse'],d['interface_velocity_sse']+d['other_velocity_sse']);value=d['all_free_velocity_sse']/d['batch_free_coordinate_denominator'];assert near(value,d['FM_contribution']);maxerr=max(maxerr,abs(value-d['FM_contribution']))
 for j,log in enumerate(logs):
  es=ex[8*j:8*j+8];ds=di[8*j:8*j+8];assert len({x['uid'] for x in es})==8 and sum(x['alpha']==1. for x in es)==4 and all(.9<=x['alpha']<=1.1 for x in es)
  assert sum(x['free_coordinate_count'] for x in es)==log['free_coordinate_denominator'];assert log['FM']==log['total_loss']==sum(x['FM_contribution'] for x in ds)
 assert max(Counter(x['uid'] for x in ex).values())-min(Counter(x['uid'] for x in ex).values())<=1
 # Saved OT labels only: no target rebuilding, no random Gaussian/time draw.
 source=read(REG['source_data']);manifest=read(REG['interface_manifest_path']);masks={};cyclic=np.array([[0,1,2],[1,2,0],[2,0,1]],np.int64)
 for tid in REG['train_task_ids']:
  row=manifest['tasks'][tid];verify(row['mask_path'],row['mask_sha256']);masks[tid]=arrays(row['mask_path'])[row['mask_key']]
 prior=read(OUT/'audits/interim_execution_review.json');assert prior['training']['interface_transport_samples']==8008;ph=digest(OUT/'plans/FULL_SEAT.json');fresh=0
 for b in range(6000):
  path=OUT/'ot/FULL_SEAT'/f'{b:05d}.npz';receipt=read(path.with_suffix('.json'));verify(path,receipt['sha256']);assert receipt['identity']['plan_sha256']==ph and receipt['identity']['batch_index']==b and receipt['actual_OT']==8
  if b<=1000:
   verify(path,prior['readback_hashes'][str(path)]);continue
  maps=arrays(path)
  for row in rec[b*8:b*8+8]:
   i=row['sample_index'];n=row['N'];k=row['K'];task=source['tasks'][row['task_id']];known=np.asarray(task['source_face_ids']);free=np.asarray(task['free_source_ids']);rng=np.random.RandomState(row['permutation_seed']);orderfree=rng.permutation(n-k);corners=cyclic[rng.randint(0,3,size=n)];order=rng.permutation(n);face=maps[str(i)+'_face'];corner=maps[str(i)+'_corner'];inv=np.argsort(face);ic=np.argsort(corner,axis=1)[inv]
   assert np.array_equal(np.sort(face),np.arange(n-k)) and np.all(np.sort(corner,axis=1)==np.arange(3))
   ids=np.r_[known,free[orderfree][inv]][order];cc=np.concatenate((corners[:k],corners[k:][inv][np.arange(n-k)[:,None],ic]))[order];mask=masks[row['task_id']][ids[:,None],cc];knownmask=(np.arange(n)<k)[order]
   assert int(mask.sum())==int(masks[row['task_id']].sum()) and not mask[knownmask].any() and ah(mask)==ex[i]['interface_mask_sha256'];assert str(maps[str(i)+'_key'].item())==ex[i]['ot_cache_key'];fresh+=1
 assert fresh+8008==48000
 return dict(updates=6000,samples=48000,calls=dict(counts),failed_calls=0,pending_calls=0,continuous_sample_and_global_step=True,diagnostic_rows=48000,max_FM_partition_roundoff=maxerr,interface_mask_samples=48000,mask_verified_prefix_reused_after_hash_check=8008,mask_independently_reconstructed_this_final_run=fresh,parent_counts=dict(Counter(x['uid'] for x in ex)),units='Scalar-coordinate velocity SSE/count; endpoint proxy=(1-t)^2 velocity error, not rollout/nearest-surface distance',source_RNG='Live source restored; sample RNG is frozen per-sample local RandomState, so unchanged global RNG hash is expected')

def evaluation():
 queries={q['condition_id']:q for q in REG['queries']};requests={q['request_id']:q for q in REG['requests']};allids=set(requests);summary=[];quality_core={};bound_core=None;newcalls=Counter();score_rows=0;boundary_rows=0;all_times={}
 for label in LABELS:
  rawindex=read(OUT/'evaluation/indices'/label/'raw_index.json');postindex=read(OUT/'evaluation/indices'/label/'post_index.json');assert {q['request_id'] for q in rawindex['rows']}==allids=={q['request_id'] for q in postindex['rows']}
  binding=read(OUT/'evaluation/bindings'/f'{label}.json') if label.startswith('FULL_') else REG['models'][label]
  raws={x['request_id']:x for x in rawindex['rows']};posts={x['request_id']:x for x in postindex['rows']};made=0;rollends=[]
  for rid,q in requests.items():
   inp=arrays(q['input_path']);src=arrays(q['condition_source']);verify(q['input_path'],q['input_sha256']);verify(q['condition_source'],q['condition_source_sha256']);assert same(inp['C'],src['C']) and int(inp['K'])==len(inp['C'])==q['K'] and int(inp['N'])==q['N']
   qr=queries[q['condition_id']];verify(qr['path'],qr['sha256']);qu=arrays(qr['path']);pts=(inp['C'][qu['C_point_faces']].astype(np.float64)*qu['C_barycentric'][...,None]).sum(axis=1);assert np.allclose(pts,qu['C_points'],rtol=0,atol=2e-6) and len(pts)==2048
   rr=raws[rid];pr=posts[rid];assert rr['checkpoint_sha256']==binding['sha256'];verify(rr['raw_path'],rr['raw_sha256']);verify(pr['post_path'],pr['post_sha256']);raw=arrays(rr['raw_path']);post=arrays(pr['post_path']);k=q['K'];n=q['N']
   assert same(raw['C'],inp['C']) and same(raw['raw'].reshape(n,3,3)[:k],inp['C']) and same(raw['z'],inp['z']);assert same(post['C'],src['C_original']) and same(post['output'][:k],src['C_original'])
   assert np.array_equal(post['C_original_source_face_ids'],src['C_original_source_face_ids']) and np.array_equal(post['C_original_source_vertex_ids'],src['C_original_source_vertex_ids']) and all(rr['contracts'].values()) and pr['source_restoration']['status']=='PASS'
   for key in ('input_sha256','Gaussian_sha256','C_sha256','condition_source_sha256'):assert rr[key]==q[key]
   if rr['generation_status']=='NEW_WORLD_FRAME_EULER50':
    calls=rows(OUT/'evaluation/raw'/label/rid/'calls.jsonl');cc=Counter((x['event'],x.get('kind')) for x in calls);assert cc['RESERVE','rollout_attempt']==1 and cc['RESERVE','sampling_forward']==50 and cc['RETURN','sampling_forward']==50
    ends=[x for x in calls if x['event']=='ROLLOUT_COMPLETE'];assert len(ends)==1 and ends[0]['nfe']==50;rollends.append(ends[0]['time_unix']);assert len(calls)==103
    reserve=[x for x in calls if x['event']=='RESERVE' and x.get('kind')=='sampling_forward'];assert [x['context']['euler_step'] for x in reserve]==list(range(50));assert all(x['context']['input_sha256']==q['input_sha256'] and x['context']['checkpoint_sha256']==binding['sha256'] for x in reserve)
    proposal=read(pr['proposal_receipt']);assert proposal['model_forwards']==1 and proposal['checkpoint_sha256']==REG['d0_checkpoint']['sha256'] and pr['new_D0_F']==1 and pr['GEO_calls']==1;made+=1;newcalls.update(rollouts=1,sampling_forward=50,D0_forward=1)
   else:assert rr['generation_status']=='VERIFIED_HISTORICAL_RAW_REUSE' and pr['new_D0_F']==0
  expect=16 if label=='FULL_S0000' else 24 if label.startswith('FULL_') else 0;assert made==expect
  if made:
   started=read(OUT/'orchestration/jobs'/('post_'+label)/'started.json');assert max(rollends)<datetime.fromisoformat(started['utc']).timestamp()
  scores=read(OUT/'evaluation/score_completion'/f'{label}.json');boundary=read(OUT/'evaluation/boundary_completion'/f'{label}.json');assert scores['status']==boundary['status']=='COMPLETE' and len(scores['receipts'])==boundary['cells']==48
  assert {(x['request_id'],x['stage']) for x in scores['receipts']}=={(rid,s) for rid in allids for s in ('RAW','POST')}
  assert {(x['request_id'],x['stage']) for x in boundary['receipts']}=={(rid,s) for rid in allids for s in ('RAW','POST')}
  for path,h in scores['definition_hashes'].items():verify(path,h)
  for ref in scores['receipts']:
   verify(ref['path'],ref['sha256']);sc=read(ref['path']);idn=sc['identity'];q=requests[ref['request_id']];qr=queries[q['condition_id']];rr=raws[q['request_id']];pr=posts[q['request_id']]
   assert idn['input_sha256']==q['input_sha256'] and idn['query_sha256']==qr['sha256'] and idn['condition_source_sha256']==q['condition_source_sha256'] and idn['mesh_sha256']==(rr['raw_sha256'] if ref['stage']=='RAW' else pr['post_sha256'])
   assert sc['K']==q['K'] and sc['N']==q['N'] and near(sc['ratio'],q['K']/q['N']) and sc['C_preserved_exact'] and sc['C_query_count']==2048
   for path,h in sc['files'].items():verify(path,h)
   edgepath=Path(ref['path']).parent/'valid_edges.csv'
   with edgepath.open(encoding='utf-8-sig') as f:edge_rows=list(csv.DictReader(f))
   cf=[x for x in edge_rows if x['is_C_F']=='True'];valid=[x for x in cf if x['valid_edge']=='True'];top=sc['topology']
   assert len(cf)==top['C_F_shared_edge_count'] and len(valid)==top['C_F_valid_edge_count']
   assert int(bool(valid))==top['C_F_has_valid_edge'] and sum(x['winding_opposite']=='True' for x in valid)==top['C_F_valid_opposite_winding_edge_count']
   assert near(sum(float(x['length_model']) for x in valid),top['C_F_valid_edge_total_length'])
   for path,h in idn['definition_hashes'].items():
    verify(path,h)
    if '/evaluation/' in path and path.endswith('.py') and 'MeshFlow-Control/src' in path:
     if path in quality_core:assert quality_core[path]==h
     quality_core[path]=h
   score_rows+=1
  for ref in boundary['receipts']:
   assert ref['status']=='COMPLETE';idn=ref['identity'];q=requests[ref['request_id']];assert idn['source_sha256']==q['condition_source_sha256'] and idn['registration_sha256']==REG_HASH
   if bound_core is None:bound_core=(idn['config'],idn['definition_hashes'])
   assert bound_core==(idn['config'],idn['definition_hashes'])
   for path,h in {**idn['definition_hashes'],**ref['files']}.items():verify(path,h)
   verify(idn['geometry_path'],idn['geometry_sha256']);tri=arrays(idn['geometry_path']);src=arrays(q['condition_source']);assert same(tri['output'][:q['K']],src['C_original']) and same(tri['vertices'][tri['faces']],tri['output']);boundary_rows+=1
  table=OUT/'evaluation/tables'/label/'per_output.csv';HASHES[str(table)]=digest(table)
  with table.open(encoding='utf-8-sig') as f:output=list(csv.DictReader(f))
  assert len(output)==48 and {(x['request_id'],x['stage']) for x in output}=={(rid,s) for rid in allids for s in ('RAW','POST')}
  for x in output:q=requests[x['request_id']];assert int(x['K'])==q['K'] and int(x['N'])==q['N'] and near(x['ratio'],q['K']/q['N'])
  assert not set(('K_over_N','area_C_over_M','area_C_over_region','source_face_ids','task_id','npz'))&set(output[0])
  summary.append(dict(label=label,POST_valid_edges=sum(int(x['C_F_valid_edge_count']) for x in output if x['stage']=='POST'),POST_valid_edge_outputs=sum(int(x['C_F_has_valid_edge']) for x in output if x['stage']=='POST'),POST_all_shared_edges=sum(int(x['C_F_shared_edge_count']) for x in output if x['stage']=='POST'),POST_valid_edge_length=sum(float(x['C_F_valid_edge_total_length']) for x in output if x['stage']=='POST'),legal_field_contract='C_F_valid_*; C_F_shared_edge_count includes rejected/nonmanifold edges',RAW=24,POST=24,new_rollouts=made,reused_rollouts=24-made,score_rows=48,boundary_rows=48,score_workers=scores.get('workers'),boundary_workers=boundary.get('workers'),all_RAW_before_single_D0=True))
 assert newcalls==Counter(rollouts=88,sampling_forward=4400,D0_forward=88) and score_rows==boundary_rows==288
 events=rows(OUT/'execution/worker_0_of_1_events.jsonl');cc=Counter(x['event'] for x in events);assert cc==Counter(D0_FORWARD_RESERVED=88,D0_FORWARD_RETURNED=88)
 assert bound_core[0]['boundary_step_over_L']==.002 and bound_core[0]['distance_thresholds_over_L']==[.001,.005,.01,.02]
 return dict(labels=summary,new_calls=dict(newcalls),D0_event_counts=dict(cc),score_rows=score_rows,boundary_rows=boundary_rows,fixed_request_count=24,fixed_condition_count=6,fixed_noise_candidates=[0,1,2,3],unchanged_original_quality_definition_hashes=quality_core,unchanged_boundary_definition=bound_core,no_new_metrics=True,no_new_queries=True,no_new_noise=True,metadata_sidecar=read(OUT/'audits/metadata_identity_sidecar.json'))

def recorded_call_overlap():
 calls=rows(OUT/'training/FULL_SEAT/calls.jsonl');starts={x['call_id']:x for x in calls if x['event']=='CALL_BEGIN' and x['context']['local_step']<=1000};train=[]
 for x in calls:
  if x['event']=='CALL_DONE' and x['call_id'] in starts:
   b=starts[x['call_id']];train.append((b['time_unix'],x['time_unix'],x['call_id'],x['kind'],x['context']['local_step']))
 generation=[]
 for path in (OUT/'evaluation/raw/FULL_S0000').glob('*/calls.jsonl'):
  pending=None
  for x in rows(path):
   if x.get('kind')!='sampling_forward':continue
   if x['event']=='RESERVE':assert pending is None;pending=x
   elif x['event']=='RETURN':
    assert pending is not None;generation.append((pending['time_unix'],x['time_unix'],path.parent.name,pending['ordinal']));pending=None
  assert pending is None
 train.sort();generation.sort();i=j=0;seconds=0.;pairs=0;seen_train={};seen_gen=set();seen_requests=set()
 while i<len(generation) and j<len(train):
  a,b=generation[i],train[j];length=min(a[1],b[1])-max(a[0],b[0])
  if length>0:
   seconds+=length;pairs+=1;seen_train[b[2]]=b;seen_gen.add((a[2],a[3]));seen_requests.add(a[2])
  if a[1]<b[1]:i+=1
  else:j+=1
 return dict(definition='Intersect existing host CALL_BEGIN/DONE and sampling RESERVE/RETURN timestamps; these are actual recorded call envelopes, not a CUDA kernel concurrency trace',overlap_seconds_sum=seconds,overlapping_interval_pairs=pairs,overlapping_generator_forward_calls=len(seen_gen),overlapping_training_calls_by_kind=dict(Counter(v[3] for v in seen_train.values())),overlapping_training_update_min=min(v[4] for v in seen_train.values()) if seen_train else None,overlapping_training_update_max=max(v[4] for v in seen_train.values()) if seen_train else None,overlapping_step0_requests=sorted(seen_requests),step0_new_forward_total=len(generation),training_scope='Initial local1..1000 segment only; controller later stage serialization verified separately',new_model_calls=0)
def resources_and_outputs():
 jobs=[]
 for folder in sorted((OUT/'orchestration/jobs').glob('*')):
  r=read(folder/'request.json');s=read(folder/'started.json');d=read(folder/'done.json');assert d['returncode']==0 and d['status']=='COMPLETE';verify(r['module_source'],r['module_sha256']);assert r['registration_sha256']==REG_HASH and d['request_sha256']==digest(folder/'request.json')
  jobs.append(dict(job=r['job_id'],kind=r['kind'],start=datetime.fromisoformat(s['utc']).timestamp(),end=datetime.fromisoformat(d['finished']).timestamp(),seconds=d['elapsed_seconds'],command=r['command']))
 gpu=[x for x in jobs if x['kind']=='GPU'];overlaps=[]
 for i,a in enumerate(gpu):
  for b in gpu[i+1:]:
   overlap=min(a['end'],b['end'])-max(a['start'],b['start'])
   if overlap>0:overlaps.append([a['job'],b['job'],overlap])
 assert not overlaps
 cpu=[x for x in jobs if x['kind']=='CPU_PIPELINE'];maxcpu=max(sum(x['start']<=y['start']<x['end'] for x in cpu) for y in cpu);assert maxcpu<=2
 ledger=rows(OUT/'ledger.jsonl');dispatch=[x for x in ledger if x['event']=='ORCHESTRATOR_JOB_DISPATCHED'];assert {x['job_id'] for x in dispatch}=={x['job'] for x in jobs}
 for x in dispatch:read(x['resources_path'])
 decisions=[x for x in ledger if x['event']=='ORCHESTRATOR_GPU_RESOURCE_DECISION']
 for x in decisions:
  read(x['resources_path'])
  if x['ready']:assert x['free_MiB']>=1.2*x['incremental_estimate_MiB']+x['headroom_MiB'] and x['GPU_workers']==1
 pre=[x for x in ledger if x['event']=='ADDITIONAL_GPU_DISPATCH'];assert len(pre)==1 and pre[0]['training_workers']==pre[0]['generator_workers']==1
 finalstates=[read(p) for p in sorted((OUT/'training/FULL_SEAT/attempts').glob('*/final_status.json'))];assert len(finalstates)==3 and sorted(x['local_step'] for x in finalstates)==[1000,3000,6000]
 for x in finalstates:assert x['status'] in ('SAFE_SEGMENT','COMPLETE') and not x['calls']['failed'] and not x['calls']['pending_call_ids']
 # Source plus first segment overlap is disclosed separately from controller serialization.
 externalgen=[]
 for p in (OUT/'evaluation/raw/FULL_S0000').glob('*/calls.jsonl'):
  cs=rows(p);beg=[x['time_unix'] for x in cs if x['event']=='RESERVE' and x.get('kind')=='rollout_attempt'];end=[x['time_unix'] for x in cs if x['event']=='ROLLOUT_COMPLETE'];externalgen.extend(zip(beg,end))
 trainlogs=rows(OUT/'training/FULL_SEAT/training_log.jsonl');firstinterval=(min(x['time_unix']-x['update_seconds'] for x in trainlogs[:1000]),max(x['time_unix'] for x in trainlogs[:1000]));externalspan=(min(x[0] for x in externalgen),max(x[1] for x in externalgen));overlap=max(0,min(firstinterval[1],externalspan[1])-max(firstinterval[0],externalspan[0]))
 outputs={}
 for rel,wanted in [('training_analysis/latest.json','COMPLETE'),('evaluation/comparison/latest.json','COMPLETE'),('evaluation/gallery/manifest.json','COMPLETE'),('evaluation/sections_gallery/manifest.json','COMPLETE_RENDERING_NOT_REVIEW')]:
  v=read(OUT/rel);assert v['status']==wanted
  if 'path' in v:verify(v['path'],v['sha256'])
  if 'images' in v:
   assert len(v['images'])==24
   for image in v['images']:verify(image['path'],image['sha256'])
  outputs[rel]=dict(status=v['status'],sha256=digest(OUT/rel),images=len(v.get('images',[])))
 return dict(controller_GPU_overlaps=overlaps,max_CPU_pipelines=maxcpu,phase_records=jobs,training_segments=[dict(local_step=x['local_step'],seconds=x['wall_seconds'],memory=x['resources'],torch_peak_reserved_bytes=x['peak_reserved_bytes']) for x in finalstates],precontroller_parallel_dispatch=pre[0],precontroller_train_sampling_envelope_overlap_seconds=overlap,recorded_call_overlap=recorded_call_overlap(),process_VRAM='NVIDIA WDDM process VRAM unavailable; global device occupancy and torch allocator scopes remain distinct',outputs=outputs,actual_visual_review='NOT_CREATED_BY_THIS_AUDIT; root reviewer records are separate',total_new_audit_model_calls=0)

def run():
 global REG,REG_HASH,SOURCE_HASH
 readiness=status()
 if readiness['status']=='PENDING':return readiness
 REG=read(OUT/'registration.json');REG_HASH=digest(OUT/'registration.json');SOURCE_HASH=digest(__file__)
 binding=dict(version='full_seat_final_execution_audit_v1',registration_sha256=REG_HASH,audit_source_sha256=SOURCE_HASH,expected_updates=6000,expected_train_forward=48000,expected_train_backward=48000,expected_rollouts=88,expected_sampling_forward=4400,expected_D0_forward=88,expected_score_rows=288,expected_boundary_rows=288)
 write(AUD/'binding.json',binding);parts={}
 for name,fn in [('contracts_and_prefix',contracts_and_prefix),('checkpoint_chain',checkpoint_chain),('training',training),('evaluation',evaluation),('resources_and_outputs',resources_and_outputs)]:parts[name]=phase(name,fn)
 value=dict(status='PASS_WITH_INHERITED_METADATA_CAVEAT_AND_DISCLOSED_PARALLEL_WINDOW',binding=binding,phases={name:dict(path=str(AUD/'phases'/f'{name}.json'),sha256=digest(AUD/'phases'/f'{name}.json')) for name in parts},checks=dict(training_updates=6000,train_forward=48000,train_backward=48000,new_rollouts=88,new_sampling_forward=4400,new_D0_forward=88,score_rows=288,boundary_rows=288,pending_or_failed_calls=0,fixed_requests=24,fixed_conditions=6,checkpoint_local_steps=list(STEPS)),blocking_issues=[],metadata_sidecar=str(OUT/'audits/metadata_identity_sidecar.json'),audit_added_calls=dict(model_construction=0,forward=0,backward=0,optimizer=0,new_Gaussian=0,OT=0,quality_scan=0),limitations=['Execution audit does not establish geometry improvement, generalization or human adoption.','Endpoint training diagnostics are teacher-forced algebraic proxies, not rollout reconstruction quality.','Eight inherited request descriptors still refer to HALF; source arrays, current K/N/ratio and metric numerics are verified separately.'])
 write(AUD/'final_execution_audit.json',value);return value

def main():
 p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group();mode.add_argument('--status',action='store_true');mode.add_argument('--run',action='store_true');a=p.parse_args()
 try:value=run() if a.run else status();print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)
 except Exception as exc:
  if a.run:
   AUD.mkdir(parents=True,exist_ok=True);write(AUD/('failure_'+str(time.time_ns())+'.json'),dict(status='AUDIT_STOPPED',error=repr(exc),traceback=traceback.format_exc(),utc=utc(),scientific_mutations=0,model_calls=0))
  raise
if __name__=='__main__':main()