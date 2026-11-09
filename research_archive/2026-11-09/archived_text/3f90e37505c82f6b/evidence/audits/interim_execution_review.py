"""Independent CPU-only interim readback; no models, OT solves or new Gaussian draws."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import ast,csv,hashlib,io,json,math,sys,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/e/MeshFlow');OUT=ROOT/'experiments/r2_full_seat_learning_curve_v1';CODE=ROOT/'r2/pilots/full_seat_learning_curve_v1';AUD=OUT/'audits'
HASHES={};CAP=1000

def read(path):
 p=Path(path);blob=p.read_bytes();HASHES[str(p)]=hashlib.sha256(blob).hexdigest();return json.loads(blob)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
 return h.hexdigest()
def stream(path):
 p=Path(path);b=p.read_bytes();end=b.rfind(b'\n')+1;b=b[:end];HASHES[str(p)+'@captured_complete_bytes='+str(end)]=hashlib.sha256(b).hexdigest();return [json.loads(x) for x in b.splitlines()]
def arrays(path):
 p=Path(path);HASHES[str(p)]=sha(p)
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def ah(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str((a.dtype,a.shape)).encode()+a.tobytes()).hexdigest()
def ch(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def area(tri):
 a=np.asarray(tri,dtype=np.float64).reshape(-1,3,3);return float(np.linalg.norm(np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]),axis=1).sum()*.5)
def close(a,b):return math.isclose(float(a),float(b),rel_tol=2e-6,abs_tol=1e-8)
def write_json(p,d):
 if Path(p).exists():
  assert json.loads(Path(p).read_text(encoding='utf-8'))==d;return
 with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def write_csv(p,rows):
 f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);blob=f.getvalue().encode('utf-8-sig')
 if Path(p).exists():assert Path(p).read_bytes()==blob
 else:Path(p).write_bytes(blob)

def metadata(r):
 conditions={x['parent_id']:x for x in r['conditions']};queries={x['condition_id']:x for x in r['queries']};rows=[];counts=Counter();identity_checks=0
 for q in r['requests']:
  c=conditions[q['parent_id']];inp=arrays(q['input_path']);src=arrays(q['condition_source']);parent=arrays(c['parent_npz']);query=arrays(queries[q['condition_id']]['path']);n,k=int(inp['N']),int(inp['K']);ids=src['source_face_ids'];free=src['free_source_ids'];actual=inp['C'];full=parent['full_target'].reshape(n,3,3)
  assert k==len(actual)==len(ids)==int(src['K'])==q['K'] and n==int(src['N'])==q['N']
  assert same(actual,src['C']) and same(actual,full[ids]) and np.array_equal(np.sort(np.r_[ids,free]),np.arange(n))
  assert q['C_sha256']==ch(actual) and q['input_sha256']==sha(q['input_path']) and q['condition_source_sha256']==sha(q['condition_source'])
  assert list(ids)==c['source_face_ids'] and list(src['C_original_source_face_ids'])==c['original_source_face_ids']
  point=(actual[query['C_point_faces']].astype(np.float64)*query['C_barycentric'][...,None]).sum(axis=1)
  assert np.allclose(point,query['C_points'],rtol=0,atol=2e-6) and len(point)==2048
  a=area(actual);m=area(full);ar=area(full[src['region_source_face_ids']]);wrong=[]
  for key,value in [('K_over_N',k/n),('area_C_over_M',a/m),('area_C_over_region',a/ar)]:
   if key in q and not close(q[key],value):wrong.append(key)
  if 'source_face_ids' in q and q.get('source_face_ids')!=ids.tolist():wrong.append('source_face_ids')
  if 'free_source_ids' in q and q.get('free_source_ids')!=free.tolist():wrong.append('free_source_ids')
  if 'task_id' in q and q.get('task_id')!=q['parent_id']+'__SEAT_FULL':wrong.append('task_id')
  if 'npz' in q and q.get('npz')!=q['condition_source']:wrong.append('npz')
  if 'original_source_face_ids' in q and q.get('original_source_face_ids')!=src['C_original_source_face_ids'].tolist():wrong.append('original_source_face_ids')
  counts.update(wrong)
  for label in ('FULL_S0000','FULL_S1000'):
   rr=read(OUT/'evaluation/raw'/label/q['request_id']/'receipt.json');pr=read(OUT/'evaluation/post'/label/q['request_id']/'receipt.json');raw=arrays(rr['raw_path']);post=arrays(pr['post_path'])
   assert same(raw['C'],actual) and same(raw['raw'].reshape(n,3,3)[:k],actual) and same(raw['z'],inp['z'])
   assert same(post['C'],src['C_original']) and same(post['output'][:k],src['C_original'])
   assert np.array_equal(post['C_original_source_face_ids'],src['C_original_source_face_ids']) and np.array_equal(post['C_original_source_vertex_ids'],src['C_original_source_vertex_ids'])
   assert all(rr['contracts'].values()) and pr['source_restoration']['status']=='PASS'
   identity_checks+=1
  rows.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],candidate=q['candidate'],current_condition_source=q['condition_source'],current_input_path=q['input_path'],actual_K=k,actual_N=n,actual_K_over_N=k/n,actual_C_area=a,actual_M_area=m,actual_region_area=ar,actual_area_C_over_M=a/m,actual_area_C_over_region=a/ar,current_canonical_source_face_ids=json.dumps(ids.tolist()),current_original_source_face_ids=json.dumps(src['C_original_source_face_ids'].tolist()),current_free_source_ids=json.dumps(free.tolist()),inherited_K_over_N=q.get('K_over_N'),inherited_ratio=q['ratio'],inherited_area_C_over_M=q.get('area_C_over_M'),inherited_area_C_over_region=q.get('area_C_over_region'),inherited_task_id=q.get('task_id'),inherited_npz=q.get('npz'),inherited_source_face_ids=json.dumps(q.get('source_face_ids')),inherited_original_source_face_ids=json.dumps(q.get('original_source_face_ids')),inherited_free_source_ids=json.dumps(q.get('free_source_ids')),stale_fields='|'.join(wrong),actual_model_input_and_POST_source_identity='PASS',current_full_query_reconstruction='PASS'))
 columns=None;table_rows=0
 for label in ('FULL_S0000','FULL_S1000','SMALL_C200','CREF_C200'):
  p=OUT/'evaluation/tables'/label/'per_output.csv';HASHES[str(p)]=sha(p)
  with p.open(encoding='utf-8-sig') as f:data=list(csv.DictReader(f));columns=list(data[0])
  assert len(data)==48
  for row in data:
   q=next(q for q in r['requests'] if q['request_id']==row['request_id']);assert int(row['K'])==q['K'] and int(row['N'])==q['N'] and close(row['ratio'],q['K']/q['N']);table_rows+=1
 assert not set(counts)&set(columns)
 write_csv(AUD/'metadata_identity_audit.csv',rows)
 sidecar=dict(status='PASS_SCIENTIFIC_INPUT_WITH_STALE_INHERITED_DESCRIPTORS',requests=24,conditions=6,affected_requests=sum(bool(x['stale_fields']) for x in rows),affected_conditions=len({x['parent_id'] for x in rows if x['stale_fields']}),field_counts=dict(counts),read_current_fields=dict(inference_geometry='request.input_path NPZ C,z,N,K; verified bytewise',source_identity='request.condition_source NPZ source_face_ids,C_original_source_face_ids,C_original_source_vertex_ids',parent_reference='registration.conditions[parent_id].parent_npz',ratios='request.ratio or actual_K/actual_N; use this sidecar for actual area ratios',queries='registration.queries by condition_id; 2048 saved FULL surface points reconstructed from barycentrics'),stale_field_scope='Inherited HALF provenance remains read-only; do not interpret as current FULL descriptors.',scientific_input_checks=identity_checks,per_output_rows_verified=table_rows,stale_fields_present_in_per_output_columns=sorted(set(counts)&set(columns)),current_scorer_numeric_effect='NONE_DETECTED: actual K/C/input arrays and saved FULL queries verified; no rescoring',model_calls=0,new_noise_draws=0,new_OT=0,source_registration_sha256=sha(OUT/'registration.json'),audit_csv_sha256=sha(AUD/'metadata_identity_audit.csv'))
 write_json(AUD/'metadata_identity_sidecar.json',sidecar);return sidecar

def training(r):
 plan=read(OUT/'plans/FULL_SEAT.json');plan_hash=sha(OUT/'plans/FULL_SEAT.json');rec=plan['records'];assert len(rec)==48000 and len(r['train_task_ids'])==47
 logs=[x for x in stream(OUT/'training/FULL_SEAT/training_log.jsonl') if x['local_step']<=CAP];ex=[x for x in stream(OUT/'training/FULL_SEAT/exposure.jsonl') if x['local_step']<=CAP+1];di=[x for x in stream(OUT/'training/FULL_SEAT/interface_diagnostics.jsonl') if x['local_step']<=CAP];calls=[x for x in stream(OUT/'training/FULL_SEAT/calls.jsonl') if x['context']['local_step']<=CAP+1]
 assert [x['local_step'] for x in logs]==list(range(1,CAP+1));assert [x['sample_index'] for x in ex]==list(range((CAP+1)*8));assert [x['sample_index'] for x in di]==list(range(CAP*8))
 counts=Counter();begun={};done={}
 for c in calls:
  cid=c['call_id'];ctx=c['context'];assert ctx['global_step']==6000+ctx['local_step'];assert c['event'] in ('CALL_BEGIN','CALL_DONE')
  target=begun if c['event']=='CALL_BEGIN' else done;assert cid not in target;target[cid]=c
  if c['event']=='CALL_DONE':counts[c['kind']]+=1
 assert set(begun)==set(done) and counts==Counter(train_forward=(CAP+1)*8,train_backward=(CAP+1)*8,optimizer_update=CAP+1)
 for cid,b in begun.items():
  d=done[cid];assert b['context']==d['context'] and b['kind']==d['kind'] and b['time_unix']<=d['time_unix']
 for b in range(CAP+1):
  rows=ex[8*b:8*b+8];assert len({x['uid'] for x in rows})==8 and sum(x['alpha']==1. for x in rows)==4
  for x in rows:
   pr=rec[x['sample_index']]
   for k in ('uid','task_id','N','K','alpha','t','condition_type','global_step'):assert x[k]==pr[k]
   assert x['local_step']==pr['step'] and x['free_coordinate_count']==9*(x['N']-x['K'])
   for kind in ('train_forward','train_backward'):
    matching=[x0 for x0 in ()] # exact ledger linkage verified below without additional model execution
 for c in begun.values():
  if c['kind']!='optimizer_update':
   e=ex[c['context']['sample_index']];assert c['context']['input_bytehash']==e['input_bytehash'] and c['context']['label_bytehash']==e['label_bytehash']
 maxfm=0.;maxendpoint=0.
 for x in di:
  e=ex[x['sample_index']];assert x['interface_mask_sha256']==e['interface_mask_sha256'] and x['interface_scalar_count']==3*e['interface_corner_count']
  assert x['interface_scalar_count']+x['other_scalar_count']==e['free_coordinate_count'];assert x['t']==e['t'];assert x['time_bin']==min(4,np.searchsorted(r['diagnostic_time_bin_edges'],x['t'],side='right')-1)
  factor=float(np.float32(1)-np.float32(x['t']))**2;assert x['endpoint_error_factor']==factor
  for part in ('interface','other'):
   s=x[part+'_velocity_sse'];num=x[part+'_scalar_count'];assert close(x[part+'_velocity_mse'],s/num);assert close(x[part+'_endpoint_scaled_sse'],s*factor);assert close(x[part+'_endpoint_scaled_mse'],s*factor/num)
  assert close(x['all_free_velocity_sse'],x['interface_velocity_sse']+x['other_velocity_sse']);fm=x['all_free_velocity_sse']/x['batch_free_coordinate_denominator'];assert close(fm,x['FM_contribution']);maxfm=max(maxfm,abs(fm-x['FM_contribution']))
 for log in logs:
  ds=di[(log['local_step']-1)*8:log['local_step']*8];assert sum(ex[x['sample_index']]['free_coordinate_count'] for x in ds)==log['free_coordinate_denominator'];assert sum(x['FM_contribution'] for x in ds)==log['FM']==log['total_loss']
 # Reconstruct only saved face/corner permutations and source labels. No Gaussian/time draws, OT or sample builder.
 manifest=read(r['interface_manifest_path']);source=read(r['source_data']);cyclic=np.array([[0,1,2],[1,2,0],[2,0,1]],np.int64);masks={};masktested=0
 for tid in r['train_task_ids']:
  row=manifest['tasks'][tid];a=arrays(row['mask_path']);masks[tid]=a[row['mask_key']];assert HASHES[row['mask_path']]==row['mask_sha256']
 for b in range(CAP+1):
  path=OUT/'ot/FULL_SEAT'/f'{b:05d}.npz';maps=arrays(path);receipt=read(path.with_suffix('.json'));assert receipt['sha256']==HASHES[str(path)] and receipt['identity']['plan_sha256']==plan_hash
  for row in rec[b*8:b*8+8]:
   i=row['sample_index'];n=row['N'];k=row['K'];task=source['tasks'][row['task_id']];known=np.asarray(task['source_face_ids']);free=np.asarray(task['free_source_ids']);rng=np.random.RandomState(row['permutation_seed']);orderfree=rng.permutation(n-k);corners=cyclic[rng.randint(0,3,size=n)];order=rng.permutation(n);face=maps[str(i)+'_face'];corner=maps[str(i)+'_corner'];inv=np.argsort(face);ic=np.argsort(corner,axis=1)[inv]
   assert np.array_equal(np.sort(face),np.arange(n-k)) and np.all(np.sort(corner,axis=1)==np.arange(3))
   ids=np.r_[known,free[orderfree][inv]][order];cc=np.concatenate((corners[:k],corners[k:][inv][np.arange(n-k)[:,None],ic]))[order];mask=masks[row['task_id']][ids[:,None],cc];knownmask=(np.arange(n)<k)[order]
   assert mask.sum()==masks[row['task_id']].sum() and not mask[knownmask].any() and ah(mask)==ex[i]['interface_mask_sha256'];assert str(maps[str(i)+'_key'].item())==ex[i]['ot_cache_key'];masktested+=1
 return dict(status='PASS',closed_updates=CAP,closed_samples=CAP*8,resume_first_update_verified=CAP+1,call_counts_through_resume_first_update=dict(counts),failed_calls=0,pending_calls_at_audited_cursor=0,interface_transport_samples=masktested,interface_source_masks=len(masks),interface_source_corners=sum(int(x.sum()) for x in masks.values()),diagnostic_rows=CAP*8,maximum_FM_contribution_roundoff=maxfm,units=dict(velocity_mse='squared model-coordinate velocity per scalar coordinate; interface and other denominators separate',endpoint_scaled_mse='(1-t)^2 times velocity MSE; teacher-forced pointwise endpoint error proxy, not rollout or nearest-surface error',total_FM='sum of eight microbatch SSE / whole-batch valid free scalar count'),no_extra_forward='normal-forward graph has one model call per microbatch; 8000+8 recorded, matching exposure; diagnostics detach and never call model',source_stream='New frozen FULL condition plan; original RNG restored, historical data-stream cursor not reused',new_Gaussians=0,new_OT_solves=0)

def checkpoints(r):
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
 files=[(Path(r['source_checkpoint']),0),(OUT/'checkpoints/FULL_SEAT_step1000.pt',1000)];out=[]
 for path,step in files:
  receipt=read(path.with_suffix('.json')) if step else None;expected=receipt['sha256'] if step else r['source_checkpoint_sha256'];actual=sha(path);assert actual==expected;HASHES[str(path)]=actual;p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
  assert p['global_step']==6000+step and p['update_in_progress'] is False and not any(k in p for k in ('ema','ema_state','generator_ema'))
  assert state(p['generator_state'])==p['generator_state_sha256'];assert tree(p['optimizer'])==p['optimizer_state_sha256'];assert tree(p['rng'])==p['rng_sha256']
  steps={int(x['step'].item()) for x in p['optimizer']['state'].values()};assert steps=={6000+step};assert len(p['generator_state'])==194==len(p['optimizer']['state']);assert all(x.dtype==torch.float32 and x.device.type=='cpu' for x in p['generator_state'].values())
  group=p['optimizer']['param_groups'][0];assert group['lr']==1e-5 and tuple(group['betas'])==(.9,.95) and group['weight_decay']==0
  if step:
   assert p['protocol']==r and p['stream_state']['batch_index']==step and p['stream_state']['next_sample_index']==step*8
   init=read(OUT/'training/FULL_SEAT/attempts/orchestrated_1000_to_3000/initialization.json')
   for a,b in [('loaded_generator_sha256','generator_state_sha256'),('loaded_optimizer_sha256','optimizer_state_sha256'),('loaded_rng_sha256','rng_sha256')]:assert init[a]==p[b]
  else:
   assert p['protocol']['EMA'] is False;init=read(OUT/'training/FULL_SEAT/attempts/segment_0000_1000/initialization.json')
   for a,b in [('loaded_generator_sha256','generator_state_sha256'),('loaded_optimizer_sha256','optimizer_state_sha256'),('loaded_rng_sha256','rng_sha256')]:assert init[a]==p[b]
  out.append(dict(path=str(path),file_sha256=actual,local_step=step,global_step=p['global_step'],generator_state_sha256=p['generator_state_sha256'],optimizer_state_sha256=p['optimizer_state_sha256'],rng_sha256=p['rng_sha256'],AdamW_steps=sorted(steps),EMA=False,actual_CPU_tensor_read=True));del p
 assert not torch.cuda.is_initialized();return dict(status='PASS',chain=out,CUDA_initialized=False,models_constructed=0,model_calls=0)

def evaluation(r):
 out=[];events=stream(OUT/'execution/worker_0_of_1_events.jsonl');counts=Counter(x['event'] for x in events)
 for label,expected_new in [('FULL_S0000',16),('FULL_S1000',24),('SMALL_C200',0),('CREF_C200',0)]:
  cache=read(OUT/'evaluation/cache_reuse'/f'{label}.json');idx=read(OUT/'evaluation/indices'/label/'complete.json');assert idx['RAW']==idx['POST']==24;assert cache['reused_requests']==24-expected_new
  allids={q['request_id'] for q in r['requests']};new=Counter();rolltimes=[]
  for q in r['requests']:
   raw=read(OUT/'evaluation/raw'/label/q['request_id']/'receipt.json');post=read(OUT/'evaluation/post'/label/q['request_id']/'receipt.json');isnew=raw['generation_status']=='NEW_WORLD_FRAME_EULER50';assert raw['label']==post['label']==label and raw['request_id']==post['request_id']==q['request_id'];assert raw['input_sha256']==q['input_sha256']
   if isnew:
    calls=stream(OUT/'evaluation/raw'/label/q['request_id']/'calls.jsonl');ends=[x for x in calls if x['event']=='ROLLOUT_COMPLETE'];assert len(ends)==1 and ends[0]['nfe']==50;cc=Counter(x['event'] for x in calls);assert cc['RESERVE']==51 and cc['RETURN']==50 and sum(x.get('kind')=='sampling_forward' and x['event']=='RESERVE' for x in calls)==50;rolltimes.append(ends[0]['time_unix']);new['rollouts']+=1;new['sampling_forward']+=50
    proposal=read(post['proposal_receipt']);assert proposal['model_forwards']==1 and proposal['checkpoint_sha256']==r['d0_checkpoint']['sha256'];assert post['new_D0_F']==1 and post['GEO_calls']==1;new['D0_forward']+=1
   else:assert raw['generation_status']=='VERIFIED_HISTORICAL_RAW_REUSE' and post['new_D0_F']==0
  assert new['rollouts']==expected_new
  for stage in ('raw','post'):
   index=read(OUT/'evaluation/indices'/label/(stage+'_index.json'));assert {x['request_id'] for x in index['rows']}==allids
  scores=read(OUT/'evaluation/score_completion'/f'{label}.json');boundary=read(OUT/'evaluation/boundary_completion'/f'{label}.json');assert len(scores['receipts'])==48 and boundary['cells']==48
  assert {(x['request_id'],x['stage']) for x in scores['receipts']}=={(x,s) for x in allids for s in ('RAW','POST')}
  if expected_new:
   job=read(OUT/'orchestration/jobs'/('post_'+label)/'started.json');assert max(rolltimes)<datetime.fromisoformat(job['utc']).timestamp()
  out.append(dict(label=label,RAW=24,POST=24,reused_requests=24-expected_new,new_calls=dict(new),score_rows=48,boundary_cells=48,all_RAW_before_D0=True))
 assert counts['D0_FORWARD_RESERVED']==counts['D0_FORWARD_RETURNED']==40
 return dict(status='PASS',labels=out,new_rollouts=40,new_generator_forward=2000,new_single_D0_forward=40,execution_event_counts=dict(counts),request_panel=24,conditions=6,noise_candidates=[0,1,2,3],new_noise_draws=0)

def resources():
 jobs=[]
 for folder in sorted((OUT/'orchestration/jobs').glob('*')):
  r=read(folder/'request.json');s=read(folder/'started.json');d=read(folder/'done.json') if (folder/'done.json').exists() else None;jobs.append(dict(job=r['job_id'],kind=r['kind'],start=datetime.fromisoformat(s['utc']).timestamp(),end=datetime.fromisoformat(d['finished']).timestamp() if d else None,returncode=d['returncode'] if d else None,command=r['command']))
 gpu=[x for x in jobs if x['kind']=='GPU'];pairs=[]
 for a in gpu:
  for b in gpu:
   if a['job']<b['job']:
    overlap=min(a['end'] or float('inf'),b['end'] or float('inf'))-max(a['start'],b['start'])
    if overlap>0:pairs.append([a['job'],b['job'],overlap])
 assert not pairs and all(x['returncode'] in (None,0) for x in jobs)
 led=stream(OUT/'ledger.jsonl');pre=[x for x in led if 'OVERLAP' in x.get('event','') or 'CONCURRENCY' in x.get('event','') or x.get('event')=='ADDITIONAL_GPU_DISPATCH']
 receipts=list((OUT/'orchestration/resources').glob('*.json'));cpu=[x for x in jobs if x['kind']=='CPU_PIPELINE'];maxcpu=max(sum(x['start']<=y['start']<(x['end'] or float('inf')) for x in cpu) for y in cpu) if cpu else 0;assert maxcpu<=2
 final=read(OUT/'training/FULL_SEAT/attempts/segment_0000_1000/final_status.json')
 return dict(status='PASS_WITH_RECORDED_PRECONTROLLER_OVERLAP',controller_GPU_stage_overlaps=pairs,max_controller_CPU_pipelines=maxcpu,jobs=jobs,resource_snapshot_files=len(receipts),precontroller_overlap_events=pre,first_segment_memory=final['resources'],torch_peak_reserved_first_segment=final['peak_reserved_bytes'],process_VRAM='UNKNOWN: nvidia-smi exposes N/A process memory; total device and torch allocated/reserved are separate',scientific_worker_settings='train batch8/micro1; one GPU stage under controller; score/boundary at most two workers each')

def main():
 started=datetime.now(timezone.utc).isoformat();r=read(OUT/'registration.json');hashchecks={}
 for path,h in {**r['training_code_hashes'],**r['evaluation_code_hashes']}.items():actual=sha(path);assert actual==h;hashchecks[path]=actual
 meta=metadata(r);print('METADATA PASS',meta['affected_requests'],flush=True)
 tr=training(r);print('TRAINING PASS',flush=True)
 ck=checkpoints(r);print('CHECKPOINTS PASS',flush=True)
 ev=evaluation(r);print('EVALUATION PASS',flush=True)
 resource=resources();print('RESOURCES PASS',flush=True)
 result=dict(status='PASS_WITH_NONSCIENTIFIC_METADATA_CAVEAT',started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),scope='Independent interim audit of completed local0 and1000 endpoints and first exact-resume update1001; active later training not certified by this snapshot',metadata=meta,training=tr,checkpoints=ck,evaluation=ev,resources=resource,readback_hashes=HASHES,frozen_code_hashes=hashchecks,audit_source_sha256=sha(__file__),call_counts_for_this_audit=dict(model_construction=0,generator_forward=0,D0_forward=0,backward=0,optimizer_update=0,new_Gaussian=0,new_OT=0,new_quality_scans=0),audit_engineering_notes=['Two initial readback attempts encountered absent optional inherited description keys; reader now explicitly distinguishes absent from stale metadata.','One readback attempt used the wrong event-name schema for generation; corrected to existing RESERVE/RETURN records. No scientific calls or outputs were repeated.'],blocking_issues=[])
 write_json(AUD/'interim_execution_review.json',result);print(json.dumps({k:result[k] for k in ('status','blocking_issues','call_counts_for_this_audit')},indent=2))
if __name__=='__main__':main()