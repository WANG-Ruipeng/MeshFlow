"""Two bounded C-input feedback updates; F_RAW is unchanged at every query."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import gc,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import common as C

def clip_delta(x,bound):
 n=np.linalg.norm(x,axis=-1,keepdims=True)
 return x*np.minimum(1.,bound/np.maximum(n,np.finfo(float).tiny))

def working_input(raw,c,delta,bound):
 working=raw.copy();K=len(c);working[:K]=(c.astype(float)+delta).astype(np.float32)
 rounded=np.zeros(c.shape[:-1],dtype=bool)
 for _ in range(32):
  bad=np.linalg.norm(working[:K].astype(float)-c,axis=-1)>bound
  if not bad.any():break
  rounded|=bad;working[:K][bad]=np.nextafter(working[:K][bad],c[bad])
 if (np.linalg.norm(working[:K].astype(float)-c,axis=-1)>bound).any():raise ValueError('FP32 query bound failed')
 return working,int(rounded.sum())

def validate_completed(reg,rh):
 proofs=[];bound=reg['methods']['FEEDBACK']['max_working_C_corner_delta_L']*reg['fixed_L'];eta=reg['methods']['FEEDBACK']['eta']
 for r in reg['records']:
  rid=r['request_id'];K,N=int(r['K']),int(r['N'])
  with np.load(r['raw_path'],allow_pickle=False) as a:raw=a['raw'].reshape(N,3,3).copy();c=a['C'].copy()
  with np.load(r['D0_denoised_path'],allow_pickle=False) as a:proposal=a['official_unconstrained_proposal'].copy()
  delta=np.zeros_like(c,dtype=float)
  for j in (1,2):
   error=c.astype(float)-proposal[:K];delta=clip_delta(delta+eta*error,bound);working,_=working_input(raw,c,delta,bound)
   path=C.OUT/'feedback_queries'/rid/f'update{j}'/'query.npz';q=C.read(path.with_name('receipt.json'))
   expected=dict(request_id=rid,update=j,raw_sha256=r['raw_sha256'],registration_sha256=rh,checkpoint_sha256=C.EXPECTED)
   if q['status']!='PASS' or any(q[k]!=v for k,v in expected.items()) or C.sha(path)!=q['output_sha256']:raise ValueError('Completed identity mismatch')
   with np.load(path,allow_pickle=False) as a:
    if a['working_input'].tobytes()!=working.tobytes() or not np.array_equal(a['delta'],delta) or not np.array_equal(a['previous_C_error'],error):raise ValueError('Completed recurrence mismatch')
    if a['working_input'][K:].tobytes()!=raw[K:].tobytes() or a['output'][:K].tobytes()!=c.tobytes():raise ValueError('Completed C/F identity mismatch')
    proposal=a['proposal'].copy()
    if a['output'][K:].tobytes()!=proposal[K:].tobytes():raise ValueError('Wrong free output')
   proofs.append(q)
 return proofs

def feedback_checks():
 """Independent analytical systems; NOT evidence that actual D0 attaches meshes."""
 rows=[]
 # D_C(c+delta)=c+delta+b; fixed F response=.4delta.
 b=np.asarray([[[.02,0,0]]]);delta=np.zeros_like(b);errs=[];free=[]
 for _ in range(3):
  err=-(delta+b);errs.append(float(np.linalg.norm(err)));free.append(float(np.linalg.norm(.4*delta)))
  delta=clip_delta(delta+.5*err,.2)
 assert np.allclose(errs,[.02,.01,.005]) and free[-1]>.0
 rows.append(dict(case='linear_cross_coupled',C_errors=errs,F_changes=free,status='PASS'))
 rows.append(dict(case='zero_cross_coupling',C_errors=errs,F_changes=[0.,0.,0.],status='PASS',meaning='C residual reduction cannot certify F improvement'))
 x=clip_delta(np.asarray([[[3.,4.,0.]]]),.02)
 assert abs(np.linalg.norm(x)-.02)<1e-12
 rows.append(dict(case='projection_saturation',norm=float(np.linalg.norm(x)),bound=.02,status='PASS',meaning='Bounded input does not certify zero output residual'))
 C.write(C.OUT/'feedback_checks.json',dict(status='PASS',cases=rows,model_forwards=0,synthetic_GPU_forwards=0))

def run():
 started=time.perf_counter();reg=C.prepare();rh=C.sha(C.OUT/'registration.json');feedback_checks()
 records=reg['records'];proofs=[]
 ledger=C.OUT/'ledger.jsonl'
 events=[json.loads(x) for x in ledger.read_text().splitlines() if x]
 attempts={(e['request_id'],e['update']) for e in events if e['event']=='D0_FEEDBACK_STARTED'}
 allkeys={(r['request_id'],j) for r in records for j in (1,2)}
 if not attempts.issubset(allkeys):raise ValueError('Unexpected forward attempt')
 pending=False
 for rid,j in sorted(allkeys):
  path=C.OUT/'feedback_queries'/rid/f'update{j}'/'receipt.json'
  if path.exists():
   q=C.read(path)
   if q['status']!='PASS' or q['registration_sha256']!=rh or C.sha(path.with_name('query.npz'))!=q['output_sha256']:raise ValueError('Invalid completed query')
  elif (rid,j) in attempts:raise RuntimeError('Started query without receipt; no automatic replay')
  else:pending=True
 if not pending:
  proofs=validate_completed(reg,rh)
  for r in records:export_endpoint(r,rh)
  summarize(reg)
  if not (C.OUT/'denoise_completion.json').exists():
   C.write(C.OUT/'denoise_completion.json',dict(status='PASS',total_forward_attempts=len(attempts),completed_queries=len(proofs),new_forwards_this_invocation=0,reused_D0_proposals=12,training_updates=0,new_generator_forwards=0,new_noise=0,synthetic_D0_forwards=0,GPU_workers=1,all_F_RAW_inputs_exact=True,all_export_C_exact=True,sum_forward_seconds=sum(p['seconds'] for p in proofs),forward_phase_wall_seconds=None,wall_time_status='NA_PRIOR_INVOCATION_INTERRUPTED_AFTER_QUERY',peak_allocated_bytes=max(p['vram']['peak_allocated_bytes'] for p in proofs),peak_reserved_bytes=max(p['vram']['peak_reserved_bytes'] for p in proofs)))
  C.verify(reg,weights=False);print('ALL_24_QUERIES_REUSED');return
 import torch
 spec=importlib.util.spec_from_file_location('_coupling_official_helper',C.ROOT/'r2/tools/postprocess_avmr_pmr_v1/denoise_run.py')
 d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
 d.OUT=C.OUT;d.SRC=C.OLD/'source_snapshot/official_denoiser_db86bc3';d.write=C.write
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 torch.use_deterministic_algorithms(True,warn_only=False)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 pre=C.resources();head=d.check_headroom(pre);C.write(C.OUT/'resources_before_denoise.json',pre)
 if head['free_MiB']-head['additional_headroom_MiB']<2600:raise RuntimeError('WAIT: insufficient measured headroom for official inference')
 if C.sha(C.ASSET)!=C.EXPECTED:raise ValueError('D0 hash mismatch')
 asset_stat=(C.ASSET.stat().st_size,C.ASSET.stat().st_mtime_ns)
 for name in ('RMSNorm_exact_extraction.py','positional_utils_exact_extraction.py'):C.checked(C.OUT/'source_snapshot/runtime'/name)
 mod,attention,manifest=d.official_modules();d.math_checks(attention)
 oldload=C.read(C.OLD/'strict_load.json');kw=oldload['init_kwargs']
 tick=time.perf_counter();C.event('MODEL_LOAD_STARTED',checkpoint_sha256=C.EXPECTED)
 ck=torch.load(C.ASSET,map_location='cpu',weights_only=True,mmap=True)
 if ck['config']!=oldload['config']:raise ValueError('Checkpoint config mismatch')
 state={k.removeprefix('module.'):v for k,v in ck['ema'].items()};model=mod.AutoencoderKL(**kw)
 status=model.load_state_dict(state,strict=True)
 if status.missing_keys or status.unexpected_keys:raise ValueError('Non-strict load')
 if sum(p.numel() for p in model.parameters())!=455592963 or any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Wrong official model')
 model.eval().requires_grad_(False)
 for m in model.modules():
  if isinstance(m,attention.SelfAttention):
   if m.dropout!=0 or m.causal:raise ValueError('Unexpected stochastic/causal attention')
   m.mixed_precision='fp32'
 C.write(C.OUT/'strict_load.json',dict(status='PASS',checkpoint_sha256=C.EXPECTED,branch='ema',parameters=455592963,init_kwargs=kw,CPU_load_seconds=time.perf_counter()-tick,missing=[],unexpected=[]))
 del state,ck;gc.collect();torch.cuda.reset_peak_memory_stats();model.to('cuda');torch.cuda.synchronize()
 versions={n:p._version for n,p in model.named_parameters()}
 resident=C.resources();d.check_headroom(resident);C.write(C.OUT/'resources_model_resident.json',resident)
 calls=0;timings=[];phase=time.perf_counter();trajectory=[];corners=[]
 eta=reg['methods']['FEEDBACK']['eta'];bound=reg['methods']['FEEDBACK']['max_working_C_corner_delta_L']*reg['fixed_L']
 for r in records:
  rid=r['request_id'];K,N=int(r['K']),int(r['N'])
  with np.load(r['raw_path'],allow_pickle=False) as a:raw=a['raw'].reshape(N,3,3).copy();c=a['C'].copy()
  with np.load(r['D0_denoised_path'],allow_pickle=False) as a:base=a['official_unconstrained_proposal'].copy()
  proposal=base;delta=np.zeros_like(c,dtype=np.float64)
  for j in (1,2):
   previous_error=c.astype(float)-proposal[:K].astype(float)
   unprojected=delta+eta*previous_error;delta=clip_delta(unprojected,bound)
   working,roundoff_corners=working_input(raw,c,delta,bound)
   if working[K:].tobytes()!=raw[K:].tobytes():raise ValueError('F_RAW changed')
   dest=C.OUT/'feedback_queries'/rid/f'update{j}';out=dest/'query.npz';receipt=dest/'receipt.json'
   identity=dict(request_id=rid,update=j,raw_sha256=r['raw_sha256'],registration_sha256=rh,checkpoint_sha256=C.EXPECTED)
   if receipt.exists():
    proof=C.read(receipt)
    if any(proof[k]!=v for k,v in identity.items()) or C.sha(out)!=proof['output_sha256']:raise ValueError('Query identity mismatch')
    with np.load(out,allow_pickle=False) as a:
     if a['working_input'].tobytes()!=working.tobytes() or not np.array_equal(a['delta'],delta):raise ValueError('Feedback recurrence mismatch')
     proposal=a['proposal'].copy()
    continue
   if (rid,j) in attempts or len(attempts)>=24 or out.exists():raise RuntimeError('No replay/overwrite of incomplete query')
   x=torch.from_numpy(working.reshape(1,N,9)).to('cuda')*C.SCALE
   cond=torch.tensor([N],device='cuda',dtype=torch.long)
   rng_cpu=torch.get_rng_state().clone();rng_gpu=torch.cuda.get_rng_state().clone()
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
   C.event('D0_FEEDBACK_STARTED',**identity,N=N,K=K);attempts.add((rid,j))
   try:
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     z=model.pe_nerf(x.reshape(1,3*N,3));residual=model.decode(z,cond,mask=None)
    proposal_d=residual.float()+x.float();proposal=(proposal_d/C.SCALE).reshape(N,3,3).cpu().numpy().copy()
    torch.cuda.synchronize();elapsed=time.perf_counter()-tick
    if proposal.dtype!=np.float32 or not np.isfinite(proposal).all():raise ValueError('Nonfinite proposal')
    if not torch.equal(rng_cpu,torch.get_rng_state()) or not torch.equal(rng_gpu,torch.cuda.get_rng_state()):raise ValueError('RNG consumed')
    output=proposal.copy();output[:K]=c
    C.save_npz(out,output=output,C=c,proposal=proposal,working_input=working,actual_input_d=x.cpu().numpy(),actual_proposal_d=proposal_d.cpu().numpy(),delta=delta,unprojected_delta=unprojected,previous_C_error=previous_error,C_error=c.astype(float)-proposal[:K],free_change_from_D0=proposal[K:].astype(float)-base[K:])
    with np.load(out,allow_pickle=False) as a:
     if a['output'].tobytes()!=output.tobytes() or a['output'][:K].tobytes()!=c.tobytes():raise ValueError('Roundtrip or C mismatch')
    vram=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),scope='Resident official FP32 model plus one BF16 inference; reset immediately before query; no optimizer')
    proof=dict(status='PASS',**identity,output_sha256=C.sha(out),seconds=elapsed,N=N,K=K,C_bitwise=True,F_RAW_input_bitwise=True,no_rng_consumed=True,residual_dtype=str(residual.dtype),C_error_mean_model=float(np.linalg.norm(c.astype(float)-proposal[:K],axis=-1).mean()),C_work_delta_max_model=float(np.linalg.norm(working[:K].astype(float)-c,axis=-1).max()),F_change_from_D0_mean_model=float(np.linalg.norm(proposal[K:].astype(float)-base[K:],axis=-1).mean()),projection_saturated_corners=int((np.linalg.norm(unprojected,axis=-1)>bound).sum()),FP32_bound_roundoff_corners=roundoff_corners,vram=vram)
    C.write(receipt,proof);C.event('D0_FEEDBACK_COMPLETE',**identity,seconds=elapsed,**vram);calls+=1;timings.append(elapsed)
    if calls==1:
     current=C.resources();d.check_headroom(current)
     C.write(C.OUT/'first_actual_query.json',dict(resources=current,result=proof,actual_GPU_workers=1,reason='Prior comparable48 official queries took approximately5s; model loading/transfer costs dominate. After first authorized measured query, retain one GPU worker for remaining23; independent CPU work overlaps. Process VRAM field may be unavailable under WDDM; reserved bytes are not treated as full process peak.',comparison_before=pre['gpu'],comparison_resident=resident['gpu']))
    del x,cond,z,residual,proposal_d
   except BaseException as e:
    C.write(dest/'failure.json',dict(status='STOPPED',**identity,error=repr(e),attempt_consumed=True));C.event('D0_FEEDBACK_FAILED',**identity,error=repr(e));raise
  export_endpoint(r,rh)
 if {n:p._version for n,p in model.named_parameters()}!=versions:raise ValueError('Model parameters mutated')
 if any(p.grad is not None or p.requires_grad for p in model.parameters()):raise ValueError('Gradient state changed')
 if (C.ASSET.stat().st_size,C.ASSET.stat().st_mtime_ns)!=asset_stat:raise ValueError('Weight file changed')
 torch.cuda.synchronize();C.write(C.OUT/'resources_after_denoise.json',C.resources())
 proofs=[C.read(C.OUT/'feedback_queries'/r['request_id']/f'update{j}'/'receipt.json') for r in records for j in (1,2)]
 if len(attempts)!=24 or any(p['status']!='PASS' for p in proofs):raise ValueError('Incomplete24 queries')
 validate_completed(reg,rh)
 summarize(reg)
 summary=dict(status='PASS',total_forward_attempts=len(attempts),completed_queries=len(proofs),new_forwards_this_invocation=calls,reused_D0_proposals=12,training_updates=0,new_generator_forwards=0,new_noise=0,synthetic_D0_forwards=0,GPU_workers=1,FP32_parameters_unchanged=True,all_F_RAW_inputs_exact=True,all_export_C_exact=True,sum_forward_seconds=sum(timings),forward_phase_wall_seconds=time.perf_counter()-phase,total_wall_seconds=time.perf_counter()-started,peak_allocated_bytes=max(p['vram']['peak_allocated_bytes'] for p in proofs),peak_reserved_bytes=max(p['vram']['peak_reserved_bytes'] for p in proofs))
 C.write(C.OUT/'denoise_completion.json',summary);C.event('DENOISE_COMPLETE',**summary);print(json.dumps(summary))

def export_endpoint(r,rh):
 rid=r['request_id'];src=C.OUT/'feedback_queries'/rid/'update2/query.npz';q=C.read(src.with_name('receipt.json'))
 if q['status']!='PASS' or q['output_sha256']!=C.sha(src):raise ValueError('Bad endpoint query')
 dest=C.OUT/'pre_geo/FEEDBACK'/rid;out=dest/'denoised.npz';receipt=dest/'receipt.json'
 identity=dict(request_id=rid,method='FEEDBACK',raw_sha256=r['raw_sha256'],registration_sha256=rh,query_sha256=q['output_sha256'])
 if receipt.exists():
  p=C.read(receipt)
  if any(p[k]!=v for k,v in identity.items()) or p['output_sha256']!=C.sha(out):raise ValueError('Endpoint mismatch')
  return
 with np.load(src,allow_pickle=False) as a:C.save_npz(out,output=a['output'],C=a['C'],official_unconstrained_proposal=a['proposal'])
 C.write(receipt,dict(status='PASS',**identity,output_sha256=C.sha(out),C_bitwise=True,final_update=2))

def summarize(reg):
 rows=[];corner_rows=[]
 for r in reg['records']:
  rid=r['request_id'];K=int(r['K'])
  with np.load(r['raw_path'],allow_pickle=False) as a:c=a['C'].copy()
  with np.load(r['D0_denoised_path'],allow_pickle=False) as a:base=a['official_unconstrained_proposal'].copy()
  for j in (0,1,2):
   if j==0:proposal=base;delta=np.zeros_like(c,dtype=float);saturated=0
   else:
    p=C.OUT/'feedback_queries'/rid/f'update{j}'/'query.npz'
    with np.load(p,allow_pickle=False) as a:proposal=a['proposal'].copy();delta=a['working_input'][:K].astype(float)-c
    saturated=C.read(p.with_name('receipt.json'))['projection_saturated_corners']
   err=np.linalg.norm(c.astype(float)-proposal[:K],axis=-1);dm=np.linalg.norm(delta,axis=-1);fm=np.linalg.norm(proposal[K:].astype(float)-base[K:],axis=-1)
   meta={k:r[k] for k in ('request_id','parent_id','condition_id','candidate','region_type','tier')}
   rows.append(dict(**meta,update=j,C_error_mean_model=float(err.mean()),C_error_max_model=float(err.max()),work_delta_max_model=float(dm.max()),F_change_from_D0_mean_model=float(fm.mean()),F_change_from_D0_max_model=float(fm.max()),saturated_corners=saturated))
   for i in range(K):
    for k in range(3):corner_rows.append(dict(**meta,update=j,C_face=i,corner=k,C_error_model=float(err[i,k]),work_delta_model=float(dm[i,k])))
 C.csvwrite(C.OUT/'feedback_per_update.csv',rows);C.csvwrite(C.OUT/'feedback_per_C_corner.csv',corner_rows)

if __name__=='__main__':run()
