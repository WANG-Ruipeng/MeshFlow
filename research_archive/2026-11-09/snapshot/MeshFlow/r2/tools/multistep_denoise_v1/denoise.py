"""Exactly four additional fixed denoiser calls per existing USER_MIX mesh."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
import gc,importlib.util,sys,time,json
from pathlib import Path
import numpy as np
import torch
from common import ROOT,OUT,PREV,ASSET,EXPECTED,SCALE,prepare,read,write,checked,sha,event,verify

def run():
 start=time.perf_counter();reg=prepare();regh=sha(OUT/'registration.json')
 spec=importlib.util.spec_from_file_location('_frozen_denoiser_helpers',ROOT/'r2/tools/postprocess_avmr_pmr_v1/denoise_run.py')
 d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
 # Old helpers only read official code/manifest. All their writes point to this new OUT.
 d.OUT=OUT;d.SRC=PREV/'source_snapshot/official_denoiser_db86bc3';d.write=write
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True,warn_only=False)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 pre=d.nv();head=d.check_headroom(pre);write(OUT/'resources_before_denoise.json',pre)
 if head['free_MiB']-head['additional_headroom_MiB']<2400:raise RuntimeError('Wait: insufficient memory for observed single worker plus safety headroom')
 if sha(ASSET)!=EXPECTED:raise ValueError('Official weight hash mismatch')
 checkpoint_stat=(ASSET.stat().st_size,ASSET.stat().st_mtime_ns)
 mod,attention,manifest=d.official_modules();d.math_checks(attention)
 oldload=read(PREV/'strict_load.json');kw=oldload['init_kwargs']
 ledger=OUT/'denoise_ledger.jsonl';prior=[json.loads(x) for x in ledger.read_text().splitlines() if x]
 attempts={(x['request_id'],x['round']) for x in prior if x['event']=='DENOISER_FORWARD_STARTED'}
 if len(attempts)>48:raise RuntimeError('Budget exceeded')
 # A complete rerun is a receipt validation; no extra model load needed.
 allkeys={(r['request_id'],n) for r in reg['records'] for n in reg['new_rounds']}
 complete={k for k in allkeys if (OUT/'denoised'/f'D{k[1]}'/k[0]/'receipt.json').is_file()}
 if complete==allkeys:
  for rid,n in allkeys:
   proof=read(OUT/'denoised'/f'D{n}'/rid/'receipt.json')
   if proof['status']!='PASS' or proof['registration_sha256']!=regh or sha(OUT/'denoised'/f'D{n}'/rid/'denoised.npz')!=proof['output_sha256']:raise ValueError('Invalid completed receipt')
  verify(reg);print(json.dumps(dict(status='REUSED_ALL_48',new_forwards=0)));return
 event('CPU_CHECKPOINT_LOAD_STARTED',checkpoint_sha256=EXPECTED)
 load_start=time.perf_counter();ck=torch.load(ASSET,weights_only=True,mmap=True,map_location='cpu')
 if ck['config']!=oldload['config']:raise ValueError('Checkpoint config changed')
 state={k.removeprefix('module.'):v for k,v in ck['ema'].items()};model=mod.AutoencoderKL(**kw)
 loaded=model.load_state_dict(state,strict=True)
 if loaded.missing_keys or loaded.unexpected_keys:raise ValueError('Strict load failed')
 if sum(p.numel() for p in model.parameters())!=455592963 or any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Parameter contract mismatch')
 model.eval().requires_grad_(False)
 for m in model.modules():
  if isinstance(m,attention.SelfAttention):
   if m.dropout!=0 or m.causal:raise ValueError('Unexpected stochastic attention')
   m.mixed_precision='fp32'
 write(OUT/'strict_load.json',dict(status='PASS',checkpoint_sha256=EXPECTED,branch='ema',parameters=455592963,init_kwargs=kw,CPU_load_seconds=time.perf_counter()-load_start,strict_missing=[],strict_unexpected=[]))
 del state,ck;gc.collect();model.to('cuda');torch.cuda.synchronize()
 on=d.nv();d.check_headroom(on);write(OUT/'resources_model_resident.json',on)
 phase=time.perf_counter();calls=0;timings=[]
 for r in reg['records']:
  rid=r['request_id'];K,N=int(r['K']),int(r['N'])
  if sha(r['raw_path'])!=r['raw_sha256'] or sha(r['D1_denoised_path'])!=r['D1_denoised_sha256']:raise ValueError('RAW or D1 mismatch')
  with np.load(r['raw_path'],allow_pickle=False) as data:raw=data['raw'].reshape(N,3,3).copy();C=data['C'].reshape(K,3,3).copy()
  with np.load(r['D1_denoised_path'],allow_pickle=False) as data:one=data['output'].reshape(N,3,3).copy()
  if raw[:K].tobytes()!=C.tobytes() or one[:K].tobytes()!=C.tobytes():raise ValueError('Original C changed')
  previous=one;previous_path=Path(r['D1_denoised_path'])
  for n in reg['new_rounds']:
   key=(rid,n);dest=checked(OUT/'denoised'/f'D{n}'/rid);path=dest/'denoised.npz';receipt=dest/'receipt.json'
   inputhash=sha(previous_path)
   identity=dict(request_id=rid,label='R2_USER_MIX',round=n,raw_sha256=r['raw_sha256'],input_path=str(previous_path),input_sha256=inputhash,registration_sha256=regh,checkpoint_sha256=EXPECTED)
   if receipt.exists():
    proof=read(receipt)
    if proof['status']!='PASS' or any(proof[k]!=v for k,v in identity.items()) or sha(path)!=proof['output_sha256']:raise ValueError('Invalid reusable round')
    with np.load(path,allow_pickle=False) as a:previous=a['output'].copy()
    if previous.dtype!=np.float32 or previous.shape!=(N,3,3) or previous[:K].tobytes()!=C.tobytes():raise ValueError('Bad saved C or N')
    previous_path=path;continue
   if key in attempts:raise RuntimeError('Started call lacks receipt; no automatic replay')
   if len(attempts)>=48:raise RuntimeError('48 call limit')
   dest.mkdir(parents=True,exist_ok=True)
   if path.exists():raise RuntimeError('Incomplete output exists; no overwrite/replay')
   if previous.dtype!=np.float32 or previous.shape!=(N,3,3) or not np.isfinite(previous).all():raise ValueError('Invalid round input')
   if previous[:K].tobytes()!=C.tobytes():raise ValueError('C drift before inference')
   x=torch.from_numpy(previous.reshape(1,N,9)).to('cuda')*SCALE;cond=torch.tensor([N],dtype=torch.long,device='cuda')
   rng_cpu=torch.get_rng_state().clone();rng_gpu=torch.cuda.get_rng_state().clone()
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
   event('DENOISER_FORWARD_STARTED',**identity,N=N,K=K);attempts.add(key)
   try:
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     z=model.pe_nerf(x.reshape(1,3*N,3));residual=model.decode(z,cond,mask=None)
    proposal=((residual.float()+x.float())/SCALE).reshape(N,3,3).cpu().numpy().copy();torch.cuda.synchronize();elapsed=time.perf_counter()-tick
    if proposal.dtype!=np.float32 or not np.isfinite(proposal).all():raise ValueError('Nonfinite output')
    if not torch.equal(rng_cpu,torch.get_rng_state()) or not torch.equal(rng_gpu,torch.cuda.get_rng_state()):raise ValueError('RNG consumed')
    output=proposal.copy();output[:K]=C
    step=output.astype(float)-previous.astype(float);total=output.astype(float)-raw.astype(float);fromone=output.astype(float)-one.astype(float)
    cproposal=np.linalg.norm(proposal[:K].astype(float)-C.astype(float),axis=-1)
    with path.open('xb') as f:np.savez_compressed(f,output=output,C=C,official_unconstrained_proposal=proposal,displacement_from_previous=step,displacement_from_RAW=total,displacement_from_D1=fromone,known_faces=np.asarray(K),input_faces=np.asarray(N),coordinate_scale=np.asarray(SCALE),round=np.asarray(n))
    with np.load(path,allow_pickle=False) as a:
     if a['output'].tobytes()!=output.tobytes() or a['output'][:K].tobytes()!=C.tobytes():raise ValueError('Output roundtrip mismatch')
    free_step=np.linalg.norm(step[K:],axis=-1);free_total=np.linalg.norm(total[K:],axis=-1)
    peaks=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),scope='Resident FP32 official network plus one BF16 forward; no optimizer')
    proof=dict(status='PASS',**identity,output_sha256=sha(path),N=N,K=K,seconds=elapsed,C_bitwise=True,C_proposal_max_displacement_model=float(cproposal.max()),max_step_free_displacement_model=float(free_step.max()),mean_step_free_displacement_model=float(free_step.mean()),max_total_free_displacement_model=float(free_total.max()),mean_total_free_displacement_model=float(free_total.mean()),no_rng_consumed=True,residual_dtype=str(residual.dtype),precision='BF16 network; FP32 parameters and residual/scale',no_intermediate_GEO=True,vram=peaks)
    write(receipt,proof);event('DENOISER_FORWARD_COMPLETE',**identity,seconds=elapsed,**peaks);calls+=1;timings.append(elapsed)
    if calls==1:
     current=d.nv();d.check_headroom(current)
     write(OUT/'first_actual_forward.json',dict(resources=current,result=proof,GPU_workers=1,dispatch_reason='Measured first authorized call; prior comparable48 calls phase5.13s, CPU model load+transfer longer. Each request round depends on prior. Available VRAM allows more copies but remaining47 brief calls do not justify duplicate process/load. CPU scoring uses independent workers.'))
    del x,cond,z,residual;previous=output;previous_path=path
   except BaseException as error:
    write(dest/'failure.json',dict(status='STOPPED',**identity,error=repr(error),forward_attempt_consumed=True));event('DENOISER_FAILED',**identity,error=repr(error));raise
 torch.cuda.synchronize();write(OUT/'resources_after_denoise.json',d.nv())
 if (ASSET.stat().st_size,ASSET.stat().st_mtime_ns)!=checkpoint_stat:raise ValueError('Checkpoint changed')
 proofs=[read(OUT/'denoised'/f'D{n}'/r['request_id']/'receipt.json') for r in reg['records'] for n in reg['new_rounds']]
 if len(attempts)!=48 or len(proofs)!=48 or any(p['status']!='PASS' for p in proofs):raise ValueError('Incomplete forward budget')
 protected=verify(reg)
 result=dict(status='PASS',requests=12,reused_D1=12,new_rounds=[2,3,4,5],total_forward_attempts=len(attempts),new_forwards_this_invocation=calls,completed_round_outputs=len(proofs),all_C_bitwise=True,all_N_preserved=True,GPU_workers=1,training_updates=0,generator_forwards=0,new_noise=0,forward_phase_wall_seconds=time.perf_counter()-phase,total_wall_seconds=time.perf_counter()-start,sum_forward_seconds=sum(timings),peak_allocated_bytes=max(p['vram']['peak_allocated_bytes'] for p in proofs),peak_reserved_bytes=max(p['vram']['peak_reserved_bytes'] for p in proofs),protected_hashes=protected)
 write(OUT/'denoise_completion.json',result);event('DENOISE_COMPLETE',**result);print(json.dumps(result))
if __name__=='__main__':run()