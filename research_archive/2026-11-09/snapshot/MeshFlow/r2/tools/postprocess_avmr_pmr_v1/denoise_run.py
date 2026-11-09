"""One official released Chair denoiser forward for each of 48 existing RAWs.
No encoder sampling, training, generator construction, or additional rollouts.
Official historical source stays read-only; C projection is explicitly local.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
import argparse, ast, csv, gc, hashlib, importlib, json, subprocess, sys, time, types
from pathlib import Path
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'experiments/r2_postprocess_avmr_pmr_v1'
ASSET=OUT/'assets/chair_denoiser_0030000.pt'
SRC=OUT/'source_snapshot/official_denoiser_db86bc3'
SCALE=.95*.3762
VERSION='official_chair_denoise_c_projection_v1'
EXPECTED='6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def event(kind,**kw):
    with (OUT/'denoise_ledger.jsonl').open('a') as f:
        f.write(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**kw))+'\n')
def nv():
    def run(args):
        p=subprocess.run(args,capture_output=True,text=True,timeout=15)
        return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    return dict(gpu=run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        processes=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
        cpu_logical=os.cpu_count(),loadavg=list(os.getloadavg()),memory=Path('/proc/meminfo').read_text())
def check_headroom(resources):
    line=resources['gpu']['stdout'].strip().splitlines()[0]
    fields=[p.strip() for p in line.split(',')];total,free=float(fields[1]),float(fields[3])
    if free<max(2048.,.1*total):raise RuntimeError('GPU headroom below protected margin')
    return dict(total_MiB=total,free_MiB=free,additional_headroom_MiB=max(2048.,.1*total))

def official_modules():
    manifest=read(OUT/'asset_audit/official_denoiser_source_manifest.json')
    for f in manifest['files']:
        if sha(SRC/f['path'])!=f['sha256']:raise ValueError('Official source hash mismatch')
    # Exact official RMSNorm class, with no unused fairscale parallel classes.
    source=(SRC/'models/rmsnorm.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='RMSNorm')
    segment=ast.get_source_segment(source,node)
    shim=types.ModuleType('models.rmsnorm');shim.__dict__['torch']=torch;shim.__dict__['nn']=torch.nn
    exec(compile(segment,str(SRC/'models/rmsnorm.py')+'[RMSNorm only]','exec'),shim.__dict__)
    sys.modules['models.rmsnorm']=shim
    adapter=OUT/'source_snapshot/runtime/RMSNorm_exact_extraction.py'
    if adapter.exists() and adapter.read_text()!='import torch\nfrom torch import nn\n\n'+segment+'\n':raise ValueError('RMSNorm extraction changed')
    adapter.parent.mkdir(parents=True,exist_ok=True);adapter.write_text('import torch\nfrom torch import nn\n\n'+segment+'\n')
    utils_source=(SRC/'models/utils.py').read_text()
    wanted={'get_2d_sincos_pos_embed','get_2d_sincos_pos_embed_from_grid','get_1d_sincos_pos_embed_from_grid'}
    functions=[ast.get_source_segment(utils_source,n) for n in ast.parse(utils_source).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    if len(functions)!=3:raise ValueError('Official positional helper extraction incomplete')
    utils_code='import numpy as np\n\n'+'\n\n'.join(functions)+'\n'
    utils_shim=types.ModuleType('models.utils');exec(compile(utils_code,str(SRC/'models/utils.py')+'[sincos helpers only]','exec'),utils_shim.__dict__)
    sys.modules['models.utils']=utils_shim
    utils_path=OUT/'source_snapshot/runtime/positional_utils_exact_extraction.py'
    if utils_path.exists() and utils_path.read_text()!=utils_code:raise ValueError('Positional helper extraction changed')
    utils_path.write_text(utils_code)
    sys.path.insert(0,str(SRC))
    model_module=importlib.import_module('models.equivae')
    attention=importlib.import_module('models.attention')
    if not str(Path(model_module.__file__).resolve()).startswith(str(SRC.resolve())):raise ValueError('Wrong models module')
    return model_module,attention,manifest

def math_checks(att):
    # Fixed CPU attention arithmetic, not a model forward or GPU probe.
    q=torch.arange(24,dtype=torch.float32).reshape(1,3,2,4)/20
    k=torch.arange(32,dtype=torch.float32).reshape(1,4,2,4)/30
    v=torch.arange(32,dtype=torch.float32).reshape(1,4,2,4)/10
    direct=torch.softmax((q.transpose(1,2)@k.transpose(1,2).transpose(-1,-2))/2,dim=-1)@v.transpose(1,2)
    actual=att.attention(q,k,v,mask_q=None,mask_kv=None,dropout=0,causal=False,backend='pytorch-sdpa')
    err=float((actual-direct.transpose(1,2)).abs().max())
    if err>1e-6:raise AssertionError('Unmasked SDPA mismatch')
    write(OUT/'denoise_math_checks.json',dict(status='PASS',CPU_fixed_attention_max_abs_error=err,
        model_forwards=0,GPU_forwards=0,no_padding=True,mask=None,scale=SCALE))

def run():
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    asset=read(OUT/'asset_audit/denoiser_asset_registration.json')
    if not ASSET.is_file() or ASSET.stat().st_size!=asset['expected_bytes']:raise ValueError('Verified full denoiser checkpoint unavailable')
    if sha(ASSET)!=EXPECTED:raise ValueError('Official checkpoint SHA256 mismatch')
    checkpoint_stat=(ASSET.stat().st_size,ASSET.stat().st_mtime_ns)
    pre=nv();head=check_headroom(pre);write(OUT/'resources_before_denoise.json',pre)
    if head['free_MiB']-head['additional_headroom_MiB']<2400:raise RuntimeError('Not enough observed free memory for single 500M inference worker plus headroom')
    mod,attention,source_manifest=official_modules();math_checks(attention)
    event('CPU_CHECKPOINT_LOAD_STARTED',checkpoint_sha256=EXPECTED,weights_only=True,mmap=True)
    t=time.perf_counter();ck=torch.load(ASSET,map_location='cpu',weights_only=True,mmap=True)
    config=ck['config'];cfg=config['model'];data=config['data']
    kw=dict(hidden_dim=cfg['hidden_dim'],num_layers=cfg['num_layers'],num_heads=cfg['num_heads'],latent_channels=cfg['latent_channels'],
        decoder_type=cfg['decoder_type'],num_bins=data['num_bins'],use_rmsnorm=cfg['use_rms'],face_bin=cfg['face_bin'],
        use_identity_encoder=cfg['use_identity_encoder'],fixed_std=cfg['fixed_std'])
    if kw!=asset['model_config_from_pickle_opcodes']:raise ValueError('Actual checkpoint config differs from registered official shape contract')
    if '03001627' not in str(data['data_path']):raise ValueError('Not the Chair checkpoint')
    branch='ema' if 'ema' in ck else 'model'
    state={k.removeprefix('module.'):v for k,v in ck[branch].items()}
    if len(state)!=len(ck[branch]):raise ValueError('State prefix collision')
    model=mod.AutoencoderKL(**kw)
    result=model.load_state_dict(state,strict=True)
    if result.missing_keys or result.unexpected_keys:raise ValueError('Non-strict official load')
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Expected FP32 parameters')
    num_params=sum(p.numel() for p in model.parameters());state_keys=len(state)
    model.eval().requires_grad_(False)
    for m in model.modules():
        if isinstance(m,attention.SelfAttention):
            if m.dropout!=0 or m.causal:raise ValueError('Unexpected stochastic/causal attention')
            m.mixed_precision='fp32' # backend selector only; outer BF16 remains
    load_info=dict(status='PASS',checkpoint_sha256=EXPECTED,checkpoint_bytes=ASSET.stat().st_size,
       state_branch=branch,state_keys=state_keys,parameters=num_params,config=config,init_kwargs=kw,
       strict_missing_keys=[],strict_unexpected_keys=[],CPU_load_seconds=time.perf_counter()-t,
       official_source_commit=source_manifest['commit'])
    write(OUT/'strict_load.json',load_info);del state,ck;gc.collect()
    reg=dict(version=VERSION,source_sha256=sha(__file__),official_source_manifest=source_manifest,
       checkpoint_sha256=EXPECTED,checkpoint_bytes=ASSET.stat().st_size,state_branch=branch,init_kwargs=kw,
       input_registration_sha256=sha(OUT/'registration.json'),raw_records=read(OUT/'registration.json')['records'],
       coordinate_scale=SCALE,coordinate_operation='Fixed multiply before network, fixed divide after; no bbox fit/clip/axis reorder',
       network='pe_nerf(x.reshape(1,3N,3)); decode(z,N,mask=None); FP32 residual sum; no encode/sample',
       C_contract='Save unconstrained proposal; replace first K faces with original RAW FP32 C before GEO_C',
       precision='BF16 network autocast; FP32 weights, residual addition, scaling and output; SDPA math; TF32 off; strict deterministic',
       compatibility=['Exact official RMSNorm class extracted to avoid unused fairscale imports',
         'Three exact official sincos helpers extracted to avoid unused megfile IO imports',
         'SelfAttention backend selector=fp32 enables SDPA; does not change outer BF16 autocast',
         'No padding N faces, mask=None; avoids inverted historical masked-SDPA code path',
         'FP32 residual sum instead of historical BF16 in-place sum; explicit local precision adaptation'],
       budget=dict(raw_outputs=48,denoiser_forwards=48,training_updates=0,generator_forwards=0,new_noise=0),
       inference_worker_policy='One already-authorized real output first; measure peaks and remaining job duration before dispatch choice')
    reg_path=OUT/'denoise_registration.json'
    if reg_path.exists() and read(reg_path)!=reg:raise ValueError('Denoise registration changed on resume')
    if not reg_path.exists():write(reg_path,reg)
    regh=sha(reg_path);prior=[]
    lp=OUT/'denoise_ledger.jsonl'
    if lp.exists():prior=[json.loads(x) for x in lp.read_text().splitlines() if x]
    attempts={(x['label'],x['request_id']) for x in prior if x['event']=='DENOISER_FORWARD_STARTED'}
    torch.cuda.reset_peak_memory_stats();model=model.to('cuda');torch.cuda.synchronize()
    on_device=nv();check_headroom(on_device);write(OUT/'resources_model_resident.json',on_device)
    before_buffer_keys=set(dict(model.named_buffers()))
    calls=0;phase=time.perf_counter();observed=[]
    for r in reg['raw_records']:
        label,rid=r['label'],r['request_id'];K,N=int(r['K']),int(r['N']);dest=OUT/'denoised'/label/rid
        receipt=dest/'receipt.json';outpath=dest/'denoised.npz'
        expected_id=dict(label=label,request_id=rid,raw_sha256=r['raw_sha256'],denoise_registration_sha256=regh,checkpoint_sha256=EXPECTED)
        if sha(r['raw_path'])!=r['raw_sha256']:raise ValueError('RAW hash changed')
        if receipt.exists():
            saved=read(receipt)
            if saved['status']!='PASS' or any(saved[k]!=v for k,v in expected_id.items()) or sha(outpath)!=saved['output_sha256']:raise ValueError('Cannot resume invalid completed denoise')
            continue
        if (label,rid) in attempts:raise RuntimeError('Prior started forward lacks valid completed receipt; no silent rerun')
        if len(attempts)>=48:raise RuntimeError('Denoiser forward budget exhausted')
        dest.mkdir(parents=True,exist_ok=False)
        with np.load(r['raw_path'],allow_pickle=False) as a:raw=a['raw'].reshape(N,3,3).copy();C=a['C'].copy()
        if raw[:K].tobytes()!=C.tobytes():raise ValueError('C identity mismatch')
        x=torch.from_numpy(raw.reshape(1,N,9)).to('cuda')*SCALE
        cond=torch.tensor([N],device='cuda',dtype=torch.long)
        rng_cpu=torch.get_rng_state().clone();rng_gpu=torch.cuda.get_rng_state().clone()
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
        event('DENOISER_FORWARD_STARTED',**expected_id,N=N,K=K);attempts.add((label,rid))
        try:
            with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                z=model.pe_nerf(x.reshape(1,3*N,3));residual=model.decode(z,cond,mask=None)
            reconstruction=(residual.float()+x.float())/SCALE
            proposed=reconstruction.reshape(N,3,3).cpu().numpy().copy()
            torch.cuda.synchronize();elapsed=time.perf_counter()-tick;calls+=1
            if proposed.dtype!=np.float32 or not np.isfinite(proposed).all():raise ValueError('Nonfinite or non-FP32 denoised coordinates')
            if not torch.equal(rng_cpu,torch.get_rng_state()) or not torch.equal(rng_gpu,torch.cuda.get_rng_state()):raise ValueError('Denoiser forward consumed RNG')
            output=proposed.copy();output[:K]=raw[:K]
            delta=output.astype(np.float64)-raw.astype(np.float64)
            distances=np.linalg.norm(delta,axis=-1);c_move=np.linalg.norm(proposed[:K].astype(float)-raw[:K].astype(float),axis=-1)
            with outpath.open('xb') as f:np.savez_compressed(f,output=output,C=C,official_unconstrained_proposal=proposed,
                 displacement_from_RAW=delta,known_faces=np.asarray(K),input_faces=np.asarray(N),coordinate_scale=np.asarray(SCALE))
            with np.load(outpath,allow_pickle=False) as saved:
                if saved['output'].tobytes()!=output.tobytes() or saved['output'][:K].tobytes()!=C.tobytes():raise ValueError('Denoise roundtrip/C failed')
            vram=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  scope='One real denoiser forward with resident FP32 model, input, PE and output; no optimizer')
            proof=dict(status='PASS',**expected_id,output_sha256=sha(outpath),N=N,K=K,seconds=elapsed,
                  max_free_displacement_model=float(distances[K:].max()),mean_free_displacement_model=float(distances[K:].mean()),
                  C_proposal_max_displacement_model=float(c_move.max()),C_proposal_mean_displacement_model=float(c_move.mean()),
                  C_original_bits_restored=True,output_FP32=True,no_rng_consumed=True,residual_dtype=str(residual.dtype),
                  model_parameters_FP32=True,vram=vram)
            write(receipt,proof);event('DENOISER_FORWARD_COMPLETE',**expected_id,seconds=elapsed,**vram);observed.append(elapsed)
            if calls==1:
                first=nv();check_headroom(first)
                write(OUT/'first_denoise_measurement.json',dict(resources=first,result=proof,actual_GPU_workers=1,
                  dispatch_reason='Measured first real output; remaining 47 small forwards complete faster than duplicating the large official checkpoint process/load. Keep one owned model; CPU scoring uses two workers later.'))
            del x,z,residual,reconstruction,cond
        except BaseException as e:
            write(dest/'failure.json',dict(status='STOPPED',**expected_id,error=repr(e),forward_attempt_consumed=True))
            event('DENOISER_FAILED',**expected_id,error=repr(e));raise
    torch.cuda.synchronize();after=nv();write(OUT/'resources_after_denoise.json',after)
    if (ASSET.stat().st_size,ASSET.stat().st_mtime_ns)!=checkpoint_stat:raise ValueError('Checkpoint file changed')
    completed=list((OUT/'denoised').glob('*/*/receipt.json'))
    if len(completed)!=48 or any(read(p)['status']!='PASS' for p in completed):raise ValueError('Incomplete denoise outputs')
    summary=dict(status='PASS',outputs=48,new_forwards_this_invocation=calls,total_forward_attempts=len(attempts),
       GPU_workers=1,model_parameters=num_params,checkpoint_sha256=EXPECTED,checkpoint_unchanged_stat=True,
       forward_phase_wall_seconds=time.perf_counter()-phase,total_wall_seconds=time.perf_counter()-start,
       sum_forward_seconds=sum(observed),generator_forwards=0,training_updates=0,new_noise=0,
       peak_allocated_bytes=max(read(p)['vram']['peak_allocated_bytes'] for p in completed),
       peak_reserved_bytes=max(read(p)['vram']['peak_reserved_bytes'] for p in completed))
    write(OUT/'denoise_completion.json',summary);print(json.dumps(summary));event('DENOISE_COMPLETE',**summary)
if __name__=='__main__':run()