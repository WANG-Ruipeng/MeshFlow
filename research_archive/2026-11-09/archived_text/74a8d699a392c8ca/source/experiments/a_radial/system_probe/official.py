"""Independent full MeshFlow CUDA correctness and effective-update profiling.

No trained checkpoint from this module is a scientific initialization or result.
Official source and the original 50,000-update configuration are read-only.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from ..official_contract import audit_loader
from ..official_train import build_model,ema_update
from ..io_utils import (new_dir,write_json,atomic_torch,file_hash,array_hash,
    rng_state,restore_rng,seed_all)
from ..layout_student.train import model_sha256

PARAMETERS=130249731
STATE_KEYS=['model','ema','optimizer','scheduler','rng','sampler_rng','step','ema_updates',
            'parent_exposures','trace','identity','config','scope']


def require(ok,message):
    if not ok:raise AssertionError(message)


def exact_same(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        require(isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),path)
    elif isinstance(a,np.ndarray):require(isinstance(b,np.ndarray) and np.array_equal(a,b),path)
    elif isinstance(a,dict):
        require(isinstance(b,dict) and a.keys()==b.keys(),path+'.keys')
        for key in a:exact_same(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(list,tuple)):
        require(type(a)==type(b) and len(a)==len(b),path+'.length')
        for i,(x,y) in enumerate(zip(a,b)):exact_same(x,y,path+'.'+str(i))
    else:require(a==b,path)


def configure(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    require(os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8',':16:8'),'Unexpected CUBLAS_WORKSPACE_CONFIG')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.set_float32_matmul_precision('highest');torch.set_num_threads(1);seed_all(seed)


def validate_config(cfg):
    expected={'model_type':'equidit','hidden_dim':768,'num_heads':12,'max_length':800,'num_layers':12,
        'gradient_checkpointing':True,'use_coord_encoding':True,'version':3,'pe_freq':20,
        'mixed_precision':'bf16','use_dit_like_pe':False,'face_cond':True,'face_bin':20,
        'use_qknorm':True,'use_rmsnorm':True}
    for key,value in expected.items():require(cfg['model'].get(key)==value,'Official model setting changed: '+key)
    for key,value in [('updates',50000),('microbatch',4),('accumulation_steps',4),('effective_batch',16),('ema_decay',.9999)]:
        require(cfg['closed16_runner'][key]==value,'Original formal budget/runner changed: '+key)
    require(cfg['optimizer']=={'lr':1e-4,'beta2':.95,'max_grad_norm':2.},'Official optimizer settings changed')
    require(cfg['transport']=={'path_type':'Linear','prediction':'velocity','use_lognorm':True,'use_jit':False},
            'Official transport settings changed')


def attention_mask_probe(attention,device,n=800,valid=512,dtype=torch.float32):
    """Real attention weights; the unmasked negative control prevents vacuous PASS."""
    x=torch.randn(2,n,attention.input_dim,device=device,dtype=dtype)
    mask=torch.arange(n,device=device)[None].expand(2,-1)<valid
    changed=x.clone();changed[:,valid:]=torch.randn_like(changed[:,valid:])*13.+7.
    with torch.no_grad():
        a=attention(x,mask);b=attention(changed,mask)
        unmasked_a=attention(x,None);unmasked_b=attention(changed,None)
    error=float((a[:,:valid].float()-b[:,:valid].float()).abs().max())
    negative=float((unmasked_a[:,:valid].float()-unmasked_b[:,:valid].float()).abs().max())
    nonzero=float(a[:,:valid].float().abs().max())
    require(error==0.,'Masked attention valid outputs changed with padding')
    require(negative>1e-5 and nonzero>1e-5,'Attention test is vacuous or perturbation ineffective')
    return {'pass':True,'masked_valid_max_abs_delta':error,'unmasked_negative_control_max_abs_delta':negative,
        'valid_output_max_abs':nonzero,'slots':n,'valid_slots':valid,'same_original_attention_module':True}


def all_parent_probe(model,transport,ds,device='cuda'):
    from datasets.mesh_dataset import collate_fn
    saved=rng_state();was=model.training;rows=[]
    try:
        seed_all(883021);model.eval()
        with torch.no_grad():
            for i in range(len(ds)):
                item=ds[i];b=collate_fn([item],max_seq_length=800)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=transport.training_losses(model,b['tokens'].to(device),b['noise'].to(device),
                        {'y':b['num_faces'].to(device),'mask':b['masks'].to(device)})['loss']
                require(bool(torch.isfinite(loss)),'Nonfinite all-parent fit probe')
                rows.append({'uid':str(np.asarray(item['uid']).item()),'loss':float(loss),
                    'tokens_sha256':array_hash(b['tokens'].numpy()),'noise_sha256':array_hash(b['noise'].numpy())})
    finally:
        model.train(was);restore_rng(saved)
    exact_same(saved,rng_state(),'probe_rng_preservation')
    require(model.training==was,'Probe changed train/eval mode')
    return {'pass':True,'rng_exactly_preserved':True,'training_mode_preserved':True,
        'noise_time_seed':883021,'rows':rows,'mean_loss':float(np.mean([x['loss'] for x in rows])),
        'scope':'training_support_only_not_convergence_or_generalization'}


class Session:
    def __init__(self,cfg,ds,seed,identity):
        from flow_matching import create_transport
        configure(seed);self.cfg=cfg;self.ds=ds;self.seed=seed;self.identity=identity
        self.model=build_model(cfg).cuda();require(sum(p.numel() for p in self.model.parameters())==PARAMETERS,'Wrong complete model')
        self.ema=deepcopy(self.model).eval().requires_grad_(False);ema_update(self.ema,self.model,0.)
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=0.,foreach=False,fused=False)
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.opt,lambda _:1.)
        self.transport=create_transport('Linear','velocity',use_lognorm=True,use_jit=False)
        self.gen=torch.Generator().manual_seed(853000+seed)
        self.step_count=0;self.ema_updates=0;self.exposures=np.zeros(16,dtype=np.int64);self.trace=[]
        self.uids=[str(np.asarray(x['uid']).item()) for x in ds.data]
        require(len(self.uids)==len(set(self.uids))==16,'Official sampler requires all16 distinct parents')
        self.model.train()

    def update(self):
        from datasets.mesh_dataset import collate_fn
        torch.cuda.synchronize();started=time.perf_counter();order=torch.randperm(16,generator=self.gen).tolist()
        self.opt.zero_grad(set_to_none=True);lossval=0.;batches=[];loader_seconds=0.
        for k in range(4):
            ids=order[k*4:(k+1)*4];self.exposures[ids]+=1
            before=time.perf_counter();b=collate_fn([self.ds[i] for i in ids],max_seq_length=800)
            loader_seconds+=time.perf_counter()-before
            require(tuple(b['tokens'].shape)==(4,512,9) and bool(b['masks'].all()),'Real loader512 batch/mask changed')
            require(bool((b['num_faces']==512).all()),'Wrong official face condition')
            batches.append({'indices':ids,'uids':[self.uids[i] for i in ids],
                'tokens_sha256':array_hash(b['tokens'].numpy()),'noise_sha256':array_hash(b['noise'].numpy())})
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=self.transport.training_losses(self.model,b['tokens'].cuda(),b['noise'].cuda(),
                    {'y':b['num_faces'].cuda(),'mask':b['masks'].cuda()})['loss']/4
            require(bool(torch.isfinite(loss)),'Official CUDA loss is nonfinite')
            loss.backward();lossval+=float(loss.detach())
        require(all(p.grad is not None for p in self.model.parameters()),'Missing full-model gradient')
        finite=torch.stack([torch.isfinite(p.grad).all() for p in self.model.parameters()]).all()
        require(bool(finite),'Nonfinite full-model gradient')
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),2.)
        require(bool(torch.isfinite(norm)) and float(norm)>0,'Zero or nonfinite gradient norm')
        self.opt.step();self.scheduler.step();ema_update(self.ema,self.model,.9999)
        self.ema_updates+=1;self.step_count+=1
        row={'step':self.step_count,'loss':lossval,'gradient_norm_before_clip':float(norm),
            'all_gradients_present_and_finite':True,'microbatches':batches,'effective_batch':16,
            'ema_updates':self.ema_updates,'scheduler_updates':self.scheduler.last_epoch,'lr':self.opt.param_groups[0]['lr']}
        self.trace.append(row);torch.cuda.synchronize()
        timing={'step':self.step_count,'seconds_full_effective_update':time.perf_counter()-started,
            'seconds_loader_collate_OT':loader_seconds,'cpu_GPU_transfer_and_sync_included':True,
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved()}
        return row,timing

    def state(self):
        return {'model':self.model.state_dict(),'ema':self.ema.state_dict(),'optimizer':self.opt.state_dict(),
            'scheduler':self.scheduler.state_dict(),'rng':rng_state(),'sampler_rng':self.gen.get_state(),
            'step':self.step_count,'ema_updates':self.ema_updates,'parent_exposures':torch.tensor(self.exposures),
            'trace':self.trace,'identity':self.identity,'config':self.cfg,'scope':'CORRECTNESS_PROFILE_ONLY_NOT_FORMAL_INITIALIZATION'}

    def load(self,path):
        ck=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        exact_same(ck['identity'],self.identity,'resume_identity');exact_same(ck['config'],self.cfg,'resume_config')
        self.model.load_state_dict(ck['model']);self.ema.load_state_dict(ck['ema']);self.opt.load_state_dict(ck['optimizer'])
        self.scheduler.load_state_dict(ck['scheduler']);self.gen.set_state(ck['sampler_rng']);restore_rng(ck['rng'])
        self.step_count=int(ck['step']);self.ema_updates=int(ck['ema_updates'])
        self.exposures=ck['parent_exposures'].numpy().copy();self.trace=ck['trace']
        require(self.ema_updates==self.step_count==self.scheduler.last_epoch,'Saved effective-update counters disagree')
        require(np.array_equal(self.exposures,np.full(16,self.step_count)),'Saved balanced-parent exposure changed')
        del ck


def segment(cfg,ds,seed,identity,out,updates,resume=None,save_at=(),profile_operators=False,fit_probe=False):
    out=new_dir(out);init_start=time.perf_counter();session=Session(cfg,ds,seed,identity)
    if resume is not None:session.load(resume)
    torch.cuda.synchronize();initialization_seconds=time.perf_counter()-init_start
    torch.cuda.reset_peak_memory_stats();first=session.step_count;timings=[];operator_names=[];write_seconds=0.
    while session.step_count<updates:
        if profile_operators and session.step_count==0:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                row,timing=session.update()
            operator_names=sorted({e.key for e in profiler.key_averages() if 'attention' in e.key.lower()})
        else:row,timing=session.update()
        timings.append(timing)
        if session.step_count in save_at or session.step_count==updates:
            before=time.perf_counter();atomic_torch(out/f'step{session.step_count:08d}.pt',session.state())
            write_seconds+=time.perf_counter()-before
        if session.step_count%10==0 or session.step_count==updates:
            print(json.dumps({'segment':out.name,'step':session.step_count,'loss':row['loss'],**timing}),flush=True)
    checkpoint=out/f'step{updates:08d}.pt'
    report={'start_step':first,'final_step':updates,'updates_executed':updates-first,
        'checkpoint':str(checkpoint),'initialization_seconds':initialization_seconds,'checkpoint_write_seconds':write_seconds,
        'parent_uids':session.uids,'parent_exposures':session.exposures.tolist(),'ema_updates':session.ema_updates,
        'scheduler_updates':session.scheduler.last_epoch,'timings':timings,'observed_attention_operators':operator_names,
        'scope':'CORRECTNESS_PROFILE_ONLY_NOT_FORMAL_INITIALIZATION'}
    if fit_probe:
        probe_started=time.perf_counter();sampler_before=session.gen.get_state().clone()
        report['current_all_parent_probe']=all_parent_probe(session.model,session.transport,ds)
        report['EMA_all_parent_probe']=all_parent_probe(session.ema,session.transport,ds)
        require(torch.equal(sampler_before,session.gen.get_state()),'Fit probe changed batch sampler RNG')
        report['fit_probe_seconds']=time.perf_counter()-probe_started
    write_json(out/'REPORT.json',report);write_json(out/'TRACE.json',session.trace)
    del session;gc.collect();torch.cuda.empty_cache()
    return checkpoint,report


def full_mask_cfg_gate(cfg,ds,checkpoint):
    from datasets.mesh_dataset import collate_fn
    from flow_matching import create_transport
    configure(774411);ck=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    model=build_model(cfg).cuda();model.load_state_dict(ck['model']);model.eval();del ck
    require(float(model.final_layer.linear.weight.abs().sum())>0,'Zero-initialized head masks attention bugs')
    require(all(float(layer.adaLN_modulation[-1].weight.abs().sum())>0 for layer in model.layers),
            'Zero attention gates would make the full-model test vacuous')
    with torch.autocast('cuda',dtype=torch.bfloat16):attn=attention_mask_probe(model.layers[0].attn,'cuda')
    b=collate_fn([ds[0]],max_seq_length=800)
    x1=torch.zeros(1,800,9,device='cuda');x0=x1.clone()
    x1[:,:512]=b['tokens'].cuda();x0[:,:512]=b['noise'].cuda()
    x1[:,512:]=torch.randn_like(x1[:,512:]);x0[:,512:]=torch.randn_like(x0[:,512:])
    alt1=x1.clone();alt0=x0.clone();alt1[:,512:]+=19.;alt0[:,512:]-=11.
    mask=torch.arange(800,device='cuda')[None]<512;y=torch.tensor([512],device='cuda');t=torch.tensor([.37],device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        pred=model(x0,t,y,mask);other=model(alt0,t,y,mask)
    output_delta=float((pred[:,:512].float()-other[:,:512].float()).abs().max())
    require(output_delta==0 and float(pred[:,:512].abs().max())>0,'Full model padding contaminated valid output')
    transport=create_transport('Linear','velocity',use_lognorm=True,use_jit=False)
    state=rng_state()
    with torch.autocast('cuda',dtype=torch.bfloat16):first_loss=transport.training_losses(model,x1,x0,{'y':y,'mask':mask})['loss']
    restore_rng(state)
    with torch.autocast('cuda',dtype=torch.bfloat16):other_loss=transport.training_losses(model,alt1,alt0,{'y':y,'mask':mask})['loss']
    loss_delta=abs(float(first_loss)-float(other_loss));require(loss_delta==0.,'Padding changed actual official masked FM loss')
    del first_loss,other_loss
    model.zero_grad(set_to_none=True);x1=x1.detach().requires_grad_(True);x0=x0.detach().requires_grad_(True)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=transport.training_losses(model,x1,x0,{'y':y,'mask':mask})['loss']
    loss.backward()
    invalid_grad=max(float(x1.grad[:,512:].abs().max()),float(x0.grad[:,512:].abs().max()))
    valid_grad=max(float(x1.grad[:,:512].abs().max()),float(x0.grad[:,:512].abs().max()))
    require(invalid_grad==0. and valid_grad>0. and bool(torch.isfinite(x0.grad).all()) and bool(torch.isfinite(x1.grad).all()),
            'Actual masked FM input-gradient gate failed')
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'Masked full-model gradient invalid')
    model.zero_grad(set_to_none=True)
    cfg_rows=[];x=x0.detach();combined=torch.cat([x,x],0);ct=torch.cat([t,t]);cm=torch.cat([mask,mask]);cy=torch.tensor([512,801],device='cuda')
    require(model.uncond_y==801 and model.num_classes==40 and model.y_embedder.embedding_table.num_embeddings==41,
            'CFG null condition or label table mismatch')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        both=model(combined,ct,cy,cm);cond,uncond=both.chunk(2)
        for scale in [0.,1.,2.]:
            actual=model.forward_with_cfg(combined,t,y,mask,cfg_scale=scale)
            expected=uncond+scale*(cond-uncond);expected=torch.cat([expected,expected],0)
            delta=float((actual.float()-expected.float()).abs().max());require(delta==0.,'Original CFG formula/null pairing mismatch')
            cfg_rows.append({'cfg_scale':scale,'max_abs_delta':delta,'pass':True})
    probe=all_parent_probe(model,transport,ds)
    result={'pass':True,'parameter_count':sum(p.numel() for p in model.parameters()),'nonzero_trained_full_model':True,
        'attention':attn,'full_model_padding_output_max_abs_delta':output_delta,'actual_masked_loss_abs_delta':loss_delta,
        'invalid_input_gradient_max_abs':invalid_grad,'valid_input_gradient_max_abs':valid_grad,
        'full_model_masked_gradients_finite':True,'face_condition':512,'condition_class':25,'null_face_condition':801,
        'null_class':40,'cfg':cfg_rows,'all_parent_probe':probe,'source_checkpoint_step':4}
    del model,x0,x1,loss,pred,other,alt0,alt1,both,combined;gc.collect();torch.cuda.empty_cache()
    return result


def run(repo,data,out,seed=10,profile_updates=100):
    require(profile_updates==100,'This entry point has only the predeclared 100-update profiling budget')
    require(torch.cuda.is_available(),'Actual CUDA required; no CPU/tiny-model substitute')
    repo=Path(repo).resolve();data=Path(data).resolve();out=Path(out).resolve()
    cfg_path=repo/'experiments/a_radial/configs/meshflow_closed16_512.yaml'
    cfg=yaml.safe_load(cfg_path.read_text());validate_config(cfg);cfg_sha=file_hash(cfg_path)
    out=new_dir(out);events=[];started=time.perf_counter()
    def event(name,value):
        events.append({'name':name,**value});write_json(out/'CHECK_EVENTS.json',events,replace=(out/'CHECK_EVENTS.json').exists())
    try:
        from models.attention import FLASH_ATTN_AVAILABLE
        ds,loader=audit_loader(repo,data);write_json(out/'OFFICIAL_LOADER_GATE.json',loader)
        protocol={'scope':'CORRECTNESS_AND_PROFILE_ONLY_NO_SCIENTIFIC_CHECKPOINT','seed':seed,
            'formal_config_path':str(cfg_path),'formal_config_sha256':cfg_sha,'formal_config_updates_unchanged':50000,
            'dataset_manifest_sha256':file_hash(data/'DATASET.json'),'data_path_override':str(data),
            'full_parameter_count':PARAMETERS,'precision':'float32_parameters_bfloat16_autocast',
            'microbatch':4,'accumulation_steps':4,'effective_batch':16,'ema_decay':.9999,
            'correctness_continuous_updates':4,'correctness_resume_after':2,'profile_updates':100,
            'attention_mask_slots':800,'attention_mask_valid':512,'cfg_scales':[0.,1.,2.],
            'tf32':False,'deterministic_algorithms':True,'scheduler':'explicit_constant_no_LR_change',
            'AdamW_foreach':False,'AdamW_fused':False,'AdamW_weight_decay':0.,
            'flash_attention_package_available':FLASH_ATTN_AVAILABLE,
            'gpu':torch.cuda.get_device_name(0),'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,
            'source_sha256':{str(p.relative_to(repo)):file_hash(p) for p in [Path(__file__),
                repo/'experiments/a_radial/official_train.py',repo/'experiments/a_radial/official_contract.py',
                repo/'experiments/a_radial/io_utils.py',repo/'models/attention.py',repo/'models/equidit.py',
                repo/'datasets/mesh_dataset.py',repo/'utils/ot_utils.py',repo/'flow_matching.py']},
            'sampling_or_long_training_authorized_by_this_entry':False}
        write_json(out/'RUN_PROTOCOL.json',protocol);write_json(out/'FORMAL_CONFIG_SNAPSHOT.json',cfg)
        identity={'protocol_sha256':file_hash(out/'RUN_PROTOCOL.json'),'dataset_sha256':file_hash(data/'DATASET.json'),
            'formal_config_sha256':cfg_sha,'base_commit':loader['repo']['base_commit'],'seed':seed}
        whole,wr=segment(cfg,ds,seed,identity,out/'continuous4',4,save_at=(2,4),profile_operators=True)
        event('full_real_cuda_forward_backward',{'pass':True,'parameter_count':PARAMETERS,'updates':4,
            'observed_attention_operators':wr['observed_attention_operators']})
        resumed,rr=segment(cfg,ds,seed,identity,out/'resume2_to4',4,resume=whole.parent/'step00000002.pt')
        a=torch.load(whole,map_location='cpu',weights_only=True,mmap=True)
        b=torch.load(resumed,map_location='cpu',weights_only=True,mmap=True)
        for key in STATE_KEYS:exact_same(a[key],b[key],'resume.'+key)
        require(a['ema_updates']==4 and a['scheduler']['last_epoch']==4,'EMA/scheduler not once per effective update')
        require(torch.equal(a['parent_exposures'],torch.full((16,),4,dtype=torch.int64)),'Actual parent exposure mismatch')
        resume_report={'pass':True,'bitwise_equal_keys':STATE_KEYS,'continuous_updates':4,'resumed_new_updates':2,
            'EMA_updates':4,'Adam_updates_per_parameter':sorted({int(v['step']) for v in a['optimizer']['state'].values()}),
            'parent_exposures':[4]*16,'checkpoint_file_hashes':{str(p.relative_to(out)):file_hash(p) for p in [whole,resumed,whole.parent/'step00000002.pt']},
            'final_model_sha256':model_sha256(a['model']),'final_ema_sha256':model_sha256(a['ema'])}
        del a,b;gc.collect();write_json(out/'RESUME_GATE.json',resume_report);event('complete_resume',resume_report)
        mask=full_mask_cfg_gate(cfg,ds,whole);write_json(out/'MASK_CFG_PROBE_GATE.json',mask);event('mask_CFG_probe',mask)
        gate={'status':'PASS','actual_official_loader_OT_full_model_CUDA':True,'complete_resume_bitwise':True,
            'masked_attention_and_loss_and_input_gradient':True,'nonzero_attention_negative_control':True,
            'CFG_null_condition':True,'probe_preserves_rng_and_mode':True,'EMA_once_per_effective_update':True,
            'scientific_training_complete':False,'formal_initialization_must_not_use_probe_checkpoints':True}
        write_json(out/'GATE.json',gate)
        profile,pr=segment(cfg,ds,seed,identity,out/'profile100',100,fit_probe=True)
        times=np.array([x['seconds_full_effective_update'] for x in pr['timings']])
        warm=times[10:];loader_times=np.array([x['seconds_loader_collate_OT'] for x in pr['timings']])
        report={'status':'COMPLETE','scope':'PROFILE_ONLY_NOT_FIT_EVIDENCE','updates':100,'effective_batch':16,
            'microbatch':4,'accumulation_steps':4,'all_100_mean_seconds_per_update':float(times.mean()),
            'warm_updates_11_to100_mean_seconds':float(warm.mean()),'warm_median_seconds':float(np.median(warm)),
            'warm_p95_seconds':float(np.quantile(warm,.95)),'full100_training_seconds':float(times.sum()),
            'loader_OT_total_seconds':float(loader_times.sum()),'checkpoint_write_seconds':pr['checkpoint_write_seconds'],
            'initialization_seconds':pr['initialization_seconds'],
            'peak_gpu_allocated_bytes':max(x['peak_gpu_allocated_bytes'] for x in pr['timings']),
            'peak_gpu_reserved_bytes':max(x['peak_gpu_reserved_bytes'] for x in pr['timings']),
            'projected_5000_training_hours_excluding_probes_checkpoints':float(warm.mean()*5000/3600),
            'projected_50000_training_hours_excluding_probes_checkpoints':float(warm.mean()*50000/3600),
            'actual_parent_exposures_each':100,'actual_EMA_updates':100,'full_timing_rows':'profile100/REPORT.json',
            'current_all_parent_probe':pr['current_all_parent_probe'],'EMA_all_parent_probe':pr['EMA_all_parent_probe'],
            'fit_probe_seconds_separate_from_update_timing':pr['fit_probe_seconds'],'EMA_initialization_coefficient':.9999**100,
            'first10_included_in_all100_and_separately_excluded_from_warm_summary':True,
            'no_model_loading_or_checkpoint_IO_hidden_in_update_timing':True,
            'source_formal_config_unchanged':file_hash(cfg_path)==cfg_sha,'elapsed_entire_probe_seconds':time.perf_counter()-started}
        require(report['source_formal_config_unchanged'],'Original formal config changed')
        write_json(out/'PROFILE100.json',report);print(json.dumps(report,indent=2),flush=True)
        return report
    except Exception as exc:
        write_json(out/'FAILED.json',{'status':'FAILED','type':type(exc).__name__,'error':str(exc),
            'elapsed_seconds':time.perf_counter()-started,'completed_checks':[x['name'] for x in events],
            'original_sources_or_thresholds_not_modified_to_hide_failure':True})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=10);p.add_argument('--profile-updates',type=int,default=100)
    a=p.parse_args();run(a.repo,a.data,a.out,a.seed,a.profile_updates)


if __name__=='__main__':main()
