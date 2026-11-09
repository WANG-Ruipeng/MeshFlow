"""Zero-update repeatability diagnostics using only archived official assets."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,contextlib,dataclasses,hashlib,itertools,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from .adapters import ControlAdapter
from .official_wrapper import load_official,OfficialWrapper,state_sha256
from .prompt_dataset import collate_prompts
from .sample import euler_sample
from .evaluate import guidance_metrics

PROJECT=Path(__file__).resolve().parent
REPO=PROJECT.parents[1]
OLD=PROJECT/'outputs/run_20260928_01'
PROMPT_KEYS=('vertices','arity','primitive_mask','points','point_primitive','point_mask')


def tensor_hash(x):
 x=x.detach().cpu().contiguous()
 return hashlib.sha256(str((tuple(x.shape),x.dtype)).encode()+x.view(torch.uint8).numpy().tobytes()).hexdigest()
def metadata(x):return dict(shape=list(x.shape),dtype=str(x.dtype),device=str(x.device),sha256=tensor_hash(x))
def diff(a,b):
 a,b=a.detach().cpu(),b.detach().cpu();d=a.double()-b.double()
 return dict(bitwise_equal=torch.equal(a,b),max_abs=float(d.abs().max()) if d.numel() else 0.,rms=float(d.square().mean().sqrt()) if d.numel() else 0.)
def comparisons(xs):return [dict(pair=[i,j],**diff(xs[i],xs[j])) for i,j in itertools.combinations(range(len(xs)),2)]
def encoded_meta(e):return {f.name:metadata(getattr(e,f.name)) for f in dataclasses.fields(e)}


def ban_optimizer_steps():
 def forbidden(*a,**k):raise RuntimeError('ZERO_TRAINING_POLICY: optimizer.step is forbidden')
 for value in vars(torch.optim).values():
  if isinstance(value,type) and issubclass(value,torch.optim.Optimizer):value.step=forbidden


def load_case(parent=0,kind='mixed_2edges_1face',seed=4101,device='cuda'):
 pm=json.loads((OLD/'prompt_manifest.json').read_text())
 record=next(r for r in pm['records'] if r['parent_index']==parent and r['kind']==kind)
 arrays=dict(np.load(record['path']))
 saved=np.load(OLD/'raw'/f'Q_center_parent{parent}_{kind}_seed{seed}_correct.npz')
 z=torch.from_numpy(saved['initial_noise'].copy()).unsqueeze(0).to(device)
 prompts=collate_prompts([{k:arrays[k] for k in PROMPT_KEYS}],device)
 return dict(record=record,arrays=arrays,z=z,y=torch.tensor([512],device=device),mask=torch.ones((1,512),dtype=torch.bool,device=device),prompts=prompts)


def load_adapter(arm):
 saved=torch.load(OLD/arm/'checkpoints/adapter_step0200.pt',map_location='cpu',weights_only=True)
 with torch.device('meta'):adapter=ControlAdapter(arm)
 adapter.load_state_dict(saved['adapter'],strict=True,assign=True)
 return adapter.to('cuda').requires_grad_(False).eval()


class FixedEncoded:
 def __init__(self,wrapper,encoded):self.wrapper,self.encoded=wrapper,encoded
 @property
 def counts(self):return self.wrapper.counts
 def __call__(self,*args,**kwargs):return self.wrapper(*args,encoded=self.encoded,**kwargs)


class Harness:
 def __init__(self,out,label,precision='bf16',deterministic=False):
  ban_optimizer_steps();torch.set_num_threads(8)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.backends.cudnn.benchmark=False
  torch.use_deterministic_algorithms(deterministic,warn_only=False)
  self.out=Path(out);self.label=label;self.precision=precision
  self.dest=self.out/(label+'.json')
  if self.dest.exists():raise FileExistsError(self.dest)
  self.counts=dict(official_forwards=0,complete_rollouts=0,guide_scores=0,condition_encodes=0,single_step_repeats=0,optimizer_updates=0,backbone_updates=0,phase3_updates=0)
  self.report=dict(status='RUNNING',label=label,counts=self.counts,precision=precision,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,environment={k:os.environ.get(k) for k in ['CUBLAS_WORKSPACE_CONFIG','CUDA_LAUNCH_BLOCKING','NVIDIA_TF32_OVERRIDE','PYTHONHASHSEED']},groups={},traces={},encoder_probes={})
  self.base,audit=load_official('/mnt/c/Users/warpwang/Downloads/last.pt',REPO/'configs/snet/base-120m-ot-v-chair.yaml','cuda')
  self.report['strict_load']=audit
  self.report['base_hash_before']=state_sha256(self.base)
  self.base.register_forward_pre_hook(lambda m,a:self.counts.__setitem__('official_forwards',self.counts['official_forwards']+1))
  import models.attention as attention
  self.report['external_flash_available']=attention.FLASH_ATTN_AVAILABLE
  self.dispatch={'sdpa':0,'external_flash':0};self.original_dispatch={}
  for name,key in [('_attention_sdpa','sdpa'),('_attention_flash','external_flash')]:
   original=getattr(attention,name);self.original_dispatch[name]=original
   def counted(*a,_original=original,_key=key,**k):self.dispatch[_key]+=1;return _original(*a,**k)
   setattr(attention,name,counted)
  self.case=load_case();self.arrays={};self.wrappers=[];self.adapters={}
  self.report['inputs']={k:metadata(self.case[k]) for k in ['z','y','mask']}
  self.report['inputs']['prompts']={k:metadata(v) for k,v in self.case['prompts'].items()}
  self.report['inputs']['checks']=metadata(torch.from_numpy(self.case['arrays']['check_points']))
  self.report['parent']=self.case['record'];self.report['time_grid']=[float(torch.tensor(k/50,dtype=torch.float32)) for k in range(51)];self.report['dt']=dict(python_float=1/50,actual_fp32=float(torch.tensor(1/50,dtype=torch.float32)),count=50)
  self.report['dropout']=[dict(name=n,type=type(m).__name__,p=getattr(m,'dropout',getattr(m,'p',None)),training=m.training) for n,m in self.base.named_modules() if isinstance(m,torch.nn.Dropout) or hasattr(m,'dropout')]
  self.input_before=self.input_hashes();self.started=time.perf_counter();self.write()
 def autocast(self):return torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.precision=='bf16')
 def input_hashes(self):return {k:tensor_hash(v) for k,v in dict(z=self.case['z'],y=self.case['y'],mask=self.case['mask'],**self.case['prompts']).items()}
 def write(self):
  self.report['actual_attention_dispatch']=self.dispatch
  self.dest.write_text(json.dumps(self.report,indent=2),encoding='utf-8')
 def wrapper(self,arm):
  for w in self.wrappers:w.close()
  self.wrappers=[]
  if arm not in self.adapters:self.adapters[arm]=load_adapter(arm)
  w=OfficialWrapper(self.base,self.adapters[arm]).eval();self.wrappers.append(w)
  orig=w.encode
  def encode(p):self.counts['condition_encodes']+=1;return orig(p)
  w.encode=encode
  self.report.setdefault('adapter_states',{})[arm]=dict(sha256=state_sha256(w.adapter),parameters=sum(p.numel() for p in w.adapter.parameters()),training=w.adapter.training,backbone_training=w.backbone.training)
  return w
 def sample_group(self,name,model,repeats=3,prompts=None,encoded=None,trace=True):
  outputs=[];traces=[];scores=[];stats=[]
  encoded_before=encoded_meta(encoded) if encoded is not None else None
  for i in range(repeats):
   xs=[];vs=[];ts=[]
   def hook(m,a,o):xs.append(a[0].detach().cpu().clone());ts.append(a[1].detach().cpu().clone());vs.append(o.detach().cpu().clone())
   handle=self.base.register_forward_hook(hook) if trace else None
   try:result,stat=euler_sample(model,self.case['z'].clone(),self.case['y'],self.case['mask'],prompts,precision=self.precision)
   finally:
    if handle:handle.remove()
   self.counts['complete_rollouts']+=1
   value=result.detach().cpu();outputs.append(value);stats.append(stat);traces.append((xs,vs,ts))
   score=guidance_metrics(value[0].numpy(),self.case['arrays']['check_points'],self.case['record']['bbox_diagonal'])
   self.counts['guide_scores']+=1;scores.append(score['guide_rms_bbox_pct'])
   self.arrays[name+'_'+str(i)]=value.numpy()
   self.write();print(json.dumps(dict(stage=self.label,group=name,repeat=i,guide=scores[-1],counts=self.counts)),flush=True)
  pairs=comparisons(outputs)
  self.report['groups'][name]=dict(pairwise=pairs,guide_scores=scores,guide_pairwise_absolute_difference=[abs(scores[i]-scores[j]) for i,j in itertools.combinations(range(repeats),2)],sample_stats=stats,input_unchanged=self.input_hashes()==self.input_before,encoded_unchanged=(encoded_meta(encoded)==encoded_before if encoded is not None else None))
  if trace:
   for j in range(1,repeats):
    xa,va,ta=traces[0];xb,vb,tb=traces[j]
    first_input=next((k for k in range(50) if not torch.equal(xa[k],xb[k])),None)
    first_velocity=next((k for k in range(50) if not torch.equal(va[k],vb[k])),None)
    record=dict(compare=[0,j],first_different_input_step=first_input,first_different_velocity_step=first_velocity,time_bitwise_equal=all(torch.equal(a,b) for a,b in zip(ta,tb)))
    if first_velocity is not None:
     k=first_velocity;record.update(input_diff=diff(xa[k],xb[k]),velocity_diff=diff(va[k],vb[k]),updated_xt_diff=diff(xa[k]+va[k].float()*(1/50),xb[k]+vb[k].float()*(1/50)))
     for side,xx,vv in [('a',xa[k],va[k]),('b',xb[k],vb[k])]:self.arrays[f'{name}_first_velocity_pair0_{j}_{side}_xt']=xx.numpy();self.arrays[f'{name}_first_velocity_pair0_{j}_{side}_v']=vv.float().numpy()
    self.report['traces'][name+f'_0_{j}']=record
  self.write();return outputs
 def encoder_probe(self,wrapper,repeats=10):
  stages=[];original=torch.Tensor.scatter_add
  features=[];sums=[];counts=[]
  handle=wrapper.adapter.condition_encoder.register_forward_hook(lambda m,a,o:features.append(o.detach().cpu().clone()))
  def traced(t,dim,index,src,*a,**k):
   result=original(t,dim,index,src,*a,**k)
   if src.ndim==3 and src.shape[-1]==128:sums.append(result.detach().cpu().clone())
   elif src.ndim==2:counts.append(result.detach().cpu().clone())
   return result
  torch.Tensor.scatter_add=traced
  try:
   with self.autocast():
    for i in range(repeats):
     e=wrapper.encode(self.case['prompts']);row=dict(features=features[-1],sums=sums[-1],counts=counts[-1],means=e.tokens.detach().cpu().clone(),encoded=e.tokens.detach().cpu().clone())
     for j,l in enumerate(wrapper.adapter.layers):row[f'K{j}']=l.key_projection(e.tokens).detach().cpu().clone();row[f'V{j}']=l.value_projection(e.tokens).detach().cpu().clone()
     stages.append(row)
  finally:torch.Tensor.scatter_add=original;handle.remove()
  summary={k:dict(metadata=metadata(stages[0][k]),pairwise=comparisons([row[k] for row in stages])) for k in stages[0]}
  summary['first_different_stage']=next((k for k in ['features','sums','counts','means','encoded','K0','V0'] if any(not r['bitwise_equal'] for r in summary[k]['pairwise'])),None)
  self.report['encoder_probes'][wrapper.adapter.arm]=summary
  for i,row in enumerate(stages):
   for k in ['features','sums','counts','means']:self.arrays[f'encoder_{wrapper.adapter.arm}_{i}_{k}']=row[k].float().numpy()
  self.write();return stages
 def finish(self):
  for w in self.wrappers:w.close()
  self.report['base_hash_after']=state_sha256(self.base);self.report['base_unchanged']=self.report['base_hash_after']==self.report['base_hash_before']
  self.report['adapters_after']={k:state_sha256(a) for k,a in self.adapters.items()}
  self.report['adapters_unchanged']=all(self.report['adapter_states'][k]['sha256']==value for k,value in self.report['adapters_after'].items())
  self.report['inputs_unchanged']=self.input_hashes()==self.input_before
  self.report['remaining_injection_hooks']=[len(self.base.layers[k-1]._forward_hooks) for k in (3,6,9,12)]
  self.report['elapsed_seconds']=time.perf_counter()-self.started
  self.report['status']='PASS_EXECUTION' if self.report['base_unchanged'] and self.report['inputs_unchanged'] and self.report['adapters_unchanged'] else 'FAIL_INTEGRITY'
  np.savez_compressed(self.out/(self.label+'_arrays.npz'),**self.arrays)
  self.write();print(json.dumps(dict(stage=self.label,status=self.report['status'],counts=self.counts,encoder_first_difference={k:v['first_different_stage'] for k,v in self.report['encoder_probes'].items()},groups={k:all(r['bitwise_equal'] for r in v['pairwise']) for k,v in self.report['groups'].items()})),flush=True)


def run_a(out,continuing=False):
 h=Harness(out,'stage_a_continue' if continuing else 'stage_a_original',precision='bf16',deterministic=False)
 try:
  with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
   if not continuing:h.sample_group('base',h.base)
   for arm in ['Q_center','G_closest']:
    w=h.wrapper(arm);h.sample_group(arm+'_fresh',w,repeats=2 if continuing and arm=='Q_center' else 3,prompts=h.case['prompts'])
    with h.autocast():encoded=w.encode(h.case['prompts'])
    h.sample_group(arm+'_fixed',FixedEncoded(w,encoded),encoded=encoded)
    with h.autocast():
     vs=[w(h.case['z'].clone(),torch.zeros(1,device='cuda'),h.case['y'],mask=h.case['mask'],encoded=encoded).cpu() for _ in range(3)]
    h.counts['single_step_repeats']+=3
    h.report.setdefault('fixed_single_step',{})[arm]=comparisons(vs)
    h.encoder_probe(w,10)
    h.report.setdefault('hook_context_checks',{})[arm]=dict(context_is_none=w._context is None,injection_hook_counts=[len(h.base.layers[k-1]._forward_hooks) for k in (3,6,9,12)])
  h.finish()
 except BaseException as e:
  h.report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc());np.savez_compressed(h.out/(h.label+'_partial_arrays.npz'),**h.arrays);h.write();raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--continue-a',action='store_true');a=p.parse_args();run_a(a.output,a.continue_a)
