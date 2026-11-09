"""Bounded zero-update C relation diagnostics; no optimizer or sampler path."""
from native_t1.runtime import configure_stable_runtime
from pathlib import Path
import argparse, gc, hashlib, json, platform, subprocess, time, traceback
import numpy as np
import torch
import yaml
from models.equidit import DiT
from models.utils import get_embedder
from native_t1.model import NativeInpaintingModel, native_execution
from native_t1.data import native_collate, model_inputs, array_hash
from native_t1.artifacts import atomic_json, file_sha256, state_sha256
from native_t1.training import masked_fm_loss
from .t1_interface_relations import load_source_relations, prepare_relations, relation_sums, endpoint_from_velocity

ROOT=Path(__file__).resolve().parents[2]
PROJECT=Path(__file__).resolve().parent
EXPECTED={
 'A500':'43e56e107b0c6d28766aebd8cf8a0a764d8d52ef62ba2ca6e756d881634f876b',
 'C500':'8a40651bcf54aebb2c48cd19a2433fb4e0679370c0da6e79e916cc727f067168',
 'A250':'aea34110ae1f75c108298a366e8b2668c28aa09ad8fc7d683890f5caa415e305',
 'C250':'a7b558451a3e733a5173f7a7a7a180df7358340418e4fe28b11d35f05a0f1ec2'}


def seal(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(r,out):atomic_json(out/'results.json',r)
def event(name,**kw):print(json.dumps(dict(event=name,**kw),allow_nan=False),flush=True)
def record(path):
 p=Path(path).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha256(p))
def release():gc.collect();torch.cuda.empty_cache()


def discover_source():
 matches=[]
 for report in (PROJECT/'outputs').rglob('THREE_ARM_INTERFACE_REPORT.md'):
  result=report.with_name('results.json')
  if not result.is_file():continue
  r=json.loads(result.read_text(encoding='utf-8-sig'))
  try:
   if r['protocol']['schema']!='t1_interface_3arm_v1':continue
   if any(r['arms'][key[0]]['checkpoints'].get(key[1:],{}).get('sha256')!=sha for key,sha in EXPECTED.items() if key.endswith('500')):continue
   if any(r['arms'][key[0]]['checkpoints'][key[1:]]['sha256']!=sha for key,sha in EXPECTED.items() if key.endswith('250') and key[1:] in r['arms'][key[0]]['checkpoints']):continue
   if not all(Path(r['arms'][a]['checkpoints']['500']['path']).is_file() for a in 'AC'):continue
  except (KeyError,TypeError):continue
  matches.append((report.parent,r))
 if len(matches)!=1:raise RuntimeError(f'Require unique report/checkpoint identity match, found{len(matches)}')
 return matches[0]


def protected_metadata(out):
 paths=[p for p in (PROJECT/'outputs').rglob('*') if p.is_file() and not p.is_relative_to(out)]
 paths += [p for p in PROJECT.glob('*.py') if not p.name.startswith('c_relation_')]
 paths += list((ROOT/'native_t1').glob('*.py'))+list((ROOT/'models').glob('*.py'))+[ROOT/'utils/ot_utils.py',ROOT/'configs/snet/base-120m-ot-v-chair.yaml']
 paths += list((PROJECT/'assets').rglob('*.npz'))
 official=Path('/mnt/c/Users/warpwang/Downloads/last.pt')
 if official.exists():paths.append(official)
 return {str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(set(paths))}


def protocol():
 return dict(schema='c_relation_zero_update_v1',models=EXPECTED,
  teacher=dict(parents=[0,1],K=[4,8,12],CPU_base_seeds=[19601,19602],times=[.1,.3,.5,.7,.9],explicit_time_squares=[.81,.49,.25,.09,.01],
   derivation='For each base seed create original NativeTrainingStream(Native_correct, seed=base). Call _sample once in P0K4,P0K8,P0K12,P1K4,P1K8,P1K12 order. Native noise RNG=base+10000, permutations=base+40000. Original sampled logit-normal t is unused. Replace with five fixed FP32 t; original free pre_OT coordinates, OT changes only Z, target final x2 after OT. Twelve original OT calls total.',
   paired_cases=12,states=60,endpoint_forwards=120,midpoint_K12_forwards=40,gradient_forwards=20,midpoint_missing='Skip, never reconstruct'),
  budget=dict(model_forward_cap=184,planned_model_forwards=180,parameter_gradient_backwards_cap=60,optimizer_updates=0,complete_generations=0,extra_model_gates_planned=0),
  runtime=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8',strict_deterministic=True,warn_only=False,TF32=False,SDPA='math',forward='BF16',parameters_states='FP32',eval=True,dropout=False),
  losses=dict(FM='Original free coordinate mean',FC='Original equal interface source-group mean, group corner mean',FF='Original equal shared source-group unordered-pair MSE',weights=[1,.1,.1],diagnostic_normalization='Single batch1 state; historical effective8 total coordinate/group denominators audited separately'),
  interpretation='Saved free-generation geometry has no GT corner correspondence. Teacher and local gradient evidence cannot establish trajectory causality. Five fixed t are not a logit-normal expectation.',
  stop='Zero updates, zero complete new samples. One subsequent experiment recommendation only, not executed.')


def initialize(out):
 if out.exists():raise FileExistsError(out)
 source,saved=discover_source()
 out.mkdir(parents=True)
 p=protocol()
 r=dict(status='INITIALIZING',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol=p,protocol_sha256=seal(p),source_run=str(source),
  counts=dict(model_forward_attempts=0,model_forward_returns=0,forward_by_stage={},parameter_gradient_attempts=0,parameter_gradient_returns=0,
   optimizer_updates=0,complete_generations=0,geometry_guidance_calls=0,geometry_repairs=0,OT_calls=0,output_gradient_analytic_evaluations=0,
   formula_autograd_calls=0,formula_finite_difference_evaluations=0),teacher_rows=[],gradient_rows=[],models={},gates={},timing={},
  active_stage='assets',assets={},protection=dict(before=protected_metadata(out)),
  AGENTS='No applicable AGENTS.md found by repository and ancestor searches',
  git_status_before=subprocess.run(['git','status','--short'],cwd=ROOT,capture_output=True,text=True).stdout,
  environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda))
 write(r,out);tick=time.perf_counter()
 r['assets']['source_report']=record(source/'THREE_ARM_INTERFACE_REPORT.md')
 r['assets']['source_results']=record(source/'results.json')
 r['assets']['old_T1_metadata_only']=saved['protocol']['initial']
 r['assets']['parent_ids']=saved['assets']['parent_ids']
 r['assets']['training_protocol']=saved['protocol']['training']
 r['assets']['inputs']=record(saved['relations']['inputs']['path'])
 if r['assets']['inputs']['sha256']!=saved['relations']['inputs']['sha256']:raise RuntimeError('Input identity mismatch')
 r['assets']['source_manifest']=record(Path(saved['relations']['inputs']['path']).with_name('results.json'))
 r['assets']['config']=record(saved['arms']['A']['load']['config'])
 if r['assets']['config']['sha256']!=saved['arms']['A']['load']['config_sha256']:raise RuntimeError('Architecture config changed')
 r['assets']['checkpoints']={}
 for key,sha in EXPECTED.items():
  entry=saved['arms'][key[0]]['checkpoints'].get(key[1:]);path=Path(entry['path']) if entry else source/key[0]/f'additional_step{key[1:]}_cumulative_step{500+int(key[1:])}.pt'
  if not path.is_file():
   if key.endswith('250'):r['assets']['checkpoints'][key]=dict(status='NOT_AVAILABLE',path=str(path));continue
   raise FileNotFoundError(path)
  actual=record(path)
  if actual['sha256']!=sha:raise RuntimeError('Checkpoint file identity mismatch:'+key)
  r['assets']['checkpoints'][key]=dict(**actual,status='PASS',state_sha256=entry['state_sha256'],backbone_state_sha256=entry['backbone_state_sha256'],additional_updates=int(key[1:]))
 r['assets']['old_execution_sources']={str(p):record(p) for p in sorted(list((ROOT/'native_t1').glob('*.py'))+[PROJECT/'t1_interface_relations.py',PROJECT/'t1_interface_train.py',ROOT/'utils/ot_utils.py'])}
 r['environment']['GPU']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
 r['timing']['assets_seconds']=time.perf_counter()-tick
 r.update(status='ASSETS_PASS',active_stage='assets_complete');write(r,out);event('assets_pass',source=str(source),out=str(out))


def prepare(r,out):
 if r['status']!='ASSETS_PASS':raise RuntimeError('No automatic prepare retry')
 from .c_relation_teacher import build_teacher_bank
 from .c_relation_saved import analyze_saved
 from .c_relation_math import run_cpu_tests
 tick=time.perf_counter();r['active_stage']='CPU_formula_gates';write(r,out)
 original_grad=torch.autograd.grad
 def counted_formula_grad(*args,**kwargs):
  r['counts']['formula_autograd_calls']+=1;write(r,out)
  answer=original_grad(*args,**kwargs)
  r['counts']['formula_autograd_returns']=r['counts'].get('formula_autograd_returns',0)+1
  return answer
 torch.autograd.grad=counted_formula_grad
 try:gates=run_cpu_tests()
 finally:torch.autograd.grad=original_grad;write(r,out)
 r['gates']['CPU_formula']=gates
 r['counts']['formula_finite_difference_evaluations']=gates.get('counts',{}).get('finite_difference_loss_evaluations',0)
 if gates.get('counts',{}).get('CPU_autograd_grad_calls')!=r['counts']['formula_autograd_calls']:raise RuntimeError('Formula autograd ledger mismatch')
 if gates.get('status')!='PASS':raise RuntimeError('CPU formula gate failed')
 relations=load_source_relations(r['assets']['inputs']['path'],expected_sha256=r['assets']['inputs']['sha256'])
 r['relations']=relations.audit
 r['active_stage']='fixed_bank_registration';write(r,out)
 from utils import ot_utils
 original_OT=ot_utils.optimal_sum_numpy
 def counted_OT(*args,**kwargs):
  r['counts']['OT_calls']+=1;write(r,out)
  answer=original_OT(*args,**kwargs)
  r['counts']['OT_returns']=r['counts'].get('OT_returns',0)+1
  return answer
 ot_utils.optimal_sum_numpy=counted_OT
 try:samples,bank=build_teacher_bank(Path(r['source_run']),out,relations)
 finally:ot_utils.optimal_sum_numpy=original_OT;write(r,out)
 r['bank']=bank
 if r['counts']['OT_calls']!=12 or r['counts'].get('OT_returns')!=12:raise RuntimeError('OT actual count differs')
 if len(samples)!=60:raise RuntimeError('Fixed teacher state count differs')
 r['active_stage']='existing_generation_statistics';write(r,out)
 r['saved_generation']=analyze_saved(Path(r['source_run']))
 r['timing']['preparation_seconds']=time.perf_counter()-tick
 r['execution_sources']={str(p):record(p) for p in PROJECT.glob('c_relation_*.py') if p.name!='c_relation_report.py'}
 r.update(status='PREPARED',active_stage='prepared');write(r,out);event('prepared',states=len(samples),counts=r['counts'])


def load_branch(r,key):
 rec=r['assets']['checkpoints'][key];tick=time.perf_counter()
 configure_stable_runtime()
 if file_sha256(rec['path'])!=rec['sha256']:raise RuntimeError('Checkpoint changed since registration')
 config=yaml.safe_load(Path(r['assets']['config']['path']).read_text(encoding='utf-8'))
 mc=dict(config['model'])
 if mc.pop('model_type')!='equidit':raise RuntimeError('Wrong model type')
 expected=dict(version=3,hidden_dim=768,num_layers=12,num_heads=12,max_length=800,face_bin=20,pe_freq=20,use_qknorm=True,use_rmsnorm=True,use_coord_encoding=True,use_dit_like_pe=False,face_cond=True)
 if any(mc.get(k)!=v for k,v in expected.items()) or config['transport']['prediction']!='velocity':raise RuntimeError('Architecture differs')
 payload=torch.load(rec['path'],map_location='cpu',weights_only=True,mmap=True)
 if payload.get('schema')!='t1_interface_branch_v1' or payload.get('arm')!=key[0] or payload.get('additional_updates')!=int(key[1:]):raise RuntimeError('Checkpoint payload identity differs')
 with torch.device('meta'):model=NativeInpaintingModel(DiT(**mc),trainable=False)
 model.backbone.x_embedder.embed_fn,_=get_embedder(mc['pe_freq'],input_dims=3)
 incompatible=model.load_state_dict(payload['model'],strict=True,assign=True);del payload
 model.set_trainable(False).eval().to('cuda')
 before=state_sha256(model)
 if before!=rec['state_sha256'] or state_sha256(model.backbone)!=rec['backbone_state_sha256']:raise RuntimeError('Strict tensor hash differs')
 def pre(module,args):
  c=r['counts'];stage=r['active_stage']
  if c['model_forward_attempts']>=184:raise RuntimeError('Hard forward budget exhausted')
  c['model_forward_attempts']+=1;c['forward_by_stage'][stage]=c['forward_by_stage'].get(stage,0)+1
 def post(module,args,value):r['counts']['model_forward_returns']+=1
 model.register_forward_pre_hook(pre);model.register_forward_hook(post)
 r['models'][key]=dict(status='LOADED',file_sha256=rec['sha256'],state_before=before,parameters=sum(p.numel() for p in model.parameters()),strict_load=True,missing_keys=list(incompatible.missing_keys),unexpected_keys=list(incompatible.unexpected_keys),load_seconds=time.perf_counter()-tick,old_T1_tensor_load=False,official_EMA_load=False)
 return model


def state_info(sample,index):
 return dict(state_id=sample.get('state_id',f'state{index:02d}'),bank_index=index,parent=int(sample['parent_index']),K=int(sample['K']),
  noise_index=int(sample['noise_index']),noise_seed=int(sample.get('noise_seed',19601+int(sample['noise_index']))),time_index=int(sample['time_index']),registered_t=[.1,.3,.5,.7,.9][int(sample['time_index'])],t=float(np.asarray(sample['t']).item()))


def actual_arrays(batch):return {k:v.detach().cpu().numpy() for k,v in batch.items() if torch.is_tensor(v)}

def input_digest(batch,labels):
 h=hashlib.sha256()
 for key in ('xt','t','y','valid_mask','known_mask','context','u','x0','x1'):
  h.update(key.encode());h.update(array_hash(batch[key].detach().cpu().numpy()).encode())
 h.update(array_hash(labels['corner_source_vertex_ids']).encode())
 return h.hexdigest()


def loss_terms(batch,velocity,prepared):
 fm=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'])
 endpoint=endpoint_from_velocity(batch['xt'],velocity,batch['t'],batch['known_mask'],batch['context'])
 terms=relation_sums(endpoint,prepared)
 fc=terms['FC_sum']/max(terms['FC_count'],1);ff=terms['FF_sum']/max(terms['FF_count'],1)
 return (fm,fc,ff),endpoint


def get_param_grads(loss,parameters,r,out,retain):
 if r['counts']['parameter_gradient_attempts']>=60:raise RuntimeError('Hard parameter-gradient budget exhausted')
 r['counts']['parameter_gradient_attempts']+=1;write(r,out)
 values=torch.autograd.grad(loss,parameters,retain_graph=retain,create_graph=False,allow_unused=True)
 r['counts']['parameter_gradient_returns']+=1
 return values


def measure(r,out):
 if r['status']!='PREPARED':raise RuntimeError('No automatic model-stage retry')
 for p,rec in r['execution_sources'].items():
  if file_sha256(p)!=rec['sha256']:raise RuntimeError('Diagnostic execution source changed since prepare')
 from .c_relation_teacher import load_teacher_bank
 from .c_relation_math import analyze_teacher,output_gradient_diagnostics,gradient_norm_dot_statistics
 relations=load_source_relations(r['assets']['inputs']['path'],expected_sha256=r['assets']['inputs']['sha256'])
 samples,_=load_teacher_bank(out/'teacher_bank.npz',relations=relations,expected_sha256=r['bank']['bank_file']['sha256'])
 labels=[relations.prepare_sample_labels(s) for s in samples]
 paired_hashes={};tick_all=time.perf_counter()
 for key in ('A500','C500','A250','C250'):
  if r['assets']['checkpoints'][key]['status']!='PASS':continue
  model=load_branch(r,key);reference={};start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
  selection=[i for i,s in enumerate(samples) if key.endswith('500') or s['K']==12]
  try:
   r['active_stage']='endpoint_teacher' if key.endswith('500') else 'midpoint_teacher'
   with torch.no_grad():
    for i in selection:
     s=samples[i];label=labels[i];batch=native_collate([s],'cuda');pack=prepare_relations(label,'cuda')
     digest=input_digest(batch,label)
     if i in paired_hashes and paired_hashes[i]!=digest:raise RuntimeError('A/C teacher state or labels differ')
     paired_hashes[i]=digest;write(r,out)
     with native_execution(model,coordinates=batch['xt']) as runtime:
      velocity=model(*model_inputs(batch)).float();terms,endpoint=loss_terms(batch,velocity,pack)
     values={n:float(v) for n,v in zip(('FM','FC','FF'),terms)}
     analysis=analyze_teacher(actual_arrays(batch),velocity.detach().cpu().numpy(),label,legacy_losses=values)
     vhash=array_hash(velocity.detach().cpu().numpy());reference[i]=vhash
     r['teacher_rows'].append(dict(model=key[0],additional_step=int(key[1:]),phase='endpoint' if key.endswith('500') else 'midpoint',state=state_info(s,i),losses=values,analysis=analysis,input_sha256=digest,velocity_sha256=vhash))
     r['models'][key]['runtime_teacher']=runtime
     if analysis['legacy_loss_reconciliation']['status']!='PASS':raise RuntimeError('Original FC/FF/FM definition reconciliation failed; preserve original C interpretation')
     tolerance=1e-12+1e-10*max(values.values())
     if any(abs(analysis['identities'][name])>tolerance for name in ('max_abs_endpoint_minus_Position_minus_Spread','max_abs_epsilon_endpoint_minus_Position_minus_Spread','FC_minus_Position_plus_Spread','FF_minus_global_partition_contributions')):raise RuntimeError('Group decomposition identity failed')
     del batch,pack,velocity,terms,endpoint
     if len(r['teacher_rows'])%10==0:write(r,out);event('teacher',model=key,total_teacher=len(r['teacher_rows']),forwards=r['counts']['model_forward_attempts'])
   if key.endswith('500'):
    model.set_trainable(True).eval();names=[n for n,p in model.named_parameters()];parameters=tuple(model.parameters())
    groups={'coordinate_embedding':[j for j,n in enumerate(names) if n.startswith('backbone.x_embedder.')],
     'first_QKV':[j for j,n in enumerate(names) if n.startswith('backbone.layers.0.attn.qkv_proj.')],
     'final_output':[j for j,n in enumerate(names) if n.startswith('backbone.final_layer.linear.')]}
    subset=[i for i,s in enumerate(samples) if s['K']==12 and int(s['noise_index'])==0]
    if len(subset)!=10:raise RuntimeError('Require exactly ten gradient states per endpoint')
    r['active_stage']='parameter_gradient_diagnosis'
    for i in subset:
     s=samples[i];label=labels[i];batch=native_collate([s],'cuda');pack=prepare_relations(label,'cuda')
     if input_digest(batch,label)!=paired_hashes[i]:raise RuntimeError('Gradient input differs from teacher')
     arrays=actual_arrays(batch);write(r,out);gtick=time.perf_counter()
     with native_execution(model,coordinates=batch['xt']) as runtime:
      velocity=model(*model_inputs(batch)).float();terms,endpoint=loss_terms(batch,velocity,pack)
      v_numpy=velocity.detach().cpu().numpy();vhash=array_hash(v_numpy)
      if vhash!=reference[i]:raise RuntimeError('Same teacher forward changed with parameter gradients enabled')
      grads=[get_param_grads(loss,parameters,r,out,j<2) for j,loss in enumerate(terms)]
     values={n:float(v.detach()) for n,v in zip(('FM','FC','FF'),terms)}
     stat={'full_model':gradient_norm_dot_statistics(*grads,names=names)}
     for name,indices in groups.items():stat[name]=gradient_norm_dot_statistics(*[tuple(g[j] for j in indices) for g in grads],names=[names[j] for j in indices])
     r['counts']['output_gradient_analytic_evaluations']+=1
     output=output_gradient_diagnostics(arrays,v_numpy,label)
     r['gradient_rows'].append(dict(model=key[0],additional_step=500,state=state_info(s,i),losses=values,parameter_space=stat,output_space=output,
      input_sha256=paired_hashes[i],teacher_velocity_bitwise_equal=True,runtime=runtime,seconds=time.perf_counter()-gtick))
     if any(p.grad is not None for p in parameters):raise RuntimeError('autograd.grad unexpectedly populated parameter.grad')
     if any(item['significant_negative'] for family in ('actual_endpoint_analytic','ideal_linear_error_reference') for item in output[family]['PSD_output_error_check'].values()):raise RuntimeError('Significantly negative output-space PSD check; inspect mapping/scale before causal interpretation')
     del grads,stat,output,batch,pack,velocity,terms,endpoint,v_numpy
     write(r,out);event('gradient',model=key,state=state_info(s,i),backwards=r['counts']['parameter_gradient_returns'])
    del parameters;model.set_trainable(False).eval()
   after=state_sha256(model)
   r['models'][key].update(status='PASS',state_after=after,parameters_unchanged=(after==r['models'][key]['state_before']),diagnostics_seconds=time.perf_counter()-start,peak_cuda_allocated=torch.cuda.max_memory_allocated())
   if after!=r['models'][key]['state_before']:raise RuntimeError('Read-only diagnostic changed model state')
  finally:
   model.zero_grad(set_to_none=True);del model;release();write(r,out)
 r['timing']['model_diagnostics_seconds']=time.perf_counter()-tick_all
 expected=120+20+20*sum(r['assets']['checkpoints'][key]['status']=='PASS' for key in ('A250','C250'))
 if r['counts']['model_forward_attempts']!=expected or r['counts']['parameter_gradient_returns']!=60:raise RuntimeError('Actual diagnostic budget differs')
 r.update(status='MEASURED',active_stage='measurements_complete');write(r,out);event('measured',counts=r['counts'])


def finalize(r,out):
 if r['status']!='MEASURED':raise RuntimeError('All fixed diagnostic states required')
 from .c_relation_report import summarize_results,render_report
 r['summary']=summarize_results(r)
 now=protected_metadata(out);changed=[p for p,v in r['protection']['before'].items() if now.get(p)!=v]
 r['protection']['after']=dict(status='PASS' if not changed else 'FAIL',files=len(r['protection']['before']),changed=changed)
 critical={**r['assets']['old_execution_sources'],**r['execution_sources']}
 critical.update({v['path']:v for v in r['assets']['checkpoints'].values() if v['status']=='PASS'})
 for key in ('source_results','source_report','inputs','source_manifest','config'):critical[r['assets'][key]['path']]=r['assets'][key]
 mismatch=[p for p,v in critical.items() if file_sha256(p)!=v['sha256']]
 r['protection']['critical_hashes_after']=dict(status='PASS' if not mismatch else 'FAIL',checked=len(critical),changed=mismatch)
 r['git_status_after']=subprocess.run(['git','status','--short'],cwd=ROOT,capture_output=True,text=True).stdout
 if changed or mismatch or r['git_status_before']!=r['git_status_after']:raise RuntimeError('Protected assets or tracked worktree status changed')
 r.update(status='PASS',active_stage='FINISHED',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
 dest=out/'C_RELATION_DIAGNOSIS_REPORT.md'
 if dest.exists():raise FileExistsError(dest)
 dest.write_text(render_report(r),encoding='utf-8')
 r['delivery']=dict(report=record(dest),diagnostic_entrypoint=str(Path(__file__).resolve()),report_source=record(PROJECT/'c_relation_report.py'))
 chunks=[]
 for p in sorted(PROJECT.glob('c_relation_*.py')):
  diff=subprocess.run(['git','diff','--no-index','--','/dev/null',str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True)
  if diff.returncode not in (0,1):raise RuntimeError(diff.stderr.decode())
  chunks.append(diff.stdout)
 patch=out/'code.diff';patch.write_bytes(b''.join(chunks))
 check=subprocess.run(['git','apply','--reverse','--check',str(patch)],cwd=ROOT,capture_output=True)
 if check.returncode:raise RuntimeError(check.stderr.decode())
 r['delivery']['code_diff']=dict(**record(patch),reverse_apply_check='PASS')
 write(r,out);event('complete',counts=r['counts'],out=str(out))


def execute(out,stage):
 if stage=='init':return initialize(out)
 r=json.loads((out/'results.json').read_text())
 try:
  if seal(r['protocol'])!=r['protocol_sha256']:raise RuntimeError('Registered protocol changed')
  configure_stable_runtime()
  if stage=='prepare':prepare(r,out)
  elif stage=='measure':measure(r,out)
  elif stage=='report':finalize(r,out)
 except BaseException as error:
  r.update(status='STOPPED',failure=dict(stage=stage,error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));write(r,out);event('stopped',error=repr(error),counts=r['counts']);raise


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--stage',choices=['init','prepare','measure','report'],required=True)
 args=parser.parse_args();execute(args.out.resolve(),args.stage)
