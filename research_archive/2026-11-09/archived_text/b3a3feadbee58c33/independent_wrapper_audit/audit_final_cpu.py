"""Independent CPU-only final-export and saved-residual audit. No model calls."""
import sys,json,math,collections
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/mnt/e/MeshFlow')
from r2.pilots.single_step_denoiser_distillation_v1 import common as C
from r2.pilots.single_step_denoiser_distillation_v1.model import tensor_state_sha256
torch.set_num_threads(2)
assert not torch.cuda.is_initialized()
audit=C.OUT/'independent_wrapper_audit'
done=C.read(C.OUT/'inference/completion.json')
assert done['status']=='COMPLETE' and done['student_forwards']==66 and done['teacher_instances']==0
assert len(done['rows'])==66
path=C.OUT/'checkpoints/student_inference.pt'; rec=C.read(path.with_suffix('.json'))
assert C.sha(path)==rec['sha256']==done['checkpoint_sha256']
payload=torch.load(path,map_location='cpu',weights_only=True)
state=payload['model'];statehash=tensor_state_sha256(state)
assert payload['step']==4000 and len(state)==104
assert all(v.dtype==torch.float32 for v in state.values())
assert sum(v.numel() for v in state.values())==38389251
assert statehash==payload['state_sha256']==rec['state_sha256']
assert payload['spec']['actual_num_layers']==8 and payload['spec']['actual_hidden_dim']==512 and payload['spec']['actual_num_heads']==8
load=C.read(C.OUT/'inference/load.json')
assert load['independent_student'] and not load['teacher_loaded']
assert load['spec']['independent_student_load'] and not load['spec']['teacher_checkpoint_loaded']
assert load['spec']['loaded_step']==4000
rows=[];inputs=[]
for row in done['rows']:
 sp=Path(row['student_cache_path']);tp=Path(row['teacher_cache_path']);rp=Path(row['student_receipt_path'])
 receipt=C.read(rp)
 assert C.sha(sp)==row['student_cache_sha256']==receipt['output_sha256']
 assert C.sha(rp)==row['student_receipt_sha256']
 assert not receipt['teacher_instance_used'] and receipt['model_calls']==1
 assert receipt['C_original_bits_restored'] and receipt['no_rng_consumed']
 assert receipt['student_checkpoint_sha256']==rec['sha256']
 with np.load(sp,allow_pickle=False) as z:
  x=z['input_d'].copy();s=z['proposal_S_d'].copy();sr=z['r_S_d'].copy();output=z['output'].copy();c=z['C'].copy();known=z['known'].copy()
 with np.load(tp,allow_pickle=False) as z:
  assert x.tobytes()==z['input_d'].tobytes()
  t=z['proposal_T_d'].copy();k=int(z['K'])
 assert c.tobytes()==output[known].tobytes()
 assert sr.tobytes()==np.subtract(s,x,dtype=np.float32).tobytes()
 rt=np.subtract(t,x,dtype=np.float32)[k:];rs=sr[k:];err=np.subtract(rs,rt,dtype=np.float32).astype(np.float64)
 e=float(np.sum(err*err));b=float(np.sum(rt.astype(np.float64)**2));se=float(np.sum(rs.astype(np.float64)**2));dot=float(np.sum(rt.astype(np.float64)*rs.astype(np.float64)))
 den=9*(len(x)-k);signal=b/den>1e-14
 rows.append(dict(sample_id=row['sample_id'],split=row['split'],domain=row['domain'],generator=row.get('label','SYNTHETIC'),
  N=len(x),K=k,free_scalar_count=den,error_SSE=e,teacher_energy_sum=b,student_energy_sum=se,teacher_student_dot_sum=dot,
  E=e/den,B=b/den,R=e/b if signal else None,signal_status='OK' if signal else 'LOW_SIGNAL',
  residual_norm_ratio=float(np.sqrt(se/b)) if signal else None,
  residual_cosine=float(np.clip(dot/np.sqrt(b*se),-1,1)) if signal and se>0 else None))
 inputs.append(dict(sample_id=row['sample_id'],teacher_sha256=C.sha(tp),student_sha256=C.sha(sp),receipt_sha256=C.sha(rp)))
summaries=[]
for grouping in ('domain','generator'):
 groups=collections.defaultdict(list)
 for r in rows:groups[r[grouping]].append(r)
 for group,rs in groups.items():
  den=sum(r['free_scalar_count'] for r in rs);e=sum(r['error_SSE'] for r in rs);b=sum(r['teacher_energy_sum'] for r in rs)
  se=sum(r['student_energy_sum'] for r in rs);dot=sum(r['teacher_student_dot_sum'] for r in rs);signal=b/den>1e-14
  summaries.append(dict(grouping=grouping,group=group,outputs=len(rs),free_scalar_count=den,E=e/den,B=b/den,R=e/b if signal else None,
   residual_norm_ratio=float(np.sqrt(se/b)) if signal else None,residual_cosine=float(dot/np.sqrt(b*se)) if signal and se else None,
   R_below_identity=sum(r['R'] is not None and r['R']<1 for r in rs),LOW_SIGNAL_outputs=sum(r['signal_status']=='LOW_SIGNAL' for r in rs)))
ledger=[json.loads(line) for line in (C.OUT/'gpu_calls.jsonl').read_text().splitlines() if line.strip()]
evalcalls=[r for r in ledger if r.get('phase')=='student_evaluation' and r['event']=='OP_COMPLETE']
assert len(evalcalls)==66 and all(r['kind']=='student_eval_F' for r in evalcalls)
assert {r['identity'] for r in evalcalls}=={r['sample_id'] for r in done['rows']}
assert not torch.cuda.is_initialized()
result=dict(status='PASS',scope='CPU-only final export/tensor/cache/receipt audit; no model instantiated or called',
 export=dict(step=4000,state_keys=104,parameters=38389251,file_sha256=rec['sha256'],state_sha256=statehash,independent_student_load=True,teacher_instances=0),
 actual_student_eval_calls=66,all_C_bytes_preserved=True,all_effective_residuals_FP32_exact=True,all_teacher_student_inputs_exact=True,
 source_sha256=C.sha(C.CODE/'inference.py'),inputs=inputs,summary=summaries)
assert result['source_sha256']==done['source_sha256']
C.write(audit/'final_student_and_residuals.json',result)
C.csvwrite(audit/'final_residuals_per_output.csv',rows)
C.csvwrite(audit/'final_residuals_summary.csv',summaries)
print(json.dumps(dict(status='PASS',export=result['export'],summary=summaries),indent=2))
