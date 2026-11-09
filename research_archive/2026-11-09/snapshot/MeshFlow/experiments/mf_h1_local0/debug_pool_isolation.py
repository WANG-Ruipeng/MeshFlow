"""Isolate the observed pooling difference on captured REAL step200 features."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .debug_repeatability import load_case,tensor_hash,metadata,comparisons,ban_optimizer_steps


def run(out):
 ban_optimizer_steps();out=Path(out);dest=out/'stage_b_pool_isolation.json'
 if dest.exists():raise FileExistsError(dest)
 evidence=json.loads((out/'stage_a_continue.json').read_text())
 arrays=np.load(out/'stage_a_continue_arrays.npz');case=load_case()
 group=case['prompts']['point_primitive'];index=group.unsqueeze(-1).expand(-1,-1,128)
 report=dict(status='RUNNING',official_forwards=0,complete_rollouts=0,optimizer_updates=0,experiments={},isolated_sum_scatter_calls=0,isolated_count_scatter_calls=0)
 for arm in ['Q_center','G_closest']:
  features=torch.from_numpy(arrays[f'encoder_{arm}_0_features']).to('cuda',dtype=torch.bfloat16)
  assert tensor_hash(features)==evidence['encoder_probes'][arm]['features']['metadata']['sha256']
  before=tensor_hash(features)
  for mode in ['original_bf16','determinism_only','fp32_accumulation_only']:
   torch.use_deterministic_algorithms(mode=='determinism_only',warn_only=False)
   sums=[];counts=[];means=[]
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for i in range(20):
     if mode=='fp32_accumulation_only':
      # Only sum accumulation is changed. Cast back before the original BF16
      # division; counts/logits/softmax dtypes are deliberately unaffected.
      value=torch.zeros((1,3,128),dtype=torch.float32,device='cuda').scatter_add(1,index,features.float()).to(features.dtype)
     else:value=features.new_zeros((1,3,128)).scatter_add(1,index,features)
     count=features.new_zeros((1,3)).scatter_add(1,group,case['prompts']['point_mask'].to(features.dtype))
     sums.append(value.cpu());counts.append(count.cpu());means.append((value/count.clamp_min(1).unsqueeze(-1)).cpu())
     report['isolated_sum_scatter_calls']+=1;report['isolated_count_scatter_calls']+=1
   report['experiments'][arm+'/'+mode]=dict(features=metadata(features),features_unchanged=tensor_hash(features)==before,sum_pairs=comparisons(sums),count_pairs=comparisons(counts),mean_pairs=comparisons(means),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),sum_dtype=str(sums[0].dtype),mean_dtype=str(means[0].dtype))
 report['status']='PASS_EXECUTION';dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({k:{'sum_all_bitwise':all(x['bitwise_equal'] for x in v['sum_pairs']),'sum_max_abs':max(x['max_abs'] for x in v['sum_pairs']),'counts_all_bitwise':all(x['bitwise_equal'] for x in v['count_pairs'])} for k,v in report['experiments'].items()}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
