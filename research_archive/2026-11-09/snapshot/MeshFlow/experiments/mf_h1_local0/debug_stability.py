"""Strict zero-training references and same/new-process stability gates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from .debug_repeatability import Harness,load_case,diff,metadata


class DisabledControl:
 def __init__(self,wrapper):self.wrapper=wrapper
 @property
 def counts(self):return self.wrapper.counts
 def __call__(self,*a,**k):return self.wrapper(*a,control_scale=0.,**k)


def set_case(h,kind):
 h.case=load_case(kind=kind);h.input_before=h.input_hashes()
 h.report.setdefault('case_inputs',{})[kind]={k:metadata(v) for k,v in h.case['prompts'].items()}


def run(out,precision,label,coverage=False,reload_from=None):
 h=Harness(out,label,precision=precision,deterministic=True)
 try:
  with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
   base=h.sample_group('base',h.base,repeats=1 if reload_from else 2)
   if reload_from:
    previous=np.load(Path(out)/(reload_from+'_arrays.npz'));h.report['fresh_process_comparisons']={'base':diff(base[0],torch.from_numpy(previous['base_0']))}
   for arm in ['Q_center','G_closest']:
    w=h.wrapper(arm)
    for kind in (['mixed_2edges_1face','single_edge','single_face'] if coverage or reload_from else ['mixed_2edges_1face']):
     set_case(h,kind)
     values=h.sample_group(arm+'_'+kind,w,repeats=1 if reload_from else 2,prompts=h.case['prompts'])
     if reload_from:h.report['fresh_process_comparisons'][arm+'_'+kind]=diff(values[0],torch.from_numpy(previous[arm+'_'+kind+'_0']))
    set_case(h,'mixed_2edges_1face')
    if not reload_from:
     h.encoder_probe(w,10)
     off=h.sample_group(arm+'_off',DisabledControl(w),repeats=1)
     h.report.setdefault('control_off_identity',{})[arm]=diff(base[0],off[0])
   h.report['bitwise_stable']=all(x['bitwise_equal'] for v in h.report['groups'].values() for x in v['pairwise'])
   h.report['bitwise_stable'] &= all(x['bitwise_equal'] for x in h.report.get('control_off_identity',{}).values())
   h.report['bitwise_stable'] &= all(x['bitwise_equal'] for x in h.report.get('fresh_process_comparisons',{}).values())
   h.report['bitwise_stable'] &= all(v['first_different_stage'] is None for v in h.report['encoder_probes'].values())
  h.finish()
  if not h.report['bitwise_stable']:raise RuntimeError('Stability gate failed; no effect comparison permitted')
  if reload_from:
   primary=json.loads((Path(out)/(reload_from+'.json')).read_text())
   if primary['status']!='PASS_EXECUTION' or not primary['bitwise_stable']:raise RuntimeError('Primary same-process gate did not pass')
   if precision!='bf16' or not coverage:raise ValueError('Unexpected final gate configuration')
   gate=dict(status='PASS',gate=dict(stage_d_allowed=True,bitwise_repeatability=True,same_process=True,fresh_process=True,all_prompt_types=True),settings=dict(precision=precision,deterministic_algorithms=True,sdpa_backend='math',tf32=False,cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG']),evidence=[reload_from+'.json',label+'.json'],semantic_note='Only strict deterministic algorithm selection differs from original BF16 runtime; weights/data/time grid/architecture/loss unchanged. Floating reduction order and resulting numbers may differ from old nondeterministic runs.')
   dest=Path(out)/'stable_gate.json'
   if dest.exists():raise FileExistsError(dest)
   dest.write_text(json.dumps(gate,indent=2),encoding='utf-8')
 except BaseException as e:
  h.report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc());h.write();raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--precision',choices=['fp32','bf16'],required=True);p.add_argument('--label',required=True);p.add_argument('--coverage',action='store_true');p.add_argument('--reload-from');a=p.parse_args();run(a.output,a.precision,a.label,a.coverage,a.reload_from)
