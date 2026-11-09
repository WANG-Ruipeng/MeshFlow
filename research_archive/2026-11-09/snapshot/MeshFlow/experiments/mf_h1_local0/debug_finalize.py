"""Finalize zero-training evidence; file checks only, no torch/model execution."""
from pathlib import Path
import argparse,difflib,hashlib,json,subprocess


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def run(output):
 out=Path(output).resolve();project=Path(__file__).resolve().parent;repo=project.parents[1]
 dest=out/'debug_evidence.json'
 if dest.exists():raise FileExistsError(dest)
 stages={}
 names=['stage_a_original','stage_a_continue','stage_b_pool_isolation','stage_c_fp32_reference','stage_c_bf16_deterministic','stage_c_fresh_process','stage_d_learning','stage_e_cache']
 for name in names:
  stages[name]=json.loads((out/(name+'.json')).read_text(encoding='utf-8-sig'))
  if name!='stage_a_original' and not stages[name].get('status','').startswith('PASS'):raise RuntimeError('Required evidence did not pass: '+name)
 before=json.loads((out/'asset_before.json').read_text())
 after={p:dict(bytes=Path(p).stat().st_size,sha256=sha(p)) for p in before['files_before']}
 if after!=before['files_before']:raise RuntimeError('Old checkpoint file changed')
 old_manifest=json.loads((project/'outputs/run_20260928_01/artifact_manifest_v2.json').read_text())
 old_verified=[]
 for row in old_manifest['files']:
  if row['path'].startswith('outputs/run_20260928_01/'):
   if sha(project/row['path'])!=row['sha256']:raise RuntimeError('Historical evidence changed: '+row['path'])
   old_verified.append(row['path'])
 git=subprocess.run(['git','-c','core.autocrlf=true','diff','--exit-code','--'],cwd=repo,capture_output=True,text=True)
 if git.returncode:raise RuntimeError('Tracked official source changed')
 total={key:0 for key in ['official_forwards','complete_rollouts','condition_encodes','guide_scores','quality_scores','completion_scores','optimizer_updates','backbone_updates','phase3_updates']}
 per_stage={}
 for name,record in stages.items():
  counts=record.get('counts',{})
  per_stage[name]=counts
  for key in total:total[key]+=int(counts.get(key,0))
 if any(total[k] for k in ['optimizer_updates','backbone_updates','phase3_updates']):raise RuntimeError('Zero update policy violated')
 total['complete_sampling_nfe']=50*total['complete_rollouts']
 total['intentional_interrupted_official_forward_calls']=1
 total['official_forwards_completed']=total['official_forwards']-1
 total['performance_warmup_rollouts']=stages['stage_e_cache']['counts']['performance_warmup_rollouts']
 total['performance_timed_rollouts']=stages['stage_e_cache']['counts']['performance_timed_rollouts']
 total['score_function_evaluations']=total['guide_scores']+total['quality_scores']+total['completion_scores']
 patch=[]
 changed=[]
 for name in ['adapters.py','official_wrapper.py','sample.py']:
  a=(out/'source_before'/name).read_text(encoding='utf-8-sig');b=(project/name).read_text(encoding='utf-8-sig')
  if a!=b:
   path='experiments/mf_h1_local0/'+name
   patch.append('diff --git a/'+path+' b/'+path+'\n')
   patch.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+path,tofile='b/'+path))
   changed.append(name)
 for source in sorted(project.glob('debug_*.py')):
  rel=str(source.relative_to(repo))
  result=subprocess.run(['git','-c','core.autocrlf=false','diff','--no-index','--','/dev/null',rel],cwd=repo,capture_output=True)
  if result.returncode not in [0,1]:raise RuntimeError(result.stderr.decode('utf-8'))
  patch.append(result.stdout.decode('utf-8'));changed.append(source.name)
 patch_file=out/'code.diff';patch_file.write_text(''.join(patch),encoding='utf-8')
 check=subprocess.run(['git','-c','core.autocrlf=false','apply','--reverse','--check','--whitespace=nowarn',str(patch_file)],cwd=repo,capture_output=True,text=True)
 if check.returncode:raise RuntimeError(check.stderr)
 evidence=dict(status='COMPLETED_ZERO_UPDATES',scope='Implementation, numerical repeatability, existing-weight response, and inference cache only',
  official_checkpoint_and_old_adapter_files=dict(before=before['files_before'],after=after,unchanged=True),
  official_tracked_source_unchanged=True,historical_evidence_files_verified=old_verified,
  stage_counts=per_stage,total_counts=total,stages=stages,
  cache_validation_detail=json.loads((out/'stage_e_cache_validation.json').read_text()),
  cache_timing_detail=json.loads((out/'stage_e_cache_timing.json').read_text()),
  isolated_pooling_counts=dict(sum_scatter=stages['stage_b_pool_isolation']['isolated_sum_scatter_calls'],count_scatter=stages['stage_b_pool_isolation']['isolated_count_scatter_calls']),
  stable_gate=json.loads((out/'stable_gate.json').read_text()),
  code_diff=dict(path='code.diff',sha256=sha(patch_file),reverse_applicability_check='PASS_CHECK_ONLY',files=changed),
  arrays=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.glob('*.npz'))],
  limitations=['No optimization, weight reset, new training or Phase 3','Historical BF16-trained weights remain unchanged; deterministic inference does not repair earlier training','Four diagnostic parents and one adapter training seed do not support significance or architecture advantage','Original linspace-vs-constant-dt failure is separate and historical evidence remains unchanged','Inference cache only; training autograd through cached geometry NOT_VALIDATED'])
 dest.write_text(json.dumps(evidence,indent=2),encoding='utf-8')
 print(json.dumps(dict(status=evidence['status'],total_counts=total,historical_files_verified=len(old_verified),old_checkpoints_verified=len(after),patch_check='PASS',arrays_bytes=sum(x['bytes'] for x in evidence['arrays'])),ensure_ascii=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
