"""Bounded loss-screen execution. No import starts a model, OT, or training."""
from .common import *
import argparse,sys,subprocess,traceback,difflib
from .distribution_losses import work_counters

def freeze_sources(r,out):
 if 'source_snapshot' in r:raise RuntimeError('Source freeze already registered')
 files=set((REPO/'native_t1').glob('*.py'))|set((REPO/'models').glob('*.py'))|set(PACKAGE.glob('*.py'))
 files|={REPO/'utils/ot_utils.py',REPO/'utils/common_utils.py'}
 files|=set((REPO/'experiments/mf_h1_local0').glob('*metrics.py'))
 files|=set((REPO/'experiments/mf_h1_local0').glob('*evaluation.py'))
 files|={REPO/'experiments/mf_h1_local0/guided_native_geometry_audit.py'}
 files={p for p in files if p.is_file() and p.name!='report.py'}
 folder=Path(out)/'source';folder.mkdir()
 manifest=[]
 for p in sorted(files):
  rel=p.relative_to(REPO);dest=folder/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
  manifest.append(dict(original_path=str(p),snapshot_path=str(dest),sha256=file_sha256(p)))
 r['source_snapshot']=dict(files=manifest,sha256=seal(manifest),report_presentation_excluded=True)
 r['source_freeze']=dict(status='FROZEN',sha256=r['source_snapshot']['sha256'])
 r['CPU_fixtures']['orchestration_prior_checks']=dict(
  evaluation_scalar_fixture_checks=6,sampling_array_count_selection_fixture_checks=10,
  status='PASS',repeat_executions=0,model_calls=0,OT_calls=0,autograd_calls=0,
  note='Evaluation fixture preceded the additional frozen-sampling-state guard; that guard was reviewed statically')
 actual_diff(out);write(r,out)

def actual_diff(out):
 lines=[]
 for p in sorted(PACKAGE.glob('*')):
  if not p.is_file() or p.suffix not in ('.py','.json'):continue
  rel=p.relative_to(REPO).as_posix()
  chunk=difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+rel)
  for line in chunk:lines.append(line if line.endswith('\n') else line+'\n\\ No newline at end of file\n')
 (Path(out)/'code.diff').write_text(''.join(lines),encoding='utf-8')

def accounting(r):
 return dict(model_forward_attempts=sum(v.get('forward_attempts',0) for v in [r['counts']['calibration'],r['counts']['wiring'],*r['counts']['training'].values()]),
   note='Sampling counts recorded separately by native clamped sampler; CPU fixtures remain separately itemized',
   training_updates=sum(x['completed_updates'] for x in r['arms'].values()),
   distribution_process=work_counters())

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--stage',choices=['all','preflight','train','development','confirmation','report'],default='all');a=parser.parse_args()
 out=Path(a.out).resolve()
 if REPO not in out.parents or out.parent!=REPO/'experiments/mf_h1_local0/outputs' or not out.name.startswith('loss_screen_'):raise ValueError('Isolated new output path required')
 if a.stage in ('all','preflight'):
  from .registration import register
  r=register(out);freeze_sources(r,out)
 else:
  r=json.loads((out/'results.json').read_text())
  from .training import source_guard
  source_guard(r)
  if seal(r['protocol'])!=r['protocol_sha256'] or seal(json.loads((out/'protocol.json').read_text()))!=r['protocol_sha256']:
   raise RuntimeError('Frozen protocol differs on stage restart')
 try:
  if a.stage in ('all','preflight'):
   from .calibration import preflight
   preflight(r,out)
  if a.stage in ('all','train'):
   from .training import train_all
   train_all(r,out)
   if r['arms']['L0']['status']!='PASS':raise RuntimeError('No complete fresh L0; formal comparison blocked')
  if a.stage in ('all','development'):
   from .sampling import sample_phase
   from .evaluation import evaluate_phase,select_candidates
   sample_phase(r,out,'development');evaluate_phase(r,out,'development');select_candidates(r,out);write(r,out)
  if a.stage in ('all','confirmation'):
   from .sampling import sample_phase,replay_once
   from .evaluation import evaluate_phase
   sample_phase(r,out,'confirmation');evaluate_phase(r,out,'confirmation');replay_once(r,out);write(r,out)
  if a.stage in ('all','report'):
   from .report import write_report
   if any(r.get('sampling_phases',{}).get(phase,{}).get('status') not in ('PASS','PASS_WITH_FAILED_CANDIDATES') for phase in ('development','confirmation')):
    raise RuntimeError('Cannot report completed execution before both fixed sampling phases')
   if set(r.get('summaries',{}))!={'development','confirmation'} or r.get('selection',{}).get('status')!='FROZEN' or r.get('sampling_replay',{}).get('status')!='PASS':
    raise RuntimeError('Completed scoring, frozen selection and one replay are required')
   r['status']='PASS_EXECUTION';r['stopped_at_registered_budget']=True
   write_report(r,out)
 except Exception as exc:
  r['status']='INTERRUPTED_OR_FAILED';r.setdefault('execution_errors',[]).append(dict(stage=a.stage,error=repr(exc),traceback=traceback.format_exc()))
  event('execution_failure',stage=a.stage,error=repr(exc));raise
 finally:
  r['actual_accounting']=accounting(r);write(r,out);actual_diff(out)
 if a.stage=='all':event('execution_stopped',updates=r['training_total_updates'],status=r['status'])

if __name__=='__main__':main()
