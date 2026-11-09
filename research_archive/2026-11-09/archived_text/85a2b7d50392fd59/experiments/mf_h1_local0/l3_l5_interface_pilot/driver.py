"""Explicit bounded pilot stages; no import starts training or generation."""
from .common import *
from ..loss_screen.training import source_guard
from ..loss_screen.distribution_losses import work_counters
import argparse,difflib,traceback

def actual_diff(out):
 lines=[]
 for p in sorted(PACKAGE.glob('*')):
  if not p.is_file() or p.suffix not in ('.py','.json'):continue
  for line in difflib.unified_diff([],p.read_text(encoding='utf-8-sig').splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+p.relative_to(REPO).as_posix()):
   lines.append(line if line.endswith('\n') else line+'\n\\ No newline at end of file\n')
 (Path(out)/'code.diff').write_text(''.join(lines),encoding='utf-8')

def freeze(r,out):
 if 'source_snapshot' in r:raise RuntimeError('Already frozen')
 files=set((REPO/'native_t1').glob('*.py'))|set((REPO/'models').glob('*.py'))|set(PACKAGE.glob('*.py'))|set((PACKAGE.parent/'loss_screen').glob('*.py'))
 files|={REPO/'utils/ot_utils.py',REPO/'utils/common_utils.py'}
 files|=set(PACKAGE.parent.glob('*metrics.py'))|set(PACKAGE.parent.glob('*evaluation.py'))|{PACKAGE.parent/'guided_native_geometry_audit.py'}
 files={p for p in files if p.is_file() and p.name!='report.py'}
 manifest=[];folder=Path(out)/'source';folder.mkdir()
 for p in sorted(files):
  dest=folder/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
  manifest.append(dict(original_path=str(p),snapshot_path=str(dest),sha256=file_sha256(p)))
 r['source_snapshot']=dict(files=manifest,sha256=seal(manifest),report_presentation_excluded=True)
 write(r,out);actual_diff(out)

def accounting(r):
 train=list(r['counts']['training'].values());cal=r['counts']['calibration'];wire=r['counts']['wiring'];teach=r['counts']['teacher'];sample=r['counts']['sampling']
 return dict(optimizer_updates=sum(x.get('optimizer_updates',0) for x in train),
  generator_forwards=sum(x.get('forward_returns',0) for x in [cal,wire,teach,*train])+sample.get('model_forward_returns',0),
  parameter_backwards=sum(x.get('backward_returns',0) for x in [wire,*train]),
  original_noise_OT_calls=sum(x.get('OT_returns',0) for x in [cal,*train]),
  velocity_leaf_autograd=sum(x.get('velocity_autograd_returns',0) for x in [cal,*train]),
  formal_generations=sample.get('formal_outputs_saved',0),replays=sample.get('replay_outputs_verified',0),
  complete_generations=sample.get('complete_rollouts',0),distribution_process=work_counters())

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--stage',required=True,choices=['register','preflight','train','teacher','sample','score','report','all']);a=parser.parse_args();out=Path(a.out).resolve()
 if out.parent!=REPO/'native_t1/runs' or not out.name.startswith('l3_l5_interface_pilot_'):raise ValueError('Isolated pilot output required')
 if a.stage in ('all','register'):
  from .registration import register
  r=register(out);freeze(r,out)
 else:
  r=json.loads((out/'results.json').read_text());source_guard(r)
  if seal(r['protocol'])!=r['protocol_sha256'] or seal(json.loads((out/'protocol.json').read_text()))!=r['protocol_sha256']:raise RuntimeError('Protocol seal changed')
  if file_sha256(OLD/'results.json')!=r['assets']['old_results']['sha256']:raise RuntimeError('Historical results changed')
 try:
  if a.stage in ('all','preflight'):
   from .calibration import preflight
   preflight(r,out)
  if a.stage in ('all','train'):
   from .training import train_all
   train_all(r,out)
  if a.stage in ('all','teacher'):
   from .teacher import evaluate_teacher
   evaluate_teacher(r,out)
  if a.stage in ('all','sample'):
   from .sampling import sample_all,replay_once
   sample_all(r,out);replay_once(r,out)
  if a.stage in ('all','score'):
   from .evaluation import evaluate
   evaluate(r,out)
  if a.stage in ('all','report'):
   from .report import write_report
   if r.get('sampling_replay',{}).get('status')!='PASS' or r.get('teacher',{}).get('status')!='PASS' or 'pilot' not in r.get('summaries',{}):raise RuntimeError('Required fixed evaluation not complete')
   if sum(v['completed_updates'] for v in r['arms'].values())!=1000 or len(r['generated'])!=120:raise RuntimeError('Completed budget accounting differs')
   r.update(status='PASS_EXECUTION',stopped_at_registered_budget=True);write_report(r,out)
 except Exception as exc:
  r['status']='INTERRUPTED_OR_FAILED';r.setdefault('execution_errors',[]).append(dict(stage=a.stage,error=repr(exc),traceback=traceback.format_exc()))
  event('pilot_execution_error',stage=a.stage,error=repr(exc));raise
 finally:
  r['actual_accounting']=accounting(r);r.setdefault('stage_distribution_counters',{})[a.stage]=work_counters();write(r,out);actual_diff(out)

if __name__=='__main__':main()
