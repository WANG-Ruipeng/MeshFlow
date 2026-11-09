"""Explicit staged runner. Every scientific stage rejects an automatic rerun."""
from .common import *
import argparse,subprocess,shutil,traceback

def register(out=OUT):
 out=Path(out)
 if (out/'results.json').exists():raise FileExistsError('Experiment already registered')
 registration_path=out/'registration/registration.json'
 registration=json.loads(registration_path.read_text())
 if registration.get('status') not in ('FROZEN','PASS'):raise RuntimeError('Evaluation must be frozen before training')
 units=registration['units'];seeds=registration['seeds']
 if len(seeds)!=4 or len(units)>10:raise RuntimeError('Registered matrix exceeds request')
 from native_t1.geometry_checkpoint import load_geometry_checkpoint
 assets={'inputs':record(INPUTS),'initial_checkpoint':record(INITIAL),'teacher_samples':record(TEACHER),
  'old_results':record(OLD/'results.json'),'pilot_results':record(PILOT/'results.json'),
  'registration':record(registration_path),'evaluation_inputs':registration['evaluation_inputs'],'endpoints':{}}
 if assets['inputs']['sha256']!=INPUT_SHA or assets['initial_checkpoint']['sha256']!=INITIAL_SHA or assets['teacher_samples']['sha256']!=TEACHER_SHA:raise RuntimeError('Core asset hash differs')
 p=torch.load(INITIAL,map_location='cpu',weights_only=True,mmap=True)
 stream_start=p['stream_state'];del p
 if seal(stream_start)!=START_STREAM_SHA:raise RuntimeError('G0 stream seal differs')
 expected_paths={'L0':('ablations/fm_cumulative3000.pt','fm'),'L5':('ablations/surface_cumulative3000.pt','surface'),
                 'JEdge5':('jedge5_cumulative3000.pt','edge5')}
 audits={}
 for name,(rel,recipe) in expected_paths.items():
  path=REPO/'native_t1/checkpoints'/rel;model,a=load_geometry_checkpoint(path,device='cpu',expected_recipe=recipe)
  if a['state_sha256']!=STATES[name] or a['parameters']!=PARAMETERS or a['cumulative_updates']!=3000:raise RuntimeError('Retained endpoint identity differs')
  if name=='JEdge5' and (a['training_objective']['lambda_edge'],a['training_objective']['lambda_surface'])!=(LAMBDA_EDGE,LAMBDA5):raise RuntimeError('Fixed full-precision coefficients differ')
  assets['endpoints'][name]=dict(**record(path),state_sha256=a['state_sha256'],cumulative_updates=3000)
  audits[name]=a;del model;release()
 old=json.loads((PILOT/'results.json').read_text())
 if (old['calibration']['lambda_edge'],old['calibration']['lambda5'])!=(LAMBDA_EDGE,LAMBDA5):raise RuntimeError('Historical coefficient differs')
 protocol=dict(name='fixed_edge_only_factorial',training=dict(arm='Edge_only',updates=500,total_update_cap=500,
  base_cumulative_updates=2500,endpoint_cumulative_updates=3000,initial_stream_sha256=START_STREAM_SHA,
  paired_stream_sha256=PAIRED_STREAM_SHA,seed=1010,batch_start=1000,sample_start=8000,effective_batch=8,microbatch=1,
  AdamW=dict(lr=1e-5,betas=[.9,.95],weight_decay=0.,fresh=True),grad_clip=1.,full_generator_trainable=True,
  time='original logit-normal',OT='original free-only nested OT reorders Z only',logs=[1,50,100,250,500],checkpoints=[250,500],
  lambda_edge=LAMBDA_EDGE,lambda5=0.,gate='original FP32 t>=0.5',ramp='min(s/50,1)',auxiliary_denominator=8,
  FM_denominator='original effective-batch free coordinates',default_model_changed=False),
  evaluation=dict(models=list(MODELS),seeds=seeds,units=units,formal_output_cap=160,expected_outputs=4*len(units)*4,
   replay_cap=2,registered_replay=dict(model='Edge_only',parent=0,patch=0,seed=seeds[0]),N=112,K=12,free=100,
   Euler_steps=50,coverage_thresholds_bbox=[.0005,.001,.002,.005,.01],GT_OT_start=False,postprocessing=False),
  teacher=dict(new_forwards=16,old_models='read-only if same state/metric',times=[.5,.75]),
  wiring=dict(forward_cap=8,planned_forwards=5,planned_zero_update_backwards=1),
  interpretation='two seen parents, new contexts only; no independent training seed; fixed unequal auxiliary strength')
 r=dict(schema='edge_only_factorial_results_v1',status='REGISTERED',output_dir=str(out),protocol=protocol,
  protocol_sha256=seal(protocol),assets=assets,registration=registration,stream_start=stream_start,
  compact_identity_audit=audits,training=dict(status='READY',completed_updates=0),
  counts={k:{} for k in ('training','wiring','teacher','sampling','CPU_parity')},loss_accounting={},generated=[],replays=[],
  git=dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
   status=subprocess.check_output(['git','-c','core.autocrlf=true','-c','core.filemode=false','status','--short'],text=True)),
  source_snapshot=dict(files=[]),timing={})
 # Only executable training dependencies are frozen here; report-only files may finish independently.
 paths=list((REPO/'native_t1').glob('*.py'))+list((REPO/'native_t1/losses').glob('*.py'))
 paths+=list((REPO/'models').rglob('*.py'))+[REPO/'utils/ot_utils.py',REPO/'datasets/mesh_dataset.py']
 paths+=[Path(__file__).parent/name for name in ('common.py','checkpoint.py','training.py','preflight.py','legacy_edge_loss.py','legacy_relation_losses.py','driver.py','teacher.py','sampling.py','registration.py','evaluation.py','edge_metric.py')]
 snap=out/'source_snapshot';snap.mkdir(exist_ok=False)
 for path in sorted(set(paths)):
  rel=path.relative_to(REPO);dest=snap/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
  r['source_snapshot']['files'].append(dict(**record(path),snapshot=str(dest)))
 atomic_json(out/'protocol.json',protocol);write(r,out);event('registered',outputs=protocol['evaluation']['expected_outputs'],seeds=seeds)
 return r

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['register','preflight','train','teacher','sample','evaluate','report'])
 args=parser.parse_args();out=OUT
 if args.stage=='register':register(out);return
 r=read(out);start=time.perf_counter()
 try:
  if args.stage=='preflight':
   from .preflight import preflight
   preflight(r,out)
  elif args.stage=='train':
   from .training import train
   train(r,out)
  elif args.stage=='teacher':
   from .teacher import evaluate_teacher
   evaluate_teacher(r,out)
  elif args.stage=='sample':
   from .sampling import sample_all
   sample_all(r,out)
  elif args.stage=='evaluate':
   from .evaluation import evaluate
   evaluate(r,out)
  elif args.stage=='report':
   from .report import write_report
   write_report(r,out)
  r['timing'][args.stage]=time.perf_counter()-start
 except BaseException as exc:
  r.setdefault('failures',[]).append(dict(stage=args.stage,error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-start))
  r['status']='STOPPED';raise
 finally:write(r,out)

if __name__=='__main__':main()
