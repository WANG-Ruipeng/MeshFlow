"""Non-destructive indexing and a single frozen two-arm pilot protocol."""
from .common import *
from .edge_loss import FORMULA_SPEC
import datetime,platform,sys,shutil,subprocess
from native_t1.artifacts import atomic_json

def register(out):
 out=Path(out).resolve();tick=time.perf_counter()
 if out.exists() or out.parent!=REPO/'native_t1/runs' or not out.name.startswith('l3_l5_interface_pilot_'):raise ValueError('New isolated native_t1/runs pilot required')
 if torch.cuda.is_initialized():raise RuntimeError('Registration before CUDA initialization required')
 audit_path=PACKAGE/'asset_audit.json';audit=json.loads(audit_path.read_text())
 if audit['status']!='PASS' or not audit['seed_audit']['status'].startswith('PASS') or tuple(audit['seed_audit']['seeds'])!=SEEDS:raise RuntimeError('Read-only asset/seed gate failed')
 history=json.loads((OLD/'results.json').read_text())
 if file_sha256(OLD/'results.json')!=audit['old_results']['sha256']:raise RuntimeError('Historical report changed since audit')
 if seal(history['protocol'])!=audit['old_protocol_sha256']:raise RuntimeError('Old protocol identity changed')
 if audit['frozen_weights']['L3']!=LAMBDA3 or audit['frozen_weights']['L5']!=LAMBDA5:raise RuntimeError('Coefficient identity differs')
 stream=copy.deepcopy(history['stream_start'])
 if seal(stream)!=START_STREAM_SHA or seal(history['arms']['L0']['stream_audit'])!=PAIRED_STREAM_SHA:raise RuntimeError('Training flow identity differs')
 for name in OLD_STATES:
  a=history['arms'][name]
  if a['paired_stream_sha256']!=PAIRED_STREAM_SHA or a['initial_stream_sha256']!=START_STREAM_SHA or a['checkpoints']['500']['state_sha256']!=OLD_STATES[name]:raise RuntimeError('Old comparison identity differs')
 assets=dict(initial_checkpoint=record(INITIAL),inputs=record(INPUTS),evaluation_inputs=record(BANK),
  old_results=record(OLD/'results.json'),old_protocol=record(OLD/'protocol.json'),asset_audit=record(audit_path),
  old_calibration_velocities=history['calibration']['velocity_archive'],endpoints={name:copy.deepcopy(history['arms'][name]['checkpoints']['500']) for name in OLD_STATES})
 for name,expected in [('initial_checkpoint',INITIAL_SHA),('inputs',INPUT_SHA),('evaluation_inputs',BANK_SHA)]:
  if assets[name]['sha256']!=expected:raise RuntimeError('Asset changed: '+name)
 if shutil.disk_usage(REPO).free<8*1024**3:raise RuntimeError('Insufficient free disk for four required checkpoints')
 training=copy.deepcopy(history['protocol']['training'])
 training.update(arms=list(ARMS),candidates=list(ARMS),total_update_cap=1000,logs_at_steps=list(LOG_STEPS),
  auxiliary='ramp(s)*mean8(I[t>=.5]*(lambda_interface*interface+lambda5*L5))',
  old500_batch_stream_sha256=PAIRED_STREAM_SHA,not_continuing_endpoint_stream=True)
 losses=dict(L3=copy.deepcopy(history['protocol']['losses']['L3']),L5=copy.deepcopy(history['protocol']['losses']['L5']),
  D_edge=FORMULA_SPEC,J35='FM+ramp*mean8(g*(lambda3*L3+lambda5*L5))',
  JEdge5='FM+ramp*mean8(g*(lambda_edge*D_edge+lambda5*L5))')
 protocol=dict(schema='l3_l5_interface_pilot_protocol_v1',training=training,losses=losses,
  fixed_coefficients={'L3':LAMBDA3,'L5':LAMBDA5},initial_state_sha256=INITIAL_STATE,
  initialization='Both new branches from G0_2500, same historical500-batch segment, fresh AdamW; final3000',
  calibration=dict(reuse_previous_16_velocities=True,reconstruct_clone_samples=8,times=[.5,.75],
   lambda_edge='lambda3*median(||g3||)/median(||gE||) on same applicable finite states',
   old_coefficients_unchanged=True,new_model_forward_cap=16,formal_stream_advanced=False,
   joint_diagnostics='norms/cos(g3,g5)/cos(gE,g5), weighted joint norms and cosine to FM; no gradient surgery',
   training_probe='step100/500 first active micro; NONE if no active; original full-batch denominator and aux /8'),
  evaluation=dict(models=list(MODELS),seeds=list(SEEDS),phase='pilot',parents=[0,1],patches=[0,1,2],
   noises_per_cell=4,N=112,K=12,free_faces=100,steps=50,C_states=51,shared_Gaussian=True,
   sampler='Original native_t1.clamped_sample, no GT OT/loss/guidance/auxiliary network/postprocessing',
   thresholds_and_points='Unchanged independent evaluator and previous frozen evaluation bank',
   aggregate='Paired within seed in each parent/patch, mean4 then equal mean6 cells; all outputs retained',
   selection=False,initial_2500_new_evaluation=False),
  factorial=dict(B00='L0',B10='L3',B01='L5',B11='J35',interaction='E(J35)-E(L3)-E(L5)+E(L0)',
   scope='Fixed coefficients; no equal-total-supervision control, no pure mechanism synergy claim'),
  edge_comparison=dict(reference='J35',additional_references=['L5','L0'],scope='D_edge replaces L3 only in fixed L5 background; no D_edge-only arm'),
  teacher=dict(models=list(MODELS),states=16,times=[.5,.75],scores=['FM','L3','L5','D_edge'],
   samples='Same saved calibrated8 samples, read-only actual3000 endpoints, not update-before500 logs',forward_cap=80),
  budgets=dict(optimizer_updates=1000,training_forward_calls=8000,training_backward_calls=8000,
   calibration_model_forward_cap=16,planned_calibration_model_forwards=0,wiring_forward_cap=2,wiring_backward_cap=2,
   teacher_forward_cap=80,formal_rollouts=120,replay_cap=3,planned_replay=1),
  failure_policy='Preserve failed calls and partial updates; no retry/retune. Invalid edge calibration stops JEdge5 with no fallback; J35 remains independently eligible.',
  inherited_BF16_permutation='Historical baseline-inheritance FAIL unchanged; no replay of old investigation or relaxed tolerance',
  stop='Two500-update branches and fixed120-output evaluation only; no added loss/seed/steps, default switch, commit/push or deletion')
 fixtures={}
 for name in ('cpu_edge_record.json','evaluation_test_record.json','report_test_record.json'):
  p=PACKAGE/name
  if not p.exists():raise RuntimeError('Required executed CPU fixture record missing: '+name)
  fixtures[name]=dict(artifact=record(p),result=json.loads(p.read_text()))
 r=dict(schema='native_t1_l3_l5_interface_pilot_results_v1',status='REGISTERED',output_dir=str(out),
  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=protocol,protocol_sha256=seal(protocol),
  old_protocol_sha256=history['protocol_sha256'],assets=assets,stream_start=stream,AGENTS=audit['AGENTS'],git=audit['git'],
  seed_audit=audit['seed_audit'],asset_audit=audit,CPU_fixtures=fixtures,
  reused={name:dict(status='READ_ONLY_VERIFIED',checkpoint=assets['endpoints'][name],new_optimizer_updates=0,old_additional_updates=500,
   original_start_state_sha256=INITIAL_STATE,old500_batch_stream_sha256=PAIRED_STREAM_SHA) for name in OLD_STATES},
  arms={name:dict(status='PENDING',completed_updates=0,checkpoints={},logs=[]) for name in ARMS},
  counts=dict(calibration={},wiring={},training={},sampling={},teacher={}),loss_accounting={},generated=[],replays=[],
  environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),CUDA_initialized=False,
   disk_free_bytes=shutil.disk_usage(REPO).free,GPU_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()),
  registration=dict(status='PASS',seconds=time.perf_counter()-tick,model_calls=0,OT_calls=0,updates=0))
 out.mkdir(parents=True,exist_ok=False);atomic_json(out/'protocol.json',protocol);write(r,out)
 index=['# Experiment index','',
  'Current default: Native T1 + Geo, original FM. Public G0 cumulative2500 is unchanged; new candidates are not defaults.','',
  'Historical six-loss screen is read-only: '+str(OLD),
  'Historical results SHA256: '+assets['old_results']['sha256'],
  'No old checkpoint, raw output, score, source snapshot or failure record is moved, deleted or overwritten.','',
  'This pilot trains only J35 and JEdge5, each from the same G0_2500 for500 updates, and evaluates only L0/L3/L5/J35/JEdge5.',
  'L1/L2/L4/L6 remain historical candidates and are absent from this pilot default run list.','',
  '| Role | Path | State SHA256 |','| --- | --- | --- |',
  '| Common initialization / current default reference | '+str(INITIAL)+' | '+INITIAL_STATE+' |']
 for name in OLD_STATES:index.append('| Read-only '+name+'3000 | '+assets['endpoints'][name]['path']+' | '+OLD_STATES[name]+' |')
 index+=['','Pilot protocol: protocol.json. New endpoints are recorded in results.json; no historical weights are copied.','']
 (out/'EXPERIMENT_INDEX.md').write_text('\n'.join(index),encoding='utf-8');r['experiment_index']=record(out/'EXPERIMENT_INDEX.md');write(r,out)
 return r
