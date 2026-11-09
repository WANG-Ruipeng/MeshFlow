"""Bounded operational readback: no training/module invocation or GPU probe."""
from pathlib import Path
import datetime,hashlib,json,subprocess,os,re,math,shutil
ROOT=Path.cwd().resolve();OUT=Path(__file__).resolve().parent
Q=ROOT/'output/a_overnight_queue_20260925_01';F=ROOT/'output/a_system_official_closed16_20260925_01'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def cmd(args):
 p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=20,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
 return {'command':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def main():
 now=datetime.datetime.now(datetime.timezone.utc);p=read(F/'RUN_PROTOCOL.json');sp=read(F/'SAMPLING_PROTOCOL.json');plan=read(Q/'PLAN.json');source=read(F/'SOURCE_MANIFEST.json');lock=read(ROOT/'experiments/a_radial/assets/UPSTREAM_LOCK.json')
 assert sha(F/'SOURCE_MANIFEST.json')==p['source_manifest_sha256']
 tree=cmd(['git','ls-tree','-r',lock['commit']]);assert tree['returncode']==0
 pin={}
 for line in tree['stdout'].splitlines():
  meta,name=line.split('\t',1);pin[name]=meta.split()[2]
 protected={}
 for name,digest in source['protected_sha256'].items():
  raw=(ROOT/name).read_bytes();blob=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
  protected[name]={'sha256':sha(ROOT/name),'expected_sha256':digest,'git_blob_sha1':blob,'pinned_blob_sha1':pin[name],'unchanged':sha(ROOT/name)==digest and blob==pin[name]}
 assert len(protected)==11 and all(x['unchanged'] for x in protected.values())
 for name,digest in lock['observed_blobs'].items():assert pin[name]==digest
 treeid=cmd(['git','rev-parse',lock['commit']+'^{tree}']);assert treeid['stdout'].strip()==lock['tree']
 bindings=[]
 for category,items in [('formal_source',source['source_sha256']),('queue_source',plan['source_sha256']),('queue_bound_files',plan['bound_files_sha256'])]:
  for name,digest in items.items():
   value={'category':category,'path':name,'expected_sha256':digest,'actual_sha256':sha(ROOT/name)};assert value['actual_sha256']==digest
   if category=='queue_source':assert sha(Q/'source_snapshot'/name)==digest
   bindings.append(value)
 status=read(Q/'STATUS.json');started=read(Q/'STARTED.json');claim=read(F/'OVERNIGHT_QUEUE_CLAIM.json')
 assert started['pid']==claim['pid'] and started['plan_sha256']==claim['plan_sha256']==sha(Q/'PLAN.json')
 assert status['status']=='RUNNING' and status['stage']=='OFFICIAL_RUN_ALL_2X50000'
 assert read(Q/'heartbeats'/f"{status['heartbeat']:06d}.json")==status
 heartbeat_age=(now-datetime.datetime.fromisoformat(status['utc'])).total_seconds()
 proc=cmd(['ps','-eo','pid,ppid,pgid,stat,etimes,pcpu,pmem,args']);assert proc['returncode']==0
 lines=[x for x in proc['stdout'].splitlines() if 'experiments.a_radial.system_formal.' in x and 'readback.py' not in x]
 controller=[x for x in lines if 'system_formal.overnight_queue ' in x];trainer=[x for x in lines if 'system_formal.train train ' in x];driver=[x for x in lines if 'system_formal.run_all ' in x]
 assert len(controller)==len(trainer)==len(driver)==1
 assert int(controller[0].split()[0])==started['pid'];assert int(driver[0].split()[0])==status['child_pid']
 assert int(trainer[0].split()[1])==status['child_pid'] and '--seed 10' in trainer[0]
 prog={};trace={};ckpts=[];fail=[]
 for seed in (10,11):
  folder=F/'training'/f'seed_{seed}';path=folder/'PROGRESS.json'
  row=read(path) if path.exists() else None
  if row:
   assert row['seed']==seed and row['EMA_updates']==row['step'] and row['parent_exposures']==[row['step']]*16
   assert math.isfinite(row['latest_loss']) and math.isfinite(row['latest_gradient_norm'])
  prog[str(seed)]={'latest_recorded':row,'recorded_updates':row['step'] if row else 0,'remaining_to_fixed_final':50000-(row['step'] if row else 0),'progress_file_mtime_utc':datetime.datetime.fromtimestamp(path.stat().st_mtime,datetime.timezone.utc).isoformat() if row else None,'progress_age_seconds':now.timestamp()-path.stat().st_mtime if row else None,'completion_present':(folder/'TRAINING_COMPLETE.json').exists()}
  chunks=sorted(folder.glob('invocations/*/trace_chunks/*.json'));alltrace=[]
  for path in chunks:alltrace+=read(path)['trace']
  finite=all(math.isfinite(x['loss']) and math.isfinite(x['gradient_norm_before_clip']) for x in alltrace)
  assert finite
  trace[str(seed)]={'saved_trace_rows':len(alltrace),'all_saved_losses_and_gradient_norms_finite':finite,'chunk_files':len(chunks)}
  for path in folder.glob('invocations/*/checkpoints/*.json'):
   rec=read(path);cp=Path(rec['path']);assert cp.exists() and cp.stat().st_size==rec['bytes']
   ckpts.append({'seed':seed,'step':rec['step'],'path':str(cp),'bytes':rec['bytes'],'completion_record_sha256':sha(path),'content_hash_not_repeated_for_large_active_checkpoint':True})
  fail += [str(x) for x in folder.rglob('FAILED.json')]
 for base in (Q,F):
  for name in ('FAILED.json','QUEUE_FAILED.json','posttraining/FINISH_FAILED.json'):
   if (base/name).exists():fail.append(str(base/name))
 assert not fail
 pattern=re.compile(r'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|(?:^|[^A-Za-z])(?:NaN|nan|Inf|Infinity)(?:[^A-Za-z]|$)|FloatingPointError|non-finite|(?:^|\W)OOM(?:\W|$)')
 logs=[]
 paths=list((Q/'logs').glob('*.log'))+list((F/'queue_logs').glob('*.log'))+[Q/'controller.log',Q/'controller_windows.stderr.log',Q/'controller_windows.stdout.log']
 for path in paths:
  if not path.exists():continue
  text=path.read_text(errors='replace');hits=[{'line':i,'text':line[:2000]} for i,line in enumerate(text.splitlines(),1) if pattern.search(line)]
  logs.append({'path':str(path),'snapshot_bytes':len(text.encode()),'checked_lines':len(text.splitlines()),'error_pattern_matches':hits,'last_2_lines':text.splitlines()[-2:]})
 assert not any(x['error_pattern_matches'] for x in logs)
 gpu=cmd(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'])
 gpu_compute=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'])
 disk=shutil.disk_usage(ROOT);rootdisk=shutil.disk_usage(Path('/'))
 # Two seeds times (step0 + 10 checkpoints). Upper planning reserve includes all4 fp32 model-equivalent arrays plus1GiB/ckpt for traces and metadata.
 checkpoint_count=2*(1+50000//p['checkpoint_every']);reserve_per_checkpoint=p['full_parameter_count']*4*4+2**30
 reserve=checkpoint_count*reserve_per_checkpoint
 live=Path('/mnt/e/MeshFlow/output/LIVE_TRAINING_20260925.md');live_bytes=live.read_bytes()
 assert plan['stages']==['WAIT_BALANCED240K','OFFICIAL_RUN_ALL_2X50000','FROZEN_FINISH']
 assert plan['official_updates_each']==50000 and plan['official_seeds']==[10,11] and plan['official_weight']=='ema' and plan['quality_loss_does_not_stop_or_select'] and not plan['automatic_retries']
 qproto=read(F/'QUEUE_PROTOCOL.json');assert not qproto['loss_based_early_stop'] and not qproto['automatic_retries']
 assert p['checkpoint_selection']=='fixed_final_50000_only' and p['primary_weight']=='ema'
 evidence={'status':'OPERATIONAL_READBACK_PASS_NOT_FUTURE_COMPLETION_GUARANTEE','observed_utc':now.isoformat(),'observed_Asia_Shanghai':now.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'source_sha256':sha(__file__),'protected_source':{'pin':lock['commit'],'tree':lock['tree'],'all11_unchanged':True,'files':protected,'scope':'active isolated worktree, not original E:/MeshFlow checkout','git_read_only_commands':['ls-tree','rev-parse'],'GIT_OPTIONAL_LOCKS':'0'},'frozen_bindings':bindings,'queue':{'plan_sha256':sha(Q/'PLAN.json'),'status_snapshot':status,'heartbeat_age_seconds':heartbeat_age,'heartbeat_file_matches_status':True,'controller_pid':started['pid'],'stage':'OFFICIAL_RUN_ALL_2X50000','one_controller_one_driver_one_trainer':True,'process_lines':lines},'progress':prog,'remaining_updates_from_latest_saved_progress':sum(x['remaining_to_fixed_final'] for x in prog.values()),'progress_is_100_step_snapshot_not_instantaneous_counter':True,'trace_checks':trace,'current_complete_checkpoints':ckpts,'runtime_failure_markers':fail,'active_log_scan':logs,'GPU_query_only_no_compute_probe':gpu,'GPU_compute_process_query':gpu_compute,'disk':{'E_mount_free_bytes':disk.free,'E_mount_total_bytes':disk.total,'WSL_root_free_bytes':rootdisk.free,'planned_checkpoint_count_including_step0':checkpoint_count,'planning_reserve_bytes':reserve,'reserve_formula':'22 * (130249731 parameters * 4 bytes * 4 arrays + 1GiB trace/metadata allowance); conservative planning estimate, not measured future bound','free_exceeds_checkpoint_reserve':disk.free>reserve},'execution_contract':{'stage_order':plan['stages'],'training_seeds_serial':[10,11],'effective_updates_per_seed':50000,'primary_weight':'EMA from fixed50000','no_loss_or_quality_early_stop':True,'runtime_or_identity_error_can_stop_and_preserves_failures':True,'no_automatic_retry':True,'posttraining_per_seed_samples':64,'primary_ours_FM_step':8000,'optional64k_is_separate':True,'raw_and_Public_Post_paper_full_false':True,'sampling_and_evaluation_not_started':not (F/'posttraining/FINISH_STARTED.json').exists()},'live_page':{'windows_path':'E:\\MeshFlow\\output\\LIVE_TRAINING_20260925.md','readback_sha256':hashlib.sha256(live_bytes).hexdigest(),'age_seconds':now.timestamp()-live.stat().st_mtime},'limitations':['Read-only operational snapshot; no guarantee of future training, sampling, evaluation success or completion time.','No new unit tests, GPU probe, training control, checkpoint load, or process start/stop.','Loss is current training batch evidence, not convergence or independent validation.']}
 dump(OUT/'OPERATIONAL_READBACK.json',evidence)
 print(json.dumps({k:evidence[k] for k in ('status','observed_Asia_Shanghai','remaining_updates_from_latest_saved_progress','disk','live_page')},indent=2))
 print('progress',[(k,v['recorded_updates'],v['remaining_to_fixed_final']) for k,v in prog.items()]);print('heartbeat age',heartbeat_age);print('GPU',gpu['stdout']);print('processes','\n'.join(lines))
if __name__=='__main__':main()
