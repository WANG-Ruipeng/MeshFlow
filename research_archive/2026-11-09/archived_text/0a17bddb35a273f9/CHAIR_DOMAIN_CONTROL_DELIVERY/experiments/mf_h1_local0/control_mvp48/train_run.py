"""Single-worker Control MVP48 training tree. No automatic replay or extra arms."""
import os
for _thread_variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable]='2'
from native_t1 import runtime as _runtime
import argparse
import csv
import difflib
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import threading
import time
import traceback
import numpy as np
import torch
from native_t1.artifacts import atomic_json, file_sha256
from native_t1.portable_checkpoint import OFFICIAL_FILE_SHA256
from native_t1.training import make_optimizer
from .train_contract import ROOT, PROTOCOL, WaitingDependency, read_binding
from .train_core import optimizer_update
from .train_stream import FrozenStream, MVPObjective
from .model_mvp import initialize_mvp, json_hash, load_mvp_checkpoint
from .train_io import gpu_lock, resource_snapshot, save_checkpoint, restore_branch
from .train_precheck import run_precheck

REPO = Path(__file__).resolve().parents[3]
COLUMNS=['phase','logical_step','branch_step','actual_optimizer_updates','batch_sha256','FM','geometry','loss',
    'grad_norm_before_clip','free_coordinate_denominator','active_samples','mean_active_D_edge','mean_active_L5',
    'OT_seconds','train_seconds','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','actual_OT_returns','OT_cache_hits']
RECOVERY = ('cd /mnt/e/MeshFlow && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '
    '/home/warpwang/miniforge3/envs/meshflow/bin/python -B -m '
    'experiments.mf_h1_local0.control_mvp48.train_run --official-checkpoint /ABSOLUTE/PATH/TO/CHAIR/last.pt')

def append_json(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())

def source_files():
    own = Path(__file__).parent
    files = list(own.glob('train_*.py')) + list(own.glob('model_*.py'))
    files += list((REPO/'native_t1').glob('*.py')) + list((REPO/'native_t1/losses').glob('*.py'))
    files += list((REPO/'models').glob('*.py'))
    files += [REPO/'utils/ot_utils.py', REPO/'configs/snet/base-120m-ot-v-chair.yaml']
    return sorted(set(files))

def code_snapshot(root):
    values = {str(p.relative_to(REPO)):file_sha256(p) for p in source_files()}
    atomic_json(root/'registration/train_code_hashes.json', values)
    chunks=[]
    for path in sorted(Path(__file__).parent.glob('*.py')):
        if path.name.startswith(('train_', 'model_')):
            relative=str(path.relative_to(REPO))
            chunks.extend(difflib.unified_diff([], path.read_text().splitlines(keepends=True),
                fromfile='/dev/null', tofile='b/'+relative))
    (root/'training/code.diff').write_text(''.join(chunks))
    return values

def report(root, status):
    log=root/'training/training_log.csv'
    if not log.exists():
        with log.open('x',newline='') as f: csv.writer(f).writerow(COLUMNS)
    index=root/'checkpoints/checkpoints.json'
    index.parent.mkdir(parents=True,exist_ok=True)
    if not index.exists(): atomic_json(index,dict(status='NOT_RUN',checkpoints=[],actual_optimizer_updates=0))
    lines = ['# Control MVP48 — Session B', '', 'Status: '+status['status'], '',
        'Actual optimizer updates: '+str(status.get('actual_optimizer_updates',0))+' / 2500.',
        'Executed phases: '+json.dumps(status.get('phase_updates',{})), '',
        'Implementation: one variable-N Native model, original Geo/Edge/L5 functions, effective batch8/microbatch1.',
        'Official Chair EMA only. Geo/Full restore the same complete shared step500 model, AdamW, RNG and data cursor.',
        'Formal GPU concurrency is one under /tmp/wrp_meshflow_jedge5_diag_gpu0.lock; CPU numerical threads are capped at two.',
        'Timing and GPU allocated/reserved peaks are phase scoped; nvidia-smi global/process observations are separate.',
        'Concurrent observations are not isolated compute benchmarks.',
        '',
        'Historical BF16 permutation boundary remains FAIL: atol=.015625, rtol=.03125; 10/1008 coordinates; max_abs=.064453125.',
        'Evidence: native_t1/runs/geometry_context_revised_20260930_153000/gates.json. No threshold changed or repair claimed.',
        '',
        'CPU checks: training/cpu_tests.json (synthetic fixtures only, zero optimizer steps).',
        'Real-data/official-weight prechecks: '+('PASS' if (root/'training/precheck.json').exists() else 'NOT_RUN')+'.',
        'Training/checkpoint implementation has not been end-to-end exercised when dependencies are blocked.',
        'Checkpoints and sidecars: checkpoints/. Ledger: training/training_log.csv and training/update_events.jsonl.',
        'Unified API: experiments.mf_h1_local0.control_mvp48.model_mvp.load_mvp_checkpoint(path, device)',
        'Returns (model, actual_metadata, hashes). Model construction/inference must not be reimplemented by Session C.',
        'Inference helpers: velocity_mvp and integrate_mvp_step. The caller owns its registered sampling budget and GPU lock.',
        '', 'No TRAIN_READY is published unless all three final endpoints and strict reloading succeed.']
    if status.get('dependencies'): lines += ['', 'Dependencies: '+json.dumps(status['dependencies'], ensure_ascii=False)]
    if status.get('error'): lines += ['', 'Failure: '+status['error']]
    lines += ['', 'Recovery before any attempted GPU precheck/training:', '', '```bash', RECOVERY, '```',
        '', 'After any numerical failure or optimizer attempt, inspect status.json and update_events.jsonl first.',
        'This runner refuses automatic restarts; no failure replay budget was granted. Completed checkpoints remain preserved.']
    (root/'training/TRAIN_REPORT.md').write_text('\n'.join(lines)+'\n')

class GPUMonitor:
    def __init__(self, root):
        self.root=root; self.stop_event=threading.Event()
        self.peak_global_used_mib=0.; self.peak_process_used_mib=None
        self.thread=threading.Thread(target=self.run, daemon=True)
    def run(self):
        while not self.stop_event.is_set():
            try:
                result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                if result.returncode==0:
                    self.peak_global_used_mib=max(self.peak_global_used_mib,float(result.stdout.split(',')[0]))
                for line in processes.stdout.splitlines():
                    parts=[p.strip() for p in line.split(',')]
                    if len(parts)>=3 and parts[0]==str(os.getpid()):
                        try: value=float(parts[-1])
                        except ValueError: continue
                        self.peak_process_used_mib=max(self.peak_process_used_mib or 0.,value)
                append_json(self.root/'training/gpu_samples.jsonl',dict(time_unix=time.time(),
                    pid=os.getpid(),gpu=result.stdout.strip(),gpu_returncode=result.returncode,
                    processes=processes.stdout.strip(),process_returncode=processes.returncode))
            except Exception as exc:
                append_json(self.root/'training/gpu_samples.jsonl',dict(monitor_error=str(exc)))
            self.stop_event.wait(2)
    def start(self): self.thread.start()
    def stop(self):
        self.stop_event.set(); self.thread.join(timeout=25)

def dispatch_check(snapshot):
    if snapshot['gpu']['returncode']:
        raise RuntimeError('nvidia-smi failed before dispatch')
    first=snapshot['gpu']['stdout'].splitlines()[0].split(',')
    total, used, free, utilization=map(float,first[-4:])
    headroom=max(2048., .10*total)
    if free <= headroom:
        raise WaitingDependency('GPU free memory does not preserve required headroom')
    if snapshot['memory']['MemAvailable_kib'] < 5*1024*1024:
        raise WaitingDependency('Less than 5 GiB available RAM for single training worker')
    return dict(gpu_workers=1, free_mib=free, total_mib=total, headroom_mib=headroom,
        utilization_percent=utilization, comparable_training_process_peak=None,
        rationale='Explicit experiment cap=1; first authorized precheck/training observes representative peak. No probe job.',
        numerical_threads=2, OT_cache=False, source='Actual nvidia-smi and /proc snapshot')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--official-checkpoint',type=Path,default=Path('/mnt/c/Users/warpwang/Downloads/last.pt'))
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    root=ROOT
    for name in ('training','checkpoints','registration'): (root/name).mkdir(parents=True,exist_ok=True)
    status_path=root/'training/status.json'
    if args.status:
        print(status_path.read_text() if status_path.exists() else 'No Session B status yet')
        return
    if (root/'TRAIN_READY.json').exists():
        print('Already complete; no further work authorized.')
        return
    attempt_marker=root/'training/gpu_attempt_started.json'
    if attempt_marker.exists():
        raise RuntimeError('An earlier GPU attempt exists. Refusing automatic precheck/training replay; inspect status and ledger.')
    for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[variable]='2'
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    _runtime.configure_stable_runtime()
    status=dict(status='PREPARING', actual_optimizer_updates=0,
        phase_updates={name:0 for name in ('T1','Geo_shared','Geo','Full')},
        optimizer_step_attempts=0, completed_checkpoints=[], recovery_command=RECOVERY,
        cuda_initialized=torch.cuda.is_initialized())
    resources=resource_snapshot()
    atomic_json(root/'registration/train_resources_before.json',resources)
    sources=code_snapshot(root)
    dependencies=[]
    try:
        binding, protocol, plan_path, data_root=read_binding(root)
    except WaitingDependency as exc:
        dependencies.append(str(exc))
    if not args.official_checkpoint.is_file():
        dependencies.append('BLOCKED_OFFICIAL_CHAIR_ASSET_MISSING: '+str(args.official_checkpoint))
    elif file_sha256(args.official_checkpoint)!=OFFICIAL_FILE_SHA256:
        dependencies.append('BLOCKED_OFFICIAL_CHAIR_IDENTITY_MISMATCH: '+str(args.official_checkpoint))
    if dependencies:
        status.update(status='WAITING_DEPENDENCY', dependencies=dependencies, cuda_initialized=torch.cuda.is_initialized())
        atomic_json(status_path,status); report(root,status)
        print(json.dumps(status,indent=2)); return
    if shutil.disk_usage(root).free < 12*1024**3:
        raise RuntimeError('Need at least 12 GiB free space for the five full-state checkpoints and staging')
    atomic_json(root/'registration/train_protocol.json',dict(protocol=protocol,protocol_sha256=json_hash(protocol)))
    cpu_checks=json.loads((root/'training/cpu_tests.json').read_text())
    if cpu_checks['status']!='PASS' or cpu_checks.get('source_hashes')!=sources:
        raise RuntimeError('Current-code CPU implementation checks must pass before GPU work')
    dispatch=dispatch_check(resources)
    atomic_json(root/'registration/train_dispatch.json',dispatch)
    phase_updates=status['phase_updates']
    batches={}
    checkpoint_records=[]
    log_path=root/'training/training_log.csv'
    monitor=None; current_counts={}; current_phase=None; stream=None
    total_tick=time.perf_counter()
    try:
        with gpu_lock(root) as lock_seconds:
            dispatch_check(resource_snapshot())
            atomic_json(attempt_marker,dict(start_time_unix=time.time(),protocol_sha256=json_hash(protocol),
                official_checkpoint=str(args.official_checkpoint),optimizer_updates=0))
            status.update(status='REAL_DATA_PRECHECK',lock_wait_seconds=lock_seconds)
            atomic_json(status_path,status)
            monitor=GPUMonitor(root);monitor.start()
            # All actual CUDA initialization happens after the shared lock.
            torch.cuda.reset_peak_memory_stats()
            precheck=run_precheck(root,args.official_checkpoint,binding,plan_path,data_root)
            status['precheck_seconds']=precheck['seconds']
            # Never train a precheck-mutated object. Reconstruct original initialization.
            random.seed(10);np.random.seed(10);torch.manual_seed(10);torch.cuda.manual_seed_all(10)
            if log_path.exists():
                old_rows=list(csv.reader(log_path.open(newline='')))
                if old_rows != [COLUMNS]: raise RuntimeError('Existing training log contains results; no replay allowed')
            with log_path.open('a',newline='') as csv_file:
                writer=csv.DictWriter(csv_file,fieldnames=COLUMNS)
                if log_path.stat().st_size==0: writer.writeheader()
                csv_file.flush()
                for phase,start,end in PROTOCOL['phases']:
                    current_phase=phase; current_counts={}
                    phase_tick=time.perf_counter()
                    observation=resource_snapshot()
                    atomic_json(root/f'training/{phase}_resources_before.json',observation)
                    dispatch_check(observation)
                    for relative,digest in sources.items():
                        if file_sha256(REPO/relative)!=digest: raise RuntimeError('Code changed during fixed run: '+relative)
                    fresh_binding,fresh_protocol,_,_=read_binding(root)
                    if fresh_protocol!=protocol: raise RuntimeError('Frozen input identities changed')
                    stream=FrozenStream(binding,plan_path,data_root)
                    load_tick=time.perf_counter();fork_sha=None
                    if start==0:
                        random.seed(10);np.random.seed(10);torch.manual_seed(10);torch.cuda.manual_seed_all(10)
                        model,identity=initialize_mvp(args.official_checkpoint,geo=phase!='T1',device='cuda')
                        optimizer=make_optimizer(model)
                        atomic_json(root/f'training/{phase}_initialization.json',identity)
                    else:
                        model,optimizer,fork_sha=restore_branch(root/'checkpoints/Geo_shared_step500.pt',stream)
                        atomic_json(root/f'training/{phase}_fork_restore.json',dict(status='PASS',fork_checkpoint_sha256=fork_sha,
                            model_optimizer_rng_stream_restored=True))
                    load_seconds=time.perf_counter()-load_tick
                    torch.cuda.reset_peak_memory_stats()
                    objective=MVPObjective(stream) if phase=='Full' else None
                    ot_seconds=train_seconds=save_seconds=0.
                    status.update(status='TRAINING',phase=phase)
                    for step in range(start+1,end+1):
                        ot_tick=time.perf_counter();samples=stream.next_effective_batch();elapsed=time.perf_counter()-ot_tick
                        ot_seconds+=elapsed
                        sample_hashes=[s['shared_input_sha256'] for s in samples]
                        bh=hashlib.sha256(''.join(sample_hashes).encode()).hexdigest()
                        if phase=='T1':
                            batches[str(step)]=dict(batch_sha256=bh,sample_hashes=sample_hashes)
                        elif batches[str(step)]['batch_sha256']!=bh:
                            raise RuntimeError('Shared data/Gaussian/OT/time/permutation bytes differ at '+str(step))
                        append_json(root/'training/update_events.jsonl',dict(event='UPDATE_BEGIN',phase=phase,logical_step=step,
                            actual_completed_before=status['actual_optimizer_updates'],batch_sha256=bh))
                        result=optimizer_update(model,optimizer,samples,objective=objective,
                            additional_step=step-500 if phase=='Full' else None,counts=current_counts)
                        phase_updates[phase]=current_counts['optimizer_updates']
                        status['actual_optimizer_updates']=sum(phase_updates.values())
                        status['optimizer_step_attempts']=sum(phase_updates.values())
                        train_seconds+=result['train_seconds']
                        diagnostic=result['geometry_diagnostics']
                        def mean_term(name):
                            values=[d['raw_losses'][name] for d in diagnostic if name in d['raw_losses']]
                            return sum(values)/len(values) if values else 0.
                        row=dict(phase=phase,logical_step=step,branch_step=step-start,
                            actual_optimizer_updates=status['actual_optimizer_updates'],batch_sha256=bh,
                            **{k:result[k] for k in ('FM','geometry','loss','grad_norm_before_clip','free_coordinate_denominator','active_samples')},
                            mean_active_D_edge=mean_term('D_edge'),mean_active_L5=mean_term('L5'),
                            OT_seconds=elapsed,train_seconds=result['train_seconds'],
                            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            actual_OT_returns=stream.counts['actual_OT_returns'],OT_cache_hits=stream.counts['OT_cache_hits'])
                        writer.writerow(row);csv_file.flush()
                        append_json(root/'training/update_events.jsonl',dict(event='UPDATE_VALIDATED',**row))
                        status.update(logical_step=step,current_counts=dict(current_counts),current_OT_counts=dict(stream.counts))
                        atomic_json(status_path,status)
                        if step in (1,100,500,1000) or step-start in (1,100,500):
                            atomic_json(root/f'training/{phase}_step{step}.json',dict(row=row,result=result))
                            print(json.dumps(row),flush=True)
                        if step in (500,1000):
                            checkpoint=save_checkpoint(root,model,optimizer,stream,protocol,variant=phase,step=step,
                                actual_updates=status['actual_optimizer_updates'],phase_updates=phase_updates,fork_sha=fork_sha)
                            checkpoint_records.append(checkpoint);save_seconds+=checkpoint['save_and_verify_seconds']
                            status['completed_checkpoints']=[r['path'] for r in checkpoint_records]
                            atomic_json(root/'checkpoints/checkpoints.json',dict(checkpoints=checkpoint_records))
                    atomic_json(root/'training/batch_hashes.json',batches)
                    phase_report=dict(phase=phase,load_seconds=load_seconds,OT_seconds=ot_seconds,
                        train_seconds=train_seconds,save_seconds=save_seconds,wall_seconds=time.perf_counter()-phase_tick,
                        actual_updates=current_counts['optimizer_updates'],updates_per_training_second=current_counts['optimizer_updates']/train_seconds,
                        counts=current_counts,OT_counts=stream.counts,scope='one phase; reset after model/optimizer load',
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    atomic_json(root/f'training/{phase}_summary.json',phase_report)
                    del model,optimizer,objective,samples,result,stream;stream=None
                    gc.collect();torch.cuda.empty_cache()
                    atomic_json(root/f'training/{phase}_resources_after.json',resource_snapshot())
            if status['actual_optimizer_updates']!=2500 or phase_updates!={'T1':1000,'Geo_shared':500,'Geo':500,'Full':500}:
                raise RuntimeError('Update budget/accounting mismatch')
            final={}
            for variant in ('T1','Geo','Full'):
                path=root/f'checkpoints/{variant}_step1000.pt'
                model,meta,hashes=load_mvp_checkpoint(path,'cpu')
                final[variant]=dict(path=str(path),metadata=meta,hashes=hashes,strict_reload='PASS')
                del model;gc.collect()
            if file_sha256(args.official_checkpoint)!=OFFICIAL_FILE_SHA256:
                raise RuntimeError('Official source file changed')
            if read_binding(root)[1]!=protocol: raise RuntimeError('Final data verification changed')
            for relative,digest in sources.items():
                if file_sha256(REPO/relative)!=digest: raise RuntimeError('Final code verification changed: '+relative)
            status.update(status='COMPLETE',wall_seconds=time.perf_counter()-total_tick,
                global_gpu_used_peak_mib=monitor.peak_global_used_mib,
                process_gpu_used_peak_mib=monitor.peak_process_used_mib,
                process_peak_note='nvidia-smi matched PID only; null if WSL reports N/A. Global used includes unrelated processes.')
            atomic_json(status_path,status);report(root,status)
            ready=dict(schema='control_mvp48_train_ready_v1',status='READY',owner='Session B',
                actual_optimizer_updates=2500,phase_updates=phase_updates,protocol_sha256=json_hash(protocol),
                data_ready_sha256=protocol['data_ready_sha256'],final_checkpoints=final,
                loader_module='experiments.mf_h1_local0.control_mvp48.model_mvp',
                loader_function='load_mvp_checkpoint',loader_returns=['model','actual_metadata','hashes'],
                loader_sha256=file_sha256(Path(__file__).parent/'model_mvp.py'),
                checkpoints_json=dict(path='checkpoints/checkpoints.json',sha256=file_sha256(root/'checkpoints/checkpoints.json')),
                report=dict(path='training/TRAIN_REPORT.md',sha256=file_sha256(root/'training/TRAIN_REPORT.md')),
                training_log=dict(path='training/training_log.csv',sha256=file_sha256(log_path)),
                code_diff=dict(path='training/code.diff',sha256=file_sha256(root/'training/code.diff')),
                source_hashes=sources,default_model_changed=False)
            atomic_json(root/'TRAIN_READY.json',ready)
    except Exception as exc:
        if current_phase:
            phase_updates[current_phase]=current_counts.get('optimizer_updates',0)
            status['actual_optimizer_updates']=sum(phase_updates.values())
            status['optimizer_step_attempts']=sum(phase_updates.values())-current_counts.get('optimizer_updates',0)+current_counts.get('optimizer_step_attempts',0)
        status.update(status='BLOCKED',error=type(exc).__name__+': '+str(exc),current_counts=current_counts,
            current_OT_counts=dict(stream.counts) if stream is not None else None,
            traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-total_tick)
        atomic_json(status_path,status);report(root,status)
        raise
    finally:
        if monitor: monitor.stop()

def entry():
    try:
        main()
    except Exception as exc:
        (ROOT/'training').mkdir(parents=True,exist_ok=True)
        path=ROOT/'training/status.json'
        state=json.loads(path.read_text()) if path.exists() else dict(actual_optimizer_updates=0,phase_updates={})
        state.update(status='WAITING_DEPENDENCY' if isinstance(exc,WaitingDependency) else 'BLOCKED',
            error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),recovery_command=RECOVERY)
        atomic_json(path,state);report(ROOT,state)
        raise

if __name__=='__main__':
    entry()
