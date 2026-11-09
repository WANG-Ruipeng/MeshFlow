"""Local unattended one-shot continuation. Waiting imports only the standard library."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MPLCONFIGDIR'] = str(__import__('pathlib').Path(__file__).resolve().parents[3]/'native_t1/runs/control_mvp48_v1/evaluation/mplconfig')
import fcntl
import subprocess
import sys
import time
import traceback
from .eval_common import ROOT, RUN, EVALUATION, REGISTRATION, read_json, write_json, sha256, snapshot
from .eval_gpu_worker import verify_execution_plan


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def mark_interrupted(gpu):
    gpu.update(status='STOPPED', reason='GPU process exited without terminal ledger',
               interrupted_request=gpu.get('active_request'), abrupt_exit_counts_may_be_incomplete=True)
    for group in ('rows','teacher_rows'):
        for row in gpu.get(group,[]):
            if row.get('status') == 'RUNNING':
                row.update(status='FAIL', attempted=True, nfe=None, actual_nfe_unknown=True,
                           reason='Interrupted request; actual NFE unknown; never replayed')
    return gpu


def run():
    EVALUATION.mkdir(parents=True, exist_ok=True)
    stream = (EVALUATION/'automation.lock').open('a+')
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (EVALUATION/'automation_started.json').exists():
        raise RuntimeError('This one-shot automation has already started; no implicit restart allowed')
    plan = verify_execution_plan()
    status = dict(schema='control_mvp48_local_automation_v1', owner='Session C',
        pid=os.getpid(), status='WAITING_TRAIN', started_utc=now(), heartbeat_utc=now(),
        data_ready_sha256=sha256(RUN/'DATA_READY.json'),
        plan_sha256=sha256(REGISTRATION/'eval_automation_plan.json'),
        cuda_models_while_waiting=0, forwards_while_waiting=0,
        gpu_workers=0, cpu_scoring_workers=0, automatic_retries=0,
        stop_after='96 registered Euler50 + 24 teacher forwards; no repeat samples',
        report='evaluation/MVP48_REPORT.md')
    write_json(EVALUATION/'automation_started.json', status, exclusive=True)
    def persist(**updates):
        status.update(updates, heartbeat_utc=now())
        write_json(EVALUATION/'automation_status.json', status)
    def halt(reason):
        write_json(EVALUATION/'automation_stop.json', dict(reason=reason, timestamp_utc=now()))
    worker = None
    try:
        persist(resources_before_wait=snapshot())
        print(now(), 'WAITING_TRAIN: CPU-only, no CUDA initialization', flush=True)
        while not (RUN/'TRAIN_READY.json').exists():
            if (EVALUATION/'automation_stop.json').exists():
                raise RuntimeError('Stop requested while waiting')
            if sha256(REGISTRATION/'eval_automation_plan.json') != status['plan_sha256']:
                raise RuntimeError('Bound automation plan changed while waiting')
            if sha256(RUN/'DATA_READY.json') != status['data_ready_sha256']:
                raise RuntimeError('Bound DATA_READY changed while waiting')
            train_path = RUN/'training/status.json'
            training = read_json(train_path) if train_path.exists() else {}
            if training.get('status') in {'BLOCKED','FAILED','FAIL','STOPPED','ERROR'}:
                raise RuntimeError('B ended without TRAIN_READY: '+str(training.get('status')))
            persist(training_status=training.get('status','NOT_READY'),
                    training_phase=training.get('phase'),
                    training_logical_step=training.get('logical_step'),
                    training_actual_optimizer_updates=training.get('actual_optimizer_updates'))
            time.sleep(30)
        verify_execution_plan(status['plan_sha256'])
        persist(status='VERIFYING_ENDPOINTS', resources_before_dispatch=snapshot())
        # The child gates all endpoints on CPU before acquiring/initializing GPU.
        worker_env = dict(os.environ, MVP48_EVAL_PLAN_SHA256=status['plan_sha256'])
        with (EVALUATION/'gpu_worker.log').open('x', encoding='utf-8') as log:
            worker = subprocess.Popen([sys.executable,'-B','-m',
                'experiments.mf_h1_local0.control_mvp48.eval_gpu_worker'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                close_fds=True, env=worker_env)
        persist(status='EVALUATING', gpu_workers=1, gpu_worker_pid=worker.pid)
        scoring = None
        while worker.poll() is None:
            gpu_path = EVALUATION/'gpu_run.json'
            gpu = read_json(gpu_path) if gpu_path.exists() else {}
            persist(gpu_status=gpu.get('status','IDENTITY_GATE'),
                    generated_rows=len(gpu.get('rows',[])), teacher_rows=len(gpu.get('teacher_rows',[])))
            terminal = [r for r in gpu.get('rows',[]) if r.get('status') in {'PASS','FAIL'}]
            if terminal:
                facts = snapshot()
                # One thread/process; keep spare CPU and RAM for inference. No extra OT.
                cpu_ok = facts['cpu_affinity_cores'] >= 2 and facts['cpu_load'][0] < facts['cpu_affinity_cores']-1
                cpu_ok = cpu_ok and int(facts['memory']['MemAvailable'].split()[0]) >= 4*1024*1024
                if cpu_ok:
                    from .eval_postprocess import score_available
                    if scoring is None:
                        persist(cpu_scoring_workers=1, cpu_scoring_start_resources=facts)
                    scoring = score_available()
                    if scoring.get('status') == 'STOPPED':
                        halt('CPU scorer stopped: '+str(scoring.get('reason',scoring.get('error'))))
                    persist(scoring_status=scoring.get('status'))
            time.sleep(5)
        persist(status='FINALIZING', gpu_exit_code=worker.returncode, resources_after_gpu=snapshot())
        gpu_path = EVALUATION/'gpu_run.json'
        if not gpu_path.exists():
            raise RuntimeError('Endpoint gate / worker initialization failed; see gpu_worker.log')
        gpu = read_json(gpu_path)
        if gpu.get('status') not in {'PASS','STOPPED'}:
            # Never replay an interrupted request; its NFE count is uncertain.
            mark_interrupted(gpu)
            write_json(gpu_path, gpu)
        from .eval_postprocess import finalize
        result = finalize()
        final_status = 'COMPLETE' if gpu.get('status') == 'PASS' and worker.returncode == 0 else 'STOPPED'
        scoring_path = EVALUATION/'scoring_run.json'
        if scoring_path.exists() and read_json(scoring_path).get('status') == 'STOPPED':
            final_status = 'STOPPED'
        persist(status=final_status, finished_utc=now(), gpu_workers=0, cpu_scoring_workers=0,
                resources_after=snapshot(), report_sha256=sha256(EVALUATION/'MVP48_REPORT.md'))
        print(now(), final_status, 'Report:', EVALUATION/'MVP48_REPORT.md', flush=True)
        return 0 if final_status == 'COMPLETE' else 1
    except BaseException as exc:
        halt(repr(exc))
        # Do not terminate a CUDA call or kill any process. Child observes the stop
        # marker between registered tasks and preserves the attempted case.
        persist(status='STOPPED', reason=repr(exc), traceback=traceback.format_exc(), finished_utc=now())
        print(traceback.format_exc(), flush=True)
        if worker is not None:
            while worker.poll() is None:
                persist(status='STOPPING', reason=repr(exc))
                time.sleep(5)
            try:
                if (EVALUATION/'gpu_run.json').exists():
                    gpu = read_json(EVALUATION/'gpu_run.json')
                    if gpu.get('status') not in {'PASS','STOPPED'}:
                        mark_interrupted(gpu)
                        write_json(EVALUATION/'gpu_run.json', gpu)
                    from .eval_postprocess import finalize
                    finalize()
            except BaseException as final_exc:
                status['finalization_error'] = repr(final_exc)
        persist(status='STOPPED', gpu_workers=0, cpu_scoring_workers=0, finished_utc=now())
        return 1


if __name__ == '__main__':
    raise SystemExit(run())
