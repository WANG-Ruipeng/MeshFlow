"""One-shot Session C GPU worker. Only the registered 96 + 24 calls are allowed."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import fcntl
import gc
import json
import threading
import time
import traceback
from .eval_common import ROOT, RUN, EVALUATION, REGISTRATION, LOCK, MODELS, read_json, write_json, sha256, snapshot


def verify_execution_plan(expected_sha=None):
    if expected_sha is not None and sha256(REGISTRATION/'eval_automation_plan.json') != expected_sha:
        raise ValueError('Bound execution plan changed')
    plan = read_json(REGISTRATION/'eval_automation_plan.json')
    for relative, digest in plan['frozen_files'].items():
        if sha256(ROOT/relative) != digest:
            raise ValueError('Frozen execution artifact changed: '+relative)
    return plan


def resources_ok(facts, *, initial):
    if facts['gpu']['returncode']:
        return False, 'nvidia-smi failed'
    fields = [x.strip() for x in facts['gpu']['stdout'].splitlines()[0].split(',')]
    total, used, free = map(float, fields[-4:-1])
    headroom = max(2048., .1*total)
    available_ram = int(facts['memory']['MemAvailable'].split()[0])/1024
    # No comparable full-process peak exists for this variable-N model. Observe
    # the first authorized sample only; 4 GiB is a conservative admission reserve.
    required = headroom + (4096 if initial else 0)
    okay = free >= required and available_ram >= 4096
    return okay, dict(total_MiB=total, used_MiB=used, free_MiB=free,
                     headroom_MiB=headroom, initial_admission_reserve_MiB=4096,
                     available_RAM_MiB=available_ram, initial=initial)


def save_npz(path, arrays):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    return dict(raw_path=str(path.relative_to(RUN)), raw_sha256=sha256(path))


def run():
    # Do not import torch, build a CUDA model, or take the GPU lock while B trains.
    expected_plan_sha = os.environ['MVP48_EVAL_PLAN_SHA256']
    plan = verify_execution_plan(expected_plan_sha)
    from .eval_checkpoint_gate import inspect_ready, validate_loaded
    gate = inspect_ready()
    write_json(REGISTRATION/'eval_endpoint_identity.json', gate, exclusive=True)
    ledger = dict(schema='control_mvp48_gpu_run_v1', status='WAITING_GPU_LOCK',
                  pid=os.getpid(), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  rows=[], teacher_rows=[], identities=gate, phase_resources=[],
                  sampling_counts={}, teacher_counts={}, requested_samples=96,
                  requested_teacher=24, repeated_samples=0, actual_gpu_workers=1,
                  cuda_models_created=0, active_request=None,
                  concurrency_reason='Explicit Session C single GPU worker; one CPU scorer may overlap.')
    target = EVALUATION/'gpu_run.json'
    if target.exists():
        raise RuntimeError('Existing GPU ledger: automatic replay prohibited')
    write_json(EVALUATION/'gpu_started.json', dict(pid=os.getpid()), exclusive=True)
    write_json(target, ledger)
    global _PROCESS_LOCK
    lock_stream = open(LOCK, 'a+')
    _PROCESS_LOCK = lock_stream  # Retain until process exit (CUDA context included).
    stop_monitor = threading.Event()
    monitor = None
    tick = time.perf_counter()
    pressure = []
    observed_peaks = dict(global_gpu_used_peak_MiB=None, process_gpu_used_peak_MiB=None, samples=0, process_peak_note='Matched actual PID in nvidia-smi; null means unavailable, never zero. Global peak includes unrelated processes.')
    def persist():
        ledger['resources'] = dict(phase_snapshots=ledger['phase_resources'], monitor='evaluation/gpu_resources.jsonl', finished=ledger.get('finished_resources'))
        ledger['phase_timing'] = dict(total_worker_seconds=time.perf_counter()-tick, sampling_seconds=sum(r.get('audit',{}).get('seconds',0) for r in ledger['rows']), teacher_seconds=sum(r.get('seconds',0) for r in ledger['teacher_rows']))
        ledger['seconds'] = time.perf_counter()-tick
        write_json(target, ledger)
    def check_phase(phase, initial=False):
        if sha256(REGISTRATION/'eval_automation_plan.json') != expected_plan_sha:
            raise RuntimeError('Bound automation plan changed during dispatch')
        if (EVALUATION/'automation_stop.json').exists():
            raise RuntimeError('Automation stop requested: '+str(read_json(EVALUATION/'automation_stop.json')))
        facts = snapshot()
        okay, detail = resources_ok(facts, initial=initial)
        ledger['phase_resources'].append(dict(phase=phase, snapshot=facts, admission=detail))
        persist()
        if pressure or not okay:
            raise RuntimeError('Resource headroom lost; dispatch stopped: '+str(detail))
    def observe():
        with (EVALUATION/'gpu_resources.jsonl').open('x', encoding='utf-8') as stream:
            while not stop_monitor.is_set():
                try:
                    facts = snapshot()
                    okay, detail = resources_ok(facts, initial=False)
                    observed_peaks['samples'] += 1
                    if isinstance(detail,dict):
                        observed_peaks['global_gpu_used_peak_MiB'] = max(observed_peaks['global_gpu_used_peak_MiB'] or 0, detail['used_MiB'])
                    for line in facts['gpu_processes']['stdout'].splitlines()[1:]:
                        parts = [v.strip() for v in line.split(',')]
                        if parts and parts[0] == str(os.getpid()):
                            try:
                                memory = float(parts[-1].split()[0])
                                observed_peaks['process_gpu_used_peak_MiB'] = max(observed_peaks['process_gpu_used_peak_MiB'] or 0,memory)
                            except ValueError:
                                pass
                    stream.write(json.dumps(dict(snapshot=facts, admission=detail))+'\n'); stream.flush()
                    if not okay:
                        pressure.append(detail)
                except Exception as exc:
                    pressure.append(repr(exc))
                stop_monitor.wait(5)
    try:
        while True:
            if (EVALUATION/'automation_stop.json').exists():
                raise RuntimeError('Stopped before GPU lock')
            try:
                fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                facts = snapshot(); okay, detail = resources_ok(facts, initial=True)
                ledger['phase_resources'].append(dict(phase='lock_admission', snapshot=facts, admission=detail)); persist()
                if okay:
                    break
                fcntl.flock(lock_stream, fcntl.LOCK_UN)
            except BlockingIOError:
                pass
            time.sleep(15)
        ledger['lock_wait_seconds'] = time.perf_counter()-tick
        ledger['status'] = 'CPU_STRICT_LOAD'; persist()
        import numpy as np
        import torch
        from .model_mvp import load_mvp_checkpoint, state_sha256
        from .eval_runtime import new_counts, sample_euler50, SamplingFailure
        from .eval_teacher import evaluate_teacher_once, TeacherFailure
        from .eval_prepare import array_hash
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        ledger['sampling_counts'] = new_counts()
        ledger['teacher_counts'] = new_counts()
        # Strictly load all three on CPU before any test forward or CUDA model.
        for label in MODELS:
            model, metadata, hashes = load_mvp_checkpoint(gate['endpoints'][label]['path'], 'cpu')
            validate_loaded(label, metadata, hashes, gate)
            del model; gc.collect()
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU strict load unexpectedly initialized CUDA')
        ledger['all_three_strict_cpu_loads'] = 'PASS'
        prep = read_json(REGISTRATION/'eval_preparation_ledger.json')
        document = read_json(REGISTRATION/'eval_requests.json')
        requests = document['requests'] if isinstance(document, dict) else document
        if len(requests) != 96 or len(prep['test_inputs']) != 32 or len(prep['teacher_inputs']) != 8:
            raise ValueError('Registered request count differs')
        monitor = threading.Thread(target=observe, daemon=True); monitor.start()
        for label in MODELS:
            check_phase(label+'_load', initial=True)
            ledger['status'] = 'RUNNING'; ledger['active_model'] = label; persist()
            model, metadata, hashes = load_mvp_checkpoint(gate['endpoints'][label]['path'], 'cuda:0')
            ledger['cuda_models_created'] += 1
            validate_loaded(label, metadata, hashes, gate)
            for request in [r for r in requests if r['model'] == label]:
                check_phase(request['request_id'])
                row = dict(request, status='RUNNING', nfe=0)
                ledger['rows'].append(row); ledger['active_request'] = row['request_id']; persist()
                before = dict(ledger['sampling_counts'])
                arrays = {}; states = None; audit = {}; raw = None
                try:
                    path = RUN/request['path']
                    if sha256(path) != request['sha256']:
                        raise ValueError('Input bank file changed')
                    with np.load(path, allow_pickle=False) as data:
                        arrays = {k:data[k].copy() for k in data.files}
                    if set(arrays) != {'C','z','N','known_mask','valid_mask'}:
                        raise ValueError('Free generation input allowlist failed')
                    N, K = request['N'], request['K']
                    if arrays['z'].shape != (1,N,9) or int(arrays['N']) != N or len(arrays['C']) != K:
                        raise ValueError('Actual variable N/K differs')
                    for key in ('C','z','known_mask','valid_mask'):
                        if array_hash(arrays[key]) != request[key+'_sha256']:
                            raise ValueError('Frozen array changed: '+key)
                    if not np.array_equal(arrays['known_mask'], np.arange(N)<K) or not arrays['valid_mask'].all():
                        raise ValueError('Known/valid mask differs')
                    z = torch.from_numpy(arrays['z']).to('cuda:0')
                    C = torch.from_numpy(arrays['C']).to('cuda:0')
                    raw, states, audit = sample_euler50(model, z, C, ledger['sampling_counts'])
                    row.update(status='PASS', audit=audit)
                except SamplingFailure as exc:
                    states = exc.partial_states; raw = states[-1]; audit = exc.audit
                    row.update(status='FAIL', reason=repr(exc), audit=audit)
                    raise
                except BaseException as exc:
                    row.update(status='FAIL', reason=repr(exc), audit=audit)
                    raise
                finally:
                    delta = {k:ledger['sampling_counts'][k]-before[k] for k in before}
                    row.update(nfe=delta['backbone_forward_attempts'], counts_delta=delta,
                               input_file_sha256=request['sha256'])
                    if states is not None:
                        try:
                            row.update(save_npz(EVALUATION/'raw'/(row['request_id']+'.npz'),
                                dict(raw=raw, states=states, C=arrays['C'], known_mask=arrays['known_mask'],
                                     valid_mask=arrays['valid_mask'], accepted_times=np.asarray(audit['accepted_times']))))
                        except BaseException as exc:
                            row.update(status='FAIL', save_error=repr(exc)); persist(); raise
                    ledger['active_request'] = None; persist()
                del z, C
            for bank in prep['teacher_inputs']:
                request_id = bank['teacher_id']+'_'+label
                check_phase(request_id)
                row = dict(parent_id=bank['parent_id'], condition_id=bank['condition_id'],
                           model=label, request_id=request_id, status='RUNNING', nfe=0)
                ledger['teacher_rows'].append(row); ledger['active_request'] = request_id; persist()
                before = dict(ledger['teacher_counts']); outputs = {}
                try:
                    if sha256(RUN/bank['path']) != bank['sha256']:
                        raise ValueError('Frozen teacher bank changed')
                    with np.load(RUN/bank['path'], allow_pickle=False) as data:
                        arrays = {k:data[k].copy() for k in data.files}
                    result, outputs = evaluate_teacher_once(model, (arrays, bank['identity']), ledger['teacher_counts'])
                    row.update(result)
                    if not row.get('finite_velocity', False):
                        raise RuntimeError('Nonfinite teacher output')
                except TeacherFailure as exc:
                    row.update(exc.row); outputs = exc.arrays; raise
                except BaseException as exc:
                    row.update(status='FAIL', reason=repr(exc)); raise
                finally:
                    delta = {k:ledger['teacher_counts'][k]-before[k] for k in before}
                    row.update(nfe=delta['backbone_forward_attempts'], counts_delta=delta,
                               input_file_sha256=bank['sha256'])
                    if outputs:
                        try:
                            saved = save_npz(EVALUATION/'teacher_raw'/(request_id+'.npz'), outputs)
                            row.update(output_path=saved['raw_path'], output_sha256=saved['raw_sha256'])
                        except BaseException as exc:
                            row.update(status='FAIL', save_error=repr(exc)); persist(); raise
                    ledger['active_request'] = None; persist()
            if state_sha256(model) != hashes['model_state_sha256']:
                raise RuntimeError('Model parameters changed during evaluation')
            del model; gc.collect(); torch.cuda.empty_cache()
            check_phase(label+'_finished')
        if ledger['sampling_counts']['backbone_forward_attempts'] != 4800 or ledger['teacher_counts']['backbone_forward_attempts'] != 24:
            raise RuntimeError('Final NFE accounting mismatch')
        if len(ledger['rows']) != 96 or len(ledger['teacher_rows']) != 24:
            raise RuntimeError('Final output count mismatch')
        ledger['status'] = 'PASS'
        verify_execution_plan(expected_plan_sha)
    except BaseException as exc:
        ledger.update(status='STOPPED', reason=repr(exc), traceback=traceback.format_exc())
    finally:
        stop_monitor.set()
        if monitor is not None:
            monitor.join(timeout=20)
        ledger['observed_process_peaks'] = observed_peaks
        ledger['throughput'] = dict(complete_rollouts_per_worker_second=ledger['sampling_counts'].get('complete_rollouts',0)/max(time.perf_counter()-tick,1e-9), timing_note='Includes loading, waits, teacher checks and overlapping CPU scoring; not an isolated compute benchmark')
        ledger['finished_resources'] = snapshot()
        ledger['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        persist()
    # lock_stream is intentionally retained until this process exits.
    return 0 if ledger['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(run())
