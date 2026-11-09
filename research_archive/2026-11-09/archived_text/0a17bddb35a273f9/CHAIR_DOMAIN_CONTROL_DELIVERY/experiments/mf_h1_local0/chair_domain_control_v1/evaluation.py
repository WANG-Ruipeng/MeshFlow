"""Fixed-budget chair-domain evaluation. The controller owns GPU dispatch."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import gc
from pathlib import Path
import time
import traceback
import numpy as np
from .common import atomic_json, read_json, digest, GPU_LOCK
from experiments.mf_h1_local0.control_mvp48_budget_single.evaluation import (
    resolve, arrays, save_npz, _verified, _bank_reference, teacher_once,
)

ARMS = ('T1', 'Geo', 'Full')
PHASES = (0, 1000, 2500, 3000)
DISPLAY = {'T1': 'Native', 'Geo': 'Native+Geo', 'Full': 'Native+Geo+Joint'}
GENERATION_LIMIT = 600
TEACHER_LIMIT = 288


def records_for_phase(panels, phase_step):
    records = [dict(r, table=table) for table in ('A', 'B', 'C')
               for r in panels[table] if phase_step in r['phase_steps']]
    if len({r['case_id'] for r in records}) != len(records):
        raise ValueError('Duplicate request identifiers in evaluation phase')
    if len(records) != {0:40, 1000:8, 2500:8, 3000:144}[phase_step]:
        raise ValueError('Requested panel count differs from the fixed protocol')
    return records


def checkpoint_path(source, out, arm, phase_step):
    if phase_step == 0:
        return Path(source)/'checkpoints'/f'{arm}_step1000.pt'
    return Path(out)/'checkpoints'/f'{arm}_phase{phase_step}_total{1000+phase_step}.pt'


def load_model(source, out, arm, phase_step):
    path = checkpoint_path(source, out, arm, phase_step)
    if phase_step == 0:
        from experiments.mf_h1_local0.control_mvp48.model_mvp import load_mvp_checkpoint
        return load_mvp_checkpoint(path, 'cuda:0')
    from .checkpoint import load_checkpoint
    return load_checkpoint(path, device='cuda:0', trainable=False)


def scoring_bank(out, record):
    bank = _verified(out, record)
    if 'gt_free' not in bank:
        bank.update(arrays(resolve(out, record.get('target_path', record['ref_npz']))))
    return bank


def score_record(out, record, raw, *, ledger=None):
    """Original independent scoring; all target access occurs after sampling."""
    from experiments.mf_h1_local0.control_mvp48.eval_scoring import score_free
    bank = scoring_bank(out, record)
    reference = _bank_reference(out, record)
    N, K = int(bank['N']), len(bank['C'])
    raw = np.asarray(raw)
    if raw.shape != (N, 9):
        raise ValueError('Actual generated N/shape differs from frozen request')
    preserved = np.array_equal(raw[:K].reshape(K, 3, 3), bank['C'])
    result = score_free(raw[K:], bank['C'], bank['gt_free'], float(bank['L']),
                        reference=reference, ledger=ledger, include_pair_audit=True)
    result['metrics']['C_preserved'] = preserved
    result['detail']['actual_condition_preserved'] = preserved
    result['detail']['GT_was_model_input'] = False
    if record['table'] == 'B':
        L, L0 = float(bank['L']), float(bank['L0'])
        if not np.isfinite(L0) or L0 <= 0:
            raise ValueError('Invalid frozen undeformed L0')
        result['metrics'].update(L_alpha=L, L0=L0)
        for name, value in list(result['metrics'].items()):
            if name.endswith('_bbox_pct'):
                result['metrics'][name.replace('_bbox_pct', '_L0_pct')] = (
                    float(value)*L/L0 if value is not None else None)
    if not preserved:
        result.update(status='INVALID_OUTPUT', failure_reason='C_MISMATCH')
    return result


def _global_attempts(out):
    generations = teachers = 0
    for path in (Path(out)/'evaluation').glob('phase*/*/ledger.json'):
        data = read_json(path)
        generations += sum(bool(r.get('attempt_reserved')) for r in data.get('rows', []))
        teachers += sum(bool(r.get('attempt_reserved')) for r in data.get('teacher_rows', []))
    return generations, teachers


def _row(record, arm, phase_step):
    return dict(case_id=record['case_id'], arm=arm, display_model=DISPLAY[arm],
                phase_step=phase_step, conditional_total_step=1000+phase_step,
                table=record.get('table', 'teacher'), uid=record.get('uid'),
                split=record.get('split'), task_id=record.get('task_id'),
                patch_id=record.get('patch_id'), ratio=record.get('ratio'),
                alpha=record.get('alpha', 1.), noise_index=record.get('noise_index'),
                teacher_t=record.get('t'))


def run(source, out, arm, phase_step, score_only=False):
    source, out = Path(source), Path(out)
    if arm not in ARMS or phase_step not in PHASES:
        raise ValueError('Unregistered model or phase')
    panels = read_json(out/'panels.json')
    records = records_for_phase(panels, phase_step)
    teachers = panels['teacher']
    if len(teachers) != 24:
        raise ValueError('Fixed teacher requests must retain all 24 registered slots')
    for record in records+teachers:
        if record.get('status') not in ('NOT_APPLICABLE', 'NOT_AVAILABLE'):
            if digest(resolve(out, record['input_path'])) != record['input_sha256']:
                raise ValueError('Frozen evaluation input changed')
    checkpoint = checkpoint_path(source, out, arm, phase_step)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    identity = dict(arm=arm, phase_step=phase_step, conditional_total_step=1000+phase_step,
                    panels_sha256=digest(out/'panels.json'),
                    protocol_sha256=digest(out/'protocol.json'),
                    checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))
    folder = out/'evaluation'/f'phase{phase_step}'/arm
    target = folder/'ledger.json'
    ledger = read_json(target) if target.exists() else dict(
        schema='chair_domain_evaluation_v1', identity=identity, status='PREPARED',
        rows=[], teacher_rows=[], sampling_counts={}, teacher_counts={},
        requested_generations=len(records), requested_teacher=len(teachers),
        actual_gpu_workers=1, scoring_counts={})
    if ledger['identity'] != identity:
        raise ValueError('Existing evaluation identity differs')
    if any(r.get('status') in ('RUNNING', 'FAIL', 'STOPPED')
           for r in ledger['rows']+ledger['teacher_rows']):
        raise RuntimeError('Interrupted/failed attempt exists; automatic replay prohibited')

    def persist():
        atomic_json(target, ledger)

    def score_pending():
        for row in ledger['rows']:
            if row['status'] != 'PASS':
                continue
            if digest(resolve(out, row['raw_path'])) != row['raw_sha256']:
                raise ValueError('Saved raw changed')
            if row.get('score_path'):
                if digest(resolve(out, row['score_path'])) != row['score_sha256']:
                    raise ValueError('Saved score changed')
                continue
            if row.get('scoring_status') in ('RUNNING', 'FAIL'):
                raise RuntimeError('Interrupted CPU scoring cannot silently replay')
            raw_path = resolve(out, row['raw_path'])
            if digest(raw_path) != row['raw_sha256']:
                raise ValueError('Saved raw changed')
            record = next(r for r in records if r['case_id'] == row['case_id'])
            row['scoring_status'] = 'RUNNING'
            ledger['scoring_counts']['score_attempts'] = ledger['scoring_counts'].get('score_attempts', 0)+1
            persist()
            started = time.perf_counter()
            try:
                score = score_record(out, record, arrays(raw_path)['raw'], ledger=ledger['scoring_counts'])
                loc = folder/'scores'/(row['case_id']+'.json')
                atomic_json(loc, score)
                ledger['scoring_counts']['score_returns'] = ledger['scoring_counts'].get('score_returns', 0)+1
                row.update(score_path=str(loc.relative_to(out)), score_sha256=digest(loc),
                           score_status=score['status'], scoring_status='COMPLETE')
            except BaseException as exc:
                row.update(scoring_status='FAIL', scoring_reason=repr(exc))
                raise
            finally:
                row['scoring_seconds'] = time.perf_counter()-started
                persist()

    for row in ledger['teacher_rows']:
        if row['status'] == 'PASS' and digest(resolve(out, row['raw_path'])) != row['raw_sha256']:
            raise ValueError('Saved teacher raw changed')
    if score_only or ledger['status'] in ('PASS', 'GPU_COMPLETE'):
        score_pending()
        if ledger['status'] == 'GPU_COMPLETE':
            ledger['status'] = 'PASS'; persist()
        return ledger
    import fcntl
    from experiments.mf_h1_local0.control_mvp48.eval_common import snapshot
    from experiments.mf_h1_local0.control_mvp48.eval_gpu_worker import resources_ok
    lock = open(GPU_LOCK, 'a+')
    start = time.perf_counter()
    ledger.update(status='WAITING_GPU_LOCK', pid=os.getpid()); persist()
    try:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(5); continue
            facts = snapshot(); ok, admission = resources_ok(facts, initial=True)
            ledger.update(resources_before=facts, admission=admission); persist()
            if ok:
                break
            fcntl.flock(lock, fcntl.LOCK_UN); time.sleep(5)
        ledger['lock_wait_seconds'] = time.perf_counter()-start
        # No CUDA initialization or model construction precedes the shared flock.
        import torch
        from native_t1.artifacts import array_hash
        from experiments.mf_h1_local0.control_mvp48.eval_runtime import new_counts, sample_euler50, SamplingFailure
        from experiments.mf_h1_local0.control_mvp48.model_mvp import state_sha256
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        ledger['sampling_counts'] = ledger['sampling_counts'] or new_counts()
        ledger['teacher_counts'] = ledger['teacher_counts'] or new_counts()
        load_start = time.perf_counter()
        model, metadata, hashes = load_model(source, out, arm, phase_step)
        ledger.update(model_metadata=metadata, model_hashes=hashes,
                      model_load_seconds=time.perf_counter()-load_start, status='RUNNING'); persist()
        for teacher, requests, output_rows, counts in (
            (False, records, ledger['rows'], ledger['sampling_counts']),
            (True, teachers, ledger['teacher_rows'], ledger['teacher_counts'])):
            from experiments.mf_h1_local0.control_mvp48.train_run import GPUMonitor
            stage_name = 'teacher' if teacher else 'sampling'
            stage_start = time.perf_counter()
            stage_counts_before = dict(counts)
            torch.cuda.reset_peak_memory_stats()
            monitor_root = folder/'resources'/stage_name
            (monitor_root/'training').mkdir(parents=True, exist_ok=True)
            monitor = GPUMonitor(monitor_root); monitor.start()
            try:
                existing = {r['case_id']:r for r in output_rows}
                for record in requests:
                    rid = record['case_id']
                    if rid in existing:
                        continue
                    row = _row(record, arm, phase_step)
                    if record.get('status') in ('NOT_APPLICABLE', 'NOT_AVAILABLE'):
                        row.update(status='NOT_APPLICABLE', reason=record.get('reason'), attempt_reserved=False)
                        output_rows.append(row); persist(); continue
                    gs, ts = _global_attempts(out)
                    if gs+int(not teacher)>GENERATION_LIMIT or ts+int(teacher)>TEACHER_LIMIT:
                        raise RuntimeError('Global evaluation hard budget exhausted')
                    facts = snapshot(); ok, admission = resources_ok(facts, initial=False)
                    ledger.setdefault('request_resources', []).append(dict(case_id=rid, teacher=teacher, snapshot=facts, admission=admission))
                    if not ok:
                        raise RuntimeError('GPU headroom lost')
                    bank = _verified(out, record)
                    row.update(status='RUNNING', attempt_reserved=True,
                               input_sha256=digest(resolve(out, record['input_path'])))
                    output_rows.append(row); persist()
                    before = dict(counts)
                    try:
                        if teacher:
                            result, raw_arrays = teacher_once(model, bank, record, counts)
                            row.update(result)
                        else:
                            # The sampler receives only iid z, C, and their implied N/role.
                            # alpha/UID/GT/L/L0/target widths stay outside model calls.
                            z = torch.from_numpy(bank['z']).to('cuda:0')
                            C = torch.from_numpy(bank['C']).to('cuda:0')
                            raw, states, audit = sample_euler50(model, z, C, counts)
                            raw_arrays = dict(raw=raw, noise=bank['z'], C=bank['C'], known_mask=bank['known_mask'])
                            if record.get('save_states', False):
                                raw_arrays['states'] = states
                            row.update(status='PASS', audit=audit,
                                       output_array_sha256={k:array_hash(v) for k,v in raw_arrays.items()})
                        loc = folder/('teacher_raw' if teacher else 'raw')/(rid+'.npz')
                        row.update(raw_path=str(loc.relative_to(out)), raw_sha256=save_npz(loc, raw_arrays))
                    except SamplingFailure as exc:
                        loc = folder/'failed_raw'/(rid+'.npz')
                        row.update(status='FAIL', reason=str(exc), audit=exc.audit,
                                   raw_path=str(loc.relative_to(out)), raw_sha256=save_npz(loc, dict(states=exc.partial_states)))
                        raise
                    except BaseException as exc:
                        partial = getattr(exc, 'partial_teacher_arrays', None)
                        if partial:
                            loc = folder/'failed_teacher_raw'/(rid+'.npz')
                            row.update(raw_path=str(loc.relative_to(out)), raw_sha256=save_npz(loc, partial))
                        row.update(status='FAIL', reason=repr(exc)); raise
                    finally:
                        row['counts_delta'] = {k:counts[k]-before[k] for k in counts}; persist()
            finally:
                monitor.stop()
                scopes = [r.get('audit', r) for r in output_rows]
                stage = dict(stage=stage_name, elapsed_seconds=time.perf_counter()-stage_start,
                    cuda_peak_allocated_bytes=max([torch.cuda.max_memory_allocated()]+[r.get('peak_cuda_allocated_bytes',0) for r in scopes]),
                    cuda_peak_reserved_bytes=max([torch.cuda.max_memory_reserved()]+[r.get('peak_cuda_reserved_bytes',0) for r in scopes]),
                    global_gpu_used_peak_mib=monitor.peak_global_used_mib,
                    process_gpu_used_peak_mib=monitor.peak_process_used_mib,
                    counts_delta={k:counts[k]-stage_counts_before[k] for k in counts},
                    exclusive_network_seconds=None,
                    peak_scope='Maximum per-request reset peaks in this sampling/teacher stage; includes resident model/cache; allocated is within reserved',
                    elapsed_scope='Stage loop including resources, native computation, raw saves and teacher CPU diagnostics; excludes model load and final geometry scoring',
                    monitor_path=str((monitor_root/'training/gpu_samples.jsonl').relative_to(out)))
                ledger.setdefault('stage_resource_attempts',[]).append(stage)
                ledger.setdefault('stage_resources',{})[stage_name]=stage; persist()
        if state_sha256(model) != hashes['model_state_sha256']:
            raise RuntimeError('Evaluation mutated model parameters')
        del model; gc.collect(); torch.cuda.empty_cache()
        ledger.update(status='GPU_COMPLETE', resources_after=snapshot(),
                      completed_generations=ledger['sampling_counts']['complete_rollouts'],
                      completed_teacher=ledger['teacher_counts']['teacher_returns'],
                      geometry_inapplicable_generations=sum(r['status']=='NOT_APPLICABLE' for r in ledger['rows']),
                      geometry_inapplicable_teacher=sum(r['status']=='NOT_APPLICABLE' for r in ledger['teacher_rows']))
        persist()
    except BaseException as exc:
        ledger.update(status='STOPPED', reason=repr(exc), traceback=traceback.format_exc()); persist(); raise
    finally:
        # Keep the shared lock until process exit, including the live CUDA context.
        global _PROCESS_LOCK
        _PROCESS_LOCK = lock
    score_pending()
    ledger.update(status='PASS', worker_seconds=time.perf_counter()-start,
                  score_valid_generations=sum(r.get('score_status')=='PASS' for r in ledger['rows']))
    persist()
    return ledger


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True); p.add_argument('--out', required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--phase-step', type=int, choices=PHASES, required=True)
    p.add_argument('--score-only', action='store_true')
    a = p.parse_args(); run(a.source, a.out, a.arm, a.phase_step, a.score_only)


if __name__ == '__main__':
    main()
