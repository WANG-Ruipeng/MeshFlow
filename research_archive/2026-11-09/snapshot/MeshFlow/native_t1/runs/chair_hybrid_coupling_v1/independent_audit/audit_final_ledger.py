"""Independent, standard-library-only final ledger/exposure audit.

Do not run while a worker is active. This file lives outside frozen experiment
code. It imports no experiment module, NumPy, torch, model, or scorer, and never
changes a source artifact. Missing/failed evidence yields BLOCKED; no rows are
filled in and no experiments are dispatched. Endpoint PT files are byte-hashed,
never deserialized; their metadata are checked from the bound JSON sidecars.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import traceback

ARMS = ('OT_BASE', 'OT_HYBRID')
FORWARDS = ('training_forward', 'sampling_forward', 'diagnostic_bf16', 'diagnostic_fp32', 'precheck_forward')
EXPECTED_TRAINING = {'training_forward': 16000, 'training_backward': 16000, 'optimizer_update': 2000}
SOURCE_SHA = '190a16e2aec96a1f84f63460b5f97b00e23499338102f82e7530218ecb185ebf'


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stat_identity(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def dump_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def dump_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows); f.flush(); os.fsync(f.fileno())


class Audit:
    def __init__(self, out):
        self.out = Path(out).resolve()
        self.dest = self.out/'independent_audit'
        self.dest.mkdir(parents=True, exist_ok=True)
        self.inputs = {}
        self.checks = Counter()
        self.failures = Counter()
        self.examples = []
        self.sections = {}

    def require(self, condition, check, detail=None):
        self.checks[check] += 1
        if not condition:
            self.failures[check] += 1
            if len(self.examples) < 150:
                self.examples.append({'check': check, 'detail': detail})
        return bool(condition)

    def record_input(self, path, before, sha):
        path = Path(path).resolve()
        self.require(stat_identity(path) == before, 'input_unchanged_during_read', str(path))
        if str(path) in self.inputs:
            self.require(self.inputs[str(path)]['sha256'] == sha, 'input_same_across_reads', str(path))
        self.inputs[str(path)] = dict(before, sha256=sha)

    def read_bytes(self, path):
        path = Path(path)
        before = stat_identity(path)
        value = path.read_bytes()
        self.record_input(path, before, hashlib.sha256(value).hexdigest())
        return value

    def json(self, path):
        return json.loads(self.read_bytes(path).decode('utf-8-sig'))

    def csv(self, path):
        return list(csv.DictReader(io.StringIO(self.read_bytes(path).decode('utf-8-sig'))))

    def jsonl(self, path):
        path = Path(path)
        before = stat_identity(path); h = hashlib.sha256(); rows = []
        with path.open('rb') as f:
            for number, raw in enumerate(f, 1):
                h.update(raw)
                if raw.strip():
                    try:
                        rows.append(json.loads(raw))
                    except Exception as exc:
                        raise ValueError(f'Invalid complete JSONL row {path}:{number}: {exc}') from exc
        self.record_input(path, before, h.hexdigest())
        return rows

    def hash_file(self, path):
        path = Path(path); before = stat_identity(path); h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
        result = h.hexdigest(); self.record_input(path, before, result)
        return result

    def section(self, name, operation):
        before = sum(self.failures.values())
        try:
            value = operation()
            self.sections[name] = {'status': 'PASS' if sum(self.failures.values()) == before else 'BLOCKED'}
            return value
        except Exception as exc:
            self.require(False, name+'_exception', repr(exc))
            self.sections[name] = {'status': 'BLOCKED', 'error': repr(exc), 'traceback': traceback.format_exc()}
            return None


def readiness(a, require_complete):
    for arm in ARMS:
        stage = a.json(a.out/'stages'/f'train_{arm}.json')
        status = a.json(a.out/'training'/f'{arm}_status.json')
        a.require(stage.get('status') == 'DONE', 'training_stage_done', arm)
        a.require(status.get('status') == 'DONE' and status.get('pilot_step') == 1000,
                  'training_status_final_1000', arm)
    for folder in ('stages', 'controller'):
        for path in sorted((a.out/folder).glob('*.json')):
            value = a.json(path)
            a.require(value.get('status') not in ('RUNNING', 'TRAINING', 'BLOCKED', 'INTERRUPTED_SAFE'),
                      'no_active_or_blocked_worker', {'path': str(path), 'status': value.get('status')})
    complete_path = a.out/'RUN_COMPLETE.json'
    if require_complete:
        value = a.json(complete_path)
        a.require(value.get('status') == 'DONE', 'run_complete_marker')
    return {'require_run_complete': require_complete, 'RUN_COMPLETE_exists': complete_path.exists()}


def classify_ledger_row(row, path):
    """Recognize exact observed CPU scoring metadata; never hide a CALL row."""
    if not isinstance(row, dict):
        raise ValueError('Ledger record is not an object')
    event = row.get('event')
    if event in ('CALL_BEGIN', 'CALL_RETURN', 'CALL_FAILED'):
        required = {'event', 'call_id', 'kind', 'stage', 'context'}
        if event != 'CALL_BEGIN':
            required |= {'error', 'wall_seconds'}
        if not required <= set(row) or not isinstance(row['context'], dict):
            raise ValueError('Malformed known CALL record: '+repr(row))
        if any(not isinstance(row[key], str) or not row[key] for key in ('call_id','kind','stage')):
            raise ValueError('Invalid CALL identity fields')
        return 'CALL'
    # A missing/unknown CALL event or call identity is an error, even if its
    # kind happens to resemble the recognized CPU metadata below.
    if 'event' in row or any(key in row for key in ('call_id','stage','context','wall_seconds','error')):
        raise ValueError('Unknown or malformed CALL/event record: '+repr(row))
    common = {'kind','pid','time_unix','checkpoint_label'}
    schemas = {
        'score_attempt': common | {'case_id'},
        'score_return': common | {'case_id','elapsed_seconds','historical_cache','status'},
        'width_group_attempt': common | {'uid','noise_index'},
        'width_group_return': common | {'uid','noise_index','elapsed_seconds','model_forwards','surface_distance_calls'},
    }
    kind = row.get('kind')
    if Path(path).name != 'cpu_scoring.jsonl' or kind not in schemas or set(row) != schemas[kind]:
        raise ValueError('Unrecognized non-CALL metadata schema: '+repr(row))
    if row['checkpoint_label'] not in ('START','OT_BASE','OT_HYBRID'):
        raise ValueError('Unknown CPU scoring checkpoint label')
    if 'model_forwards' in row and row['model_forwards'] != 0:
        raise ValueError('CPU metadata claims a model forward; cannot exclude from call audit')
    if 'elapsed_seconds' in row and (not math.isfinite(float(row['elapsed_seconds'])) or row['elapsed_seconds'] < 0):
        raise ValueError('Invalid CPU scoring metadata duration')
    return 'CPU_SCORING_METADATA'


def audit_calls(a, protocol):
    attempts, returns, failed, seconds = Counter(), Counter(), Counter(), Counter()
    begins, terminals, pending = {}, {}, {}
    metadata_rows = []; metadata_counts = Counter()
    paths = sorted((a.out/'ledger').glob('*.jsonl'))
    a.require(bool(paths), 'actual_call_ledgers_present')
    for path in paths:
        for record_index, row in enumerate(a.jsonl(path), 1):
            classification = classify_ledger_row(row, path)
            if classification == 'CPU_SCORING_METADATA':
                metadata_counts[row['kind']] += 1
                metadata_rows.append(dict(source_file=str(path), nonblank_record_index=record_index,
                    classification=classification, excluded_from_model_CALL_counts=True, original_record=row))
                continue
            event, cid, kind = row['event'], row['call_id'], row['kind']
            a.require(kind in protocol['budget'], 'registered_call_kind', {'call_id': cid, 'kind': kind})
            a.require(row['stage'] == path.stem, 'call_stage_matches_file', cid)
            if event == 'CALL_BEGIN':
                a.require(cid not in begins, 'unique_call_id', cid)
                begins[cid] = row; pending[cid] = row; attempts[kind] += 1
            elif event in ('CALL_RETURN', 'CALL_FAILED'):
                a.require(cid in pending, 'terminal_has_pending_begin', cid)
                a.require(cid not in terminals, 'single_terminal_per_call', cid)
                begin = pending.pop(cid, None)
                if begin is not None:
                    a.require(all(begin.get(k) == row.get(k) for k in ('kind', 'stage', 'context')),
                              'begin_terminal_identity', cid)
                terminals[cid] = row
                (returns if event == 'CALL_RETURN' else failed)[kind] += 1
                duration = float(row['wall_seconds'])
                a.require(math.isfinite(duration) and duration >= 0, 'finite_nonnegative_call_time', cid)
                seconds[kind] += duration
                a.require((row.get('error') is None) == (event == 'CALL_RETURN'), 'terminal_error_semantics', cid)
            else:
                a.require(False, 'known_call_event', row)
    metadata_summary = dict(total_rows=len(metadata_rows), kinds=dict(metadata_counts),
        scope='Exact recognized cpu_scoring.jsonl attempt/return metadata; preserved explicitly and excluded only from actual model CALL_* counters. Unknown CALL/non-CALL schemas fail.')
    dump_json(a.dest/'non_call_metadata_rows.json',metadata_rows)
    dump_json(a.dest/'non_call_metadata_summary.json',metadata_summary)
    a.require(not pending, 'no_unresolved_actual_calls', list(pending)[:10])
    a.require(not sum(failed.values()), 'no_failed_actual_calls', dict(failed))
    accounting = a.json(a.out/'accounting.json')
    for name, calculated in (('attempts', attempts), ('returns', returns), ('failed', failed)):
        actual = accounting[name]
        for kind in set(actual) | set(calculated):
            a.require(int(actual.get(kind, 0)) == calculated[kind], 'accounting_'+name,
                      {'kind': kind, 'recorded': actual.get(kind, 0), 'reconstructed': calculated[kind]})
    for kind in set(seconds) | set(accounting['wall_seconds_by_kind']):
        a.require(math.isclose(seconds[kind], accounting['wall_seconds_by_kind'].get(kind, 0.), rel_tol=1e-10, abs_tol=1e-7),
                  'accounting_call_seconds', kind)
    total_forward = sum(attempts[k] for k in FORWARDS)
    a.require(total_forward == accounting['total_forward_attempts'], 'accounting_forward_total')
    a.require(not accounting.get('pending'), 'accounting_no_pending')
    for kind, limit in protocol['budget'].items():
        a.require(attempts[kind] <= limit, 'hard_kind_budget', {'kind': kind, 'actual': attempts[kind], 'limit': limit})
    a.require(total_forward <= protocol['overall_forward_limit'], 'hard_total_forward_budget', total_forward)
    for kind, expected in EXPECTED_TRAINING.items():
        a.require(attempts[kind] == returns[kind] == expected, 'fixed_training_actual_calls',
                  {'kind': kind, 'attempts': attempts[kind], 'returns': returns[kind], 'expected': expected})
    generated = []
    for path in sorted((a.out/'evaluation').glob('*/generated_rows.json')):
        value = a.json(path)
        a.require(value['status'] == 'COMPLETE', 'generated_panel_complete', str(path))
        ids = [r['case_id'] for r in value['rows']]
        a.require(len(ids) == len(set(ids)), 'unique_generation_case_per_label', path.parent.name)
        a.require(len(ids) == 144, 'registered_generation_rows_per_label', {'label': path.parent.name, 'rows': len(ids)})
        generated.extend(value['rows'])
    summary = dict(optimizer_updates=returns['optimizer_update'],
        registered_unique_generation_requests=len(generated),
        applicable_unique_generation_requests=sum(r['status'] != 'NOT_APPLICABLE' for r in generated),
        actual_full_sampling_attempts=attempts['sampling'], actual_full_sampling_returns=returns['sampling'],
        historical_raw_cache_hits=sum(r['status'] == 'CACHED' for r in generated),
        generation_not_applicable=sum(r['status'] == 'NOT_APPLICABLE' for r in generated),
        extra_diagnostic_bf16_forwards=attempts['diagnostic_bf16'], extra_diagnostic_fp32_forwards=attempts['diagnostic_fp32'],
        precheck_forwards=attempts['precheck_forward'], precheck_zero_update_backwards=attempts['precheck_backward'],
        training_microbatch_forwards=attempts['training_forward'], training_backwards=attempts['training_backward'],
        sampling_forwards=attempts['sampling_forward'], total_model_forwards=total_forward,
        failed_actual_calls=sum(failed.values()), unresolved_calls=len(pending))
    for key, value in summary.items():
        a.require(accounting.get('summary', {}).get(key) == value, 'accounting_summary',
                  {'field': key, 'recorded': accounting.get('summary', {}).get(key), 'reconstructed': value})
    a.require(len(generated) <= protocol['unique_generation_request_limit'], 'unique_request_limit')
    count_rows = [dict(kind=k, actual_begin=attempts[k], actual_return=returns[k], actual_failed=failed[k],
                       accounting_attempts=accounting['attempts'].get(k, 0),
                       accounting_returns=accounting['returns'].get(k, 0), limit=protocol['budget'][k],
                       call_wall_seconds=seconds[k]) for k in protocol['budget']]
    dump_csv(a.dest/'call_count_audit.csv', count_rows)
    return dict(begins=begins, terminals=terminals, summary=summary, non_call_metadata=metadata_summary)


def audit_training(a, protocol, calls):
    plan = a.json(a.out/'train_plan.json')
    plan_records = plan['records']
    a.require(len(plan_records) == 8000, 'frozen_plan_8000')
    a.require(a.inputs[str((a.out/'train_plan.json').resolve())]['sha256'] == protocol['training_plan']['sha256'],
              'frozen_plan_file_hash')
    declared_parents = set(plan['object_counts'])
    a.require(len(declared_parents) == 32 and set(plan['object_counts'].values()) == {250}, 'frozen_parent_counts')
    exposures = {}; batch_rows = []; parent_rows = []
    events_by_arm = {}; logs_by_arm = {}
    for arm in ARMS:
        rows = a.csv(a.out/'training'/f'{arm}_sample_exposure.csv')
        logs = a.csv(a.out/'training'/f'{arm}_training_log.csv')
        events = a.jsonl(a.out/'training'/f'{arm}_events.jsonl')
        a.require(len(rows) == 8000, 'exposure_row_count', {'arm': arm, 'actual': len(rows)})
        a.require(len(logs) == 1000, 'training_log_row_count', {'arm': arm, 'actual': len(logs)})
        a.require([int(r['sample_index']) for r in rows] == list(range(8000)), 'ordered_unique_sample_indices', arm)
        a.require([int(r['pilot_step']) for r in logs] == list(range(1, 1001)), 'ordered_unique_training_steps', arm)
        exposures[arm] = rows
        logs_by_arm[arm] = {int(r['pilot_step']): r for r in logs}
        selected = {}
        for event in ('INPUT_BEGIN', 'INPUT_DONE', 'UPDATE_BEGIN', 'UPDATE_DONE'):
            values = [r for r in events if r.get('event') == event]
            a.require([r['pilot_step'] for r in values] == list(range(1, 1001)), 'event_exact_1000_ordered',
                      {'arm': arm, 'event': event, 'actual': len(values)})
            selected[event] = {r['pilot_step']: r for r in values}
        events_by_arm[arm] = selected
        a.require(not [r for r in events if r.get('event') == 'ARM_FAILED'], 'no_training_failure_event', arm)
        by_parent = defaultdict(list); by_batch = defaultdict(list)
        for row in rows:
            index = int(row['sample_index']); step = int(row['pilot_step'])
            a.require(row['arm'] == arm and step == index//8+1, 'exposure_arm_step_cursor', {'arm': arm, 'index': index})
            a.require(0 <= index < len(plan_records), 'exposure_registered_index', index)
            if not 0 <= index < len(plan_records):
                continue
            record = plan_records[index]
            equal = all(row[key] == str(record[key]) for key in ('uid', 'task_id', 'N', 'K'))
            equal &= all(float(row[key]) == float(record[key]) for key in ('ratio', 'alpha'))
            a.require(equal, 'exposure_matches_frozen_task_alpha_NK', {'arm': arm, 'index': index})
            n, k = int(row['N']), int(row['K'])
            a.require(int(row['free_coordinates']) == 9*(n-k) and 0 < k < n, 'sample_effective_free_coordinates',
                      {'arm': arm, 'index': index})
            a.require(math.isfinite(float(row['t'])) and 0 <= float(row['t']) <= 1, 'finite_training_time', index)
            a.require(.9-1e-7 <= float(row['alpha']) <= 1.1+1e-7, 'registered_alpha_range', index)
            for key in ('shared_input_sha256', 'coupled_input_sha256', 'ot_cache_key'):
                value = row[key]
                a.require(len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'complete_actual_tensor_hash',
                          {'arm': arm, 'index': index, 'key': key})
            by_parent[row['uid']].append(row); by_batch[step].append(row)
        a.require(set(by_parent) == declared_parents, 'exact_original_training_parents', arm)
        for uid, values in sorted(by_parent.items()):
            a.require(len(values) == 250, 'actual_parent_exposure_250', {'arm': arm, 'uid': uid, 'count': len(values)})
            parent_rows.append(dict(arm=arm,uid=uid,exposures=len(values),unique_C_tasks=len({r['task_id'] for r in values}),
                alpha1_count=sum(float(r['alpha']) == 1. for r in values),
                ratio20_count=sum(float(r['ratio']) == .2 for r in values),
                ratio40_count=sum(float(r['ratio']) == .4 for r in values),
                free_coordinates_total=sum(int(r['free_coordinates']) for r in values)))
        for step in range(1, 1001):
            values = by_batch[step]
            denominator = sum(int(r['free_coordinates']) for r in values)
            unit_count = sum(float(r['alpha']) == 1. for r in values)
            a.require(len(values) == 8 and unit_count == 4, 'batch_8_with_4_unit_alpha', {'arm': arm, 'step': step})
            shared_hash = hashlib.sha256(''.join(r['shared_input_sha256'] for r in values).encode()).hexdigest()
            coupled_hash = hashlib.sha256(''.join(r['coupled_input_sha256'] for r in values).encode()).hexdigest()
            log = logs_by_arm[arm][step]; event = selected['UPDATE_DONE'][step]; inp = selected['INPUT_DONE'][step]
            a.require(int(log['free_coordinate_denominator']) == event['free_coordinate_denominator'] ==
                      inp['effective_free_coordinates'] == denominator, 'batch_global_coordinate_denominator', {'arm': arm, 'step': step})
            a.require(log['shared_batch_sha256'] == event['shared_batch_sha256'] == inp['shared_batch_sha256'] == shared_hash,
                      'batch_hash_reconstructed_from_actual_sample_hashes', {'arm': arm, 'step': step})
            a.require(log['coupled_batch_sha256'] == event['coupled_batch_sha256'] == coupled_hash,
                      'coupled_batch_hash_reconstructed', {'arm': arm, 'step': step})
            a.require(float(log['FM']) == float(log['loss']) == float(event['FM']) == float(event['loss']),
                      'original_FM_only_logged_loss', {'arm': arm, 'step': step})
            diagnostics = event['sample_diagnostics']
            a.require(len(diagnostics) == 8 and [r['sample_index'] for r in diagnostics] == [int(r['sample_index']) for r in values],
                      'FM_diagnostic_sample_alignment', {'arm': arm, 'step': step})
            contribution = sum(float(r['FM_contribution']) for r in diagnostics)
            a.require(math.isclose(contribution, float(log['FM']), rel_tol=1e-12, abs_tol=1e-12),
                      'FM_contributions_sum_to_batch', {'arm': arm, 'step': step})
            for diagnostic, exposure in zip(diagnostics, values):
                a.require(int(diagnostic['free_coordinates']) == int(exposure['free_coordinates']), 'FM_effective_coordinates_match_exposure')
                expected = float(diagnostic['FM_contribution'])*denominator/int(exposure['free_coordinates'])
                a.require(math.isclose(expected,float(diagnostic['FM_per_sample_MSE']),rel_tol=1e-12,abs_tol=1e-12),
                          'per_sample_MSE_uses_true_coordinate_count')
            ot_attempts, ot_returns, ot_hits = (int(log[k]) for k in ('actual_OT_attempts', 'actual_OT_returns', 'OT_cache_hits'))
            a.require(ot_attempts == ot_returns and ot_returns+ot_hits == 8, 'complete_batch_OT_accounting', {'arm': arm, 'step': step})
            batch_rows.append(dict(arm=arm,pilot_step=step,samples=len(values),alpha1_count=unit_count,
                expected_free_coordinates=denominator,logged_free_coordinates=int(log['free_coordinate_denominator']),
                denominator_equal=denominator==int(log['free_coordinate_denominator']),
                shared_batch_sha256=shared_hash,FM=float(log['FM']),FM_contribution_sum=contribution,
                actual_OT_attempts=ot_attempts,actual_OT_returns=ot_returns,OT_cache_hits=ot_hits))
    paired=[]
    if all(len(exposures.get(arm, [])) == 8000 for arm in ARMS):
        keys=('pilot_step','sample_index','task_id','uid','N','K','ratio','alpha','t','free_coordinates','shared_input_sha256','ot_cache_key')
        for left,right in zip(exposures['OT_BASE'],exposures['OT_HYBRID']):
            same=all(left[k] == right[k] for k in keys)
            a.require(same,'paired_actual_input_identity',int(left['sample_index']))
            a.require(left['coupled_input_sha256'] != right['coupled_input_sha256'],'coupling_changes_actual_path_label',int(left['sample_index']))
            paired.append(dict(sample_index=int(left['sample_index']),pilot_step=int(left['pilot_step']),uid=left['uid'],
                task_id=left['task_id'],N=int(left['N']),K=int(left['K']),alpha=float(left['alpha']),t=float(left['t']),
                free_coordinates=int(left['free_coordinates']),paired_identity_equal=same,
                base_shared_sha256=left['shared_input_sha256'],hybrid_shared_sha256=right['shared_input_sha256']))
    shared_ledger=a.jsonl(a.out/'training/shared_batch_hashes.jsonl')
    a.require([r['pilot_step'] for r in shared_ledger] == list(range(1,1001)), 'registered_shared_batches_exact_1000')
    for row in shared_ledger:
        step=row['pilot_step']
        a.require(all(logs_by_arm[arm][step]['shared_batch_sha256']==row['shared_batch_sha256'] for arm in ARMS),
                  'paired_batch_registration_equal',step)
    if calls is not None:
        call_slots=Counter()
        for cid,begin in calls['begins'].items():
            kind=begin['kind']
            if kind not in EXPECTED_TRAINING:
                continue
            ctx=begin['context'];arm=ctx['arm'];step=int(ctx['pilot_step'])
            a.require(arm in ARMS and 1<=step<=1000,'training_call_registered_arm_step',cid)
            if arm not in exposures or not 1<=step<=1000:
                continue
            micro=int(ctx.get('microbatch',0))
            call_slots[(arm,step,kind,micro)]+=1
            a.require(begin['stage']=='train_'+arm,'training_call_stage_arm',cid)
            a.require(calls['terminals'].get(cid,{}).get('event')=='CALL_RETURN','training_call_completed',cid)
            if kind!='optimizer_update':
                index=8*(step-1)+micro
                row=exposures[arm][index]
                a.require(ctx['sample_index']==index and ctx['shared_input_sha256']==row['shared_input_sha256'],
                          'actual_forward_backward_matches_exposure',cid)
        for arm in ARMS:
            for step in range(1,1001):
                for kind in EXPECTED_TRAINING:
                    for micro in (range(8) if kind!='optimizer_update' else (0,)):
                        a.require(call_slots[(arm,step,kind,micro)]==1,'exact_one_actual_call_per_registered_slot',
                                  {'arm':arm,'step':step,'kind':kind,'micro':micro,'count':call_slots[(arm,step,kind,micro)]})
    dump_csv(a.dest/'paired_input_audit.csv',paired)
    dump_csv(a.dest/'parent_exposure_audit.csv',parent_rows)
    dump_csv(a.dest/'batch_normalization_audit.csv',batch_rows)
    return dict(exposure_rows={k:len(v) for k,v in exposures.items()},paired_rows=len(paired),parents_per_arm=32,
                updates_per_arm=1000,normalization='sum squared errors / (9 * total valid free faces across all 8 microbatches)')


def audit_endpoints(a,protocol):
    identities=[]
    for arm in ARMS:
        path=a.out/'checkpoints'/f'{arm}_pilot1000_total5000.pt'
        side=a.json(path.with_suffix('.json'));meta=side['metadata']
        actual=a.hash_file(path)
        a.require(actual==side['checkpoint_sha256'],'endpoint_actual_file_SHA256',arm)
        a.require(Path(side['path']).resolve()==path.resolve(),'endpoint_sidecar_path',arm)
        expected=dict(arm=arm,pilot_step=1000,conditional_total_step=5000,completed_updates=1000,
                      source_checkpoint_sha256=SOURCE_SHA,context_encoder='geo',loss_recipe='fm',
                      lambda_variance=0. if arm=='OT_BASE' else .25,full_recovery_state=True,
                      optimizer_reset=False,model_parameters='FP32',optimizer_state='FP32',protocol_sha256=canonical_hash(protocol))
        for key,value in expected.items():
            a.require(meta.get(key)==value,'endpoint_fixed_metadata',{'arm':arm,'field':key,'actual':meta.get(key),'expected':value})
        for key in ('model_state_sha256','optimizer_state_sha256','rng_sha256','stream_state_sha256','config_sha256'):
            a.require(isinstance(meta.get(key),str) and len(meta[key])==64,'endpoint_complete_state_identity',{'arm':arm,'field':key})
        pointer=a.json(a.out/'training'/f'{arm}_pointer.json')
        a.require(pointer['metadata']==meta and pointer['checkpoint_sha256']==actual and Path(pointer['path']).resolve()==path.resolve(),
                  'endpoint_recovery_pointer_matches_fixed',arm)
        events=a.jsonl(a.out/'training'/f'{arm}_events.jsonl')
        fixed=[r for r in events if r.get('event')=='CHECKPOINT_DONE' and r.get('pilot_step')==1000]
        a.require(len(fixed)==1 and fixed[0]['checkpoint']['checkpoint_sha256']==actual,'endpoint_durable_save_event',arm)
        identities.append(dict(arm=arm,path=str(path),bytes=path.stat().st_size,actual_sha256=actual,metadata=meta,
                               scope='Actual PT file SHA256 plus bound sidecar metadata; no tensor deserialization'))
    manifest=a.out/'checkpoint_manifest.json'
    if manifest.exists():
        records=a.json(manifest)['endpoints']
        for identity in identities:
            matching=[r for r in records if r['metadata']['arm']==identity['arm']]
            a.require(len(matching)==1 and matching[0]['checkpoint_sha256']==identity['actual_sha256'] and
                      matching[0]['metadata']==identity['metadata'],'final_manifest_endpoint_identity',identity['arm'])
    dump_json(a.dest/'endpoint_identity_audit.json',identities)
    return identities


def run(out,require_complete):
    a=Audit(out);started=time.perf_counter()
    ready=a.section('readiness',lambda:readiness(a,require_complete))
    if a.failures:
        # Do not read mutating ledgers or derive a final result during training.
        result=dict(status='BLOCKED',reason='Required stages are missing, active, blocked, or incomplete; no live training audit performed',
                    sections=a.sections,failed_checks=dict(a.failures),failure_examples=a.examples,
                    model_forwards=0,model_backwards=0,optimizer_updates=0)
        dump_json(a.dest/'audit_result.json',result)
        return result
    protocol=a.section('protocol',lambda:a.json(a.out/'protocol.json'))
    calls=training=endpoints=None
    if protocol is not None:
        calls=a.section('actual_call_ledger',lambda:audit_calls(a,protocol))
        training=a.section('paired_training_exposure',lambda:audit_training(a,protocol,calls))
        endpoints=a.section('fixed_endpoints',lambda:audit_endpoints(a,protocol))
    for path,identity in a.inputs.items():
        a.require(stat_identity(path)=={k:identity[k] for k in ('bytes','mtime_ns')},'input_stable_through_complete_audit',path)
    result=dict(status='BLOCKED' if a.failures else 'PASS',readiness=ready,sections=a.sections,
        check_counts=dict(a.checks),failed_checks=dict(a.failures),failure_examples=a.examples,
        reconstructed_accounting=calls['summary'] if calls else None,training=training,
        non_call_metadata=calls['non_call_metadata'] if calls else None,
        endpoint_files_checked=len(endpoints) if endpoints else 0,
        input_files_checked=len(a.inputs),wall_seconds=time.perf_counter()-started,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_forwards=0,model_backwards=0,optimizer_updates=0,OT_calls=0,
        limits='This is an independent artifact/ledger consistency audit, not a model-quality judgement or tensor re-execution. Endpoint tensors are not deserialized. No absent or failed rows are filled in.')
    dump_json(a.dest/'audited_input_sha256.json',a.inputs)
    dump_json(a.dest/'audit_result.json',result)
    summary=calls['summary'] if calls else {}
    lines=['# Independent ledger and exposure audit','',f"Status: **{result['status']}**",'',
           f"Actual optimizer returns: {summary.get('optimizer_updates','BLOCKED')}; training forwards: {summary.get('training_microbatch_forwards','BLOCKED')}; training backwards: {summary.get('training_backwards','BLOCKED')}.",
           f"Failed actual calls: {summary.get('failed_actual_calls','BLOCKED')}; unresolved calls: {summary.get('unresolved_calls','BLOCKED')}.",'',
           'Checks reconstruct CALL_BEGIN/RETURN/FAILED counts from durable ledgers, compare accounting.json, require 1000 UPDATE_DONE and 8000 exposure rows per arm, compare paired sample input hashes, enforce 32 parents × 250 exposures and four alpha=1 slots per batch, and compare the full batch free-coordinate denominator with logs and per-sample FM contributions.',
           '', 'Recognized non-CALL CPU scoring metadata are listed in non_call_metadata_rows.json and counted in non_call_metadata_summary.json. They do not count as model forwards; unknown or malformed CALL/event records remain audit errors. Endpoint identity checks use actual PT bytes and their metadata sidecars: pilot_step=1000, conditional_total_step=5000, original Geo4000 source, full AdamW/RNG/stream identity, and the frozen protocol. PT files are never deserialized.',
           '', 'Missing, failed, duplicated or inconsistent evidence is BLOCKED. This audit does not modify training, model weights, protocol, frozen code or data, and does not repair missing rows.',
           '', 'Artifacts: audit_result.json; call_count_audit.csv; paired_input_audit.csv; parent_exposure_audit.csv; batch_normalization_audit.csv; endpoint_identity_audit.json; audited_input_sha256.json.']
    if a.failures:
        lines+=['','Failed checks:','']+[f'- {name}: {count}' for name,count in sorted(a.failures.items())]
    (a.dest/'INDEPENDENT_LEDGER_EXPOSURE_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--require-run-complete',action='store_true',help='Also require the final RUN_COMPLETE.json marker (recommended for final delivery).')
    args=p.parse_args();result=run(args.out,args.require_run_complete)
    print(json.dumps({k:result.get(k) for k in ('status','reconstructed_accounting','failed_checks','wall_seconds')},sort_keys=True))
    raise SystemExit(0 if result['status']=='PASS' else 2)


if __name__=='__main__':main()
