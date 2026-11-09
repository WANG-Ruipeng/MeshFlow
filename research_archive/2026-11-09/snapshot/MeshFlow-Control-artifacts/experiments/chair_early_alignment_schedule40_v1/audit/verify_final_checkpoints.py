#!/usr/bin/env python3
"""CPU-only audit of immutable schedule40 endpoints; never reads latest.pt.

No model is constructed. No forward, backward, optimizer step or CUDA call is
performed. Missing completion artifacts remain PENDING; this tool never trains.
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import torch
from meshflow_control.training import schedule40 as s
from meshflow_control.data.io import atomic_json, digest, json_hash, read_json

torch.set_num_threads(2)
ENDPOINTS = {'COMMON': 5000, 'EARLY': 6000, 'FM_ONLY': 6000, 'LATE': 6000}
EXPECTED_HEAD_INITIAL = '3a546f6531ee4228ffb64309c2a45d135403789728d9a7d51b48d6b4ce682c6d'
EXPECTED_GENERATOR_INITIAL = '792ad8d6249ed4ccd1b5c5a81dfba0e527fc61911be96144729f8a2fc7f4811a'
INFERENCE_CONTRACT = dict(readout_mode='none', teacher=False, aligner=False,
    capture=False, forward='BF16', parameters='FP32', integration='FP32')


def check(items, name, ok=None, actual=None, expected=None, status=None, detail=None):
    row = dict(check=name, status=status or ('PASS' if ok else 'FAIL'))
    if actual is not None: row['actual'] = actual
    if expected is not None: row['expected'] = expected
    if detail is not None: row['detail'] = detail
    items.append(row)
    return row['status'] == 'PASS'


def aggregate(items):
    states = {r['status'] for r in items}
    return 'FAIL' if 'FAIL' in states else 'PENDING' if 'PENDING' in states else 'PASS'


def optimizer_audit(p, checks):
    """Inspect stored tensors directly, without constructing an optimizer."""
    segment, step = p['segment'], p['global_step']
    head = p['aligner_state']
    opt, names = p['optimizer'], p['optimizer_parameter_names']
    groups = opt['param_groups']
    expected_groups = 2 if segment in ('EARLY', 'LATE') else 1
    check(checks, 'optimizer_group_count', len(groups) == expected_groups,
        len(groups), expected_groups)
    lookup = {'generator.' + n: v for n, v in p['generator_state'].items()}
    if head is not None: lookup.update({'aligner.' + n: v for n, v in head.items()})
    expected_names = [['generator.' + n for n in p['generator_state']]]
    if head is not None: expected_names.append(['aligner.' + n for n in head])
    check(checks, 'optimizer_parameter_registration_order', names == expected_names,
        actual=[len(x) for x in names], expected=[len(x) for x in expected_names])
    owned = [i for group in groups for i in group['params']]
    check(checks, 'optimizer_unique_complete_state_ownership',
        len(owned) == len(set(owned)) == len(lookup) and set(opt['state']) == set(owned),
        actual=dict(owned=len(owned), unique=len(set(owned)), states=len(opt['state'])), expected=len(lookup))
    details = []
    for group_index, (group, group_names) in enumerate(zip(groups, names)):
        expected_lr = 1e-5 if group_index == 0 else 1e-4
        expected_step = step if group_index == 0 else s.head_steps(segment, step)
        settings = dict(lr=group['lr'], betas=list(group['betas']), weight_decay=group['weight_decay'],
                        eps=group['eps'], amsgrad=group['amsgrad'], maximize=group['maximize'])
        expected_settings = dict(lr=expected_lr, betas=[.9, .95], weight_decay=0., eps=1e-8, amsgrad=False, maximize=False)
        check(checks, 'optimizer_group_' + str(group_index) + '_settings', settings == expected_settings,
            settings, expected_settings)
        for index, name in zip(group['params'], group_names):
            v, st = lookup[name], opt['state'].get(index, {})
            actual_step = float(st['step']) if isinstance(st.get('step'), torch.Tensor) else st.get('step')
            valid = set(st) == {'step', 'exp_avg', 'exp_avg_sq'} and actual_step == expected_step
            moments = {}
            for k in ('exp_avg', 'exp_avg_sq'):
                x = st.get(k)
                facts = dict(tensor=isinstance(x, torch.Tensor))
                if isinstance(x, torch.Tensor):
                    facts.update(shape=list(x.shape), dtype=str(x.dtype), device=str(x.device), finite=bool(torch.isfinite(x).all()))
                    valid = valid and x.shape == v.shape and x.dtype == torch.float32 and x.device.type == 'cpu' and facts['finite']
                else: valid = False
                moments[k] = facts
            details.append(dict(name=name, optimizer_id=index, expected_step=expected_step,
                actual_step=actual_step, status='PASS' if valid else 'FAIL', moments=moments))
    check(checks, 'all_optimizer_parameter_steps_shapes_dtypes_finite',
        len(details) == len(lookup) and all(r['status'] == 'PASS' for r in details),
        actual=dict(inspected=len(details), failed=sum(r['status'] != 'PASS' for r in details)), expected=len(lookup))
    return details


def endpoint(run, segment, identities):
    step = ENDPOINTS[segment]
    path = run / 'checkpoints' / f'{segment}_step{step:04d}.pt'
    sidecar = path.with_suffix('.json')
    done_path = run / 'training' / segment / 'completed.json'
    result = dict(segment=segment, path=str(path), status='PENDING', checks=[])
    checks = result['checks']
    missing = [str(x) for x in (path, sidecar, done_path) if not x.is_file()]
    if missing:
        check(checks, 'immutable_endpoint_ready', status='PENDING', detail=dict(missing=missing)); return result
    done = read_json(done_path)
    if done.get('status') != 'COMPLETE':
        check(checks, 'immutable_endpoint_ready', status='PENDING', actual=done.get('status')); return result
    payload = None
    started = time.perf_counter()
    try:
        receipt = read_json(sidecar)
        file_hash = digest(path)
        check(checks, 'file_sha256_vs_sidecar_and_completion',
            file_hash == receipt.get('sha256') == done.get('checkpoint', {}).get('sha256'),
            file_hash, receipt.get('sha256'))
        check(checks, 'completion_cursor', done.get('step') == step and done.get('segment') == segment,
            dict(step=done.get('step'), segment=done.get('segment')), dict(step=step, segment=segment))
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        p = payload
        check(checks, 'schema', p.get('schema') == s.SCHEMA and p.get('experiment') == s.EXPERIMENT,
            dict(schema=p.get('schema'), experiment=p.get('experiment')))
        check(checks, 'completed_update_boundary', p.get('update_in_progress') is False and p.get('global_step') == step and p.get('segment') == segment,
            dict(step=p.get('global_step'), segment=p.get('segment'), update_in_progress=p.get('update_in_progress')))
        check(checks, 'training_protocol_and_model_configuration', p['protocol'] == s.PROTOCOL and p['protocol_sha256'] == json_hash(s.PROTOCOL) and p['model_config'] == s.MODEL_CONFIG,
            p['protocol_sha256'], json_hash(s.PROTOCOL))
        state = p['generator_state']
        counts = dict(tensors=len(state), parameters=sum(x.numel() for x in state.values()))
        check(checks, 'generator_capacity', counts == dict(tensors=194, parameters=130401155), counts)
        check(checks, 'generator_finite_CPU_FP32_no_extra_branches',
            all(x.device.type == 'cpu' and x.dtype == torch.float32 and bool(torch.isfinite(x).all()) for x in state.values())
            and not any(n.startswith(('readouts.', 'aligner.', 'teacher.')) for n in state))
        actual_generator = s.state_hash(state)
        check(checks, 'generator_state_sha256', actual_generator == p['generator_state_sha256'] == receipt.get('generator_state_sha256'), actual_generator, p['generator_state_sha256'])
        actual_optimizer = s.tree_hash(p['optimizer'])
        check(checks, 'optimizer_state_sha256', actual_optimizer == p['optimizer_state_sha256'] == receipt.get('optimizer_state_sha256'), actual_optimizer, p['optimizer_state_sha256'])
        actual_rng = s.tree_hash(p['rng'])
        check(checks, 'rng_state_sha256', actual_rng == p['rng_sha256'] == receipt.get('rng_sha256'), actual_rng, p['rng_sha256'])
        actual_stream = json_hash(p['stream_state'])
        check(checks, 'stream_state_sha256', actual_stream == p['stream_state_sha256'] == receipt.get('stream_state_sha256'), actual_stream, p['stream_state_sha256'])
        stream = p['stream_state']
        check(checks, 'stream_cursor', stream.get('batch_index') == step and stream.get('sample_index') == 8*step,
            dict(batch=stream.get('batch_index'), sample=stream.get('sample_index')), dict(batch=step, sample=8*step))
        check(checks, 'stream_input_binding', stream.get('plan_sha256') == p['identities']['plan_sha256'] and stream.get('data_manifest_sha256') == p['identities']['data_manifest_sha256'])
        for key, actual in identities.items():
            check(checks, 'actual_' + key, status='PENDING' if actual is None else None,
                ok=p['identities'].get(key) == actual, actual=p['identities'].get(key), expected=actual,
                detail='Original input file absent' if actual is None else None)
        head = p['aligner_state']; expected_head_steps = 1000 if segment in ('EARLY', 'LATE') else 0
        check(checks, 'head_completed_updates', p['head_completed_updates'] == expected_head_steps,
            p['head_completed_updates'], expected_head_steps)
        if expected_head_steps:
            actual_head = s.state_hash(head)
            check(checks, 'head_state_sha256', actual_head == p['aligner_state_sha256'] == receipt.get('aligner_state_sha256'), actual_head, p['aligner_state_sha256'])
            check(checks, 'head_finite_CPU_FP32_capacity', sum(x.numel() for x in head.values()) == s.HEAD_PARAMETERS
                and all(x.device.type == 'cpu' and x.dtype == torch.float32 and bool(torch.isfinite(x).all()) for x in head.values()))
            check(checks, 'head_initialization_identity', p['lineage'].get('head_initial_sha256') == EXPECTED_HEAD_INITIAL,
                p['lineage'].get('head_initial_sha256'), EXPECTED_HEAD_INITIAL)
        else:
            actual_head = None
            check(checks, 'FM_has_no_head', head is None and p['aligner_state_sha256'] is None)
        lineage = p['lineage']
        check(checks, 'official_initialization_identity',
            lineage.get('official_file_sha256') == s.OFFICIAL_FILE_SHA256 and lineage.get('official_backbone_sha256') == s.OFFICIAL_BACKBONE_SHA256
            and lineage.get('generator_initial_sha256') == EXPECTED_GENERATOR_INITIAL and lineage.get('role_nonzero') == 0 and lineage.get('geo_exit_nonzero') == 0)
        result['optimizer_parameters'] = optimizer_audit(p, checks)
        result['summary'] = dict(file_sha256=file_hash, generator_state_sha256=actual_generator,
            optimizer_state_sha256=actual_optimizer, rng_sha256=actual_rng, stream_state_sha256=actual_stream,
            aligner_state_sha256=actual_head, global_step=step, identities=p['identities'], lineage=lineage,
            protocol_sha256=p['protocol_sha256'], model_config=p['model_config'], execution_receipt=p['execution_receipt'])
    except BaseException as exc:
        check(checks, 'audit_exception', False, actual=repr(exc), detail=traceback.format_exc())
    finally:
        # Do not retain tensors in the result or retain a list of open endpoints.
        payload = None
        if 'p' in locals(): del p
        if 'state' in locals(): del state
        if 'head' in locals(): del head
        gc.collect()
    result.update(status=aggregate(checks), seconds=time.perf_counter()-started)
    return result


def inference_export(run, segment, trained):
    path = run / 'inference' / f'{segment}_generator_step6000.pt'
    result = dict(segment=segment, path=str(path), status='PENDING', checks=[])
    checks = result['checks']; payload = None
    if not path.is_file() or 'summary' not in trained:
        check(checks, 'export_ready', status='PENDING', detail='Missing export or complete training endpoint audit'); return result
    started = time.perf_counter()
    try:
        done = read_json(run/'training'/segment/'completed.json')
        receipt = done.get('inference_export', {})
        actual_file = digest(path)
        check(checks, 'export_file_sha256', actual_file == receipt.get('sha256'), actual_file, receipt.get('sha256'))
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        p = payload; prior = trained['summary']
        expected_keys = {'schema','experiment','segment','global_step','model_config','generator_state','generator_state_sha256',
            'protocol_sha256','identities','lineage','training_checkpoint_sha256','inference_contract'}
        check(checks, 'exact_pure_generator_schema', set(p) == expected_keys and p.get('schema') == s.INFERENCE_SCHEMA
            and p.get('experiment') == s.EXPERIMENT, actual=sorted(p), expected=sorted(expected_keys))
        check(checks, 'export_arm_and_step', p['segment'] == segment and p['global_step'] == 6000)
        check(checks, 'export_training_checkpoint_binding', p['training_checkpoint_sha256'] == prior['file_sha256'],
            p['training_checkpoint_sha256'], prior['file_sha256'])
        check(checks, 'export_protocol_and_identity', p['protocol_sha256'] == prior['protocol_sha256']
            and p['model_config'] == prior['model_config'] and p['identities'] == prior['identities'] and p['lineage'] == prior['lineage'])
        check(checks, 'inference_contract', p['inference_contract'] == INFERENCE_CONTRACT, p['inference_contract'], INFERENCE_CONTRACT)
        state = p['generator_state']; actual_state = s.state_hash(state)
        check(checks, 'export_generator_exact_tensor_identity', actual_state == p['generator_state_sha256'] == prior['generator_state_sha256'],
            actual_state, prior['generator_state_sha256'])
        check(checks, 'export_capacity_finite_CPU_FP32', len(state) == 194 and sum(x.numel() for x in state.values()) == 130401155
            and all(x.dtype == torch.float32 and x.device.type == 'cpu' and bool(torch.isfinite(x).all()) for x in state.values())
            and not any(n.startswith(('aligner.', 'teacher.', 'readouts.')) for n in state))
        result['summary'] = dict(file_sha256=actual_file, generator_state_sha256=actual_state, training_checkpoint_sha256=p['training_checkpoint_sha256'])
    except BaseException as exc:
        check(checks, 'audit_exception', False, actual=repr(exc), detail=traceback.format_exc())
    finally:
        payload = None
        if 'p' in locals(): del p
        if 'state' in locals(): del state
        gc.collect()
    result.update(status=aggregate(checks), seconds=time.perf_counter()-started)
    return result


def event_rows(run, segment):
    for path in sorted((run/'training'/segment/'attempts').glob('*/events.jsonl')):
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip(): yield path, json.loads(line)


def cross_checks(run, trained):
    checks = []
    common = trained['COMMON'].get('summary')
    for segment in ('FM_ONLY', 'LATE'):
        branch = trained[segment].get('summary')
        if common is None or branch is None:
            check(checks, segment + '_common_fork', status='PENDING', detail='Required immutable endpoint absent'); continue
        lineage = branch['lineage']
        expected = dict(common_checkpoint_sha256=common['file_sha256'],
            common_generator_sha256=common['generator_state_sha256'],
            common_optimizer_sha256=common['optimizer_state_sha256'],
            common_rng_sha256=common['rng_sha256'], fork_step=5000,
            fork_destination=segment, generator_optimizer_reset=False)
        for key, value in expected.items():
            check(checks, segment + '_fork_' + key, lineage.get(key) == value, lineage.get(key), value)
        check(checks, segment + '_fork_input_identities', branch['identities'] == common['identities'])
        forks = [(str(path), row) for path, row in event_rows(run, segment) if row.get('event') == 'FORK']
        check(checks, segment + '_actual_fork_event', bool(forks) and all(row.get('step') == 5000
            and row.get('source_sha256') == common['file_sha256'] and Path(row.get('source', '')).name == 'COMMON_step5000.pt'
            for _, row in forks), actual=[dict(path=path, step=row.get('step'), source_sha256=row.get('source_sha256')) for path,row in forks])
    early = trained['EARLY'].get('summary'); late = trained['LATE'].get('summary')
    if early is None or late is None:
        check(checks, 'EARLY_LATE_same_initial_head', status='PENDING')
    else:
        check(checks, 'EARLY_LATE_same_initial_head', early['lineage'].get('head_initial_sha256') == late['lineage'].get('head_initial_sha256') == EXPECTED_HEAD_INITIAL,
            actual=dict(EARLY=early['lineage'].get('head_initial_sha256'), LATE=late['lineage'].get('head_initial_sha256')), expected=EXPECTED_HEAD_INITIAL)
    if early is None:
        check(checks, 'EARLY_head_freeze_after_1000', status='PENDING')
    else:
        records = [(str(path), row) for path,row in event_rows(run,'EARLY')
            if row.get('event') == 'CHECKPOINT_DONE' and int(row.get('step', -1)) >= 1000]
        boundary = [row['checkpoint'].get('aligner_state_sha256') for _,row in records if row['step'] == 1000]
        final_head = early['aligner_state_sha256']
        check(checks, 'EARLY_head_step1000_event_identity', bool(boundary) and all(x == final_head for x in boundary),
            actual=boundary, expected=final_head,
            detail='Historical step1000 event hash retained; old rolling checkpoint file is intentionally not read.')
        check(checks, 'EARLY_all_post1000_checkpoint_head_hashes_frozen', bool(records)
            and all(row['checkpoint'].get('aligner_state_sha256') == final_head for _,row in records),
            actual=dict(records=len(records), distinct_hashes=sorted({row['checkpoint'].get('aligner_state_sha256','MISSING') for _,row in records})), expected=final_head)
        seen, violations, row_count = set(), [], 0
        for path in sorted((run/'training/EARLY/attempts').glob('*/training_log.csv')):
            with path.open(newline='', encoding='utf-8') as handle:
                for row in csv.DictReader(handle):
                    step = int(row['step'])
                    if step <= 1000: continue
                    seen.add(step); row_count += 1
                    ok = row['alignment_active'].lower() == 'false' and int(row['aligned_samples']) == 0 \
                        and int(row['head_adamw_step']) == 1000 and float(row['raw_align']) == 0. \
                        and float(row['weighted_align']) == 0. and float(row['aligner_gradient_norm']) == 0.
                    if not ok: violations.append(dict(path=str(path), step=step, row=row))
        check(checks, 'EARLY_post1000_logs_head_inactive_counter1000', seen == set(range(1001,6001)) and not violations,
            actual=dict(unique_steps=len(seen), rows_including_replay=row_count, violations=violations), expected=dict(unique_steps=5000, violations=[]))
        check(checks, 'EARLY_step1000_to6000_head_moment_bitwise_comparison', status='NA',
            detail='No separate step1000 head-moment tensor archive/hash was retained. Final moment step=1000 is inspected per parameter; parameter hash freeze and all inactive training logs are verified. No historical moment bitwise-equivalence claim.')
    return checks


def audit(run, output):
    started = time.perf_counter()
    result = dict(schema='schedule40_final_checkpoint_cpu_audit_v1', status='PENDING',
        experiment=s.EXPERIMENT, run=str(run), started_unix=time.time(), script_sha256=digest(__file__),
        execution=dict(device='cpu', cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            cuda_initialized_before=torch.cuda.is_initialized(), torch_threads=torch.get_num_threads(),
            model_constructions=0, forwards=0, backwards=0, optimizer_steps=0,
            immutable_endpoints_only=True, latest_checkpoint_reads=0, missing_policy='PENDING; never rerun training'),
        checks=[], endpoints={}, exports={})
    input_paths = dict(data_manifest_sha256=run.parent.parent/'data/train_manifest.json',
        plan_sha256=run/'stream/train_plan.json', paired_inputs_sha256=run/'stream/paired_inputs.json',
        teacher_registration_sha256=run/'teacher/teacher_manifest.json')
    identities = {key: digest(path) if path.is_file() else None for key,path in input_paths.items()}
    result['actual_input_identities'] = {key: dict(path=str(input_paths[key]), sha256=value) for key,value in identities.items()}
    source_root = Path(s.__file__).resolve().parents[3]
    frozen_path = run/'audit/source_hashes_before.json'
    if frozen_path.is_file():
        frozen = read_json(frozen_path)
        required = ('src/meshflow_control/training/schedule40.py', 'src/meshflow_control/checkpoints.py',
                    'src/meshflow_control/training/context_checkpoint.py', 'src/meshflow_control/data/io.py')
        for name in required:
            path = source_root/name
            actual = digest(path) if path.is_file() else None
            check(result['checks'], 'frozen_helper_' + name, actual is not None and actual == frozen.get(name), actual, frozen.get(name))
    else: check(result['checks'], 'frozen_helper_sources', status='PENDING', detail='Missing source_hashes_before.json')
    for segment in ENDPOINTS:
        result['endpoints'][segment] = endpoint(run, segment, identities)
        atomic_json(output, result)
        print(segment, result['endpoints'][segment]['status'], flush=True)
    result['cross_checks'] = cross_checks(run, result['endpoints'])
    for segment in ('EARLY','FM_ONLY','LATE'):
        result['exports'][segment] = inference_export(run, segment, result['endpoints'][segment])
        atomic_json(output, result)
        print(segment + '_EXPORT', result['exports'][segment]['status'], flush=True)
    check(result['checks'], 'no_CUDA_initialization', not torch.cuda.is_initialized(), torch.cuda.is_initialized(), False)
    result['execution']['cuda_initialized_after'] = torch.cuda.is_initialized()
    all_checks = result['checks'] + result['cross_checks']
    for item in list(result['endpoints'].values()) + list(result['exports'].values()): all_checks += item['checks']
    result.update(status=aggregate(all_checks), seconds=time.perf_counter()-started,
        check_counts={status:sum(x['status']==status for x in all_checks) for status in ('PASS','FAIL','PENDING','NA')})
    atomic_json(output, result)
    return result


def self_test():
    """Only tiny stored-tensor dictionaries; no model or optimizer execution."""
    generator = {'weight':torch.zeros(2)}; head = {'q':torch.ones(1)}
    states = {0:dict(step=torch.tensor(6000.), exp_avg=torch.zeros(2), exp_avg_sq=torch.ones(2)),
              1:dict(step=torch.tensor(1000.), exp_avg=torch.zeros(1), exp_avg_sq=torch.ones(1))}
    def group(index, lr):
        return dict(params=[index],lr=lr,betas=(.9,.95),weight_decay=0.,eps=1e-8,amsgrad=False,maximize=False)
    p = dict(segment='EARLY',global_step=6000,aligner_state=head,generator_state=generator,
        optimizer_parameter_names=[['generator.weight'],['aligner.q']],
        optimizer=dict(state=states,param_groups=[group(0,1e-5),group(1,1e-4)]))
    checks=[]; details=optimizer_audit(p,checks)
    assert aggregate(checks)=='PASS' and len(details)==2
    p['optimizer']['state'][1]['step']=torch.tensor(6000.)
    bad=[]; optimizer_audit(p,bad); assert aggregate(bad)=='FAIL'
    assert aggregate([dict(status='PASS'),dict(status='PENDING')])=='PENDING'
    assert aggregate([dict(status='PASS'),dict(status='NA')])=='PASS'
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='PASS',kind='CPU_LIGHTWEIGHT_LOGIC_ONLY',real_weight_loads=0,
        model_constructions=0,forwards=0,backwards=0,optimizer_steps=0,cuda_initialized=False)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run=args.run.resolve()
        if run.name!='chair_early_alignment_schedule40_v1': raise ValueError('Wrong experiment root')
        output=(args.output or run/'audit/final_checkpoint_verification.json').resolve()
        if run not in output.parents: raise ValueError('Audit output must remain inside this experiment')
        report=audit(run,output)
        print(json.dumps(dict(status=report['status'],output=str(output),seconds=report['seconds'],check_counts=report['check_counts'])))
        sys.exit(1 if report['status']=='FAIL' else 0)
