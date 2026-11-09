"""Reuse the frozen K12 teacher bank for the four-model gated factorial study.

Loading and M00 reuse perform no OT, model call, autograd, or artifact write.
Only evaluate_state performs a single frozen forward; the root owns its ledger.
Teacher losses are the original ungated diagnostics for every model and time.
"""
from __future__ import annotations

from native_t1.runtime import configure_stable_runtime
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from native_t1.artifacts import array_hash, file_sha256
from native_t1.data import native_collate, model_inputs
from native_t1.model import native_execution
from .c_relation_teacher import load_teacher_bank
from .c_relation_math import analyze_teacher, _record as array_record
from .c_relation_run import actual_arrays, input_digest, loss_terms, state_info
from .t1_interface_relations import prepare_relations


BANK_SHA256 = 'e9dc4574b7610aa46988b926262c93cd97f622a39ad67e821e6f2112b6d80572'
A_FILE_SHA256 = '43e56e107b0c6d28766aebd8cf8a0a764d8d52ef62ba2ca6e756d881634f876b'
A_STATE_SHA256 = '6a2f003d5997b1bbe996e872041ef06e0443ab4e3af4cc13bd42bd2d157a6520'
MODEL_ORDER = ('M00', 'M10', 'M01', 'M11')
TIMES = (.1, .3, .5, .7, .9)
SUBSETS = ('interface', 'free_only_shared')
PROTOCOL = dict(
    K=12, parents=[0, 1], noise_seeds=[19601, 19602], times=list(TIMES),
    states_per_model=20, reused_M00_states=20, new_models=['M10', 'M01', 'M11'],
    new_model_forwards=60, new_OT_calls=0, autograd_calls=0,
    teacher_gate=False, normalization='Original per-state group-equal MSE_xyz; no bbox normalization',
    FF_partition='Interface and free-only contributions each retain the full FF group denominator',
    input_scope='Only xt,t,y,valid_mask,known_mask reach the model; GT/source IDs remain supervision',
    purpose='Fixed teacher endpoint error decomposition; not a free-sampling geometry score')


def _record(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha256(path))


def _seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _validate_analysis(analysis):
    if (analysis.get('schema') != 'c_relation_teacher_decomposition_v1'
            or analysis.get('source_mapping_status') != 'PASS_EXACT_GROUP_TARGET_AND_ANCHOR_CHECK'
            or analysis.get('legacy_loss_reconciliation', {}).get('status') != 'PASS'):
        raise RuntimeError('Original teacher definition/source-label reconciliation did not pass.')
    tolerance = 1e-12 + 1e-10 * max(analysis['losses'][k] for k in ('FM', 'FC', 'FF'))
    for name in ('max_abs_endpoint_minus_Position_minus_Spread',
                 'max_abs_epsilon_endpoint_minus_Position_minus_Spread',
                 'FC_minus_Position_plus_Spread', 'FF_minus_global_partition_contributions'):
        value = analysis['identities'][name]
        if not np.isfinite(value) or abs(value) > tolerance:
            raise RuntimeError('Original teacher decomposition identity failed: ' + name)


def _compact_row(model_name, old_row, *, reused):
    analysis = old_row['analysis']
    _validate_analysis(analysis)
    losses = analysis['losses']
    metrics = {key:float(losses[key]) for key in (
        'FM', 'FC', 'FF', 'FC_position_component', 'FC_spread_component',
        'FF_interface_contribution', 'FF_free_only_contribution')}
    metrics.update(FC_interface_contribution=metrics['FC'], FC_free_only_contribution=0.)
    group_counts, corner_counts = {}, {}
    for name in SUBSETS:
        subset = analysis['subsets'][name]
        group_counts[name], corner_counts[name] = subset['group_count'], subset['corner_count']
        for key in ('Position', 'Spread', 'EndpointError'):
            metrics[name + '_' + key] = subset['endpoint']['group_equal'][key]
    group_counts['FF_shared'] = losses['denominators']['FF_shared_groups']
    compact = {key:copy.deepcopy(analysis[key]) for key in (
        'schema', 'losses', 'identities', 'legacy_loss_reconciliation',
        'coordinate_units', 'decomposition_reduction_dtype', 'source_mapping_status')}
    compact['subsets'] = {key:copy.deepcopy(analysis['subsets'][key]) for key in SUBSETS}
    row = dict(model=model_name, state=copy.deepcopy(old_row['state']),
        input_sha256=old_row['input_sha256'], velocity_sha256=old_row['velocity_sha256'],
        losses=copy.deepcopy(old_row['losses']), metrics=metrics,
        group_counts=group_counts, corner_counts=corner_counts, analysis=compact,
        reused_existing_teacher=reused, new_forward_calls=0 if reused else 1,
        teacher_gate_applied=False, autograd_calls=0, OT_calls=0)
    if reused:
        row['reuse_source'] = dict(model='A', additional_step=500, phase='endpoint',
                                  full_original_row_sha256=_seal(old_row))
    json.dumps(row, allow_nan=False)
    return row


def _check_array_records(sample, analysis):
    records = analysis['array_records']
    for key in ('xt', 'x0', 'x1', 'u', 'context'):
        if array_record(np.asarray(sample[key]).reshape(112, 3, 3)) != records[key]:
            raise RuntimeError('Stored teacher coordinate identity differs: ' + key)
    for key in ('known_mask', 'valid_mask'):
        if array_record(np.asarray(sample[key]).reshape(112)) != records[key]:
            raise RuntimeError('Stored teacher mask identity differs: ' + key)
    if array_record(sample['labels']['corner_source_vertex_ids']) != records['corner_source_vertex_ids']:
        raise RuntimeError('Stored trusted source-corner labels differ.')


def load_fixed_teacher(diagnosis_run, relations):
    """Return (20 exact samples, 20 reused M00 rows, audit); never reconstruct OT.

    The unchanged bank loader checks every saved array/state hash and the real
    source mapper. A reuse is permitted only when checkpoint identity, original
    helper files, actual collated inputs, labels and numerical definition match.
    Any mismatch stops rather than silently spending additional A forwards.
    Large checkpoint byte hashes remain the root asset gate's responsibility.
    """
    run = Path(diagnosis_run).resolve()
    report, result_path = run/'C_RELATION_DIAGNOSIS_REPORT.md', run/'results.json'
    if not report.is_file():
        raise ValueError('Require the completed diagnosis report identity.')
    old = json.loads(result_path.read_text(encoding='utf-8-sig'))
    if old.get('status') != 'PASS':
        raise ValueError('Only the completed zero-update diagnosis can supply teacher values.')
    saved_report = old['delivery']['report']
    if Path(saved_report['path']).resolve() != report or file_sha256(report) != saved_report['sha256']:
        raise RuntimeError('Completed diagnosis report identity changed.')
    checkpoint, model = old['assets']['checkpoints']['A500'], old['models']['A500']
    if (checkpoint['sha256'] != A_FILE_SHA256 or checkpoint['state_sha256'] != A_STATE_SHA256
            or model.get('status') != 'PASS' or model.get('file_sha256') != A_FILE_SHA256
            or model.get('state_before') != A_STATE_SHA256 or model.get('state_after') != A_STATE_SHA256
            or not model.get('parameters_unchanged') or not model.get('strict_load')
            or model.get('missing_keys') or model.get('unexpected_keys')):
        raise RuntimeError('M00 is not the previously measured frozen A500 endpoint.')
    expected_runtime = dict(cublas_workspace_config=':4096:8', deterministic_algorithms=True,
        deterministic_warn_only=False, tf32_matmul=False, tf32_cudnn=False,
        sdpa_math_enabled=True, sdpa_flash_enabled=False, sdpa_mem_efficient_enabled=False,
        cuda_autocast_enabled=True, cuda_autocast_dtype='torch.bfloat16',
        gradient_enabled=False, parameter_dtype='torch.float32', all_modules_eval=True,
        trainable_parameters=0)
    if any(model['runtime_teacher'].get(key) != value for key, value in expected_runtime.items()):
        raise RuntimeError('Saved A teacher runtime differs from the fixed stable BF16 path.')
    bank_path = run/'teacher_bank.npz'
    if (old['bank']['bank_file']['sha256'] != BANK_SHA256
            or Path(old['bank']['bank_file']['path']).resolve() != bank_path):
        raise RuntimeError('Teacher-bank identity differs.')
    saved_sources = {**old['assets']['old_execution_sources'], **old['execution_sources']}
    required_paths = [Path(__file__).with_name(name) for name in (
        'c_relation_teacher.py', 'c_relation_math.py', 'c_relation_run.py', 't1_interface_relations.py')]
    repo = Path(__file__).resolve().parents[2]
    required_paths += [repo/'native_t1'/name for name in ('data.py', 'model.py', 'runtime.py', 'training.py')]
    verified_sources = {}
    for path in required_paths:
        rec = saved_sources.get(str(path.resolve()))
        if rec is None or file_sha256(path) != rec['sha256']:
            raise RuntimeError('Original teacher/data/model helper changed: ' + str(path))
        verified_sources[str(path.resolve())] = copy.deepcopy(rec)
    states, bank_audit = load_teacher_bank(bank_path, relations=relations, expected_sha256=BANK_SHA256)
    selected = [(i, sample) for i, sample in enumerate(states) if sample['K'] == 12]
    old_rows = [row for row in old['teacher_rows'] if row.get('model') == 'A'
                and row.get('additional_step') == 500 and row.get('phase') == 'endpoint'
                and row['state']['K'] == 12]
    row_map = {row['state']['state_id']:row for row in old_rows}
    expected = {(parent, seed, index) for parent in (0, 1) for seed in (19601, 19602) for index in range(5)}
    if (len(selected) != 20 or len(old_rows) != 20 or len(row_map) != 20
            or {(s['parent_index'], s['noise_seed'], s['time_index']) for _, s in selected} != expected):
        raise RuntimeError('Require exactly the fixed two-parent/two-noise/five-time K12 matrix.')
    samples, reused, state_records = [], [], []
    for index, sample in selected:
        prior = row_map[sample['state_id']]
        if state_info(sample, index) != prior['state']:
            raise RuntimeError('Old teacher state metadata differs.')
        _check_array_records(sample, prior['analysis'])
        batch = native_collate([sample], 'cpu')
        digest = input_digest(batch, sample['labels'])
        if digest != prior['input_sha256']:
            raise RuntimeError('Actual collated input/label hash differs from saved A teacher input.')
        sample['teacher_bank_index'] = index
        sample['teacher_reference_input_sha256'] = digest
        samples.append(sample)
        reused.append(_compact_row('M00', prior, reused=True))
        state_records.append(dict(**prior['state'], state_sha256=sample['state_sha256'], input_sha256=digest))
    audit = dict(status='PASS', protocol=copy.deepcopy(PROTOCOL),
        source_report=_record(report), source_results=_record(result_path),
        bank_file=bank_audit['bank_file'], source_inputs=copy.deepcopy(old['assets']['inputs']),
        M00_checkpoint_metadata=copy.deepcopy(checkpoint), M00_runtime=copy.deepcopy(model['runtime_teacher']),
        source_files=verified_sources, states=state_records, loaded_states_before_K12_selection=60,
        selected_states=20, reused_M00_rows=20, new_model_forwards=0, OT_calls=0, autograd_calls=0,
        source_labels_revalidated=True, original_A_analysis_recomputed=False,
        checkpoint_hash_scope='Root validates actual checkpoint bytes; this loader verifies recorded identity')
    return samples, reused, audit


@torch.no_grad()
def evaluate_state(model, sample, model_name, *, device='cuda'):
    """One new frozen teacher forward; return (compact row, runtime facts).

    The root must use the fixed 20 returned states once each for M10/M01/M11,
    validate checkpoint identities, and account attempts/returns with its hook.
    No M00 fallback, parameter-gradient calculation, or file write is present.
    """
    if model_name not in ('M10', 'M01', 'M11') or sample.get('K') != 12:
        raise ValueError('Only the three registered new K12 teacher evaluations are allowed.')
    if (getattr(model, 'experiment_trainable', True)
            or any(p.requires_grad or p.grad is not None for p in model.parameters())
            or any(module.training for module in model.modules())):
        raise RuntimeError('Teacher evaluation requires a frozen eval model without stored gradients.')
    configure_stable_runtime()
    label = sample['labels']
    batch = native_collate([sample], device)
    digest = input_digest(batch, label)
    if digest != sample['teacher_reference_input_sha256']:
        raise RuntimeError('Actual teacher inputs differ before the new forward.')
    pack = prepare_relations(label, device)
    with native_execution(model, coordinates=batch['xt']) as runtime:
        velocity = model(*model_inputs(batch)).float()
        terms, endpoint = loss_terms(batch, velocity, pack)
    if not bool(torch.isfinite(velocity).all()):
        raise RuntimeError('Nonfinite teacher velocity; no retry or fallback.')
    values = {name:float(value) for name, value in zip(('FM', 'FC', 'FF'), terms)}
    decoded = velocity.detach().cpu().numpy()
    analysis = analyze_teacher(actual_arrays(batch), decoded, label, legacy_losses=values)
    original = dict(state=state_info(sample, sample['teacher_bank_index']), losses=values,
                    input_sha256=digest, velocity_sha256=array_hash(decoded), analysis=analysis)
    row = _compact_row(model_name, original, reused=False)
    return row, runtime
