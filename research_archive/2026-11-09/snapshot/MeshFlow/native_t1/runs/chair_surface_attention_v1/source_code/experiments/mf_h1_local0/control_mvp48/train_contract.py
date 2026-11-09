"""Immutable update budget and fail-closed frozen-data binding contract."""
from pathlib import Path
import json
from native_t1.artifacts import file_sha256
from native_t1.objectives import objective_spec

ROOT = Path(__file__).resolve().parents[3] / 'native_t1/runs/control_mvp48_v1'
PROTOCOL = dict(schema='control_mvp48_training_protocol_v1', training_seed=10,
    model_initialization_seed=1010, total_actual_optimizer_updates=2500,
    phases=[['T1', 0, 1000], ['Geo_shared', 0, 500], ['Geo', 500, 1000], ['Full', 500, 1000]],
    effective_batch=8, microbatch=1, N_range=[128, 256],
    optimizer=dict(name='AdamW', lr=1e-5, betas=[.9, .95], weight_decay=0., clip=1.),
    scheduler=None, ema=None, precision='BF16_forward_FP32_parameters_coordinates_geometry',
    determinism='strict_TF32_off_SDPA_math', loss=objective_spec('edge5'),
    full_ramp_index='logical_step_minus_500', shared_optimizer_copied=True,
    FM_denominator='all_valid_free_scalar_coordinates_in_effective_batch',
    feature_statistics_fitted=False, gpu_workers=1,
    gpu_lock='/tmp/wrp_meshflow_jedge5_diag_gpu0.lock')


def validate_protocol(protocol):
    if {k: protocol.get(k) for k in PROTOCOL} != PROTOCOL:
        raise ValueError('Fixed Control MVP48 protocol changed')
    if not all(protocol.get(k) for k in ('data_ready_sha256', 'binding_sha256', 'plan_sha256')):
        raise ValueError('Protocol lacks frozen data identities')


class WaitingDependency(RuntimeError):
    pass


def checked_file(root, record):
    path = (root / record['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Frozen data path escapes run root')
    if not path.is_file():
        raise WaitingDependency('Missing frozen dependency: ' + str(path))
    if file_sha256(path) != record['sha256']:
        raise ValueError('Frozen dependency hash mismatch: ' + str(path))
    return path


def read_binding(root=ROOT):
    from .train_binding import bind_data
    return bind_data(root)
