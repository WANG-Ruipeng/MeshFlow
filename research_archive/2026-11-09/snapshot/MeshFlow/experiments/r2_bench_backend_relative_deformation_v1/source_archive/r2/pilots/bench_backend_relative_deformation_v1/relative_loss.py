"""Target-only local normalization and differentiable relative edge supervision.

No geometry scorer, model, CUDA initialization or source-topology operation is
imported here. Corner order is the immutable input/target correspondence. The
term penalizes target-relative vector error, including shear and rotation; it
does not certify topology or enforce a hard stretch bound at inference.
"""
from __future__ import annotations

import math
import numpy as np

SCHEMA = 'target_relative_edges_qr_svd_radial_huber_v1'
REDUCTION = 'per_face_scalar_huber_frobenius_rms_then_instance_mean'


def validate_config(config):
    required = {'schema', 'lambda_relative', 'huber_delta',
                'singular_floor_denoiser', 'rank_epsilon_denoiser', 'reduction'}
    if set(config) != required or config['schema'] != SCHEMA or config['reduction'] != REDUCTION:
        raise ValueError('Relative-loss contract must explicitly fix all six fields')
    for name in ('lambda_relative', 'huber_delta', 'singular_floor_denoiser', 'rank_epsilon_denoiser'):
        value = config[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError('Non-finite or missing loss hyperparameter: ' + name)
    if config['lambda_relative'] < 0 or min(config['huber_delta'], config['singular_floor_denoiser'], config['rank_epsilon_denoiser']) <= 0:
        raise ValueError('Lambda must be nonnegative; delta/floor/rank epsilon positive')
    if config['rank_epsilon_denoiser'] >= config['singular_floor_denoiser']:
        raise ValueError('Numerical rank exclusion must be below the inverse clipping floor')
    return dict(config)


def prepare_target(target_model, loss_mask, scale, config):
    """CPU float64 QR/SVD; frozen FP32 matrices are used by the training term."""
    config = validate_config(config)
    target = np.asarray(target_model, dtype=np.float32) * np.float32(scale)
    mask = np.asarray(loss_mask)
    if target.ndim != 3 or target.shape[1:] != (3, 3) or mask.shape != target.shape[:2] or mask.dtype != np.bool_:
        raise ValueError('Expected ordered Nx3x3 target and Nx3 bool coordinate mask')
    if not np.isfinite(target).all():
        raise ValueError('Nonfinite target geometry')
    edges = np.stack((target[:, 1] - target[:, 0], target[:, 2] - target[:, 0]), axis=-1)
    singular = np.linalg.svd(edges.astype(np.float64), compute_uv=False)
    trusted = mask.all(axis=-1)
    eligible = trusted & (singular[:, 1] > config['rank_epsilon_denoiser'])
    face_ids = np.flatnonzero(eligible).astype(np.int64)
    inverse = np.empty((len(face_ids), 2, 2), dtype=np.float32)
    clipped = 0
    for output_index, face_index in enumerate(face_ids):
        _, r = np.linalg.qr(edges[face_index].astype(np.float64), mode='reduced')
        # Positive QR diagonal fixes otherwise arbitrary target basis signs.
        signs = np.where(np.diag(r) < 0, -1.0, 1.0)
        r = signs[:, None] * r
        u, s, vh = np.linalg.svd(r, full_matrices=False)
        clipped += int(np.any(s < config['singular_floor_denoiser']))
        inverse[output_index] = (vh.T * (1.0 / np.maximum(s, config['singular_floor_denoiser']))) @ u.T
    if not np.isfinite(inverse).all():
        raise ValueError('Nonfinite capped inverse')
    return dict(face_ids=face_ids, target_edges=edges[face_ids].copy(), inverse=inverse,
        trusted_faces=int(trusted.sum()), eligible_faces=len(face_ids),
        excluded_rank_faces=int(trusted.sum()) - len(face_ids), clipped_faces=clipped,
        minimum_trusted_sigma=float(singular[trusted, 1].min()) if trusted.any() else None,
        maximum_inverse_norm=float(np.linalg.norm(inverse.astype(np.float64), ord=2, axis=(-2, -1)).max()) if len(face_ids) else 0.0)


def relative_loss(prediction_denoiser, prepared, config, torch):
    """FP32 scalar Huber of Frobenius RMS; gradients only through prediction."""
    if prediction_denoiser.dtype != torch.float32:
        raise ValueError('Relative geometry loss must use FP32 residual proposals')
    ids = torch.from_numpy(prepared['face_ids']).to(device=prediction_denoiser.device)
    if not len(prepared['face_ids']):
        return prediction_denoiser.sum() * 0.0
    prediction = prediction_denoiser[ids]
    edges = torch.stack((prediction[:, 1] - prediction[:, 0], prediction[:, 2] - prediction[:, 0]), dim=-1)
    target = torch.from_numpy(prepared['target_edges']).to(device=prediction_denoiser.device)
    inverse = torch.from_numpy(prepared['inverse']).to(device=prediction_denoiser.device)
    deformation = torch.matmul(edges - target, inverse)
    norm = torch.linalg.vector_norm(deformation.reshape(-1, 6), dim=-1) / math.sqrt(6.0)
    delta = float(config['huber_delta'])
    values = torch.where(norm <= delta, 0.5 * norm.square(), delta * (norm - 0.5 * delta))
    return values.mean()
