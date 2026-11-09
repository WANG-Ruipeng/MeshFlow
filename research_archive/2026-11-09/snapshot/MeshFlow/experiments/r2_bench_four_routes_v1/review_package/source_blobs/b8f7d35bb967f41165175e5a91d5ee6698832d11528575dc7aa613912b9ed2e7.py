"""Chair18049-parameter contextual head with frozen TRAIN-only normalization.

Buffers add no trainable parameters. Closed-C inference bypasses the head.
Feature scaling is explicitly additional to the historical Chair setup.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from r2.pilots.chair_multipart_seam_v1.head import MultipartOccupancyHead, HEAD_CONFIG as BASE_CONFIG
from r2.pilots.bench_parts_convergence_v1.data import edge_features, inference_layout
from meshflow_control.seam.slot_data import L

FEATURE_SCHEMA = 'C_edge24_relative_bbox3_world_y1_v1'
HEAD_CONFIG = copy.deepcopy(BASE_CONFIG)
NORMALIZATION_SCHEMA = 'BENCH_C_ONLY_TRAIN_STANDARDIZATION_V1'
THRESHOLD_LOGIT = 0.


def features(C_vertices, C_faces_original, total_N):
    v = np.asarray(C_vertices, np.float32)
    edges, base = edge_features(v, C_faces_original, int(total_N))
    if not len(edges):
        return np.empty((0, 28), np.float32)
    mid = (v[edges[:, 1]] + v[edges[:, 2]]) * np.float32(.5)
    center = (v.min(0) + v.max(0)) * np.float32(.5)
    extent = np.maximum(np.ptp(v, axis=0), np.float32(1e-6*L))
    return np.concatenate((base, (mid-center)/extent, mid[:, 1:2]/np.float32(L)), 1).astype(np.float32)


class NormalizedOccupancyHead(MultipartOccupancyHead):
    def __init__(self, **config):
        super().__init__(**(config or HEAD_CONFIG))
        self.register_buffer('feature_mean', torch.zeros(28, dtype=torch.float32))
        self.register_buffer('feature_scale', torch.ones(28, dtype=torch.float32))

    def forward(self, x, mask=None):
        return super().forward((x.float()-self.feature_mean)/self.feature_scale, mask)


def new_head(seed, scaler):
    if scaler['schema'] != NORMALIZATION_SCHEMA or scaler['DEV_used']:
        raise ValueError('Frozen eligible TRAIN-only scaler required')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = NormalizedOccupancyHead(**HEAD_CONFIG)
    model.feature_mean.copy_(torch.as_tensor(scaler['mean'], dtype=torch.float32))
    model.feature_scale.copy_(torch.as_tensor(scaler['scale'], dtype=torch.float32))
    if not bool(torch.isfinite(model.feature_mean).all() and torch.isfinite(model.feature_scale).all()) or not bool((model.feature_scale > 0).all()):
        raise ValueError('Invalid normalization')
    return model


@torch.no_grad()
def predict_layout(head, bundle):
    required = {'C_vertices', 'C_faces_original', 'C_faces_model', 'total_N'}
    if set(bundle) != required or head.training:
        raise ValueError('Only C/N inference bundle with eval head allowed')
    feat = features(bundle['C_vertices'], bundle['C_faces_original'], bundle['total_N'])
    if len(feat):
        logits = head(torch.from_numpy(feat).to(next(head.parameters()).device)).float().cpu().numpy()
    else:
        logits = np.empty(0, np.float32)
    return inference_layout(**bundle, occupied=logits >= THRESHOLD_LOGIT), logits


def load_head(path, device='cpu'):
    p = torch.load(path, map_location='cpu', weights_only=True)
    if p['head_config'] != HEAD_CONFIG or p['threshold_logit'] != THRESHOLD_LOGIT or p['normalization_schema'] != NORMALIZATION_SCHEMA:
        raise ValueError('Head contract differs')
    h = NormalizedOccupancyHead(**p['head_config'])
    h.load_state_dict(p['head_state'], strict=True)
    return h.to(device).eval().requires_grad_(False), p
