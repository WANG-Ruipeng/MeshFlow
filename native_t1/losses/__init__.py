"""Exact, training-only finite interface edge and surface MMD objectives.

Importing never reads data, initializes CUDA, or creates a model. Time gates,
ramps, effective-batch normalization and optimizer updates belong to the caller.
"""
import torch
from .targets import TrainingGeometryTargets
from .edge import prepare_edge_target, edge_loss
from .surface import prepare_distribution_target, distribution_loss, work_counters

LAMBDA_EDGE = 4.071385484299878
LAMBDA_L5 = 0.3266104383520167


def prepare_targets(label):
    return {'D_edge':prepare_edge_target(label),
            'L5':prepare_distribution_target(torch.as_tensor(label['free_gt']),label['bbox_L'])}


def evaluate(name,pred_free,target):
    if name=='D_edge':
        return edge_loss(pred_free,target)
    if name=='L5':
        return distribution_loss(name,pred_free,target)
    raise ValueError('Only D_edge and L5 are supported')
