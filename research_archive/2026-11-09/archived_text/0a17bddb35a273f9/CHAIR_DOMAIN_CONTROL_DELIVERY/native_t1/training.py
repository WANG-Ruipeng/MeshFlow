"""Native masked FM training building blocks; importing never starts training.

The original mixed-K effective batch is 8, microbatch is 1, and FM normalization
covers all free coordinates in the effective batch. Optional geometry objectives
share the backward pass and average their sample losses over all eight samples.
No scheduler, EMA or checkpoint selection is provided. Callers own the budget,
persistence and interruptions.
"""
from __future__ import annotations
import torch
from .data import native_collate, model_inputs


def masked_fm_loss(velocity, target_velocity, valid_mask, known_mask, *, denominator=None):
    """Unchanged free-only squared velocity error; known/padding excluded."""
    if velocity.shape!=target_velocity.shape or velocity.ndim!=3 or velocity.shape[-1]!=9:
        raise ValueError('Velocity arrays must share [B,N,9] shape.')
    if valid_mask.shape!=velocity.shape[:2] or known_mask.shape!=valid_mask.shape:
        raise ValueError('Mask shapes differ from velocity faces.')
    if valid_mask.dtype!=torch.bool or known_mask.dtype!=torch.bool:
        raise ValueError('Boolean masks required.')
    mask=valid_mask&~known_mask
    count=int(mask.sum())*9
    if count==0:
        raise ValueError('A training batch must contain valid free coordinates.')
    divisor=count if denominator is None else int(denominator)
    if divisor<count:
        raise ValueError('Effective-batch denominator cannot be smaller than this microbatch.')
    return (velocity.float()-target_velocity.float()).square()[mask].sum()/divisor


def make_optimizer(model):
    """Exact original full-model AdamW settings; does not perform an update."""
    parameters=list(model.parameters())
    if not parameters or not all(p.requires_grad for p in parameters):
        raise ValueError('Caller must explicitly enable every Native parameter for training.')
    return torch.optim.AdamW(parameters,lr=1e-5,betas=(.9,.95),weight_decay=0.)


def train_step(model,optimizer,samples,*,device='cuda',counts=None,objective=None,additional_step=None):
    """One explicitly requested original update; called only by an explicit trainer.

    Numerical or deterministic failures propagate without fallback. Counters
    retain attempted forwards/backwards/steps if an exception interrupts work.
    A caller must persist them and must not assume a failed step was atomic.
    """
    from .model import native_execution
    if len(samples)!=8:
        raise ValueError('Preserve original effective batch8 with microbatch1.')
    if model.training or not all(p.requires_grad for p in model.parameters()):
        raise ValueError('Native training requires eval mode with every parameter trainable.')
    parameters=list(model.parameters())
    if any(p.dtype!=torch.float32 for p in parameters):
        raise ValueError('Native parameters must remain FP32.')
    if not isinstance(optimizer,torch.optim.AdamW):
        raise ValueError('Original Native training requires AdamW.')
    if {id(p) for group in optimizer.param_groups for p in group['params']} != {id(p) for p in parameters}:
        raise ValueError('Optimizer must own every Native parameter exactly once.')
    for group in optimizer.param_groups:
        if (group['lr'],tuple(group['betas']),group['weight_decay']) != (1e-5,(.9,.95),0.):
            raise ValueError('Original Native optimizer settings changed.')
    counts={} if counts is None else counts
    def increment(key): counts[key]=counts.get(key,0)+1
    denominator=9*sum(int((s['valid_mask']&~s['known_mask']).sum()) for s in samples)
    if denominator not in (7524,7452,7488):
        raise ValueError('Original mixed-K effective denominator changed.')
    prepared = objective.prepare(samples) if objective is not None else None
    optimizer.zero_grad(set_to_none=True); total=0.; fm_total=0.; geometry_total=0.; active_count=0
    geometry_diagnostics=[]
    for micro, sample in enumerate(samples):
        batch=native_collate([sample],device)
        with native_execution(model,coordinates=batch['xt']) as runtime:
            increment('forward_attempts')
            velocity=model(*model_inputs(batch)).float()
            increment('forward_returns')
            loss=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'],denominator=denominator)
            fm_total += float(loss.detach())
            if objective is not None:
                auxiliary, diagnostic = objective.auxiliary(velocity, batch, prepared[micro],
                                                            additional_step=additional_step)
                loss = loss + auxiliary
                geometry_total += float(auxiliary.detach())
                active_count += int(diagnostic["active"])
                geometry_diagnostics.append(diagnostic)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite free FM loss; no numerical fallback.')
            increment('backward_attempts'); loss.backward(); increment('backward_returns')
        total+=float(loss.detach())
    norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
    increment('optimizer_step_attempts'); optimizer.step(); increment('optimizer_updates')
    if parameters[0].is_cuda:
        torch.cuda.synchronize()
    if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()):
        raise RuntimeError('Nonfinite parameter after attempted update.')
    return dict(loss=total,FM=fm_total,geometry=geometry_total,active_samples=active_count,
                geometry_diagnostics=geometry_diagnostics,grad_norm_before_clip=norm,free_coordinate_denominator=denominator,
                effective_batch=8,microbatch=1,runtime=runtime,counts=dict(counts))
