"""One training implementation for all four phases; precheck never steps AdamW."""
import time
import torch
from native_t1.model import native_execution
from native_t1.data import model_inputs
from native_t1.training import masked_fm_loss
from .train_stream import collate


def backward_batch(model, samples, *, objective=None, additional_step=None, counts=None):
    if len(samples) != 8:
        raise ValueError('Effective batch8/microbatch1 is fixed')
    device = next(model.parameters()).device
    counts = {} if counts is None else counts
    def inc(key): counts[key] = counts.get(key, 0) + 1
    denominator = 9 * sum(int((s['valid_mask'] & ~s['known_mask']).sum()) for s in samples)
    prepared = objective.prepare(samples) if objective is not None else None
    report = dict(FM=0., geometry=0., loss=0., free_coordinate_denominator=denominator,
                  active_samples=0, geometry_diagnostics=[])
    for index, sample in enumerate(samples):
        batch = collate([sample], device)
        with native_execution(model, coordinates=(batch['xt'], batch['t']), autocast=device.type == 'cuda') as runtime:
            inc('forward_attempts')
            velocity = model(*model_inputs(batch)).float()
            inc('forward_returns')
            loss = masked_fm_loss(velocity, batch['u'], batch['valid_mask'],
                                 batch['known_mask'], denominator=denominator)
            report['FM'] += float(loss.detach())
            if objective is not None:
                aux, diagnostic = objective.auxiliary(velocity, batch, prepared[index], additional_step=additional_step)
                report['geometry'] += float(aux.detach())
                report['active_samples'] += int(diagnostic['active'])
                report['geometry_diagnostics'].append(diagnostic)
                loss = loss + aux
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite actual loss')
            report['loss'] += float(loss.detach())
            inc('backward_attempts'); loss.backward(); inc('backward_returns')
    report['runtime'] = runtime
    return report


def optimizer_update(model, optimizer, samples, *, objective=None, additional_step=None, counts=None):
    counts = {} if counts is None else counts
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError('AdamW required')
    params = list(model.parameters())
    if model.training or not all(p.requires_grad and p.dtype == torch.float32 for p in params):
        raise ValueError('All model parameters must remain trainable FP32 in eval mode')
    opt_params = [p for group in optimizer.param_groups for p in group['params']]
    if len(opt_params) != len(params) or {id(p) for p in opt_params} != {id(p) for p in params}:
        raise ValueError('Optimizer must own all parameters exactly once')
    for group in optimizer.param_groups:
        if (group['lr'], tuple(group['betas']), group['weight_decay']) != (1e-5, (.9, .95), 0.):
            raise ValueError('Optimizer configuration differs')
    tick = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    report = backward_batch(model, samples, objective=objective, additional_step=additional_step, counts=counts)
    report['grad_norm_before_clip'] = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
    counts['optimizer_step_attempts'] = counts.get('optimizer_step_attempts', 0) + 1
    counts['optimizer_step_in_progress'] = True
    counts['update_validated'] = False
    optimizer.step()
    counts['optimizer_step_in_progress'] = False
    counts['optimizer_updates'] = counts.get('optimizer_updates', 0) + 1
    if params[0].is_cuda:
        torch.cuda.synchronize()
    if not all(bool(torch.isfinite(p).all()) for p in params):
        raise FloatingPointError('Nonfinite parameter after optimizer update; no retry')
    counts['update_validated'] = True
    report['train_seconds'] = time.perf_counter() - tick
    report['counts'] = dict(counts)
    return report
