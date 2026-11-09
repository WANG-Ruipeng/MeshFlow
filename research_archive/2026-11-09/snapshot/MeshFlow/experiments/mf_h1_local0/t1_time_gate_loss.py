"""Fixed t>=0.5 relation gate for the separately authorized C_gate arm.

The caller supplies the original effective-batch FC/FF group denominators,
including every inactive sample. This module never changes FM, the time
distribution, the optimizer, or the original relation definitions. Importing
does not execute fixtures, autograd, a model, data OT, or an update.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np
import torch

from .t1_interface_relations import endpoint_from_velocity, relation_sums


GATE_THRESHOLD = 0.5
FC_WEIGHT = 0.1
FF_WEIGHT = 0.1


def _denominator(value, local_count, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(name + ' must be the integer effective-batch group count.')
    value = int(value)
    if value < local_count or value < 0:
        raise ValueError(name + ' excludes groups in this microbatch (including inactive groups).')
    return value


def gated_terms(batch, velocity, prepared, fc_den, ff_den):
    """Return (FC, FF, JSON-ready audit), normalized over the effective batch.

    ``prepared`` is an original prepare_relations mapping for B=1, or one
    mapping per sample for B>1. Group indices remain local to each sample.
    The caller must pass *all* effective-batch groups, not only active groups;
    local validation cannot prove the composition of samples outside this call.

    Below threshold, the sample contributes exactly ``velocity.sum()*0`` to
    each auxiliary term and does not execute endpoint/relation calculations.
    On/above threshold, endpoint and relation arithmetic are the old functions
    in their original order. This training helper reads the detached scalar
    times to CPU for an explicit gate/audit; it is not a sampling-loop kernel.
    Returned FC/FF do not include the fixed 0.1 outer coefficients.
    """
    if velocity.ndim != 3 or velocity.shape[-1] != 9 or velocity.dtype != torch.float32:
        raise ValueError('Require original FP32 velocity [B,N,9].')
    batch_size = velocity.shape[0]
    if not batch_size:
        raise ValueError('Empty microbatches are not supported.')
    for key in ('xt', 'context'):
        value = batch[key]
        if value.shape != velocity.shape or value.dtype != torch.float32 or value.device != velocity.device:
            raise ValueError(key + ' must match original FP32 velocity shape/device.')
    times = batch['t']
    mask = batch['known_mask']
    if times.shape != (batch_size,) or times.dtype != torch.float32 or times.device != velocity.device:
        raise ValueError('Require one original FP32 time per sample on the same device.')
    if mask.shape != velocity.shape[:2] or mask.dtype != torch.bool or mask.device != velocity.device:
        raise ValueError('Require original boolean known mask.')
    if isinstance(prepared, Mapping):
        if batch_size != 1:
            raise ValueError('B>1 requires one local prepared relation mapping per sample.')
        packs = [prepared]
    else:
        packs = list(prepared)
        if len(packs) != batch_size or not all(isinstance(p, Mapping) for p in packs):
            raise ValueError('Prepared relation mappings must match batch size.')
    counts = [(int(p['FC_count']), int(p['FF_count'])) for p in packs]
    if any(fc < 0 or ff < 0 for fc, ff in counts):
        raise ValueError('Negative relation group count.')
    fc_den = _denominator(fc_den, sum(x[0] for x in counts), 'fc_den')
    ff_den = _denominator(ff_den, sum(x[1] for x in counts), 'ff_den')
    stored_times = times.detach().cpu().tolist()
    if any(not np.isfinite(t) or not 0.0 <= t <= 1.0 for t in stored_times):
        raise ValueError('Nonfinite/out-of-range original training time.')

    fc_parts, ff_parts, rows = [], [], []
    with torch.autocast(device_type=velocity.device.type, enabled=False):
        for i, (pack, t, (fc_count, ff_count)) in enumerate(zip(packs, stored_times, counts)):
            active = t >= GATE_THRESHOLD
            v = velocity[i:i + 1]
            if active:
                endpoint = endpoint_from_velocity(
                    batch['xt'][i:i + 1], v, times[i:i + 1],
                    mask[i:i + 1], batch['context'][i:i + 1])
                terms = relation_sums(endpoint, pack)
                fc = terms['FC_sum'] / max(fc_den, 1)
                ff = terms['FF_sum'] / max(ff_den, 1)
            else:
                zero = v.sum() * 0
                fc, ff = zero, zero
            fc_parts.append(fc)
            ff_parts.append(ff)
            rows.append(dict(sample_index=i, stored_t=t, active=active,
                             FC_count=fc_count, FF_count=ff_count))
        # B=1 does not gain an extra zero-addition relative to the old formula.
        fc, ff = fc_parts[0], ff_parts[0]
        for next_fc, next_ff in zip(fc_parts[1:], ff_parts[1:]):
            fc, ff = fc + next_fc, ff + next_ff
    active_count = sum(row['active'] for row in rows)
    audit = dict(
        threshold=GATE_THRESHOLD, predicate='stored_FP32_t >= 0.5',
        FC_weight=FC_WEIGHT, FF_weight=FF_WEIGHT,
        FC_group_denominator=fc_den, FF_group_denominator=ff_den,
        local_FC_groups=sum(x[0] for x in counts), local_FF_groups=sum(x[1] for x in counts),
        active_FC_groups=sum(row['FC_count'] for row in rows if row['active']),
        active_FF_groups=sum(row['FF_count'] for row in rows if row['active']),
        active_samples=active_count, inactive_samples=batch_size-active_count,
        normalization='Original effective-batch group totals including inactive groups',
        active_fraction_compensation=False, FM_changed=False,
        endpoint_calls=active_count, legacy_relation_calls=active_count, samples=rows)
    return fc, ff, audit


def gated_relation_loss(fm, batch, velocity, prepared, fc_den, ff_den):
    """Return (FM + .1 FC + .1 FF, unweighted components, audit)."""
    if not torch.is_tensor(fm) or fm.ndim != 0:
        raise ValueError('Pass the already normalized scalar original FM loss.')
    fc, ff, audit = gated_terms(batch, velocity, prepared, fc_den, ff_den)
    return fm + FC_WEIGHT * fc + FF_WEIGHT * ff, dict(FM=fm, FC=fc, FF=ff), audit


def run_cpu_tests():
    """Explicit small CPU fixtures; caller owns invocation and accounting.

    All equivalence checks use identical original FP32 operations and require
    bitwise equality (no relaxed numerical tolerance). No model or OT is used.
    """
    from .t1_interface_relations import build_relation_labels, prepare_relations
    from native_t1.training import masked_fm_loss

    counts = dict(model_forwards=0, model_backwards=0, optimizer_updates=0,
                  OT_calls=0, CPU_autograd_grad_calls=0, CPU_autograd_grad_returns=0,
                  gated_terms_calls=0, legacy_reference_endpoint_calls=0,
                  legacy_reference_relation_calls=0, gated_endpoint_calls=0,
                  gated_legacy_relation_calls=0, validation_rejection_calls=0)
    fixture_names = []
    rng = np.random.RandomState(64051)
    ids = np.asarray([[0,1,2], [0,2,3], [0,3,4], [1,3,4]], dtype=np.int64)
    vertices = rng.randn(5,3).astype(np.float32)

    def fixture(times):
        b = len(times)
        known = np.asarray([[True, True, False, False] if i % 2 == 0
                            else [True, False, False, False] for i in range(b)])
        context = np.zeros((b,4,3,3), dtype=np.float32)
        packs = []
        for i in range(b):
            context[i,known[i]] = vertices[ids[known[i]]]
            labels = build_relation_labels(ids, known[i], context[i])
            pack = prepare_relations(labels, 'cpu')
            pack['fc_anchors'].requires_grad_(True)
            packs.append(pack)
        batch = dict(xt=torch.from_numpy(rng.randn(b,4,9).astype(np.float32)),
                     context=torch.from_numpy(context.reshape(b,4,9)).requires_grad_(),
                     known_mask=torch.from_numpy(known), valid_mask=torch.ones((b,4),dtype=torch.bool),
                     t=torch.tensor(times,dtype=torch.float32),
                     u=torch.from_numpy(rng.randn(b,4,9).astype(np.float32)))
        velocity = torch.from_numpy(rng.randn(b,4,9).astype(np.float32)).requires_grad_()
        return batch, velocity, packs

    def grad(loss, args):
        counts['CPU_autograd_grad_calls'] += 1
        result = torch.autograd.grad(loss, args, allow_unused=True)
        counts['CPU_autograd_grad_returns'] += 1
        return result

    def gated(batch, velocity, packs, fc_den, ff_den):
        counts['gated_terms_calls'] += 1
        result = gated_terms(batch, velocity, packs, fc_den, ff_den)
        counts['gated_endpoint_calls'] += result[2]['endpoint_calls']
        counts['gated_legacy_relation_calls'] += result[2]['legacy_relation_calls']
        return result

    def old(batch, v, pack, fc_den, ff_den):
        counts['legacy_reference_endpoint_calls'] += 1
        endpoint = endpoint_from_velocity(batch['xt'], v, batch['t'], batch['known_mask'], batch['context'])
        counts['legacy_reference_relation_calls'] += 1
        terms = relation_sums(endpoint, pack)
        return terms['FC_sum']/max(fc_den,1), terms['FF_sum']/max(ff_den,1)

    for t in (.25, .5, .75):
        batch, v, packs = fixture([t])
        pack = packs[0]
        den = (8*pack['FC_count'], 8*pack['FF_count'])
        fc, ff, audit = gated(batch, v, pack, *den)
        ref_fc, ref_ff = old(batch, v, pack, *den)
        args = (v, batch['context'], pack['fc_anchors'])
        actual_grad = grad(.1*fc+.1*ff, args)
        old_grad = grad(.1*ref_fc+.1*ref_ff, args)
        assert actual_grad[1:] == (None,None) and old_grad[1:] == (None,None)
        if t < .5:
            assert fc.item() == ff.item() == 0 and not torch.count_nonzero(actual_grad[0])
            assert ref_fc.item()+ref_ff.item()>0 and torch.count_nonzero(old_grad[0])>0
            assert audit['endpoint_calls'] == audit['legacy_relation_calls'] == 0
        else:
            assert torch.equal(fc,ref_fc) and torch.equal(ff,ref_ff)
            assert torch.equal(actual_grad[0],old_grad[0])
        assert torch.count_nonzero(actual_grad[0][batch['known_mask']]) == 0
    fixture_names.append('below_at_above_threshold_original_loss_and_gradient_bitwise_or_exact_zero')
    fixture_names.append('known_context_and_C_anchors_detached')

    near = [float(np.nextafter(np.float32(.5),np.float32(0))),
            float(np.nextafter(np.float32(.5),np.float32(1)))]
    batch, v, packs = fixture(near)
    fc_den, ff_den = (sum(p[k] for p in packs) for k in ('FC_count','FF_count'))
    _, _, audit = gated(batch,v,packs,fc_den,ff_den)
    assert [row['active'] for row in audit['samples']] == [False,True]
    fixture_names.append('adjacent_representable_FP32_times')

    batch, v, packs = fixture([.25,.5,.75,.1,.2,.9,.49,.5001])
    fc_den, ff_den = (sum(p[k] for p in packs) for k in ('FC_count','FF_count'))
    fc, ff, audit = gated(batch,v,packs,fc_den,ff_den)
    ref_parts, micro_parts = [], []
    for i, pack in enumerate(packs):
        one = {k:value[i:i+1] for k,value in batch.items()}
        if float(batch['t'][i]) >= .5:
            ref_parts.append(old(one,v[i:i+1],pack,fc_den,ff_den))
        else:
            z = v[i:i+1].sum()*0
            ref_parts.append((z,z))
        micro_parts.append(gated(one,v[i:i+1],pack,fc_den,ff_den)[:2])
    def add(parts, column):
        value = parts[0][column]
        for part in parts[1:]:
            value = value + part[column]
        return value
    ref_fc,ref_ff = add(ref_parts,0),add(ref_parts,1)
    micro_fc,micro_ff = add(micro_parts,0),add(micro_parts,1)
    assert fc_den==24 and ff_den==20 and audit['active_FC_groups']==12 and audit['active_FF_groups']==11
    assert torch.equal(fc,ref_fc) and torch.equal(ff,ref_ff)
    assert torch.equal(fc,micro_fc) and torch.equal(ff,micro_ff)
    g, = grad(.1*fc+.1*ff,(v,))
    ref_g, = grad(.1*ref_fc+.1*ref_ff,(v,))
    micro_g, = grad(.1*micro_fc+.1*micro_ff,(v,))
    assert torch.equal(g,ref_g) and torch.equal(g,micro_g)
    inactive = batch['t'] < .5
    assert torch.count_nonzero(g[inactive]) == 0
    fixture_names.append('mixed_effective8_denominators_include_inactive_and_microbatch_accumulation_matches')
    try:
        gated_terms(batch,v,packs,audit['active_FC_groups'],audit['active_FF_groups'])
    except ValueError:
        counts['validation_rejection_calls'] += 1
    else:
        raise AssertionError('Active-only group denominators were accepted.')
    fixture_names.append('active_only_denominator_rejected_for_full_effective_batch')

    batch, v, packs = fixture([.1,.3])
    fc_den, ff_den = (sum(p[k] for p in packs) for k in ('FC_count','FF_count'))
    fc,ff,audit = gated(batch,v,packs,fc_den,ff_den)
    g, = grad(fc+ff,(v,))
    assert fc.item()==ff.item()==0 and torch.count_nonzero(g)==0 and audit['active_samples']==0
    fixture_names.append('all_inactive_graph_connected_zero')

    batch, v, packs = fixture([.25,.75])
    fc_den,ff_den = (sum(p[k] for p in packs) for k in ('FC_count','FF_count'))
    fm = masked_fm_loss(v,batch['u'],batch['valid_mask'],batch['known_mask'])
    total,components,audit = gated_relation_loss(fm,batch,v,packs,fc_den,ff_den)
    counts['gated_terms_calls'] += 1
    counts['gated_endpoint_calls'] += audit['endpoint_calls']
    counts['gated_legacy_relation_calls'] += audit['legacy_relation_calls']
    assert components['FM'] is fm
    actual, = grad(total,(v,))
    reference_fm = masked_fm_loss(v,batch['u'],batch['valid_mask'],batch['known_mask'])
    one = {k:value[1:2] for k,value in batch.items()}
    ref_fc,ref_ff = old(one,v[1:2],packs[1],fc_den,ff_den)
    reference, = grad(reference_fm+.1*ref_fc+.1*ref_ff,(v,))
    assert torch.equal(actual,reference) and torch.count_nonzero(actual[0])>0
    fixture_names.append('FM_keeps_full_time_support_and_fixed_outer_weights')

    return dict(status='PASS',fixtures=fixture_names,counts=counts,
                numerical_policy='Identical FP32 formula: bitwise equivalence; inactive exact zero',
                model_forwards=0,model_backwards=0,optimizer_updates=0,
                CPU_autograd_grad_calls=counts['CPU_autograd_grad_calls'])
