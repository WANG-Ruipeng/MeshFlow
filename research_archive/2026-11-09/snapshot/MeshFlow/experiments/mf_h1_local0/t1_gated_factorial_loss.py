"""Fixed FC/FF switches on the already validated C_gate relation objective.

M10 applies only FC; M01 applies only FF. FF retains its original interface
and free-only shared groups. No model, optimizer, OT, fixture, or backward
operation runs on import. The caller owns training budgets and denominators.
"""
from __future__ import annotations

import torch

from .t1_time_gate_loss import FC_WEIGHT, FF_WEIGHT, gated_terms


NUMERICAL_POLICY = dict(
    identical_formula='Bitwise FP32 equality',
    oracle='Exactly zero on the fixed dyadic-coordinate oracle fixture',
    inactive='Graph-connected exact zero for selected terms',
    permutation_atol=5e-7,
    permutation_rtol=5e-6,
)


def switch_loss(fm, batch, velocity, prepared, fc_den, ff_den, use_fc, use_ff):
    """Return (total, {FM, FC, FF}, audit); disabled components are None.

    The original gated_terms computes both relation sums. Only selected terms
    enter total, so an omitted term is not a reported zero loss. No relation
    groups are filtered: in particular FF keeps interface-sharing groups.
    The original t>=.5 predicate, .1 coefficients, and full effective-batch
    group denominators (including inactive samples) are unchanged. All-on
    uses the same addition order as C_gate; all-off returns fm itself.
    """
    if type(use_fc) is not bool or type(use_ff) is not bool:
        raise TypeError('FC/FF switches must be explicit fixed booleans.')
    if not torch.is_tensor(fm) or fm.ndim != 0:
        raise ValueError('Pass the already normalized scalar original FM loss.')
    fc, ff, gate = gated_terms(batch, velocity, prepared, fc_den, ff_den)
    total = fm
    if use_fc:
        total = total + FC_WEIGHT * fc
    if use_ff:
        total = total + FF_WEIGHT * ff
    components = dict(FM=fm, FC=fc if use_fc else None, FF=ff if use_ff else None)
    audit = dict(gate)
    audit.update(
        use_fc=use_fc, use_ff=use_ff,
        FC_status='APPLIED_WITH_TIME_GATE' if use_fc else 'NOT_APPLIED',
        FF_status='APPLIED_WITH_TIME_GATE' if use_ff else 'NOT_APPLIED',
        FC_weight=FC_WEIGHT if use_fc else None,
        FF_weight=FF_WEIGHT if use_ff else None,
        computed_relation_terms=['FC','FF'],
        applied_relation_terms=([name for name, enabled in (('FC',use_fc),('FF',use_ff)) if enabled]),
        applied_active_FC_groups=gate['active_FC_groups'] if use_fc else None,
        applied_active_FF_groups=gate['active_FF_groups'] if use_ff else None,
        disabled_component_value=None,
        FF_group_filtering=False,
        FF_includes_interface_shared_groups=True,
        changes_to_original_gate_or_denominators=False,
    )
    return total, components, audit


def run_cpu_tests():
    """Minimal fixed math gates; caller executes once and records the counts.

    Four CPU autograd calls are expected: all-on and its old reference, then
    each selected auxiliary term separately. No old complete fixture suite is
    invoked. There is no model, optimizer, OT, asset read, or geometry test.
    """
    import numpy as np
    from native_t1.training import masked_fm_loss
    from .t1_interface_relations import build_relation_labels, prepare_relations
    from .t1_time_gate_loss import gated_relation_loss

    counts = dict(model_forwards=0, model_backwards=0, optimizer_updates=0,
                  OT_calls=0, CPU_autograd_grad_calls=0, CPU_autograd_grad_returns=0,
                  switch_calls=0, old_gate_reference_calls=0, gated_terms_calls=0,
                  endpoint_calls=0, legacy_relation_calls=0)
    fixtures = []
    ids = np.asarray([[0,1,2],[0,2,3],[0,3,4],[1,3,4]],dtype=np.int64)
    known = np.asarray([True,True,False,False])
    vertices = np.arange(15,dtype=np.float32).reshape(5,3)/8-.5
    context = np.zeros((4,3,3),dtype=np.float32)
    context[known] = vertices[ids[known]]
    target = np.zeros((4,3,3),dtype=np.float32)
    target[~known] = vertices[ids[~known]]
    z = np.arange(36,dtype=np.float32).reshape(4,3,3)/16-1
    z[known] = 0
    labels = build_relation_labels(ids,known,context)
    assert labels['FC_count']==3 and labels['FF_count']==2
    interface_ids = {g['source_vertex_id'] for g in labels['fc_groups']}
    ff_ids = {g['source_vertex_id'] for g in labels['ff_groups']}
    assert interface_ids & ff_ids and ff_ids - interface_ids

    def fixture(times, oracle=False):
        b = len(times)
        batch = dict(
            t=torch.tensor(times,dtype=torch.float32),
            known_mask=torch.from_numpy(np.tile(known[None],(b,1))),
            valid_mask=torch.ones((b,4),dtype=torch.bool),
            context=torch.from_numpy(np.tile(context.reshape(1,4,9),(b,1,1))).requires_grad_(),
            x0=torch.from_numpy(np.tile(z.reshape(1,4,9),(b,1,1))),
            x1=torch.from_numpy(np.tile(target.reshape(1,4,9),(b,1,1))),
        )
        batch['u'] = batch['x1']-batch['x0']
        t = batch['t'][:,None,None]
        xt = (1-t)*batch['x0']+t*batch['x1']
        batch['xt'] = torch.where(batch['known_mask'][...,None],batch['context'].detach(),xt)
        v = batch['u'].clone()
        if not oracle:
            v = v + torch.arange(36,dtype=torch.float32).reshape(1,4,9)/64
        v.requires_grad_(True)
        packs = [prepare_relations(labels,'cpu') for _ in times]
        for pack in packs:
            pack['fc_anchors'].requires_grad_(True)
        return batch,v,packs

    def fm_loss(batch,v):
        return masked_fm_loss(v,batch['u'],batch['valid_mask'],batch['known_mask'])

    def record(audit):
        counts['gated_terms_calls'] += 1
        counts['endpoint_calls'] += audit['endpoint_calls']
        counts['legacy_relation_calls'] += audit['legacy_relation_calls']

    def call(fm,batch,v,packs,fc_den,ff_den,use_fc,use_ff):
        counts['switch_calls'] += 1
        value = switch_loss(fm,batch,v,packs,fc_den,ff_den,use_fc,use_ff)
        record(value[2])
        return value

    def grad(loss,args):
        counts['CPU_autograd_grad_calls'] += 1
        result = torch.autograd.grad(loss,args,allow_unused=True)
        counts['CPU_autograd_grad_returns'] += 1
        return result

    batch,v,packs = fixture([.5],oracle=True)
    _,parts,_ = call(fm_loss(batch,v),batch,v,packs,24,16,True,True)
    assert parts['FM'].item()==parts['FC'].item()==parts['FF'].item()==0
    fixtures.append('dyadic_oracle_preserves_zero_FM_FC_FF')

    for t in (.25,.5,.75):
        batch,v,packs = fixture([t])
        _,parts,audit = call(fm_loss(batch,v),batch,v,packs,24,16,True,True)
        assert audit['samples'][0]['active']==(t>=.5)
        if t<.5:
            assert parts['FC'].item()==parts['FF'].item()==0
            assert audit['endpoint_calls']==audit['legacy_relation_calls']==0
        else:
            assert parts['FC'].item()>0 and parts['FF'].item()>0
    fixtures.append('below_at_above_original_FP32_threshold')

    batch,v,packs = fixture([.25,.5,.75,.1,.2,.9,.49,.5001])
    fc_den,ff_den = 24,16
    total,parts,audit = call(fm_loss(batch,v),batch,v,packs,fc_den,ff_den,True,True)
    counts['old_gate_reference_calls'] += 1
    reference,ref_parts,ref_audit = gated_relation_loss(
        fm_loss(batch,v),batch,v,packs,fc_den,ff_den)
    record(ref_audit)
    assert torch.equal(total,reference)
    assert all(torch.equal(parts[k],ref_parts[k]) for k in ('FM','FC','FF'))
    g, = grad(total,(v,))
    reference_g, = grad(reference,(v,))
    assert torch.equal(g,reference_g)
    assert audit['FC_group_denominator']==24 and audit['FF_group_denominator']==16
    assert audit['active_FC_groups']==12 and audit['active_FF_groups']==8
    assert not audit['active_fraction_compensation']
    fixtures.append('all_on_loss_and_gradient_bitwise_equal_old_C_gate_full_effective8_denominators')

    fm = fm_loss(batch,v)
    off,parts,audit = call(fm,batch,v,packs,fc_den,ff_den,False,False)
    assert off is fm and parts['FM'] is fm
    assert parts['FC'] is None and parts['FF'] is None
    assert audit['FC_status']==audit['FF_status']=='NOT_APPLIED'
    fixtures.append('all_off_is_FM_identity_and_disabled_terms_are_None')

    for name, switches in [('FC',(True,False)),('FF',(False,True))]:
        _,parts,audit = call(fm_loss(batch,v),batch,v,packs,fc_den,ff_den,*switches)
        other = 'FF' if name=='FC' else 'FC'
        assert parts[other] is None and audit[other+'_status']=='NOT_APPLIED'
        args = (v,batch['context'])+tuple(p['fc_anchors'] for p in packs)
        gradients = grad(parts[name],args)
        assert all(x is None for x in gradients[1:])
        g = gradients[0]
        assert torch.count_nonzero(g[batch['t']<.5])==0
        assert torch.count_nonzero(g[batch['known_mask']])==0
        assert torch.count_nonzero(g[(batch['t']>=.5),:])>0
        assert audit['FF_group_filtering'] is False and audit['FF_includes_interface_shared_groups'] is True
    fixtures.append('single_terms_nonzero_free_gradient_inactive_zero_detached_C_and_unfiltered_FF')

    # Permute face and corner identities together; build trusted labels again.
    batch,v,packs = fixture([.5])
    _,before,_ = call(fm_loss(batch,v),batch,v,packs,24,16,True,True)
    order = np.asarray([2,0,3,1])
    corners = np.asarray([[2,0,1],[0,2,1],[1,0,2],[2,1,0]])
    def permute_array(a):
        return np.take_along_axis(a[order],corners[...,None],axis=1)
    ids2 = np.take_along_axis(ids[order],corners,axis=1)
    context2 = permute_array(context)
    labels2 = build_relation_labels(ids2,known[order],context2)
    pack2 = prepare_relations(labels2,'cpu')
    batch2 = dict(t=batch['t'].clone(),known_mask=torch.from_numpy(known[order][None]),
                  valid_mask=torch.ones((1,4),dtype=torch.bool))
    for key in ('xt','context','u'):
        array = batch[key].detach().numpy().reshape(4,3,3)
        batch2[key] = torch.from_numpy(permute_array(array).reshape(1,4,9))
    v2 = torch.from_numpy(permute_array(v.detach().numpy().reshape(4,3,3)).reshape(1,4,9))
    _,after,_ = call(fm_loss(batch2,v2),batch2,v2,pack2,24,16,True,True)
    for key in ('FM','FC','FF'):
        assert torch.allclose(before[key],after[key],
                              atol=NUMERICAL_POLICY['permutation_atol'],
                              rtol=NUMERICAL_POLICY['permutation_rtol'])
    assert {g['source_vertex_id'] for g in labels2['ff_groups']}==ff_ids
    fixtures.append('synchronized_face_corner_permutation_keeps_FM_FC_FF_and_FF_membership')
    assert counts['CPU_autograd_grad_calls']==counts['CPU_autograd_grad_returns']==4
    return dict(status='PASS',fixtures=fixtures,counts=counts,
                numerical_policy=dict(NUMERICAL_POLICY),
                CPU_autograd_grad_calls=counts['CPU_autograd_grad_calls'],
                model_forwards=0,model_backwards=0,optimizer_updates=0)
