"""One formal shared-tip update; no probe forwards or optimizer replay."""
import time
import numpy as np
import torch
from meshflow_control.models.native import native_execution
from meshflow_control.seam.occupancy import occupancy_bce_loss
from .train import tensors, validate_optimizers
from . import support_c as S
from ..shared_tip import shared_tip_forward, independent_fm_loss, decode_predictions
from ..pointer import pointer_ce_loss


def validate_pointer(pointer, optimizer, step, model, occupancy):
    if sum(p.numel() for p in pointer.parameters()) != 3970 or len(optimizer.param_groups) != 1:
        raise ValueError('Pointer capacity/group changed')
    group = optimizer.param_groups[0]
    if (group['lr'], tuple(group['betas']), group['weight_decay']) != (.001, (.9, .95), 0.):
        raise ValueError('Pointer fresh AdamW hyperparameters changed')
    if [id(p) for p in group['params']] != [id(p) for p in pointer.parameters()]:
        raise ValueError('Pointer AdamW ownership mismatch')
    if {id(p) for p in pointer.parameters()} & {id(p) for m in (model, occupancy) for p in m.parameters()}:
        raise ValueError('Pointer must not share parameters with generator/occupancy')
    for p in pointer.parameters():
        state = optimizer.state.get(p, {})
        if step == 0 and not state: continue
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'} or float(state['step']) != step:
            raise ValueError('Pointer independent AdamW step differs')
        if p.dtype != torch.float32 or any(state[k].dtype != torch.float32 or state[k].shape != p.shape for k in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('Pointer parameter/moment precision changed')


def update(model, optimizer, head, head_optimizer, pointer, pointer_optimizer, samples, arm, step, ledger, hyper, bins):
    denominator, edge_denominator = S.validate_batch(samples)
    pointer_denominator = sum(len(s['pointer_labels']) for s in samples)
    validate_optimizers(model, optimizer, head, head_optimizer, step - 1, hyper)
    validate_pointer(pointer, pointer_optimizer, step - 1, model, head)
    torch.cuda.reset_peak_memory_stats()
    for opt in (optimizer, head_optimizer, pointer_optimizer): opt.zero_grad(set_to_none=True)
    started = time.perf_counter()
    result = dict(FM=0., occupancy_BCE=0., pointer_CE=0.,
                  free_coordinate_denominator=denominator,
                  original_free_coordinate_denominator=sum(int(s['free_coordinate_count']) for s in samples),
                  alias_corners=sum(s['aliased_corner_count'] for s in samples),
                  occupancy_edge_denominator=edge_denominator,pointer_query_denominator=pointer_denominator,
                  forward_seconds=0.,backward_seconds=0.,head_forward_seconds=0.,head_backward_seconds=0.,
                  pointer_forward_seconds=0.,pointer_backward_seconds=0.)
    diagnostics, head_rows, pointer_rows = [], [], []
    for micro, sample in enumerate(samples):
        b = tensors(sample)
        context = dict(local_step=step,global_step=S.BASE_STEP+step,microbatch=micro,
                       sample_index=sample['sample_index'],uid=sample['uid'],alpha=sample['alpha'])
        with native_execution(model, coordinates=(b['x'], b['t']), autocast=True):
            prediction, elapsed = ledger.call('train_forward', lambda: shared_tip_forward(
                model,b['x'],b['t'],b['y'],b['valid'],b['C'],b['corner'],sample['alias_corner_ids']).float(), context)
            result['forward_seconds'] += elapsed
            loss = independent_fm_loss(prediction,b['target'],sample['train_free_corner_mask'],denominator)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite shared-tip FM')
            value = float(loss.detach())
            _, elapsed = ledger.call('train_backward', loss.backward, context)
            result['backward_seconds'] += elapsed
        predicted = prediction.detach().cpu().numpy()[0].reshape(sample['N'],3,3)
        row = S.diagnostic(sample,predicted,bins)
        independent_sse = float(((predicted.astype(np.float64)-sample['u'])**2)[sample['train_free_corner_mask']].sum())
        if not np.isclose(independent_sse/denominator,value,rtol=3e-6,atol=1e-7):
            raise ValueError('Independent FM diagnostic denominator differs')
        row.update(**context,FM_contribution=value,independent_velocity_sse=independent_sse,
                   batch_free_coordinate_denominator=denominator,
                   sample_free_coordinate_denominator=sample['train_free_coordinate_count'],
                   original_free_coordinate_count=int(sample['free_coordinate_count']),
                   alias_corner_count=sample['aliased_corner_count'],
                   all_free_diagnostic_includes_aliases=True)
        diagnostics.append(row); result['FM'] += value
        features = torch.as_tensor(sample['edge_features'],device='cuda')
        labels = torch.as_tensor(sample['occupied_labels'],device='cuda')
        mask = torch.ones(labels.shape,dtype=torch.bool,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits, elapsed = ledger.call('head_forward',lambda:head(features,mask),context)
            result['head_forward_seconds'] += elapsed
            bce = occupancy_bce_loss(logits,labels,mask,class_weights=[8.25,66/124])*len(labels)/edge_denominator
            if not bool(torch.isfinite(bce)): raise FloatingPointError('Nonfinite occupancy BCE')
            _, elapsed = ledger.call('head_backward',bce.backward,context)
            result['head_backward_seconds'] += elapsed
        result['occupancy_BCE'] += float(bce.detach())
        occupied = (logits.detach().float()>=0).cpu().numpy(); truth=sample['occupied_labels']
        head_rows.append(dict(**context,edge_count=len(truth),positive_count=int(truth.sum()),null_count=int((~truth).sum()),
                              true_positive=int((occupied&truth).sum()),true_negative=int((~occupied&~truth).sum()),
                              false_positive=int((occupied&~truth).sum()),false_negative=int((~occupied&truth).sum()),
                              whole_layout_match=bool(np.array_equal(occupied,truth)),weighted_BCE_contribution=float(bce.detach()),
                              edge_batch_denominator=edge_denominator,logits=logits.detach().float().cpu().tolist(),labels=truth.tolist()))
        pair = torch.as_tensor(sample['pointer_pair_features'],device='cuda')
        null = torch.as_tensor(sample['pointer_null_features'],device='cuda')
        pointer_labels = torch.as_tensor(sample['pointer_labels'],device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pointer_logits, elapsed = ledger.call('pointer_forward',lambda:pointer(pair,null),context)
            result['pointer_forward_seconds'] += elapsed
            ce = pointer_ce_loss(pointer_logits,pointer_labels,denominator=pointer_denominator)
            if not bool(torch.isfinite(ce)): raise FloatingPointError('Nonfinite pointer CE')
            _, elapsed = ledger.call('pointer_backward',ce.backward,context)
            result['pointer_backward_seconds'] += elapsed
        result['pointer_CE'] += float(ce.detach())
        scores = pointer_logits.detach().float().cpu().numpy()
        decoded, facts = decode_predictions(scores,sample,sample['anchor_ids'],sample['known_mask'])
        truth_alias = sample['alias_corner_ids']; query=sample['pointer_query_corner_ids']
        pred_ids=decoded.reshape(-1)[query]; true_ids=truth_alias.reshape(-1)[query]
        tp=int(((pred_ids==true_ids)&(true_ids>=0)).sum())
        pointer_rows.append(dict(**context,query_count=len(query),source_nonnull=int((true_ids>=0).sum()),
                                 predicted_nonnull=int((pred_ids>=0).sum()),correct_nonnull=tp,
                                 missed_nonnull=int(((true_ids>=0)&(pred_ids!=true_ids)).sum()),
                                 false_positive_null=int(((true_ids<0)&(pred_ids>=0)).sum()),
                                 correct_null=int(((true_ids<0)&(pred_ids<0)).sum()),
                                 whole_alias_match=bool(np.array_equal(pred_ids,true_ids)),
                                 same_face_conflict_rejections=facts['pointer_conflict_null'],
                                 CE_contribution=float(ce.detach()),batch_query_denominator=pointer_denominator,
                                 true_pointer_root_edges_disabled_by_predicted_occupancy=int(sum(
                                     not occupied[sample['pointer_tip_edge_ids'][lab-1]] for lab in sample['pointer_labels'] if lab>0)),
                                 interpretation='Teacher source occupancy candidates; generator uses source aliases. Inference predicts both heads; not rollout accuracy.'))
        del prediction,loss,b,features,labels,mask,logits,bce,pair,null,pointer_labels,pointer_logits,ce
    for name, module in [('generator',model),('head',head),('pointer',pointer)]:
        if any(p.grad is None for p in module.parameters()): raise RuntimeError(name+' has unused parameters')
        result[name+'_gradient_norm']=float(torch.nn.utils.clip_grad_norm_(module.parameters(),1.,error_if_nonfinite=True))
    context=dict(local_step=step,global_step=S.BASE_STEP+step)
    for kind, opt in [('optimizer',optimizer),('head_optimizer',head_optimizer),('pointer_optimizer',pointer_optimizer)]:
        _,result[kind+'_seconds']=ledger.call(kind+'_update',opt.step,context)
    if any(not bool(torch.isfinite(p).all()) for m in (model,head,pointer) for p in m.parameters()):
        raise FloatingPointError('Nonfinite updated parameter')
    validate_optimizers(model,optimizer,head,head_optimizer,step,hyper)
    validate_pointer(pointer,pointer_optimizer,step,model,head)
    result.update(update_seconds=time.perf_counter()-started,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  peak_scope='One formal batch: generator and two heads, all backward and three AdamW steps')
    return result,diagnostics,head_rows,pointer_rows
