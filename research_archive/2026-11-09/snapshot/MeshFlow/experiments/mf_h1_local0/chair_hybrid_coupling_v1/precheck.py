"""Finite real-model wiring checks: ten forwards, two zero-update backwards."""
import numpy as np
import torch
from native_t1.model import native_execution
from native_t1.data import model_inputs
from native_t1.training import masked_fm_loss
from experiments.mf_h1_local0.control_mvp48.train_stream import collate
from experiments.mf_h1_local0.control_mvp48.train_io import capture_rng,restore_rng,tree_hash
from .common import SOURCE,CHAIR,START,digest,atomic_json
from .checkpoint import tensor_state_hash
from .stream import HybridStream
from .coupling import old_slot_sample,map_velocity_old_to_common


def run(out,model,optimizer,meta,budget):
    initial=dict(model=tensor_state_hash(model.state_dict()),optimizer=tree_hash(optimizer.state_dict()),file=digest(START))
    assert initial['model']==meta['model_state_sha256'] and initial['optimizer']==meta['optimizer_state_sha256']
    rng=capture_rng()
    base_stream=HybridStream(SOURCE,CHAIR,out,'OT_BASE');hybrid_stream=HybridStream(SOURCE,CHAIR,out,'OT_HYBRID')
    base=base_stream.next_effective_batch();hybrid=hybrid_stream.next_effective_batch()
    assert [s['shared_input_sha256'] for s in base]==[s['shared_input_sha256'] for s in hybrid]
    indices=[next(i for i,s in enumerate(base) if s['alpha']==1.),next(i for i,s in enumerate(base) if s['alpha']!=1.)]
    rows=[]
    for j,i in enumerate(indices):
        s=base[i];old=old_slot_sample(s);results={}
        for mode,sample,bf16 in [('base_bf16',s,True),('old_bf16',old,True),('base_fp32',s,False),('old_fp32',old,False),('hybrid_bf16',hybrid[i],True)]:
            do_backward=j==0 and mode in ('base_bf16','hybrid_bf16')
            b=collate([sample],next(model.parameters()).device)
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(do_backward),native_execution(model,coordinates=(b['xt'],b['t']),autocast=bf16) as facts:
                v=budget.call('precheck_forward',model,*model_inputs(b),context=dict(sample_index=i,mode=mode))
                if not bool(torch.isfinite(v).all()):raise FloatingPointError('Nonfinite precheck velocity')
                results[mode]=v.detach().float().cpu().numpy()[0]
                if do_backward:
                    loss=masked_fm_loss(v.float(),b['u'],b['valid_mask'],b['known_mask'],denominator=int(b['loss_mask'].sum())*9)
                    budget.call('precheck_backward',loss.backward,context=dict(sample_index=i,mode=mode))
                    grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True))
                else:grad_norm=None
            rows.append(dict(sample_index=i,uid=s['object_id'],alpha=s['alpha'],mode=mode,N=int(s['y']),K=s['K'],
                             dtype=str(v.dtype),runtime=facts,zero_update_backward=do_backward,gradient_norm=grad_norm))
        for mode in ('bf16','fp32'):
            mapped=map_velocity_old_to_common(s,results['old_'+mode])
            actual=results['base_'+mode];d=np.abs(mapped-actual)
            tolerance=.015625+.03125*np.abs(actual) if mode=='bf16' else 1e-4+1e-4*np.abs(actual)
            rows.append(dict(sample_index=i,mode=mode+'_slot_comparison',max_abs=float(d.max()),rms=float(np.sqrt(np.mean(d*d))),
                             coordinates_outside_descriptive_tolerance=int((d>tolerance).sum()),
                             scope='Numerical observation only. Historical BF16 permutation gate is not redefined or claimed repaired.'))
    optimizer.zero_grad(set_to_none=True);restore_rng(rng)
    final=dict(model=tensor_state_hash(model.state_dict()),optimizer=tree_hash(optimizer.state_dict()),file=digest(START))
    if initial!=final or tree_hash(capture_rng())!=tree_hash(rng):raise ValueError('Precheck mutated initial model/AdamW/RNG')
    result=dict(status='PASS',actual_forwards=10,actual_zero_update_backwards=2,optimizer_updates=0,initial=initial,after=final,
                rng_restored=True,rows=rows,OT_counts_base=base_stream.counts,OT_counts_hybrid=hybrid_stream.counts,
                quality_is_not_an_implementation_gate=True)
    atomic_json(out/'precheck/result.json',result);return result
