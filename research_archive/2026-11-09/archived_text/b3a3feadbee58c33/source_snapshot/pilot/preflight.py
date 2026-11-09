"""Bounded preflight helpers. GPU execution is explicitly dispatched by root.

teacher_preflight(records, phase): 2 historical inputs, two calls each (4F).
student_preflight(batches, phase): two real effective batches of eight (16F/B,
2 discarded AdamW updates), plus seven diagnostic forwards (23F total).
No teacher/student simultaneous residency; run in separate processes/phases.
The caller supplies verified arrays/cache identities; this module never draws
new training inputs and never selects a scientific endpoint.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='4'
import gc
import math
from pathlib import Path

import numpy as np
import torch

from . import common as C
from .model import make_student,load_teacher,effective_proposal,export_student,load_student,tensor_state_sha256
from .inference import single_student_proposal,residual_diagnostics


def _input(row):
    x=np.asarray(row['input_d']);p=np.asarray(row['proposal_T_d'])
    if x.dtype!=np.float32 or p.dtype!=np.float32 or x.shape!=p.shape or x.ndim!=3 or x.shape[1:]!=(3,3):
        raise ValueError('Preflight requires exact FP32 input_d and teacher proposal_T_d')
    n=len(x);k=int(row['K']);known=np.arange(n)<k
    if not 0<k<n or not np.isfinite(x).all() or not np.isfinite(p).all():raise ValueError('Invalid preflight input')
    return x,p,known


def teacher_preflight(records,phase,*,verify_checkpoint_hash=True):
    """records: exactly2 {raw,C,proposal_model, identity}; no approximate d target.

    proposal_model must be the original historical pre-C projection; matching
    it and restored preGEO bytes validates the input to unchanged CPU GEO.
    The controller/evaluator owns any final indexed POST replay audit.
    """
    if len(records)!=2:raise ValueError('Teacher preflight fixed to two historical inputs')
    teacher,spec=load_teacher(verify_checkpoint_hash=verify_checkpoint_hash)
    teacher.to('cuda').eval().requires_grad_(False)
    before=tensor_state_sha256(teacher.state_dict());rows=[]
    for index,row in enumerate(records):
        raw=np.asarray(row['raw']);c=np.asarray(row['C']);historic=np.asarray(row['proposal_model'])
        if raw.dtype!=np.float32 or raw.shape!=historic.shape or raw[:len(c)].tobytes()!=c.tobytes():raise ValueError('Historical input invalid')
        x=torch.from_numpy(raw.copy()).to('cuda').reshape(1,len(raw),9)*C.SCALE
        rng_before=C.rng_hash();outputs=[]
        for repeat in range(2):
            def call():
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):return effective_proposal(teacher,x)
            proposal,residual=phase.call('teacher_preflight_F',call,dict(case=index,repeat=repeat,identity=row.get('identity')))
            outputs.append(proposal.detach().cpu().numpy().copy())
        if C.rng_hash()!=rng_before:raise AssertionError('Teacher consumed RNG')
        assert np.array_equal(outputs[0],outputs[1])
        model=(proposal.float()/C.SCALE).reshape(raw.shape).cpu().numpy()
        diff=float(np.max(np.abs(model.astype(np.float64)-historic.astype(np.float64))))
        if model.tobytes()!=historic.tobytes():raise AssertionError('Historical teacher proposal replay not byte exact: '+str(diff))
        restored=model.copy();restored[:len(c)]=c
        if 'restored' in row and restored.tobytes()!=np.asarray(row['restored']).tobytes():raise AssertionError('Historical restored preGEO mismatch')
        rows.append(dict(index=index,identity=row.get('identity'),N=len(raw),K=len(c),historical_proposal_bitwise=True,
                         repeated_proposal_bitwise=True,C_original_bits_restored=restored[:len(c)].tobytes()==c.tobytes(),max_abs=diff,no_rng_consumed=True))
    if tensor_state_sha256(teacher.state_dict())!=before or any(p.grad is not None or p.requires_grad for p in teacher.parameters()):
        raise AssertionError('Teacher modified or gradients enabled')
    out=dict(status='PASS',teacher_forwards=4,teacher_backward=0,teacher_updates=0,rows=rows,
             state_unchanged=True,load_spec=spec,peak=C.memory(),
             POST_contract='Exact historical proposal/restored preGEO; same fixed GEO is checked in independent CPU evaluator')
    C.write(C.OUT/'preflight/teacher.json',out)
    del teacher;gc.collect();torch.cuda.empty_cache()
    return out


def _permuted(a,faces,corners):
    return np.stack([a[f,corners[i]] for i,f in enumerate(faces)],axis=0)


def _unpermuted(a,faces,corners):
    b=np.empty_like(a)
    for i,f in enumerate(faces):b[f,corners[i]]=a[i]
    return b


def student_preflight(batches,phase,*,initial_seed=None):
    """batches=list of2 lists of8 exact teacher-cache rows.

    Each row: input_d FP32[N,3,3], proposal_T_d same, K, optional identity.
    No resampling, GT target, teacher query, or persistent formal update occurs.
    """
    if len(batches)!=2 or any(len(batch)!=8 for batch in batches):raise ValueError('Exactly2 batch8 required')
    for batch in batches:
        for row in batch:_input(row)
    initial_seed=C.seed('student_initialization') if initial_seed is None else int(initial_seed)
    student,spec=make_student(initial_seed);initial_hash=tensor_state_sha256(student.state_dict())
    student.to('cuda').train()
    first_x,first_t,first_known=_input(batches[0][0]);n=len(first_x)
    original_C=np.asarray(batches[0][0].get('C',np.divide(first_x[:int(first_known.sum())],np.float32(C.SCALE),dtype=np.float32)))
    arrays,init=single_student_proposal(student,C_original=original_C,input_d=first_x,
                       phase=phase,call_kind='student_preflight_F',identity='zero_output')
    if np.count_nonzero(arrays['effective_residual_d']) or arrays['proposal_d'].tobytes()!=first_x.tobytes():raise AssertionError('Initial student residual not zero')
    # Scaling roundtrip is recorded separately; never assert RAW free bytes are
    # identical simply because the denoiser residual initializes at zero.
    scale_roundtrip=np.divide(np.multiply(first_x,np.float32(1/C.SCALE),dtype=np.float32),np.float32(1/C.SCALE),dtype=np.float32)
    init['diagnostic_scale_roundtrip_max_abs_d']=float(np.max(np.abs(scale_roundtrip-first_x)))
    init['free_identity_claim']='exact in denoiser coordinates only; inverse-scale roundoff is separate'
    student.train()
    opt=torch.optim.AdamW(student.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=0.,foreach=False)
    logs=[];micro_checks=[];torch.cuda.reset_peak_memory_stats();rng_start=C.rng_hash()
    parameter_before=student.base.post_quant_linear.weight.detach().clone()
    for step,batch in enumerate(batches,1):
        opt.zero_grad(set_to_none=True)
        denominator=9*sum(len(_input(r)[0])-int(r['K']) for r in batch)
        batch_loss=0.;input_c_grad=[]
        for micro,row in enumerate(batch):
            x_np,t_np,known=_input(row)
            x=torch.from_numpy(x_np.copy()).to('cuda').reshape(1,len(x_np),9).requires_grad_(True)
            t=torch.from_numpy(t_np.copy()).to('cuda').reshape_as(x)
            mask=torch.from_numpy(~known).to('cuda')
            def call():
                with torch.autocast('cuda',dtype=torch.bfloat16):return effective_proposal(student,x)
            p,r=phase.call('student_preflight_F',call,dict(step=step,micro=micro,identity=row.get('identity')))
            target=(t.float()-x.detach().float()).detach()
            loss=((r[:,mask]-target[:,mask])**2).sum()/denominator
            if not torch.isfinite(loss):raise FloatingPointError('Invalid KD loss')
            rn=r.detach().cpu().numpy()[0].reshape(x_np.shape)
            tn=np.subtract(t_np,x_np,dtype=np.float32)
            diff=np.subtract(rn[~known],tn[~known],dtype=np.float32)
            reference=float(np.sum(np.multiply(diff,diff,dtype=np.float32),dtype=np.float64)/denominator)
            value=float(loss.detach());relative=abs(value-reference)/max(abs(reference),1e-30)
            if not math.isclose(value,reference,rel_tol=1e-5,abs_tol=1e-12):raise AssertionError('Independent free scalar loss mismatch')
            # Changing known targets alone must leave this independent loss exact.
            altered=tn.copy();altered[known]=12345.
            if not np.array_equal(diff,np.subtract(rn[~known],altered[~known],dtype=np.float32)):raise AssertionError('Known target leaked into loss')
            phase.call('student_preflight_B',lambda:loss.backward(),dict(step=step,micro=micro))
            input_c_grad.append(float(x.grad[:,~mask].float().abs().max().item()))
            batch_loss+=value
            micro_checks.append(dict(step=step,micro=micro,N=len(x_np),K=int(known.sum()),denominator=denominator,
                                     KD=value,independent_reference=reference,relative_difference=relative,C_target_excluded=True))
        backbone_grad=float(student.base.post_quant_linear.weight.grad.float().norm().item())
        grad_norm=float(torch.nn.utils.clip_grad_norm_(student.parameters(),1.,foreach=False).item())
        if not math.isfinite(grad_norm):raise FloatingPointError('Invalid gradient')
        phase.call('student_discarded_updates',opt.step,dict(step=step,discarded=True))
        torch.cuda.synchronize()
        logs.append(dict(step=step,KD=batch_loss,denominator=denominator,backbone_grad_norm=backbone_grad,
                         global_grad_norm_before_clip=grad_norm,C_input_grad_max=max(input_c_grad),memory=C.memory()))
        if step==2 and (backbone_grad==0 or max(input_c_grad)==0):raise AssertionError('Second update did not propagate through backbone/C features')
    if C.rng_hash()!=rng_start:raise AssertionError('Dropout/noise consumed RNG during student updates')
    if torch.equal(parameter_before,student.base.post_quant_linear.weight.detach()):raise AssertionError('Backbone parameters did not update after two steps')
    train_memory=C.memory()

    # Nonzero trained disposable state prevents a vacuous zero-output
    # permutation check. Small fixture is a deterministic slice of cache input.
    student.eval();small=first_x[:min(12,n)].copy();N=len(small)
    rng=np.random.default_rng(C.seed('preflight_permutation'))
    faces=rng.permutation(N);corners=np.stack([rng.permutation(3) for _ in range(N)])
    perm=_permuted(small,faces,corners)
    if not np.array_equal(small,_unpermuted(perm,faces,corners)):raise AssertionError('Joint inverse map incorrect')
    perm_results={};reference_bf16=None
    for bf16 in (False,True):
        outputs=[]
        for label,data in (('original',small),('permuted',perm)):
            x=torch.from_numpy(data.copy()).to('cuda').reshape(1,N,9)
            def call():
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):return effective_proposal(student,x)
            p,r=phase.call('student_preflight_F',call,dict(check='permutation',BF16=bf16,ordering=label))
            outputs.append(p.reshape(N,3,3).cpu().numpy().copy())
        restored=_unpermuted(outputs[1],faces,corners)
        err=np.abs(outputs[0].astype(np.float64)-restored.astype(np.float64))
        if not np.isfinite(err).all():raise FloatingPointError('Permutation nonfinite')
        perm_results['BF16' if bf16 else 'FP32']=dict(max_abs=float(err.max()),RMS=float(np.sqrt(np.mean(err**2))),
            original_proposal_norm=float(np.linalg.norm(outputs[0])),bitwise=bool(np.array_equal(outputs[0],restored)),
            fixed_allclose_atol_1e_6_rtol_1e_5=bool(np.allclose(outputs[0],restored,atol=1e-6,rtol=1e-5)),
            note='Joint mapping exact; floating reduction difference reported, not silently relaxed or interpreted as structural proof.')
        if bf16:reference_bf16=outputs[0]
    # One repeated trained call checks train/eval dropout is absent; prior zero
    # output cannot establish that all stochastic paths were disabled.
    x=torch.from_numpy(small.copy()).to('cuda').reshape(1,N,9);student.train()
    def repeated():
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):return effective_proposal(student,x)
    p,r=phase.call('student_preflight_F',repeated,'trained_train_mode_dropout0')
    if not np.array_equal(p.reshape(N,3,3).cpu().numpy(),reference_bf16):raise AssertionError('Train/eval output mismatch with dropout0')
    student.eval()
    exported=export_student(C.OUT/'preflight/discarded_student_inference.pt',student,2,dict(discarded_preflight_only=True))
    del student,opt,x,p,r,loss,target,t,mask,parameter_before;gc.collect();torch.cuda.empty_cache()
    # Teacher is absent, training instance and optimizer removed. Only student
    # export is loaded. This is not a second training initialization/branch.
    loaded,loaded_spec,payload=load_student(C.OUT/'preflight/discarded_student_inference.pt',require_step=2)
    del payload
    loaded.to('cuda').eval();x=torch.from_numpy(small.copy()).to('cuda').reshape(1,N,9)
    def reloaded():
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):return effective_proposal(loaded,x)
    p,r=phase.call('student_preflight_F',reloaded,'independent_student_export')
    if not np.array_equal(p.reshape(N,3,3).cpu().numpy(),reference_bf16):raise AssertionError('Student export output differs')
    zeros=np.zeros((2,3,3),np.float32);tiny=zeros.copy();tiny[1]=1e-8
    low=residual_diagnostics(zeros,zeros,tiny,np.array([True,False]))
    if low['R'] is not None or low['relative_status']!='LOW_SIGNAL' or low['cosine'] is not None:raise AssertionError('Low-signal division guard failed')
    out=dict(status='PASS',student_forwards=23,student_backwards=16,discarded_updates=2,formal_updates=0,
             discarded_update_optimizer=dict(lr=1e-4,betas=[.9,.95],weight_decay=0.,foreach=False,clip=1.,note='Nonzero-gradient diagnostic only; differs from formal warmup and is discarded'),
             initial_seed=initial_seed,initial_state_sha256=initial_hash,initial_structure=spec,zero_output=init,
             update_checks=logs,micro_loss_checks=micro_checks,permutation=perm_results,
             low_signal=low,export_receipt=exported,export_independent_reload=True,teacher_loaded=False,
             no_rng_consumed=True,full_training_peak=train_memory,
             disposition='Discard both preflight updates; formal trainer loads exact original initial export, fresh AdamW and stream cursor0')
    C.write(C.OUT/'preflight/student.json',out)
    del loaded;gc.collect();torch.cuda.empty_cache()
    return out


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('teacher','student'),required=True);args=p.parse_args()
    C.runtime()
    data=C.read(C.OUT/'manifest/data.json')
    completion=C.OUT/'preflight'/f'{args.stage}_completion.json'
    if completion.exists():
        saved=C.read(completion)
        if saved['status']!='PASS' or C.sha(saved['result_path'])!=saved['result_sha256']:raise ValueError('Preflight completion changed')
        print('SKIP_VERIFIED_COMPLETE',args.stage);return
    def arrays(path):
        with np.load(path,allow_pickle=False) as z:return {key:z[key].copy() for key in z.files}
    if args.stage=='teacher':
        byid={r['sample_id']:r for r in data['dev_real']}
        ids=[data['benchmark_ids'][0],data['benchmark_ids'][-1]]
        if len(set(ids))!=2:raise ValueError('Need two distinct preselected replay records')
        records=[]
        for sid in ids:
            row=byid[sid]
            if C.sha(row['raw_path'])!=row['raw_sha256']:raise ValueError('Historical RAW changed')
            b=arrays(row['input_path']);old=arrays(row['historical_teacher_model_space_cache'])
            records.append(dict(raw=b['input_model'],C=b['C'],proposal_model=old['official_unconstrained_proposal'],
                                restored=old['output'],identity=sid))
    else:
        if C.read(C.OUT/'teacher_cache/completion.json')['status']!='COMPLETE':raise ValueError('All teacher labels must be frozen first')
        byid={r['sample_id']:r for r in data['train_inputs']}
        stream=C.read(data['train_stream_path'])
        if C.sha(data['train_stream_path'])!=data['train_stream_sha256']:raise ValueError('Frozen train stream changed')
        batches=[]
        for batch in stream['batches'][:2]:
            items=[]
            for sid in batch['sample_ids']:
                row=byid[sid];path=Path(row['teacher_cache_path']);receipt=C.read(path.with_suffix('.json'))
                if C.sha(path)!=receipt['sha256'] or C.sha(row['input_path'])!=receipt['input_sha256'] or receipt['teacher_sha256']!=C.EXPECTED:
                    raise ValueError('Teacher preflight label identity mismatch')
                b=arrays(path);items.append(dict(input_d=b['input_d'],proposal_T_d=b['proposal_T_d'],K=int(b['K']),C=b['C'],identity=sid))
            if 9*sum(len(v['input_d'])-v['K'] for v in items)!=batch['free_scalar_denominator']:raise ValueError('Batch denominator differs')
            batches.append(items)
    phase=C.Phase(args.stage+'_preflight',estimated_mib=2500 if args.stage=='teacher' else 3000)
    try:
        result=teacher_preflight(records,phase) if args.stage=='teacher' else student_preflight(batches,phase)
        result_path=C.OUT/'preflight'/f'{args.stage}.json'
        C.write(completion,dict(status='PASS',result_path=str(result_path),result_sha256=C.sha(result_path),
                               counts=dict(phase.counts),source_sha256=C.sha(__file__)))
        phase.finish();print(args.stage,'preflight PASS',phase.counts)
    except BaseException as exc:
        phase.finish('FAILED',repr(exc));raise


if __name__=='__main__':main()
