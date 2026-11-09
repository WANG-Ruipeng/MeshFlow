"""Single-call student inference and exact residual diagnostics.

These functions never dispatch a model themselves on import. The controller
owns C.Phase and supplies a phase/call_kind when a model call is authorized.
Pure student load uses only its export and verified original source code.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='4'
from contextlib import nullcontext
from pathlib import Path
import math
import re
import time

import numpy as np
import torch

from . import common as C
from .model import StudentDenoiser,effective_proposal,load_student


def _fp32(a,name):
    a=np.asarray(a)
    if a.dtype!=np.float32 or a.ndim!=3 or a.shape[1:]!=(3,3) or not np.isfinite(a).all():
        raise ValueError(name+' must be finite FP32 [N,3,3]')
    return a


def single_student_proposal(student:StudentDenoiser, *, C_original:np.ndarray,
        input_model:np.ndarray|None=None, input_d:np.ndarray|None=None,
        known_mask:np.ndarray|None=None, device='cuda', bf16=True,
        phase=None,call_kind='student_eval_F',identity=None):
    """One forward; exact FP32 denoiser-domain contract and original C restore.

    Supply input_model for actual RAW. Synthetic frozen inputs supply input_d,
    optionally input_model for a byte-exact displacement origin. known_mask
    defaults to the first K faces. It never enters the student network.
    Return (arrays, diagnostic receipt fields), without writing files or GEO.
    """
    if not isinstance(student,StudentDenoiser):raise TypeError('Expected independent student')
    student.eval()
    c=_fp32(C_original,'C_original')
    raw=_fp32(input_model,'input_model') if input_model is not None else None
    xd=_fp32(input_d,'input_d') if input_d is not None else None
    if raw is None and xd is None:raise ValueError('Need input_model or exact input_d')
    n=len(raw) if raw is not None else len(xd)
    if raw is not None and xd is not None and raw.shape!=xd.shape:raise ValueError('Input shape mismatch')
    known=np.arange(n)<len(c) if known_mask is None else np.asarray(known_mask)
    if known.dtype!=np.bool_ or known.shape!=(n,) or int(known.sum())!=len(c):raise ValueError('Invalid C mask')
    if raw is not None and raw[known].tobytes()!=c.tobytes():raise ValueError('RAW differs from original C')
    x=torch.from_numpy(np.ascontiguousarray(xd if xd is not None else raw)).to(device=device,dtype=torch.float32).reshape(1,n,9)
    if xd is None:x=x*C.SCALE
    count=torch.tensor([n],device=x.device,dtype=torch.long)
    before=C.rng_hash()
    def call():
        with torch.inference_mode(),torch.autocast(device_type=x.device.type,dtype=torch.bfloat16,enabled=bf16):
            return effective_proposal(student,x,count)
    if x.device.type=='cuda':torch.cuda.synchronize()
    started=time.perf_counter()
    proposal_d,residual_d=(phase.call(call_kind,call,identity) if phase is not None else call())
    if x.device.type=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    after=C.rng_hash()
    if before!=after:raise AssertionError('Student single-step inference consumed RNG')
    with torch.inference_mode(),torch.autocast(device_type=x.device.type,enabled=False):
        proposal_model=(proposal_d.float()/C.SCALE).reshape(n,3,3).cpu().numpy().copy()
        input_model_roundtrip=(x.float()/C.SCALE).reshape(n,3,3).cpu().numpy().copy()
    output=proposal_model.copy();output[known]=c
    if output[known].tobytes()!=c.tobytes():raise AssertionError('C restoration not byte exact')
    input_saved=x.detach().reshape(n,3,3).cpu().numpy().copy()
    proposal_saved=proposal_d.detach().reshape(n,3,3).cpu().numpy().copy()
    residual_saved=residual_d.detach().reshape(n,3,3).cpu().numpy().copy()
    if not np.array_equal(residual_saved,np.subtract(proposal_saved,input_saved,dtype=np.float32)):
        raise AssertionError('Effective residual cache mismatch')
    arrays=dict(input_d=input_saved,proposal_d=proposal_saved,effective_residual_d=residual_saved,
                residual_d=residual_saved,proposal_model=proposal_model,proposal=proposal_model,
                proposal_S_d=proposal_saved,r_S_d=residual_saved,proposal_S_model=proposal_model,
                output=output,C=c.copy(),known=known.copy(),
                displacement_from_RAW=output-(raw if raw is not None else input_model_roundtrip))
    if not all(np.isfinite(a).all() for a in arrays.values()):raise FloatingPointError('Nonfinite inference')
    info=dict(N=n,K=len(c),model_calls=1,no_rng_consumed=True,C_original_bits_restored=True,
              BF16=bool(bf16),parameters_dtype='FP32',residual_add_sub_dtype='FP32',
              scale=C.SCALE,model_forward_wall_seconds=seconds,teacher_instance_used=False,
              input_d_source='frozen exact FP32 input' if xd is not None else 'FP32 RAW*SCALE on device',
              displacement_reference='original RAW bytes' if raw is not None else 'input_d inverse-scale diagnostic')
    return arrays,info


def residual_diagnostics(input_d,proposal_s_d,proposal_t_d,known_mask,*,L:float|None=None):
    """Compatibility wrapper around the single registered CPU metric definition."""
    from . import distill_metrics as DM
    known=np.asarray(known_mask)
    if known.dtype!=np.bool_ or known.shape!=(len(input_d),):raise ValueError('KD mask shape mismatch')
    if L is not None and float(L)!=DM.L:raise ValueError('Do not silently change registered L')
    # General public mask becomes a C-prefix for the registered scorer only;
    # this has no effect on the sum or quantiles and never runs a model.
    order=np.concatenate([np.flatnonzero(known),np.flatnonzero(~known)])
    out=DM.residual_metrics(np.asarray(input_d)[order],np.asarray(proposal_t_d)[order],
                            np.asarray(proposal_s_d)[order],int(known.sum()))
    out.update(relative_status=out['signal_status'],cosine=out['residual_cosine'])
    return out


def save_real_result(arrays,info,*,generator_label,request_id,raw_sha256,student_checkpoint_sha256):
    for name in (generator_label,request_id):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',name) or name in ('.','..'):raise ValueError('Unsafe output identity')
    dest=C.OUT/'inference'/C.STUDENT/generator_label/request_id
    path=dest/'denoised.npz';receipt_path=dest/'denoised.json'
    if path.exists() or receipt_path.exists():raise FileExistsError('Controller must verify/skip completed result before model call')
    C.save_npz(path,**arrays)
    identity=dict(label=generator_label,request_id=request_id,raw_sha256=raw_sha256,
                  student_checkpoint_sha256=student_checkpoint_sha256,model_checkpoint_sha256=student_checkpoint_sha256)
    receipt=dict(status='PASS',**identity,identity=identity,output_sha256=C.sha(path),**info)
    C.write(receipt_path,receipt)
    return receipt


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,default=C.OUT/'checkpoints/student_inference.pt')
    args=parser.parse_args()
    C.runtime()
    data=C.read(C.OUT/'manifest/data.json')
    rows=data['dev_real']+data['dev_synthetic']
    if len(rows)!=66:raise ValueError('Exactly48 real +18 synthetic student outputs are authorized')
    done=C.OUT/'inference/completion.json'
    if done.exists():
        result=C.read(done)
        if result['status']!='COMPLETE' or len(result['rows'])!=66:raise ValueError('Incomplete student evaluation receipt')
        for row in result['rows']:
            if C.sha(row['student_cache_path'])!=row['student_cache_sha256'] or C.sha(row['input_path'])!=row['input_sha256']:
                raise ValueError('Completed student evaluation identity changed')
        print('SKIP_VERIFIED_COMPLETE student evaluation');return
    checkpoint=args.checkpoint
    chk=C.read(checkpoint.with_suffix('.json'))
    if chk['status']!='PASS' or chk['step']!=4000 or C.sha(checkpoint)!=chk['sha256']:raise ValueError('Final student export receipt mismatch')
    phase=C.Phase('student_evaluation',estimated_mib=1500)
    completed=[]
    try:
        tick=time.perf_counter();student,spec,payload=load_student(checkpoint,require_step=4000);del payload
        student.to('cuda').eval().requires_grad_(False)
        torch.cuda.reset_peak_memory_stats()
        C.write(C.OUT/'inference/load.json',dict(status='PASS',independent_student=True,teacher_loaded=False,
                                                checkpoint_sha256=chk['sha256'],load_seconds=time.perf_counter()-tick,spec=spec))
        for row in rows:
            if row['split']=='dev_real':
                dest=C.OUT/'inference'/C.STUDENT/row['label']/row['request_id']
                filename='denoised.npz';receiptname='denoised.json'
            else:
                dest=C.OUT/'synthetic_results'/C.STUDENT/row['synthetic_id']
                filename='prediction.npz';receiptname='receipt.json'
            path=C.checked(dest/filename);receiptpath=C.checked(dest/receiptname)
            if path.exists() or receiptpath.exists():raise RuntimeError('Existing output without closed stage: no unaccounted replay')
            inputpath=Path(row['input_path'])
            if C.sha(inputpath)!=row['input_sha256']:raise ValueError('Frozen DEV input changed')
            with np.load(inputpath,allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
            arrays,info=single_student_proposal(student,C_original=b['C'],input_model=b.get('input_model'),
                                               input_d=b['input_d'],known_mask=b['known_mask'],phase=phase,
                                               call_kind='student_eval_F',identity=row['sample_id'])
            C.save_npz(path,**arrays)
            identity=dict(sample_id=row['sample_id'],label=row.get('label'),request_id=row.get('request_id'),
                          raw_sha256=row.get('raw_sha256'),model_checkpoint_sha256=chk['sha256'],
                          student_checkpoint_sha256=chk['sha256'],input_sha256=row['input_sha256'])
            receipt=dict(status='PASS',**identity,identity=identity,output_sha256=C.sha(path),**info)
            C.write(receiptpath,receipt)
            entry=dict(row,student_cache_path=str(path),student_cache_sha256=receipt['output_sha256'],
                       student_receipt_path=str(receiptpath),student_receipt_sha256=C.sha(receiptpath))
            completed.append(entry)
        C.write(done,dict(status='COMPLETE',student=C.STUDENT,rows=completed,student_forwards=66,
                          real_outputs=48,synthetic_outputs=18,teacher_instances=0,checkpoint_sha256=chk['sha256'],
                          source_sha256=C.sha(__file__),peak=C.memory(),wall_seconds=time.perf_counter()-tick))
        phase.finish();print('Student evaluation COMPLETE',len(completed))
    except BaseException as exc:
        C.write(C.OUT/'inference/failure.json',dict(error=repr(exc),completed=len(completed),no_automatic_retry=True))
        phase.finish('FAILED',repr(exc));raise


if __name__=='__main__':main()
