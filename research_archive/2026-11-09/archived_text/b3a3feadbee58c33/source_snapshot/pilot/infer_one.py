"""Independent one-shot student entrypoint; no teacher weights are loaded.

No arguments shows help. Supplying inputs without --execute is a CPU-only dry
validation. This pilot does not invoke --execute: its 66 formal evaluations
are owned by inference.py. A future explicit standalone invocation records
its single forward separately under --out and refuses implicit replay.

Verified decoder source: MeshFlow db86bc32423c316c00bcc2f645a26243e8010f20.
The original source snapshot and its existing third-party license/provenance
remain dependencies; no teacher checkpoint/state is a runtime dependency.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='4'
import argparse,json,time,sys
from pathlib import Path
import numpy as np
from . import common as C


def load_raw(path,key,mask_key,known_faces):
    path=Path(path)
    before=C.sha(path)
    with np.load(path,allow_pickle=False) as z:
        if key not in z:raise ValueError('Explicit RAW key is absent: '+key)
        raw=z[key].copy()
        if raw.dtype!=np.float32 or raw.ndim not in (2,3):raise ValueError('RAW must be existing FP32 triangles')
        if raw.ndim==2 and raw.shape[1:]!=(9,):raise ValueError('Expected [N,9]')
        if raw.ndim==3 and raw.shape[1:]!=(3,3):raise ValueError('Expected [N,3,3]')
        raw=raw.reshape(-1,3,3)
        n=len(raw)
        if not 1<n<=800 or not np.isfinite(raw).all():raise ValueError('Invalid N or nonfinite RAW')
        if 'N' in z and int(z['N'])!=n:raise ValueError('N differs from full RAW face count')
        if known_faces is not None:
            known=np.arange(n)<known_faces
            if mask_key in z and not np.array_equal(known,z[mask_key].reshape(-1)):raise ValueError('Explicit K and mask conflict')
        else:
            if mask_key not in z:raise ValueError('Need bool C mask in NPZ or explicit --known-faces')
            known=z[mask_key].copy().reshape(-1)
        if known.dtype!=np.bool_ or known.shape!=(n,) or not 0<int(known.sum())<n:raise ValueError('Invalid known mask')
        c=raw[known].copy()
        if 'C' in z and (z['C'].dtype!=np.float32 or z['C'].reshape(-1,3,3).tobytes()!=c.tobytes()):
            raise ValueError('Saved C differs from original RAW at known mask')
    if C.sha(path)!=before:raise ValueError('RAW changed while reading')
    order=np.concatenate([np.flatnonzero(known),np.flatnonzero(~known)]).astype(np.int64)
    inverse=np.argsort(order)
    if not np.array_equal(raw,raw[order][inverse]):raise AssertionError('Stable C-prefix inverse failed')
    return raw,c,known,order,inverse,before


def run(args):
    raw,c,known,order,inverse,raw_sha=load_raw(args.input,args.input_key,args.mask_key,args.known_faces)
    dest=C.checked(args.out)
    for filename in ('pregeo.npz','pregeo_Cprefix.npz','receipt.json','execution.json','post','post_original_index_maps.npz'):
        C.checked(dest/filename)
    if dest.exists() and any(dest.iterdir()):raise FileExistsError('Output directory is not empty; no replay or overwrite')
    checkpoint=Path(args.checkpoint)
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    receipt=C.read(checkpoint.with_suffix('.json'))
    if receipt['status']!='PASS' or receipt['step']!=4000 or C.sha(checkpoint)!=receipt['sha256']:raise ValueError('Final student export identity mismatch')
    plan=dict(status='DRY_RUN',N=len(raw),K=len(c),input=str(Path(args.input).resolve()),input_sha256=raw_sha,
              checkpoint=str(checkpoint.resolve()),checkpoint_sha256=receipt['sha256'],output=str(dest),
              new_model_calls=0,planned_model_calls=1,teacher_weights_required=False,optional_fixed_GEO=bool(args.geo),
              ordering='Stable C-prefix for one student call; preGEO arrays restored to original face order. POST keeps C-prefix retained order with explicit original source-face/corner maps.',
              note='No model instantiated in dry run. --execute is a new explicit standalone inference, outside this completed pilot evaluation.')
    if not args.execute:
        print(json.dumps(plan,indent=2));return plan
    # Lazy imports ensure help and dry validation never construct a model.
    import gc,torch
    from .model import load_student
    from .inference import single_student_proposal
    C.runtime();C.write(dest/'execution.json',dict(plan,status='RUNNING',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    calls=[0]
    class SingleCall:
        def call(self,kind,fn,identity=None):
            if calls[0]!=0:raise RuntimeError('Exactly one standalone student forward is allowed')
            calls[0]+=1
            C.write(dest/'model_call.json',dict(status='STARTED',kind='STANDALONE_USER_INFERENCE',call=1,
                                               part_of_pilot_formal_budget=False,checkpoint_sha256=receipt['sha256']))
            value=fn()
            C.write(dest/'model_call.json',dict(status='COMPLETE',kind='STANDALONE_USER_INFERENCE',call=1,
                                               part_of_pilot_formal_budget=False,checkpoint_sha256=receipt['sha256']))
            return value
    try:
        resources=C.resources();C.write(dest/'resources_before.json',resources)
        line=resources['gpu']['stdout'].strip().splitlines()[0].split(',')
        total,free=float(line[1]),float(line[3])
        if free-max(2048.,.1*total)<1500:raise RuntimeError('Insufficient measured GPU headroom')
        student,spec,payload=load_student(checkpoint,require_step=4000);del payload
        student.to('cuda').eval().requires_grad_(False)
        torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        arrays,info=single_student_proposal(student,C_original=c,input_model=raw[order],
                                            phase=SingleCall(),identity='explicit_user_single_step')
        pregeo={name:(value[inverse].copy() if isinstance(value,np.ndarray) and value.shape[:1]==(len(raw),) else value)
                for name,value in arrays.items()}
        pregeo.update(known_mask=known.copy(),prefix_to_original_face_ids=order,original_to_prefix_face_ids=inverse)
        if pregeo['output'][known].tobytes()!=c.tobytes():raise AssertionError('Original C bytes were not restored')
        C.save_npz(dest/'pregeo.npz',**pregeo)
        C.save_npz(dest/'pregeo_Cprefix.npz',**arrays,prefix_to_original_face_ids=order)
        peak=C.memory();del student;gc.collect();torch.cuda.empty_cache()
        post_info=None
        if args.geo:
            # NumPy-only maintained GEO module; do not import evaluation.py,
            # which deliberately changes CUDA visibility for CPU workers.
            from native_t1 import postprocess_cli as PC
            from native_t1.postprocessing import FIXED_CONFIG
            expected=C.read(C.OLD/'registration.json')['postprocess_config']
            if dict(FIXED_CONFIG)!=expected:raise ValueError('Fixed GEO_C contract changed')
            postdir=C.checked(dest/'post')
            post_info=PC.run(dest/'pregeo_Cprefix.npz',len(c),postdir,input_key='output')
            with np.load(postdir/'post.npz',allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
            source_original=order[p['source_face_ids']]
            target=p['corner_target_input_ids']
            target_original=3*order[target//3]+target%3
            vertex_source=p['vertex_source_input_corner_ids']
            vertex_source_original=3*order[vertex_source//3]+vertex_source%3
            C.save_npz(dest/'post_original_index_maps.npz',
                       POST_face_to_original_face_id=source_original,
                       POST_vertex_to_original_input_corner_id=vertex_source_original,
                       original_input_corner_to_target_original_corner_id=target_original[inverse],
                       original_input_corner_to_POST_vertex_id=p['corner_vertex_ids'][inverse],
                       POST_known_mask=known[source_original],prefix_to_original_face_ids=order)
            if p['output'][:len(c)].tobytes()!=c.tobytes():raise AssertionError('GEO changed C')
        if C.sha(args.input)!=raw_sha or C.sha(checkpoint)!=receipt['sha256']:raise ValueError('Protected inference input changed')
        result=dict(plan,status='COMPLETE',new_model_calls=calls[0],C_original_bits_restored=True,info=info,
                    student_spec=spec,peak=peak,wall_seconds=time.perf_counter()-tick,POST=post_info,
                    files={str(p.relative_to(dest)):C.sha(p) for p in dest.rglob('*') if p.is_file() and p.name not in ('execution.json','receipt.json')},
                    source_sha256=C.sha(__file__))
        C.write(dest/'receipt.json',result)
        C.write(dest/'execution.json',dict(status='COMPLETE',model_calls=1,receipt_sha256=C.sha(dest/'receipt.json')))
        print(json.dumps(dict(status='COMPLETE',out=str(dest),model_calls=1,GEO=bool(args.geo))))
        return result
    except BaseException as exc:
        C.write(dest/'execution.json',dict(status='FAILED',model_call_attempts=calls[0],error=repr(exc),no_automatic_replay=True))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input',type=Path,help='Existing FP32 RAW NPZ; source is read-only')
    parser.add_argument('--input-key',default='raw')
    parser.add_argument('--mask-key',default='known_mask',help='NPZ bool face mask, any face order')
    parser.add_argument('--known-faces',type=int,help='Alternative explicit first-K mask')
    parser.add_argument('--checkpoint',type=Path,default=C.OUT/'checkpoints/student_inference.pt')
    parser.add_argument('--out',type=Path,help='New empty output directory inside this pilot OUT')
    parser.add_argument('--geo',action='store_true',help='One unchanged fixed GEO_C pass after restoring C')
    parser.add_argument('--execute',action='store_true',help='Actually perform one new standalone GPU forward; otherwise dry validation only')
    args=parser.parse_args(argv)
    if args.input is None or args.out is None:
        parser.print_help()
        if args.execute:parser.error('--execute requires --input and --out')
        return
    run(args)


if __name__=='__main__':main()
