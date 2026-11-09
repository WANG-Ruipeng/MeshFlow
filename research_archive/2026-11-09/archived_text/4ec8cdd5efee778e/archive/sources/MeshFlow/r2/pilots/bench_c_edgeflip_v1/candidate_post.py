"""Registered candidate postprocessing: one official Bench D0 and the old chain.

This module does nothing at import time except configure thread/CUBLAS settings.
GPU work requires an explicit d0_batch call after all 24 registered RAW receipts
exist.  post() runs in a separate process with no torch and reuses the unchanged
BASE_POST -> HINGE_OPP015 -> ALPHA -> HOLE scoring/cascade implementation.

tasks.json is a list of {cell_id, label, q, original_q, raw_path, source_npz,
baseline_origin}. Labels are EXPOSURE (four flipped inputs), PLAIN200 (ten), and
FLIP200 (ten). Optional raw_receipt_path overrides raw.parent/complete.json or
receipt.json. raw.npz is the existing Native raw_arrays schema, including its
C-only anchor arrays; no separate layout file is needed.
"""
from __future__ import annotations

import os
if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (None, ':4096:8'):
    raise RuntimeError('D0 requires CUBLAS_WORKSPACE_CONFIG=:4096:8')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from r2.pilots.seam_abc_v1 import eval_common as C

sys.path.insert(0, str(C.ROOT/'r2/src'))
ROOT = C.ROOT/'experiments/r2_bench_c_edgeflip_v1/candidate_evaluation'
TASKS = ROOT/'tasks.json'
VERSION = 'bench_edgeflip_candidate_single_Bench_D0_original_CPU_chain_v1'
EXPECTED = {'EXPOSURE':4, 'PLAIN200':10, 'FLIP200':10}


def _verify(book):
    for path,digest in book.items():
        if C.sha(path) != digest:
            raise ValueError('Frozen input/result differs: '+str(path))


def _safe_id(value):
    if not isinstance(value,str) or not value or any(ch in value for ch in '/\\:') or value in ('.','..'):
        raise ValueError('Unsafe task/worker identifier')
    return value


def _save_npz(path, values):
    path=C.resolve(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise RuntimeError('Refusing to overwrite output: '+str(path))
    with path.open('xb') as handle:
        np.savez_compressed(handle, **values)
        handle.flush();os.fsync(handle.fileno())


def _raw_receipt(task):
    if task.get('raw_receipt_path'):
        return C.resolve(task['raw_receipt_path'])
    folder=C.resolve(task['raw_path']).parent
    for name in ('complete.json','receipt.json'):
        if (folder/name).exists():
            return folder/name
    raise FileNotFoundError('No completed RAW receipt: '+str(folder))


def _post_dir(task):
    return ROOT/'post'/ _safe_id(task['cell_id'])


def _task_identity(task, tasks_path):
    return dict(version=VERSION,cell_id=task['cell_id'],label=task['label'],
        request_id=task['q']['request_id'],tasks_sha256=C.sha(tasks_path),
        raw_sha256=C.sha(task['raw_path']),source_sha256=C.sha(task['source_npz']),
        input_sha256=C.sha(task['q']['inference_npz']),query_sha256=task['q']['query_descriptor']['sha256'])


def load_tasks(tasks_path=TASKS):
    tasks=C.read(tasks_path)
    if not isinstance(tasks,list) or Counter(t['label'] for t in tasks)!=EXPECTED:
        raise ValueError('Expected exactly EXPOSURE4 + PLAIN20010 + FLIP20010 new post tasks')
    if len({t['cell_id'] for t in tasks})!=24:
        raise ValueError('Task IDs must be unique')
    panels={label:set() for label in EXPECTED}
    for task in tasks:
        for key in ('cell_id','label','q','original_q','raw_path','source_npz','baseline_origin'):
            if key not in task:
                raise ValueError('Task missing '+key)
        _safe_id(task['cell_id'])
        q,old=task['q'],task['original_q']
        for key in ('request_id','parent_id','condition_id','noise_index','N','K','split'):
            if q[key]!=old[key]:
                raise ValueError('Registered original/flip request differs in '+key)
        flipped=C.sha(q['inference_npz'])!=C.sha(old['inference_npz'])
        panel=(q['request_id'],flipped)
        if panel in panels[task['label']]:
            raise ValueError('Duplicate task panel identity')
        panels[task['label']].add(panel)
    if panels['PLAIN200']!=panels['FLIP200']:
        raise ValueError('Candidate model panels differ')
    if sum(flip for _,flip in panels['PLAIN200'])!=4:
        raise ValueError('Each candidate needs six original and four legal flipped requests')
    if panels['EXPOSURE']!={p for p in panels['PLAIN200'] if p[1]}:
        raise ValueError('EXPOSURE tasks must be exactly the four already generated flips')
    return tasks


def verify_raw_barrier(tasks_path=TASKS):
    """Authenticate all 24 saved RAWs and Native C-only anchors; no models."""
    tasks=load_tasks(tasks_path);protected={str(C.resolve(tasks_path)):C.sha(tasks_path)}
    from meshflow_control.source_winding import validate_source
    required=('raw','output','known_faces','input_faces','C_vertices','C_faces_original','C_faces_model',
        'raw_anchor_ids','raw_seam_slot_mask','raw_slot_edge_id','occupied','occupancy_logits','edge_records')
    for task in tasks:
        q=task['q'];raw_path=C.resolve(task['raw_path']);receipt_path=_raw_receipt(task)
        receipt=C.read(receipt_path)
        if receipt.get('status') not in ('COMPLETE','PASS') or receipt.get('generator_forwards')!=50:
            raise ValueError('RAW is not a sealed Euler50 result: '+task['cell_id'])
        if receipt.get('D0_forwards',0)!=0:
            raise ValueError('Task RAW already declares denoising')
        _verify(receipt['files'])
        if not any(C.resolve(p)==raw_path and h==C.sha(raw_path) for p,h in receipt['files'].items()):
            raise ValueError('RAW file is not sealed by its producer receipt')
        raw=C.arrays(raw_path);inp=C.arrays(q['inference_npz']);source=C.arrays(task['source_npz'])
        if any(key not in raw for key in required):
            raise ValueError('RAW lacks the existing anchor/geometry schema')
        n,k=int(q['N']),int(q['K'])
        if raw['raw'].shape!=(n,3,3) or raw['raw'].dtype!=np.float32 or not np.isfinite(raw['raw']).all():
            raise ValueError('RAW shape/dtype/nonfinite failure')
        if int(raw['known_faces'])!=k or int(raw['input_faces'])!=n:
            raise ValueError('RAW N/K differ')
        for key in ('C_vertices','C_faces_original','C_faces_model'):
            if not np.array_equal(raw[key],inp[key]):
                raise ValueError('RAW C metadata differs in '+key)
        if not np.array_equal(raw['raw'][:k],inp['C_vertices'][inp['C_faces_model']]):
            raise ValueError('Canonical RAW C differs')
        if not np.array_equal(raw['output'][:k],inp['C_vertices'][inp['C_faces_original']]):
            raise ValueError('Source-restored RAW C differs')
        anchor=raw['raw_anchor_ids']
        if anchor.shape!=(n,3) or not np.array_equal(anchor[:k],inp['C_faces_model']) or np.any(anchor[k:]>=0):
            raise ValueError('Not the frozen Native C-only anchor layout')
        if np.any(raw['raw_seam_slot_mask']) or np.any(raw['occupied']):
            raise ValueError('Native task unexpectedly has seam slots')
        validate_source(source,raw['raw'][:k])
        for path in (raw_path,receipt_path,C.resolve(task['source_npz']),C.resolve(q['inference_npz']),C.resolve(q['query_descriptor']['path'])):
            protected[str(path)]=C.sha(path)
        if C.sha(q['query_descriptor']['path'])!=q['query_descriptor']['sha256']:
            raise ValueError('Frozen query bank differs')
    return tasks,dict(status='PASS',outputs=24,labels=EXPECTED,new_noise=0,new_queries=0,files=protected)


def _checked_post(task,tasks_path):
    folder=_post_dir(task);saved=C.read(folder/'receipt.json')
    if saved['identity']!=_task_identity(task,tasks_path) or saved['D0_forwards']!=1 or saved['GEO_calls']!=1:
        raise ValueError('Completed D0/GEO identity differs')
    _verify(saved['files'])
    return saved


def d0_batch(tasks_path=TASKS, indices=None, worker='d0_0', estimated_peak_mib=3500):
    """One independently owned official D0 model; exactly one call per task.

    The controller chooses indices/concurrency from measured resources.  There
    are no implicit retries: any partial per-task directory blocks this call.
    Completed tasks are authenticated and skipped without loading a model.
    """
    tasks,barrier=verify_raw_barrier(tasks_path)
    chosen=tasks if indices is None else [tasks[i] for i in indices]
    if not chosen or len({t['cell_id'] for t in chosen})!=len(chosen):
        raise ValueError('Select unique, nonempty registered tasks')
    pending=[]
    for task in chosen:
        folder=_post_dir(task)
        if (folder/'receipt.json').exists():
            _checked_post(task,tasks_path)
        elif folder.exists():
            raise RuntimeError('Partial D0/GEO attempt requires explicit engineering recovery: '+str(folder))
        else:
            pending.append(task)
    if not pending:
        return dict(status='COMPLETE_VERIFIED_NO_MODEL_LOAD',outputs=len(chosen),new_D0_forwards=0,new_GEO_calls=0)
    from r2.pilots.bench_four_routes_v1.resources import snapshot
    from r2.pilots.seam_slot_c200_v1.generate import dispatch_guard
    before=snapshot();dispatch_guard(before,estimated_peak_mib)
    stage=ROOT/'d0_workers'/_safe_id(worker)
    if stage.exists():
        raise RuntimeError('Worker attempt already exists; no implicit replay')
    stage.mkdir(parents=True)
    C.write(stage/'resources_before.json',before)
    C.write(stage/'all_RAW_barrier.json',barrier)
    C.write(stage/'dispatch.json',dict(workers_in_this_process=1,controller_owns_global_concurrency=True,
        estimated_process_peak_mib=estimated_peak_mib,headroom='max(2GiB,10%total) plus 1.2 times comparable measured peak',
        tasks=[t['cell_id'] for t in pending],threads_per_process=1))
    model=None;observer=None;calls=0;completed=[];started=time.perf_counter()
    try:
        from r2.pilots.bench_control_transfer_v1 import postprocess as OLD
        from r2.pilots.seam_slot_c200_v1.postprocess import geometry_from_proposal
        from r2.pilots.user_condition_finetune_v1.trainer import GPUObserver
        from meshflow_control.geo_io import write_indexed_ply
        from meshflow_control.denoise_model import predict_proposal, _assert_runtime
        import torch
        observer=GPUObserver('registered_Bench_D0_model_load_and_real_D0_GEO_tasks',interval=1.)
        observer.__enter__()
        model,model_identity=OLD.load_bench_denoiser()
        C.write(stage/'model_identity.json',model_identity)
        C.write(stage/'resources_loaded.json',snapshot())
        code={str(Path(p)):C.sha(p) for p in (__file__,OLD.__file__,sys.modules[geometry_from_proposal.__module__].__file__)}
        C.write(stage/'code_hashes.json',code)
        for task in pending:
            folder=_post_dir(task);folder.mkdir(parents=True)
            identity=_task_identity(task,tasks_path)
            C.write(folder/'attempt.json',dict(identity=identity,D0_budget=1,GEO_budget=1,worker=worker,
                official_Bench_D0_sha256=OLD.D0_SHA,no_automatic_retry=True))
            raw=C.arrays(task['raw_path']);source=C.arrays(task['source_npz'])
            C.write(folder/'resources_before_D0.json',snapshot())
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            OLD.event(folder/'calls.jsonl','D0_FORWARD_STARTED',budget=1,cell_id=task['cell_id'])
            proposal=predict_proposal(model,raw['raw'])
            torch.cuda.synchronize();calls+=1
            OLD.event(folder/'calls.jsonl','D0_FORWARD_RETURNED',seconds=time.perf_counter()-tick)
            _save_npz(folder/'proposal.npz',dict(official_unconstrained_proposal=proposal))
            C.write(folder/'proposal_receipt.json',dict(status='COMPLETE',D0_forwards=1,
                identity=identity,official_checkpoint_sha256=OLD.D0_SHA,seconds=time.perf_counter()-tick,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_scope='One real D0 forward after counter reset; resident model included; allocator is not full process VRAM',
                files={str(folder/'proposal.npz'):C.sha(folder/'proposal.npz')}))
            C.write(folder/'geometry_attempt.json',dict(calls=1,source_orientation_calls=1,utc=C.utc()))
            geo_tick=time.perf_counter()
            pre,canonical,post,geometry,oriented,anchors=geometry_from_proposal(raw,proposal,source)
            _save_npz(folder/'pre_geo.npz',dict(output=pre,C=raw['raw'][:int(raw['known_faces'])],
                known_faces=raw['known_faces'],raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask']))
            _save_npz(folder/'post_canonical.npz',canonical);_save_npz(folder/'post.npz',post)
            write_indexed_ply(folder/'post.ply',post['vertices'],post['faces'])
            C.write(folder/'geometry.json',dict(stats=geometry['stats'],face_records=geometry['face_records']))
            C.write(folder/'anchor_identity.json',anchors)
            C.write(folder/'source_restoration.json',oriented['restoration_summary'])
            orient=oriented['orientation_result']
            C.write(folder/'orientation.json',dict(summary=orient['summary'],components=orient['component_rows'],edges=orient['edge_rows']))
            C.write(folder/'resources_finished.json',snapshot())
            result=dict(status='COMPLETE',version=VERSION,identity=identity,D0_forwards=1,GEO_calls=1,
                source_orientation_calls=1,official_D0_sha256=OLD.D0_SHA,geometry_seconds=time.perf_counter()-geo_tick,
                new_noise=0,new_queries=0,anchors=anchors,source_labels_read=False,
                files={str(p):C.sha(p) for p in folder.iterdir() if p.is_file()})
            C.write(folder/'receipt.json',result);completed.append(task['cell_id'])
            print(json.dumps(dict(stage='D0_GEO',cell_id=task['cell_id'],completed=len(completed),tasks=len(pending))),flush=True)
        _verify(code);_verify(barrier['files']);runtime=_assert_runtime()
        result=dict(status='COMPLETE',worker=worker,new_D0_forwards=calls,new_GEO_calls=len(completed),
            completed=completed,seconds=time.perf_counter()-started,runtime_after=runtime,
            files={str(_post_dir(t)/'receipt.json'):C.sha(_post_dir(t)/'receipt.json') for t in chosen})
        C.write(stage/'complete.json',result)
        return result
    except BaseException as exc:
        C.write(stage/'failure.json',dict(status='STOPPED',error=repr(exc),traceback=traceback.format_exc(),
            returned_D0_calls=calls,completed=completed,no_automatic_retry=True))
        raise
    finally:
        if observer is not None:
            observer.__exit__(None,None,None)
            C.write(stage/'resource_observation.json',dict(summary=observer.summary(),samples=observer.samples,
                scope='Resident model load plus registered real D0/GEO work; no probe tasks'))
        C.write(stage/'resources_finished.json',snapshot())
        if model is not None:
            del model
            import torch
            torch.cuda.empty_cache()


def post(task, tasks_path=TASKS):
    """Pure CPU single-task cascade, with original BASE/FINAL 177-field scores."""
    if 'torch' in sys.modules:
        raise RuntimeError('Run CPU cascade in a fresh process without torch')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    tasks,barrier=verify_raw_barrier(tasks_path)
    registered={t['cell_id']:t for t in tasks}
    if task!=registered.get(task['cell_id']):
        raise ValueError('CPU task differs from frozen tasks.json')
    done=_checked_post(task,tasks_path)
    from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
    folder=_post_dir(task);cache=ROOT/'base_pair_caches'/task['cell_id'];completion=cache/'complete.json'
    mesh_sha=C.sha(folder/'post.npz')
    if completion.exists():
        saved=C.read(completion)
        if saved['mesh_sha256']!=mesh_sha:
            raise ValueError('BASE pair cache mesh differs')
        _verify(saved['files'])
    else:
        if cache.exists():
            raise RuntimeError('Partial BASE scan requires explicit engineering recovery')
        C.write(cache/'started.json',dict(mesh_sha256=mesh_sha,model_calls=0))
        pairs=CPU.H.scan(C.arrays(folder/'post.npz')['output'],int(task['q']['K']))
        C.write(cache/'pairs.json',pairs)
        C.write(completion,dict(status='COMPLETE',mesh_sha256=mesh_sha,full_pair_scans=1,
            files={str(p):C.sha(p) for p in cache.iterdir() if p.is_file()}))
    return CPU.process_one(task['q'],task['label'],folder/'post.npz',folder/'pre_geo.npz',task['raw_path'],
        cache/'pairs.json',ROOT/'cpu'/task['cell_id'],seam_model=False,registration_sha=C.sha(tasks_path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify','d0','post'))
    parser.add_argument('--tasks',default=str(TASKS))
    parser.add_argument('--indices',nargs='+',type=int)
    parser.add_argument('--worker',default='d0_0')
    parser.add_argument('--estimated-peak-mib',type=float,default=3500)
    args=parser.parse_args()
    if args.action=='verify':
        _,result=verify_raw_barrier(args.tasks)
    elif args.action=='d0':
        result=d0_batch(args.tasks,args.indices,args.worker,args.estimated_peak_mib)
    else:
        if args.indices is None or len(args.indices)!=1:
            raise ValueError('CPU post CLI takes one registered task index per process')
        result=post(load_tasks(args.tasks)[args.indices[0]],args.tasks)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','identity')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
