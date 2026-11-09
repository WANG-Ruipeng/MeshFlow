"""Finite CPU-only source restoration, geometry, stream and recovery checks."""
import argparse
import copy
import gc
from pathlib import Path
import time
import numpy as np
import torch
from native_t1.context_geometry import face_geometry
from native_t1.losses import prepare_targets,evaluate
from native_t1.training import make_optimizer
from native_t1.artifacts import state_sha256
from experiments.mf_h1_local0.control_mvp48.model_mvp import load_mvp_checkpoint
from experiments.mf_h1_local0.control_mvp48.train_io import capture_rng,restore_rng,tree_hash
from experiments.mf_h1_local0.control_mvp48.train_stream import collate
from .common import atomic_json,read_json,digest
from .prepare import load_task,bits
from .stream import ChairStream
from .training import check_recovery


def recovery_tests():
    good=[dict(event=e,phase_step=s) for s in range(1,4) for e in ('INPUT_BEGIN','INPUT_READY','UPDATE_BEGIN','UPDATE_VALIDATED')]
    check_recovery(good,3,0)
    cases=[good+[dict(event='UPDATE_BEGIN',phase_step=4)],good+good[:1],
           good+[dict(event='INPUT_BEGIN',phase_step=4)],
           good+[dict(event='INPUT_BEGIN',phase_step=4),dict(event='INPUT_READY',phase_step=4)],
           [dict(event='UPDATE_VALIDATED',phase_step=2)]]
    for records in cases:
        try:check_recovery(records,3,0)
        except RuntimeError:pass
        else:raise AssertionError('Invalid/replayed recovery accepted')
    return dict(contiguous_accepted=True,uncheckpointed_attempt_rejected=True,duplicate_rejected=True,gap_rejected=True,unfinished_input_rejected=True,uncheckpointed_ready_input_rejected=True)


def run(source,out):
    source,out=Path(source),Path(out)
    target=out/'registration/training_cpu_tests.json'
    if target.exists():raise RuntimeError('CPU checks already recorded; do not silently repeat')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    if torch.cuda.is_initialized():raise RuntimeError('CPU test started after CUDA initialization')
    tick=time.perf_counter();result=dict(status='RUNNING',model_forwards=0,OT_calls=0,optimizer_updates=0,arms={})
    atomic_json(target,result)
    try:
        result['recovery']=recovery_tests()
        for arm in ('T1','Geo','Full'):
            path=source/'checkpoints'/f'{arm}_step1000.pt'
            model,meta,hashes=load_mvp_checkpoint(path,'cpu')
            model.set_trainable(True).eval();optimizer=make_optimizer(model)
            payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            optimizer.load_state_dict(payload['optimizer'])
            assert tree_hash(optimizer.state_dict())==meta['optimizer_state_sha256']
            assert len(optimizer.state)==len(list(model.parameters()))
            assert all(int(s['step'])==1000 for s in optimizer.state.values())
            assert meta['variant']==arm and meta['logical_step']==1000
            assert state_sha256(model)==meta['model_state_sha256']
            # Restore the CPU RNG domains; CUDA bytes are validated, never activated here.
            cpu_rng=dict(payload['rng'],torch_cuda=[]);restore_rng(cpu_rng)
            assert tree_hash(capture_rng())==tree_hash(cpu_rng)
            assert tree_hash(payload['rng'])==meta['rng_sha256']
            assert payload['rng']['torch_cuda']
            result['arms'][arm]=dict(status='PASS',restore_mode='RESTORE_ALL',**hashes,
                source_checkpoint=str(path),optimizer_state_sha256=meta['optimizer_state_sha256'],
                rng_sha256=meta['rng_sha256'],optimizer_states=len(optimizer.state),optimizer_step=1000,
                phase_step=0,conditional_total_step=1000,strict_model_load=True,
                CPU_RNG_restore_verified=True,CUDA_RNG_bytes_verified=True,
                CUDA_RNG_execution_restore='REQUIRED_IN_LOCKED_WORKER',
                fork_checkpoint_sha256=meta.get('fork_checkpoint_sha256'))
            del model,optimizer,payload;gc.collect();atomic_json(target,result)
        stream=ChairStream(source,out,'T1')
        initial=stream.state_dict();last=dict(initial,batch_index=3000,sample_index=24000)
        stream.load_state_dict(last);assert stream.batch_index==3000
        try:stream.next_effective_batch()
        except RuntimeError:pass
        else:raise AssertionError('Exhausted stream accepted')
        stream.load_state_dict(initial)
        tasks=read_json(out/'task_manifest.json')['tasks']
        selected=[next(t for t in tasks if t['role']=='train' and t['status']=='READY' and t['ratio']==ratio) for ratio in (.2,.4)]
        result['geometry']=[];samples=[]
        for task in selected:
            base=load_task(source,out,task['task_id'],1.)
            base_phi=face_geometry(torch.from_numpy(base['constraints']))[0]
            for alpha in (.9,1.,1.1):
                case=load_task(source,out,task['task_id'],alpha)
                known=case['source_face_ids'];free=case['free_source_ids'];n=case['N'];f=case['source_face_vertex_ids']
                assert sorted(np.r_[known,free].tolist())==list(range(n)) and not set(known)&set(free)
                expected=base['full_target'].reshape(n,3,3).copy();expected[...,0]*=np.float32(alpha)
                assert bits(expected,case['full_target'].reshape(n,3,3))
                assert bits((case['pre_ot_full']*2).reshape(n,9),case['full_target'])
                assert bits(case['model_vertices'][f],case['full_target'].reshape(n,3,3))
                phi,adj,facts=face_geometry(torch.from_numpy(case['constraints']))
                assert torch.isfinite(phi).all()
                if alpha!=1.:assert not torch.equal(phi,base_phi)
                label=dict(free_gt=case['free_target'].reshape(-1,3,3),known=case['constraints'],
                    free_ids=f[free],known_ids=f[known],bbox_L=case['bbox_diagonal'])
                targets=prepare_targets(label)
                expected_edges=sorted(sorted(map(int,e)) for e in case['interface']['edge_source_vertex_ids'])
                assert targets['D_edge']['edge_ids']==expected_edges
                prediction=torch.from_numpy(label['free_gt'])
                losses={name:float(evaluate(name,prediction,targets[name])['loss']) for name in ('D_edge','L5')}
                # Existing FP32 geometry comparison tolerance, test_edge5_losses.py.
                assert all(abs(value)<=2e-6 for value in losses.values())
                result['geometry'].append(dict(task_id=task['task_id'],ratio=task['ratio'],alpha=alpha,N=n,K=case['K'],
                    self_losses=losses,FP32_tolerance=2e-6,interface_edges=len(expected_edges),
                    Geo_recomputed_from_transformed_C=True,Joint_reprepared_from_transformed_GT=True))
                # No OT: zero-noise arithmetic fixture, only partition/collate validation.
                x1=np.zeros((n,9),np.float32);x1[free]=case['full_target'][free]
                context=np.zeros_like(x1);context[known]=case['full_target'][known]
                sample=dict(x0=np.zeros_like(x1),x1=x1,u=x1,xt=x1*.5,context=context,t=np.asarray(.5,np.float32),
                    y=np.asarray(n,np.int64),valid_mask=np.ones(n,bool),known_mask=np.isin(np.arange(n),known),
                    source_face_ids=np.arange(n),corner_permutations=np.tile(np.arange(3),(n,1)),object_id=task['uid'],
                    _geometry=dict(V=case['model_vertices'],F=f,bbox_L=case['bbox_diagonal']))
                stream.label(sample);samples.append(sample)
        padded=collate(samples,pad_to=max(int(s['y']) for s in samples)+5)
        assert torch.equal(padded['valid_mask'].sum(1),padded['y'])
        assert torch.equal(padded['xt'][padded['known_mask']],padded['context'][padded['known_mask']])
        assert not (padded['loss_mask'] & ~padded['valid_mask']).any()
        result['stream']=dict(plan_sha256=stream.plan_sha256,task_manifest_sha256=stream.task_manifest_sha256,
            records=len(stream.records),phase_cursor_restore=True,hard_exhaustion=True,OT_calls=stream.counts,
            K_strata_and_padding=True,N_values_tested=sorted({int(s['y']) for s in samples}),free_scalar_denominator=int(padded['loss_mask'].sum())*9,
            zero_noise_fixture_is_training=False)
        result.update(status='PASS',Geo_feature_calls=8,Joint_target_preparations=6,geometry_loss_calls=12,
            CUDA_initialized=torch.cuda.is_initialized(),elapsed_seconds=time.perf_counter()-tick)
        assert not result['CUDA_initialized']
        atomic_json(out/'registration/source_restore_audit.json',dict(status='PASS',restore_mode='RESTORE_ALL',
            arms=result['arms'],model_forwards=0,optimizer_updates=0,CUDA_initialized=False))
        atomic_json(target,result)
    except BaseException as exc:
        result.update(status='STOPPED',error=repr(exc),elapsed_seconds=time.perf_counter()-tick)
        atomic_json(target,result);raise
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(run(args.source,args.out),flush=True)
