"""Finite CPU implementation checks; zero real models, CUDA or updates.

Results and actual CPU OT / geometry calls are recorded under cpu_tests only.
Synthetic arrays are unit tests; the real frozen task fixtures are separately
identified and are never presented as formal training or evaluation results.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .coupling import (array_hash, file_hash, nested_ot_map, to_target_slots,
    to_noise_slots, coupling_path, old_slot_sample, map_velocity_old_to_common)
from .stream import HybridStream, prepare_training_plan, read_json, make_plan
from experiments.mf_h1_local0.chair_domain_control_v1.stream import ChairStream
from experiments.mf_h1_local0.chair_domain_control_v1.prepare import load_task_sample
from experiments.mf_h1_local0.control_mvp48.train_stream import collate
from native_t1.context_geometry import face_geometry
from native_t1.metric_geometry import surface_distance_squared
from utils.ot_utils import optimal_sum_numpy


def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def run(source_mvp,source_chair,out):
    source_mvp,source_chair,out=map(lambda p:Path(p).resolve(),(source_mvp,source_chair,out))
    testout=out/'cpu_tests';testout.mkdir(parents=True,exist_ok=True)
    summary=testout/'summary.json'
    if summary.exists():
        old=read_json(summary)
        if old['status']=='PASS':
            for p,h in old['local_source_sha256'].items():
                if file_hash(p)!=h:raise RuntimeError('CPU tested source changed; preserve old evidence and rerun explicitly')
            return old
        raise RuntimeError('Prior CPU checks failed; retain evidence, investigate before another run')
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only checks found initialized CUDA')
    started=time.perf_counter()
    result=dict(status='RUNNING',model_forwards=0,model_backwards=0,optimizer_updates=0,
        synthetic_scalar_autograd_calls=0,CPU_OT_attempts=0,CPU_OT_returns=0,
        CPU_OT_cache_hits=0,point_triangle_distance_calls=0,checks={},
        local_source_sha256={str(Path(__file__).with_name(p)):file_hash(Path(__file__).with_name(p))
                             for p in ('coupling.py','stream.py','cpu_tests.py')})
    save(testout/'attempt.json',result)
    def check(name,**details):
        result['checks'][name]=dict(status='PASS',**details)
        save(testout/'progress.json',result)
    try:
        fixture=testout/'fixture_stream'
        identity=prepare_training_plan(source_mvp,source_chair,fixture)
        manifest=read_json(source_chair/'task_manifest.json')
        plan=read_json(fixture/'train_plan.json')
        assert len(plan['records'])==8000 and len(plan['object_counts'])==32 and set(plan['object_counts'].values())=={250}
        assert all(sum(r['alpha']==1. for r in plan['records'][i:i+8])==4 for i in range(0,8000,8))
        assert {r['task_id'] for r in plan['records']}<=set(t['task_id'] for t in manifest['tasks'] if t['role']=='train' and t['status']=='READY')
        check('frozen_plan',**identity,parents=32,exposure_per_parent=250,all_1000_batches_four_unit_alpha=True)

        base=HybridStream(source_mvp,source_chair,fixture,'OT_BASE')
        hybrid=HybridStream(source_mvp,source_chair,fixture,'OT_HYBRID')
        legacy=ChairStream(source_mvp,source_chair,'Geo')
        samples=base.next_effective_batch();mixed=hybrid.next_effective_batch()
        result['CPU_OT_attempts']+=base.counts['actual_OT_attempts']+hybrid.counts['actual_OT_attempts']
        result['CPU_OT_returns']+=base.counts['actual_OT_returns']+hybrid.counts['actual_OT_returns']
        result['CPU_OT_cache_hits']+=base.counts['OT_cache_hits']+hybrid.counts['OT_cache_hits']
        reference_hashes=[]
        for sample,coupled,record in zip(samples,mixed,plan['records'][:8]):
            old=legacy._sample(record)
            result['CPU_OT_attempts']+=1;result['CPU_OT_returns']+=1
            remapped=old_slot_sample(sample)
            for key in ('x0','x1','u','xt','context','t','y','known_mask','valid_mask','source_face_ids','corner_permutations','face_permutation'):
                assert bits(remapped[key],old[key]),key
            assert bits(sample['free_noise_after_OT'],old['free_noise_after_OT'])
            assert bits(map_velocity_old_to_common(sample,remapped['u']),sample['u'])
            assert bits(to_target_slots(to_noise_slots(old['free_target_before_OT'],sample['ot_face_map'],sample['ot_corner_map']),sample['ot_face_map'],sample['ot_corner_map']),old['free_target_before_OT'])
            assert sample['shared_input_sha256']==coupled['shared_input_sha256']
            assert sample['coupled_input_sha256']!=coupled['coupled_input_sha256']
            assert bits(sample['x1'],coupled['x1']) and bits(sample['context'],coupled['context'])
            assert bits(sample['xt'][sample['known_mask']],sample['context'][sample['known_mask']])
            assert bits(coupled['xt'][coupled['known_mask']],sample['context'][sample['known_mask']])
            base.label(sample);hybrid.label(coupled)
            reference_hashes.append(dict(task_id=sample['task_id'],uid=sample['object_id'],N=int(sample['y']),K=sample['K'],
                alpha=sample['alpha'],shared_input_sha256=sample['shared_input_sha256'],
                base_coupled_sha256=sample['coupled_input_sha256'],hybrid_coupled_sha256=coupled['coupled_input_sha256'],
                ot_cache_key=sample['ot_cache_key']))
        save(testout/'paired_real_fixture_hashes.json',reference_hashes)
        check('face_corner_inverse_and_legacy_lambda0',real_fixtures=8,all_original_path_label_metadata_bits_equal=True,
            target_triangle_multiset_preserved_by_bijection=True,shared_tensor_hashes_equal=True,
            original_OT_direction='noise reordered into target slots, all six corner permutations',
            new_direction='inverse map puts targets into original iid noise slots')

        state=base.state_dict();clone=HybridStream(source_mvp,source_chair,fixture,'OT_BASE');clone.load_state_dict(state)
        assert clone.state_dict()==state
        exhausted=dict(state,batch_index=1000,sample_index=8000);clone.load_state_dict(exhausted)
        try:clone.next_effective_batch()
        except RuntimeError:pass
        else:raise AssertionError('Budget exhaustion did not stop stream')
        check('stream_cursor_restore_and_budget',restored_batch_index=state['batch_index'],hard_stop_at_updates=1000)

        s=samples[0];face,corner=s['ot_face_map'],s['ot_corner_map']
        z=s['free_gaussian_before_OT'];y=to_noise_slots((s['free_target_before_OT']*2).astype(np.float32),face,corner)
        eps=np.random.RandomState(700041).randn(*z.shape)
        original_mapping=(face.copy(),corner.copy())
        for lam in (0.,.25,1.):
            for t in (0.,.5,1.):
                zm,xt,u=coupling_path(z,y,eps,t,lam)
                assert bits(u,y-zm)
                if t==0:assert bits(xt,zm)
                if t==1:assert bits(xt,y)
                if lam==0:assert bits(zm,z.astype(np.float32))
                if lam==1:assert bits(zm,eps.astype(np.float32))
                assert bits(face,original_mapping[0]) and bits(corner,original_mapping[1])
        coupling_path(z,y,object(),.5,0.)  # Lambda0 cannot inspect epsilon.
        check('coupling_endpoints',lambda_test_only=[0,.25,1],times=[0,.5,1],lambda1_does_not_repair_OT=True,
              interpolation_and_velocity_change_together=True,lambda0_does_not_access_epsilon=True)

        batch=collate(samples,pad_to=max(int(s['y']) for s in samples)+11)
        count=sum((int(s['y'])-s['K'])*9 for s in samples)
        assert int(batch['loss_mask'].sum())*9==count
        assert not (batch['loss_mask'] & ~batch['valid_mask']).any()
        assert not (batch['loss_mask'] & batch['known_mask']).any()
        for i,s in enumerate(samples):
            assert int(batch['valid_mask'][i].sum())==int(s['y'])
            assert torch.equal(batch['xt'][i,batch['known_mask'][i]],batch['context'][i,batch['known_mask'][i]])
        check('known_free_partition_padding',effective_free_coordinates=count,all_C_source_faces_preserved=True,
              added_padding_faces_per_sample_at_least=11,padding_in_OT=False,padding_in_loss=False)

        # Tiny scalar autograd fixture: exactly the production sum-of-squares
        # normalization with deliberately unequal microbatch lengths.
        w=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
        term_values=[]
        lengths=[int((~s['known_mask']).sum())*9 for s in samples]
        for i,length in enumerate(lengths):
            a=torch.linspace(-1,1,length,dtype=torch.float64)
            b=torch.full((length,),float(i)/8,dtype=torch.float64)
            term=((w*a-b).square().sum()/sum(lengths));term_values.append(float(term.detach()))
            term.backward();result['synthetic_scalar_autograd_calls']+=1
        accumulated_grad=float(w.grad)
        v=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
        a=torch.cat([torch.linspace(-1,1,length,dtype=torch.float64) for length in lengths])
        b=torch.cat([torch.full((length,),float(i)/8,dtype=torch.float64) for i,length in enumerate(lengths)])
        whole=(v*a-b).square().sum()/sum(lengths);whole.backward();result['synthetic_scalar_autograd_calls']+=1
        assert abs(sum(term_values)-float(whole.detach()))<1e-14
        assert abs(accumulated_grad-float(v.grad))<1e-14
        wrong=np.mean([term_values[i]*sum(lengths)/lengths[i] for i in range(8)])
        assert abs(wrong-float(whole.detach()))>1e-6
        check('effective_batch_normalization',denominator=sum(lengths),micro_lengths=lengths,
            accumulated_loss=sum(term_values),whole_loss=float(whole.detach()),
            gradient_abs_difference=abs(accumulated_grad-float(v.grad)),naive_micro_MSE_mean=wrong,
            no_model_or_optimizer_used=True)

        count=200000
        zs=np.random.RandomState(87101).randn(count,3).astype(np.float32)
        es=np.random.RandomState(87102).randn(count,3).astype(np.float32)
        mix=np.float32(np.sqrt(.75))*zs+np.float32(.5)*es
        means=mix.mean(0,dtype=np.float64);variance=mix.var(0,dtype=np.float64)
        corr=np.corrcoef(mix.astype(np.float64),rowvar=False)
        independent_corr=float(np.corrcoef(zs.ravel(),es.ravel())[0,1])
        corr_z=float(np.corrcoef(mix.ravel(),zs.ravel())[0,1])
        assert np.max(abs(means))<.02 and np.max(abs(variance-1))<.02
        assert np.max(abs(corr-np.eye(3)))<.02 and abs(independent_corr)<.02
        assert abs(corr_z-np.sqrt(.75))<.01
        check('Gaussian_marginal_sanity',fixed_samples=count,dimensions=3,means=means.tolist(),variances=variance.tolist(),
            cross_coordinate_correlation=corr.tolist(),epsilon_Z_correlation=independent_corr,mixed_Z_correlation=corr_z,
            analytic_basis='Independent iid N(0,I) Z and epsilon give sqrt(1-lambda)Z+sqrt(lambda)epsilon ~ N(0,I); coefficients squared sum to one.',
            scope='Common raw iid noise slots only; finite FP32 sampling sanity check is not a mathematical proof.')

        first_task=plan['records'][0]['task_id']
        original=load_task_sample(source_mvp,manifest,first_task,1.)
        geo=[]
        for alpha in (.9,1.,1.1):
            case=load_task_sample(source_mvp,manifest,first_task,alpha)
            expected=original['full_target'].copy().reshape(-1,3);expected[:,0]*=np.float32(alpha)
            assert bits(expected.reshape(-1,9),case['full_target'])
            assert bits(case['constraints'],case['full_target'][case['source_face_ids']].reshape(-1,3,3))
            assert bits(case['free_target'],case['full_target'][case['free_source_ids']])
            phi,_,_=face_geometry(torch.from_numpy(case['constraints']))
            geo.append(phi.numpy())
        assert not bits(geo[0],geo[1]) and not bits(geo[1],geo[2])
        check('affine_full_target_before_partition_and_Geo',alphas=[.9,1.,1.1],task_id=first_task,
              no_renormalization=True,geometry_recomputed_directly_from_corresponding_C=True,
              geometry_feature_hashes=[array_hash(x) for x in geo],encoder_model_forwards=0)

        panel=read_json(source_chair/'panels.json');matrix_rows=[]
        chosen=[sorted(manifest['B_train_parents'])[0],sorted(manifest['B_validation_parents'])[0]]
        for uid in chosen:
            r=next(r for r in panel['B'] if r['uid']==uid and r['noise_index']==0 and r['alpha']==1.)
            case0=load_task_sample(source_mvp,manifest,r['task_id'],1.)
            with np.load(source_chair/r['control_points_path'],allow_pickle=False) as bank:points0=bank['points0'].copy()
            targets=[load_task_sample(source_mvp,manifest,r['task_id'],a)['free_target'].reshape(-1,3,3) for a in (.9,1.,1.1)]
            for candidate in ('perfect_response','no_response'):
                E=np.empty((3,3),np.float64)
                for i,alpha in enumerate((.9,1.,1.1)):
                    generated=targets[i] if candidate=='perfect_response' else targets[1]
                    for j,beta in enumerate((.9,1.,1.1)):
                        q=points0.copy();q[:,0]*=np.float32(beta)
                        sq=surface_distance_squared(q,generated)
                        result['point_triangle_distance_calls']+=1
                        E[i,j]=float(np.sqrt(np.mean(sq))/case0['bbox_L0'])
                        matrix_rows.append(dict(uid=uid,candidate=candidate,alpha=alpha,beta=beta,E=E[i,j]))
                column=np.array([(E[:,j].sum()-E[j,j])/2-E[j,j] for j in range(3)])
                if candidate=='perfect_response':
                    assert np.max(np.diag(E))<1e-6 and np.min(column)>0
                else:
                    assert bits(E[0],E[1]) and bits(E[1],E[2]) and np.max(abs(column))<1e-15
                    assert E[0,0]>1e-7 and E[2,2]>1e-7
                check('column_positive_control_'+uid+'_'+candidate,matrix=E.tolist(),column_advantage=column.tolist(),
                      formula='E[alpha,beta]=sqrt(mean_q d_triangle(Q_beta,F_alpha)^2)/untransformed_L0')
        with (testout/'column_positive_matrices.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(matrix_rows[0]));writer.writeheader();writer.writerows(matrix_rows)
        assert not torch.cuda.is_initialized()
        result.update(status='PASS',CUDA_initialized=False,wall_seconds=time.perf_counter()-started)
        save(summary,result)
        return result
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started)
        save(summary,result)
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path)
    p.add_argument('--chair',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();result=run(a.source,a.chair,a.out)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},sort_keys=True),flush=True)


if __name__=='__main__':main()
