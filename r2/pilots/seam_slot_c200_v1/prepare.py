"""Freeze all paired inputs and compute the authorized 3200 training OT maps on CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse, concurrent.futures, json, shutil, time
import numpy as np
from . import common as C
from r2.prototypes.seam_slot_c200_v1.run import make_case
from r2.prototypes.seam_slot_c200_v1 import slot_data as S

PROTO=C.ROOT/'experiments/r2_seam_slot_c200_prototype_v1'
def freeze():
    if (C.OUT/'registration.json').exists():return C.read(C.OUT/'registration.json')
    C.OUT.mkdir(parents=True,exist_ok=True);C.resources(C.OUT/'resources_initial.json')
    pr=C.read(PROTO/'registration.json');records=pr['records'];protected=dict(pr['protected_hashes'])
    for p in (C.ROOT/'r2/prototypes/seam_slot_c200_v1').glob('*.py'):protected[str(p)]=C.sha(p)
    source=C.ROOT/'experiments/r2_full_seat_learning_curve_v1/checkpoints/FULL_SEAT_step6000.pt'
    cp=C.read(source.with_suffix('.json'));assert C.sha(source)==cp['sha256'];protected[str(source)]=cp['sha256']
    protected[str(source.with_suffix('.json'))]=C.sha(source.with_suffix('.json'))
    previous=C.read(C.ROOT/'experiments/r2_regular_patch_control_v1/registration.json')
    extra=C.read(C.ROOT/'experiments/r2_seat_failure_diagnosis_v1/registration.json')
    requests=[];conditions=[]
    for ci,record in enumerate(records):
        case=make_case(record,1.);n,k=int(case['N']),int(case['K']);prefix=record['parent_id'][:8]
        base=C.OUT/'inputs'/prefix;base.mkdir(parents=True,exist_ok=True)
        inp={key:case[key] for key in ['C_vertices','C_faces_original','C_faces_model']};inp['total_N']=np.array(n,np.int64)
        np.savez_compressed(base/'C_only.npz',**inp)
        # This separate file is for supervised training/evaluation labels only, never the sampler.
        np.savez_compressed(base/'training_labels.npz',**case)
        p=C.arrays(record['parent_npz'])
        meta=dict(C_source_vertex_ids=case['C_source_vertex_ids'],source_C_face_ids=case['known_ids'],
            source_C_corner_permutations=p['source_corner_permutations'][case['known_ids']],
            C_faces_original=case['C_faces_original'],C_faces_model=case['C_faces_model'])
        np.savez_compressed(base/'identity_metadata.npz',**meta)
        qs=previous['queries']+extra['queries'];query=next(q for q in qs if q['parent_id']==record['parent_id'] and 'SEAT_FULL' in q['condition_id'])
        assert C.sha(query['path'])==query['sha256'];protected[query['path']]=query['sha256']
        vertices=case['C_vertices'];assert len({v.tobytes() for v in vertices})==len(vertices),'C identity alias unsupported'
        cond=dict(condition_id=record['condition_id'],parent_id=record['parent_id'],split='TRAIN',N=n,K=k,
            inference_npz=str(base/'C_only.npz'),metadata_npz=str(base/'identity_metadata.npz'),labels_npz=str(base/'training_labels.npz'),
            query_path=query['path'],query_sha256=query['sha256'],query_descriptor=query,source_record=record,
            edges=len(case['occupied_labels']),occupied=int(case['occupied_labels'].sum()),Null=int((~case['occupied_labels']).sum()))
        conditions.append(cond)
        for ni in range(4):
            seed=3100000000+100*ci+ni;rid=f'{prefix}_n{ni}';noise=base/f'noise_n{ni}.npz'
            z=np.random.RandomState(seed).randn(n-k,3,3).astype(np.float32)
            np.savez_compressed(noise,z=z,seed=np.array(seed,np.int64))
            requests.append({**{key:cond[key] for key in ('condition_id','parent_id','N','K','inference_npz','metadata_npz','query_path')},
                'request_id':rid,'noise_index':ni,'candidate':ni,'split':'TRAIN','noise_npz':str(noise),'noise_sha256':C.sha(noise),'query_descriptor':query,'seed':seed})
    evalm=dict(schema='seam_c200_eval_v1',conditions=conditions,requests=requests,arms=list(C.ARMS),L=C.L,
        training_noise_disjoint=True,new_rollouts=32,D0_calls=32,solver='Euler50',occupancy='logit>=0 once per C, all four noises share layout',source_GT_rank=False)
    C.write(C.OUT/'eval_manifest.json',evalm)
    rng=np.random.RandomState(2026101607);steps=[]
    for step in range(1,201):
        rows=[]
        order=rng.permutation(4)
        for aug in range(2):
            for ci in order:
                si=len(rows);alpha=1. if aug==0 else float(rng.uniform(.9,1.1));seed=10000000+step*100+int(ci)*10+aug
                rows.append(dict(condition_index=int(ci),uid=records[ci]['parent_id'],sample_index=si,alpha=alpha,
                    noise_seed=seed,epsilon_seed=seed+100000000,time_seed=seed+200000000,permutation_seed=seed+300000000))
        steps.append(dict(step=step,rows=rows))
    C.write(C.OUT/'input_schedule.json',dict(steps=steps,seed=2026101607))
    reg=dict(schema='R2_SEAM_SLOT_C200_V1',status='FROZEN',utc=C.utc(),source_checkpoint=str(source),source_checkpoint_sha256=cp['sha256'],
        source_generator_sha256=cp['generator_state_sha256'],source_optimizer_sha256=cp['optimizer_state_sha256'],source_rng_sha256=cp['rng_sha256'],source_global_step=12000,
        updates_per_arm=200,batch=8,batch8=8,microbatch=1,arms=list(C.ARMS),head_seed=20261016,head_lr=.001,head_class_weights=[8.25,66/124],
        head_threshold_logit=0.,head_parameters=833,head_features_schema=S.FEATURE_SCHEMA,diagnostic_time_bin_edges=[0,.2,.4,.6,.8,1],
        d0_checkpoint=previous['d0_checkpoint'],records=records,protected_hashes=protected,eval_manifest_sha256=C.sha(C.OUT/'eval_manifest.json'),input_schedule_sha256=C.sha(C.OUT/'input_schedule.json'),
        budget=dict(training_updates=400,generator_train_forward=3200,generator_train_backward=3200,allNull_integration_forward=2,
            occupancy_train_forward=1600,occupancy_backward=1600,occupancy_updates=200,sampling_forward=1600,RAW=32,D0=32,POST=32,OT=3200),
        training_contract=dict(start='FULL_S6000 live + matched AdamW; new head separate fresh AdamW',augmentation='4 clean and 4 x-width U(.9,1.1); each father twice/batch',
            attention_time='1 only original C faces; t for every face with unknown corner',corner_time='1 on C and active seam anchors; t elsewhere',
            FM='free-only scalars 9(N-K)-6S; all bulk included; original OT before HYBRID0.25',head='weighted BCE only, no generator-quality feedback',
            paired='same source,alpha,time,randomarrays,finalfacepermutation; OT and Xt differ',precision='BF16 forward / FP32 parameters and integration / strict determinism'),
        evaluation_contract=dict(panel='4 seen TRAIN C x4 fresh frozen noises; no DEV generalization',predicted_occupancy_only=True,
            post='one official D0, restore C and candidate anchors, original GEO_C, source winding; candidate extra anchors are a real pipeline change',
            seam_bulk='retained legal-CF seam seeds; ONLY free legal adjacency graph; reached BULK area/retained BULK area; no C traversal',
            NULL_baseline=dict(accuracy='62/66',occupied_recall='62/62',Null_recall='0/4',exact_layout='2/4'),review='nonblind actual AI_PROXY',
            metrics='original CF/FF/I_smallangle_AVMR_PMR, actual legalCF length and overall reachablearea; RAW adoption/body; candidate seam-bulk; full perrequest',
            no_candidate_step0=True,main_endpoint=200,normal_probe='normal source winding; degenerate invalid; only existing training forward'),
        resources=dict(headroom_MiB=2048,CPU_OT_workers=2,dispatch='one authorized worker first; observe backward/optimizer process peak; remeasure before any extra'),
        closest_prior='R10 partner closure teacher success did not become rollout closure; here hard shared C vertex IDs from initial state plus learned C-only occupancy, no partner loss',
        failure_policy='Preserve partial calls/updates and failure record; no silent scientific replay/settings change. Resume only exact checkpoint cursor; engineering repairs separately recorded.',
        stopping='Finish fixed200 endpoint and all32paired outputs; no 6000 extension, default switch, cleanup, commit or push')
    C.write(C.OUT/'registration.json',reg);C.event('REGISTERED',sha256=C.sha(C.OUT/'registration.json'))
    return reg

def worker(item):
    step,rows,records=item;out={arm:[] for arm in C.ARMS};count=0
    for row in rows:
        case=make_case(records[row['condition_index']],row['alpha']);n,k=int(case['N']),int(case['K']);parent=C.arrays(records[row['condition_index']]['parent_npz'])
        z=np.random.RandomState(row['noise_seed']).randn(n-k,3,3);eps=np.random.RandomState(row['epsilon_seed']).randn(n-k,3,3)
        t=np.float32(1/(1+np.exp(-np.random.RandomState(row['time_seed']).randn())))
        source_third=np.zeros((n,3),bool);source_owner=np.zeros(n,bool)
        for ei in np.flatnonzero(case['occupied_labels']):
            fi=int(case['edge_owner_face'][ei]);third=case['edge_source_triple'][ei,2];source_third[fi]=case['source_face_vertex_ids'][fi]==third;source_owner[fi]=True
        pair=[]
        for arm in C.ARMS:
            path=C.OUT/'data'/arm/f's{step:03d}_m{row["sample_index"]}.npz'
            receipt=path.with_suffix('.json')
            if path.exists():
                if not receipt.exists() or C.sha(path)!=C.read(receipt)['sha256']:raise RuntimeError('Incomplete data cell: '+str(path))
                sample=C.arrays(path)
            else:
                sample=S.supervised_sample(case,z,eps,t,row['permutation_seed'],arm);count+=1
                ids=sample['source_face_ids'];cp=sample['source_corner_permutations'];currentids=case['source_face_vertex_ids'][ids[:,None],cp]
                original=parent['source_faces'][ids]
                winding=np.asarray([[int(np.flatnonzero(cur==v).item()) for v in ori] for cur,ori in zip(currentids,original)],np.int64)
                sample.update(seam_third_corner_mask=source_third[ids[:,None],cp],source_interface_face_mask=source_owner[ids],source_winding_permutation=winding,
                    uid=np.asarray(row['uid']),alpha=np.asarray(row['alpha']),sample_index=np.asarray(row['sample_index']))
                assert np.array_equal(np.take_along_axis(currentids,winding,1),original)
                assert int(sample['seam_third_corner_mask'].sum())==int(case['occupied_labels'].sum())
                path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**sample);os.replace(temp,path)
                C.write(receipt,dict(sha256=C.sha(path),step=step,arm=arm,OT_calls=1))
            pair.append(sample)
            out[arm].append(dict(path=str(path),sha256=C.sha(path),uid=row['uid'],alpha=row['alpha'],sample_index=row['sample_index']))
        assert np.array_equal(pair[0]['face_permutation'],pair[1]['face_permutation'])
    return dict(step=step,samples=out,new_OT_calls_this_invocation=count)

def data():
    reg=freeze()
    if (C.OUT/'data_manifest.json').exists():return C.read(C.OUT/'data_manifest.json')
    t=time.perf_counter();schedule=C.read(C.OUT/'input_schedule.json');results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,(s['step'],s['rows'],reg['records'])) for s in schedule['steps']]
        for future in concurrent.futures.as_completed(jobs):
            results.append(future.result())
            if len(results)%20==0:print(json.dumps(dict(data_steps=len(results),seconds=time.perf_counter()-t)),flush=True)
    results.sort(key=lambda r:r['step'])
    manifest=dict(status='COMPLETE',steps=results,samples=3200,OT_calls_logical=3200,seconds=time.perf_counter()-t,CPU_workers=2)
    C.write(C.OUT/'data_manifest.json',manifest);C.event('TRAINING_DATA_READY',samples=3200,seconds=manifest['seconds']);return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['freeze','data'],default='data');a=ap.parse_args()
    r=freeze() if a.stage=='freeze' else data();print(json.dumps(dict(status=r['status'])),flush=True)
