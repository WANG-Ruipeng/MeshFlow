"""Freeze the dedup-only paired experiment and its actual formal CPU inputs."""
import collections, hashlib, json, shutil
import numpy as np
from . import common as C
C.package()
from meshflow_control.data.dataset import training_time


def seed(domain, *parts):
    return int.from_bytes(hashlib.sha256(json.dumps([C.VERSION,domain,*parts],separators=(',',':')).encode()).digest()[:4],'little')


def plans(manifests, registration_sha):
    parents=sorted(manifests[C.ARMS[0]]['parents']); schedule=[]
    rng=np.random.RandomState(seed('parent_cycles'))
    for _ in range(100):schedule.extend(parents[int(i)] for i in rng.permutation(16))
    result={}
    for arm,m in manifests.items():
        tasks={t['uid']:t for t in m['tasks'].values()}; rows=[]
        for b in range(200):
            rng=np.random.RandomState(seed('augmentation',b))
            alphas=np.r_[np.ones(4,np.float32),rng.uniform(.9,1.1,4).astype(np.float32)]
            alphas=alphas[rng.permutation(8)]
            assert np.count_nonzero(alphas==1)==4
            for j in range(8):
                i=8*b+j; uid=schedule[i]; t=tasks[uid]; ts=seed('time',i)
                rows.append(dict(sample_index=i,effective_batch_index=b,slot_index=j,step=b+1,global_step=12001+b,
                    arm=arm,uid=uid,task_id=t['task_id'],N=t['N'],K=t['K'],ratio=t['K']/t['N'],
                    alpha=float(alphas[j]),clean=bool(alphas[j]==1),condition_type='SEAT_FULL',region_type='seat',
                    t_seed=ts,t=float(training_time(ts)),gaussian_seed=seed('Gaussian',i),
                    epsilon_seed=seed('epsilon',i),permutation_seed=seed('permutation',i)))
        counts=dict(collections.Counter(r['uid'] for r in rows)); assert set(counts.values())=={100}
        result[arm]=dict(schema='r2_target_dedup_paired_plan_v1',arm=arm,registration_sha256=registration_sha,
            updates=200,records=rows,by_parent=counts,free_coordinate_count=sum(9*(r['N']-r['K']) for r in rows),
            rng_contract='Same parent/alpha/time and Gaussian/epsilon/permutation seed identities. N and free count differ; recompute OT independently.')
    for a,b in zip(result[C.ARMS[0]]['records'],result[C.ARMS[1]]['records']):
        assert all(a[k]==b[k] for k in ('uid','K','alpha','t','gaussian_seed','epsilon_seed','permutation_seed'))
    return result


def prepare():
    target=C.OUT/'registration.json'
    if target.exists():
        r=C.read(target)
        for p,h in {**r['protected'],**r['training_code_hashes']}.items():assert C.sha(p)==h,p
        return dict(status='VERIFIED_FROZEN',sha256=C.sha(target))
    oldout=C.ROOT/'experiments/r2_full_seat_learning_curve_v1'; old=C.read(oldout/'registration.json')
    source=C.read(C.SOURCE.with_suffix('.json')); assert source['global_step']==12000
    assert C.sha(C.SOURCE)==source['sha256']
    root=C.ROOT/'experiments/r2_target_cleanup_v1'
    manifests={};datasets={};protected={};identity=[]
    for arm,which in zip(C.ARMS,('ORIGINAL','DEDUP')):
        p=root/'datasets'/which/'train_manifest.json';m=C.read(p)
        assert m['schema']=='r2_target_cleanup_training_data_v1' and len(m['parents'])==len(m['tasks'])==16
        manifests[arm]=m;datasets[arm]=dict(path=str(p),sha256=C.sha(p));protected[str(p)]=C.sha(p)
        for row in list(m['parents'].values())+list(m['tasks'].values()):
            assert C.sha(row['npz'])==row['sha256'];protected[row['npz']]=row['sha256']
    ma,mb=[manifests[a] for a in C.ARMS];assert set(ma['parents'])==set(mb['parents'])
    for uid in sorted(ma['parents']):
        a=next(t for t in ma['tasks'].values() if t['uid']==uid);b=next(t for t in mb['tasks'].values() if t['uid']==uid)
        assert a['condition_type']==b['condition_type']=='SEAT_FULL' and a['K']==b['K']>=2
        assert C.same(C.arrays(a['npz'])['C'],C.arrays(b['npz'])['C'])
        identity.append(dict(uid=uid,K=a['K'],N_original=a['N'],N_dedup=b['N'],C_exact=True,exposures_per_arm=100))
    binding=C.read(oldout/'evaluation/bindings/FULL_S6000.json');cache={}
    for q in old['requests']:
        item={'historical_label':'FULL_S6000'}
        for stage in ('raw','post'):
            p=oldout/'evaluation'/stage/'FULL_S6000'/q['request_id']/'receipt.json'
            if not p.exists():
                matches=list((oldout/'evaluation'/stage/'FULL_S6000').rglob('*.json')) if (oldout/'evaluation'/stage/'FULL_S6000').exists() else []
                raise FileNotFoundError(str(p)+'; receipts='+str(matches[:2]))
            item[stage+'_path']=str(p);item[stage+'_sha256']=C.sha(p)
        cache[q['request_id']]=item
    paths=sorted((C.ROOT/'r2/src/meshflow_control').rglob('*.py'))
    paths += [C.CODE/p for p in ('common.py','prepare.py','train.py')]
    paths += [C.ROOT/p for p in ('r2/pilots/full_seat_learning_curve_v1/common.py','r2/pilots/user_data_scaling_32_67_v1/training_data.py','r2/pilots/user_condition_finetune_v1/trainer.py')]
    disk=shutil.disk_usage(C.ROOT);assert disk.free>10*1024**3
    r=dict(version=C.VERSION,status='FROZEN',utc=C.utc(),source_checkpoint=str(C.SOURCE),source_checkpoint_sha256=source['sha256'],
        source_generator_sha256=source['generator_state_sha256'],source_optimizer_sha256=source['optimizer_state_sha256'],source_rng_sha256=source['rng_sha256'],
        source_global_step=12000,updates_per_arm=200,batch=8,microbatch=1,datasets=datasets,protected=protected,
        training_code_hashes={str(p):C.sha(p) for p in paths},evaluation_labels=list(C.ARMS)+['FULL_S6000'],
        requests=old['requests'],queries=old['queries'],conditions=old['conditions'],d0_checkpoint=old['d0_checkpoint'],
        models={'FULL_S6000':binding},cache={'FULL_S6000':cache},
        resources={**old['resources'],'training_incremental_estimate_MiB':4096,'CPU_OT_workers':2},
        budget=dict(arms=2,updates_each=200,train_forward_each=1600,train_backward_each=1600,new_rollouts=48,new_sampling_forward=2400,new_D0_forward=48,cached_reference_requests=24),
        training_contract=dict(model='Native/T1+role+Geo; WORLD coordinates',objective='original free-only FM; OT then HYBRID0.25',
            optimizer='restore same FULL_S6000 live generator/AdamW/RNG, no reset, no EMA mixing',
            sampling='16 affected TRAIN parents, one coherent SEAT_FULL each; 100 exposures each',
            intervention='exact duplicate free target copies only; C and interface owners protected; N/OT dimension/free loss denominator change',
            precision='BF16 forward FP32 parameters/integration, deterministic math SDPA',no_new_loss=True,no_new_seam=True),
        evaluation_contract=dict(panel='unchanged six FULL C x four existing noises',all_raw_before_D0=True,
            post='one official D0 / GEO_C / original-source C winding and free orientation',source_GT_rank=False,
            main_endpoint=200,no_best_checkpoint=True,review='actual AI_PROXY, RAW/POST and TRAIN/DEV separate'),
        failure_policy='No silent failed replay. Only exact safe checkpoints resume. Engineering repairs preserve code and receipts.',
        third_item='Preparation only: no mixed-data, longer-SEAM or new third-point-constraint training',
        final_actions='No default switch, automatic commit/push, or old asset cleanup')
    C.write(target,r);rh=C.sha(target)
    for arm,plan in plans(manifests,rh).items():C.write(C.OUT/'plans'/f'{arm}.json',plan)
    C.csvwrite(C.OUT/'inputs/paired_parents.csv',identity)
    C.csvwrite(C.OUT/'inputs/evaluation_exposure.csv',[dict(condition_id=q['condition_id'],parent_id=q['parent_id'],historical_split=q['split'],continued_in_this_pilot=q['parent_id'] in ma['parents']) for q in old['queries']])
    C.write(C.OUT/'resources/initial.json',dict(resources=C.resources(),disk_free_bytes=disk.free,
        dispatch='Start one authorized arm, observe backward/optimizer process VRAM and utilization; launch second only if resource/throughput permits.'))
    C.event('REGISTRATION_FROZEN',sha256=rh,arms=list(C.ARMS),updates_each=200,parents=16,new_rollouts=48)
    return dict(status='FROZEN',sha256=rh,parents=16,new_rollouts=48)


if __name__=='__main__':print(json.dumps(prepare(),indent=2))
