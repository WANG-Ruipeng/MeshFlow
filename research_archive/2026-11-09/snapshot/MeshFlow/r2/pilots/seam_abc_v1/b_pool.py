"""Independent CPU preparation of all TRAIN47 owner-slot data and old DEV inputs.

No model imports, forward, GPU initialization, new evaluation noise or old-file writes.
Imports do not mutate CUDA visibility. CLI sets CPU thread limits before numpy import.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'experiments/r2_seam_abc_v1'
PRIOR = ROOT/'experiments/r2_idea_1b_3_4a_diagnostic_v1/idea3'
OLD = ROOT/'experiments/r2_seam_slot_c200_v1'
_ARRAY_CACHE = {}
_INTERFACES = None
ALLOW_ASSET_RESUME = False


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def csvwrite(path, rows):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def arrays(path):
    import numpy as np
    key=str(path)
    if key not in _ARRAY_CACHE:
        with np.load(path,allow_pickle=False) as z:
            _ARRAY_CACHE[key]={k:z[k].copy() for k in z.files}
            for x in _ARRAY_CACHE[key].values():x.setflags(write=False)
    return _ARRAY_CACHE[key]


def interfaces():
    global _INTERFACES
    if _INTERFACES is None:
        _INTERFACES=defaultdict(list)
        with (PRIOR/'per_source_interface.csv').open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):_INTERFACES[row['condition_id']].append(row)
    return _INTERFACES


def make_case(record,alpha):
    import numpy as np
    from meshflow_control.seam.slot_data import edge_features
    p,c=arrays(record['parent_npz']),arrays(record['condition_path'])
    known=c['source_face_ids'].astype(np.int64);free=c['free_source_ids'].astype(np.int64)
    n=len(p['source_faces']);k=len(known)
    v=p['model_vertices'].copy();v[:,0]*=np.float32(alpha)
    full=p['full_target'].reshape(n,3,3).copy();full[:,:,0]*=np.float32(alpha)
    pre=p['pre_ot_full'].copy();pre[:,:,0]*=np.float32(alpha)
    source_ids=np.unique(p['source_faces'][known])
    lookup={int(vi):i for i,vi in enumerate(source_ids)}
    original=np.asarray([[lookup[int(vi)] for vi in f] for f in p['source_faces'][known]],np.int64)
    canonical=np.asarray([[lookup[int(vi)] for vi in f] for f in p['source_face_vertex_ids'][known]],np.int64)
    cv=v[source_ids];edges,features=edge_features(cv,original,n)
    legal=interfaces()[record['condition_id']]
    byedge={tuple(sorted((int(r['vertex_u']),int(r['vertex_v'])))):r for r in legal}
    if len(byedge)!=len(legal):raise ValueError('Multiple legal Y rows on one C edge require a new policy')
    byface=defaultdict(list)
    for r in legal:byface[int(r['Y_source_face'])].append(r)
    owneredge={fi:min(int(r['source_edge_id']) for r in rows) for fi,rows in byface.items()}
    labels=np.zeros(len(edges),bool);supported=np.zeros(len(edges),bool)
    alias=np.zeros(len(edges),bool);thirdC=np.zeros(len(edges),bool)
    owner=np.full(len(edges),-1,np.int64);triple=np.full((len(edges),3),-1,np.int64)
    audit=[]
    for ei,(cfi,u,w) in enumerate(edges):
        su,sw=int(source_ids[u]),int(source_ids[w]);r=byedge.get(tuple(sorted((su,sw))))
        if r is None:continue
        assert int(r['source_edge_id'])==ei and r['winding_opposite']=='True'
        fi=int(r['Y_source_face']);third=int(r['third_vertex_id'])
        isowner=ei==owneredge[fi]
        assert int(r['owner_slot_edge_id'])==owneredge[fi]
        assert (r['assignment']=='FACE_OWNER')==isowner
        supported[ei]=True;labels[ei]=isowner;alias[ei]=not isowner
        thirdC[ei]=third in lookup
        owner[ei]=fi;triple[ei]=[sw,su,third]
        audit.append(dict(condition_id=record['condition_id'],parent_id=record['parent_id'],group=record['group'],
            source_edge_id=ei,Y_source_face=fi,owner_slot_edge_id=owneredge[fi],
            source_supported=True,occupied_owner_label=isowner,alias_returned_bulk=not isowner,
            third_vertex_id=third,third_in_C=bool(thirdC[ei]),
            third_policy='FREE_TIP_NO_EXTRA_GT_ANCHOR' if isowner else 'NO_SEAM_FOR_ALIAS'))
    assert len(set(owner[labels]))==int(labels.sum())==len(byface)
    assert np.array_equal(np.sort(np.r_[known,free]),np.arange(n))
    assert np.array_equal(v[p['source_face_vertex_ids']],full)
    assert np.array_equal(pre*np.float32(2),full)
    return dict(full_target=full,pre_ot_full=pre,source_face_vertex_ids=p['source_face_vertex_ids'],
        known_ids=known,free_ids=free,C_vertices=cv,C_faces_original=original,C_faces_model=canonical,
        C_source_vertex_ids=source_ids,occupied_labels=labels,edge_owner_face=owner,edge_source_triple=triple,
        edge_features=features,N=n,K=k,source_supported_labels=supported,
        source_alias_labels=alias,source_third_in_C=thirdC),audit


def resources():
    import subprocess, psutil
    commands={}
    for name,args in [('gpu',['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),
                      ('processes',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])]:
        result=subprocess.run(args,capture_output=True,text=True)
        commands[name]=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    return dict(cpu_cores=os.cpu_count(),available_ram_bytes=psutil.virtual_memory().available,
        total_ram_bytes=psutil.virtual_memory().total,cpu_percent=psutil.cpu_percent(interval=.1),
        cpu_OT_workers=2,threads_per_worker=1,GPU_workers=0,nvidia_smi=commands,
        scope='Read-only resource snapshot before CPU source packaging and OT cache preparation')


def save_npz(path,values):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not ALLOW_ASSET_RESUME:raise RuntimeError('No overwrite '+str(path))
        with np.load(path,allow_pickle=False) as old:
            assert set(old.files)==set(values),str(path)
            for key,value in values.items():
                value=np.asarray(value)
                assert old[key].dtype==value.dtype and old[key].shape==value.shape and np.array_equal(old[key],value),(str(path),key)
        return
    with path.open('xb') as f:np.savez_compressed(f,**values)


def schedule(records):
    import numpy as np
    rng=np.random.RandomState(2026101607)
    old=[]
    priorrecords=read(OLD/'registration.json')['records']
    for step in range(1,201):
        rows=[];order=rng.permutation(4)
        for aug in range(2):
            for ci in order:
                alpha=1. if aug==0 else float(rng.uniform(.9,1.1));seed=10000000+step*100+int(ci)*10+aug
                rows.append(dict(condition_index=int(ci),uid=priorrecords[ci]['parent_id'],sample_index=len(rows),alpha=alpha,
                    noise_seed=seed,epsilon_seed=seed+100000000,time_seed=seed+200000000,permutation_seed=seed+300000000))
        old.append(dict(step=step,rows=rows))
    assert old==read(OLD/'input_schedule.json')['steps']
    queue=[];cursor=0;steps=[];exposures=Counter();allseeds=set()
    for step in range(1,801):
        # Random permutations without replacement balance parent positions.
        while len(queue)-cursor<4:queue.extend(map(int,rng.permutation(len(records))))
        chosen=queue[cursor:cursor+4];cursor+=4;rows=[];source_step=step+200
        # At an epoch boundary a repeated parent may occur in one batch. Seed
        # uses position rather than parent identity only in that documented case.
        for aug in range(2):
            for position,ci in enumerate(chosen):
                alpha=1. if aug==0 else float(rng.uniform(.9,1.1))
                seed=10000000+source_step*1000+ci*10+aug
                if chosen.count(ci)>1:seed=10000000+source_step*1000+500+position*10+aug
                if seed in allseeds:raise ValueError('Registered noise seed collision')
                allseeds.add(seed);exposures[records[ci]['parent_id']]+=1
                rows.append(dict(condition_index=ci,uid=records[ci]['parent_id'],sample_index=len(rows),alpha=alpha,
                    noise_seed=seed,epsilon_seed=seed+100000000,time_seed=seed+200000000,permutation_seed=seed+300000000,
                    epoch_boundary_repeated_parent=chosen.count(ci)>1))
        steps.append(dict(step=step,source_step=source_step,rows=rows))
    assert sum(exposures.values())==6400 and sorted(Counter(exposures.values()).items())==[(136,43),(138,4)]
    return dict(schema='B_POOL47_balanced_parent_schedule_v1',seed=2026101607,replayed_old_steps=200,
        steps=steps,parent_exposure=dict(sorted(exposures.items())),
        augmentation='4 parent positions per batch, each clean then x-width U(.9,1.1)',
        seeds='10000000+source_step*1000+condition_index*10+aug; repeated epoch-boundary parents use 500+position*10+aug',
        draw_policy='Independent per-example MT19937 z, epsilon, logistic-normal t and target/face permutations; OT before HYBRID0.25')


def freeze():
    import numpy as np
    from meshflow_control.seam.slot_data import inference_layout
    regpath=OUT/'B_DATA_REGISTRATION.json'
    if regpath.exists():
        reg=read(regpath);verify(reg);return reg
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'B_DATA_STARTED.json').exists():
        if not ALLOW_ASSET_RESUME:raise RuntimeError('Incomplete source preparation requires explicit engineering review')
        write(OUT/'B_RESOURCES_ASSET_REPAIR01.json',resources())
    else:
        write(OUT/'B_RESOURCES_CPU.json',resources())
        write(OUT/'B_DATA_STARTED.json',dict(stage='ASSET_PREPARATION',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    old=read(PRIOR/'registration.json')
    records=[dict(r) for r in old['records'] if r['group']=='TRAIN47']
    dev=[dict(r) for r in old['records'] if r['group']=='PANEL_DEV']
    # Historical registration calls all six evaluation parents PANEL6.
    if not dev:dev=[dict(r) for r in old['records'] if r['split']=='DEV']
    assert len(records)==47 and len(dev)==3
    protected={str(p):sha(p) for p in [PRIOR/'registration.json',PRIOR/'per_source_interface.csv',
        PRIOR/'per_condition.csv',OLD/'registration.json',OLD/'input_schedule.json',
        ROOT/'r2/src/meshflow_control/seam/slot_data.py',ROOT/'r2/src/meshflow_control/training/coupling.py']}
    owncode={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('slot_data_extension.py')]}
    audits=[];counts=[];dev_conditions=[];dev_requests=[]
    oldeval=read(OLD/'eval_manifest.json')
    historical=read(ROOT/'experiments/r2_regular_patch_control_v1/registration.json')
    protected[str(ROOT/'experiments/r2_regular_patch_control_v1/registration.json')]=sha(ROOT/'experiments/r2_regular_patch_control_v1/registration.json')
    for record in records+dev:
        for pk,hk in [('condition_path','condition_sha256'),('parent_npz','parent_sha256'),('mask_path','mask_sha256'),('proof_path','proof_sha256'),('quality_path','quality_sha256')]:
            if pk not in record:
                assert record['split']=='DEV' and pk in ('mask_path','proof_path')
                continue
            assert sha(record[pk])==record[hk],record[pk]
            protected[record[pk]]=record[hk]
        case,audit=make_case(record,1.);audits.extend(audit)
        n,k=int(case['N']),int(case['K']);prefix=record['parent_id'][:8]
        base=OUT/'B_data'/'inputs'/prefix
        inp={key:case[key] for key in ['C_vertices','C_faces_original','C_faces_model']};inp['total_N']=np.array(n,np.int64)
        save_npz(base/'C_only.npz',inp);save_npz(base/'training_labels.npz',case)
        parent=arrays(record['parent_npz'])
        meta=dict(C_source_vertex_ids=case['C_source_vertex_ids'],source_C_face_ids=case['known_ids'],
            source_C_corner_permutations=parent['source_corner_permutations'][case['known_ids']],
            C_faces_original=case['C_faces_original'],C_faces_model=case['C_faces_model'])
        save_npz(base/'identity_metadata.npz',meta)
        for pattern in [np.zeros(len(case['occupied_labels']),bool),case['occupied_labels'],np.ones(len(case['occupied_labels']),bool)]:
            layout=inference_layout(**inp,occupied=pattern)
            assert len(layout['seam_slots'])+len(layout['bulk_slots'])==n-k
        record.update(inference_npz=str(base/'C_only.npz'),labels_npz=str(base/'training_labels.npz'),
            metadata_npz=str(base/'identity_metadata.npz'),N=n,K=k,edges=len(case['occupied_labels']),
            occupied=int(case['occupied_labels'].sum()),Null=int((~case['occupied_labels']).sum()))
        counts.append(dict(parent_id=record['parent_id'],group=record['group'],split=record['split'],N=n,K=k,
            C_edges=len(case['occupied_labels']),source_supported=int(case['source_supported_labels'].sum()),
            owner_slots=int(case['occupied_labels'].sum()),alias_slots=int(case['source_alias_labels'].sum()),
            source_Null=int((~case['source_supported_labels']).sum()),owner_label_Null=int((~case['occupied_labels']).sum()),
            third_in_C_edges=int(case['source_third_in_C'].sum()),third_in_C_owners=int((case['source_third_in_C']&case['occupied_labels']).sum()),
            source_faces_exact_once=True,allNull_allOn_layouts_valid=True))
        oldcond=next((c for c in oldeval['conditions'] if c['parent_id']==record['parent_id']),None)
        if oldcond:
            original=arrays(oldcond['labels_npz'])
            for key in original:
                assert np.array_equal(original[key],case[key]),f'Old four-C case changed: {prefix}/{key}'
        if record['split']=='DEV':
            query=next(q for q in historical['queries'] if q['parent_id']==record['parent_id'] and 'SEAT_FULL' in q['condition_id'])
            assert sha(query['path'])==query['sha256'];protected[query['path']]=query['sha256']
            cond=dict(condition_id=record['condition_id'],parent_id=record['parent_id'],split='DEV',N=n,K=k,
                inference_npz=record['inference_npz'],metadata_npz=record['metadata_npz'],labels_npz=record['labels_npz'],
                query_path=query['path'],query_sha256=query['sha256'],query_descriptor=query,source_record=record,
                edges=record['edges'],occupied=record['occupied'],Null=record['Null'])
            dev_conditions.append(cond)
            for ni in range(4):
                prior=next(r for r in historical['requests'] if r['parent_id']==record['parent_id'] and int(r['candidate'])==ni)
                assert sha(prior['input_path'])==prior['input_sha256']
                protected[prior['input_path']]=prior['input_sha256']
                oldinput=arrays(prior['input_path'])
                assert np.array_equal(oldinput['C'],case['full_target'][case['known_ids']])
                z=oldinput['z'].reshape(n,3,3)[k:].copy()
                noisepath=base/f'noise_n{ni}.npz'
                save_npz(noisepath,dict(z=z,seed=np.array(prior['noise_seed'],np.int64)))
                dev_requests.append(dict(condition_id=record['condition_id'],parent_id=record['parent_id'],N=n,K=k,
                    inference_npz=record['inference_npz'],metadata_npz=record['metadata_npz'],query_path=query['path'],
                    request_id=f'{prefix}_n{ni}',noise_index=ni,candidate=ni,split='DEV',noise_npz=str(noisepath),
                    noise_sha256=sha(noisepath),seed=prior['noise_seed'],query_descriptor=query,
                    source_request_id=prior['request_id'],source_input_path=prior['input_path'],source_input_sha256=prior['input_sha256'],
                    noise_provenance='Exact saved full-N request z[K:] copied; zero random draws; original free slots unchanged',
                    z_array_sha256=hashlib.sha256(z.tobytes()).hexdigest()))
    csvwrite(OUT/'B_SOURCE_CONDITIONS.csv',counts);csvwrite(OUT/'B_SOURCE_ALIAS_THIRD_C.csv',audits)
    sched=schedule(records);write(OUT/'B_INPUT_SCHEDULE.json',sched)
    write(OUT/'B_EVAL_DEV_ASSETS.json',dict(status='COMPLETE_C_ONLY_BUNDLES_AND_FROZEN_NOISE',conditions=dev_conditions,
        requests=dev_requests,new_noise_draws=0,source_occupancy_used_in_inference=False,
        labels_semantics='Owner slots only; aliases are ordinary bulk; third-in-C tips remain free',
        controls='Reuse old four-C16 SEAM200/LONG; DEV3x4 must be generated from each frozen comparison model'))
    registration=dict(schema='B_POOL47_owner_data_v1',status='FROZEN',records=records,DEV_records=dev,
        updates=800,batch=8,microbatch=1,samples=6400,production_OT_budget=6400,
        source_checkpoint=str(OLD/'checkpoints/SEAM_SLOT_C200_step200.pt'),
        source_checkpoint_receipt=read(OLD/'checkpoints/SEAM_SLOT_C200_step200.json'),
        source_global_step=12200,final_global_step=13000,head_class_weights=[8.25,66/124],head_lr=.001,
        label_semantics='One minimum-registered-edge owner per legal Y face; source aliases return to bulk; third-C is free tip',
        old_four_C_all_existing_arrays_equal=True,source_y_faces_removed=0,
        protected_hashes=protected,code_hashes=owncode,input_schedule_sha256=sha(OUT/'B_INPUT_SCHEDULE.json'),
        budget_fairness='Same SEAM200 state +800 as LONG; data exposure differs and general owner-label extension is explicit, not a perfect single-variable 4-to47 comparison',
        artifact_hashes={str(p):sha(p) for p in (OUT/'B_data/inputs').rglob('*') if p.is_file()},
        source_conditions=counts)
    write(regpath,registration)
    train=[r for r in counts if r['split']=='TRAIN']
    report=f'''# B TRAIN47 数据实核\n\n已实际构造47 TRAIN及3 DEV来源包，旧4C原有case数组全部相等。无删坏父、无来源几何修复、无新评测噪声、无模型调用。\n\nTRAIN共有{sum(r['C_edges'] for r in train)}条C开放边、{sum(r['source_supported'] for r in train)}条来源合法接口边、{sum(r['owner_slots'] for r in train)}个独立owner槽、{sum(r['alias_slots'] for r in train)}个alias返回bulk。third-in-C合法边{sum(r['third_in_C_edges'] for r in train)}条，其中owner{sum(r['third_in_C_owners'] for r in train)}条仍作为自由tip，不加GT锚。每父原始目标面恰好一次，N不变。\n\n原SEAM最小代码不支持这些情况；本轮显式扩展label为owner-slot，不能称仅扩大父数而任何代码/标签都没变。所有来源对照和别名明细在B_SOURCE_CONDITIONS.csv、B_SOURCE_ALIAS_THIRD_C.csv。来源31/47有旧非法FF及3父UNKNOWN仍原样保留，不把数据宣称无交。\n\n固定6400曝光，每父136或138；clean与x-width U(.9,1.1)各3200；恢复旧200步data RNG后按47父排列循环；新seed步距1000避免跨step重号，epoch边界同父重复用500+batch位置独立命名。t、z、epsilon、permutation、OT后HYBRID.25沿原算法。不能与LONG声称逐实例随机数组一一相同。\n\nDEV资产在B_EVAL_DEV_ASSETS.json：3父×4既有噪声，直接取旧FULL请求z[K:]，C数组与原请求逐位一致；仅C和N进入预测占用布局。source标签单独文件，只用于训练/离线诊断。\n\nOT缓存尚需data阶段实际完成；本文件是数据合同核验，不是训练结果。当前scope与先前R31四TRAIN接续信号不同，检验更多训练C能否在原4C及历史DEV3上兼顾接续与质量，不预设更广池必然防记忆或获胜。\n'''
    with (OUT/'B_DATA_AUDIT.md').open('x',encoding='utf-8') as f:f.write(report)
    verify(registration);return registration


def verify(reg):
    for name in ['protected_hashes','code_hashes','artifact_hashes']:
        for path,value in reg[name].items():
            if sha(path)!=value:raise RuntimeError('Frozen asset changed: '+path)
    assert sha(OUT/'B_INPUT_SCHEDULE.json')==reg['input_schedule_sha256']


def worker(item):
    import numpy as np
    from .slot_data_extension import supervised_sample
    from meshflow_control.seam.slot_data import boundary_from_C
    entry,records=item;out=[];calls=0
    for row in entry['rows']:
        path=OUT/'B_data/cache'/f's{entry["source_step"]:04d}_m{row["sample_index"]}.npz'
        receipt=path.with_suffix('.json');started=path.with_suffix('.started.json')
        if path.exists():
            if not receipt.exists() or sha(path)!=read(receipt)['sha256']:raise RuntimeError('Incomplete existing cache '+str(path))
        else:
            write(started,dict(step=entry['step'],source_step=entry['source_step'],row=row,OT_calls_reserved=1))
            record=records[row['condition_index']];case,_=make_case(record,row['alpha']);parent=arrays(record['parent_npz'])
            n,k=int(case['N']),int(case['K'])
            z=np.random.RandomState(row['noise_seed']).randn(n-k,3,3)
            eps=np.random.RandomState(row['epsilon_seed']).randn(n-k,3,3)
            t=np.float32(1/(1+np.exp(-np.random.RandomState(row['time_seed']).randn())))
            sample=supervised_sample(case,z,eps,t,row['permutation_seed']);calls+=1
            source_third=np.zeros((n,3),bool);source_owner=np.zeros(n,bool)
            for ei in np.flatnonzero(case['occupied_labels']):
                fi=int(case['edge_owner_face'][ei]);third=case['edge_source_triple'][ei,2]
                source_third[fi]=case['source_face_vertex_ids'][fi]==third;source_owner[fi]=True
            ids,cp=sample['source_face_ids'],sample['source_corner_permutations']
            current=case['source_face_vertex_ids'][ids[:,None],cp];original=parent['source_faces'][ids]
            winding=np.asarray([[int(np.flatnonzero(cur==v).item()) for v in ori] for cur,ori in zip(current,original)],np.int64)
            final_slots=np.argsort(sample['face_permutation'])[k+np.flatnonzero(case['occupied_labels'])]
            edge=boundary_from_C(case['C_vertices'],case['C_faces_original'])
            sample.update(seam_third_corner_mask=source_third[ids[:,None],cp],source_interface_face_mask=source_owner[ids],
                source_winding_permutation=winding,uid=np.asarray(row['uid']),alpha=np.asarray(row['alpha']),
                sample_index=np.asarray(row['sample_index']),source_stream_step=np.asarray(entry['source_step']),
                C_faces_original=case['C_faces_original'],anti_cover_seam_slots=final_slots,
                anti_cover_owner_faces=edge[case['occupied_labels'],0],source_supported_labels=case['source_supported_labels'],
                source_alias_labels=case['source_alias_labels'],source_third_in_C=case['source_third_in_C'])
            assert np.array_equal(np.take_along_axis(current,winding,1),original)
            assert np.array_equal(sample['anchor_ids'][final_slots,:2],edge[case['occupied_labels'],1:][:,::-1])
            assert int(sample['seam_third_corner_mask'].sum())==int(case['occupied_labels'].sum())
            assert (sample['anchor_ids'][final_slots,2]<0).all()
            save_npz(path,sample)
            write(receipt,dict(sha256=sha(path),step=entry['step'],source_step=entry['source_step'],OT_calls=1,
                source_face_corner_bijection=True,anchors_exact=True,third_in_C_remains_free=True))
        out.append(dict(path=str(path),sha256=sha(path),uid=row['uid'],alpha=row['alpha'],sample_index=row['sample_index']))
    return dict(step=entry['step'],source_step=entry['source_step'],samples={'B_POOL47':out},new_OT_calls=calls)


def data(workers):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    reg=freeze();target=OUT/'B_DATA_MANIFEST.json'
    if target.exists():
        result=read(target)
        for step in result['steps']:
            for r in step['samples']['B_POOL47']:assert sha(r['path'])==r['sha256']
        return result
    if workers not in (1,2):raise ValueError('Approved one/two CPU worker only')
    tick=time.perf_counter();done=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,(entry,reg['records'])) for entry in read(OUT/'B_INPUT_SCHEDULE.json')['steps']]
        for fut in as_completed(futures):
            done.append(fut.result())
            if len(done)%25==0:print(json.dumps(dict(B_completed_steps=len(done),seconds=time.perf_counter()-tick)),flush=True)
    done.sort(key=lambda x:x['step']);verify(reg)
    result=dict(status='COMPLETE',steps=done,samples=6400,OT_logical_calls=6400,
        OT_calls_this_invocation=sum(r['new_OT_calls'] for r in done),CPU_workers=workers,
        seconds=time.perf_counter()-tick,registration_sha256=sha(OUT/'B_DATA_REGISTRATION.json'),
        all_source_faces_exact_once=True,all_source_winding_roundtrips=True,all_C_seam_anchors_exact=True,
        all_third_in_C_owner_tips_free=True,old_assets_unchanged=True)
    write(target,result);return result


def main():
    global ALLOW_ASSET_RESUME
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    sys.path.insert(0,str(ROOT/'r2/src'))
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['freeze','data'],default='freeze')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--resume-asset-repair01',action='store_true');args=parser.parse_args()
    ALLOW_ASSET_RESUME=args.resume_asset_repair01
    if ALLOW_ASSET_RESUME and not (OUT/'B_PREP_ENGINEERING_REPAIR01.json').exists():raise RuntimeError('Explicit engineering receipt required')
    result=freeze() if args.stage=='freeze' else data(args.workers)
    print(json.dumps(dict(status=result['status'],stage=args.stage)),flush=True)


if __name__=='__main__':main()
