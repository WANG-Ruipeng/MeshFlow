"""CPU-only frozen task/coordinate preparation for Chair domain control.

All source assets are read-only. No model is instantiated in this module.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path
import numpy as np
from experiments.mf_h1_local0.control_mvp48_budget_single.prepare import (
    read_json, sha256, _json, _npz, _load_api, _source_audit, _area)

EXPERIMENT = 'chair_domain_control_v1'

def seed(domain, *identity):
    message = json.dumps([EXPERIMENT, domain, *identity], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:4], 'little')

def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

@lru_cache(maxsize=4)
def _api(source):
    return _load_api(Path(source))

@lru_cache(maxsize=96)
def _base(source, uid):
    return _api(source).load_sample(uid, 'p0', Path(source)/'data')

@lru_cache(maxsize=4)
def _manifest(path):
    return read_json(path)

def _task(manifest, task_id):
    matches = [row for row in manifest['tasks'] if row['task_id'] == task_id]
    if len(matches) != 1 or matches[0]['status'] != 'READY':
        raise ValueError('Task is absent, duplicate, or not applicable: '+task_id)
    return matches[0]

def load_task(source, out, task_id, alpha=1.0):
    return load_task_sample(source, _manifest(str(Path(out)/'task_manifest.json')), task_id, alpha)

def load_task_sample(source, manifest, task_id, alpha=1.0):
    """Supervision-only sample. Never pass the entire result to a model."""
    task = _task(manifest, task_id)
    base = _base(str(Path(source).resolve()), task['uid'])
    sample = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in base.items()}
    a = np.float32(alpha)
    if not np.isfinite(a) or not .8999999 <= a <= 1.1000001:
        raise ValueError('Alpha outside registered range')
    n, k = task['N'], task['K']
    ids = np.asarray(task['source_face_ids'], np.int64)
    free = np.asarray(task['free_source_ids'], np.int64)
    if sorted(np.r_[ids, free].tolist()) != list(range(n)) or len(ids) != k:
        raise ValueError('Task C/free partition changed')
    # Transform the saved model and pre-OT coordinates, never rerun normalization.
    for key in ('full_target', 'pre_ot_full', 'model_vertices'):
        if a != np.float32(1):
            view = sample[key].reshape(-1, 3)
            view[:, 0] *= a
    if not bits((sample['pre_ot_full']*np.float32(2)).reshape(n, 9), sample['full_target']):
        raise ValueError('Transformed original pre-OT x2 contract failed')
    if not bits(sample['model_vertices'][sample['source_face_vertex_ids']], sample['full_target'].reshape(n, 3, 3)):
        raise ValueError('Transformed vertex/corner reconstruction failed')
    sample.update(task_id=task_id, patch_id=task_id, N=n, K=k, alpha=float(a),
        source_face_ids=ids, free_source_ids=free,
        original_source_face_ids=sample['retained_to_original_face'][ids].copy(),
        original_free_source_ids=sample['retained_to_original_face'][free].copy(),
        constraints=sample['full_target'][ids].reshape(k, 3, 3).copy(),
        free_target=sample['full_target'][free].copy(), pre_ot_free=sample['pre_ot_full'][free].copy(),
        interface={key:np.asarray(value,np.int64) for key,value in task['interface'].items()},
        patch_metrics=task['metrics'], bbox_L0=float(base['bbox_diagonal']),
        bbox_diagonal=float(np.linalg.norm(np.ptp(sample['full_target'].reshape(-1,3),axis=0))))
    return sample

def _topology(base):
    incident, raw = defaultdict(list), defaultdict(list)
    for target, faces in ((incident,base['source_faces']), (raw,base['original_source_faces'])):
        for i, face in enumerate(faces):
            for a,b in ((0,1),(1,2),(2,0)):
                target[tuple(sorted((int(face[a]),int(face[b]))))].append(i)
    adjacency = [set() for _ in base['source_faces']]
    for ids in incident.values():
        for i in ids:
            adjacency[i].update(j for j in ids if j != i)
    unseen=set(range(base['N'])); components=[]
    while unseen:
        pending=[min(unseen)]; comp=set()
        while pending:
            i=pending.pop()
            if i in comp: continue
            comp.add(i); pending.extend(adjacency[i]-comp)
        unseen-=comp; components.append(sorted(comp))
    return incident,raw,adjacency,components

def _patch(base, ids, topology):
    incident,raw,adjacency,components=topology
    ids=np.asarray(ids,np.int64); known=set(ids.tolist()); free=sorted(set(range(base['N']))-known)
    interface=defaultdict(list); ambiguous=[]
    for edge,faces in sorted(incident.items()):
        inside=[i for i in faces if i in known]; outside=[i for i in faces if i not in known]
        if not inside or not outside: continue
        expected={int(base['retained_to_original_face'][i]) for i in faces}
        if len(faces)!=2 or len(inside)!=1 or len(outside)!=1 or len(raw[edge])!=2 or set(raw[edge])!=expected:
            ambiguous.append(dict(edge=list(edge),retained_faces=faces,original_faces=raw[edge])); continue
        interface['edge_source_vertex_ids'].append(list(edge))
        for role,face in (('known',inside[0]),('free',outside[0])):
            interface[role+'_source_face_ids'].append(face)
            interface[role+'_original_source_face_ids'].append(int(base['retained_to_original_face'][face]))
            interface[role+'_corner_ids'].append([int(np.flatnonzero(base['source_face_vertex_ids'][face]==v)[0]) for v in edge])
    if ambiguous: return None,'AMBIGUOUS_OR_ORIGINAL_NONMANIFOLD_INTERFACE',ambiguous
    if not interface['edge_source_vertex_ids']: return None,'NO_TRUSTED_KNOWN_FREE_INTERFACE',[]
    reached=set(); pending=[int(ids[0])]
    while pending:
        i=pending.pop()
        if i in reached: continue
        reached.add(i); pending.extend((adjacency[i]&known)-reached)
    if reached != known: return None,'DISCONNECTED_SOURCE_EDGE_PATCH',[]
    tri=base['full_target'].reshape(-1,3,3); c=tri[ids]; target=tri[free]
    edges=base['model_vertices'][np.asarray(interface['edge_source_vertex_ids'])]
    metrics=dict(area_fraction=_area(c)/_area(tri),interface_edge_count=len(edges),
        interface_length_model=float(np.linalg.norm(edges[:,1].astype(np.float64)-edges[:,0],axis=-1).sum()),
        C_centroid=c.astype(np.float64).mean((0,1)).tolist(),C_span=np.ptp(c.reshape(-1,3),axis=0).astype(float).tolist(),
        GT_free_X_robust_width=float(np.diff(np.quantile(target.reshape(-1,3)[:,0],[.05,.95]))[0]),
        known_connected_by_complete_source_edges=True,original_degree2_interface=True,
        interface_ambiguities=0,C_original_open_edges_not_interfaces=True)
    return dict(source_face_ids=ids.tolist(),free_source_ids=free,interface=dict(interface),metrics=metrics),None,[]

def _candidates(base, ratio, topology):
    k=int(np.floor(ratio*base['N']+.5)); adjacency=topology[2]
    unique={}; failures=Counter(); ambiguous=[]
    for start in range(base['N']):
        queue=deque([start]); seen={start}; ordered=[]
        while queue and len(ordered)<k:
            i=queue.popleft(); ordered.append(i)
            for j in sorted(adjacency[i]):
                if j not in seen: seen.add(j); queue.append(j)
        if len(ordered)!=k:
            failures['COMPONENT_SMALLER_THAN_K']+=1; continue
        key=tuple(sorted(ordered))
        if key in unique:
            failures['DUPLICATE_SOURCE_FACE_SET']+=1; continue
        patch,reason,details=_patch(base,ordered,topology)
        if patch is None:
            failures[reason]+=1
            if details: ambiguous.append(dict(start=start,details=details))
            continue
        patch.update(N=base['N'],K=k,ratio=ratio,start_source_face_id=start)
        unique[key]=patch
    return list(unique.values()),dict(failures),ambiguous

def _set(task): return frozenset(task['source_face_ids'])
def _jaccard(a,b): return len(a&b)/len(a|b)

def _select(candidates,excluded,anchors, *, width_L=None):
    pool=[x for x in candidates if _set(x) not in excluded]
    if width_L is not None:
        pool=[x for x in pool if x['metrics']['C_span'][0]>=1e-3*width_L and x['metrics']['GT_free_X_robust_width']>1e-8*width_L]
    if not pool: return None
    def rank(item):
        current=_set(item)
        overlap=max((_jaccard(current,_set(a)) for a in anchors),default=0.)
        spatial=min((float(np.linalg.norm(np.asarray(item['metrics']['C_centroid'])-a['metrics']['C_centroid'])) for a in anchors),default=0.)
        area_spread=min((abs(item['metrics']['area_fraction']-a['metrics']['area_fraction']) for a in anchors),default=0.)
        tie=seed('task_candidate_tie',item['N'],item['K'],item['start_source_face_id'])
        return (overlap,-spatial,-area_spread,tie,item['start_source_face_id'])
    return min(pool,key=rank)

def _strata(uids,objects,count):
    ranked=sorted(uids,key=lambda u:(objects[u]['N'],u)); width=len(ranked)//count
    return [min(ranked[width*i:width*(i+1)]) for i in range(count)]

def geometry_manifest(source,out):
    source,out=Path(source).resolve(),Path(out).resolve()
    ready,splits,objects,checked=_source_audit(source)
    frozen=out/'task_manifest.json'
    if frozen.exists():
        manifest=read_json(frozen)
        if manifest['source_data_ready_sha256']!=sha256(source/'DATA_READY.json'): raise ValueError('Source identity changed')
        return manifest
    old=read_json(source/'data/patches.json')['records']; old_by=defaultdict(list)
    for row in old: old_by[row['uid']].append(row)
    train=_strata(splits['train'],objects,8); val=sorted(splits['validation'])
    width_train=_strata(train,objects,4); width_val=_strata(val,objects,4)
    manifest=dict(schema='chair_domain_task_manifest_v1',experiment_id=EXPERIMENT,source=str(source),
        source_data_ready_sha256=sha256(source/'DATA_READY.json'),data_revision='repair_v2',
        splits=splits,A_train_parents=train,A_validation_parents=val,B_train_parents=width_train,B_validation_parents=width_val,
        selection='A train: (N,UID) eight four-parent strata, first UID each; B within each A split: (N,UID) four two-parent strata, first UID each',
        axis='fixed canonical X; metadata does not certify per-object semantic left-right axis',
        coordinate_contract='one archived whole-object normalization; FP32 T_alpha on saved full_target/model_vertices/pre_ot_full, no renormalization; original pre_OT * 2 identity retained',
        candidate_rule='all retained source-face starts; sorted-neighbor FIFO BFS over complete source-edge adjacency; exact K; deduplicate source face sets; reject ambiguous or absent trusted interfaces',
        ranking_rule='min maximum Jaccard versus anchors; max minimum centroid distance; max minimum area-fraction distance; fixed domain hash tie; start ID',
        evaluation_reserved_before_new_train=True,tasks=[],availability=[],source_files_checked=checked)
    for split in ('train','validation','test'):
        for uid in sorted(splits[split]):
            base=_base(str(source),uid); topology=_topology(base)
            old_tasks=[]
            for row in sorted(old_by[uid],key=lambda x:x['patch_id']):
                # Recover every old C through its strict original loader before any selection.
                actual=_api(str(source)).load_sample(uid,row['patch_id'],source/'data')
                patch,reason,_=_patch(base,row['source_face_ids'],topology)
                if reason or not bits(actual['constraints'],base['full_target'][row['source_face_ids']].reshape(-1,3,3)):
                    raise ValueError('Existing 20% task cannot be recovered: '+uid+'/'+row['patch_id'])
                patch.update(N=base['N'],K=row['K'],ratio=.2,start_source_face_id=int(row['source_face_ids'][0]),old_patch_id=row['patch_id'])
                old_tasks.append(patch)
            pools={}; reasons={}; ambiguous={}
            for ratio in (.2,.4): pools[ratio],reasons[str(ratio)],ambiguous[str(ratio)]=_candidates(base,ratio,topology)
            excluded={_set(t) for t in old_tasks}; eval_tasks=[]; training=[]
            def add(patch,role,suffix,ratio):
                row=dict(task_id=uid+'_'+suffix,patch_id=uid+'_'+suffix,uid=uid,split=split,role=role,ratio=ratio,N=base['N'],K=int(np.floor(ratio*base['N']+.5)))
                if patch is None:
                    row.update(status='NOT_APPLICABLE',reason='NO_DISTINCT_TRUSTED_CONNECTED_EXACT_K_CANDIDATE_AFTER_RESERVATION',source_face_ids=[],free_source_ids=[])
                else:
                    row.update(patch,status='READY',reason=None)
                manifest['tasks'].append(row)
                return row
            if uid in train+val:
                for ratio,suffix in ((.2,'A20'),(.4,'A40')):
                    selected=_select(pools[ratio],excluded,old_tasks+eval_tasks)
                    task=add(selected,'A',suffix,ratio)
                    if selected is not None: excluded.add(_set(task)); eval_tasks.append(task)
            if uid in width_train+width_val:
                selected=_select(pools[.2],excluded,old_tasks+eval_tasks,width_L=base['bbox_diagonal'])
                task=add(selected,'B','Bwidth20',.2)
                if selected is not None: excluded.add(_set(task)); eval_tasks.append(task)
                elif any(_set(t) not in excluded for t in pools[.2]): task['reason']='NO_ELIGIBLE_WIDTH_PATCH_X_SPAN_OR_FREE_WIDTH_AFTER_RESERVATION'
            if split=='train':
                for i,patch in enumerate(old_tasks): training.append(add(patch,'train','train20_'+str(i),.2))
                for ratio,first,last in ((.2,4,8),(.4,0,8)):
                    for index in range(first,last):
                        selected=_select(pools[ratio],excluded,old_tasks+eval_tasks+training)
                        task=add(selected,'train','train'+str(int(ratio*100))+'_'+str(index),ratio)
                        if selected is not None: excluded.add(_set(task)); training.append(task)
            usable=[t for t in training if t['status']=='READY']
            if split=='train' and len(usable)<4: raise ValueError('Training parent lost old C: '+uid)
            old_sets=[_set(t) for t in old_tasks] if split=='train' else []
            train_sets=[_set(t) for t in usable]
            for task in manifest['tasks']:
                if task['uid']!=uid or task['status']!='READY': continue
                here=_set(task)
                task['overlap']=dict(max_old_training_jaccard=max((_jaccard(here,s) for s in old_sets),default=None),
                    max_new_training_jaccard=max((_jaccard(here,s) for s in train_sets),default=None),
                    parent_seen_in_conditional_training=split=='train')
                if task['role'] in ('A','B') and (here in old_sets or here in train_sets): raise ValueError('Evaluation C leaked into old/new train')
            manifest['availability'].append(dict(uid=uid,split=split,N=base['N'],original_N=base['original_N'],
                component_sizes=sorted(map(len,topology[3]),reverse=True),candidate_counts={str(k):len(v) for k,v in pools.items()},
                candidate_rejection_counts=reasons,ambiguous_candidate_records=ambiguous,
                old_conditions_recovered=len(old_tasks),train20_available=sum(t['ratio']==.2 for t in usable),
                train40_available=sum(t['ratio']==.4 for t in usable),training_parent_retained=split=='train'))
    manifest['counts']=dict(Counter(t['role']+'_'+t['status'] for t in manifest['tasks']))
    _json(frozen,manifest)
    _json(out/'registration/geometric_applicability.json',dict(status='PASS',parents=manifest['availability'],counts=manifest['counts'],model_forwards=0,OT_calls=0))
    _write_task_csv(out/'C_availability_overlap.csv',manifest)
    return manifest

def _write_task_csv(path,manifest):
    keys=['task_id','uid','split','role','ratio','N','K','status','reason','area_fraction','interface_edge_count','interface_length_model','C_X_span','max_old_training_jaccard','max_new_training_jaccard']
    rows=[]
    for task in manifest['tasks']:
        row={k:task.get(k) for k in keys}; m=task.get('metrics',{}); o=task.get('overlap',{})
        for key in ('area_fraction','interface_edge_count','interface_length_model'): row[key]=m.get(key,'NA')
        row['C_X_span']=m.get('C_span',['NA'])[0]
        for key in ('max_old_training_jaccard','max_new_training_jaccard'): row[key]=o.get(key) if o.get(key) is not None else 'NA'
        rows.append(row)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    import io
    stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    content=stream.getvalue()
    if path.exists():
        if path.read_bytes()!=content.encode(): raise ValueError('Frozen task CSV changed')
    else: path.write_bytes(content.encode())

def make_train_plan(manifest):
    parents=sorted(manifest['splits']['train']); tasks=defaultdict(lambda:defaultdict(list))
    for task in manifest['tasks']:
        if task['role']=='train' and task['status']=='READY': tasks[task['uid']][task['ratio']].append(task)
    rng=np.random.RandomState(seed('training','parent_schedule'))
    assignments=[parents[int(i)] for _ in range(750) for i in rng.permutation(32)]
    alpha_rng=np.random.RandomState(seed('training','alpha')); shuffle=np.random.RandomState(seed('training','alpha_slot_shuffle'))
    choice=np.random.RandomState(seed('training','task_schedule')); records=[]
    for batch in range(3000):
        alphas=np.r_[np.ones(4,np.float32),alpha_rng.uniform(.9,1.1,4).astype(np.float32)][shuffle.permutation(8)]
        for slot in range(8):
            index=8*batch+slot; uid=assignments[index]; ratios=sorted(tasks[uid]); ratio=ratios[int(choice.randint(len(ratios)))]; options=tasks[uid][ratio]
            task=options[int(choice.randint(len(options)))]; alpha=float(alphas[slot])
            records.append(dict(sample_index=index,effective_batch_index=batch,slot_index=slot,phase_step=batch+1,conditional_total_step=1001+batch,
                uid=uid,patch_id=task['task_id'],task_id=task['task_id'],N=task['N'],K=task['K'],ratio=ratio,alpha=alpha,
                gaussian_seed=seed('training','gaussian',index),t_seed=seed('training','time',index),permutation_seed=seed('training','permutation',index)))
    if any(sum(r['alpha']==1. for r in records[8*i:8*i+8])!=4 for i in range(3000)): raise ValueError('Alpha allocation differs')
    return dict(schema='chair_domain_train_plan_v1',experiment_id=EXPERIMENT,effective_batch_count=3000,batch_size=8,sample_count=24000,
        seed=seed('training','parent_schedule'),RNG_mode='stateless_per_sample_frozen_seeds',effective_batch_index_base=0,
        object_counts=dict(Counter(r['uid'] for r in records)),task_counts=dict(Counter(r['task_id'] for r in records)),
        object_ratio_counts=dict(Counter(r['uid']+'/'+str(r['ratio']) for r in records)),records=records)
def _source_checkpoints(source,out):
    historical=source.parent/'control_mvp48_budget_single_v1/checkpoint_manifest.json'
    old=read_json(historical)['checkpoints']
    models={}
    for arm in ('T1','Geo','Full'):
        path=source/'checkpoints'/(arm+'_step1000.pt'); side=read_json(path.with_suffix('.json'))
        digest=sha256(path); meta=side['metadata']
        matches=[r for r in old if r['path']==str(path)]
        if digest!=side['checkpoint_sha256'] or len(matches)!=1 or matches[0]['checkpoint_sha256']!=digest or matches[0]['metadata']!=meta:
            raise ValueError('Source1000 checkpoint disagrees with protected budget manifest: '+arm)
        if meta['logical_step']!=1000 or meta['variant']!=arm or meta['context_encoder']!=(None if arm=='T1' else 'geo') or meta['loss_recipe']!=('edge5' if arm=='Full' else 'fm'):
            raise ValueError('Invalid strict source1000 model identity')
        for field in ('model_state_sha256','optimizer_state_sha256','rng_sha256','stream_state_sha256'):
            if len(meta[field])!=64: raise ValueError('Incomplete recovery identity')
        models[arm]=dict(path=str(path),sha256=digest,metadata=meta,sidecar_sha256=sha256(path.with_suffix('.json')))
    value=dict(status='PASS',source1000=models,old_budget_checkpoint_manifest_sha256=sha256(historical),
        checkpoint_tensor_loading='NOT_RUN_CPU_PREPARATION; strict state verification belongs to authorized model load',
        old5000_used_for_initialization=False)
    _json(out/'registration/source_checkpoint_identity.json',value)
    return value

def prepare(source,out):
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source in out.parents: raise ValueError('Output must be separate from source')
    out.mkdir(parents=True,exist_ok=True)
    manifest=geometry_manifest(source,out)
    source_identity=_source_checkpoints(source,out)
    if (out/'panels.json').exists():
        panels=read_json(out/'panels.json')
        for relative,digest in panels['frozen_files'].items():
            path=Path(relative) if Path(relative).is_absolute() else out/relative
            if sha256(path)!=digest: raise ValueError('Frozen preparation input changed: '+relative)
        if read_json(out/'registration/data_audit.json')['status']!='PASS': raise ValueError('Prior data audit not PASS')
        return panels
    from experiments.mf_h1_local0.control_mvp48.eval_scoring import prepare_reference_from_sample,score_free,sample_mesh_surface
    from native_t1.artifacts import array_hash
    started=time.perf_counter()
    panels=dict(schema='chair_domain_panels_v1',source=str(source),source_data_ready_sha256=manifest['source_data_ready_sha256'],
        data_revision='repair_v2',A=[],B=[],C=[],teacher=[],frozen_files={},
        random_domains=['initialization','training','teacher','generation_A','generation_B','control_diagnostic','plotting','dispatch'],
        A_train_parents=manifest['A_train_parents'],A_validation_parents=manifest['A_validation_parents'],
        B_train_parents=manifest['B_train_parents'],B_validation_parents=manifest['B_validation_parents'])
    def freeze(relative,*,arrays=None,value=None):
        path=out/relative
        if arrays is not None: _npz(path,arrays)
        else: _json(path,value)
        panels['frozen_files'][relative]=sha256(path)
        return relative
    for name in ('task_manifest.json','registration/geometric_applicability.json','C_availability_overlap.csv','registration/source_checkpoint_identity.json'):
        panels['frozen_files'][name]=sha256(out/name)
    freeze('train_plan.json',value=make_train_plan(manifest))
    events=out/'registration/prepare_events.jsonl'
    def event(action,**data):
        with events.open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(action=action,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**data),sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
    references={}
    def reference(task_id,alpha):
        key=(task_id,float(alpha)); tag=str(int(round(float(alpha)*100)))
        if key not in references:
            stem='inputs/references/'+task_id+'_a'+tag
            if (out/(stem+'.npz')).exists() and (out/(stem+'.json')).exists():
                ref=dict(arrays={k:v.copy() for k,v in np.load(out/(stem+'.npz'),allow_pickle=False).items()},audit=read_json(out/(stem+'.json')))
            else:
                item=load_task_sample(source,manifest,task_id,alpha)
                ref=prepare_reference_from_sample(item)
                freeze(stem+'.npz',arrays=ref['arrays']); freeze(stem+'.json',value=ref['audit'])
                event('reference_prepared',task_id=task_id,alpha=float(alpha),reference_only=True)
            references[key]=dict(ref_npz=freeze(stem+'.npz',arrays=ref['arrays']),ref_json=freeze(stem+'.json',value=ref['audit']))
        return references[key]
    def reference_value(record):
        with np.load(out/record['ref_npz'],allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
        return dict(arrays=arrays,audit=read_json(out/record['ref_json']))
    def make_generation(task,alpha,noise_index,table):
        tag=str(int(round(alpha*100))); case_id=table+'_'+task['task_id']+'_a'+tag+'_n'+str(noise_index)
        phases=[3000]
        if table=='A' and task['ratio']==.2 and noise_index==0: phases=[0,1000,2500,3000] if task['split']=='validation' else [0,3000]
        if table=='B' and noise_index==0: phases=[0,3000]
        record={key:task[key] for key in ('uid','task_id','patch_id','split','N','K','ratio','status','reason')}
        record.update(case_id=case_id,parent_id=task['uid'],table=table,alpha=alpha,noise_index=noise_index,phase_steps=phases,save_states=False)
        if task['status']!='READY': return record
        sample=load_task_sample(source,manifest,task['task_id'],alpha); n,k=sample['N'],sample['K']
        noise_seed=seed('generation_'+table,task['uid'],noise_index)
        noise=_api(str(source)).gaussian(noise_seed,n-k,fp32=True)
        z=np.zeros((1,n,9),np.float32); z[:,k:]=noise.reshape(1,n-k,9)
        arrays=dict(C=sample['constraints'],gaussian=noise,z=z,N=np.asarray(n),K=np.asarray(k),L=np.asarray(sample['bbox_diagonal']),
            L0=np.asarray(sample['bbox_L0']),known_mask=np.arange(n)<k,valid_mask=np.ones(n,bool),full_target=sample['full_target'],
            gt_free=sample['free_target'],source_face_ids=sample['source_face_ids'],free_source_ids=sample['free_source_ids'])
        path=freeze('inputs/generation/'+case_id+'.npz',arrays=arrays)
        record.update(input_path=path,input_sha256=sha256(out/path),gaussian_seed=noise_seed,L=sample['bbox_diagonal'],L0=sample['bbox_L0'],
            C_area_fraction=_area(sample['constraints'])/_area(sample['full_target']),overlap=task['overlap'],**reference(task['task_id'],alpha))
        return record
    for task in manifest['tasks']:
        if task['role']=='A':
            for noise in (0,1): panels['A'].append(make_generation(task,1.,noise,'A'))
        if task['role']=='B':
            original=load_task_sample(source,manifest,task['task_id'],1.) if task['status']=='READY' else None
            point_path='inputs/control_points/'+task['task_id']+'_base.npz'
            if original is not None:
                if (out/point_path).exists():
                    with np.load(out/point_path,allow_pickle=False) as f: points0=f['points0'].copy()
                else:
                    points0=sample_mesh_surface(original['free_target'].reshape(-1,3,3),512,np.random.RandomState(seed('control_diagnostic',task['task_id'])))
                    if points0 is None: raise ValueError('Control GT point bank failed')
                    event('control_diagnostic_point_sample',task_id=task['task_id'],point_count=512)
                freeze(point_path,arrays=dict(points0=points0,L0=np.asarray(original['bbox_L0'])))
            for alpha in (.9,1.,1.1):
                for noise in (0,1):
                    record=make_generation(task,alpha,noise,'B')
                    if original is not None:
                        points=points0.copy(); points[:,0]*=np.float32(alpha)
                        path=freeze('inputs/control_points/'+task['task_id']+'_a'+str(int(round(alpha*100)))+'.npz',arrays=dict(points0=points0,points=points,L0=np.asarray(original['bbox_L0'])))
                        record.update(control_points_path=path,control_points_sha256=sha256(out/path),control_points_scope='512 original GT free-area samples transported under T_alpha; not new area-uniform variant samples')
                    panels['B'].append(record)
    old=read_json(source/'registration/eval_preparation_ledger.json'); conditions={(r['parent_id'],r['condition_id']):r for r in old['conditions']}
    for request in old['test_inputs']:
        path=source/request['path']; ref=conditions[(request['parent_id'],request['condition_id'])]; ref_path=source/ref['path']
        if sha256(path)!=request['sha256'] or sha256(ref_path)!=ref['sha256']: raise ValueError('Original Session C actual bank changed')
        panels['frozen_files'][str(path)]=request['sha256']; panels['frozen_files'][str(ref_path)]=ref['sha256']
        meta=freeze('inputs/holdout_metadata/'+request['case_id']+'.json',value=ref['reference_metadata'])
        panels['C'].append(dict(request,table='C',uid=request['parent_id'],task_id=request['parent_id']+'_old_'+request['condition_id'],
            patch_id=request['condition_id'],split='holdout',noise_index=request['noise_id'],alpha=1.,ratio=.2,status='READY',reason=None,
            phase_steps=[3000],save_states=False,input_path=str(path),input_sha256=request['sha256'],ref_npz=str(ref_path),ref_json=meta,
            reference_key_prefix='reference_',reference_contains_geometry=True,target_path=str(ref_path),source_case_id=request['case_id']))
    for table in ('A','B','C'):
        first=next((r for r in panels[table] if r['status']=='READY'),None)
        if first: first['save_states']=True
    # CPU tests are finite, declared before any real-data score call, and resumable by actual artifacts.
    control_uids=[sorted(manifest['B_train_parents'])[0],sorted(manifest['B_validation_parents'])[0]]
    cpu_spec=dict(schema='chair_domain_cpu_positive_spec_v1',parents=control_uids,alphas=[.9,1.,1.1],
        candidates=['perfect_reference','no_response'],score_calls=12,include_pair_audit=False,perfect_RMS_over_L_max=1e-6,
        post_scale='zero-forward direct affine width identity only; no additional score call',model_forwards=0,OT_calls=0)
    freeze('registration/cpu_positive_spec.json',value=cpu_spec)
    control_results=[]
    for uid in control_uids:
        rows=[r for r in panels['B'] if r['uid']==uid and r['noise_index']==0 and r['status']=='READY']
        if len(rows)!=3: raise ValueError('Predeclared finite CPU control parent unavailable')
        original=load_task_sample(source,manifest,rows[0]['task_id'],1.); width0=float(np.diff(np.quantile(original['free_target'].reshape(-1,3)[:,0],[.05,.95]))[0])
        for record in rows:
            alpha=record['alpha']; sample=load_task_sample(source,manifest,record['task_id'],alpha)
            target_width=float(np.diff(np.quantile(sample['free_target'].reshape(-1,3)[:,0],[.05,.95]))[0]); L0=sample['bbox_L0']
            if not np.isclose(target_width,alpha*width0,rtol=2e-6,atol=1e-7*L0): raise ValueError('GT width affine identity failed')
            if not bits(sample['full_target'].reshape(-1,3)[:,1:],original['full_target'].reshape(-1,3)[:,1:]): raise ValueError('Y/Z changed under X control')
            x1=original['constraints'][...,0].astype(np.float64); xa=sample['constraints'][...,0].astype(np.float64)
            alpha_hat=float(np.sum(x1*xa)/np.sum(x1*x1)); post=original['free_target'].reshape(-1,3,3).copy(); post[...,0]*=np.float32(alpha_hat)
            post_width=float(np.diff(np.quantile(post.reshape(-1,3)[:,0],[.05,.95]))[0])
            if not np.isclose(post_width,target_width,rtol=2e-6,atol=1e-7*L0): raise ValueError('Direct post-scale control width failed')
            for candidate in ('perfect_reference','no_response'):
                result_path='registration/cpu_controls/'+uid+'_a'+str(int(round(alpha*100)))+'_'+candidate+'.json'
                attempt_path=out/(result_path+'.attempt.json')
                if (out/result_path).exists(): result=read_json(out/result_path)
                else:
                    if attempt_path.exists(): raise ValueError('Interrupted CPU score attempt requires evidence review; no silent repeat: '+result_path)
                    _json(attempt_path,dict(action='CPU_score_attempt',uid=uid,alpha=alpha,candidate=candidate))
                    event('CPU_positive_score_attempt',uid=uid,alpha=alpha,candidate=candidate)
                    predicted=sample['free_target'] if candidate=='perfect_reference' else original['free_target']
                    result=score_free(predicted,sample['constraints'],sample['free_target'],sample['bbox_diagonal'],reference=reference_value(record),include_pair_audit=False)
                    freeze(result_path,value=result); event('CPU_positive_score_return',uid=uid,alpha=alpha,candidate=candidate)
                freeze(result_path,value=result)
                rms=result['metrics']['remainder_symmetric_RMS_bbox_pct']/100.
                measured_width=target_width if candidate=='perfect_reference' else width0
                if result['status']!='PASS' or not np.isfinite(rms): raise ValueError('CPU control scoring failed')
                if candidate=='perfect_reference' and rms>1e-6: raise ValueError('Perfect CPU reference not near zero')
                if candidate=='no_response' and alpha!=1. and (abs(measured_width-target_width)/L0<=1e-7 or rms<=1e-7): raise ValueError('No-response control not detected')
                control_results.append(dict(uid=uid,alpha=alpha,candidate=candidate,RMS_over_L=rms,width=measured_width,target_width=target_width,
                    absolute_width_error_over_L0=abs(measured_width-target_width)/L0,post_scale_alpha_hat=alpha_hat,post_scale_width=post_width,result_path=result_path))
    # Eight unique teacher pairings, reused across three t and every model/checkpoint.
    teacher_choices=[]
    for split,uids in (('train',manifest['B_train_parents']),('validation',manifest['B_validation_parents'])):
        for uid in uids:
            role='train' if split=='train' else 'A'
            task=sorted([t for t in manifest['tasks'] if t['uid']==uid and t['role']==role and t['ratio']==.2 and t['status']=='READY'],key=lambda t:t['task_id'])[0]
            teacher_choices.append(dict(uid=uid,split=split,task_id=task['task_id']))
    freeze('registration/teacher_spec.json',value=dict(selected=teacher_choices,t=[.2,.5,.8],alpha=1.,unique_OT_budget=8,noise_domain='teacher'))
    from experiments.mf_h1_local0.control_mvp48.eval_teacher import prepare_teacher_sample
    for chosen in teacher_choices:
        uid,task_id,split=chosen['uid'],chosen['task_id'],chosen['split']; sample=load_task_sample(source,manifest,task_id,1.)
        stem='inputs/teacher_pairs/'+task_id; pnpz,pjson=out/(stem+'.npz'),out/(stem+'.json'); attempt=out/(stem+'.attempt.json')
        if pnpz.exists() and pjson.exists():
            identity=read_json(pjson)
            with np.load(pnpz,allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
            if any(array_hash(arrays[k])!=digest for k,digest in identity['array_sha256'].items()): raise ValueError('Teacher OT cache identity changed')
        else:
            if attempt.exists(): raise ValueError('Interrupted teacher OT; no silent repeat: '+task_id)
            _json(attempt,dict(task_id=task_id,action='teacher_OT_attempt'))
            event('teacher_OT_attempt',task_id=task_id,uid=uid)
            arrays,identity=prepare_teacher_sample(sample,_api(str(source)).gaussian(seed('teacher',task_id),sample['N']-sample['K']))
            _npz(pnpz,arrays); _json(pjson,identity); event('teacher_OT_return',task_id=task_id,uid=uid)
        freeze(stem+'.npz',arrays=arrays); freeze(stem+'.json',value=identity)
        for value in (.2,.5,.8):
            current={k:v.copy() for k,v in arrays.items()}; current['t']=np.asarray(value,np.float32)
            current['xt']=(np.float32(1)-current['t'])*current['x0']+current['t']*current['x1']; current['xt'][current['known_mask']]=current['context'][current['known_mask']]
            current['free_vertex_ids']=sample['source_face_vertex_ids'][sample['free_source_ids']].copy(); current['known_vertex_ids']=sample['source_face_vertex_ids'][sample['source_face_ids']].copy()
            case='teacher_'+task_id+'_t'+str(int(round(value*10))); path=freeze('inputs/teacher/'+case+'.npz',arrays=current)
            info=dict(identity,teacher_t=value,array_sha256={k:array_hash(v) for k,v in current.items()})
            info.update(model_input_sha256=hashlib.sha256(b''.join(current[k].tobytes() for k in ('xt','t','y','known_mask','valid_mask'))).hexdigest(),input_hash_definition='sha256 concatenated xt,t,y,known_mask,valid_mask bytes')
            panels['teacher'].append(dict(case_id=case,teacher_id=case,uid=uid,parent_id=uid,task_id=task_id,patch_id=task_id,split=split,alpha=1.,t=value,
                N=sample['N'],K=sample['K'],status='READY',phase_steps=[0,1000,2500,3000],input_path=path,input_sha256=sha256(out/path),identity=info,**reference(task_id,1.)))
    expected=dict(A=64,B=48,C=32,teacher=24)
    if {k:len(panels[k]) for k in expected}!=expected: raise ValueError('Request denominator mismatch')
    active=sum(len(row['phase_steps']) for table in ('A','B','C') for row in panels[table] if row['status']=='READY')*3
    total=sum(len(row['phase_steps']) for table in ('A','B','C') for row in panels[table])*3
    if total!=600 or active>600: raise ValueError('Frozen stage matrix exceeds generation budget')
    events_read=[json.loads(line) for line in events.read_text().splitlines() if line.strip()]
    counts=dict(Counter(x['action'] for x in events_read))
    if counts.get('teacher_OT_attempt',0)!=8 or counts.get('teacher_OT_return',0)!=8 or counts.get('CPU_positive_score_attempt',0)!=12 or counts.get('CPU_positive_score_return',0)!=12:
        raise ValueError('Finite CPU budget/attempt accounting mismatch')
    audit=dict(schema='chair_domain_data_audit_v1',status='PASS',source_data_ready_sha256=manifest['source_data_ready_sha256'],source_checkpoints=source_identity,
        parents_retained=48,train_parents_retained=32,old_C_recovered=160,task_counts=manifest['counts'],
        generation_requests=600,applicable_generation_requests=active,not_applicable_generation_requests=600-active,
        finite_CPU_positive_results=control_results,preparation_counts=counts,teacher_unique_OT=8,teacher_states=24,
        fixed_X_axis_no_semantic_claim=True,Gaussian_never_transformed=True,model_forwards=0,optimizer_updates=0,CUDA_initialized=False,
        GT_post_scale_reference_only=True,completed_seconds=time.perf_counter()-started)
    freeze('registration/data_audit.json',value=audit)
    panels.update(counts=dict(requested_generation=600,applicable_generation=active,not_applicable_generation=600-active,teacher_states=24,teacher_forwards=288),preparation_counts=counts)
    _json(out/'panels.json',panels)
    _json(out/'preparation_manifest.json',dict(status='PASS',panels_sha256=sha256(out/'panels.json'),task_manifest_sha256=sha256(out/'task_manifest.json'),train_plan_sha256=sha256(out/'train_plan.json'),counts=panels['counts']))
    return panels

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--source',required=True); parser.add_argument('--out',required=True); parser.add_argument('--geometry-only',action='store_true')
    args=parser.parse_args()
    result=geometry_manifest(args.source,args.out) if args.geometry_only else prepare(args.source,args.out)
    print(json.dumps(result.get('counts',{}),sort_keys=True))