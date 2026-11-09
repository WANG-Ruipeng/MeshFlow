"""Authorized minimal source-derived cleanup for Control MVP48 repair_v2.

Preserves strict-v1 evidence and official sources. No model, OT or CUDA work.
Prepare then independently audit, then freeze. No selection repair shortcuts.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('control_data_base',HERE/'data_prepare.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
np=b.np
from scipy.spatial import cKDTree
RUN,REG,DATA,REPO,SOURCE=b.RUN,b.REG,b.DATA,b.REPO,b.SOURCE


def clean(raw):
    v=raw['vertices']; faces=raw['faces']; tri=v[faces].astype(np.float64)
    cross=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    zero=np.flatnonzero(cross==0).tolist();keep=[];dups=[];representatives=[];seen={}
    for i,t in enumerate(tri):
        if cross[i]==0:continue
        key=tuple(sorted(tuple(float(c) for c in point) for point in t))
        if key in seen:dups.append(i);representatives.append(seen[key])
        else:seen[key]=i;keep.append(i)
    kept=np.asarray(keep,np.int64);inverse=np.full(len(faces),-1,np.int64);inverse[kept]=np.arange(len(kept))
    repaired=dict(raw,faces=faces[kept].copy(),faces_num=np.asarray(len(kept),dtype=np.int64))
    info=dict(original_N=len(faces),retained_N=len(kept),retained_original_face_ids=keep,
        removed_zero_original_face_ids=zero,removed_duplicate_original_face_ids=dups,
        duplicate_representative_original_face_ids=representatives,
        repair_applied=bool(zero or dups),vertices_modified=False,vertices_removed=False,
        positive_area_surface_union_preserved=True,operation_order='exact_zero_then_exact_duplicate_min_original_id')
    return repaired,kept,inverse,info


def patches(vertices,faces,original_faces,retained_to_original):
    n=len(faces);k=int(np.floor(.2*n+.5));centers=vertices[faces].astype(np.float64).mean(1)
    edges,adj,topo=b.topology(faces);orig_edges,_,orig_topo=b.topology(original_faces)
    remaining=set(range(n));attempts=[];out=[];sets=set();seedcoords=set();rejected=Counter()
    mindist=((centers-centers.mean(0))**2).sum(1)
    while remaining and len(out)<4:
        start=min(remaining,key=lambda i:(-float(mindist[i]),i));remaining.remove(start)
        dist=((centers-centers[start])**2).sum(1)
        mindist=dist if not attempts else np.minimum(mindist,dist)
        queue=deque([start]);seen={start};known=[]
        while queue and len(known)<k:
            f=queue.popleft();known.append(f)
            for other in adj[f]:
                if other not in seen:seen.add(other);queue.append(other)
        reason=None;interface=[];ks=set(known)
        if len(known)!=k:reason='component_smaller_than_K'
        else:
            for edge,members in sorted(edges.items()):
                inside=[f for f in members if f in ks];outside=[f for f in members if f not in ks]
                if inside and outside:
                    if len(members)!=2 or len(inside)!=1 or len(outside)!=1:
                        reason='nonmanifold_retained_crossing';break
                    originals=orig_edges[edge]
                    expected=sorted(int(retained_to_original[f]) for f in members)
                    if len(originals)!=2 or sorted(originals)!=expected:
                        reason='abnormal_original_source_crossing';break
                    interface.append((edge,inside[0],outside[0]))
            if reason is None and not interface:reason='empty_known_free_interface'
            if reason is None and tuple(sorted(known)) in sets:reason='duplicate_condition_source_set'
            if reason is None and tuple(centers[start]) in seedcoords:reason='duplicate_start_centroid'
        attempts.append(dict(start_source_face_id=start,start_original_face_id=int(retained_to_original[start]),rejection_reason=reason))
        if reason:rejected[reason]+=1;continue
        sets.add(tuple(sorted(known)));seedcoords.add(tuple(centers[start]))
        out.append(dict(start_source_face_id=start,known=known,free=[i for i in range(n) if i not in ks],interface=interface))
    return out,dict(topology=topo,original_topology=orig_topo,attempts=attempts,
        rejected_patch_candidates=dict(rejected),accepted_patch_count=len(out),K=k)


def inspect_sources():
    original=b.read_json(DATA/'candidate_audit.json');history=b.read_json(REG/'data_historical_audit.json')
    originals={x['uid']:x for x in original['objects']};historical=set(history['historical_uids'])
    raws={};derived={};maps={};inverse={};infos={};rows=[];allpatch={}
    exact=defaultdict(list);normalized=defaultdict(list);near_data={}
    for i,uid in enumerate(sorted(originals)):
        rec=originals[uid];path=REPO/rec['source_path']
        if b.sha(path)!=rec['source_sha256']:raise RuntimeError('Original source hash changed: '+uid)
        with np.load(path,allow_pickle=False) as z:
            raw={k:z[k].copy() for k in ('vertices','faces','uid','faces_num')}
        v,f=raw['vertices'],raw['faces'];raw_n=len(f)
        if str(raw['uid'].item())!=uid or raw_n!=rec['N'] or not np.isfinite(v).all():raise ValueError('Invalid raw source '+uid)
        fixed,keep,inv,info=clean(raw);n=len(keep);extent=float(np.ptp(v,axis=0).max())
        reasons=[]
        if not 128<=raw_n<=256:reasons.append('original_face_count_outside_128_256')
        if not 128<=n<=256:reasons.append('retained_face_count_outside_128_256')
        if extent<=0:reasons.append('zero_scale')
        cross=np.linalg.norm(np.cross(v[fixed['faces']][:,1].astype(np.float64)-v[fixed['faces']][:,0],
                                     v[fixed['faces']][:,2].astype(np.float64)-v[fixed['faces']][:,0]),axis=1)
        tiny=int((cross<=1e-12*extent**2).sum())
        if tiny:reasons.append('retained_small_nonzero_area_faces')
        gh=b.geometry_hash(v[fixed['faces']])
        nv=(v.astype(np.float64)-(v.min(0).astype(np.float64)+v.max(0).astype(np.float64))/2)/extent
        nh=b.geometry_hash(np.round(nv[fixed['faces']],6))
        exact[gh].append(uid);normalized[nh].append(uid)
        row=dict(uid=uid,official_split=rec['official_split'],source_path=rec['source_path'],source_sha256=rec['source_sha256'],
            original_N=raw_n,N=n,source_vertex_count=len(v),raw_exact_geometry_sha256=rec['exact_geometry_sha256'],
            exact_geometry_sha256=gh,normalized_rounded6_geometry_sha256=nh,repair=info,
            retained_small_nonzero_area_face_count=tiny,rejection_reasons=reasons)
        if not reasons:
            pp,pa=patches(v,fixed['faces'],f,keep);allpatch[uid]=pp;row['patch_audit']=pa
            if len(pp)<2:reasons.append('fewer_than_2_applicable_connected_patches')
        if (128<=raw_n<=256 or uid in historical) and n>0:
            points=np.unique(nv[fixed['faces']].reshape(-1,3),axis=0);centers=nv[fixed['faces']].mean(1)
            near_data[uid]=(points,centers,cKDTree(points),cKDTree(centers))
        raws[uid]=raw;derived[uid]=fixed;maps[uid]=keep;inverse[uid]=inv;infos[uid]=info;rows.append(row)
        if i%500==0:print('repair scanned',i,flush=True)
    uf=b.Union(raws)
    for members in original['grouping']['groups'].values():
        for u in members[1:]:uf.merge(members[0],u)
    for groups in (exact,normalized):
        for members in groups.values():
            for u in members[1:]:uf.merge(members[0],u)
    near_edges=[];pairs=0;bboxpass=0;us=sorted(near_data)
    bounds={u:(a[0].min(0),a[0].max(0),a[1].min(0),a[1].max(0)) for u,a in near_data.items()}
    for i,u in enumerate(us):
        for w in us[i+1:]:
            pairs+=1
            if max(float(np.abs(a-c).max()) for a,c in zip(bounds[u],bounds[w]))>.01:continue
            bboxpass+=1;a,c=near_data[u],near_data[w]
            distances=[]
            for pts,tree in ((a[0],c[2]),(c[0],a[2]),(a[1],c[3]),(c[1],a[3])):
                ds=tree.query(pts,distance_upper_bound=.010000000000001,workers=1)[0]
                if not np.isfinite(ds).all():break
                distances.append(float(ds.max()))
            if len(distances)==4 and max(distances)<=.01:
                near_edges.append(dict(uids=[u,w],vertex_hausdorff=max(distances[:2]),centroid_hausdorff=max(distances[2:])))
                uf.merge(u,w)
    groups=defaultdict(list)
    for uid in sorted(raws):groups[uf.find(uid)].append(uid)
    exposed={uf.find(u) for u in historical}
    for r in rows:r.update(group_id=uf.find(r['uid']),historically_exposed_group=uf.find(r['uid']) in exposed)
    counts={}
    for split in ('train','test'):
        rs=[r for r in rows if r['official_split']==split]
        valid=[r for r in rs if not r['rejection_reasons']]
        counts[split]=dict(all=len(rs),original_in_range=sum(128<=r['original_N']<=256 for r in rs),
            both_counts_in_range=sum(128<=r['original_N']<=256 and 128<=r['N']<=256 for r in rs),
            applicable_2patch=len(valid),applicable_4patch=sum(len(allpatch[r['uid']])>=4 for r in valid),
            unexposed_applicable_groups=len({r['group_id'] for r in valid if not r['historically_exposed_group']}),
            rejection_reasons=dict(Counter(k for r in rs for k in r['rejection_reasons'])))
    audit=dict(schema='control_mvp48_repair_candidate_audit_v2',experiment_id=b.EXPERIMENT,data_revision='repair_v2',
        protocol_sha256=b.sha(REG/'data_repair_protocol.json'),original_audit_sha256=b.sha(DATA/'candidate_audit.json'),
        source_split_files=original['source_split_files'],objects=rows,counts_by_filter=counts,
        grouping=dict(groups=dict(groups),inherited_all_original_groups=True,
            exact_retained_groups=[x for x in exact.values() if len(x)>1],
            normalized_rounded6_retained_groups=[x for x in normalized.values() if len(x)>1],
            near_scope_uids=us,near_pairs_tested=pairs,near_bbox_prefilter_pass=bboxpass,near_edges=near_edges,
            historically_exposed_group_ids=sorted(exposed),limitations=b.read_json(REG/'data_repair_protocol.json')['grouping']))
    return raws,derived,maps,inverse,infos,allpatch,audit


def choose(audit,candidates):
    used=set();chosen={};short=[];protocol=b.read_json(REG/'data_protocol.json')
    def rank(role,u):return b.hashlib.sha256(f"{protocol['selection_seed']}|{role}|{u}".encode()).hexdigest()
    for role,official,count,npatch in [('test','test',8,2),('validation','train',8,2),('train','train',32,4)]:
        pool=defaultdict(list)
        for r in audit['objects']:
            if r['official_split']!=official or r['rejection_reasons'] or r['group_id'] in used:continue
            if role!='train' and r['historically_exposed_group']:continue
            if len(candidates.get(r['uid'],[]))<npatch:continue
            pool[r['group_id']].append(r['uid'])
        gs=sorted(pool,key=lambda g:(rank(role,g),g));chosen[role]=[min(pool[g],key=lambda u:(rank(role,u),u)) for g in gs[:count]]
        used.update(gs[:count])
        if len(gs)<count:short.append(dict(role=role,required=count,available_groups=len(gs),shortfall=count-len(gs)))
    if short:
        b.write_json(DATA/'repair_selection_shortfall.json',dict(shortages=short,partial_selection=chosen))
        raise RuntimeError('Repair still insufficient: '+json.dumps(short))
    rows={r['uid']:r for r in audit['objects']}
    return dict(schema='control_mvp48_splits_v1',experiment_id=b.EXPERIMENT,data_revision='repair_v2',
        selection_seed=protocol['selection_seed'],**chosen,
        group_ids={u:rows[u]['group_id'] for us in chosen.values() for u in us},
        official_splits={u:rows[u]['official_split'] for us in chosen.values() for u in us},historical_exclusion_applied=True)


def emit(uid,original,raw,keep,inv,info,candidates,count,meta):
    arrays=b.preprocess(raw);N=len(raw['faces']);K=int(np.floor(.2*N+.5))
    arrays.update(original_source_faces=original['faces'],original_N=np.asarray(len(original['faces']),np.int64),
        retained_to_original_face=keep,original_to_retained_face=inv,
        repair_metadata_json=np.asarray(json.dumps(info,sort_keys=True)))
    target=arrays['full_target'].reshape(-1,3,3);vid=arrays['source_face_vertex_ids'];records=[]
    rawtri=raw['vertices'][raw['faces']].astype(np.float64)
    area=np.linalg.norm(np.cross(rawtri[:,1]-rawtri[:,0],rawtri[:,2]-rawtri[:,0]),axis=1)/2
    selected=candidates[:count]
    overlap=[[len(set(a['known'])&set(c['known']))/K for c in selected] for a in selected]
    jaccard=[[len(set(a['known'])&set(c['known']))/len(set(a['known'])|set(c['known'])) for c in selected] for a in selected]
    for i,p in enumerate(selected):
        label=f'p{i}';edges=[];knownf=[];freef=[];kc=[];fc=[]
        for edge,k,f in p['interface']:
            edges.append(list(edge));knownf.append(k);freef.append(f)
            kc.append([int(np.flatnonzero(vid[k]==v)[0]) for v in edge]);fc.append([int(np.flatnonzero(vid[f]==v)[0]) for v in edge])
        edgeids=np.asarray(edges,np.int64);me=arrays['model_vertices'][edgeids].astype(np.float64);re=raw['vertices'][edgeids].astype(np.float64)
        arrays[label+'_constraints']=target[p['known']].copy()
        records.append(dict(uid=uid,patch_id=label,N=N,K=K,original_N=len(original['faces']),
            source_face_ids=p['known'],free_source_ids=p['free'],original_source_face_ids=keep[p['known']].tolist(),
            original_free_source_ids=keep[p['free']].tolist(),start_source_face_id=p['start_source_face_id'],
            start_original_face_id=int(keep[p['start_source_face_id']]),constraints_array_key=label+'_constraints',
            constraints_sha256=b.ah(arrays[label+'_constraints']),
            interface=dict(edge_source_vertex_ids=edges,known_source_face_ids=knownf,free_source_face_ids=freef,
                known_original_source_face_ids=keep[knownf].tolist(),free_original_source_face_ids=keep[freef].tolist(),
                known_corner_ids=kc,free_corner_ids=fc),
            patch_metrics=dict(K_over_N=K/N,area_fraction=float(area[p['known']].sum()/area.sum()),
                interface_edge_count=len(edges),interface_length_model=float(np.linalg.norm(me[:,1]-me[:,0],axis=1).sum()),
                interface_length_raw=float(np.linalg.norm(re[:,1]-re[:,0],axis=1).sum()),
                overlap_intersection_over_K=overlap[i],overlap_jaccard=jaccard[i],
                interface_applicable=True,original_degree2_interface=True,known_connected_by_complete_source_edges=True)))
    path=DATA/'meshes'/f'{uid}.npz';b.save_npz(path,arrays)
    ob={k:meta[k] for k in ('uid','official_split','source_path','source_sha256','original_N','N','source_vertex_count','group_id','historically_exposed_group','exact_geometry_sha256','raw_exact_geometry_sha256')}
    ob.update(prepared_path=str(path.relative_to(RUN)),prepared_sha256=b.sha(path),prepared_bytes=path.stat().st_size,
        K=K,free=N-K,patch_count=count,repair=info,topology=meta['patch_audit']['topology'],
        original_topology=meta['patch_audit']['original_topology'],
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=b.ah(v)) for k,v in arrays.items()})
    return records,ob


def prepare():
    if (RUN/'DATA_READY.json').exists():raise RuntimeError('Already frozen; repair changes prohibited')
    protocol=b.read_json(REG/'data_repair_protocol.json');contract=b.read_json(REG/'data_contract_repair_v2.json')
    if protocol['experiment_id']!=b.EXPERIMENT or contract['data_revision']!='repair_v2':raise RuntimeError('Repair identity mismatch')
    before={str(p.relative_to(REPO)):b.sha(p) for p in [REPO/'datasets/mesh_dataset.py',REPO/'native_t1/prepare.py',
        REPO/'native_t1/data.py',REPO/'utils/ot_utils.py',REG/'data_repair_protocol.json',REG/'data_contract_repair_v2.json']}
    tick=time.perf_counter();raws,derived,maps,inverse,infos,candidates,audit=inspect_sources();scan_sec=time.perf_counter()-tick
    b.write_json(DATA/'repair_candidate_audit.json',audit)
    if not (REG/'data_repair_scan_runtime.json').exists():b.write_json(REG/'data_repair_scan_runtime.json',dict(scan_seconds=scan_sec,gpu_workers=0,cpu_workers=1,cpu_threads=1))
    splits=choose(audit,candidates);b.write_json(DATA/'splits.json',splits)
    rows={r['uid']:r for r in audit['objects']};records=[];objects=[];normal_start=time.perf_counter()
    for role,count in [('train',4),('validation',2),('test',2)]:
        for u in splits[role]:
            p,o=emit(u,raws[u],derived[u],maps[u],inverse[u],infos[u],candidates[u],count,rows[u]);o['split']=role;objects.append(o);records.extend(p)
    norm_sec=time.perf_counter()-normal_start
    b.write_json(DATA/'patches.json',dict(schema='control_mvp48_patches_v1',experiment_id=b.EXPERIMENT,data_revision='repair_v2',records=records))
    b.plans(splits,objects)
    for uid,r in rows.items():
        if b.sha(REPO/r['source_path'])!=r['source_sha256']:raise RuntimeError('Original source changed during repair: '+uid)
    for path,hash_ in before.items():
        if b.sha(REPO/path)!=hash_:raise RuntimeError('Preprocessing/contract changed during repair: '+path)
    m=dict(schema='control_mvp48_manifest_v1',experiment_id=b.EXPERIMENT,data_revision='repair_v2',status='PREPARED_AWAITING_INDEPENDENT_AUDIT',
        object_count=48,objects=objects,source_code_and_registration=before,official_split_files=audit['source_split_files'],
        contract=dict(path='registration/data_contract_repair_v2.json',sha256=b.sha(REG/'data_contract_repair_v2.json')),
        repair_protocol=dict(path='registration/data_repair_protocol.json',sha256=b.sha(REG/'data_repair_protocol.json')),
        candidate_audit_path='data/repair_candidate_audit.json',candidate_audit_sha256=b.sha(DATA/'repair_candidate_audit.json'),
        splits_sha256=b.sha(DATA/'splits.json'),patches_sha256=b.sha(DATA/'patches.json'),
        plan_hashes={k:b.sha(DATA/f'{k}_plan.json') for k in ('train','validation','test')},
        source_geometries_repaired=sum(x['repair']['repair_applied'] for x in objects),
        original_source_files_modified=0,original_vertices_modified=0,original_vertices_removed=0,model_forwards=0,optimizer_updates=0,OT_calls=0,
        dataset_downloads=0,noise_arrays_persisted=0,full_object_preprocessing_calls=48,
        normalization_contract=contract['coordinate_warning'],inference_allowlist=['C','free_state','t','N','known_mask','valid_mask'],
        pretraining_exposure='Official-train training/validation may have appeared in official unconditional pretraining; validation is held out only from this conditional finetuning. Official-test per-checkpoint pretraining absence is not independently certified.',
        near_duplicate_scope='All old groups retained; all3135 repaired exact/normalized signatures;528 original-range/historical repaired vertex+centroid comparisons at .01. No rotation/reflection/semantic ancestry completeness claim.')
    b.write_json(DATA/'manifest.json',m)
    runtime=dict(scan_group_patch_seconds=scan_sec,preprocess48_seconds=norm_sec,total_prepare_seconds=time.perf_counter()-tick,
        source_rehash_count=len(rows),cpu_workers=1,cpu_threads=1,gpu_workers=0,model_forwards=0,optimizer_updates=0,OT_calls=0,
        objects_per_preprocess_second=48/norm_sec,concurrent_with_independent_cpu_audits=True,
        cuda_peak='Not applicable: CUDA never initialized',normalization_scope='One official full-object preprocessing per selected object')
    if not (REG/'data_repair_runtime.json').exists():b.write_json(REG/'data_repair_runtime.json',runtime)
    if not (REG/'data_repair_resources_after.json').exists():b.write_json(REG/'data_repair_resources_after.json',b.resources())
    print(json.dumps(dict(status='PREPARED_REPAIR_V2',counts=audit['counts_by_filter'],objects=len(objects),patches=len(records),repaired_objects=m['source_geometries_repaired'],runtime=runtime)),flush=True)


def freeze():
    if (RUN/'DATA_READY.json').exists():b.verify_ready();return
    validation=b.read_json(DATA/'independent_repair_validation.json')
    if validation['status']!='PASS':raise RuntimeError('Independent repair audit not PASS')
    for rec in validation['checked_files']:
        if b.sha(RUN/rec['path'])!=rec['sha256']:raise RuntimeError('Audited artifact changed: '+rec['path'])
    m=b.read_json(DATA/'manifest.json');a=b.read_json(DATA/'repair_candidate_audit.json');sp=b.read_json(DATA/'splits.json')
    old_report=DATA/'DATA_REPORT.md';archive=DATA/'history/strict_v1/DATA_REPORT.md'
    if not archive.exists():b.write_bytes(archive,old_report.read_bytes())
    if not (REG/'data_blocked_strict_v1.json').exists():b.write_bytes(REG/'data_blocked_strict_v1.json',(RUN/'DATA_BLOCKED.json').read_bytes())
    objs=m['objects'];repaired=[o for o in objs if o['repair']['repair_applied']]
    zeros=sum(len(o['repair']['removed_zero_original_face_ids']) for o in objs);dups=sum(len(o['repair']['removed_duplicate_original_face_ids']) for o in objs)
    text=f'''# Control MVP48 — Session A 数据报告（repair_v2）

**READY：48对象、32/8/8划分、128训练C＋16验证C＋16测试C全部完成。** 用户授权修复后，本修订只清理精确零面积面与完全重复面；官方原文件只读，上一轮严格筛选阻塞证据完整保留。

| 划分 | 官方split | 对象 | 原面数 min/median/max | 保留面数 min/median/max | 条件数 |
|---|---|---:|---|---|---:|
'''
    for role in ('train','validation','test'):
        os_=[o for o in objs if o['split']==role];old=np.asarray([o['original_N'] for o in os_]);ns=np.asarray([o['N'] for o in os_])
        text+=f"|{role}|{'test' if role=='test' else 'train'}|{len(os_)}|{old.min()}/{np.median(old):g}/{old.max()}|{ns.min()}/{np.median(ns):g}/{ns.max()}|{sum(o['patch_count'] for o in os_)}|\n"
    text+=f'''
48对象中{len(repaired)}个进行了清理，合计删除{zeros}个精确零面积面、{dups}份完全重复三角形。没有移动/焊接/删除顶点，没有创造面、补洞、简化独立正面积面、细分或重网格。重复面保留最小原始face ID；所有有效三角形的位置不变，正面积三角形的几何并集保持。原始N和保留N均在128–256，N/K/free按保留面数定义。每个NPZ保留原始faces/vertices、删除原因、保留↔原始ID映射；这批数据应称“经授权最小清理的官方源派生数据”，不再称完全未经清理。

test供给已从严格版6对象恢复：本轮官方test原范围50个，保留面数合格{a['counts_by_filter']['test']['both_counts_in_range']}个，2条件适用{a['counts_by_filter']['test']['applicable_2patch']}个，未曝光合格组{a['counts_by_filter']['test']['unexposed_applicable_groups']}个。train相应为462/{a['counts_by_filter']['train']['both_counts_in_range']}/{a['counts_by_filter']['train']['applicable_2patch']}，4条件适用{a['counts_by_filter']['train']['applicable_4patch']}个。按预登记seed2026100201和组/UID哈希选择，不以模型表现或人工外观选择。

所有排除原因逐UID、逐起点记录于repair_candidate_audit.json；对象过滤计数（可重叠）为 `{json.dumps({s:a['counts_by_filter'][s]['rejection_reasons'] for s in ('train','test')},sort_keys=True)}`。原始/保留面数出界、极小非零面和缺少足量适用C均拒绝；已合格但未抽中的只是未选。异常跨known/free边拒绝patch而非无条件排除对象。严格水密、自交仍不是筛选门槛；自交未核验，未声称完美拓扑。

K=floor(.2N+.5)，共享完整源索引边FIFO BFS、几何分散起点。160块C在同对象内不同源面集合、不同起点位置，保存补集、坐标、面积比例、接口长度/边数和重合率。每条接口在清理后的拓扑及清理前原始拓扑中都恰好一known一free两面；删除重复/退化面不用于把异常原接口伪装为正常监督。

分组在划分前完成：继承全部严格版同源/精确/近重复分组，再添加所有3135个清理后几何的精确/规范化重复和528个范围/历史对象的139128对近重复检查。48对象来自48个不同最终组；历史20个开发父对象及最终组闭包不进入验证/test，P0/P1均不入选。近重复阈值为规范化顶点和面质心双向Hausdorff均≤.01；未旋转/镜像对齐或穷举语义家族。UID后缀仅是保守代理，不能声称完全无潜在近重复泄漏。

一次完整对象官方预处理，保留全部原始顶点保证bbox尺度约定；C/free共用变换。实际捕获pre_OT，严格核验`full_target=(pre_OT*2).astype(float32)`，禁止再次乘2或对C单独规范化。dense source_face_ids保持B/C行索引接口；original_source_face_ids显式对应原始面。完整GT、bbox、映射、接口仅用于监督/评分；load_condition只返回C、N和mask。

训练任务冻结1000×8=8000身份，每对象4C，128个组合各62或63次。训练任务序、Gaussian、t、排列、验证、测试、绘图/模型初始化随机域分离。测试8×2C×2noise=32case，三模型复用C/N/Gaussian共96生成；同对象两C复用自由Gaussian，不声称不同N有逐坐标对应。验证仅8对象p0、一份noise、t=.5，最终3端点共24只读teacher；不用于选checkpoint。本Session未生成模型输出。

曝光边界：32训练＋8验证来自官方train，可能已见于官方无条件预训练；验证只表示本轮条件微调未见。8测试来自官方test，但具体checkpoint预训练内容未独立认证。历史双对象调参模型不构成新对象独立证据。

独立CPU审计PASS：48对象/160条件、原始hash与最小清理映射、官方尺度、原始接口、分组/划分、全部任务流和随机规则均校验。详见independent_repair_validation.json；其中实际检查范围与耗时为准。0模型调用、0训练更新、0OT、0GPU worker；1 CPU构造worker/1数值线程，独立审计按24可用CPU与约61GiB WSL RAM并行。资源与阶段耗时在registration/data_repair_resources_*.json、data_repair_runtime.json；并发时间不当作隔离benchmark。

最新契约registration/data_contract_repair_v2.json，READY绑定其hash；B原接口签名/密集索引/尺度保持兼容，B应重新绑定最新READY。A未修改B/C代码或启动它们。旧DATA_BLOCKED原样保留为严格版历史，由带repair_v2标识的DATA_READY明确取代；旧报告/加载器在data/history/strict_v1/。不reset/clean、commit/push。
'''
    content=text.encode('utf-8')
    if old_report.read_bytes()!=content:
        tmp=DATA/'DATA_REPORT.repair_pending';b.write_bytes(tmp,content);os.replace(tmp,old_report)
    handoff=dict(schema='control_mvp48_data_handoff_repair_v2',status='READY',experiment_id=b.EXPERIMENT,data_revision='repair_v2',
        contract='registration/data_contract_repair_v2.json',contract_sha256=b.sha(REG/'data_contract_repair_v2.json'),
        compatibility='Existing B/C call signatures, shapes, retained-dense IDs, FP32 coords and Gaussian recipes unchanged; original raw IDs are explicit additive fields.',
        original_files_modified=0,model_forwards=0,optimizer_updates=0,report='data/DATA_REPORT.md',
        superseded_blocked_sha256=b.sha(REG/'data_blocked_strict_v1.json'),downstream='Bind latest DATA_READY files/contract hashes before any B/C execution.')
    b.write_json(REG/'data_handoff_repair_v2.json',handoff)
    files={str(p.relative_to(RUN)):dict(sha256=b.sha(p),bytes=p.stat().st_size) for p in sorted(DATA.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pending' not in p.name}
    files.update({str(p.relative_to(RUN)):dict(sha256=b.sha(p),bytes=p.stat().st_size) for p in sorted(REG.glob('data_*')) if p.is_file()})
    ready=dict(schema='control_mvp48_data_ready_v1',experiment_id=b.EXPERIMENT,data_revision='repair_v2',status='READY',owner='Session A',
        object_count=48,splits=dict(train=32,validation=8,test=8),patch_counts=dict(train=128,validation=16,test=16),
        effective_batches=1000,effective_batch_size=8,test_case_count=32,three_model_generations=96,teacher_checks=24,
        manifest=dict(path='data/manifest.json',sha256=b.sha(DATA/'manifest.json')),
        split=dict(path='data/splits.json',sha256=b.sha(DATA/'splits.json')),
        contract=dict(path='registration/data_contract_repair_v2.json',sha256=b.sha(REG/'data_contract_repair_v2.json')),
        independent_audit=dict(path='data/independent_repair_validation.json',sha256=b.sha(DATA/'independent_repair_validation.json')),
        task_plans={k:dict(path=f'data/{k}_plan.json',sha256=b.sha(DATA/f'{k}_plan.json')) for k in ('train','validation','test')},
        files=files,code={str(p.relative_to(REPO)):b.sha(p) for p in sorted(HERE.glob('data_*.py'))},
        supersedes_blocked=dict(path='DATA_BLOCKED.json',sha256=b.sha(RUN/'DATA_BLOCKED.json'),revision='strict_v1',
            reason='User-authorized minimal cleanup repaired the source shortage; historical marker preserved, READY is authoritative.'),
        historical_snapshot_remap={'data/DATA_REPORT.md':'data/history/strict_v1/DATA_REPORT.md','data/load_sample.py':'data/history/strict_v1/load_sample.py'},
        repaired_objects=len(repaired),removed_exact_zero_faces=zeros,removed_duplicate_faces=dups,
        original_source_files_modified=0,model_forwards=0,optimizer_updates=0,OT_calls=0,noise_arrays_persisted=0)
    b.write_json(RUN/'DATA_READY.json',ready);b.verify_ready()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.prepare==args.freeze:raise ValueError('Choose prepare or freeze')
    try:
        if args.prepare:prepare()
        else:freeze()
    except Exception as e:
        import traceback
        tb=traceback.format_exc();print(tb,file=sys.stderr)
        if not (RUN/'DATA_READY.json').exists():
            record=dict(status='BLOCKED',data_revision='repair_v2',reason=str(e),traceback=tb,model_forwards=0,optimizer_updates=0)
            p=REG/'data_repair_failure.json'
            if not p.exists():b.write_json(p,record)
        raise


if __name__=='__main__':main()
