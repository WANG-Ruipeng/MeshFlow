"""Session A: CPU-only, preregistered Control MVP48 source/condition preparation.

No model construction, forward, OT, training, download or topology repair.
Use --prepare then an independent audit, then --freeze. Never replaces a
different existing artifact. All outputs are in A's explicitly owned paths.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
RUN = REPO / 'native_t1/runs/control_mvp48_v1'
DATA = RUN / 'data'
REG = RUN / 'registration'
SOURCE = REPO / 'experiments/mf_h1_local0/assets/shapenet-03001627'
EXPERIMENT = 'control_mvp48_v1'
PERMS = np.asarray(list(itertools.permutations(range(3))), dtype=np.int64)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def ah(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.dtype, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def write_bytes(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f'Refusing to replace different existing artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive atomic link prevents overwrite even if another session races.
    tmp = path.with_name(path.name + '.pending_A')
    if tmp.exists():
        if tmp.read_bytes() != content:
            raise RuntimeError(f'Different incomplete pending artifact: {tmp}')
    else:
        with tmp.open('xb') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
    os.link(tmp, path)
    tmp.unlink()  # only this newly created, verified staging file


def write_json(path, value):
    write_bytes(path, json_bytes(value))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_npz(path, arrays):
    # Fixed ZIP metadata makes resumed byte hashes reproducible.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for key in sorted(arrays):
            entry = io.BytesIO()
            np.lib.format.write_array(entry, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(key + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, entry.getvalue())
    write_bytes(path, buf.getvalue())


def seed(namespace, identity):
    return int.from_bytes(hashlib.sha256(f'{EXPERIMENT}|{namespace}|{identity}'.encode()).digest()[:8], 'little') % (2**32 - 1)


def resources():
    def output(args):
        p = subprocess.run(args, capture_output=True, text=True, check=False)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        gpu=output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv']),
        gpu_processes=output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv']),
        cpu_cores=os.cpu_count(), cpu_affinity=len(os.sched_getaffinity(0)),
        meminfo=Path('/proc/meminfo').read_text(), loadavg=Path('/proc/loadavg').read_text(),
        relevant_processes=output(['ps', '-eo', 'pid,comm,pcpu,pmem,rss,args']),
        dispatch=dict(gpu_workers=0, cpu_geometry_workers=1, cpu_threads_per_worker=1,
            independent_readonly_auditor_max=2,
            rationale='No GPU operations authorized. One geometry scan is fast enough and shares no output writer; independent source/code audits overlap.'))


class Union:
    def __init__(self, names):
        self.parent = {x:x for x in names}
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def merge(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a,b)] = min(a,b)


def canonical_triangle_rows(tri):
    # Corner/face-order independent; dtype converted explicitly; -0 normalized.
    t = np.asarray(tri, dtype='<f8').copy()
    t[t == 0] = 0
    order = np.lexsort((t[:,:,2], t[:,:,1], t[:,:,0]), axis=1)
    t = t[np.arange(len(t))[:,None], order].reshape(-1,9)
    return t[np.lexsort(t.T[::-1])]


def geometry_hash(tri):
    return ah(canonical_triangle_rows(tri))


def topology(faces):
    edges = defaultdict(list)
    for i, face in enumerate(faces):
        for a,b in ((0,1),(1,2),(2,0)):
            edges[tuple(sorted((int(face[a]),int(face[b]))))].append(i)
    adjacency = [set() for _ in faces]
    for members in edges.values():
        for f in members:
            adjacency[f].update(g for g in members if g != f)
    unseen = set(range(len(faces)))
    sizes = []
    while unseen:
        todo = [min(unseen)]; unseen.remove(todo[0]); n=0
        while todo:
            f=todo.pop(); n+=1
            nxt=adjacency[f]&unseen; unseen-=nxt; todo.extend(nxt)
        sizes.append(n)
    return edges, [sorted(a) for a in adjacency], dict(
        edge_count=len(edges), boundary_edge_count=sum(len(x)==1 for x in edges.values()),
        nonmanifold_edge_count=sum(len(x)>2 for x in edges.values()),
        component_sizes=sorted(sizes,reverse=True), self_intersection_checked=False,
        strict_watertightness_required=False)


def patch_candidates(vertices, faces):
    N=len(faces); K=int(np.floor(.2*N+.5))
    tri=vertices[faces].astype(np.float64); centers=tri.mean(1)
    edges, adjacency, topo=topology(faces)
    remaining=set(range(N)); previous=[]; selected=[]; rejections=Counter(); attempts=[]
    mindist=((centers-centers.mean(0))**2).sum(1)
    sets=set(); seed_coords=set()
    while remaining and len(selected)<4:
        start=min(remaining,key=lambda i:(-float(mindist[i]),i))
        remaining.remove(start); previous.append(start)
        distances=((centers-centers[start])**2).sum(1)
        mindist=distances if len(previous)==1 else np.minimum(mindist,distances)
        todo=deque([start]); seen={start}; known=[]
        while todo and len(known)<K:
            f=todo.popleft(); known.append(f)
            for other in adjacency[f]:
                if other not in seen:
                    seen.add(other); todo.append(other)
        reason=None; interface=[]
        if len(known)!=K:
            reason='component_smaller_than_K'
        else:
            ks=set(known)
            for edge,members in sorted(edges.items()):
                left=[i for i in members if i in ks]; right=[i for i in members if i not in ks]
                if left and right:
                    if len(members)!=2 or len(left)!=1 or len(right)!=1:
                        reason='nonmanifold_known_free_crossing'; break
                    interface.append((edge,left[0],right[0]))
            if reason is None and not interface:
                reason='empty_known_free_interface'
            if reason is None and tuple(sorted(known)) in sets:
                reason='duplicate_condition_source_set'
            if reason is None and tuple(centers[start]) in seed_coords:
                reason='duplicate_start_centroid'
        attempts.append(dict(start_source_face_id=start,rejection_reason=reason))
        if reason:
            rejections[reason]+=1; continue
        sets.add(tuple(sorted(known))); seed_coords.add(tuple(centers[start]))
        selected.append(dict(start_source_face_id=start,known=known,
            free=[i for i in range(N) if i not in set(known)],interface=interface))
    return selected, dict(topology=topo,attempts=attempts,rejected_patch_candidates=dict(rejections),
        accepted_patch_count=len(selected),K=K)


def scan(protocol, historical):
    split={}; split_files={}
    for role in ('train','test'):
        path=SOURCE/'split'/f'{role}.npz'
        split_files[role]=dict(path=str(path.relative_to(REPO)),sha256=sha(path))
        with np.load(path,allow_pickle=True) as z:  # trusted local official metadata object array
            entries=z['npz_list'].tolist()
        for rec in entries:
            uid=str(rec['uid'])
            if uid in split: raise RuntimeError('Duplicate/cross-split official UID '+uid)
            split[uid]=role
    raws={}; rows=[]; patches={}; counts=Counter(); suffix=defaultdict(list)
    exact=defaultdict(list); normalized=defaultdict(list)
    for i,path in enumerate(sorted((SOURCE/'objaverse_occ_v5_ids').glob('*.npz'))):
        with np.load(path,allow_pickle=False) as z:
            raw={k:z[k].copy() for k in ('vertices','faces','uid','faces_num')}
        uid=str(raw['uid'].item()); v=raw['vertices']; f=raw['faces']; N=len(f)
        if uid!=path.stem or uid in raws: raise RuntimeError('Source UID/path mismatch or duplicate '+uid)
        if uid not in split: raise RuntimeError('Source missing official split '+uid)
        reasons=[]
        base_valid=(v.ndim==2 and v.shape[1]==3 and np.isfinite(v).all() and f.shape==(N,3)
            and np.issubdtype(f.dtype,np.integer) and np.all((f>=0)&(f<len(v)))
            and int(raw['faces_num'])==N)
        if not base_valid: reasons.append('invalid_source_shape_coordinates_or_indices')
        extent=float(np.ptp(v,axis=0).max()) if base_valid else 0
        if extent<=0: reasons.append('zero_scale')
        if not 128<=N<=256: reasons.append('face_count_outside_128_256')
        row=dict(uid=uid,official_split=split[uid],source_path=str(path.relative_to(REPO)),
            source_sha256=sha(path),N=N,source_vertex_count=len(v),rejection_reasons=reasons)
        if base_valid and extent>0:
            tri=v[f].astype(np.float64)
            raw['normalized64']=(v.astype(np.float64)-(v.min(0).astype(np.float64)+v.max(0).astype(np.float64))/2)/extent
            gh=geometry_hash(tri); nh=geometry_hash(np.round(raw['normalized64'][f],6))
            exact[gh].append(uid); normalized[nh].append(uid)
            source_rows=canonical_triangle_rows(tri)
            duplicates=len(source_rows)-len(np.unique(source_rows,axis=0))
            cross=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
            bad=int((cross<=1e-12*extent**2).sum())
            row.update(exact_geometry_sha256=gh,normalized_rounded6_geometry_sha256=nh,
                zero_or_unstable_area_face_count=bad,duplicate_source_triangle_count=duplicates)
            if duplicates: reasons.append('duplicate_source_triangles')
            if bad: reasons.append('zero_or_unstable_area_faces')
            if not reasons:
                pc,pa=patch_candidates(v,f); patches[uid]=pc; row['patch_audit']=pa
                if len(pc)<2: reasons.append('fewer_than_2_applicable_connected_patches')
        suffix[uid[-16:]].append(uid); raws[uid]=raw; rows.append(row)
        counts[f'{split[uid]}_all']+=1
        if i%500==0: print('scanned',i,flush=True)
    if set(raws)!=set(split): raise RuntimeError('Official split has missing local source files')
    uf=Union(raws)
    for groups in (suffix,exact,normalized):
        for members in groups.values():
            for uid in members[1:]: uf.merge(members[0],uid)
    # Near check is deliberately limited, disclosed and preregistered.
    near_uids=sorted(u for u,r in raws.items() if 'normalized64' in r and
        (128<=len(r['faces'])<=256 or u in historical))
    trees={}; extents={}
    for u in near_uids:
        v=raws[u]['normalized64']; f=raws[u]['faces']
        points=np.unique(v[f].reshape(-1,3),axis=0); centers=v[f].mean(1)
        trees[u]=(points,centers,cKDTree(points),cKDTree(centers))
        extents[u]=(points.min(0),points.max(0),centers.min(0),centers.max(0))
    near_edges=[]; pairs=0; bbox_pass=0
    for ii,u in enumerate(near_uids):
        for w in near_uids[ii+1:]:
            pairs+=1
            if max(float(np.abs(a-b).max()) for a,b in zip(extents[u],extents[w]))>.01: continue
            bbox_pass+=1
            a,b=trees[u],trees[w]
            dv1=b[2].query(a[0],distance_upper_bound=.010000000000001,workers=1)[0]
            if not np.isfinite(dv1).all(): continue
            dv2=a[2].query(b[0],distance_upper_bound=.010000000000001,workers=1)[0]
            if not np.isfinite(dv2).all(): continue
            dc1=b[3].query(a[1],distance_upper_bound=.010000000000001,workers=1)[0]
            if not np.isfinite(dc1).all(): continue
            dc2=a[3].query(b[1],distance_upper_bound=.010000000000001,workers=1)[0]
            if not np.isfinite(dc2).all(): continue
            vd=float(max(dv1.max(),dv2.max())); cd=float(max(dc1.max(),dc2.max()))
            if vd<=.01 and cd<=.01:
                near_edges.append(dict(uids=[u,w],vertex_hausdorff=vd,centroid_hausdorff=cd));uf.merge(u,w)
    groups=defaultdict(list)
    for uid in sorted(raws): groups[uf.find(uid)].append(uid)
    exposed={uf.find(u) for u in historical if u in raws}
    for row in rows:
        row.update(group_id=uf.find(row['uid']),historically_exposed_group=uf.find(row['uid']) in exposed)
    audit=dict(schema='control_mvp48_candidate_audit_v1',experiment_id=EXPERIMENT,
        source_split_files=split_files,official_counts=dict(counts),objects=rows,
        grouping=dict(groups=dict(groups),suffix16_groups=[x for x in suffix.values() if len(x)>1],
            exact_groups=[x for x in exact.values() if len(x)>1],
            normalized_rounded6_groups=[x for x in normalized.values() if len(x)>1],
            near_scope_uids=near_uids,near_pairs_tested=pairs,near_bbox_prefilter_pass=bbox_pass,near_edges=near_edges,
            limitations=protocol['grouping']['near_duplicate'],historically_exposed_group_ids=sorted(exposed)),
        counts_by_filter={s:dict(
            all=sum(r['official_split']==s for r in rows),
            in_face_range=sum(r['official_split']==s and 128<=r['N']<=256 for r in rows),
            applicable_2patch=sum(r['official_split']==s and not r['rejection_reasons'] for r in rows),
            applicable_4patch=sum(r['official_split']==s and not r['rejection_reasons'] and len(patches.get(r['uid'],[]))>=4 for r in rows),
            unexposed_applicable_groups=len({r['group_id'] for r in rows if r['official_split']==s and not r['rejection_reasons'] and not r['historically_exposed_group']}),
            rejection_reasons=dict(Counter(reason for r in rows if r['official_split']==s for reason in r['rejection_reasons']))) for s in ('train','test')})
    return raws,patches,audit


def choose(audit,patches,protocol):
    used=set(); chosen={}; shortages=[]
    def rank(role,id): return hashlib.sha256(f"{protocol['selection_seed']}|{role}|{id}".encode()).hexdigest()
    for role,official,count,num in [('test','test',8,2),('validation','train',8,2),('train','train',32,4)]:
        candidates=defaultdict(list)
        for row in audit['objects']:
            if row['official_split']!=official or row['rejection_reasons'] or row['group_id'] in used:continue
            if role!='train' and row['historically_exposed_group']:continue
            if len(patches.get(row['uid'],[]))<num:continue
            candidates[row['group_id']].append(row['uid'])
        groups=sorted(candidates,key=lambda g:(rank(role,g),g))
        chosen[role]=[min(candidates[g],key=lambda u:(rank(role,u),u)) for g in groups[:count]]
        used.update(groups[:count])
        if len(groups)<count:shortages.append(dict(role=role,required=count,available_groups=len(groups),shortfall=count-len(groups)))
    if shortages:
        write_json(DATA/'selection_shortfall.json',dict(shortages=shortages,partial_selection=chosen))
        raise RuntimeError('Insufficient eligible independent groups: '+json.dumps(shortages))
    lookup={r['uid']:r for r in audit['objects']}
    return dict(schema='control_mvp48_splits_v1',experiment_id=EXPERIMENT,selection_seed=protocol['selection_seed'],
        **chosen,group_ids={u:lookup[u]['group_id'] for us in chosen.values() for u in us},
        official_splits={u:lookup[u]['official_split'] for us in chosen.values() for u in us},
        historical_exclusion_applied=True)


def transform_steps(v):
    bounds=np.asarray([v.min(0),v.max(0)]); center1=(bounds[0]+bounds[1])/2
    divisor1=(bounds[1]-bounds[0]).max(); a=(v-center1[None,:])/divisor1
    bounds2=np.asarray([a.min(0),a.max(0)]);center2=(bounds2[0]+bounds2[1])/2;a=a-center2[None,:]
    bounds3=np.asarray([a.min(0),a.max(0)]);center3=(bounds3[0]+bounds3[1])/2
    divisor3=(bounds3[1]-bounds3[0]).max();a=((a-center3[None,:])/divisor3).clip(-.5,.5)
    return a,dict(raw_dtype=str(v.dtype),raw_bounds=bounds.tolist(),center1=center1.tolist(),divisor1=float(divisor1),
        center2=center2.tolist(),center3=center3.tolist(),divisor3=float(divisor3),clip=[-.5,.5],
        dataset_std=.3762,final_model_scale=2.,normalization_passes_per_object=1,
        operation_sequence='official FP32 operations: (raw-center1)/divisor1; subtract center2; (result-center3)/divisor3; clip; divide .3762; model multiply 2',
        note='Internal recenter operations belong to ONE official full-object preprocessing call. Patches never normalized separately. Stored raw geometry is unmodified.')


def preprocess(raw):
    from datasets.mesh_dataset import ObjaverseDataset,sort_triangle_soup
    N=len(raw['faces']); captures=[]; vc=[]
    obj=ObjaverseDataset.__new__(ObjaverseDataset)
    obj.__dict__.update(training=True,use_custom_prior=False,do_dataset_normalize=True,
        max_face_length=800,use_rot_aug=False,use_scale_aug=False,noise_sort='ot',
        use_permut_aug=False,std=.3762,data=[raw])
    def capture_noise(x): captures.append(x.copy());return x,np.zeros_like(x)
    def capture_vertices(v,f,num_tokens=None):
        vc.append(v.copy());return ObjaverseDataset.sort_vertices_and_faces(obj,v,f,num_tokens)
    obj.sample_noise=capture_noise;obj.sort_vertices_and_faces=capture_vertices
    rng=np.random.get_state();sample=obj[0];rng_after=np.random.get_state()
    if any(not np.array_equal(x,y) for x,y in zip(rng,rng_after)):raise RuntimeError('Official preprocessing drew RNG')
    if len(captures)!=1 or len(vc)!=1:raise RuntimeError('Multiple normalization/capture calls')
    pre=sort_triangle_soup(captures[0]).reshape(N,3,3)
    target=sort_triangle_soup(np.asarray(sample['coords'])).astype(np.float32).reshape(N,3,3)
    if not np.array_equal((pre*2).astype(np.float32),target):raise RuntimeError('Official pre-OT/model scale mismatch')
    expected,transform=transform_steps(raw['vertices'])
    if not np.array_equal(expected,vc[0]):raise RuntimeError('Transform operands do not reproduce official coordinates')
    model_vertices=(vc[0]/obj.std*2).astype(np.float32)
    source_tri=model_vertices[raw['faces']]; mapping=[]; corner_map=np.zeros((N,3),dtype=np.int64)
    for tri in target:
        match=(source_tri[:,PERMS,:]==tri[None,None]).all(axis=(2,3))
        ids=np.flatnonzero(match.any(1))
        if len(ids)!=1:raise RuntimeError('Official-source face map is not unique')
        src=int(ids[0]);perm=PERMS[np.flatnonzero(match[src])[0]]
        mapping.append(src);corner_map[src]=perm
    if len(set(mapping))!=N:raise RuntimeError('Not all source faces preserved')
    final=np.empty_like(target);pre_source=np.empty_like(pre);final[mapping]=target;pre_source[mapping]=pre
    vertex_ids=np.take_along_axis(raw['faces'],corner_map,axis=1)
    if not np.array_equal(model_vertices[vertex_ids],final):raise RuntimeError('Corner/source mapping changed geometry')
    return dict(uid=raw['uid'],source_vertices=raw['vertices'],source_faces=raw['faces'],
        full_target=final.reshape(N,9),pre_ot_full=pre_source,source_face_vertex_ids=vertex_ids,
        canonical_to_source_face=np.asarray(mapping,dtype=np.int64),source_corner_permutations=corner_map,
        normalized_vertices=vc[0],model_vertices=model_vertices,transform_json=np.asarray(json.dumps(transform,sort_keys=True)),
        bbox_diagonal=np.asarray(float(np.linalg.norm(np.ptp(final.reshape(-1,3),axis=0)))))


def emit_mesh(uid,raw,candidates,count,meta):
    arrays=preprocess(raw);target=arrays['full_target'].reshape(-1,3,3);N=len(target);K=int(np.floor(.2*N+.5))
    rawtri=raw['vertices'][raw['faces']].astype(np.float64)
    area=np.linalg.norm(np.cross(rawtri[:,1]-rawtri[:,0],rawtri[:,2]-rawtri[:,0]),axis=1)/2
    vid=arrays['source_face_vertex_ids'];records=[]
    selected=candidates[:count]
    overlap=[[len(set(a['known'])&set(b['known']))/K for b in selected] for a in selected]
    jaccard=[[len(set(a['known'])&set(b['known']))/len(set(a['known'])|set(b['known'])) for b in selected] for a in selected]
    for i,p in enumerate(selected):
        label=f'p{i}';edges=[];knownf=[];freef=[];kc=[];fc=[]
        for edge,k,f in p['interface']:
            edges.append(list(edge));knownf.append(k);freef.append(f)
            kc.append([int(np.flatnonzero(vid[k]==v)[0]) for v in edge])
            fc.append([int(np.flatnonzero(vid[f]==v)[0]) for v in edge])
        edgeids=np.asarray(edges,dtype=np.int64)
        model_edges=arrays['model_vertices'][edgeids].astype(np.float64)
        raw_edges=raw['vertices'][edgeids].astype(np.float64)
        lengths=np.linalg.norm(model_edges[:,1]-model_edges[:,0],axis=1)
        arrays[label+'_constraints']=target[p['known']].copy()
        records.append(dict(uid=uid,patch_id=label,N=N,K=K,source_face_ids=p['known'],free_source_ids=p['free'],
            start_source_face_id=p['start_source_face_id'],constraints_array_key=label+'_constraints',
            constraints_sha256=ah(arrays[label+'_constraints']),
            interface=dict(edge_source_vertex_ids=edges,known_source_face_ids=knownf,free_source_face_ids=freef,
                known_corner_ids=kc,free_corner_ids=fc),
            patch_metrics=dict(K_over_N=K/N,area_fraction=float(area[p['known']].sum()/area.sum()),
                interface_edge_count=len(edges),interface_length_model=float(lengths.sum()),
                interface_length_raw=float(np.linalg.norm(raw_edges[:,1]-raw_edges[:,0],axis=1).sum()),
                overlap_intersection_over_K=overlap[i],overlap_jaccard=jaccard[i],
                interface_applicable=True,known_connected_by_complete_source_edges=True)))
    path=DATA/'meshes'/f'{uid}.npz';save_npz(path,arrays)
    return records,dict(**{k:meta[k] for k in ('uid','official_split','source_path','source_sha256','N','source_vertex_count','group_id','historically_exposed_group','exact_geometry_sha256')},
        prepared_path=str(path.relative_to(RUN)),prepared_sha256=sha(path),prepared_bytes=path.stat().st_size,
        K=K,free=N-K,patch_count=count,topology=meta['patch_audit']['topology'],
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=ah(v)) for k,v in arrays.items()})


def plans(splits,objects):
    lookup={o['uid']:o for o in objects};pairs=[(u,f'p{i}') for u in sorted(splits['train']) for i in range(4)]
    identities=[]
    for cycle in range(63):
        order=np.random.RandomState(seed('training_task_order',str(cycle))).permutation(128)
        identities.extend(pairs[int(i)] for i in order[:64 if cycle==62 else 128])
    records=[]
    for idx,(u,p) in enumerate(identities):
        identity=f'{idx}|{u}|{p}';o=lookup[u]
        records.append(dict(sample_index=idx,effective_batch_index=idx//8,slot_index=idx%8,uid=u,patch_id=p,N=o['N'],K=o['K'],
            gaussian_seed=seed('training_gaussian',identity),t_seed=seed('training_time',identity),
            permutation_seed=seed('training_permutation',identity)))
    counts=Counter((r['uid'],r['patch_id']) for r in records)
    train=dict(schema='control_mvp48_train_plan_v1',experiment_id=EXPERIMENT,records=records,
        effective_batch_count=1000,batch_size=8,sample_count=8000,model_arms_share_exact_plan=True,
        pair_counts={f'{u}/{p}':counts[u,p] for u,p in pairs},
        object_counts={u:sum(r['uid']==u for r in records) for u in sorted(splits['train'])})
    validation=[];test=[]
    for u in sorted(splits['validation']):
        o=lookup[u];identity=f'{u}|p0|noise0'
        validation.append(dict(case_id=f'validation/{identity}',uid=u,patch_id='p0',noise_index=0,N=o['N'],K=o['K'],t=.5,
            gaussian_seed=seed('validation_gaussian',identity),t_seed=seed('validation_time',identity),
            permutation_seed=seed('validation_permutation',identity)))
    for u in sorted(splits['test']):
        o=lookup[u]
        for p in ('p0','p1'):
            for noise in (0,1):
                identity=f'{u}|{p}|noise{noise}'
                test.append(dict(case_id=f'test/{identity}',uid=u,patch_id=p,noise_index=noise,N=o['N'],K=o['K'],
                    gaussian_seed=seed('test_gaussian',f'{u}|noise{noise}'),
                    t_seed=seed('test_time',identity),permutation_seed=seed('test_permutation',identity),
                    model_input_order='C first, free in shared Gaussian order; no GT-based assignment'))
    write_json(DATA/'train_plan.json',train)
    write_json(DATA/'validation_plan.json',dict(schema='control_mvp48_validation_plan_v1',experiment_id=EXPERIMENT,
        records=validation,case_count=8,endpoint_count=3,total_teacher_checks=24,checkpoint_selection=False,t=.5))
    write_json(DATA/'test_plan.json',dict(schema='control_mvp48_test_plan_v1',experiment_id=EXPERIMENT,
        records=test,case_count=32,model_count=3,total_generations=96,
        shared_across_models=True,shared_free_gaussian_between_same_object_patches=True,
        cross_N_coordinate_correspondence=False))
    return train


def prepare():
    start=time.perf_counter();protocol=read_json(REG/'data_protocol.json')
    contract=read_json(REG/'data_contract.json');history=read_json(REG/'data_historical_audit.json')
    for x in (protocol,contract,history):
        if x['experiment_id']!=EXPERIMENT:raise RuntimeError('Experiment identity mismatch')
    if (RUN/'DATA_READY.json').exists():
        print('Already frozen: verify only; do not prepare again.');return
    # Source fingerprints also serve as before/after checks.
    initial={str(p.relative_to(REPO)):sha(p) for p in [REPO/'AGENTS.md',REPO/'WORKSPACE.md',
        REPO/'datasets/mesh_dataset.py',REPO/'native_t1/prepare.py',REPO/'native_t1/data.py',
        REPO/'utils/ot_utils.py',REG/'data_protocol.json',REG/'data_contract.json',REG/'data_historical_audit.json']}
    if not (REG/'data_resources_before.json').exists():write_json(REG/'data_resources_before.json',resources())
    t=time.perf_counter();raws,patches,audit=scan(protocol,set(history['historical_uids']));scan_seconds=time.perf_counter()-t
    write_json(DATA/'candidate_audit.json',audit)
    splits=choose(audit,patches,protocol);write_json(DATA/'splits.json',splits)
    allpatch=[];objects=[];lookup={x['uid']:x for x in audit['objects']};t=time.perf_counter()
    for role,num in [('train',4),('validation',2),('test',2)]:
        for u in splits[role]:
            pr,ob=emit_mesh(u,raws[u],patches[u],num,lookup[u]);ob['split']=role;objects.append(ob);allpatch.extend(pr)
    mesh_seconds=time.perf_counter()-t
    write_json(DATA/'patches.json',dict(schema='control_mvp48_patches_v1',experiment_id=EXPERIMENT,records=allpatch))
    train=plans(splits,objects)
    source_hashes={str(SOURCE/'objaverse_occ_v5_ids'/f'{u}.npz'):r['source_sha256'] for u,r in lookup.items()}
    if any(sha(p)!=h for p,h in source_hashes.items()):raise RuntimeError('Official source modified during preparation')
    if any(sha(REPO/p)!=h for p,h in initial.items()):raise RuntimeError('Preprocessing/instructions/registration changed during preparation')
    sources={p:dict(sha256=h) for p,h in initial.items()}
    manifest=dict(schema='control_mvp48_manifest_v1',experiment_id=EXPERIMENT,status='PREPARED_AWAITING_INDEPENDENT_AUDIT',
        object_count=len(objects),objects=objects,source_code_and_registration=sources,
        official_split_files=audit['source_split_files'],normalization_contract=contract['coordinate_warning'],
        inference_allowlist=protocol['coordinates']['inference_allowlist'],
        historical_exclusion_audit='registration/data_historical_audit.json',
        candidate_audit_path='data/candidate_audit.json',candidate_audit_sha256=sha(DATA/'candidate_audit.json'),
        splits_sha256=sha(DATA/'splits.json'),patches_sha256=sha(DATA/'patches.json'),
        plan_hashes={kind:sha(DATA/f'{kind}_plan.json') for kind in ('train','validation','test')},
        model_forwards=0,optimizer_updates=0,OT_calls=0,noise_arrays_persisted=0,
        dataset_downloads=0,source_geometries_repaired=0,
        pretraining_exposure='Official-train train and validation may have been seen by official unconditional pretraining. Validation is only unseen in this conditional finetuning; official-test is held out by released official split, but checkpoint-specific provenance is not independently proven.',
        near_duplicate_scope=audit['grouping']['limitations'])
    write_json(DATA/'manifest.json',manifest)
    timing=dict(schema='control_mvp48_data_runtime_v1',scan_group_patch_seconds=scan_seconds,
        preprocess_48_seconds=mesh_seconds,total_prepare_seconds=time.perf_counter()-start,
        all_source_rehash_checked=True,protected_source_count=len(source_hashes),gpu_workers=0,
        cpu_geometry_workers=1,cpu_threads=1,all_models_forwards=0,optimizer_updates=0,
        cuda_peak='not applicable: no CUDA initialized; observed GPU usage belongs to other processes',
        throughput_objects_per_second=len(objects)/mesh_seconds,phases_concurrent_with_other_sessions=True)
    if not (REG/'data_runtime.json').exists():write_json(REG/'data_runtime.json',timing)
    if not (REG/'data_resources_after.json').exists():write_json(REG/'data_resources_after.json',resources())
    print(json.dumps(dict(status='PREPARED',counts=audit['counts_by_filter'],objects=len(objects),patches=len(allpatch),timing=timing)),flush=True)


def report(manifest, audit, validation):
    objs=manifest['objects'];Ns=np.asarray([o['N'] for o in objs]);patches=read_json(DATA/'patches.json')['records']
    train=read_json(DATA/'train_plan.json');counts=list(train['pair_counts'].values());oc=list(train['object_counts'].values())
    roles={s:[o for o in objs if o['split']==s] for s in ('train','validation','test')}
    reasons={s:audit['counts_by_filter'][s]['rejection_reasons'] for s in ('train','test')}
    text=f'''# Control MVP48 — Session A DATA REPORT

48 objects complete: 32 training / 8 validation / 8 test, 48 disjoint conservative parent groups. Natural source faces preserved: min {Ns.min()}, median {float(np.median(Ns)):g}, max {Ns.max()}. No repair, remesh, model selection, model call, GPU work or training.

| Split | Official split | Objects | Face min/median/max | Conditions |
|---|---|---:|---|---:|
'''
    for s,oset in roles.items():
        ns=[o['N'] for o in oset]
        text+=f"| {s} | {'test' if s=='test' else 'train'} | {len(oset)} | {min(ns)}/{float(np.median(ns)):g}/{max(ns)} | {sum(o['patch_count'] for o in oset)} |\n"
    text+=f'''
All 128 training conditions and 16 validation + 16 test conditions are distinct within each object and connected by complete source-index edges. K=floor(.2*N+.5), free=N-K. All {len(patches)} patches have nonempty applicable interfaces: exactly one known and one free face on each original degree-2 edge. Candidate-level rejection, topology, overlap, area fraction and interface lengths/counts are recorded in `candidate_audit.json` and `patches.json`. No global watertight/self-intersection gate; self-intersections were not checked.

Selection was registered before scanning in `registration/data_protocol.json`. Candidates: official train/test face-range counts {audit['counts_by_filter']['train']['in_face_range']}/{audit['counts_by_filter']['test']['in_face_range']}; 2-patch eligible {audit['counts_by_filter']['train']['applicable_2patch']}/{audit['counts_by_filter']['test']['applicable_2patch']}; unexposed eligible groups {audit['counts_by_filter']['train']['unexposed_applicable_groups']}/{audit['counts_by_filter']['test']['unexposed_applicable_groups']}. No shortage or split substitution.

All object exclusions (reasons may overlap): `{json.dumps(reasons,sort_keys=True)}`. Full per-UID reasons, patch candidate failures, source paths/hashes, family membership and nonselected eligible objects are retained. Eligible but unselected objects are unused by the fixed seed/group ranking; they are not classified as geometrically invalid. Historical group membership separately bars evaluation selection. Training additionally requires 4 applicable patches.

Leakage audit groups all 3,135 local Chair UIDs by exact unordered triangle geometry, normalized rounded geometry, and shared trailing-16 UID family proxy. Near check: {len(audit['grouping']['near_scope_uids'])} range/historical objects, {audit['grouping']['near_pairs_tested']} pairs, {len(audit['grouping']['near_edges'])} matched pairs at 0.01 normalized vertex AND face-centroid Hausdorff thresholds. These groups were formed before split. UID suffix is a conservative heuristic, not verified ancestry. No rotation/reflection/semantic matching; hidden or differently triangulated near relatives may remain. This is bounded evidence, not a claim of zero possible leakage.

The 20 discovered historical training/diagnostic parents (including P0/P1) and discovered relatives are excluded from validation/test. Source evidence and census limits: `registration/data_historical_audit.json`. P0/P1 have 112 faces and are not selected. Validation comes from official train and may have been seen in official unconditional pretraining; it is unseen only in this conditional finetuning. Test comes from official test, subject to released-split provenance; per-checkpoint training contents were not independently proved. Existing two-parent tuned checkpoints do not make these fresh conditional comparisons automatically fair; model initialization is Session B's responsibility.

Coordinates follow the actual official dataset implementation with Native T1 flags. Each full object is preprocessed once; source-order row/corner mapping and untouched raw vertices/faces are stored. Captured pre-OT coordinates and model targets are both retained and checked bitwise: model=(pre_OT*2).astype(float32). C/free share this transform; never multiply model targets by 2 again, nor normalize each C. Original transform operands and bounds are supervision/scoring only. B/C interface is `registration/data_contract.json`; `load_condition` returns C, N and masks only.

Frozen train plan: exactly 1,000 effective batches ×8 =8,000 identities; every object/patch occurs {min(counts)}–{max(counts)} times and every object {min(oc)}–{max(oc)} times. Separate seeds for task order, Gaussian, t and permutations; no full Gaussian dataset stored. Validation: patch p0, one fixed Gaussian, t=.5, 8 cases ×3 endpoints =24 teacher checks, no checkpoint selection. Test: 8×2 patches×2 noises=32 cases shared across three models=96 outputs; same object/noise shares free Gaussian across patches. Different N has no claimed per-coordinate correspondence. Model initialization and plotting use independent namespaces.

CPU verification: {validation['status']}; all sources rehashed before/after, 48 official preprocess outputs mapped exactly, 160 partition/connectivity/interface checks, plan counts/balance/RNG reproducibility and inference allowlist checked. No OT/model call. The independent audit file and all deliverables are hash-bound by DATA_READY.json. Resource snapshots and phase timings are in `registration/data_resources_*.json` / `data_runtime.json`. No GPU worker; one CPU geometry worker with one numerical thread, independent read-only audits overlapped. Concurrent wall times are not isolated compute benchmarks.

Deliverables: `meshes/<uid>.npz`, `manifest.json`, `splits.json`, `patches.json`, `train_plan.json`, `validation_plan.json`, `test_plan.json`, `load_sample.py`, `candidate_audit.json`, independent audit, and this report. DATA_READY is published only after all checks; downstream must bind and verify its hashes. Session A stops after freeze.
'''
    return text.encode()


def freeze():
    if (RUN/'DATA_READY.json').exists():
        verify_ready();return
    manifest=read_json(DATA/'manifest.json');audit=read_json(DATA/'candidate_audit.json')
    validation=read_json(DATA/'independent_validation.json')
    if validation.get('status')!='PASS':raise RuntimeError('Independent audit has not passed')
    for rec in validation['checked_files']:
        if sha(RUN/rec['path'])!=rec['sha256']:raise RuntimeError('Audited file changed: '+rec['path'])
    write_bytes(DATA/'DATA_REPORT.md',report(manifest,audit,validation))
    files={str(p.relative_to(RUN)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(DATA.rglob('*')) if p.is_file() and '.pending_' not in p.name}
    files.update({str(p.relative_to(RUN)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(REG.glob('data_*')) if p.is_file()})
    ready=dict(schema='control_mvp48_data_ready_v1',experiment_id=EXPERIMENT,status='READY',owner='Session A',
        object_count=48,splits=dict(train=32,validation=8,test=8),patch_counts=dict(train=128,validation=16,test=16),
        effective_batches=1000,effective_batch_size=8,test_case_count=32,three_model_generations=96,teacher_checks=24,
        manifest=dict(path='data/manifest.json',sha256=sha(DATA/'manifest.json')),
        split=dict(path='data/splits.json',sha256=sha(DATA/'splits.json')),
        task_plans={k:dict(path=f'data/{k}_plan.json',sha256=sha(DATA/f'{k}_plan.json')) for k in ('train','validation','test')},
        contract=dict(path='registration/data_contract.json',sha256=sha(REG/'data_contract.json')),
        independent_audit=dict(path='data/independent_validation.json',sha256=sha(DATA/'independent_validation.json')),
        files=files,code=dict(path='experiments/mf_h1_local0/control_mvp48/data_prepare.py',sha256=sha(Path(__file__))),
        model_forwards=0,optimizer_updates=0,OT_calls=0,noise_arrays_persisted=0,
        previous_blocked_record_present=(RUN/'DATA_BLOCKED.json').exists())
    write_json(RUN/'DATA_READY.json',ready)
    verify_ready()


def verify_ready():
    ready=read_json(RUN/'DATA_READY.json')
    if ready['experiment_id']!=EXPERIMENT or ready['status']!='READY':raise RuntimeError('Different READY identity')
    for path,rec in ready['files'].items():
        if sha(RUN/path)!=rec['sha256']:raise RuntimeError('Frozen artifact changed: '+path)
    print(json.dumps(dict(status='READY_VERIFIED',sha256=sha(RUN/'DATA_READY.json'),files=len(ready['files']))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--freeze',action='store_true');p.add_argument('--verify',action='store_true');a=p.parse_args()
    try:
        if sum((a.prepare,a.freeze,a.verify))!=1:raise ValueError('Choose exactly one operation')
        if a.prepare:prepare()
        elif a.freeze:freeze()
        else:verify_ready()
    except Exception as e:
        import traceback
        detail=traceback.format_exc();print(detail,file=sys.stderr,flush=True)
        if not (RUN/'DATA_READY.json').exists():
            blocked=dict(schema='control_mvp48_data_blocked_v1',experiment_id=EXPERIMENT,status='BLOCKED',
                reason=str(e),traceback=detail,completed_files=[str(p.relative_to(RUN)) for p in sorted(DATA.rglob('*')) if p.is_file()],
                model_forwards=0,optimizer_updates=0)
            if not (RUN/'DATA_BLOCKED.json').exists():write_json(RUN/'DATA_BLOCKED.json',blocked)
        raise


if __name__=='__main__':main()
