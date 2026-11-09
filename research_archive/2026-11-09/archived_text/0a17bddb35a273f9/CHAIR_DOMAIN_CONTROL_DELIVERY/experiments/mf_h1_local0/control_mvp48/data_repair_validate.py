"""Independent CPU audit of the authorized Control MVP48 repair revision.

Does not import the builder, construct models, run OT, use CUDA, or write any
artifact except data/independent_repair_validation.json after all checks pass.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import collections
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'native_t1/runs/control_mvp48_v1'
DATA = RUN / 'data'
REG = RUN / 'registration'
SOURCE = ROOT / 'experiments/mf_h1_local0/assets/shapenet-03001627'
EXPERIMENT = 'control_mvp48_v1'
sys.path.insert(0, str(ROOT))


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.dtype, value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def bits(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def close(left, right, label):
    check(np.allclose(left, right, rtol=1e-12, atol=1e-12), label)


def triangle_key(triangle):
    return tuple(sorted(tuple(float(c) for c in vertex) for vertex in triangle))


def geometry_key(triangles):
    rows = np.asarray(sorted(triangle_key(t) for t in triangles), dtype='<f8')
    rows[rows == 0] = 0
    return hashlib.sha256(rows.tobytes()).digest()


def repair(vertices, original_faces):
    triangles = vertices[original_faces].astype(np.float64)
    cross = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1)
    zero, duplicate, representatives, retained, first = [], [], [], [], {}
    for index, triangle in enumerate(triangles):
        if cross[index] == 0:
            zero.append(index)
            continue
        key = triangle_key(triangle)
        if key in first:
            duplicate.append(index)
            representatives.append(first[key])
        else:
            first[key] = index
            retained.append(index)
    return np.asarray(retained, dtype=np.int64), cross, {
        'removed_zero_original_face_ids': zero,
        'removed_duplicate_original_face_ids': duplicate,
        'duplicate_representative_original_face_ids': representatives,
        'retained_original_face_ids': retained,
    }


def topology(faces):
    # Lists preserve repeated incidence of a degenerate original face edge.
    incidence = collections.defaultdict(list)
    for index, face in enumerate(faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            incidence[tuple(sorted((int(face[a]), int(face[b]))))].append(index)
    adjacent = [set() for _ in faces]
    for members in incidence.values():
        for index in members:
            adjacent[index].update(set(members) - {index})
    return incidence, adjacent


def independently_make_patches(vertices, original_faces, retained):
    faces = original_faces[retained]
    incidence, adjacent = topology(faces)
    original_incidence, _ = topology(original_faces)
    n = len(faces)
    k = int(np.floor(0.2 * n + 0.5))
    centers = vertices[faces].astype(np.float64).mean(axis=1)
    attempted, accepted, sets, starts, attempts = [], [], [], [], []
    while len(attempted) < n and len(accepted) < 4:
        available = sorted(set(range(n)) - set(attempted))
        if attempted:
            scores = np.min(np.sum((centers[:, None] - centers[attempted][None]) ** 2, axis=2), axis=1)
        else:
            scores = np.sum((centers - centers.mean(axis=0)) ** 2, axis=1)
        start = min(available, key=lambda i: (-float(scores[i]), i))
        attempted.append(start)
        pending, seen, known = collections.deque([start]), {start}, []
        while pending and len(known) < k:
            index = pending.popleft()
            known.append(index)
            for other in sorted(adjacent[index]):
                if other not in seen:
                    seen.add(other)
                    pending.append(other)
        reason, boundary = None, []
        if len(known) != k:
            reason = 'component_smaller_than_K'
        else:
            ks = set(known)
            for edge, members in sorted(incidence.items()):
                left = [i for i in members if i in ks]
                right = [i for i in members if i not in ks]
                if left and right:
                    if len(members) != 2 or len(left) != 1 or len(right) != 1:
                        reason = 'nonmanifold_retained_crossing'
                        break
                    expected_original = sorted(int(retained[i]) for i in members)
                    if len(original_incidence[edge]) != 2 or sorted(original_incidence[edge]) != expected_original:
                        reason = 'abnormal_original_source_crossing'
                        break
                    boundary.append((edge, left[0], right[0]))
            if reason is None and not boundary:
                reason = 'empty_known_free_interface'
            if reason is None and ks in sets:
                reason = 'duplicate_condition_source_set'
            if reason is None and tuple(centers[start]) in starts:
                reason = 'duplicate_start_centroid'
        attempts.append(dict(start_source_face_id=start, start_original_face_id=int(retained[start]), rejection_reason=reason))
        if reason is None:
            sets.append(set(known))
            starts.append(tuple(centers[start]))
            accepted.append(dict(known=known, free=sorted(set(range(n)) - set(known)),
                                 start=start, interface=boundary))
    return accepted, attempts


def official_preprocess(vertices, faces, uid, stored):
    from datasets.mesh_dataset import ObjaverseDataset, sort_triangle_soup
    import torch
    check(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized')
    before = np.random.get_state()
    captures, normalized = [], []
    dataset = ObjaverseDataset.__new__(ObjaverseDataset)
    dataset.__dict__.update(training=True, use_custom_prior=False, do_dataset_normalize=True,
        max_face_length=800, use_rot_aug=False, use_scale_aug=False, noise_sort='ot',
        use_permut_aug=False, std=0.3762,
        data=[dict(vertices=vertices.copy(), faces=faces.copy(), uid=np.asarray(uid), faces_num=len(faces))])
    def capture_noise(value):
        captures.append(value.copy())
        return value, np.zeros_like(value)
    def capture_vertices(value, face_ids, num_tokens=None):
        normalized.append(value.copy())
        check(num_tokens is None, 'Unexpected quantization')
        return ObjaverseDataset.sort_vertices_and_faces(dataset, value, face_ids, num_tokens)
    dataset.sample_noise = capture_noise
    dataset.sort_vertices_and_faces = capture_vertices
    result = dataset[0]
    check(len(captures) == len(normalized) == 1, 'one complete-object preprocessing call')
    check(all(np.array_equal(a, b) for a, b in zip(before, np.random.get_state())), 'official RNG draw')
    canonical = sort_triangle_soup(result['coords']).reshape(-1, 3, 3).astype(np.float32)
    pre_canonical = sort_triangle_soup(captures[0]).reshape(-1, 3, 3)
    n = len(faces)
    mapping = stored['canonical_to_source_face']
    check(np.array_equal(np.sort(mapping), np.arange(n)), 'canonical retained-row bijection')
    check(bits(canonical, stored['full_target'][mapping].reshape(n, 3, 3)), 'official exact model coordinates '+uid)
    check(bits(pre_canonical, stored['pre_ot_full'][mapping]), 'official exact captured pre-OT '+uid)
    check(bits((stored['pre_ot_full'] * 2).astype(np.float32).reshape(n, 9), stored['full_target']), 'x2 relation')
    check(bits(normalized[0], stored['normalized_vertices']), 'normalized vertices')
    # Recompute each official operation separately, retaining FP32 arithmetic.
    value = vertices.copy()
    bounds1 = np.asarray([value.min(0), value.max(0)])
    center1 = (bounds1[0] + bounds1[1]) / 2
    divisor1 = (bounds1[1] - bounds1[0]).max()
    value = (value - center1[None]) / divisor1
    bounds2 = np.asarray([value.min(0), value.max(0)])
    center2 = (bounds2[0] + bounds2[1]) / 2
    value = value - center2[None]
    bounds3 = np.asarray([value.min(0), value.max(0)])
    center3 = (bounds3[0] + bounds3[1]) / 2
    divisor3 = (bounds3[1] - bounds3[0]).max()
    value = ((value - center3[None]) / divisor3).clip(-0.5, 0.5)
    check(bits(value, normalized[0]), 'independent transform replay')
    transform = json.loads(str(stored['transform_json'].item()))
    for key, expected in dict(raw_bounds=bounds1.tolist(), center1=center1.tolist(), divisor1=float(divisor1),
        center2=center2.tolist(), center3=center3.tolist(), divisor3=float(divisor3),
        dataset_std=0.3762, final_model_scale=2.0, normalization_passes_per_object=1).items():
        check(transform[key] == expected, 'transform operand '+key)
    check(bits((value / 0.3762 * 2).astype(np.float32), stored['model_vertices']), 'model vertices')
    corners = stored['source_corner_permutations']
    check(np.array_equal(np.sort(corners, axis=1), np.tile(np.arange(3), (n, 1))), 'corner permutations')
    ids = np.take_along_axis(faces, corners, axis=1)
    check(np.array_equal(ids, stored['source_face_vertex_ids']), 'corner vertex provenance')
    check(bits(stored['model_vertices'][ids], stored['full_target'].reshape(n, 3, 3)), 'mapped full target')
    close(stored['bbox_diagonal'].item(), np.linalg.norm(np.ptp(stored['full_target'].reshape(-1, 3), axis=0)), 'bbox')
    check(not torch.cuda.is_initialized(), 'preprocessing initialized CUDA')


def seed(namespace, identity):
    return int.from_bytes(hashlib.sha256(f'{EXPERIMENT}|{namespace}|{identity}'.encode()).digest()[:8], 'little') % (2**32 - 1)


def validate_plans(splits, objects, loader):
    documents = {kind: read(DATA/f'{kind}_plan.json') for kind in ('train', 'validation', 'test')}
    for kind, document in documents.items():
        check(document == loader.load_plan(kind), 'loader plan '+kind)
    train = documents['train']
    check((train['effective_batch_count'], train['batch_size'], train['sample_count']) == (1000, 8, 8000), 'train budget')
    check(train['model_arms_share_exact_plan'] is True, 'shared train plan')
    pairs = [(uid, f'p{index}') for uid in sorted(splits['train']) for index in range(4)]
    expected = []
    for cycle in range(63):
        order = np.random.RandomState(seed('training_task_order', str(cycle))).permutation(128)
        expected.extend(pairs[i] for i in order[:64 if cycle == 62 else 128])
    check(len(train['records']) == 8000, 'train plan length')
    counts = collections.Counter()
    streams = collections.defaultdict(set)
    for index, (record, identity_pair) in enumerate(zip(train['records'], expected)):
        uid, patch = identity_pair
        check((record['uid'], record['patch_id']) == identity_pair, 'training order')
        check((record['sample_index'], record['effective_batch_index'], record['slot_index']) == (index, index//8, index%8), 'training index')
        check((record['N'], record['K']) == (objects[uid]['N'], objects[uid]['K']), 'training shape')
        identity = f'{index}|{uid}|{patch}'
        for name, namespace in [('gaussian_seed', 'training_gaussian'), ('t_seed', 'training_time'), ('permutation_seed', 'training_permutation')]:
            check(record[name] == seed(namespace, identity), 'train seed '+name)
            streams[namespace].add(record[name])
        counts[identity_pair] += 1
    check(set(counts.values()) == {62, 63} and len(counts) == 128, '128 balanced training pairs')
    check(train['pair_counts'] == {f'{u}/{p}': counts[u,p] for u,p in pairs}, 'pair counts')
    check(train['object_counts'] == {u:sum(v for (uid,p),v in counts.items() if uid==u) for u in sorted(splits['train'])}, 'object counts')
    val = documents['validation']
    check((val['case_count'],val['endpoint_count'],val['total_teacher_checks']) == (8,3,24) and val['checkpoint_selection'] is False and val['t']==0.5, 'teacher budget')
    check(len(val['records']) == 8, 'teacher count')
    for uid, record in zip(sorted(splits['validation']), val['records']):
        identity = f'{uid}|p0|noise0'
        check(record['case_id'] == f'validation/{identity}' and record['uid']==uid and record['patch_id']=='p0' and record['noise_index']==0 and record['t']==0.5, 'teacher identity')
        for name, namespace in [('gaussian_seed','validation_gaussian'),('t_seed','validation_time'),('permutation_seed','validation_permutation')]:
            check(record[name] == seed(namespace,identity), 'teacher seed')
            streams[namespace].add(record[name])
    test = documents['test']
    check((test['case_count'],test['model_count'],test['total_generations']) == (32,3,96), 'test budget')
    check(test['shared_across_models'] is True and test['shared_free_gaussian_between_same_object_patches'] is True and test['cross_N_coordinate_correspondence'] is False, 'test sharing contract')
    check(len(test['records']) == 32 and len({r['case_id'] for r in test['records']})==32, 'test count')
    expected_test = itertools.product(sorted(splits['test']), ('p0','p1'), (0,1))
    shared = {}
    for record, (uid, patch, noise) in zip(test['records'], expected_test):
        identity=f'{uid}|{patch}|noise{noise}'
        check(record['case_id']==f'test/{identity}' and (record['uid'],record['patch_id'],record['noise_index'])==(uid,patch,noise), 'test identity')
        for name, namespace, ident in [('gaussian_seed','test_gaussian',f'{uid}|noise{noise}'),('t_seed','test_time',identity),('permutation_seed','test_permutation',identity)]:
            check(record[name]==seed(namespace,ident), 'test seed')
            streams[namespace].add(record[name])
        generated=loader.gaussian(record['gaussian_seed'],record['N']-record['K'])
        check(bits(generated,np.random.RandomState(record['gaussian_seed']).randn(record['N']-record['K'],3,3)), 'Gaussian canonical API')
        if (uid,noise) in shared:check(bits(generated,shared[uid,noise]), 'same-object Gaussian sharing')
        else:shared[uid,noise]=generated
    # Literal seed integers can collide across 32-bit namespaces by chance;
    # independence is local RNG ownership plus domain-separated derivation.
    for record in train['records'][::127] + val['records'] + test['records']:
        n,k=record['N'],record['K']
        rng=np.random.RandomState(record['permutation_seed'])
        exp_free=rng.permutation(n-k)
        exp_corners=np.asarray([[0,1,2],[1,2,0],[2,0,1]])[rng.randint(0,3,size=n)]
        exp_faces=rng.permutation(n)
        actual=loader.permutations(record['permutation_seed'],n,k)
        check(all(bits(a,b) for a,b in zip(actual,(exp_free,exp_corners,exp_faces))), 'permutation API')
        t=np.float32(1/(1+np.exp(-np.random.RandomState(record['t_seed']).randn())))
        check(bits(loader.training_time(record['t_seed']),t), 'time API')
    check('model_initialization' not in streams and 'plot' not in streams, 'reserved RNG namespace')
    return {name:len(values) for name,values in streams.items()}


def run():
    started=time.monotonic()
    check(not (DATA/'independent_repair_validation.json').exists(), 'Refuse overwriting independent audit')
    check(not (RUN/'DATA_READY.json').exists(), 'Already frozen; do not rerun preparation audit')
    protocol=read(REG/'data_protocol.json')
    repair_protocol=read(REG/'data_repair_protocol.json')
    contract=read(REG/'data_contract_repair_v2.json')
    history=read(REG/'data_historical_audit.json')
    old_audit=read(DATA/'candidate_audit.json')
    old_validation=read(DATA/'independent_shortfall_validation.json')
    audit=read(DATA/'repair_candidate_audit.json')
    manifest=read(DATA/'manifest.json')
    splits=read(DATA/'splits.json')
    patches=read(DATA/'patches.json')['records']
    for doc in (protocol,repair_protocol,contract,history,old_audit,old_validation,audit,manifest,splits):
        check(doc['experiment_id']==EXPERIMENT, 'experiment identity')
    check(repair_protocol['data_revision']==contract['data_revision']==manifest['data_revision']=='repair_v2','revision identity')
    check(protocol['selection_seed']==2026100201 and protocol['face_count_inclusive']==[128,256], 'base selection unchanged')
    check(contract['repair_protocol']['sha256']==digest(REG/'data_repair_protocol.json'), 'contract repair binding')
    check(old_validation['status']=='CONFIRMED_BLOCKED', 'strict evidence preserved')
    for record in old_validation['checked_files']:
        if record['path']=='DATA_BLOCKED.json':continue
        check(digest(RUN/record['path'])==record['sha256'], 'preserved strict audit '+record['path'])
    required=[REG/'data_protocol.json',REG/'data_repair_protocol.json',REG/'data_contract_repair_v2.json',
        REG/'data_historical_audit.json',DATA/'candidate_audit.json',DATA/'independent_shortfall_validation.json',
        DATA/'repair_candidate_audit.json',DATA/'manifest.json',DATA/'splits.json',DATA/'patches.json',DATA/'load_sample.py']
    required += [DATA/f'{kind}_plan.json' for kind in ('train','validation','test')]
    check((REG/'data_repair_protocol.json').stat().st_mtime_ns < (DATA/'repair_candidate_audit.json').stat().st_mtime_ns,'repair rules preregistered')
    checked_before={p:digest(p) for p in required}
    before_rng=np.random.get_state()
    official={}
    for role in ('train','test'):
        path=SOURCE/'split'/f'{role}.npz'
        check(digest(path)==audit['source_split_files'][role]['sha256'], 'official split hash')
        with np.load(path,allow_pickle=True) as archive:entries=archive['npz_list'].tolist()
        for record in entries:
            uid=str(record['uid'])
            check(uid not in official, 'official duplicate UID')
            official[uid]=role
    rows={r['uid']:r for r in audit['objects']}
    old_rows={r['uid']:r for r in old_audit['objects']}
    check(len(rows)==len(audit['objects'])==3135 and set(rows)==set(official), 'complete local official inventory')
    parent={u:u for u in rows}
    def find(uid):
        while parent[uid]!=uid:
            parent[uid]=parent[parent[uid]]
            uid=parent[uid]
        return uid
    def union(a,b):
        a,b=find(a),find(b)
        parent[max(a,b)]=min(a,b)
    for group in old_audit['grouping']['groups'].values():
        for uid in group[1:]:union(group[0],uid)
    historical=set(history['historical_uids'])
    near_data={}
    repaired_signature_maps=[{},{}]
    source_hashes=[]
    raw_selected={}
    all_selected=set(splits['train']+splits['validation']+splits['test'])
    candidate_patches={}
    repaired_metadata={}
    paths=sorted((SOURCE/'objaverse_occ_v5_ids').glob('*.npz'))
    check({p.stem for p in paths}==set(rows),'local file inventory')
    for index,path in enumerate(paths):
        with np.load(path,allow_pickle=False) as archive:
            vertices=archive['vertices'].copy()
            original_faces=archive['faces'].copy()
            check(str(archive['uid'].item())==path.stem and int(archive['faces_num'])==len(original_faces),'raw source identity')
        uid=path.stem
        row=rows[uid]
        raw_n=len(original_faces)
        check(row['official_split']==official[uid], 'official assignment')
        actual_hash=digest(path)
        check(actual_hash==row['source_sha256'],'official raw hash '+uid)
        check(actual_hash==old_rows[uid]['source_sha256'],'source changed since strict attempt')
        source_hashes.append(dict(path=str(path.relative_to(ROOT)),sha256=actual_hash))
        check(vertices.ndim==2 and vertices.shape[1]==3 and np.isfinite(vertices).all(),'finite vertices')
        check(original_faces.shape==(raw_n,3) and np.issubdtype(original_faces.dtype,np.integer) and np.all((original_faces>=0)&(original_faces<len(vertices))),'valid original indices')
        extent=float(np.ptp(vertices,axis=0).max())
        check(extent>0,'nonzero raw scale')
        retained,cross,metadata=repair(vertices,original_faces)
        faces=original_faces[retained]
        check(row['original_N']==raw_n and row['N']==len(retained),'raw/retained counts '+uid)
        repaired_metadata[uid]=metadata
        for key,expected in metadata.items():check(row['repair'][key]==expected,'all-source repair metadata '+uid+' '+key)
        check(row['repair']['original_N']==raw_n and row['repair']['retained_N']==len(retained),'repair count metadata')
        norm=(vertices.astype(np.float64)-(vertices.min(0).astype(np.float64)+vertices.max(0).astype(np.float64))/2)/extent
        for mapping,key in zip(repaired_signature_maps,(geometry_key(vertices[faces]),geometry_key(np.round(norm[faces],6)))):
            if key in mapping:union(uid,mapping[key])
            else:mapping[key]=uid
        if 128<=raw_n<=256 or uid in historical:
            points=np.unique(norm[faces].reshape(-1,3),axis=0)
            centers=norm[faces].mean(axis=1)
            if len(points) and len(centers):near_data[uid]=(points,centers,cKDTree(points),cKDTree(centers))
        reasons=[]
        if not 128<=raw_n<=256:reasons.append('original_face_count_outside_128_256')
        if not 128<=len(retained)<=256:reasons.append('retained_face_count_outside_128_256')
        tiny=int(np.sum(cross[retained]<=1e-12*extent**2))
        check(row['retained_small_nonzero_area_face_count']==tiny,'tiny nonzero count')
        if tiny:reasons.append('retained_small_nonzero_area_faces')
        if not reasons:
            accepted,attempts=independently_make_patches(vertices,original_faces,retained)
            candidate_patches[uid]=accepted
            if 'patch_audit' in row:
                check(len(accepted)==row['patch_audit']['accepted_patch_count'],'candidate patch count '+uid)
                check(attempts==row['patch_audit']['attempts'],'candidate start and acceptance attempts '+uid)
            if len(accepted)<2:reasons.append('fewer_than_2_applicable_connected_patches')
        check(row['rejection_reasons']==reasons,'all exclusion reasons '+uid)
        if uid in all_selected:
            check(not row['rejection_reasons'] and 128<=raw_n<=256 and 128<=len(retained)<=256,'selected range/validity')
            raw_selected[uid]=(vertices,original_faces,retained)
        if index%500==0:print('independent repair source audit',index,flush=True)
    grouping=audit['grouping']
    check(sorted(near_data)==grouping['near_scope_uids'],'repaired near scope')
    near_edges=[]
    pair_count=0
    for u,w in itertools.combinations(sorted(near_data),2):
        pair_count+=1
        a,b=near_data[u],near_data[w]
        distance=max(float(b[2].query(a[0],workers=1)[0].max()),float(a[2].query(b[0],workers=1)[0].max()))
        if distance>0.01:continue
        center_distance=max(float(b[3].query(a[1],workers=1)[0].max()),float(a[3].query(b[1],workers=1)[0].max()))
        if center_distance<=0.01:
            near_edges.append((u,w))
            union(u,w)
    check(pair_count==grouping['near_pairs_tested'],'near pair count')
    check(near_edges==[tuple(e['uids']) for e in grouping['near_edges']],'repaired near edge completeness')
    groups=collections.defaultdict(list)
    exposed={find(uid) for uid in historical}
    for uid in sorted(rows):
        group=find(uid)
        groups[group].append(uid)
        check(rows[uid]['group_id']==group and rows[uid]['historically_exposed_group']==(group in exposed),'group/history closure')
    check(dict(groups)==grouping['groups'],'complete group memberships')
    used=set()
    for role,official_role,count,required_patches in [('test','test',8,2),('validation','train',8,2),('train','train',32,4)]:
        choices=collections.defaultdict(list)
        for uid,accepted in candidate_patches.items():
            group=find(uid)
            if official[uid]!=official_role or group in used or len(accepted)<required_patches:continue
            if role!='train' and group in exposed:continue
            choices[group].append(uid)
        rank=lambda identifier:hashlib.sha256(f"2026100201|{role}|{identifier}".encode()).hexdigest()
        chosen_groups=sorted(choices,key=lambda g:(rank(g),g))[:count]
        expected=[min(choices[g],key=lambda u:(rank(u),u)) for g in chosen_groups]
        check(len(expected)==count and splits[role]==expected,'fixed registered selection '+role)
        used.update(chosen_groups)
    check(len(all_selected)==len(used)==48,'48 independent groups')
    check(len(manifest['objects'])==manifest['object_count']==48,'manifest48')
    check(len(patches)==160 and len({(r['uid'],r['patch_id']) for r in patches})==160,'160 unique conditions')
    objects={o['uid']:o for o in manifest['objects']}
    spec=importlib.util.spec_from_file_location('control_mvp48_verified_loader',DATA/'load_sample.py')
    loader=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    patch_count=0
    for uid,obj in objects.items():
        vertices,original_faces,retained=raw_selected[uid]
        faces=original_faces[retained]
        path=RUN/obj['prepared_path']
        check(path==DATA/'meshes'/f'{uid}.npz','prepared path')
        check(digest(path)==obj['prepared_sha256'],'prepared NPZ hash')
        required.append(path)
        checked_before[path]=digest(path)
        with np.load(path,allow_pickle=False) as archive:stored={key:archive[key].copy() for key in archive.files}
        check(bits(stored['source_vertices'],vertices),'untouched original raw vertices')
        check(bits(stored['original_source_faces'],original_faces),'untouched original raw faces')
        check(bits(stored['source_faces'],faces),'retained unchanged faces')
        check(np.array_equal(stored['retained_to_original_face'],retained),'retained source mapping')
        inverse=np.full(len(original_faces),-1,dtype=np.int64)
        inverse[retained]=np.arange(len(retained))
        check(np.array_equal(stored['original_to_retained_face'],inverse),'original inverse mapping')
        metadata=json.loads(str(stored['repair_metadata_json'].item()))
        for key,expected in repaired_metadata[uid].items():check(metadata[key]==expected,'repair removal metadata '+key)
        for key,value in stored.items():
            info=obj['arrays'][key]
            check(info==dict(shape=list(value.shape),dtype=str(value.dtype),sha256=array_digest(value)),'array manifest '+key)
        official_preprocess(vertices,faces,uid,stored)
        selected=[r for r in patches if r['uid']==uid]
        expected_count=4 if obj['split']=='train' else 2
        check(len(selected)==obj['patch_count']==expected_count,'per-object patch count')
        check(obj['N']==len(retained) and obj['original_N']==len(original_faces),'manifest counts')
        area=np.linalg.norm(np.cross(vertices[faces].astype(np.float64)[:,1]-vertices[faces].astype(np.float64)[:,0],vertices[faces].astype(np.float64)[:,2]-vertices[faces].astype(np.float64)[:,0]),axis=1)/2
        allsets=[set(r['source_face_ids']) for r in selected]
        for index,(record,expected) in enumerate(zip(selected,candidate_patches[uid])):
            n,k=len(retained),int(np.floor(0.2*len(retained)+0.5))
            check(record['patch_id']==f'p{index}' and record['N']==n and record['K']==k,'patch identity')
            check(record['source_face_ids']==expected['known'] and record['free_source_ids']==expected['free'],'BFS complete partition')
            check(record['start_source_face_id']==expected['start'],'fixed patch start')
            known=np.asarray(expected['known'],np.int64)
            free=np.asarray(expected['free'],np.int64)
            check(record['original_source_face_ids']==retained[known].tolist() and record['original_free_source_ids']==retained[free].tolist(),'patch original source IDs')
            c=stored[f'p{index}_constraints']
            check(bits(c,stored['full_target'][known].reshape(k,3,3)),'once-normalized C')
            check(record['constraints_sha256']==array_digest(c),'C hash')
            interface=record['interface']
            edges=[list(x[0]) for x in expected['interface']]
            known_faces=[x[1] for x in expected['interface']]
            free_faces=[x[2] for x in expected['interface']]
            check(interface['edge_source_vertex_ids']==edges and interface['known_source_face_ids']==known_faces and interface['free_source_face_ids']==free_faces,'all genuine interfaces')
            for label,ids in [('known',known_faces),('free',free_faces)]:
                check(interface[f'{label}_original_source_face_ids']==retained[ids].tolist(),'interface original mapping')
                expected_corners=[[int(np.flatnonzero(stored['source_face_vertex_ids'][face]==vertex)[0]) for vertex in edge] for face,edge in zip(ids,edges)]
                check(interface[f'{label}_corner_ids']==expected_corners,'interface endpoint/corner correspondence')
            metrics=record['patch_metrics']
            close(metrics['K_over_N'],k/n,'K/N')
            close(metrics['area_fraction'],area[known].sum()/area.sum(),'area fraction')
            check(metrics['interface_edge_count']==len(edges) and metrics['interface_applicable'] and metrics['known_connected_by_complete_source_edges'],'interface/connectivity metrics')
            for label,verts in [('raw',vertices),('model',stored['model_vertices'])]:
                endpoints=verts[np.asarray(edges)].astype(np.float64)
                close(metrics[f'interface_length_{label}'],np.linalg.norm(endpoints[:,1]-endpoints[:,0],axis=1).sum(),'interface length')
            close(metrics['overlap_intersection_over_K'],[len(set(known)&other)/k for other in allsets],'intersection overlap')
            close(metrics['overlap_jaccard'],[len(set(known)&other)/len(set(known)|other) for other in allsets],'Jaccard overlap')
            sample=loader.load_sample(uid,record['patch_id'])
            check(sample['N']==n and sample['K']==k and sample['original_N']==len(original_faces),'loader counts')
            for key in ('source_vertices','source_faces','original_source_faces','retained_to_original_face','original_to_retained_face','full_target','pre_ot_full'):
                check(bits(sample[key],stored[key]),'loader arrays '+key)
            check(bits(sample['free_target'],stored['full_target'][free]) and bits(sample['pre_ot_free'],stored['pre_ot_full'][free]),'loader free supervision')
            condition=loader.load_condition(uid,record['patch_id'])
            check(set(condition)=={'C','N','known_mask','valid_mask'},'strict inference allowlist')
            check(bits(condition['C'],c) and condition['N']==n and np.array_equal(condition['known_mask'],np.arange(n)<k) and np.array_equal(condition['valid_mask'],np.ones(n,dtype=bool)),'condition values')
            patch_count+=1
    check(set(objects)==all_selected and patch_count==160,'full delivery')
    rng_streams=validate_plans(splits,objects,loader)
    for source in source_hashes:check(digest(ROOT/source['path'])==source['sha256'],'official sources unchanged after audit')
    for source in history['source_files']:
        relative=source['path'].replace('\\','/').split('MeshFlow/',1)[1]
        check(digest(ROOT/relative)==source['sha256'],'historical evidence hash')
    for relative,info in manifest['source_code_and_registration'].items():
        expected=info['sha256'] if isinstance(info,dict) else info
        check(digest(ROOT/relative)==expected,'protected code/registration hash')
    check(manifest['splits_sha256']==digest(DATA/'splits.json') and manifest['patches_sha256']==digest(DATA/'patches.json'),'manifest identities')
    for kind,value in manifest['plan_hashes'].items():check(value==digest(DATA/f'{kind}_plan.json'),'manifest plans')
    check(all(np.array_equal(a,b) for a,b in zip(before_rng,np.random.get_state())),'global numpy RNG unchanged')
    for path,value in checked_before.items():check(digest(path)==value,'artifact changed during independent audit '+str(path))
    report=dict(schema='control_mvp48_independent_repair_validation_v2',experiment_id=EXPERIMENT,data_revision='repair_v2',status='PASS',
        independent_implementation=True,builder_imported=False,complete_48_delivery=True,object_count=48,
        split_counts={role:len(splits[role]) for role in ('train','validation','test')},patch_count=patch_count,
        train_condition_count=128,validation_condition_count=16,test_condition_count=16,
        official_source_count=len(paths),checked_raw_source_hash_count=len(source_hashes),
        source_hashes_before_after_equal=True,retained_source_geometry_unchanged=True,
        exact_zero_and_duplicate_only_repair_verified=True,positive_area_triangle_union_preserved=True,
        original_and_retained_count_bounds=[128,256],official_preprocess_bitwise_object_count=48,
        source_corner_and_face_maps_verified=True,original_degree2_interfaces_verified=True,
        grouping_old_groups_never_split=True,near_scope_count=len(near_data),near_pairs_independently_evaluated=pair_count,
        near_matches=len(near_edges),grouping_limitations=protocol['grouping']['near_duplicate'],
        historical_groups_excluded_from_validation_test=True,fixed_seed_selection_reproduced=True,
        plan_counts=dict(train_samples=8000,effective_batches=1000,batch_size=8,teacher_cases=8,teacher_checks=24,test_cases=32,test_generations=96),
        training_pair_count_range=[62,63],rng_namespaces=rng_streams,global_numpy_rng_unchanged=True,
        inference_allowlist=['C','N','known_mask','valid_mask'],model_calls=0,training_updates=0,OT_calls=0,gpu_workers=0,cpu_threads=1,
        wall_seconds=time.monotonic()-started,
        checked_files=[dict(path=str(p.relative_to(RUN)),sha256=digest(p)) for p in sorted(set(required))],
        source_hashes=source_hashes,validator=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=digest(__file__)))
    out=DATA/'independent_repair_validation.json'
    content=(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    pending=out.with_name(out.name+'.pending_auditor')
    with pending.open('xb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending,out)
    pending.unlink()
    print(json.dumps({k:report[k] for k in ('status','object_count','patch_count','near_scope_count','near_pairs_independently_evaluated','wall_seconds')}),flush=True)


if __name__=='__main__':
    run()
