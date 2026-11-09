"""Freeze eight existing Chair BACK_FULL inputs; CPU only, no model/OT calls.

Selection uses C geometry and source identity only. Re-entry verifies frozen
assets and never redraws noise or replaces an incomplete preparation attempt.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'r2/src'))
from meshflow_control.seam.slot_data import edge_features, inference_layout
from meshflow_control.source_winding import capture_source, validate_source
from r2.pilots.seam_slot_c200_v1 import generate as G
from r2.pilots.user_condition_finetune_v1 import evaluation as Q

OUT = ROOT / 'experiments/r2_chair_back_transfer_v1'
TRAIN = ROOT / 'experiments/r2_user_data_scaling_32_67_v1/data/train_manifest.json'
DEV = ROOT / 'experiments/r2_user_data_scaling_32_67_v1/data/panel_manifest.json'
QUALIFIED = ROOT / 'experiments/r2_user_data_scaling_12h_v1/data/qualified_condition_manifest.json'
POOL47 = ROOT / 'experiments/r2_seam_abc_v1/B_SOURCE_CONDITIONS.csv'
VERSION = 'chair_back_transfer_C_only_fixed_inputs_v1'
L = float(np.sqrt(3) * 2 / .3762)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(G.clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def save(p, values):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    G.save_npz(p, {k: np.asarray(v) for k, v in values.items()})


def components(faces):
    book = collections.defaultdict(list)
    for i, face in enumerate(faces):
        for j in range(3):
            book[tuple(sorted((int(face[j]), int(face[(j + 1) % 3]))))].append(i)
    graph = [set() for _ in faces]
    for owners in book.values():
        for i in owners:
            graph[i].update(owners)
    unseen = set(range(len(faces))); count = 0
    while unseen:
        count += 1; todo = [unseen.pop()]
        while todo:
            for j in graph[todo.pop()] & unseen:
                unseen.remove(j); todo.append(j)
    return count


def inspect(row):
    a = arrays(row['npz']); c = np.asarray(a['C'], np.float32).reshape(-1, 3, 3)
    original = np.asarray(a['C_original'], np.float32).reshape(-1, 3, 3)
    ids = np.asarray(a['C_original_source_vertex_ids'], np.int64)
    if ids.shape != original.shape[:2]:
        raise ValueError('Original C source IDs shape mismatch')
    vertices = []; source_ids = np.unique(ids)
    for vid in source_ids:
        xyz = original[ids == vid]
        if any(not G.same(xyz[0], x) for x in xyz):
            raise ValueError('One source vertex has multiple coordinates')
        vertices.append(xyz[0])
    vertices = np.asarray(vertices, np.float32)
    if len({x.tobytes() for x in vertices}) != len(vertices):
        raise ValueError('Distinct C source IDs share exact coordinates')
    lookup = {int(v): i for i, v in enumerate(source_ids)}
    faces = np.asarray([[lookup[int(v)] for v in f] for f in ids], np.int64)
    perm = np.asarray(a['C_source_corner_permutations'], np.int64)
    if perm.shape != faces.shape or not np.array_equal(np.sort(perm, axis=1), np.tile(np.arange(3), (len(faces), 1))):
        raise ValueError('C source corner mapping not invertible')
    model = np.take_along_axis(faces, perm, axis=1)
    if not G.same(vertices[model], c) or not G.same(vertices[faces], original):
        raise ValueError('C original/canonical coordinates mismatch')
    n, k = int(a['N']), int(a['K'])
    if not (128 <= n <= 256 and 2 <= k < n and k == len(c)):
        raise ValueError('N/K unsupported')
    area2 = np.linalg.norm(np.cross(original[:, 1] - original[:, 0], original[:, 2] - original[:, 0]), axis=1)
    if not np.isfinite(original).all() or np.any(area2 <= 1e-12):
        raise ValueError('Nonfinite or zero-area C')
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != k:
        raise ValueError('Exact duplicate C indexed face')
    cc = components(faces)
    if cc != 1:
        raise ValueError('C not source-edge connected')
    edge, features = edge_features(vertices, faces, n)
    bundle = dict(C_vertices=vertices, C_faces_original=faces, C_faces_model=model, total_N=np.asarray(n, np.int64))
    # Compatibility check for both extreme C-only layouts, not a prediction.
    for on in (False, True):
        layout = inference_layout(**bundle, occupied=np.full(len(edge), on, bool))
        assert layout['N'] == n and len(layout['seam_slots']) + len(layout['bulk_slots']) == n - k
    metadata = dict(C_source_vertex_ids=source_ids, source_C_face_ids=np.asarray(a['C_original_source_face_ids'], np.int64),
                    source_C_corner_permutations=perm, C_faces_original=faces, C_faces_model=model)
    source = capture_source(original, source_vertex_ids=ids, corner_permutation=perm,
                            source_face_ids=metadata['source_C_face_ids'])
    validate_source(source, c)
    facts = dict(K=k, N=n, C_components=cc, C_boundary_edges=len(edge), C_area=float(area2.sum()/2),
                 C_span=np.ptp(vertices, axis=0).tolist(), source_identity_reversible=True,
                 original_winding_consistent=True, exact_degenerate_faces=0, exact_duplicate_faces=0)
    return bundle, metadata, source, facts


def render(selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    dest = OUT / 'input_review'; dest.mkdir(exist_ok=True)
    pages = []
    for row in selected:
        inp = arrays(row['inference_npz']); parent = arrays(row['source_record']['parent_npz'])
        v = parent['model_vertices']; f = parent['source_faces']
        whole = v[f]; c = inp['C_vertices'][inp['C_faces_original']]
        known = np.asarray(row['source_record']['parent_C_face_rows'], np.int64)
        colors = np.tile([.73, .74, .76, 1], (len(f), 1)); colors[known] = [.06, .42, .95, 1]
        bounds = whole.reshape(-1, 3); mid = (bounds.min(0) + bounds.max(0))/2; radius = np.ptp(bounds, axis=0).max()*.55
        fig = plt.figure(figsize=(13, 7))
        for i, (tri, col, title, azim) in enumerate(((whole, colors, 'Source M with C highlighted', -65),
                (c, '#197be3', 'C only / same camera and scale', -65),
                (whole, colors, 'Source M / reverse view', 115), (c, '#197be3', 'C only / reverse view', 115))):
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolors=(.1,.15,.2,.55), linewidths=.4))
            ax.set(xlim=(mid[0]-radius,mid[0]+radius), ylim=(mid[1]-radius,mid[1]+radius), zlim=(mid[2]-radius,mid[2]+radius))
            ax.set_box_aspect((1,1,1)); ax.view_init(elev=18, azim=azim, vertical_axis='y'); ax.set_axis_off(); ax.set_title(title)
        fig.suptitle(f"{row['split']} {row['parent_id']}  BACK_FULL  K={row['K']} N={row['N']}\nInput/source review only; source completion is not a unique generation target")
        fig.tight_layout(); image = dest/(row['condition_id']+'.png')
        if image.exists():
            raise FileExistsError('Refuse replacing input review')
        fig.savefig(image, dpi=130); plt.close(fig)
        pages.append(dict(condition_id=row['condition_id'], parent_id=row['parent_id'], image=str(image), image_sha256=sha(image)))
    write(dest/'manifest.json', dict(pages=pages, review='PENDING_AI_PROXY', generated_outputs_used=False))
    return pages


def run():
    done = OUT / 'data_completion.json'
    if done.exists():
        result = read(done)
        for p, h in result['files'].items():
            if sha(p) != h:
                raise ValueError('Frozen asset changed: '+p)
        return dict(status='VERIFIED_EXISTING_NO_REDRAW', conditions=8, requests=16)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'data_attempt.json').exists():
        raise RuntimeError('Incomplete CPU preparation exists; inspect explicitly, no silent redraw')
    import psutil
    resource = dict(cpu_count=os.cpu_count(), cpu_percent=psutil.cpu_percent(interval=.1), ram_available=psutil.virtual_memory().available,
                    GPU_workers=0, CPU_workers=1, threads=1)
    for name, args in [('GPU', ['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),
                       ('GPU_processes',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])]:
        try:
            rr=subprocess.run(args,capture_output=True,text=True,timeout=20);resource[name]=dict(returncode=rr.returncode,stdout=rr.stdout,stderr=rr.stderr)
        except OSError as exc:
            resource[name]=dict(error=repr(exc))
    write(OUT/'data_resources.json', resource)
    write(OUT/'data_attempt.json', dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), version=VERSION, source_code_sha256=sha(__file__)))
    train, dev, qualified = read(TRAIN), read(DEV), read(QUALIFIED)
    with POOL47.open(encoding='utf-8-sig') as f:
        ids47 = {r['parent_id'] for r in csv.DictReader(f) if r['group']=='TRAIN47'}
    parents = {p.get('parent_uid',p.get('uid')):p for p in qualified['parents']}
    parents.update(train['parents'])
    groups = dict(FAMILIAR=[r for r in train['tasks'].values() if r['condition_type']=='BACK_FULL' and r['parent_id'] in ids47],
                  DEV=[r for r in dev['conditions'] if r['condition_type']=='BACK_FULL'])
    if [len(groups[k]) for k in ('FAMILIAR','DEV')] != [35,19]:
        raise ValueError('Expected frozen 35/19 BACK_FULL pools')
    selected=[]; audits=[]; cache={}
    for group, rows in groups.items():
        qualified_rows=[]
        for row in rows:
            try:
                package=inspect(row);cache[row['condition_id']]=package
                qualified_rows.append(row);audit=dict(status='ELIGIBLE', reason='PASS', **package[-1])
            except (ValueError, AssertionError) as exc:
                audit=dict(status='INELIGIBLE', reason=str(exc), K=row['K'], N=row['N'])
            audits.append(dict(group=group,parent_id=row['parent_id'],condition_id=row['condition_id'],**audit))
        ordered=sorted(qualified_rows,key=lambda r:(int(r['K']),r['parent_id'],r['condition_id']))
        if len(ordered)<4:
            raise RuntimeError('Fewer than four eligible C in '+group)
        indices=[int(np.floor((j+.5)*len(ordered)/4)) for j in range(4)]
        for j,i in enumerate(indices):
            selected.append((group,ordered[i],dict(eligible_count=len(ordered),sorted_rank_zero_based=i,quartile=j)))
    write(OUT/'eligibility.json',dict(rows=audits,policy='C only: indexed connected, open manifold consistently oriented boundary, K>=2, finite nonzero area, invertible source identity; no Y quality or generated output filter'))
    frozen_selection=[dict(group=g,condition_id=r['condition_id'],parent_id=r['parent_id'],K=r['K'],N=r['N'],**rank) for g,r,rank in selected]
    write(OUT/'selection.json',dict(policy='Within each eligible pool sort by (K,parent_id,condition_id), choose ranks floor((j+.5)*n/4), j=0..3',rows=frozen_selection,generated_outputs_used=False))
    conditions=[];requests=[];protected={str(p):sha(p) for p in (TRAIN,DEV,QUALIFIED,POOL47)}
    for group,row,rank in selected:
        cid=row['condition_id'];bundle,meta,source,facts=cache[cid];base=OUT/'inputs'/cid;base.mkdir(parents=True)
        parent=parents[row['parent_id']];pn=parent['npz'];pa=arrays(pn);ca=arrays(row['npz']);known=ca['source_face_ids'].astype(np.int64)
        assert G.same(pa['full_target'].reshape(-1,3,3)[known],bundle['C_vertices'][bundle['C_faces_model']])
        for p in (row['npz'],pn):protected[p]=sha(p)
        save(base/'C_only.npz',bundle);save(base/'identity_metadata.npz',meta);save(base/'source_condition.npz',source)
        # Source-completion data remains outside the deployable C-only inputs.
        rec=dict(condition_path=row['npz'],condition_sha256=sha(row['npz']),parent_npz=pn,parent_sha256=sha(pn),parent_C_face_rows=known.tolist(),
                 official_pretrain_seen=row.get('official_pretrain_seen','UNKNOWN'),conditional_pretrain_seen=row.get('conditional_pretrain_seen','UNKNOWN'))
        c=bundle['C_vertices'][bundle['C_faces_model']]
        query,query_seed=Q.query_bank(c,cid);qp=OUT/'queries'/(cid+'.npz');save(qp,query)
        qd=dict(condition_id=cid,parent_id=row['parent_id'],panel=group,path=str(qp),sha256=sha(qp),array_sha256={k:Q.array_sha(v) for k,v in query.items()},
                identity=dict(condition_id=cid,C_seed=int(query_seed),C_sha256=Q.array_sha(c),L0=L))
        co=dict(condition_id=cid,parent_id=row['parent_id'],uid=row['parent_id'],split=group,group=group,panel=group,condition_type='BACK_FULL',region_type='BACK',tier='FULL',
                N=int(row['N']),K=int(row['K']),ratio=float(row['K']/row['N']),inference_npz=str(base/'C_only.npz'),metadata_npz=str(base/'identity_metadata.npz'),
                source_npz=str(base/'source_condition.npz'),query_path=str(qp),query_sha256=sha(qp),query_descriptor=qd,source_record=rec,selection=rank,input_facts=facts,
                familiar_definition='Source parent seen in TRAIN47; BACK_FULL may already have been exposed during EXP67. Not a globally unseen C.' if group=='FAMILIAR' else 'Held-out conditional-training DEV parent; official pretraining exposure UNKNOWN')
        for key in ('inference_npz','metadata_npz','source_npz'):co[key+'_sha256']=sha(co[key])
        G.read_bundle(co); reconstructed=G.source_from_bundle(co,bundle)
        for k in source:assert G.same(source[k],reconstructed[k]),k
        conditions.append(co)
        for noise_index in range(2):
            seed=int.from_bytes(hashlib.sha256((VERSION+'|'+cid+'|'+str(noise_index)).encode()).digest()[:4],'little')
            z=np.random.RandomState(seed).randn(co['N']-co['K'],3,3).astype(np.float32);npz=base/f'noise_n{noise_index}.npz';save(npz,dict(z=z,seed=np.asarray(seed,np.int64)))
            rq={**co,'request_id':row['parent_id'][:8]+f'_BACK_FULL_n{noise_index}','noise_index':noise_index,'candidate':noise_index,'seed':seed,'noise_npz':str(npz),
                'noise_npz_sha256':sha(npz),'noise_sha256':sha(npz),'query_descriptor':qd}
            G.read_noise(rq);requests.append(rq)
    if len(requests)!=16 or len({r['request_id'] for r in requests})!=16:raise ValueError('Request identity mismatch')
    manifest=dict(schema=VERSION,status='FROZEN',conditions=conditions,requests=requests,arms=['S2000'],L=L,source_hashes=protected,
                  layout='Frozen LONG C-only occupancy head, new C; no source/Y owner oracle',trained_updates=0,gaussian_draws=16,C_query_draws=8,
                  original_data_unchanged=True,source_GT_rank=False,review='Input images pending actual AI_PROXY inspection; no generated outputs exist at selection')
    write(OUT/'eval_manifest.json',manifest)
    pages=render(conditions)
    write(OUT/'data_registration.json',dict(schema=VERSION,status='FROZEN',input_manifest=str(OUT/'eval_manifest.json'),input_manifest_sha256=sha(OUT/'eval_manifest.json'),
          selection_sha256=sha(OUT/'selection.json'),source_pools={k:len(v) for k,v in groups.items()},selection_policy=read(OUT/'selection.json')['policy'],
          conditions=8,requests=16,noises_per_C=2,C_queries=2048,L=L,coordinates='Existing full-parent model coordinates, no per-C normalization/remeshing/repair',
          model_forwards=0,OT_calls=0,training_updates=0,gaussian_arrays_created_once=16,source_metadata='Pure C identity, no Y or source occupancy',
          nearest_prior='SEAM/S2000 retained 28 requests all SEAT_FULL; this is BACK_FULL frozen-weight transfer, not new training or new category',
          source_code_sha256=sha(__file__)))
    own_files=[OUT/name for name in ('data_resources.json','data_attempt.json','eligibility.json','selection.json','eval_manifest.json','data_registration.json')]
    for name in ('inputs','queries','input_review'):
        own_files.extend(p for p in (OUT/name).rglob('*') if p.is_file())
    files={str(p):sha(p) for p in sorted(own_files)}
    for p,h in protected.items():assert sha(p)==h,p
    write(done,dict(status='COMPLETE_CPU_INPUTS',conditions=8,requests=16,pages=pages,model_forwards=0,OT_calls=0,files=files))
    return dict(status='COMPLETE_CPU_INPUTS',selection=frozen_selection,requests=16)


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,indent=2))
