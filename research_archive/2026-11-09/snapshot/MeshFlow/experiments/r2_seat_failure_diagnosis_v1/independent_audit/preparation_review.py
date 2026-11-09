"""Independent CPU re-calculation from frozen arrays; no prepare functions/models."""
from pathlib import Path
import collections, csv, json, sys
import numpy as np
from scipy.spatial import cKDTree
from r2.pilots.seat_failure_diagnosis_v1 import common as C

r=C.load_registration(); out=C.OUT/'independent_audit'; protected={}
def bind(p,h):
    assert C.sha(p)==h, str(p)
    protected[str(p)]=h
for p,h in {**r['preparation_source_hashes'],**r['source_documents']}.items():bind(p,h)
for q in r['requests']:
    bind(q['input_path'],q['input_sha256']);bind(q['condition_source'],q['condition_source_sha256'])
for q in r['queries']:bind(q['path'],q['sha256'])
conds={q['condition_id']:q for q in r['conditions']};queries={q['condition_id']:q for q in r['queries']}
requests={(q['panel'],q['base_condition_id'],q['variant'],q['candidate']):q for q in r['requests']}
assert len(conds)==20 and len(requests)==40
assert sum(q['action']=='NEW' for q in r['requests'])==24
assert all(128<=q['N']<=256 and 0<q['K']<q['N'] for q in r['requests'])

def normal(a):
    a=a.astype(np.float64);return np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0])
def invert_source(s):
    p=s['C_source_corner_permutations'];assert np.array_equal(np.sort(p,axis=1),np.tile(np.arange(3),(len(p),1)))
    inv=np.argsort(p,axis=1);return s['C'][np.arange(len(p))[:,None],inv]

subdivision=[]
for q in r['conditions']:
    s=C.arrays(q['condition_source']);assert C.same(invert_source(s),s['C_original'])
    if q['variant']!='sub4':continue
    b=conds[q['base_condition_id']+'__base'];bs=C.arrays(b['condition_source'])
    assert q['K']==4*b['K'] and q['N']==b['N']+3*b['K'] and q['N']-q['K']==b['N']-b['K']
    par=s['subdivision_parent_C_row'];assert np.array_equal(par,np.repeat(np.arange(b['K']),4))
    bary=s['child_vertices_in_parent_canonical_barycentric'];assert np.all(np.isin(bary,[0.,.5,1.])) and np.array_equal(bary.sum(-1),np.ones((q['K'],3)))
    reconstructed=np.einsum('nij,njk->nik',bary,bs['C'].astype(np.float64)[par]).astype(np.float32)
    assert C.same(reconstructed,s['C'])
    seen={};mid_ids=set();original_ids=set(map(int,bs['C_source_vertex_ids'].ravel()))
    for f,t,ii,bb in zip(par,s['C'],s['C_source_vertex_ids'],bary):
        for xyz,vid,coef in zip(t,ii,bb):
            nz=np.flatnonzero(coef);pid=bs['C_source_vertex_ids'][f]
            if len(nz)==1:
                assert int(vid)==int(pid[nz[0]]) and C.same(xyz,bs['C'][f,nz[0]])
            else:
                assert len(nz)==2 and np.all(coef[nz]==.5)
                key=tuple(sorted(map(int,pid[nz])));expected=((bs['C'][f,nz[0]].astype(float)+bs['C'][f,nz[1]].astype(float))*.5).astype(np.float32)
                assert C.same(xyz,expected) and int(vid) not in original_ids
                current=(int(vid),xyz.tobytes());assert key not in seen or seen[key]==current
                seen[key]=current;mid_ids.add(int(vid))
    assert len(mid_ids)==len(seen)
    n0=normal(bs['C_original']);nc=normal(s['C_original']);dot=np.einsum('ij,ij->i',nc,n0[par])
    assert np.all(dot>0)
    a0=np.linalg.norm(n0,axis=1)*.5;ac=np.linalg.norm(nc,axis=1)*.5
    areas=np.bincount(par,weights=ac,minlength=b['K']);rel=np.abs(areas-a0)/a0
    assert np.max(rel)<1e-5
    qr=queries[q['condition_id']];qb=queries[b['condition_id']];qa=C.arrays(qr['path']);ba=C.arrays(qb['path'])
    assert C.same(qa['C_points'],ba['C_points']) and C.same(qa['C_weights'],ba['C_weights'])
    faces=qa['C_point_faces'];cb=qa['C_barycentric'];pb=ba['C_barycentric']
    assert np.all(faces//4==ba['C_point_faces']) and np.min(cb)>-1e-12 and np.max(cb)<1+1e-12
    assert np.max(np.abs(cb.sum(1)-1))<1e-12
    original_bary=np.einsum('ni,nij->nj',cb,bary[faces]);assert np.max(np.abs(original_bary-pb))<1e-12
    points=np.einsum('ni,nij->nj',cb,s['C'].astype(float)[faces]);err=np.max(np.abs(points-qa['C_points']))
    assert err<1e-6*C.L
    cn=normal(s['C']);cn=cn/np.linalg.norm(cn,axis=1)[:,None]
    assert np.max(np.abs(cn[faces]-qa['C_point_normals']))<1e-12
    seg=qa['C_open_edges'].astype(float);v=seg[:,1]-seg[:,0];length2=(v*v).sum(1)
    assert np.all(length2>0)
    physical=qa['C_points'];t=np.einsum('qsi,si->qs',physical[:,None]-seg[None,:,0],v)/length2
    nearest=seg[None,:,0]+np.clip(t,0,1)[...,None]*v[None]
    distance=np.sqrt(((physical[:,None]-nearest)**2).sum(2).min(1))
    assert np.max(np.abs(distance-qa['C_border_distance']))<1e-12
    for noise in (0,1):
        rq=requests[q['panel'],q['base_condition_id'],'sub4',noise];rb=requests[q['panel'],q['base_condition_id'],'base',noise]
        x=C.arrays(rq['input_path']);y=C.arrays(rb['input_path'])
        assert C.same(x['z'][:,q['K']:],y['z'][:,b['K']:]) and rq['free_Gaussian_sha256']==rb['free_Gaussian_sha256']
    subdivision.append(dict(condition_id=q['condition_id'],K_before=b['K'],K_after=q['K'],N_before=b['N'],N_after=q['N'],free_count=q['N']-q['K'],midpoint_coordinates_bitwise=True,source_ID_edge_midpoints_shared=True,original_winding_preserved=True,max_per_parent_relative_area_rounding=float(rel.max()),physical_query_point_weight_bitwise=True,query_owner_barycentric_verified=True,max_query_surface_rounding_model=float(err),source_boundary_distances_independently_recomputed=True,full_free_noise_bitwise_both_candidates=True))

manifest=C.read(C.OUT/'coverage_audit/selected_train_dev_manifest.json');noise_rows=[]
for p in manifest['pairs']:
    for candidate in (0,1):
        tq=requests['TRAIN',p['train_condition_id'],'base',candidate];dq=requests['DEV',p['dev_condition_id'],'base',candidate]
        ta=C.arrays(tq['input_path']);da=C.arrays(dq['input_path']);tz=ta['z'][:,tq['K']:].reshape(-1,3,3);dz=da['z'][:,dq['K']:].reshape(-1,3,3)
        assert tq['noise_seed']==dq['noise_seed']
        m=min(len(tz),len(dz));assert C.same(tz[:m],dz[:m])
        # CPU independent replay of frozen random arrays; never creates new inference inputs.
        for q,z in ((tq,tz),(dq,dz)):
            reference=np.random.RandomState(q['noise_seed']).randn(len(z),3,3).astype(np.float32)
            assert C.same(reference,z)
        noise_rows.append(dict(pair_id=p['pair_id'],candidate=candidate,seed=tq['noise_seed'],train_free=len(tz),dev_free=len(dz),common_prefix_faces=m,common_prefix_bitwise=True,full_array_bitwise=C.same(tz,dz),MT19937_recalculation=True))

p4=next(p for p in manifest['pairs'] if p['pair_id']=='PAIR04')
train=C.arrays(p4['train_condition_source'])['C'].reshape(-1,3).astype(float);dev=C.arrays(p4['dev_condition_source'])['C'].reshape(-1,3).astype(float)
train[:,0]*=-1
nearest=np.r_[cKDTree(train).query(dev)[0],cKDTree(dev).query(train)[0]]/C.L
near=dict(pair_id='PAIR04',registered_pair_retained=True,comparison='C corner coordinates, TRAIN mirror X around model origin, independent cKDTree',symmetric_mean_L=float(nearest.mean()),symmetric_p95_L=float(np.quantile(nearest,.95)),symmetric_max_L=float(nearest.max()),interpretation='Near-duplicate geometry risk; not independent new-shape generalization evidence. Report full results plus separate sensitivity excluding PAIR04, without changing registered sampling.',group_uid_field='Registration group_uid equals parent_id by construction; do not interpret as verified source deduplication group.')
for path,h in protected.items():assert C.sha(path)==h
assert 'torch' not in sys.modules
value=dict(status='PASS_WITH_INTERPRETATION_LIMITS',utc=C.utc(),registration_sha256=C.sha(C.OUT/'registration.json'),requests=40,conditions=20,new_rollouts_registered=24,N_min=min(q['N'] for q in r['requests']),N_max=max(q['N'] for q in r['requests']),subdivision_conditions=subdivision,paired_noise=noise_rows,near_duplicate_limitation=near,protected_registered_files_checked=len(protected),protected_registered_hashes_unchanged=True,training=0,model_forwards=0,new_inference_inputs=0,audit_random_recalculation='CPU deterministic reconstruction of already-frozen arrays only; no new noise registered or used',methods='Independent NumPy geometry/barycentric and SciPy cKDTree; no prepare.subdivide or prepare.child_queries called',limitations=['TRAIN/DEV matching is observational, unequal free counts and training exposure remain.','sub4 keeps free variables but simultaneously changes K, N and C vertex-based metric denominators.','Midpoints are FP64 averages rounded once to FP32; surface equivalence has measured rounding, not identical mesh arrays.','All C open boundaries are descriptive, not a required-join contract.'],protected_files=protected,audit_source_sha256=C.sha(__file__))
C.write(out/'preparation_review.json',value)
report='''# Independent preparation review

PASS with interpretation limits. Zero model forwards and zero training.

- Verified all 40 frozen inference inputs and all 20 condition/query identities; N remains in 128..256.
- Independently reconstructed four sub4 conditions: child coordinates are exact FP32 source-edge midpoints; shared source edge IDs share midpoint IDs; original source winding is retained; per-parent triangle areas remain within the registered rounding tolerance.
- Verified 2048 physical query points and weights per sub4 condition are byte-identical to base. Recomputed child ownership through barycentric transforms, normals, and finite-segment boundary distances.
- Reconstructed each paired MT19937 stream on CPU and compared frozen arrays: 16 TRAIN/DEV noise pairs share exact prefixes, not necessarily complete arrays. All eight sub4 outputs use the complete unchanged free array of their own base request.
- PAIR04 is a near-duplicate-risk pair: independently recomputed mirrored C corner mean distance is %.12g L. Retain its outputs, show it separately and report an exclusion sensitivity; do not present it as independent new-shape generalization.
- The registration group_uid is a parent-ID alias, not the source dataset deduplication group. Consult coverage_audit/supplement_v1 for true group metadata and source similarity.
- sub4 changes K, N and vertex-based metric denominators together. Interpret as operational representation sensitivity; it does not isolate token dilution or attention causality.

All %d registered source/input/query hashes checked here remained unchanged. Detailed measurements and exact audit source hash are in preparation_review.json.
'''%(near['symmetric_mean_L'],len(protected))
C._write_bytes(out/'PREPARATION_REVIEW.md',report.encode('utf-8'))
print(json.dumps({k:value[k] for k in ('status','requests','conditions','N_min','N_max','protected_registered_files_checked','model_forwards')}))