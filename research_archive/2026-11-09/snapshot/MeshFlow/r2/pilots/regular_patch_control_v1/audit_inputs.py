"""Independent CPU-only preregistered full-patch input and cache audit."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections, csv, json, sys, time
from pathlib import Path
import numpy as np
from . import common as C

PRIOR=C.ROOT/'experiments/r2_c_reference_frame_v1'
NOISE=C.ROOT/'experiments/r2_seat_noise_transfer_v1'
PREFIXES=('73b369','3f7417','85b169','2d08a6','610b16','608b46')
EXPECTED_K={'73b369':20,'3f7417':14,'85b169':12,'2d08a6':14,'610b16':2,'608b46':2}
CACHE_PREFIXES={'85b169','608b46'}
CHECKS=0

def ck(value,message):
    global CHECKS
    CHECKS+=1
    if not bool(value):raise AssertionError(message)
def eq(a,b,message):ck(C.same(np.asarray(a),np.asarray(b)),message)
def near(a,b,message,tol=1e-7):ck(np.allclose(a,b,rtol=1e-7,atol=tol,equal_nan=False),message)
def check_file(p,h):ck(C.sha(p)==h,'source/file identity '+str(p))
def csvrows(p):
    with C.resolve(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def center(Ctri):
    x=np.asarray(Ctri,dtype=np.float32).reshape(-1,3,3)
    cr=np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0])
    area=np.sqrt(np.sum(cr*cr,axis=1,dtype=np.float32))*np.float32(.5)
    return np.ascontiguousarray(np.sum(area[:,None]*np.mean(x,axis=1,dtype=np.float32),axis=0,dtype=np.float32)/np.sum(area,dtype=np.float32),dtype=np.float32)

def source_check(item):
    check_file(item['condition_path'],item['condition_sha256']);check_file(item['parent_npz'],item['parent_sha256'])
    s=C.arrays(item['condition_path']);p=C.arrays(item['parent_npz'])
    tri=s['C_original'];known=s['C'];ids=s['C_original_source_vertex_ids'];faceids=s['source_face_ids'];freeids=s['free_source_ids'];K=len(known);N=int(s['N'])
    ck(K==int(s['K'])==item['K'] and 128<=N<=256 and 2<=K<N,'full patch K/N')
    ck(K==EXPECTED_K[item['parent_id'][:6]],'fixed per-parent full K')
    ck(known.dtype==tri.dtype==np.float32 and np.isfinite(known).all(),'FP32 finite source C')
    eq(p['model_vertices'][ids],tri,'source vertex-ID geometry')
    eq(p['model_vertices'][p['source_faces'][faceids]],tri,'original source face ordering')
    eq(np.take_along_axis(tri,s['C_source_corner_permutations'][:,:,None],axis=1),known,'canonical/source permutation')
    ck(np.array_equal(np.sort(faceids),np.sort(s['region_source_face_ids'])),'entire declared FULL surface')
    ck(len(set(faceids.tolist()))==K and not set(faceids.tolist())&set(freeids.tolist()),'C/Y disjoint source IDs')
    ck(sorted(faceids.tolist()+freeids.tolist())==list(range(N)),'C/Y complete source partition')
    edge=collections.defaultdict(list);adj=[set() for _ in range(K)]
    for i,t in enumerate(ids):
        ck(len(set(map(int,t)))==3,'no repeated source vertex in triangle')
        for a,b in zip(t,np.roll(t,-1)):edge[tuple(sorted((int(a),int(b))))].append((i,int(a),int(b)))
    for e,owners in edge.items():
        ck(len(owners)<=2,'source C no nonmanifold edge')
        if len(owners)==2:
            a,b=owners;ck(a[1:]==b[1:][::-1],'source C coherent internal winding')
            adj[a[0]].add(b[0]);adj[b[0]].add(a[0])
    todo=[0];seen={0}
    while todo:
        i=todo.pop()
        for x in adj[i]-seen:seen.add(x);todo.append(x)
    ck(len(seen)==K,'source edge-connected patch')
    boundary=[e for e,v in edge.items() if len(v)==1];badj=collections.defaultdict(set)
    for a,b in boundary:badj[a].add(b);badj[b].add(a)
    ck(bool(boundary) and all(len(v)==2 for v in badj.values()),'simple closed boundary degrees')
    todo=[next(iter(badj))];seen={todo[0]}
    while todo:
        i=todo.pop()
        for x in badj[i]-seen:seen.add(x);todo.append(x)
    ck(len(seen)==len(badj),'single source boundary loop')
    V=len(set(ids.ravel().tolist()));ck(V-len(edge)+K==1,'disk Euler identity')
    t=tri.astype(np.float64);cr=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);dbl=np.linalg.norm(cr,axis=1)
    ck((dbl>2e-12).all(),'valid finite source face areas')
    n=cr/dbl[:,None];mean=cr.sum(0);mean=mean/np.linalg.norm(mean)
    angles=np.degrees(np.arccos(np.clip(n@mean,-1,1)))
    points=np.unique(t.reshape(-1,3),axis=0);ctr=points.mean(0);_,_,vt=np.linalg.svd(points-ctr,full_matrices=False)
    plane=float(np.max(abs((points-ctr)@vt[-1])))
    minangles=[]
    for x in t:
        for i in range(3):
            a=x[(i+1)%3]-x[i];b=x[(i+2)%3]-x[i]
            minangles.append(float(np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1)))))
    quad=None
    if K==2:
        ck(V==4 and len(boundary)==4 and len(edge)==5,'native two-triangle quadrilateral')
        ck(plane<1e-6*C.L and float(angles.max())<1e-3,'two-triangle planar coherent quad')
        order=[min(badj)]
        while len(order)<4:order.append(next(x for x in badj[order[-1]] if x not in order))
        xyz=p['model_vertices'][order].astype(np.float64);uv=(xyz-ctr)@vt[:2].T
        turn=np.array([np.cross(uv[(i+1)%4]-uv[i],uv[(i+2)%4]-uv[(i+1)%4]) for i in range(4)])
        ck((turn>0).all() or (turn<0).all(),'convex quad outline')
        quad=dict(source_vertices=4,boundary_edges=4,finite_triangles=2,planarity_max_model=plane,convex=True)
    ck(item['whole_region_AI_proxy_status']=='REVIEWED_NONBLIND_AI_PROXY','input actually reviewed metadata')
    ck(item['input_review']['actually_viewed'] is True and item['input_eligibility'].startswith('USABLE'),'input review accepted')
    check_file(item['candidate_image'],item['candidate_image_sha256'])
    ck(item['reviewed_image_sha256']==item['candidate_image_sha256'],'review image identity')
    for part in ('CC','CF'):ck(item['existing_source_quality'][part]=={'illegal':0,'unknown':0},'inherited source '+part+' qualification')
    return dict(parent_id=item['parent_id'],K=K,N=N,condition_source=item['condition_path'],condition_source_sha256=item['condition_sha256'],
        source_disk=True,boundary_edges=len(boundary),source_vertices=V,area_model=float(dbl.sum()/2),
        maximum_source_normal_deviation_deg=float(angles.max()),minimum_triangle_angle_deg=min(minangles),
        max_vertex_best_plane_offset_model=plane,native_quad=quad,input_eligibility=item['input_eligibility'],
        inherited_FF_risk=item['existing_source_quality']['FF'],CREF_FP32_area_center=center(known).tolist())

def query_check(qr,request):
    check_file(qr['path'],qr['sha256']);a=C.arrays(qr['path']);inp=C.arrays(request['input_path']);tri=inp['C'].astype(np.float64)
    ck(qr['condition_id']==request['condition_id'] and qr['identity']['C_sha256']==request['C_sha256'],'query condition identity')
    pts=a['C_points'];ids=a['C_point_faces'];w=a['C_weights'];b=a['C_barycentric'];K=len(tri)
    ck(pts.shape==(2048,3) and ids.shape==(2048,) and w.shape==(2048,) and b.shape==(2048,3),'full query bank shape')
    ck(np.isfinite(pts).all() and np.isfinite(w).all() and (w>0).all(),'finite positive query')
    ck(0<=ids.min() and ids.max()<K and np.isfinite(b).all() and (b>=0).all() and (b<=1).all(),'query source face/barycentric bounds')
    near(b.sum(1),1,'query barycentric sum');near(w.sum(),1,'full-C weight denominator')
    near(np.einsum('qi,qij->qj',b,tri[ids]),pts,'query is on new finite C',1e-6*C.L)
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm=np.linalg.norm(normal,axis=1);normal/=norm[:,None]
    near(a['C_point_normals'],normal[ids],'query canonical normal owners')
    ck(a['C_point_normal_valid'].all(),'query valid normal owners')
    # Independent finite open-edge distances, using saved current-C topology.
    incidence=collections.Counter()
    for t in tri:
        for u,v in ((0,1),(1,2),(2,0)):incidence[tuple(sorted((tuple(t[u]),tuple(t[v]))))]+=1
    edges=np.asarray([x for x,n in incidence.items() if n==1],np.float64)
    ck(len(edges)>0,'nonempty full-C boundary')
    got={tuple(sorted(map(tuple,x))) for x in a['C_open_edges']}
    ck(got=={tuple(map(tuple,x)) for x in edges},'query boundary uses FULL C')
    e=edges[:,1]-edges[:,0];delta=pts[:,None]-edges[None,:,0];d=np.sum(e*e,axis=1)
    u=np.clip(np.sum(delta*e[None],axis=-1)/d[None],0,1)
    dist=np.linalg.norm(pts[:,None]-(edges[None,:,0]+u[:,:,None]*e[None]),axis=-1).min(1)
    near(a['C_border_distance'],dist,'query FULL boundary distance')
    return dict(condition_id=qr['condition_id'],query_sha256=qr['sha256'],queries=2048,all_C_weight=float(w.sum()),boundary_edges=len(edges))

def cache_check(q,arm,model,oldq,query,d0):
    cache=q['cache'][arm]
    check_file(cache['raw_path'],cache['raw_sha256']);check_file(cache['post_path'],cache['post_sha256'])
    raw=C.read(cache['raw_path']);post=C.read(cache['post_path'])
    for row in (raw,post):
        ck(row['label']==arm and row['request_id']==q['request_id']==oldq['request_id'],'cached model/request unchanged')
        for key in ('input_sha256','C_sha256','Gaussian_sha256','condition_source_sha256'):ck(row[key]==q[key]==oldq[key],'cache identity '+key)
        ck(row['checkpoint_sha256']==model['sha256'],'cached fixed model checkpoint')
    check_file(raw['raw_path'],raw['raw_sha256']);check_file(post['post_path'],post['post_sha256'])
    a=C.arrays(raw['raw_path']);inp=C.arrays(q['input_path']);K=q['K']
    eq(a['C'].reshape(K,3,3),inp['C'],'cached RAW C');eq(a['z'],inp['z'],'cached original Gaussian')
    src=C.arrays(q['condition_source']);b=C.arrays(post['post_path'])
    eq(b['output'][:K],src['C_original'],'cached source-oriented POST C')
    eq(b['vertices'][b['faces']],b['output'],'cached actual POST topology')
    d=C.read(post['proposal_receipt']);ck(d['model_forwards']==1 and d['checkpoint_sha256']==d0['sha256'],'cached single official D0')
    ck(post['GEO_calls']==1,'cached single GEO_C')
    return dict(label=arm,request_id=q['request_id'],status='PASS',raw_sha256=raw['raw_sha256'],post_sha256=post['post_sha256'])

def exposure():
    f=C.SCALING/'training_analysis/final/exposures.csv';counts=collections.Counter();total=0
    with f.open(encoding='utf-8-sig') as h:
        for r in csv.DictReader(h):
            if r['arm']=='USER_EXPANDED_67' and r['optimizer_update_completed']=='True' and 0<int(r['step'])<=6000:
                counts[r['task_id']]+=1;total+=1
    ck(total==48000,'completed EXP67 exposure denominator')
    a=C.read(C.QUICK/'plans/SMALL_C200.json');b=C.read(PRIOR/'plans/CREF_C200.json')
    ck(a['condition_counts']==b['condition_counts'],'same focused continuation task exposures')
    return counts,a['condition_counts'],dict(path=str(f),sha256=C.sha(f),completed_samples=total)

def main():
    start=time.monotonic();reg=C.load_registration();rh=C.sha(C.OUT/'registration.json')
    ck(tuple(reg['labels'])==('SMALL_C200','CREF_C200'),'two frozen models')
    ck(reg['status']=='FROZEN','registration frozen')
    check_file(C.OUT/'inputs/manifest.json',reg['manifest_sha256'])
    check_file(C.OUT/'preparation/condition_audit.json',reg['input_audit_sha256'])
    for path,digest in reg['code_hashes'].items():check_file(path,digest)
    ck(reg['budget']==dict(training_updates=0,new_rollouts=32,new_generator_forwards=1600,new_D0_forwards=32,new_Gaussian_draws=0,logical_RAW=48,logical_POST=48,cached_RAW=16,cached_POST=16,new_stage_scores=64,cached_stage_scores=32),'complete frozen call budget')
    qs=reg['requests'];queries=reg['queries'];ck(len(qs)==24 and len(queries)==6,'24 requests and six C')
    ck(len({q['request_id'] for q in qs})==24 and len({q['parent_id'] for q in qs})==6,'unique request/parent budget')
    ck({q['parent_id'][:6] for q in qs}==set(PREFIXES),'fixed six parents')
    ck(collections.Counter(q['action'] for q in qs)=={'NEW':16,'CACHE':8},'16 new requests, eight cached')
    old=C.read(NOISE/'inputs/manifest.json');oldlookup={(q['parent_id'],int(q['candidate'])):q for q in old['requests']}
    oldqueries={q['condition_id']:q for q in old['queries']}
    oldreg=C.read(PRIOR/'evaluation/registration.json')
    expected={'SMALL_C200':oldreg['models']['SMALL_C200'],'CREF_C200':C.read(PRIOR/'evaluation/bindings/CREF_C200.json')}
    for arm,b in expected.items():
        ck(C.resolve(reg['models'][arm]['path'])==C.resolve(b['path']) and reg['models'][arm]['sha256']==b['sha256'],'fixed endpoint binding '+arm)
        ck(b.get('global_step')==6200 and b.get('local_step')==200,'fixed 200-step endpoint')
    ck(reg['d0_checkpoint']['sha256']==oldreg['d0_checkpoint']['sha256'],'same official D0')
    audit=C.read(C.OUT/'preparation/condition_audit.json')
    ck(audit['status']=='COMPLETE_INPUT_AUDIT' and len(audit['conditions'])==6,'source/input review complete')
    source={r['parent_id']:r for r in audit['conditions']};source_results=[source_check(x) for x in audit['conditions']]
    qmap={q['condition_id']:q for q in queries};request_results=[];cache_results=[]
    for q in qs:
        uid=q['parent_id'];pre=uid[:6];oq=oldlookup[uid,int(q['candidate'])];item=source[uid]
        ck(int(q['candidate']) in range(4) and q['N']==oq['N']==item['N'] and q['K']==item['K'],'frozen N/candidate/full K')
        ck(q['condition_type']=='SEAT_FULL' and q['action']==('CACHE' if pre in CACHE_PREFIXES else 'NEW'),'FULL treatment/cache identity')
        check_file(q['input_path'],q['input_sha256']);check_file(q['condition_source'],q['condition_source_sha256'])
        ck(C.resolve(q['condition_source'])==C.resolve(item['condition_path']),'declared full source')
        inp=C.arrays(q['input_path']);src=C.arrays(q['condition_source']);K=q['K'];N=q['N']
        eq(inp['C'],src['C'],'actual input is full canonical C');ck(int(inp['N'])==N and int(inp['K'])==K,'input N/K fields')
        ck(q['C_sha256']==C.array_sha(inp['C']),'registered C array hash')
        check_file(oq['full_noise_path'],oq['full_noise_file_sha256']);full=C.arrays(oq['full_noise_path'])['full_Gaussian']
        check_file(q['full_noise_path'],q['full_noise_file_sha256']);eq(C.arrays(q['full_noise_path'])['full_Gaussian'],full,'new metadata binds unchanged full Gaussian')
        ck(full.shape==(1,N,9) and full.dtype==np.float32 and np.isfinite(full).all(),'old full standard Gaussian stored')
        ck(q['full_Gaussian_sha256']==oq['full_Gaussian_sha256']==C.array_sha(full),'same entire stored Gaussian')
        expectedz=full.copy();expectedz[:,:K]=0;eq(inp['z'],expectedz,'full Gaussian then zero NEW known slots')
        ck(C.array_sha(inp['z'])==q['Gaussian_sha256'],'new zero-known Gaussian hash')
        free_for_hash=inp['z'][:,K:] if q['action']=='NEW' else inp['z'][:,K:].reshape(-1,3,3)
        ck(C.array_sha(free_for_hash)==q['free_Gaussian_sha256'],'explicit NEW [1,M,9] versus legacy CACHE [M,3,3] free hash')
        oi=C.arrays(oq['input_path']);eq(inp['z'][:,K:],oi['z'][:,K:],'common absolute free slots preserved')
        oldsrc=C.arrays(oq['condition_source']);ck(set(oldsrc['source_face_ids'])<=set(src['source_face_ids']),'old C is source-face subset of FULL')
        ck(q['noise_seed']==oq['noise_seed'],'no seed redraw')
        if q['action']=='CACHE':
            ck(q['request_id']==oq['request_id'] and C.resolve(q['input_path'])==C.resolve(oq['input_path']),'unchanged cached request/input path')
            qr=qmap[q['condition_id']];oqr=oldqueries[oq['condition_id']]
            ck(qr['sha256']==oqr['sha256'],'unchanged FULL original query bytes')
            for arm in reg['labels']:cache_results.append(cache_check(q,arm,reg['models'][arm],oq,qr,reg['d0_checkpoint']))
        else:
            ck(q['request_id']!=oq['request_id'] and q['condition_id']!=oq['condition_id'],'changed C has new identities')
            ck(K>oq['K'],'registered FULL expands old HALF')
            ck(qmap[q['condition_id']]['sha256']!=oldqueries[oq['condition_id']]['sha256'],'HALF query cannot masquerade as FULL')
        request_results.append(dict(request_id=q['request_id'],old_request_id=oq['request_id'],status='PASS',action=q['action'],
            N=N,old_K=oq['K'],new_K=K,old_free=N-oq['K'],new_free=N-K,same_full_Gaussian=True,common_absolute_free_slots_bitwise=True,
            old_CREF_area_center=center(oi['C']).tolist(),new_CREF_area_center=center(inp['C']).tolist(),
            CREF_noise_contract='unchanged standard Gaussian in LOCAL coordinates; world initial free = z + new C area center; never subtract center/demean z'))
    for uid in source:ck(sorted(int(q['candidate']) for q in qs if q['parent_id']==uid)==[0,1,2,3],'four fixed noises per parent')
    query_results=[query_check(qr,next(q for q in qs if q['condition_id']==qr['condition_id'])) for qr in queries]
    exp,focus,exp_source=exposure();exposures=[]
    for item in audit['conditions']:
        cid=Path(item['condition_path']).stem
        exposures.append(dict(parent_id=item['parent_id'],condition_id=cid,EXP67_completed_full_samples=exp[cid],
            SMALL_C200_focused_full_samples=focus.get(cid,0),CREF_C200_focused_full_samples=focus.get(cid,0),
            official_pretraining_seen='UNKNOWN',meaning='Condition identity exposure, including former augmentation; not byte-identical input memorization proof'))
    ck(C.sha(C.OUT/'registration.json')==rh,'registration unchanged during input audit')
    ck(not any(n=='torch' or n.startswith('torch.') for n in sys.modules),'no model framework imported')
    result=dict(status='PASS',version='regular_full_patch_independent_input_audit_v1',registration_sha256=rh,
        auditor_sha256=C.sha(__file__),checks=CHECKS,wall_seconds=time.monotonic()-start,
        conditions=source_results,queries=query_results,requests=request_results,cache_cells=cache_results,
        exposure_records=exposures,exposure_source=exp_source,logical_RAW=48,logical_POST=48,new_RAW=32,cached_RAW=16,
        new_generator_forwards=1600,new_D0=32,new_score_stages=64,cached_score_stages=32,
        audit_engineering_note='First CPU audit stopped on free hash schema before model calls: NEW records hash [1,M,9], retained CACHE records hash [M,3,3]. Auditor now checks each explicitly. Scientific code/registration/inputs unchanged.',
        actual_calls=dict(models_loaded=0,model_forward=0,training=0,generation=0,old_quality_scorer=0,new_random_draws=0),
        limits=['This tests larger meaningful source regions, not a pure K ablation. Area, boundary, free count and CREF origin change together; total N and face-bin remain fixed.',
                'Same stored Gaussian is not an identical entire free array after K changes, nor identical world-space starting coordinates across SMALL and CREF.',
                'Main objective comparison is SMALL vs CREF on the same new FULL C and query bank. HALF/FULL I and boundary fractions have different domains.',
                'Source FF risks remain disclosed; CC/CF qualification does not certify a globally clean parent.',
                'Input regularity does not guarantee semantic adoption or legal completion; actual generation remains separate.',
                'Large weight files are not rehashed here; bindings are matched to previously frozen records and the controller owns the single weight verification.'])
    C.write(C.OUT/'independent_audit/input_audit.json',result,replace=True)
    print(json.dumps({k:result[k] for k in ('status','checks','wall_seconds','new_RAW','cached_RAW')}),flush=True)
    return result
if __name__=='__main__':main()

