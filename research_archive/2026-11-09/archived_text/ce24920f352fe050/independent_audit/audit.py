"""Independent CPU audit of frozen cached meshes; no production geometry/scorer import.

Uses dominant-axis 2-D triangle coordinates independently of production's cross
product barycentrics. Rechecks every query against every finite free triangle.
Only writes this independent_audit directory. No torch/model/framework import.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv, json, hashlib, time, sys, traceback
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE.parent
ROOT=OUT.parents[1]
BASE=ROOT/'experiments/r2_c_reference_frame_v1'
CHECKS=0
L=float(2*np.sqrt(3)/.3762)

def path(x):
    s=str(x).replace('\\','/')
    if os.name!='nt' and len(s)>2 and s[1]==':':s='/mnt/'+s[0].lower()+s[2:]
    if os.name=='nt' and s.startswith('/mnt/'):s=s[5].upper()+':'+s[6:]
    return Path(s)
def read(x):return json.loads(path(x).read_text(encoding='utf-8-sig'))
def rows(x):
    with path(x).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def arrays(x):
    with np.load(path(x),allow_pickle=False) as a:return {k:a[k] for k in a.files}
def sha(x):
    h=hashlib.sha256()
    with path(x).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ck(value,why):
    global CHECKS
    CHECKS+=1
    if not bool(value):raise AssertionError(why)
def eq(a,b,why):ck(np.array_equal(a,b,equal_nan=True),why)
def near(a,b,why,atol=2e-8):ck(np.allclose(a,b,rtol=2e-8,atol=atol,equal_nan=True),why)
def boolean(x):return str(x).lower()=='true'
def normal(t):
    n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);length=np.linalg.norm(n,axis=1)
    return np.divide(n,length[:,None],out=np.zeros_like(n),where=length[:,None]>0),length/2
def bary(points,t):
    """Project onto dominant normal axis, solve the two-dimensional determinant."""
    n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);drop=np.argmax(abs(n),axis=-1)
    out=np.full(points.shape,np.nan,np.float64)
    for axis in range(3):
        ids=np.flatnonzero(drop==axis)
        if not len(ids):continue
        u,v=[i for i in range(3) if i!=axis]
        e=t[ids,1]-t[ids,0];g=t[ids,2]-t[ids,0];delta=points[:,ids]-t[None,ids,0]
        den=e[:,u]*g[:,v]-e[:,v]*g[:,u]
        b1=np.divide(delta[:,:,u]*g[None,:,v]-delta[:,:,v]*g[None,:,u],den[None],out=np.full(delta.shape[:2],np.nan),where=den[None]!=0)
        b2=np.divide(e[None,:,u]*delta[:,:,v]-e[None,:,v]*delta[:,:,u],den[None],out=np.full(delta.shape[:2],np.nan),where=den[None]!=0)
        out[:,ids]=np.stack((1-b1-b2,b1,b2),axis=-1)
    return out

def distances(q,t):
    n,_=normal(t);delta=q[:,None]-t[None,:,0]
    proj=q[:,None]-np.sum(delta*n[None],axis=-1)[:,:,None]*n[None]
    bc=bary(proj,t);inside=np.isfinite(bc).all(-1)&(bc>=0).all(-1)&(bc<=1).all(-1)
    dist=np.where(inside,np.sum((proj-q[:,None])**2,-1),np.inf)
    for u,v in ((0,1),(1,2),(2,0)):
        e=t[:,v]-t[:,u];dd=np.sum(e*e,-1)
        alpha=np.divide(np.sum((q[:,None]-t[None,:,u])*e[None],-1),dd[None],out=np.zeros(dist.shape),where=dd[None]>0)
        edge=t[None,:,u]+np.clip(alpha,0,1)[:,:,None]*e[None]
        dist=np.minimum(dist,np.sum((edge-q[:,None])**2,-1))
    return dist

def load(cell):
    inp=arrays(cell['input_path']);src=arrays(cell['condition_source_path']);g=arrays(cell['geometry_path']);query=arrays(cell['query_path'])
    C=np.asarray(inp['C']);original=np.asarray(src['C_original']);K=len(C)
    ck(K==cell['K'],'K identity')
    for a,b in zip(C,original):ck(sorted(map(tuple,a))==sorted(map(tuple,b)),'source C face row identity')
    if cell['stage']=='SOURCE':
        vertices=g[cell['vertices_key']];faces=g[cell['faces_key']][np.concatenate((src['source_face_ids'],src['free_source_ids']))]
        F=vertices[faces[K:]]
    elif cell['stage']=='POST':
        vertices=g['vertices'];faces=g['faces'];F=g['free']
        eq(vertices[faces],g['output'],'POST actual index geometry');eq(F,g['output'][K:],'POST free row order')
    else:
        eq(C,g['C'],'RAW C bytes');F=g['free'];tri=np.concatenate((C,F));vertices,idx=np.unique(tri.reshape(-1,3),axis=0,return_inverse=True);faces=idx.reshape(-1,3)
        ck(K+len(F)==cell['N'],'RAW original N')
    for a,b in zip(C,vertices[faces[:K]]):ck(sorted(map(tuple,a))==sorted(map(tuple,b)),'indexed C face geometry')
    ck(np.isfinite(F).all(),'finite F')
    return dict(C=C.astype(np.float64),original=original.astype(np.float64),F=F.astype(np.float64),vertices=vertices.astype(np.float64),faces=faces,K=K,query=query)

def graph_audit(d,g):
    faces=g['faces'];vertices=g['vertices'];K=g['K'];adj=[set() for _ in faces];edges=defaultdict(list)
    for i,t in enumerate(faces):
        for a,b in zip(t,np.roll(t,-1)):
            if a!=b:edges[tuple(sorted((int(a),int(b))))].append(i)
    stored=rows(d/'exact_edges.csv');ck(len(stored)==len(edges),'exact edge count')
    for r in stored:
        key=(int(r['vertex0']),int(r['vertex1']));inc=edges[key];owners=sorted(set(inc))
        ck(json.loads(r['incident_face_ids'])==owners,'exact edge owner set')
        ck(int(r['incidence_count'])==len(inc) and int(r['distinct_faces'])==len(owners),'raw/distinct incidence')
        ck(boolean(r['nonmanifold'])==(len(inc)>2),'nonmanifold flag')
        ck(boolean(r['boundary'])==(len(owners)==1),'boundary flag')
        ck(boolean(r['has_C'])==any(x<K for x in owners) and boolean(r['has_F'])==any(x>=K for x in owners),'C/F incidence')
        near(float(r['length']),np.linalg.norm(vertices[key[1]]-vertices[key[0]]),'edge length')
        for i in owners:adj[i].update(set(owners)-{i})
    def comps(ids):
        remain=set(ids);out=[]
        while remain:
            todo=[min(remain)];remain.remove(todo[0]);group=[]
            while todo:
                i=todo.pop();group.append(i);new=adj[i]&remain;remain-=new;todo.extend(new)
            out.append(sorted(group))
        return out
    sm=read(d/'summary.json')['graph'];allg=comps(range(len(faces)));freeg=comps(range(K,len(faces)))
    ck(sm['exact_components']==len(allg) and sm['exact_free_components']==len(freeg),'component count')
    ck(sm['exact_CF_edges']==sum(any(i<K for i in e) and any(i>=K for i in e) for e in edges.values()),'C/F observed edge count')
    ck(sm['nonmanifold_edges']==sum(len(e)>2 for e in edges.values()),'nonmanifold summary')
    face_rows=rows(d/'free_face_geometry.csv');regions=rows(d/'parallel_regions.csv');walls=rows(d/'sidewall_adjacency.csv')
    cn,ca=normal(g['original']);fn,fa=normal(g['F']);good=fa>1e-11*L*L;centers=g['F'].mean(1)
    witnesses=arrays(d/'near_parallel_witnesses.npz');owner=g['query']['C_point_faces'];wset=set(zip(owner[witnesses['query_id']].tolist(),witnesses['free_face_id'].tolist()))
    by_C=defaultdict(list)
    for r in face_rows:by_C[int(r['C_face_row'])].append(r)
    for ci,table in by_C.items():
        ck(len(table)==len(g['F']),'per C/F geometry denominator')
        signed=(g['F']-g['original'][ci].mean(0))@cn[ci]
        parallel=good&(abs(fn@cn[ci])>=np.cos(np.deg2rad(15)))
        corridor=(signed.min(1)<=.1*L)&(signed.max(1)>=-.1*L)
        pg=comps(K+np.flatnonzero(parallel&corridor));ng=comps(K+np.flatnonzero(~parallel&good&corridor))
        pr={i:k for k,m in enumerate(pg) for i in m};nr={i:k for k,m in enumerate(ng) for i in m}
        for r in table:
            f=int(r['free_face_id']);gi=f+K;ck(int(r['global_face_id'])==gi,'global/local face row')
            near(float(r['free_area']),fa[f],'free area');near(float(r['normal_abs_dot']),abs(fn[f]@cn[ci]),'normal dot')
            ck(boolean(r['normal_good'])==good[f] and boolean(r['parallel_15deg'])==parallel[f],'parallel/area filter')
            near([float(r['signed_vertex_min']),float(r['signed_vertex_max']),float(r['signed_centroid'])],[signed[f].min(),signed[f].max(),np.mean(signed[f])],'source C normal offsets')
            ck(boolean(r['plane_corridor_intersects'])==corridor[f],'corridor extent')
            ck(boolean(r['registered_near_witness'])==((ci,f) in wset),'candidate witness flag')
            ck(int(r['parallel_region_id'])==pr.get(gi,-1) and int(r['nonparallel_component_id'])==nr.get(gi,-1),'regional component labels')
        actual=[r for r in regions if int(r['C_face_row'])==ci];ck(len(actual)==len(pg),'parallel region count')
        for r,group in zip(actual,pg):
            fids=[x-K for x in group];ck(json.loads(r['free_face_ids'])==fids,'parallel region members')
            near(float(r['area']),fa[fids].sum(),'parallel region area')
            ck(boolean(r['has_registered_near_witness'])==any((ci,x) in wset for x in fids),'parallel witness')
        expected=[]
        for ni,group in enumerate(ng):
            nbr=set().union(*(adj[i] for i in group))-set(group);preg=sorted({pr[i] for i in nbr if i in pr});cc=sorted(i for i in nbr if i<K)
            if preg or cc:expected.append((ni,[i-K for i in group],preg,cc))
        actual=[r for r in walls if int(r['C_face_row'])==ci];ck(len(actual)==len(expected),'sidewall count')
        for r,(ni,ids,preg,cc) in zip(actual,expected):
            ck(int(r['nonparallel_component_id'])==ni and json.loads(r['free_face_ids'])==ids,'sidewall members')
            ck(json.loads(r['adjacent_parallel_region_ids'])==preg and json.loads(r['adjacent_C_face_rows'])==cc,'sidewall graph neighbors')
            ck(boolean(r['candidate_C_to_parallel'])==bool(preg and cc) and boolean(r['candidate_parallel_to_parallel'])==(len(preg)>=2),'sidewall proxy flags')
    return {'exact_CF_edges':sm['exact_CF_edges'],'RAW_graph_is_observational_only':True}

def cell_audit(cell):
    cid=cell['cell_id'];d=OUT/'diagnostics'/cid;s=read(d/'summary.json');receipt=read(OUT/'receipts'/f'{cid}.json')
    ck(receipt['status']=='PASS' and not receipt['model_framework_loaded'],'CPU receipt')
    for p,h in receipt['files'].items():ck(sha(p)==h,'receipt hash '+p)
    for p,h in s['files'].items():ck(sha(p)==h,'summary file hash '+p)
    ck(s['cell_id']==cid and s['status']=='PASS','summary identity')
    ck(s['new_model_calls']==0 and s['new_old_scorer_calls']==0,'zero neural / old scoring')
    for k,v in cell.items():
        if k.endswith('_path') and k[:-5]+'_sha256' in cell:
            ck(s['geometry_source_hashes'][str(path(v))]==cell[k[:-5]+'_sha256'],'summary source provenance '+k)
    g=load(cell);q0=g['query'];q=np.asarray(q0['C_points'],np.float64);owner=q0['C_point_faces'].astype(np.int64);weights=q0['C_weights'].astype(np.float64);border=q0['C_border_distance'].astype(np.float64)
    ck(len(q)==2048 and s['query_count']==2048,'all frozen queries');near(weights.sum(),1,'query denominator',1e-12)
    v=arrays(d/'query_geometry.npz');F=g['F'];M=len(F);cn,ca=normal(g['original']);fn,fa=normal(F);directions=cn[owner]
    eq(v['query_points'],q,'query coordinates');eq(v['query_weights'],weights,'query weights');eq(v['query_C_face'],owner,'query owners');near(v['source_C_normals'],directions,'source winding normals')
    goodC=ca>1e-11*L*L;goodF=fa>1e-11*L*L;interior=border>=.01*L;eq(v['old_interior'],interior,'original interior rule')
    nearest=np.empty(2048);parallel=np.full(2048,np.inf);allhits=[];allintervals=[];witness=[];witnessdist=[];maxdist=0.
    for start in range(0,2048,64):
        stop=min(2048,start+64);points=q[start:stop];dirs=directions[start:stop];D=distances(points,F);ii=np.arange(stop-start)
        nearest[start:stop]=np.sqrt(D.min(1));aligned=(abs(dirs@fn.T)>=np.cos(np.deg2rad(15)))&goodF[None]&goodC[owner[start:stop],None]
        parallel[start:stop]=np.sqrt(np.where(aligned,D,np.inf).min(1))
        ni=v['nearest_F_face_id'][start:stop];near(np.sqrt(D[ii,ni]),nearest[start:stop],'stored nearest face globally minimal')
        pi=v['nearest_parallel_F_face_id'][start:stop];has=pi>=0
        ck(np.array_equal(has,np.isfinite(parallel[start:stop])),'parallel finite/missing identity')
        if has.any():
            ck(aligned[ii[has],pi[has]].all(),'parallel selected normal valid');near(np.sqrt(D[ii[has],pi[has]]),parallel[start:stop][has],'nearest eligible parallel face')
        qualifies=aligned&(D<(.005*L)**2)&interior[start:stop,None];a,b=np.nonzero(qualifies)
        witness.extend(zip((a+start).tolist(),b.tolist()));witnessdist.extend(np.sqrt(D[a,b]).tolist())
        dot=dirs@fn.T;num=np.einsum('qfj,fj->qf',F[None,:,0]-points[:,None],fn)
        reg=(abs(dot)>1e-12)&goodF[None]&(np.linalg.norm(dirs,axis=1)>.5)[:,None]
        ts=np.divide(num,dot,out=np.zeros_like(dot),where=reg);hitpoint=points[:,None]+ts[:,:,None]*dirs[:,None]
        bc=bary(hitpoint,F);inside=reg&(bc>=-1e-9).all(-1)&(bc<=1+1e-9).all(-1);a,b=np.nonzero(inside)
        allhits.extend((int(aa+start),int(bb),float(ts[aa,bb])) for aa,bb in zip(a,b))
        cop=(abs(dot)<=1e-12)&goodF[None]&(abs(num)<=1e-9*L)&(np.linalg.norm(dirs,axis=1)>.5)[:,None]
        if cop.any():
            a0=bary(np.broadcast_to(points[:,None],(len(points),M,3)),F);a1=bary(np.broadcast_to((points+dirs)[:,None],(len(points),M,3)),F)
            for aa,bb in zip(*np.nonzero(cop)):
                lo,hi=-np.inf,np.inf;valid=True
                for val,slope in zip(a0[aa,bb],a1[aa,bb]-a0[aa,bb]):
                    if abs(slope)<1e-15:
                        if val < -1e-9:valid=False
                    elif slope>0:lo=max(lo,(-1e-9-val)/slope)
                    else:hi=min(hi,(-1e-9-val)/slope)
                if valid and np.isfinite(lo) and np.isfinite(hi) and lo<=hi:allintervals.append((aa+start,bb,lo,hi))
    near(v['nearest_F_distance'],nearest,'all finite nearest distances');near(v['nearest_parallel_F_distance'],parallel,'all eligible finite distances')
    maxdist=float(np.max(abs(v['nearest_F_distance']-nearest)))
    reconstructed=interior&goodC[owner]&(parallel<.005*L);eq(v['reconstructed_dup'],reconstructed,'independent old dup reconstruction')
    for name,idname,distname in [('nearest_F_point','nearest_F_face_id','nearest_F_distance'),('nearest_parallel_F_point','nearest_parallel_F_face_id','nearest_parallel_F_distance')]:
        ids=v[idname];has=ids>=0;point=v[name][has];tri=F[ids[has]]
        # One query/face pair per row, independently establish finite containment.
        n,_=normal(tri);near(np.sum((point-tri[:,0])*n,-1),0,'nearest lies on source triangle plane')
        bb=np.concatenate([bary(point[i:i+1,None],tri[i:i+1])[0] for i in range(len(point))],axis=0) if len(point) else np.empty((0,3))
        ck((bb>=-2e-7).all() and (bb<=1+2e-7).all(),'nearest lies inside finite face')
        near(np.linalg.norm(point-q[has],axis=1),v[distname][has],'saved nearest point distance')
    near(v['signed_nearest_point_offset'],np.sum((v['nearest_F_point']-q)*directions,-1),'signed source-normal nearest offset')
    near(v['signed_nearest_parallel_point_offset'],np.sum((v['nearest_parallel_F_point']-q)*directions,-1),'signed eligible offset')
    savedw=arrays(d/'near_parallel_witnesses.npz');expectedw=np.asarray(witness,np.int64).reshape(-1,2)
    eq(np.column_stack((savedw['query_id'],savedw['free_face_id'])),expectedw,'all qualifying near-parallel face witnesses');near(savedw['finite_distance'],witnessdist,'witness finite distances')
    counts=np.bincount(expectedw[:,1],minlength=M);cweights=np.bincount(expectedw[:,1],weights=weights[expectedw[:,0]],minlength=M)
    cr=rows(d/'candidate_faces.csv');ck(len(cr)==M,'candidate face denominator')
    h=arrays(d/'normal_line_hits.npz');expectedh=np.asarray(allhits,np.float64).reshape(-1,3)
    eq(np.column_stack((h['query_id'],h['free_face_id'])),expectedh[:,:2].astype(np.int64),'ALL finite-line hit completeness')
    near(h['signed_t'],expectedh[:,2],'finite-line signed t');eq(h['in_corridor'],abs(h['signed_t'])<=.1*L,'corridor membership')
    hp=q[h['query_id']]+h['signed_t'][:,None]*directions[h['query_id']];tri=F[h['free_face_id']]
    near(np.sum(h['barycentric'][:,:,None]*tri,axis=1),hp,'finite-hit barycentric point')
    near(h['barycentric'].sum(1),1,'finite-hit barycentric sum');ck((h['barycentric']>=-1e-9).all() and (h['barycentric']<=1+1e-9).all(),'finite-hit barycentric bounds')
    eq(h['barycentric_strict'],(h['barycentric']>1e-9).all(1)&(h['barycentric']<1-1e-9).all(1),'strict interior hit flag')
    near(h['normal_dot'],np.sum(directions[h['query_id']]*fn[h['free_face_id']],-1),'line F-normal dot')
    intervals=arrays(d/'coplanar_line_intervals.npz');expinterval=np.asarray(allintervals,np.float64).reshape(-1,4)
    eq(np.column_stack((intervals['query_id'],intervals['free_face_id'])),expinterval[:,:2].astype(np.int64),'all coplanar finite intervals')
    near(intervals['t_min'],expinterval[:,2],'coplanar interval lower');near(intervals['t_max'],expinterval[:,3],'coplanar interval upper')
    eq(intervals['intersects_corridor'],(intervals['t_min']<=.1*L)&(intervals['t_max']>=-.1*L),'coplanar corridor flag')
    layer=rows(d/'normal_line_layers.csv');gid=np.full(len(h['query_id']),-1);groupcounts=np.zeros(2048,np.int64);ng=0
    for qi in range(2048):
        inds=np.flatnonzero((h['query_id']==qi)&h['in_corridor']);inds=inds[np.lexsort((h['free_face_id'][inds],h['signed_t'][inds]))]
        pieces=np.split(inds,1+np.flatnonzero(np.diff(h['signed_t'][inds])>1e-4*L)) if len(inds) else []
        for li,ids in enumerate(pieces):
            r=layer[ng];gid[ids]=ng;ck(int(r['query_id'])==qi and int(r['query_layer_index'])==li,'layer query/local index')
            ck(int(r['hit_count'])==len(ids) and json.loads(r['free_face_ids'])==sorted(set(h['free_face_id'][ids].tolist())),'layer multiplicity and sorted face identity')
            near([float(r['t_min']),float(r['t_max']),float(r['signed_t_mean'])],[h['signed_t'][ids].min(),h['signed_t'][ids].max(),h['signed_t'][ids].mean()],'layer signed summary')
            ng+=1;groupcounts[qi]+=1
    ck(ng==len(layer)==s['corridor_layer_groups'],'layer total');eq(h['layer_group_id'],gid,'all hit group IDs');eq(v['corridor_layer_count'],groupcounts,'query group counts')
    ck(s['layer_count_query_histogram']=={str(k):int(vv) for k,vv in Counter(groupcounts).items()},'layer histogram')
    corridorface=np.bincount(h['free_face_id'][h['in_corridor']],minlength=M)
    for j,r in enumerate(cr):
        ck(int(r['free_face_id'])==j and boolean(r['qualifies'])==bool(counts[j]),'candidate face ID/flag')
        ck(int(r['qualifying_query_count'])==counts[j] and int(r['corridor_hit_query_count'])==corridorface[j],'candidate counts')
        near(float(r['qualifying_query_weight']),cweights[j],'candidate query weight')
        ck(json.loads(r['witness_C_face_rows'])==np.unique(owner[expectedw[expectedw[:,1]==j,0]]).tolist(),'candidate C witness IDs')
    ck(s['isolated_normal_hits']==len(h['query_id']) and s['coplanar_normal_line_intervals']==len(intervals['query_id']),'hit summary counts')
    ck(s['near_parallel_free_indices']==np.flatnonzero(counts).tolist(),'candidate summary')
    qr=rows(d/'per_query.csv');ck(len(qr)==2048,'query CSV denominator')
    lc=np.bincount(h['query_id'],minlength=2048);cc=np.bincount(h['query_id'][h['in_corridor']],minlength=2048);ic=np.bincount(intervals['query_id'],minlength=2048)
    for qi,r in enumerate(qr):
        ck(int(r['query_id'])==qi and int(r['C_face_row'])==owner[qi],'query CSV identity')
        ck([int(r[k]) for k in ('all_isolated_normal_hits','corridor_isolated_normal_hits','corridor_layer_groups','coplanar_normal_line_intervals')]==[lc[qi],cc[qi],groupcounts[qi],ic[qi]],'query CSV hit counts')
        ck(boolean(r['reconstructed_dup'])==reconstructed[qi] and boolean(r['old_interior'])==interior[qi],'query CSV masks')
    dup=None
    if cell['stage']!='SOURCE':
        old=arrays(cell['vectors_path']);eq(v['old_conflict_dup'],old['conflict_dup'],'frozen old mask');eq(reconstructed,old['conflict_dup'],'all 2048 original dup parity')
        oldscore=read(cell['score_path']);dup=float(weights@reconstructed/weights.sum());near(dup,oldscore['metrics']['ConflictCov_C_dup'],'original dup weighted fraction',1e-12)
        ck(s['old_dup_comparison']['status']=='PASS' and s['old_dup_comparison']['mismatched_queries']==0,'old dup comparison status')
    else:ck(s['old_dup_comparison']['status']=='NA_MISSING_OLD_MASK','source is local context / no inherited mask')
    graph=graph_audit(d,g)
    return dict(cell_id=cid,status='PASS',queries=2048,all_F_faces=M,all_query_F_pairs=2048*M,isolated_hits=len(h['query_id']),coplanar_intervals=len(intervals['query_id']),old_dup_fraction=dup,max_nearest_distance_absolute_error=maxdist,**graph)

def main():
    start=time.time();errors=[];results=[];reg=read(OUT/'registration.json');freeze=read(OUT/'science_freeze.json');protected=read(OUT/'protected_hashes.json')
    ck(sha(OUT/'registration.json')==freeze['registration_sha256'],'registration frozen hash')
    for p,h in freeze['code_hashes'].items():ck(sha(p)==h,'frozen implementation '+p)
    for p,h in protected.items():ck(sha(p)==h,'protected source before '+p)
    ck(len(protected)==210 and len(reg['cells'])==44 and len(reg['references'])==6 and len(reg['requests'])==11,'registered budget')
    ck(reg['budgets']['new_model_calls']==reg['budgets']['old_quality_rescores']==reg['budgets']['new_noise']==reg['budgets']['training_updates']==0,'zero neural/scorer/noise/train budget')
    for q in reg['requests']:
        cells=[c for c in reg['cells'] if c['request_id']==q['request_id']]
        ck({(x['label'],x['stage']) for x in cells}=={(m,s) for m in ('SMALL_C200','CREF_C200') for s in ('RAW','POST')},'paired 2x2 matrix')
    ck(Counter(x['selection_reason'] for x in reg['requests'])=={'RAW_NO_TO_YES':7,'RAW_YES_TO_UNCERTAIN':2,'HIGH_I_FF_TAIL':1,'INTERFACE_LOSS_STABLE_ADOPTION':1},'selection reasons')
    ck(any(x['short_id']=='85b169_n7' and x['selection_reason']=='INTERFACE_LOSS_STABLE_ADOPTION' for x in reg['requests']),'85 n7 is interface loss, not adoption loss')
    original={(x['label'],x['request_id'],x['stage']):x for x in rows(BASE/'evaluation/per_output.csv')};copy=rows(OUT/'inherited_per_output.csv')
    ck(len(copy)==44,'inherited old score rows')
    for r in copy:ck(r==original[(r['label'],r['request_id'],r['stage'])],'old output row unchanged')
    receipts=list((OUT/'receipts').glob('*.json'));ck(len(receipts)==50,'exactly 50 geometry receipts')
    ledger=[json.loads(x) for x in (OUT/'ledger.jsonl').read_text().splitlines()]
    completed=[x for x in ledger if x['event']=='CELL_COMPLETE'];ck(len(completed)==50,'ledger cell completion count')
    ck(len({x['cell_id'] for x in completed})==50,'no duplicate scientific cells')
    ck(all(not x['model_framework_loaded'] for x in completed),'all worker no framework')
    frozen=next(x['utc'] for x in ledger if x['event']=='SCIENCE_FROZEN');ck(all(x['utc']>frozen for x in completed),'freeze before every completion')
    for cell in reg['cells']+reg['references']:
        try:
            result=cell_audit(cell);results.append(result);print(cell['cell_id'],'PASS',flush=True)
        except BaseException as e:
            errors.append(dict(cell_id=cell['cell_id'],error=repr(e),traceback=traceback.format_exc()));print(cell['cell_id'],'FAIL',repr(e),flush=True)
    for p,h in protected.items():ck(sha(p)==h,'protected source after '+p)
    ck(not any(n=='torch' or n.startswith('torch.') for n in sys.modules),'independent process has no neural framework')
    result=dict(status='PASS' if not errors and len(results)==50 else 'FAIL',version='INDEPENDENT_CREF_LAYER_AUDIT_V1',utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.time()-start,checks=CHECKS,
        registration_sha256=sha(OUT/'registration.json'),science_freeze_sha256=sha(OUT/'science_freeze.json'),auditor_sha256=sha(__file__),protected_sources_checked_before_and_after=len(protected),
        cells=results,errors=errors,total_queries=sum(x['queries'] for x in results),total_finite_query_triangle_pairs=sum(x['all_query_F_pairs'] for x in results),
        new_model_calls=0,new_old_quality_scoring_calls=0,method='Independent dominant-axis finite-triangle distance/ray calculation for all 2048 queries x every free face; exact adjacency, sidewall, signed-gap clusters and old dup parity recomputed.',
        audit_development=['Attempt 01 preserved: audit assumed layer face IDs in t order rather than sorted unique, and used FP32 edge-length arithmetic instead of FP64. Only these audit-side errors were corrected; production outputs unchanged.'],
        limits=['Result-selected explanatory subset; not a population success-rate estimate.','Normal-line clusters are geometric observations, not semantic layers.','Source normal side is not inside/outside.','Observed exact adjacency may include nonmanifold or illegal contacts; not the previous valid-edge graph.','References are local thickness context only, not unique-GT success labels.','This numerical audit does not itself perform visual review or certify semantic structural thickness.'])
    HERE.mkdir(parents=True,exist_ok=True);(HERE/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    md=f"# Independent cached layer audit\n\nStatus: {result['status']}\n\n- Passed cells: {len(results)}/50 (44 model-stage cells and 6 source references).\n- Independent checks: {CHECKS}.\n- Frozen source files: {len(protected)} checked before and after.\n- Queries: {result['total_queries']}; finite query-triangle pairs: {result['total_finite_query_triangle_pairs']}.\n- No model, generation, training or original quality scorer calls.\n- All original dup masks are independently reproduced when present.\n- 85b/n7 retains its registered meaning: POST interface loss with stable RAW adoption.\n\nMethod: {result['method']}\n\n"+'\n'.join('- '+x for x in result['limits'])+'\n'
    if errors:md+='\nFailures:\n'+'\n'.join('- '+str(x) for x in errors)+'\n'
    (HERE/'audit.md').write_text(md,encoding='utf-8');print(json.dumps({k:result[k] for k in ('status','checks','wall_seconds','total_queries','total_finite_query_triangle_pairs')}),flush=True)
    return 0 if result['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
