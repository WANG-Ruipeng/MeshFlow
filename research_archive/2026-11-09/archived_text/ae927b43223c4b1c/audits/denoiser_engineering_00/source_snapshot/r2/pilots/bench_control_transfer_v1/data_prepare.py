"""Prepare a frozen, source-index-preserving Bench condition pilot on CPU.

Reads the already downloaded archive in memory. Does not alter source meshes,
weld, deduplicate, repair, triangulate, or invoke any learned model. Regions are
geometric PLANAR_PATCH conditions, not asserted semantic seat annotations.
"""
from __future__ import annotations
import argparse, collections, hashlib, importlib, io, json, os, sys, tarfile
from pathlib import Path
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_k]='1'
import numpy as np
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_control_transfer_v1/data'
ARCHIVE=ROOT/'downloaded_data/archives/shapenet-cls.tar.gz'
sys.path.insert(0,str(ROOT/'r2/src'))
sys.path.insert(0,str(ROOT))
from meshflow_control.seam.slot_data import edge_features, boundary_from_C

def rank(uid,role='split'):
    return hashlib.sha256(('BENCH_CONTROL_TRANSFER_V1|20261018|'+role+'|'+uid).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),''):
            if not b:break
            h.update(b)
    return h.hexdigest()

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    txt=json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
    if path.exists():
        if path.read_text('utf-8')!=txt:raise FileExistsError(str(path))
    else:path.write_text(txt,encoding='utf-8')

def normalize(v):
    bounds=np.asarray([v.min(0),v.max(0)]); c1=(bounds[0]+bounds[1])/2
    d1=(bounds[1]-bounds[0]).max();a=(v-c1[None,:])/d1
    b2=np.asarray([a.min(0),a.max(0)]);c2=(b2[0]+b2[1])/2;a=a-c2[None,:]
    b3=np.asarray([a.min(0),a.max(0)]);c3=(b3[0]+b3[1])/2
    d3=(b3[1]-b3[0]).max();a=((a-c3[None,:])/d3).clip(-.5,.5)
    return a,dict(center1=c1.tolist(),divisor1=float(d1),center2=c2.tolist(),center3=c3.tolist(),divisor3=float(d3),dataset_std=.3762,model_scale=2.)

def preprocess(raw):
    v=np.asarray(raw['vertices'],np.float32);f=np.asarray(raw['faces'],np.int64)
    if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or not np.isfinite(v).all():raise ValueError('INVALID_MESH')
    if f.min()<0 or f.max()>=len(v) or np.ptp(v,axis=0).max()<=0:raise ValueError('INVALID_INDEX_OR_SPAN')
    nv,trans=normalize(v)
    vertex_order=np.lexsort((nv[:,0],nv[:,2],nv[:,1]));vr=np.argsort(vertex_order)
    p=np.argsort(vr[f],axis=1,kind='stable');pv=nv/np.float32(.3762)
    pre=pv[np.take_along_axis(f,p,axis=1)]
    begin=np.lexsort((pre[:,:,2],pre[:,:,1],pre[:,:,0]),axis=1)[:,0]
    p=np.take_along_axis(p,(begin[:,None]+np.arange(3)[None,:])%3,axis=1)
    cf=np.take_along_axis(f,p,axis=1);pre=pv[cf].astype(np.float32);mv=(pv*2).astype(np.float32)
    return dict(source_vertices=v,source_faces=f,normalized_vertices=nv,model_vertices=mv,
        full_target=(pre*2).astype(np.float32),pre_ot_full=pre,source_face_vertex_ids=cf,
        source_corner_permutations=p,transform_json=np.asarray(json.dumps(trans,sort_keys=True)))

def edgebook(f):
    e=collections.defaultdict(list)
    for i,t in enumerate(f):
        for j in range(3):e[tuple(sorted((int(t[j]),int(t[(j+1)%3]))))].append((i,int(t[j]),int(t[(j+1)%3])))
    return e

def regions(p):
    v=p['normalized_vertices'].astype(float);f=p['source_faces'];tri=v[f]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);twice=np.linalg.norm(normal,axis=1)
    normal=normal/np.maximum(twice[:,None],1e-30);cent=tri.mean(1)
    adj=[set() for _ in f]
    for owners in edgebook(f).values():
        for a,u,w in owners:
            for b,x,y in owners:
                if a>=b or (u,w)!=(y,x) or twice[a]<=1e-10 or twice[b]<=1e-10:continue
                if np.dot(normal[a],normal[b])>.999 and abs(np.dot(cent[b]-cent[a],normal[a]))<1e-4:
                    adj[a].add(b);adj[b].add(a)
    unseen=set(range(len(f)));candidates=[]
    while unseen:
        todo=[min(unseen)];comp=[]
        while todo:
            i=todo.pop()
            if i not in unseen:continue
            unseen.remove(i);comp.append(i);todo.extend(sorted(adj[i]&unseen,reverse=True))
        ids=np.asarray(sorted(comp),np.int64)
        if not 2<=len(ids)<=64 or twice[ids].min()<=1e-10:continue
        n=(normal[ids]*twice[ids,None]).sum(0);n/=np.linalg.norm(n)
        area=float(twice[ids].sum()/2)
        if area<.01:continue
        vids=np.unique(f[ids]);pts=v[vids];_,_,basis=np.linalg.svd(pts-pts.mean(0),full_matrices=False)
        planar=(pts-pts.mean(0))@basis[:2].T
        if np.min(np.ptp(planar,axis=0))<.05:continue
        hullarea=float(ConvexHull(planar).volume);compact=area/hullarea
        if compact<.60 or compact>1.001:continue
        lookup={int(u):j for j,u in enumerate(vids)};cf=np.asarray([[lookup[int(u)] for u in t] for t in f[ids]],np.int64)
        try:e=boundary_from_C(p['model_vertices'][vids],cf)
        except ValueError:continue
        if len(e)>len(f)-len(ids):continue
        y=float((cent[ids]*twice[ids,None]).sum(0)[1]/twice[ids].sum())
        # Prefer a broad horizontal surface above the bottom. Generic planar
        # fallback is explicitly labelled instead of inventing a seat meaning.
        horizontal=bool(abs(n[1])>=.9 and y>v[:,1].min()+.18*np.ptp(v[:,1]))
        candidates.append(dict(ids=ids,area=area,compactness=compact,horizontal=horizontal,normal=n.tolist(),center=cent[ids].mean(0).tolist(),boundary_edges=len(e)))
    return sorted(candidates,key=lambda x:(not x['horizontal'],-x['area'],-x['center'][1],tuple(x['ids'])))

def source_labels(p,known):
    from r2.data_tools.user_conditions_v1.quality import frozen_detector
    frozen_detector()
    detector=importlib.import_module('_r2_quality_frozen_geometry_v1.intersections')
    f=p['source_faces'];v=p['model_vertices'];vid=np.unique(f[known]);lookup={int(x):i for i,x in enumerate(vid)}
    original=np.asarray([[lookup[int(x)] for x in t] for t in f[known]],np.int64)
    canonical=np.asarray([[lookup[int(x)] for x in t] for t in p['source_face_vertex_ids'][known]],np.int64)
    edges,features=edge_features(v[vid],original,len(f));book=edgebook(f);ks=set(map(int,known));audit=[];byface=collections.defaultdict(list)
    owner=np.full(len(edges),-1,np.int64);triple=np.full((len(edges),3),-1,np.int64)
    supported=np.zeros(len(edges),bool);alias=np.zeros(len(edges),bool);labels=np.zeros(len(edges),bool)
    for ei,(ci,u,w) in enumerate(edges):
        su,sw=int(vid[u]),int(vid[w]);rows=book[tuple(sorted((su,sw)))];yr=[r for r in rows if r[0] not in ks]
        reason='NOT_SINGLE_C_SINGLE_FREE_INCIDENCE';cls='NA'
        if len(rows)==2 and len(yr)==1:
            fi,x,y=yr[0]
            if (x,y)!=(sw,su):reason='SOURCE_WINDING_NOT_OPPOSITE'
            else:
                q=detector.verified_pair(v[f[known[ci]]],v[f[fi]],float(np.linalg.norm(np.ptp(v,axis=0))))
                cls=q['classification'];reason='PASS' if cls=='shared_edge_contact' and q['verified'] else 'NOT_VERIFIED_LEGAL_CONTACT'
                if reason=='PASS':
                    third=[int(a) for a in f[fi] if a not in (su,sw)]
                    if len(third)!=1:reason='INVALID_THIRD'
                    else:supported[ei]=True;owner[ei]=fi;triple[ei]=[sw,su,third[0]];byface[fi].append(ei)
        audit.append(dict(edge_id=ei,source_u=su,source_v=sw,reason=reason,classification=cls,source_rows=[r[0] for r in rows]))
    for fi,es in byface.items():
        labels[min(es)]=True
        for ei in es:
            alias[ei]=ei!=min(es);audit[ei].update(owner_edge=min(es),owner_source_face=fi,role='OWNER' if ei==min(es) else 'ALIAS')
    for row in audit:row.setdefault('role','NULL_UNSUPPORTED')
    result=dict(C_vertices=v[vid],C_faces_original=original,C_faces_model=canonical,C_source_vertex_ids=vid,
        occupied_labels=labels,edge_owner_face=owner,edge_source_triple=triple,edge_features=features,
        source_supported_labels=supported,source_alias_labels=alias,source_third_in_C=np.asarray([bool(s and t[2] in lookup) for s,t in zip(supported,triple)]),
        known_ids=known,free_ids=np.setdiff1d(np.arange(len(f)),known),N=np.asarray(len(f)),K=np.asarray(len(known)))
    return result,audit

def geometry_hash(p):
    tri=p['normalized_vertices'][p['source_faces']];ordered=np.sort(tri.view([('x','<f4'),('y','<f4'),('z','<f4')]).reshape(tri.shape[:2]),axis=1)
    rows=np.ascontiguousarray(ordered).view('<f4').reshape(-1,9);rows=rows[np.lexsort(rows.T[::-1])]
    return hashlib.sha256(rows.tobytes()).hexdigest()

def load_archive():
    records=[];counts=collections.defaultdict(collections.Counter)
    with tarfile.open(ARCHIVE) as t:
        official={}
        for split in ('train','test'):
            z=np.load(io.BytesIO(t.extractfile('shapenet-02828884/split/'+split+'.npz').read()),allow_pickle=True)
            for r in z['npz_list']:official[str(r['uid'])]=split
        for m in t.getmembers():
            if not m.isfile() or '/objaverse_occ_v5_ids/' not in m.name or not m.name.endswith('.npz'):continue
            cat=m.name.split('/')[0]
            with np.load(io.BytesIO(t.extractfile(m).read()),allow_pickle=False) as z:
                n=len(z['faces']);counts[cat].update(total=1,n128_256=int(128<=n<=256))
                if cat!='shapenet-02828884' or not 128<=n<=256:continue
                raw={k:z[k].copy() for k in z.files}
            uid=Path(m.name).stem;p=preprocess(raw);rs=regions(p)
            records.append(dict(parent_id=uid,source_member=m.name,official_split=official.get(uid,'UNKNOWN'),N=n,
                candidate_count=len(rs),p=p,region=rs[0] if rs else None,exact_geometry_sha256=geometry_hash(p)))
    return records,dict(counts)

def render(records):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    OUT.joinpath('gallery').mkdir(exist_ok=True)
    for group in ('TRAIN','DEV'):
        subset=[r for r in records if r['group']==group]
        for start in range(0,len(subset),8):
            fig=plt.figure(figsize=(16,8))
            for j,r in enumerate(subset[start:start+8]):
                p=r['p'];a=r['case'];tri=p['model_vertices'][p['source_faces']];colors=np.tile([.70,.72,.74,1],(len(tri),1));colors[a['known_ids']]=[.08,.42,.95,1]
                for view,(elev,azim) in enumerate(((20,-65),(80,-90))):
                    ax=fig.add_subplot(4,4,(j//4)*8+(j%4)*2+view+1,projection='3d')
                    ax.add_collection3d(Poly3DCollection(tri,facecolors=colors,edgecolors=(.2,.2,.2,.3),linewidths=.2))
                    lo=tri.reshape(-1,3).min(0);hi=tri.reshape(-1,3).max(0);mid=(lo+hi)/2;rad=np.max(hi-lo)/2
                    ax.set(xlim=(mid[0]-rad,mid[0]+rad),ylim=(mid[1]-rad,mid[1]+rad),zlim=(mid[2]-rad,mid[2]+rad));ax.set_box_aspect((1,1,1));ax.view_init(elev=elev,azim=azim,vertical_axis='y');ax.set_axis_off()
                    ax.set_title(r['parent_id'][:6]+f" N={r['N']} K={r['K']}"+(' '+r['region_kind'] if view==0 else ''),fontsize=7)
            fig.tight_layout();fig.savefig(OUT/'gallery'/f'{group}_{start//8:02d}.png',dpi=120);plt.close(fig)

def render_c_only():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    m=json.loads((OUT/'manifest.json').read_text('utf-8'))
    dest=OUT/'gallery_C_only';dest.mkdir(exist_ok=True)
    for group in ('train','dev'):
        rows=m[group]
        for start in range(0,len(rows),16):
            fig,axs=plt.subplots(4,4,figsize=(14,12))
            for ax in axs.flat:ax.set_axis_off()
            for ax,r in zip(axs.flat,rows[start:start+16]):
                with np.load(r['case_path'],allow_pickle=False) as z:v=z['C_vertices'].astype(float);f=z['C_faces_original']
                center=v.mean(0);_,_,basis=np.linalg.svd(v-center,full_matrices=False);p=(v-center)@basis[:2].T
                ax.add_collection(PolyCollection(p[f],facecolors='#3486df',edgecolors='#15365c',linewidths=.6))
                span=np.max(np.ptp(p,axis=0));mid=(p.min(0)+p.max(0))/2
                ax.set_xlim(mid[0]-.56*span,mid[0]+.56*span);ax.set_ylim(mid[1]-.56*span,mid[1]+.56*span);ax.set_aspect('equal')
                ax.set_title(r['parent_id'][:8]+f"  K={r['K']}  owner={r['source_owner_count']}\n"+r['region_kind'],fontsize=8)
            fig.tight_layout();fig.savefig(dest/(group.upper()+'_'+str(start//16).zfill(2)+'.png'),dpi=130);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scan-only',action='store_true');parser.add_argument('--render-c-only',action='store_true');parser.add_argument('--train-count',type=int,default=64);parser.add_argument('--dev-count',type=int,default=8);args=parser.parse_args()
    if args.render_c_only:render_c_only();return
    records,counts=load_archive();eligible=[r for r in records if r['region'] is not None]
    unique=[];seen=set()
    for r in sorted(eligible,key=lambda r:rank(r['parent_id'],'geometry_representative')):
        if r['exact_geometry_sha256'] not in seen:unique.append(r);seen.add(r['exact_geometry_sha256'])
    summary=dict(archive=str(ARCHIVE),archive_sha256=sha(ARCHIVE),category_counts=counts,bench_N_eligible=len(records),bench_planar_eligible=len(eligible),bench_unique_planar_eligible=len(unique),official_split_counts=dict(collections.Counter(r['official_split'] for r in unique)))
    print(json.dumps(summary,indent=2),flush=True)
    if args.scan_only:return
    ordered=sorted(unique,key=lambda r:rank(r['parent_id']));nd=min(args.dev_count,max(1,len(ordered)//5))
    official_test=[r for r in ordered if r['official_split']=='test'];official_train=[r for r in ordered if r['official_split']!='test']
    dev=(official_test+official_train)[:nd];dev_ids={r['parent_id'] for r in dev}
    train=[r for r in ordered if r['parent_id'] not in dev_ids][:args.train_count]
    if len(train)<16 or len(dev)<4:raise RuntimeError('Insufficient eligible independent parents for bounded transfer pilot')
    for group,rs in (('TRAIN',train),('DEV',dev)):
        for r in rs:r['group']=group
    selected=train+dev;manifest=[];all_audit=[]
    OUT.mkdir(parents=True,exist_ok=True);OUT.joinpath('cases').mkdir(exist_ok=True)
    for r in selected:
        known=r['region']['ids'];case,audit=source_labels(r['p'],known);case.update(r['p']);r['case']=case;r['K']=len(known)
        r['region_kind']='HORIZONTAL_PLANAR_PATCH' if r['region']['horizontal'] else 'PLANAR_PATCH'
        cid=r['parent_id']+'__PLANAR_FULL';dest=OUT/'cases'/(cid+'.npz')
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:
                assert set(z.files)==set(case)
                for k,a in case.items():assert np.array_equal(z[k],a),(dest,k)
        else:
            with dest.open('xb') as h:np.savez_compressed(h,**case)
        assert np.array_equal(case['pre_ot_full']*np.float32(2),case['full_target'])
        assert np.array_equal(case['C_vertices'][case['C_faces_model']],case['full_target'][known])
        f=r['p']['source_faces'];tri=r['p']['model_vertices'][f]
        dup=len(f)-len(np.unique(np.sort(f,axis=1),axis=0));zero=int((np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)<=1e-12).sum())
        m=dict(case_path=str(dest),case_sha256=sha(dest),parent_id=r['parent_id'],condition_id=cid,N=r['N'],K=r['K'],group=r['group'],official_split=r['official_split'],official_pretraining_exposure='PRETRAIN_TRAIN_SPLIT' if r['official_split']=='train' else 'OFFICIAL_TEST_SPLIT',source_archive_member=r['source_member'],region_kind=r['region_kind'],region_source_face_ids=known.tolist(),exact_geometry_sha256=r['exact_geometry_sha256'],region_area_normalized=r['region']['area'],region_compactness=r['region']['compactness'],C_boundary_edges=len(case['occupied_labels']),source_owner_count=int(case['occupied_labels'].sum()),source_alias_count=int(case['source_alias_labels'].sum()),source_null_unsupported_count=int((~case['source_supported_labels']).sum()),source_exact_duplicate_ID_faces=dup,source_degenerate_faces=zero,source_quality_policy='RECORDED_NOT_FILTERED_NO_REPAIR',full_FF_audit='NOT_RUN')
        manifest.append(m)
        for a in audit:all_audit.append(dict(parent_id=r['parent_id'],condition_id=cid,group=r['group'],**a))
    trainm=[r for r in manifest if r['group']=='TRAIN'];devm=[r for r in manifest if r['group']=='DEV'];eval_train=sorted(trainm,key=lambda r:rank(r['parent_id'],'eval'))[:4]
    requests=[]
    for r in eval_train+devm:
        for ni in range(2):
            seed=int(rank(r['parent_id']+'|'+str(ni),'noise')[:8],16)%(2**31-1)
            requests.append(dict(request_id=r['parent_id'][:8]+'_n'+str(ni),parent_id=r['parent_id'],condition_id=r['condition_id'],case_path=r['case_path'],inference_npz=r['case_path'],metadata_npz=r['case_path'],group=r['group'],split=r['group'],official_split=r['official_split'],official_pretraining_exposure=r['official_pretraining_exposure'],N=r['N'],K=r['K'],noise_id=ni,noise_index=ni,noise_seed=seed))
    summary.update(train_count=len(trainm),dev_count=len(devm),request_count=len(requests),train_owner_count=sum(r['source_owner_count'] for r in trainm),dev_owner_count=sum(r['source_owner_count'] for r in devm),split_policy='SHA256 parent-group split before any model outputs; exact normalized geometry duplicates represented once',condition_definition='Largest coherent nearly planar source-index region; prefer broad horizontal surface above bottom; K2..64, no C repair; no semantic seat annotation claim',normalization='Exact official full-object FP32 transform; /0.3762 pre-OT, x2 model; C never independently normalized')
    save_json(OUT/'manifest.json',dict(schema='bench_control_transfer_data_v1',train=trainm,dev=devm,evaluation_requests=requests,summary=summary))
    save_json(OUT/'train_manifest.json',trainm);save_json(OUT/'dev_manifest.json',devm);save_json(OUT/'evaluation_requests.json',requests);save_json(OUT/'source_interface_audit.json',all_audit);save_json(OUT/'summary.json',summary)
    render(selected);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
