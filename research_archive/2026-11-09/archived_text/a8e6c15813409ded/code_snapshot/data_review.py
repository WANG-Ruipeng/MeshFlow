"""Input-only region sheets and immutable, explicitly entered AI_PROXY reviews."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import argparse, collections, heapq, json, time
from pathlib import Path
import numpy as np
from data_prepare import OUT,ROOT,old,proposal,read,save,sha,rank

def half_region(parent,ids,kind):
    with np.load(parent['npz'],allow_pickle=False) as z:t=z['full_target'].reshape(-1,3,3);f=z['source_faces']
    cent=t.astype(np.float64).mean(1);area=np.linalg.norm(np.cross(t[:,1].astype(float)-t[:,0],t[:,2].astype(float)-t[:,0]),axis=1)/2
    ids=sorted(ids);edges=collections.defaultdict(list);adj={i:set() for i in ids}
    for i in ids:
        for j in range(3):edges[tuple(sorted((int(f[i,j]),int(f[i,(j+1)%3]))))].append(i)
    for fs in edges.values():
        if len(fs)==2:adj[fs[0]].add(fs[1]);adj[fs[1]].add(fs[0])
    groups=proposal.components(ids,adj)
    if len(groups)!=1:return None,dict(status='NA_DISCONNECTED_REGION_HALF_NEEDS_MEANINGFUL_SELECTION',components=len(groups))
    pts=t[ids].reshape(-1,3);lo=pts.min(0);hi=pts.max(0);anchor=(lo+hi)/2
    anchor[0]=lo[0]
    if kind=='SEAT':anchor[2]=lo[2]
    else:anchor[1]=hi[1]
    boundary=set(i for fs in edges.values() if len(fs)==1 for i in fs)
    seed=min(boundary or set(ids),key=lambda i:(float(np.sum((cent[i]-anchor)**2)),i))
    heap=[(0.,seed)];dist={seed:0.};order=[];seen=set()
    while heap:
        d,i=heapq.heappop(heap)
        if i in seen:continue
        seen.add(i);order.append(i)
        for j in sorted(adj[i]):
            nd=d+float(np.linalg.norm(cent[i]-cent[j]))
            if nd<dist.get(j,float('inf')):dist[j]=nd;heapq.heappush(heap,(nd,j))
    assert len(order)==len(ids)
    cumulative=np.cumsum(area[order]);target=area[ids].sum()*.5
    k=min(range(1,len(order)+1),key=lambda k:(abs(cumulative[k-1]-target),k))
    selected=sorted(order[:k]);alias=selected==ids
    return selected,dict(status='ALIAS_FULL' if alias else 'UNREVIEWED',seed_face_id=seed,area_fraction=float(area[selected].sum()/area[ids].sum()),growth='source-edge Dijkstra centroid-distance prefix; area nearest .5; ID ties')

def render(start=0,stop=None,manifest=None,prefix="front"):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    rows=read(manifest or OUT/'data/parent_candidates.json')['rows'];rows=[r for r in rows if r['variant_order']==0]
    rows.sort(key=lambda r:r['candidate_rank'])
    stop=min(len(rows),len(rows) if stop is None else stop)
    index=[]
    for b in range(start,stop,4):
        group=rows[b:min(b+4,stop)];fig=plt.figure(figsize=(19.2,3.1*len(group)));entry=[]
        for ri,r in enumerate(group):
            p=read(r['proposal_path']);cand={c['candidate_id']:c for c in p['candidates']}
            with np.load(r['npz'],allow_pickle=False) as z:t=z['full_target'].reshape(-1,3,3)
            ids={};half={};meta={}
            for k in ('SEAT','BACK'):
                cid=p['proposed_regions'][k];ids[k]=[] if cid is None else cand[cid]['source_face_ids']
                half[k],meta[k]=([],{'status':'NA'}) if not ids[k] else half_region(r,ids[k],k)
            panels=[('M front',[],15,90),('M side',[],10,0),
                ('SEAT '+str(p['proposed_regions']['SEAT'])+' FULL',ids['SEAT'],28,58),
                ('SEAT HALF',half['SEAT'] or [],28,58),
                ('BACK '+str(p['proposed_regions']['BACK'])+' FULL',ids['BACK'],18,60),
                ('BACK HALF',half['BACK'] or [],18,60)]
            bounds=np.array([t.reshape(-1,3).min(0),t.reshape(-1,3).max(0)]);mid=bounds.mean(0);span=float(np.ptp(bounds,axis=0).max())*1.08
            plot=t[:,:,[0,2,1]]
            cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);norm=cross/np.maximum(np.linalg.norm(cross,axis=1)[:,None],1e-30)
            shade=.45+.5*np.abs(norm@np.array([.25,.7,.6]))
            for col,(title,keep,elev,azim) in enumerate(panels):
                ax=fig.add_subplot(len(group),6,ri*6+col+1,projection='3d');colors=np.repeat(shade[:,None],4,axis=1);colors[:,3]=1
                if keep:colors[keep]=[.08,.55,.95,1]
                ax.add_collection3d(Poly3DCollection(plot,facecolors=colors,edgecolors=(.1,.12,.15,.36),linewidths=.24))
                center=mid[[0,2,1]]
                ax.set_xlim(center[0]-span/2,center[0]+span/2);ax.set_ylim(center[1]-span/2,center[1]+span/2);ax.set_zlim(center[2]-span/2,center[2]+span/2)
                ax.set_box_aspect((1,1,1));ax.set_proj_type('ortho');ax.view_init(elev,azim);ax.set_axis_off()
                ax.set_title(title+' K='+str(len(keep)),fontsize=9)
                if col==0:ax.text2D(.02,.02,f"{b+ri:03d} {r['uid'][:8]} N={r['N']}",transform=ax.transAxes,fontsize=9)
            entry.append(dict(index=b+ri,uid=r['uid'],variant_uid=r['variant_uid'],proposed_ids=ids,half_ids=half,half_meta=meta,proposal_path=r['proposal_path']))
        fig.subplots_adjust(left=0,right=1,bottom=.005,top=.98,wspace=0,hspace=.02)
        dest=OUT/'data/review_sheets'/f'{prefix}_{b:03d}_{b+len(group)-1:03d}.png';dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():fig.savefig(dest,dpi=125,facecolor='white')
        plt.close(fig);record=dict(image=str(dest),image_sha256=sha(dest),status='NOT_REVIEWED',rows=entry)
        save(dest.with_suffix('.json'),record);index.append(record)
        print(str(dest),flush=True)
    return index

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int);ap.add_argument('--manifest');ap.add_argument('--prefix',default='front');a=ap.parse_args();render(a.start,a.stop,a.manifest,a.prefix)
