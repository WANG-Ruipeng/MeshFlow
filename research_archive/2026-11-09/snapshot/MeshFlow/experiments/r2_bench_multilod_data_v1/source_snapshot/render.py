"""Actual source/candidate mesh inspection sheets; CPU matplotlib only."""
from . import common as c
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import html


def draw(ax,v,f,known,bounds,elev,azim,title):
    tri=v[f][...,[0,2,1]]
    colors=np.tile([.62,.65,.67,.82],(len(f),1))
    colors[np.asarray(known,dtype=np.int64)]=[.02,.58,.85,1.]
    p=Poly3DCollection(tri,facecolors=colors,edgecolors=(.16,.19,.21,.35),linewidths=.25,zsort='average')
    ax.add_collection3d(p)
    lo,hi=bounds
    for setter,i in ((ax.set_xlim,0),(ax.set_ylim,2),(ax.set_zlim,1)):setter(lo[i],hi[i])
    ax.set_box_aspect((1,1,1));ax.view_init(elev=elev,azim=azim)
    ax.set_axis_off();ax.set_title(title,fontsize=9)


def main():
    protocol=c.read(c.OUT/'protocol.json');records=[]
    directory=c.OUT/'gallery';directory.mkdir(exist_ok=True)
    for row in protocol['rows']:
        folder=c.OUT/'parents'/row['parent_id'];receipt=folder/'receipt.json'
        if not receipt.exists() or not (folder/'proposal.npz').exists():continue
        result=c.read(receipt);old=c.arrays(row['case_path']);new=c.arrays(folder/'proposal.npz')
        image=directory/(row['parent_id']+'.png')
        v,f=old['model_vertices'],old['source_faces'];nv,nf=new['model_vertices'],new['source_faces']
        lookup={int(oldid):i for i,oldid in enumerate(new['kept_original_face_ids'])}
        known=np.asarray([lookup[int(i)] for i in old['known_ids']],np.int64)
        used=v[np.unique(f)];center=(used.max(0)+used.min(0))*.5;extent=np.ptp(used,axis=0).max()*1.12
        bounds=(center-extent*.5,center+extent*.5)
        if not image.exists():
            fig=plt.figure(figsize=(15,10))
            views=((22,-60,'Front oblique'),(22,120,'Rear oblique'),(90,-90,'Top'))
            for i,(elev,azim,name) in enumerate(views):
                a=fig.add_subplot(3,3,i*3+1,projection='3d')
                draw(a,v,f,old['known_ids'],bounds,elev,azim,f'{name} | ORIGINAL N={len(f)}')
                a=fig.add_subplot(3,3,i*3+2,projection='3d')
                draw(a,nv,nf,known,bounds,elev,azim,f'{name} | CANDIDATE N={len(nf)}')
                a=fig.add_subplot(3,3,i*3+3,projection='3d')
                # Same-scale C-only reference makes the immutable input explicit.
                cf=f[old['known_ids']]
                draw(a,v,cf,np.arange(len(cf)),bounds,elev,azim,f'{name} | unchanged C, K={len(cf)}')
            fig.suptitle(f"{row['parent_id']} | {result['status']}\nFixed source frame and cameras; blue=C, gray=free. No generated output.",fontsize=12)
            fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.92,wspace=.01,hspace=.04)
            fig.savefig(image,dpi=125,facecolor='white');plt.close(fig)
        records.append(dict(parent_id=row['parent_id'],status=result['status'],image=str(image),sha256=c.sha(image),
                            original_N=len(f),candidate_N=len(nf),views=3))
    c.write(c.OUT/'gallery_manifest.json',dict(rows=records,kind='Source target augmentation inspection',model_calls=0))
    index=directory/'index.html'
    if not index.exists():
        index.write_text('<!doctype html><meta charset="utf-8"><title>Bench fixed-C multi-LOD</title><style>body{font:16px sans-serif}img{max-width:100%}section{border-top:1px solid #bbb}</style><h1>Bench fixed-C multi-LOD targets</h1>'+''.join(
            '<section><h2>'+html.escape(r['parent_id']+' '+r['status'])+'</h2><img loading="lazy" src="'+c.Path(r['image']).name+'"></section>' for r in records),encoding='utf-8')
    c.event('INSPECTION_RENDERED',sheets=len(records),model_calls=0)
    print(c.json.dumps(dict(sheets=len(records))))


if __name__=='__main__': main()
