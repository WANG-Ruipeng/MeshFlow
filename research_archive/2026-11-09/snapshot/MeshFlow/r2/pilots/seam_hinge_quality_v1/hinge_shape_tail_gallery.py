"""Two explicitly selected local shape tails from the saved baseline HINGE audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .evaluation import C,OUT,mesh_path,requests


def run():
    dest=OUT/'visual/hinge_shape_tails';dest.mkdir(parents=True,exist_ok=True)
    qs={q['request_id']:q for q in requests()};result=[]
    for rid,slot in [('73b369b9_n3',192),('608b46b0_n0',54)]:
        p=dest/(rid+'_slot'+str(slot)+'.png');rp=p.with_suffix('.json')
        if rp.exists():
            r=C.read(rp);assert C.sha(p)==r['image_sha256'];result.append(r);continue
        q=qs[rid];meshes={};protected={};targets=[]
        for stage in ('POST','HINGE015','HINGE025'):
            source=mesh_path('PLAIN_S2000',rid,'POST') if stage=='POST' else OUT/'hinge/cells'/('PLAIN_S2000_'+stage)/rid/'post.npz'
            a=C.arrays(source);protected[str(source)]=C.sha(source);meshes[stage]=a
            index=np.flatnonzero(a['source_face_ids']==slot);assert len(index)==1
            targets.append(a['output'][index[0]])
        allp=np.concatenate(targets);low=allp.min(0)-.025*C.L;high=allp.max(0)+.025*C.L
        center=(low+high)/2;half=max(high-low)/2
        fig=plt.figure(figsize=(15,10),dpi=150)
        for col,(stage,a) in enumerate(meshes.items()):
            tri=a['output'];target=np.flatnonzero(a['source_face_ids']==slot)[0]
            near=np.all(tri.max(1)>=center-half,axis=1)&np.all(tri.min(1)<=center+half,axis=1)
            ids=np.flatnonzero(near)
            for row,az in enumerate((-60,120)):
                ax=fig.add_subplot(2,3,row*3+col+1,projection='3d')
                for fid in ids:
                    color='#cf3427' if fid==target else '#1686c0' if fid<q['K'] else '#a3a3a3'
                    coll=Poly3DCollection([tri[fid][:,[0,2,1]]],facecolors=color,edgecolors='#8c1b12' if fid==target else '#666666',
                        linewidths=2.0 if fid==target else .35,alpha=.9 if fid==target else .25)
                    ax.add_collection3d(coll)
                ax.set_xlim(center[0]-half,center[0]+half);ax.set_ylim(center[2]-half,center[2]+half)
                ax.set_zlim(center[1]-half,center[1]+half)
                ax.set_box_aspect((1,1,1));ax.view_init(elev=20,azim=az)
                ax.set(xlabel='X',ylabel='Z',zlabel='Y',title=stage+f' / azimuth {az}')
                ax.tick_params(labelsize=7)
        fig.suptitle(rid+f' | tracked original RAW slot {slot} | same local 3D bounds; two views',fontsize=14)
        fig.text(.5,.02,'Red selected face; blue C; translucent gray neighboring saved faces. Selected from numeric shape-tail audit, not a random sample.',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.045,1,.96))
        with p.open('xb') as f:fig.savefig(f,format='png')
        plt.close(fig)
        r=dict(status='PENDING_ACTUAL_REVIEW',request_id=rid,source_raw_slot=slot,path=str(p),image_sha256=C.sha(p),
            sources=protected,code_sha256=C.sha(__file__),bounds_center=center.tolist(),bounds_halfwidth=float(half),
            selection='Explicit largest relative face expansion / largest relative area shrinkage from independent HINGE shape-tail audit',
            view='Two local 3D camera views; translucency is for target visibility, not a final photoreal rendering',model_calls=0)
        C.write(rp,r);result.append(r)
    return result

if __name__=='__main__':print([r['path'] for r in run()],flush=True)
