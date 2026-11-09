"""Render strictly mapped historical regions for new, actual AI_PROXY review.

Only the region face IDs proven to map bitwise are used. No automatic candidate
selection, semantic acceptance or model calls are performed by this renderer.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from r2.pilots.user_data_scaling_12h_v1 import evaluation as E
from data_review import half_region

def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    path=E.OUT/'data/annotated_existing_proposals.json';existing=E.read(path)
    candidates=E.read(E.OUT/'data/parent_candidates.json')['rows'];lookup={(p['parent_uid'],p['variant_uid']):p for p in candidates}
    grouped=collections.defaultdict(list)
    for row in existing['rows']:
        if row['eligible_for_root_review']:grouped[row['parent_uid']].append(row)
    chosen=[]
    for uid,rows in grouped.items():
        variants=sorted({r['variant_uid'] for r in rows},key=lambda v:lookup[(uid,v)]['variant_rank_key'])
        variants=[v for v in variants if lookup[(uid,v)]['finite_and_valid_indices'] and lookup[(uid,v)]['exact_zero_area_faces']==0]
        if not variants:continue
        variant=variants[0];p=lookup[(uid,variant)];rr={r['region_type']:r for r in rows if r['variant_uid']==variant}
        chosen.append((p,rr))
    chosen.sort(key=lambda item:item[0]['candidate_rank']);sheets=[];tick=time.perf_counter()
    for offset in range(0,len(chosen),4):
        E.deadline();group=chosen[offset:offset+4];fig=plt.figure(figsize=(24,3.3*len(group)));records=[]
        for row_index,(parent,region_rows) in enumerate(group):
            a=E.arrays(parent['npz']);tri=a['full_target'].reshape(-1,3,3);N=len(tri)
            if E.sha(parent['npz'])!=parent['sha256']:raise ValueError('Parent changed before mapped-region rendering')
            full={k:region_rows[k.lower()]['mapped_face_ids'] if k.lower() in region_rows else [] for k in ('SEAT','BACK')};half={};meta={}
            for k in full:half[k],meta[k]=half_region(parent,full[k],k) if full[k] else ([],dict(status='NA_NO_PRIOR_ACCEPTED_REGION'))
            panels=[('M front',[],15,90,False),('M side',[],10,0,False),('SEAT FULL',full['SEAT'],28,58,False),('SEAT HALF',half['SEAT'] or [],28,58,False),
                    ('BACK FULL',full['BACK'],18,60,False),('BACK HALF',half['BACK'] or [],18,60,False),('BACK opposite',full['BACK'],18,-120,False),('C isolated',full['SEAT']+full['BACK'],28,58,True)]
            xyz=tri.reshape(-1,3).astype(float);lo,hi=xyz.min(0),xyz.max(0);mid=(lo+hi)/2;span=float((hi-lo).max())*1.08;plot=tri[:,:,[0,2,1]]
            cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal=cross/np.maximum(np.linalg.norm(cross,axis=1)[:,None],1e-30);shade=.45+.5*np.abs(normal@np.array([.25,.7,.6]))
            for col,(title,ids,elev,azim,isolated) in enumerate(panels):
                ax=fig.add_subplot(len(group),8,row_index*8+col+1,projection='3d');color=np.repeat(shade[:,None],4,axis=1);color[:,3]=1
                if ids:color[ids]=[.08,.55,.95,1]
                shown=np.asarray(ids,np.int64) if isolated else np.arange(N)
                if len(shown):ax.add_collection3d(Poly3DCollection(plot[shown],facecolors=color[shown],edgecolors=(.1,.12,.15,.38),linewidths=.25))
                center=mid[[0,2,1]];ax.set_xlim(center[0]-span/2,center[0]+span/2);ax.set_ylim(center[1]-span/2,center[1]+span/2);ax.set_zlim(center[2]-span/2,center[2]+span/2)
                ax.set_box_aspect((1,1,1));ax.set_proj_type('ortho');ax.view_init(elev,azim);ax.set_axis_off();ax.set_title(title+' K='+str(len(ids)),fontsize=9)
                if col==0:ax.text2D(.01,.01,f"{offset+row_index:02d} {parent['uid'][:8]} N={N}\n{parent['variant_uid']}",transform=ax.transAxes,fontsize=8)
            records.append(dict(index=offset+row_index,uid=parent['uid'],variant_uid=parent['variant_uid'],npz=parent['npz'],parent_sha256=parent['sha256'],
                full_ids=full,half_ids=half,half_meta=meta,prior_regions=region_rows,current_review_status='NOT_REVIEWED',auto_accept=False))
        fig.subplots_adjust(left=0,right=1,bottom=.005,top=.98,wspace=0,hspace=.02)
        image=E.checked(E.OUT/'data/existing_region_review_sheets'/f'mapped_{offset:02d}_{offset+len(group)-1:02d}.png');image.parent.mkdir(parents=True,exist_ok=True)
        identity=dict(source_mapping_sha256=E.sha(path),renderer_sha256=E.sha(__file__),rows=records)
        if image.exists():
            record=E.read(image.with_suffix('.json'))
            if record['identity']!=E.clean(identity) or record['image_sha256']!=E.sha(image):raise ValueError('Existing mapped-region sheet differs')
        else:
            fig.savefig(image,dpi=125,facecolor='white');record=dict(image=str(image),image_sha256=E.sha(image),status='NOT_REVIEWED',identity=identity);E.write(image.with_suffix('.json'),record)
        plt.close(fig);sheets.append(record);print(str(image),flush=True)
    E.write(E.OUT/'data/existing_region_review_sheets/index.json',dict(status='RENDERED_NOT_REVIEWED',parent_count=len(chosen),sheets=sheets,seconds=time.perf_counter()-tick,new_model_calls=0,new_quality_scans=0))
    return sheets

if __name__=='__main__':render()
