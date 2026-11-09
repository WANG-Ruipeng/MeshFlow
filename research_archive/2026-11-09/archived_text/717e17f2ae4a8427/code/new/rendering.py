"""Complete RAW/POST panel: C plus all methods, one shared camera and scale.

Rendered does not mean reviewed. Writes an unreviewed AI_PROXY template only;
actual visual observations must be supplied after inspecting the images.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, html, io, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.user_data_scaling_32_67_v1 import evaluation as E

CAMERAS=(('front',(5.,-90.)),('side',(5.,0.)),('oblique',(22.,-60.)),('wire',(22.,-60.)))

def wire(ax,mesh,K,bounds,camera):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    for segment,color in ((mesh[:K],'#2387c9'),(mesh[K:],'#555e66')):
        edges=np.concatenate([segment[:,[a,b]][:,:,[0,2,1]] for a,b in ((0,1),(1,2),(2,0))])
        ax.add_collection3d(Line3DCollection(edges,colors=color,linewidths=.4,alpha=1.))
    center=np.asarray(bounds['center_model_xyz'])[[0,2,1]];half=bounds['span']/2
    ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),zlim=(center[2]-half,center[2]+half))
    ax.set_proj_type('ortho');ax.set_box_aspect((1,1,1));ax.view_init(elev=camera[0],azim=camera[1]);ax.set_axis_off()

def run():
    E.deadline();reg=E.load_registration();raw=E.collect_raw()['rows'];post=E.read(E.OUT/'evaluation/post_index.json')['rows']
    rawmap={(r['label'],r['request_id']):r for r in raw};postmap={(r['label'],r['request_id']):r for r in post}
    labels=reg['labels'];anonymous={label:chr(65+i) for i,label in enumerate(labels)}
    E.write(E.OUT/'evaluation/gallery/method_mapping.json',dict(mapping=anonymous,blind_review=False,
        note='Anonymous display names do not establish blinding; operator has access to mapping and registration'))
    G=E.module('_scaling_original_mesh_renderer',E.ROOT/'r2/tools/student_multistep_denoise_v1/gallery.py')
    import matplotlib.pyplot as plt
    from PIL import Image
    receipts=[];cards=[];review=[];tick=time.perf_counter()
    for q in reg['requests']:
        E.deadline();rid=q['request_id'];K=int(q['K']);meshes={};hashes={};C=E.old().read_input(q)['C']
        for label in labels:
            for stage,mapping,key in (('RAW',rawmap,'raw_path'),('POST',postmap,'post_path')):
                row=mapping[(label,rid)];path=row[key];a=E.arrays(path)
                tri=np.asarray(a['raw'] if stage=='RAW' else a['output']).reshape(-1,3,3)
                meshes[(label,stage)]=tri;hashes[str(path)]=E.sha(path)
        # Union includes all RAW and POST geometry, even remote fragments.
        xyz=np.concatenate([C.reshape(-1,3),*[m.reshape(-1,3) for m in meshes.values()]]).astype(float)
        lo,hi=xyz.min(0),xyz.max(0);span=float((hi-lo).max())*1.12
        if not np.isfinite(span) or span<=0:raise ValueError('Degenerate shared display bounds')
        bounds=dict(center_model_xyz=((lo+hi)/2).tolist(),span=span,bounds_model_xyz=[lo.tolist(),hi.tolist()])
        sections=[]
        for stage in ('POST','RAW'):
            dest=E.OUT/'evaluation/gallery'/stage/(rid+'.jpg');rp=dest.with_suffix('.json')
            identity=dict(request_id=rid,stage=stage,labels=labels,cameras=CAMERAS,bounds=bounds,hashes=hashes,
                dimensions=[1560,1820],colors='C blue; F gray',all_geometry_visible=True,renderer_sha256=E.sha(__file__))
            identity=E.clean(identity)
            if rp.exists():
                rr=E.read(rp)
                if rr['identity']!=identity or E.sha(dest)!=rr['image_sha256']:raise ValueError('Frozen render differs')
            else:
                fig=plt.figure(figsize=(12,14),dpi=130);columns=1+len(labels)
                for i,(view,camera) in enumerate(CAMERAS):
                    for j,label in enumerate([None,*labels]):
                        ax=fig.add_subplot(4,columns,i*columns+j+1,projection='3d')
                        mesh=C if label is None else meshes[(label,stage)]
                        if view=='wire':wire(ax,mesh,K,bounds,camera)
                        else:G.draw(ax,mesh,K,bounds,camera)
                        ax.set_title(('Input C' if label is None else anonymous[label]+' | free '+str(len(mesh)-K))+' | '+view,fontsize=9)
                fig.suptitle(q['parent_id']+' | '+q['region_type']+' '+q['tier']+' | n'+str(q['candidate'])+' | '+stage,fontsize=11)
                fig.text(.5,.012,'Blue C / Gray free. Shared RAW+POST bounds and orthographic views. No smoothing, normalization or hidden fragment removal.',ha='center',fontsize=8)
                fig.subplots_adjust(left=0,right=1,top=.96,bottom=.028,wspace=0,hspace=0)
                dest=E.checked(dest);dest.parent.mkdir(parents=True,exist_ok=True);tmp=E.checked(dest.with_name(dest.name+'.tmp'))
                with tmp.open('xb') as f:fig.savefig(f,format='jpg',dpi=130,pil_kwargs={'quality':88,'optimize':True})
                plt.close(fig);os.replace(tmp,dest)
                with Image.open(dest) as im:im.verify()
                rr=dict(status='RENDERED_NOT_REVIEWED',identity=identity,image_path=str(dest),image_sha256=E.sha(dest));E.write(rp,rr)
            receipts.append(rr);sections.append('<details '+('open' if stage=='POST' else '')+'><summary>'+stage+'</summary><a href="'+stage+'/'+rid+'.jpg"><img loading="lazy" src="'+stage+'/'+rid+'.jpg" style="width:100%"></a></details>')
        cards.append('<section><h2>'+html.escape(rid)+'</h2>'+''.join(sections)+'</section>')
        for label in labels:
            for stage in ('RAW','POST'):
                review.append(dict(request_id=rid,parent_id=q['parent_id'],condition_id=q['condition_id'],candidate=q['candidate'],panel=q['panel'],
                    region_type=q['region_type'],tier=q['tier'],label=label,display_label=anonymous[label],stage=stage,
                    review_status='NOT_REVIEWED',reviewer='AI_PROXY',blind=False,reviewed_image_sha256='',
                    C_adoption='UNKNOWN',C_floating='UNKNOWN',C_replaced_by_other_surface='UNKNOWN',C_wrong_layer='UNKNOWN',
                    whole_body_complete='UNKNOWN',support_missing='UNKNOWN',fragments_or_folded_layers='UNKNOWN',visible_join='UNKNOWN',notes=''))
    page='<!doctype html><meta charset="utf-8"><title>USER data scaling complete comparison</title><h1>All registered requests: RAW / POST</h1><p>Views: front, side, oblique and wire. Display mapping is stored separately; no claim of blinding. Rendering is not completed visual review.</p>'+''.join(cards)
    p=E.checked(E.OUT/'evaluation/gallery/index.html');p.write_text(page,encoding='utf8')
    template=E.OUT/'evaluation/gallery/AI_PROXY_review_template.csv'
    if not template.exists():E.csvwrite(template,review)
    E.write(E.OUT/'evaluation/gallery/manifest.json',dict(status='RENDERED_NOT_REVIEWED',requests=len(reg['requests']),images=len(receipts),views_per_image=4,receipts=receipts,
        seconds=time.perf_counter()-tick,actual_review_status='NOT_RUN_BY_RENDERER',no_metric_rank_read=True))
    return receipts

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();run()
