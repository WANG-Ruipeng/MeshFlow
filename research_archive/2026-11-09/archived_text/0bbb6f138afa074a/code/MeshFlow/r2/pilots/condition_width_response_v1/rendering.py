
"""Four fixed views of all three C-width variants, RAW and POST separately.

No metric ranking is read. Rendering never supplies visual judgments: the
96-row template is NOT_REVIEWED until a person/AI actually inspects all pages.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, collections, html, json, time
from pathlib import Path
import numpy as np
from . import common as E
from . import evaluate as Q

CAMERAS=(('front',(5.,-90.)),('side',(5.,0.)),('oblique',(22.,-60.)),('wire',(22.,-60.)))
ORDER=('m095','base','p105')
TITLES={'m095':'C- | alpha=0.95','base':'C0 | alpha=1.00','p105':'C+ | alpha=1.05'}
GALLERY=E.OUT/'evaluation/gallery'

def wire(ax,mesh,K,bounds,camera):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    for segment,color in ((mesh[:K],'#2387c9'),(mesh[K:],'#555e66')):
        if not len(segment):continue
        edges=np.concatenate([segment[:,[a,b]][:,:,[0,2,1]] for a,b in ((0,1),(1,2),(2,0))])
        ax.add_collection3d(Line3DCollection(edges,colors=color,linewidths=.4,alpha=1.))
    center=np.asarray(bounds['center_model_xyz'])[[0,2,1]];half=bounds['span']/2
    ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),zlim=(center[2]-half,center[2]+half))
    ax.set_proj_type('ortho');ax.set_box_aspect((1,1,1));ax.view_init(elev=camera[0],azim=camera[1]);ax.set_axis_off()

def definitions():
    return {str(p):E.sha(p) for p in (Path(__file__),E.CODE/'common.py',E.CODE/'evaluate.py',
        E.ROOT/'r2/tools/student_multistep_denoise_v1/gallery.py')}

def run():
    reg=Q.registration();raw,post=Q.indices(reg);definition=definitions()
    manifest=GALLERY/'manifest.json'
    if manifest.exists():
        prior=E.read(manifest)
        if prior['status']!='RENDERED_NOT_REVIEWED' or prior['images']!=32 or prior['definition_hashes']!=definition:
            raise ValueError('Completed gallery definition differs')
        for rr in prior['receipts']:
            Q.verify({rr['image_path']:rr['image_sha256']})
            if E.read(E.resolve(rr['image_path']).with_suffix('.json'))!=rr:raise ValueError('Page sidecar changed')
        Q.verify(prior['support_hashes'])
        return prior
    before=GALLERY/'resources_before.json'
    if before.exists():before=GALLERY/('resources_resume_'+str(time.time_ns())+'.json')
    E.resources(before)
    G=E.module('_width_original_mesh_renderer',E.ROOT/'r2/tools/student_multistep_denoise_v1/gallery.py')
    import matplotlib.pyplot as plt
    from PIL import Image
    grouped=collections.defaultdict(list)
    for q in reg['requests']:grouped[q['base_request_id']].append(q)
    receipts=[];cards=[];review=[];tick=time.perf_counter()
    for base_rid,rows in sorted(grouped.items(),key=lambda item:(item[1][0]['selection_order'],item[1][0]['candidate'])):
        variants={q['variant']:q for q in rows};K=int(rows[0]['K']);meshes={};hashes={};input_hashes={};Cs={}
        for variant in ORDER:
            q=variants[variant];C=Q.old().read_input(q)['C'];_,src=Q.load_source(q,C);Cs[variant]=C
            input_hashes[q['input_path']]=q['input_sha256'];input_hashes[q['condition_source']]=q['condition_source_sha256']
            for stage,mapping,key in (('RAW',raw,'raw_path'),('POST',post,'post_path')):
                row=mapping[q['request_id']];path=row[key];a=E.arrays(path)
                tri=np.asarray(a['raw'] if stage=='RAW' else a['output']).reshape(-1,3,3)
                if tri.dtype!=np.float32 or not np.isfinite(tri).all():raise ValueError('Nonfinite/non-FP32 render input')
                if not E.same(tri[:K],C if stage=='RAW' else src['C_original']):raise ValueError('Render C identity fails')
                if stage=='POST' and not E.same(a['vertices'][a['faces']],tri):raise ValueError('POST index/triangle mismatch')
                meshes[variant,stage]=tri;hashes[str(path)]=E.sha(path)
        xyz=np.concatenate([*[c.reshape(-1,3) for c in Cs.values()],*[m.reshape(-1,3) for m in meshes.values()]]).astype(float)
        lo,hi=xyz.min(0),xyz.max(0);span=float((hi-lo).max())*1.12
        if not np.isfinite(span) or span<=0:raise ValueError('Degenerate shared display bounds')
        bounds=dict(center_model_xyz=((lo+hi)/2).tolist(),span=span,bounds_model_xyz=[lo.tolist(),hi.tolist()],
            same_all_three_variants_and_RAW_POST=True,normalization=False,remote_fragments_included=True)
        sections=[]
        for stage in ('RAW','POST'):
            dest=GALLERY/stage/(base_rid+'.jpg');rp=dest.with_suffix('.json')
            identity=E.clean(dict(base_request_id=base_rid,parent_id=rows[0]['parent_id'],stage=stage,label=Q.LABEL,
                requests=[{k:variants[v][k] for k in Q.META} for v in ORDER],
                variants=list(ORDER),cameras=CAMERAS,bounds=bounds,hashes=hashes,input_hashes=input_hashes,
                registration_sha256=E.sha(E.OUT/'registration.json'),definition_hashes=definition,
                dimensions=[1560,1820],colors='C blue; F gray',all_geometry_visible=True,blind=False,
                screenshot_claim='Four orthographic views do not certify attachment or reveal every occluded triangle'))
            if rp.exists():
                rr=E.read(rp)
                if rr['identity']!=identity or E.sha(dest)!=rr['image_sha256']:raise ValueError('Frozen render differs')
            else:
                if dest.exists():raise RuntimeError('Incomplete page requires explicit engineering recovery')
                fig=plt.figure(figsize=(12,14),dpi=130,facecolor='white')
                for i,(view,camera) in enumerate(CAMERAS):
                    for j,variant in enumerate(ORDER):
                        ax=fig.add_subplot(4,3,i*3+j+1,projection='3d');mesh=meshes[variant,stage]
                        if view=='wire':wire(ax,mesh,K,bounds,camera)
                        else:G.draw(ax,mesh,K,bounds,camera)
                        ax.set_title(TITLES[variant]+' | F='+str(len(mesh)-K)+' | '+view,fontsize=10)
                q=rows[0]
                fig.suptitle(q['parent_id']+' | '+q['region_type']+' '+q['tier']+' | n'+str(q['candidate'])+' | '+stage+'\nEXP67 frozen | N='+str(q['N'])+' | K='+str(K),fontsize=12)
                fig.text(.5,.015,'Blue: preserved variant C / Gray: free. Fixed views and union RAW+POST scale. No smoothing or fragment removal.',ha='center',fontsize=8)
                fig.text(.5,.005,'Known variants; nonblind AI_PROXY review. Proximity in an image does not certify a legal shared-edge connection.',ha='center',fontsize=8)
                fig.subplots_adjust(left=0,right=1,top=.95,bottom=.03,wspace=0,hspace=0)
                dest=E.checked(dest);dest.parent.mkdir(parents=True,exist_ok=True);tmp=E.checked(dest.with_name(dest.name+'.tmp'))
                with tmp.open('xb') as f:fig.savefig(f,format='jpg',dpi=130,pil_kwargs={'quality':88,'optimize':True})
                plt.close(fig);os.replace(tmp,dest)
                with Image.open(dest) as im:im.verify()
                Q.verify(hashes);Q.verify(input_hashes)
                rr=dict(status='RENDERED_NOT_REVIEWED',identity=identity,image_path=str(dest),image_sha256=E.sha(dest));E.write(rp,rr)
            receipts.append(rr)
            sections.append('<details open><summary>'+stage+'</summary><a href="'+stage+'/'+html.escape(base_rid)+'.jpg"><img loading="lazy" src="'+stage+'/'+html.escape(base_rid)+'.jpg" style="width:100%"></a></details>')
            for variant in ORDER:
                q=variants[variant]
                review.append(dict(**{k:q[k] for k in Q.META},label=Q.LABEL,display_label=variant,stage=stage,
                    image_path=str(dest),image_sha256=rr['image_sha256'],review_status='NOT_REVIEWED',reviewer='AI_PROXY',blind=False,
                    reviewed_image_sha256='',**{f:'UNKNOWN' for f in Q.REVIEW_FIELDS},notes=''))
        cards.append('<section><h2>'+html.escape(base_rid)+'</h2>'+''.join(sections)+'</section>')
        E.write(GALLERY/'progress.json',dict(status='RENDERING',images=len(receipts),registered_images=32),replace=True)
        print(json.dumps(dict(stage='RENDER',base_request_id=base_rid,images=len(receipts),registered_images=32)),flush=True)
    if len(receipts)!=32 or len(review)!=96:raise ValueError('Wrong gallery/review budget')
    page='<!doctype html><meta charset="utf-8"><title>EXP67 condition width response</title><h1>All eight parents, two original noises, three C scales: RAW / POST</h1><p>Front, side, oblique and wire. C-/C0/C+ are explicitly labeled: nonblind AI_PROXY review. Shared scale includes every fragment across all three variants and both stages of each noise request. Rendering alone is not completed review.</p>'+''.join(cards)
    index=GALLERY/'index.html';E._write_bytes(index,page.encode('utf8'))
    template=GALLERY/'AI_PROXY_review_template.csv';E.csvwrite(template,review)
    mapping=GALLERY/'variant_mapping.json';E.write(mapping,dict(order=ORDER,titles=TITLES,blind=False,
        original_C_array_is_not_the_transformed_variant='Every variant is validated against its own frozen C and source mapping'))
    value=dict(status='RENDERED_NOT_REVIEWED',requests=48,base_requests=16,images=32,review_rows=96,views_per_image=4,
        receipts=receipts,definition_hashes=definition,support_hashes={str(p):E.sha(p) for p in (index,template,mapping)},
        seconds=time.perf_counter()-tick,actual_review_status='NOT_RUN_BY_RENDERER',no_metric_rank_read=True,models_loaded=0,model_calls=0)
    E.write(manifest,value)
    E.write(GALLERY/'progress.json',dict(status='RENDERED_NOT_REVIEWED',images=32,registered_images=32),replace=True)
    return value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if args.check_only:
        reg=Q.registration()
        result=dict(status='CPU_RENDER_INTERFACE_CHECK_PASS',requests=len(reg['requests']),expected_images=32,
            actual_renders=0,model_calls=0,scientific_scans=0,definition_files=len(definitions()))
    else:result=run()
    print(json.dumps({k:v for k,v in result.items() if k not in ('receipts','definition_hashes','support_hashes')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()

