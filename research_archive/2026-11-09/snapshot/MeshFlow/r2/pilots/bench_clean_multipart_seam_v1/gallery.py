"""48 request pages, main+aux sheets, exactly inherited Bench cut positions."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,concurrent.futures,html,io,multiprocessing,resource,time
from pathlib import Path
import numpy as np
from .eval_prepare import OUT,OLD,HISTORICAL,P,requests,verify
from .comparison import cpu_dest
from r2.pilots.chair_multipart_seam_v1 import gallery as G
from r2.pilots.chair_back_transfer_v1.evaluate import worker_count
C,D,H=G.C,G.D,G.H
KIND='BENCH_CLEAN_FIXED48_MAIN_AUX'

def cell(q,model,stage):
    base=(OLD if model in HISTORICAL else OUT)/'evaluation'/model/q['request_id']
    if stage=='CLEAN_FINAL_POST':mesh=cpu_dest(model,q)/stage/'post.npz';receipt=cpu_dest(model,q)/'complete.json'
    else:
        kind='raw' if stage=='RAW' else 'post';mesh=base/kind/(kind+'.npz');receipt=base/kind/'receipt.json'
    return dict(model=model,stage=stage,label=model+' '+stage,path=str(mesh),receipt=str(receipt),request_id=q['request_id'])

def prepare():
    dp=OUT/'gallery_manifests'/(KIND+'.json')
    if dp.exists():
        value=P.read(dp);verify(value['protected']);return value
    definition=OLD/'gallery/NATIVE_S21000__SEAM_S21000_full/definition.json'
    old=P.read(definition);qq=requests()
    specs={
      'MAIN':[(m,s) for m in ('SEAM_S15000','NEW_HEAD') for s in ('RAW','BASE_POST','CLEAN_FINAL_POST')],
      'AUX':[(m,s) for m in ('NATIVE_S21000','SEAM_S21000','OLD_S15000_LAYOUT') for s in ('BASE_POST','CLEAN_FINAL_POST')]}
    pages=[]
    for q in qq:
      for sheet,items in specs.items():
        pages.append(dict(**{k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','N','K','split','part_type')},
            page_id=sheet+'__'+q['request_id'],sheet=sheet,cells=[cell(q,m,s) for m,s in items],
            inference_npz=q['inference_npz'],kind='Bench '+q['part_type']+' | '+q['split'],
            bounds=old['bounds'][q['condition_id']],depth_bounds=old['depth_bounds'][q['condition_id']]))
    value=dict(status='FROZEN',version='bench_clean_original_views_main_aux_v1',pages=pages,request_pages=48,sheets=96,
        protected={str(p):P.sha(p) for p in (definition,OUT/'eval_manifest.json',Path(__file__),Path(G.__file__),Path(D.__file__),Path(H.__file__))},
        cameras='Inherited azimuth-60/120 elevation23; exact original center/span for all models',
        sections='Original central front/side finite sections; same coordinates and bounds for every stage/model',
        source_definition=str(definition),roles='BlueC orangeSEAM grayBULK greenBRIDGE; role is notquality',
        review='PENDING_ACTUAL_NONBLIND_AI_PROXY; 48requestpages eachMAIN+AUX; render is not review',
        model_calls=0,new_scores=0,geometry_edits=0,GT_display=False)
    P.immutable(dp,value);return value

def one(job):
    page,out,definition_sha=job;out=Path(out);png=out/(page['page_id']+'.png');receipt=png.with_suffix('.json')
    if receipt.exists():
        r=P.read(receipt);assert r['definition_sha256']==definition_sha and P.sha(png)==r['image_sha256'];return r
    tick=time.perf_counter();plt=H._mpl();K=page['K'];bounds=page['bounds'];depth=page['depth_bounds']
    fig=plt.figure(figsize=(31.8,13.5),dpi=110,facecolor='white')
    grid=fig.add_gridspec(4,6,left=.025,right=.99,bottom=.055,top=.91,wspace=.15,hspace=.23)
    a=C.arrays(page['inference_npz']);original=a['C_vertices'][a['C_faces_original']];display=[]
    for col,c in enumerate(page['cells']):
        mesh=C.arrays(c['path']);tri,role=G.roles(mesh,K)
        g=dict(C=tri[:K],F=tri[K:],C_original=original,tri=tri)
        facts=dict(label=c['label'],role_counts={r:int((role==r).sum()) for r in ('C','SEAM','BULK','BRIDGE')},views=[])
        title=c['model']+'\n'+c['stage']
        for row,az in enumerate((-60,120)):
            image,scope=G.role_raster(tri,role,K,D.basis(az,23),np.array(depth['center']),depth['span'])
            ax=fig.add_subplot(grid[row,col]);ax.imshow(image);ax.set_axis_off();ax.set_title(title+' | '+str(az),fontsize=9)
            facts['views'].append(dict(azimuth=az,**scope))
            if scope['projected_vertices_outside_inherited_frame']:ax.text(.01,.01,'Inherited-frame clipping: see receipt',transform=ax.transAxes,fontsize=7,color='#b34b27')
        for row,view in ((2,'front'),(3,'side')):
            ax=fig.add_subplot(grid[row,col]);seams=np.flatnonzero(role[K:]=='SEAM')
            H.draw_section(ax,g,seams,bounds,C.L,view,1)
            from matplotlib.collections import LineCollection
            axis,project=(2,(0,1)) if view=='front' else (0,(2,1))
            pos=bounds['front_positions' if view=='front' else 'side_positions'][1]
            ff,ids=H.finite_sections(g['F'],axis,pos,1e-12*C.L);bm=role[K:][ids]=='BRIDGE'
            if bm.any():ax.add_collection(LineCollection(ff[bm][:,:,project],colors='#41a46c',linewidths=1.6,zorder=3))
            ax.set_title(c['model']+' | '+view+f'={pos:.5f}',fontsize=8)
        display.append(facts)
    fig.suptitle(page['request_id']+' | '+page['kind']+f" | {page['sheet']} | K={K} N={page['N']}\n"
        'Blue C | orange SEAM | gray BULK | green added BRIDGE. Same noise, original camera/scale/cuts.\n'
        'Historical steps are unequal-budget references; NEW_HEAD versus OLD_S15000_LAYOUT shares the new generator.',fontsize=12)
    fig.text(.5,.018,'AI_PROXY actual review recorded separately. Cross-section endpoints are not proof of cracks; topology/intersection require numerical evidence. No source-style grading.',ha='center',fontsize=9)
    try:
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=110)
        with png.open('xb') as f:f.write(buffer.getvalue())
    finally:plt.close(fig)
    r=dict(status='PENDING_ACTUAL_VIEW',request_id=page['request_id'],page_id=page['page_id'],condition_id=page['condition_id'],
        part_type=page['part_type'],split=page['split'],sheet=page['sheet'],path=str(png),image_sha256=P.sha(png),
        definition_sha256=definition_sha,cells=page['cells'],display=display,seconds=time.perf_counter()-tick,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_calls=0,scoring_calls=0)
    P.immutable(receipt,r);return r

def run(workers=2):
    plan=prepare();out=OUT/'gallery'/KIND;sources={}
    for page in plan['pages']:
        for c in page['cells']:sources.update(G.authenticate(c))
    out.mkdir(parents=True,exist_ok=True)
    definition=dict(manifest_sha256=P.sha(OUT/'gallery_manifests'/(KIND+'.json')),sources=sources,protected=plan['protected'])
    dp=out/'definition.json';P.immutable(dp,definition);dh=P.sha(dp)
    done=out/'completion.json'
    if done.exists():
        r=P.read(done)
        for image in r['images']:assert P.sha(image['path'])==image['image_sha256']
        return r
    jobs=[(page,str(out),dh) for page in plan['pages']];tick=time.perf_counter()
    first=one(jobs[0]);snap=C.resources();workers=worker_count(snap,first,workers)
    P.immutable(out/'dispatch.json',dict(workers=workers,resources=snap,first_peak_RSS_KiB=first['peak_RSS_KiB'],GPU_calls=0))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        images=[first,*list(pool.map(one,jobs[1:]))]
    by=collections_by_request(images)
    text='<meta charset="utf-8"><h1>Bench CLEAN: all48 fixed requests</h1><p>Each request has MAIN and AUX comparison sheets. Same original cameras and cuts; actual AI_PROXY review is separate.</p>'
    for rid,imgs in by.items():
        text+='<h2>'+html.escape(rid)+'</h2>'
        for im in imgs:
            name=Path(im['path']).name;text+='<h3>'+im['sheet']+'</h3><a href="'+name+'"><img loading="lazy" style="width:100%" src="'+name+'"></a>'
    (out/'index.html').write_text(text,encoding='utf8')
    r=dict(status='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW',request_pages=48,image_sheets=len(images),images=images,
        definition_sha256=dh,workers=workers,seconds=time.perf_counter()-tick,model_calls=0,new_scores=0)
    P.immutable(done,r);return r

def collections_by_request(images):
    from collections import defaultdict
    out=defaultdict(list)
    for im in images:out[im['request_id']].append(im)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    r=prepare() if a.action=='prepare' else run(a.workers);print({k:v for k,v in r.items() if k not in ('pages','images','protected')},flush=True)
