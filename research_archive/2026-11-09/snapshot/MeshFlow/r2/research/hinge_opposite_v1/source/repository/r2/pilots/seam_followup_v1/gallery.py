"""CPU comparison renders from a frozen explicit mesh manifest; no metrics-to-labels."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse, concurrent.futures, multiprocessing, html, io, time, resource
from pathlib import Path
import numpy as np
from r2.pilots.seam_abc_v1 import eval_common as C
from r2.pilots.seam_abc_v1.render import measured_workers
from r2.pilots.regular_patch_control_v1 import depth_gallery as D
from r2.tools.cref_layer_audit_v1 import render as H

OUT=C.ROOT/'experiments/r2_seam_followup_v1'

def one(job):
    page,out,definition_sha,bounds,db=job
    out=Path(out);p=out/(page['page_id']+'.png');receipt=p.with_suffix('.json')
    if receipt.exists():
        r=C.read(receipt)
        assert r['definition_sha256']==definition_sha and C.sha(p)==r['image_sha256']
        return r
    tick=time.perf_counter();plt=H._mpl();nc=len(page['cells']);K=page['K']
    fig=plt.figure(figsize=(5.3*nc,13),dpi=110,facecolor='white')
    grid=fig.add_gridspec(4,nc,left=.025,right=.99,bottom=.05,top=.94,wspace=.15,hspace=.21)
    for col,cell in enumerate(page['cells']):
        a=C.arrays(cell['path']);tri=a['output'];g=dict(C=tri[:K],F=tri[K:],C_original=tri[:K],tri=tri)
        for row,b in enumerate([D.basis(-60,23),D.basis(120,23)]):
            im,_=D.raster(tri,K,b,np.array(db['center']),db['span'],size=420)
            ax=fig.add_subplot(grid[row,col]);ax.imshow(im);ax.set_axis_off();ax.set_title(cell['label']+' | '+str((-60,120)[row]),fontsize=10)
        for row,view in ((2,'front'),(3,'side')):
            ax=fig.add_subplot(grid[row,col]);H.draw_section(ax,g,[],bounds,C.L,view,1);ax.set_title(cell['label']+' | '+view,fontsize=9)
    fig.suptitle(page['request_id']+' | '+page['stage']+' | '+page['kind']+'\nBlue C / gray free; same scale and fixed original Gaussian',fontsize=12)
    fig.text(.5,.015,'Nonblind AI_PROXY only. Finite central sections may miss off-plane details. Images do not certify topology.',ha='center',fontsize=8)
    try:
        b=io.BytesIO();fig.savefig(b,format='png',dpi=110)
        with p.open('xb') as f:f.write(b.getvalue())
    finally:plt.close(fig)
    r=dict(page_id=page['page_id'],request_id=page['request_id'],stage=page['stage'],kind=page['kind'],
        path=str(p),image_sha256=C.sha(p),definition_sha256=definition_sha,seconds=time.perf_counter()-tick,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,status='PENDING_ACTUAL_VIEW',cells=page['cells'])
    C.write(receipt,r);return r

def run(kind):
    manifest=OUT/'gallery_manifests'/(kind+'.json');m=C.read(manifest);pages=m['pages']
    out=OUT/'gallery'/kind;out.mkdir(parents=True,exist_ok=True)
    hashes={};gg={};allpts={}
    for page in pages:
        cid=page['condition_id'];gg.setdefault(cid,[]);allpts.setdefault(cid,[])
        for cell in page['cells']:
            p=C.resolve(cell['path']);hashes[str(p)]=C.sha(p);tri=C.arrays(p)['output'];K=page['K']
            gg[cid].append(dict(C=tri[:K],F=tri[K:],C_original=tri[:K],tri=tri));allpts[cid].append(tri.reshape(-1,3))
    bounds={};db={}
    for cid,gs in gg.items():
        bounds[cid]=H.bounds_for_condition(gs,None,C.L);points=np.concatenate(allpts[cid]);center=(points.min(0)+points.max(0))/2
        span=max(np.ptp((points-center)@b,axis=0)[:2].max() for b in [D.basis(-60,23),D.basis(120,23)])
        db[cid]=dict(center=center.tolist(),span=float(span))
    definition=dict(version='seam_followup_gallery_v1',manifest_sha256=C.sha(manifest),code_sha256=C.sha(__file__),
        sources={str(Path(x.__file__)):C.sha(x.__file__) for x in (D,H)},geometry_hashes=hashes,bounds=bounds,depth_bounds=db,
        scale='Common per C across this comparison, all noises and registered stages',GT_display=False)
    C.write(out/'definition.json',definition);dh=C.sha(out/'definition.json')
    done=out/'completion.json'
    if done.exists():
        r=C.read(done);assert r['definition_sha256']==dh
        for im in r['images']:assert C.sha(im['path'])==im['image_sha256']
        print('RENDER_COMPLETE_VERIFIED',len(r['images']));return
    jobs=[(p,str(out),dh,bounds[p['condition_id']],db[p['condition_id']]) for p in pages]
    tick=time.perf_counter();C.write(out/'resources_before.json',C.resources())
    first=one(jobs[0]);snap=C.resources();workers,policy=measured_workers(snap,first)
    C.write(out/'dispatch.json',dict(policy=policy,resources=snap))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        images=[first,*list(pool.map(one,jobs[1:]))]
    page='<meta charset="utf-8"><h1>'+html.escape(kind)+'</h1>'
    page+=''.join('<h2>'+html.escape(r['page_id'])+'</h2><a href="'+Path(r['path']).name+'"><img loading="lazy" style="width:100%" src="'+Path(r['path']).name+'"></a>' for r in images)
    (out/'index.html').write_text(page,encoding='utf-8')
    C.write(done,dict(status='RENDER_COMPLETE_PENDING_REVIEW',definition_sha256=dh,images=images,workers=workers,wall_seconds=time.perf_counter()-tick))
    print('RENDER_COMPLETE',kind,len(images),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--kind',required=True);run(a.parse_args().kind)
