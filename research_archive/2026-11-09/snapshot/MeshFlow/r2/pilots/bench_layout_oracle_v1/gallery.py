"""Fixed cameras and original finite sections; image creation never makes ratings."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
import concurrent.futures
import html
import io
import json
import multiprocessing
import resource
import sys
import time
import numpy as np
from . import common as C, evaluation as E

GALLERY=C.OUT/'gallery'
VERSION='bench_layout_oracle_four_mode_three_stage_gallery_v1'


def mesh_spec(label,q,stage):
    e=E.engine()
    if stage=='RAW':
        folder=e.producer(label,q,'raw');path=folder/'raw.npz';receipt=folder/'receipt.json'
    elif stage=='BASE_POST':
        folder=e.producer(label,q,'post');path=folder/'post.npz';receipt=folder/'receipt.json'
    else:
        path=E.root_for(label)/'cpu'/label/q['request_id']/stage/'post.npz'
        receipt=E.root_for(label)/'cpu'/label/q['request_id']/'complete.json'
    saved=C.read(receipt);C.verify(saved['files'])
    if not any(C.resolve(p)==path and h==C.sha(path) for p,h in saved['files'].items()):
        raise ValueError('Image source mesh is not sealed by its original producer')
    return dict(label=label,path=str(path),sha256=C.sha(path),producer=str(receipt),producer_sha256=C.sha(receipt))


def render_page(page):
    if 'torch' in sys.modules:raise RuntimeError('Gallery must run in fresh CPU process')
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    q,stage=page['q'],page['stage'];pageid=stage+'__'+q['request_id']
    folder=GALLERY/pageid;png=folder/'comparison.png';receipt=folder/'comparison.json'
    identity=dict(version=VERSION,page_id=pageid,stage=stage,request_id=q['request_id'],
        source_sha256=C.sha(__file__),cells=page['cells'],bounds=q['gallery_bounds'],depth_bounds=q['gallery_depth_bounds'])
    if receipt.exists():
        r=C.read(receipt)
        if r['identity']!=identity or C.sha(png)!=r['image_sha256']:raise ValueError('Frozen image changed')
        return r
    if png.exists():raise RuntimeError('Unsealed image; no overwrite')
    folder.mkdir(parents=True,exist_ok=True);tick=time.perf_counter();plt=G.H._mpl()
    fig=plt.figure(figsize=(20,16.5),dpi=110,facecolor='white')
    grid=fig.add_gridspec(len(E.LABELS),4,left=.025,right=.99,bottom=.055,top=.89,wspace=.19,hspace=.29)
    depth,bounds=q['gallery_depth_bounds'],q['gallery_bounds'];display=[]
    try:
        for row,cell in enumerate(page['cells']):
            k=int(q['K']);arr=C.arrays(cell['path']);tri,roles=G.roles(arr,k)
            inp=C.arrays(q['inference_npz']);original=inp['C_vertices'][inp['C_faces_original']]
            if not np.array_equal(tri[:k],original):raise ValueError('Rendered C differs from input')
            geometry=dict(C=tri[:k],F=tri[k:],C_original=original,tri=tri)
            facts=dict(label=cell['label'],stage=stage,N_requested=int(q['N']),N_output=len(tri),K=k,views=[])
            for col,azimuth in enumerate((-60,120)):
                raster,scope=G.role_raster(tri,roles,k,G.D.basis(azimuth,23),np.asarray(depth['center']),depth['span'])
                ax=fig.add_subplot(grid[row,col]);ax.imshow(raster);ax.set_axis_off()
                ax.set_title(cell['label']+f' | {stage} | az={azimuth}\nK={k}, requestedN={q["N"]}, outputN={len(tri)}',fontsize=8)
                facts['views'].append(dict(azimuth=azimuth,**scope))
                if scope['projected_vertices_outside_inherited_frame']:
                    ax.text(.01,.01,'Inherited frame clipping; see receipt',transform=ax.transAxes,fontsize=7,color='#b34b27')
            for col,view in ((2,'front'),(3,'side')):
                ax=fig.add_subplot(grid[row,col]);seams=np.flatnonzero(roles[k:]=='SEAM')
                G.H.draw_section(ax,geometry,seams,bounds,G.C.L,view,1)
                position=bounds['front_positions' if view=='front' else 'side_positions'][1]
                from matplotlib.collections import LineCollection
                axis,project=(2,(0,1)) if view=='front' else (0,(2,1))
                sections,ids=G.H.finite_sections(geometry['F'],axis,position,1e-12*G.C.L)
                bridge=roles[k:][ids]=='BRIDGE'
                if bridge.any():ax.add_collection(LineCollection(sections[bridge][:,:,project],colors='#41a46c',linewidths=1.6,zorder=3))
                ax.set_title(cell['label']+' | '+stage+' | '+view+f'={position:.5f}',fontsize=8)
            display.append(facts)
        fig.suptitle(q['request_id']+' | '+stage+' | '+q['report_group']+'\n'
            'EXPOSURE / PLAIN1000 / LAYOUT1000 (SOURCE ORACLE) / LAYOUT_NULL | Same C, N, saved noise, cameras and finite sections\n'
            'Blue: condition C | Gray: generated bulk | Green: added bridge',fontsize=11)
        fig.text(.5,.018,'Actual AI_PROXY viewing is recorded separately. Sections and graph components alone do not certify cracks/fragments. No unique-source-style grading.',ha='center',fontsize=8)
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=110)
        with png.open('xb') as f:f.write(buffer.getvalue())
    finally:plt.close(fig)
    r=dict(status='PENDING_ACTUAL_VIEW',identity=identity,page_id=pageid,stage=stage,image_path=str(png),
        image_sha256=C.sha(png),display=display,seconds=time.perf_counter()-tick,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_calls=0,scoring_calls=0,ratings_generated=0)
    C.write(receipt,r);return r


def render(workers=2):
    if workers not in (1,2):raise ValueError('Controller-selected bounded CPU workers')
    done=GALLERY/'complete.json'
    if done.exists():
        r=C.read(done);C.verify(r['files']);C.verify(r['protected']);return r
    from r2.pilots.bench_four_routes_v1.resources import snapshot
    pages=[];protected={str(E.MANIFEST):C.sha(E.MANIFEST)}
    for q in E.requests():
        for stage in E.STAGES:
            cells=[mesh_spec(label,q,stage) for label in E.LABELS]
            pages.append(dict(q=q,stage=stage,cells=cells))
            for c in cells:protected[c['path']]=c['sha256'];protected[c['producer']]=c['producer_sha256']
    if len(pages)!=3*len(E.requests()):raise ValueError('Fixed gallery cardinality')
    C.write(GALLERY/'definition.json',dict(version=VERSION,pages=pages,protected=protected,
        camera='Original saved frame; azimuth -60/120 elevation23',section='Original front/side position index1',
        layout='Four modes x two oblique views and two original finite sections; LAYOUT1000 source regions are privileged',ratings_generated=0))
    dispatch=GALLERY/'dispatch'/str(time.time_ns());tick=time.perf_counter()
    C.write(dispatch/'before.json',dict(workers=workers,threads_per_worker=1,resources=snapshot(),pages=len(pages)))
    if workers==1:results=[render_page(page) for page in pages]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            results=list(pool.map(render_page,pages))
    rows=sorted(results,key=lambda r:r['page_id'])
    chunks=['<!doctype html><meta charset="utf-8"><title>Bench coarse-layout privileged-information comparison</title>',
        '<h1>EXPOSURE / PLAIN1000 / LAYOUT1000 (SOURCE ORACLE) / LAYOUT_NULL</h1><p>Fixed registered pages. RAW, BASE_POST and CLEAN_FINAL_POST. Original noise, scale and cross sections. LAYOUT1000 has privileged source coarse regions; it is not a C-only deployment result. AI_PROXY review is separate.</p>']
    for r in rows:
        rel=r['page_id']+'/comparison.png'
        chunks.append('<h2>'+html.escape(r['page_id'])+'</h2><a href="'+html.escape(rel,quote=True)+'"><img style="max-width:100%" src="'+html.escape(rel,quote=True)+'"></a>')
    text='\n'.join(chunks);index=GALLERY/'index.html'
    if index.exists():
        if index.read_text('utf-8')!=text:raise ValueError('Gallery index changed')
    else:
        with index.open('x',encoding='utf-8') as f:f.write(text)
    C.verify(protected)
    C.write(dispatch/'complete.json',dict(status='COMPLETE',seconds=time.perf_counter()-tick,resources=snapshot()))
    paths=[GALLERY/'definition.json',index]+[C.resolve(r['image_path']) for r in rows]+[GALLERY/r['page_id']/'comparison.json' for r in rows]
    r=dict(status='COMPLETE',pages=len(pages),review_status='PENDING_ACTUAL_AI_PROXY_REVIEW',ratings_generated=0,
        model_calls=0,scoring_calls=0,workers=workers,seconds=time.perf_counter()-tick,protected=protected,
        files={str(p):C.sha(p) for p in paths})
    C.write(done,r);return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,choices=(1,2),default=2)
    a=p.parse_args();r=render(a.workers)
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','protected')},ensure_ascii=False))


if __name__=='__main__':main()
