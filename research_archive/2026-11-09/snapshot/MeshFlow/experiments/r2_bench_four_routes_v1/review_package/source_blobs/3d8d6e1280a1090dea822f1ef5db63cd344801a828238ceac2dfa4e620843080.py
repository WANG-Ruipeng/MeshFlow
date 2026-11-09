"""Original frozen cameras/sections; FINAL main and BASE supplemental pages."""
from . import common as C
from .run import LABEL
import concurrent.futures,html,io,multiprocessing,resource,time
import numpy as np

def prepare():
    reg=C.read(C.OUT/'registration.json');pages=[]
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    for task in reg['tasks']:
        q=task['request'];rid=q['request_id'];p=task['paths']
        for stage in ('BASE_POST','CLEAN_FINAL_POST'):
            new=C.OUT/'outputs'/rid/'post.npz' if stage=='BASE_POST' else C.OUT/'cpu'/LABEL/rid/'CLEAN_FINAL_POST/post.npz'
            old=p['original_base' if stage=='BASE_POST' else 'original_final']
            cells=[dict(model='RAW',stage='RAW',path=p['raw']),dict(model='M_D0',stage=stage,path=old),dict(model=LABEL,stage=stage,path=str(new))]
            pages.append(dict(page_id=stage+'__'+rid,sheet=stage,request=q,cells=cells))
    declaration=dict(status='FROZEN',pages=pages,primary='CLEAN_FINAL_POST (46)',supplement='BASE_POST (46)',
        cameras='Original q.gallery_depth_bounds, azimuth -60/120, elevation23',sections='Original q.gallery_bounds front/side_positions[1]',
        actual_review_required=True,ratings_generated=0,GT_display=False,
        sources={str(p):C.sha(p) for p in (C.CODE/'render.py',G.__file__,G.D.__file__,G.H.__file__)},
        prior_v2_definition_sha256=C.sha(C.OLD/'visual_review/v2_definition.json'))
    C.write(C.OUT/'gallery/manifest.json',declaration);return declaration

def one(job):
    page,dh=job;q=page['request'];path=C.OUT/'gallery'/(page['page_id']+'.png');done=path.with_suffix('.json')
    if done.exists():
        r=C.read(done)
        if r['image_sha256']!=C.sha(path) or r['definition_sha256']!=dh:raise ValueError('Gallery changed')
        return r
    if path.exists():raise RuntimeError('Unsealed image')
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    D,H=G.D,G.H;plt=H._mpl();tick=time.perf_counter();fig=plt.figure(figsize=(15.9,13.5),dpi=110,facecolor='white')
    grid=fig.add_gridspec(4,3,left=.025,right=.99,bottom=.045,top=.92,wspace=.15,hspace=.23)
    inp=C.arrays(q['inference_npz']);original=inp['C_vertices'][inp['C_faces_original']];K=q['K'];bounds=q['gallery_bounds'];depth=q['gallery_depth_bounds'];sources={};display=[]
    try:
        for col,cell in enumerate(page['cells']):
            tri,role=G.roles(C.arrays(cell['path']),K);sources[cell['path']]=C.sha(cell['path']);g=dict(C=tri[:K],F=tri[K:],C_original=original,tri=tri)
            facts=dict(model=cell['model'],stage=cell['stage'],role_counts={r:int((role==r).sum()) for r in ('C','SEAM','BULK','BRIDGE')},views=[])
            for row,az in enumerate((-60,120)):
                raster,scope=G.role_raster(tri,role,K,D.basis(az,23),np.asarray(depth['center']),depth['span'])
                ax=fig.add_subplot(grid[row,col]);ax.imshow(raster);ax.set_axis_off();ax.set_title(cell['model']+' '+cell['stage']+' | '+str(az),fontsize=9)
                if scope['projected_vertices_outside_inherited_frame']:ax.text(.01,.01,'Inherited-frame clipping',transform=ax.transAxes,fontsize=7,color='#b34b27')
                facts['views'].append(dict(azimuth=az,**scope))
            for row,view in ((2,'front'),(3,'side')):
                ax=fig.add_subplot(grid[row,col]);H.draw_section(ax,g,np.flatnonzero(role[K:]=='SEAM'),bounds,C.L,view,1)
                from matplotlib.collections import LineCollection
                axis,project=(2,(0,1)) if view=='front' else (0,(2,1));pos=bounds['front_positions' if view=='front' else 'side_positions'][1]
                ff,ids=H.finite_sections(g['F'],axis,pos,1e-12*C.L);bm=role[K:][ids]=='BRIDGE'
                if bm.any():ax.add_collection(LineCollection(ff[bm][:,:,project],colors='#41a46c',linewidths=1.6,zorder=3))
                ax.set_title(cell['model']+' | '+view+f'={pos:.5f}',fontsize=8)
            display.append(facts)
        fig.suptitle(q['request_id']+' | '+q['report_group']+f' | N={q["N"]} K={K}\nSame RAW; official D0 versus fixed one-ring RAW/D0 selector; blue C / gray free / green bridge',fontsize=11)
        fig.text(.5,.012,'V2 AI_PROXY actual review separate. Support PRESENT requires major support surfaces. Fixed sections alone do not certify cracks.',ha='center',fontsize=8)
        b=io.BytesIO();fig.savefig(b,format='png',dpi=110)
        with path.open('xb') as f:f.write(b.getvalue())
    finally:plt.close(fig)
    result=dict(status='PENDING_ACTUAL_VIEW',page_id=page['page_id'],request_id=q['request_id'],parent_id=q['parent_id'],report_group=q['report_group'],
        path=str(path),image_sha256=C.sha(path),definition_sha256=dh,cells=page['cells'],display=display,source_meshes=sources,
        seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_calls=0,scoring_calls=0)
    C.write(done,result);return result

def finish(plan,images,workers,seconds,streaming=False):
    content='<meta charset="utf-8"><h1>HYBRID RAW/D0 selector</h1><p>46 primary FINAL pages; BASE supplemental. Same noise/cameras/sections. Actual review separate.</p>'
    for image in images:
        if image['page_id'].startswith('CLEAN_FINAL_POST'):
            primary=C.resolve(image['path']).name;base='BASE_POST'+primary[len('CLEAN_FINAL_POST'):]
            content+='<h2>'+html.escape(image['request_id'])+'</h2><a href="'+base+'">BASE supplemental</a><br><a href="'+primary+'"><img loading="lazy" style="width:100%" src="'+primary+'"></a>'
    index=C.OUT/'gallery/index.html'
    with index.open('x',encoding='utf-8') as f:f.write(content)
    C.verify(plan['sources']);result=dict(status='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW',primary_pages=46,supplemental_pages=46,total_cells=276,
        seconds=seconds,workers=workers,streaming_alongside_scoring=streaming,model_calls=0,scoring_calls=0,ratings_generated=0,
        files={str(C.resolve(r['path'])):r['image_sha256'] for r in images})
    for r in images:result['files'][str(C.resolve(r['path']).with_suffix('.json'))]=C.sha(C.resolve(r['path']).with_suffix('.json'))
    result['files'][str(index)]=C.sha(index);C.write(C.OUT/'gallery/completion.json',result);return result

def run():
    plan=prepare();C.verify(C.read(C.OUT/'completions/geometry_scores.json')['files']);dh=C.sha(C.OUT/'gallery/manifest.json');done=C.OUT/'gallery/completion.json'
    if done.exists():r=C.read(done);C.verify(r['files']);return r
    before=C.resources();tick=time.perf_counter();jobs=[(p,dh) for p in plan['pages']];first=one(jobs[0])
    from r2.pilots.chair_back_transfer_v1.evaluate import worker_count
    workers=worker_count(C.resources(),first,2)
    C.write(C.OUT/'gallery/dispatch.json',dict(workers=workers,threads_per_worker=1,resources_before=before,resources_after_first=C.resources()))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:images=[first,*pool.map(one,jobs[1:])]
    return finish(plan,images,workers,time.perf_counter()-tick)

if __name__=='__main__':print(run())
