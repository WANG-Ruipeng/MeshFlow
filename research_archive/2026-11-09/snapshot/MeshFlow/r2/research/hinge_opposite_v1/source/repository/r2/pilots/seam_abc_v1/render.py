"""All sixteen requests, four frozen models, RAW/POST separated; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import html
import io
import json
import concurrent.futures
import multiprocessing
import resource
import time
from pathlib import Path
import numpy as np
from . import eval_common as C
from . import evaluate as E
from r2.pilots.regular_patch_control_v1 import depth_gallery as D
from r2.tools.cref_layer_audit_v1 import render as H

LABELS=C.ARMS
BASE=C.ROOT/'experiments/r2_seam_slot_c200_v1'


def render_one(job):
    """One registered page, one owned PNG/receipt; no shared output writer."""
    q,stage,out_text,definition_sha,cell,bounds,db=job
    out=Path(out_text);p=out/(q['request_id']+'_'+stage+'.png');receipt=p.with_suffix('.json')
    if receipt.exists():
        rec=C.read(receipt)
        if C.sha(p)!=rec['image_sha256'] or rec['definition_sha256']!=definition_sha:
            raise ValueError('Rendered page identity changed')
        return dict(path=str(p),reused=True,**rec)
    if p.exists() or p.with_suffix('.started.json').exists():
        raise ValueError('Incomplete rendered page; explicit recovery required')
    C.write(p.with_suffix('.started.json'),dict(request_id=q['request_id'],stage=stage,
        definition_sha256=definition_sha,pid=os.getpid(),registered_page=True))
    tick=time.perf_counter();plt=H._mpl();views=[D.basis(-60,23),D.basis(120,23)]
    fig=plt.figure(figsize=(16.8,15),dpi=125,facecolor='white')
    try:
        grid=fig.add_gridspec(4,4,left=.025,right=.99,bottom=.055,top=.94,wspace=.18,hspace=.22)
        for col,label in enumerate(LABELS):
            g=cell[label]
            for row,b in enumerate(views):
                im,_=D.raster(g['tri'],q['K'],b,np.array(db['center']),db['span'],size=460)
                ax=fig.add_subplot(grid[row,col]);ax.imshow(im);ax.set_axis_off();ax.set_title(label+' | '+stage+(' | -60' if row==0 else ' | +120'),fontsize=10)
            for row,view in ((2,'front'),(3,'side')):
                ax=fig.add_subplot(grid[row,col]);H.draw_section(ax,g,[],bounds,C.L,view,1);ax.set_title(label+' | '+view,fontsize=9)
        fig.suptitle(q['request_id']+' | '+stage+' | K='+str(q['K'])+' N='+str(q['N'])+'\nBlue preserved C / gray free; shared fixed Gaussian; four TRAIN / three DEV conditions',fontsize=13)
        fig.text(.5,.018,'Common scale across RAW/POST and all models/noises of this C. Finite central cuts; no source completion shown. Nonblind AI_PROXY review.',ha='center',fontsize=8)
        data=io.BytesIO();fig.savefig(data,format='png',dpi=125)
        with p.open('xb') as f:f.write(data.getvalue())
    finally:
        plt.close(fig)
    rec=dict(request_id=q['request_id'],stage=stage,image_sha256=C.sha(p),definition_sha256=definition_sha,
        status='RENDERED_PENDING_ACTUAL_VIEW',seconds=time.perf_counter()-tick,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        RSS_scope='Peak worker process RSS through one actual registered page; includes renderer imports',
        worker_pid=os.getpid(),threads=1)
    C.write(receipt,rec)
    return dict(path=str(p),reused=False,**rec)


def memory_available(snapshot):
    values={line.split(':')[0]:int(line.split()[1])*1024
        for line in (snapshot.get('memory') or '').splitlines()
        if len(line.split())>1 and line.split()[1].isdigit()}
    return values.get('MemAvailable',0)


def measured_workers(snapshot,first):
    available=memory_available(snapshot);cores=int(snapshot.get('cpu_logical') or 1)
    load=(snapshot.get('loadavg') or [None])[0]
    peak=int(first['peak_RSS_KiB'])*1024
    # Reserve two GiB for unrelated work and allow 20% over the observed process peak.
    slots=max(0,int((available-(2<<30))//max(1,1.2*peak)))
    if slots<1:raise RuntimeError('RESOURCE_WAIT: no CPU renderer slot with measured RSS headroom')
    workers=2 if slots>=2 and cores>=4 and load is not None and load<cores-2 else 1
    return workers,dict(available_RAM_bytes=available,CPU_cores=cores,load_average_1m=load,
        observed_worker_peak_RSS_bytes=peak,RAM_headroom_bytes=2<<30,peak_margin=1.2,
        memory_slots=slots,workers=workers,threads_per_worker=1,
        reason='At most two workers after first actual registered page; RAM and CPU-load gated')


def main():
    root=C.OUT;m=C.read(root/'eval_manifest.json');out=root/'evaluation/gallery';out.mkdir(parents=True,exist_ok=True)
    cells={};hashes={};views=[D.basis(-60,23),D.basis(120,23)]
    for q in m['requests']:
        for label in LABELS:
            for stage in ('RAW','POST'):
                srcroot=root;arm=label
                p,receipt=E.artifact(srcroot,arm,q['request_id'],stage);mesh=C.arrays(p);tri=mesh['output']
                cells[q['request_id'],label,stage]=dict(C=tri[:q['K']],F=tri[q['K']:],C_original=tri[:q['K']],tri=tri)
                hashes[str(p)]=C.sha(p)
    bounds={};depth={}
    for cid in sorted({q['condition_id'] for q in m['requests']}):
        gg=[cells[q['request_id'],label,stage] for q in m['requests'] if q['condition_id']==cid for label in LABELS for stage in ('RAW','POST')]
        bounds[cid]=H.bounds_for_condition(gg,None,C.L)
        points=np.concatenate([g['tri'].reshape(-1,3) for g in gg]);center=(points.min(0)+points.max(0))/2
        span=max(np.ptp((points-center)@b,axis=0)[:2].max() for b in views)
        depth[cid]=dict(center=center.tolist(),span=float(span))
    C.write(out/'definition.json',dict(version='seam_abc_paired_depth_sections_v1',labels=LABELS,
        source_hashes={str(Path(x.__file__)):C.sha(x.__file__) for x in (D,H)},renderer_sha256=C.sha(__file__),geometry_hashes=hashes,
        physical_scale='same per-C world bounds over all four models, four noises, RAW and POST',
        GT_display=False,depth_views=[[-60,23],[120,23]],central_finite_sections=['front','side'],
        caveats='Depth ties may favor blue C. Central sections can miss off-plane geometry. Images do not certify connectivity or outward orientation.',
        review='AI_PROXY nonblind actual view required; not human ratings; no automatic semantic labels',bounds=bounds,depth_bounds=depth))
    definition_sha=C.sha(out/'definition.json')
    done=out/'render_completion.json'
    if done.exists():
        previous=C.read(done)
        if previous['definition_sha256']!=definition_sha:raise ValueError('Completed render definition changed')
        E.verify(previous['artifacts'])
        print(json.dumps(dict(status='COMPLETE_VERIFIED',new_renders=0,new_model_calls=0)));return
    jobs=[]
    for q in m['requests']:
        for stage in ('RAW','POST'):
            jobs.append((q,stage,str(out),definition_sha,
                {label:cells[q['request_id'],label,stage] for label in LABELS},
                bounds[q['condition_id']],depth[q['condition_id']]))
    if len(jobs)!=56 or len({(x[0]['request_id'],x[1]) for x in jobs})!=56:
        raise ValueError('Exactly 56 unique registered pages required')
    # All 224 source cells were loaded/checked before dispatch. No partial-arm rendering.
    attempt='render_dispatch_'+str(time.time_ns());started=time.perf_counter()
    before=C.resources();C.write(out/(attempt+'_before.json'),before)
    if memory_available(before)<(3<<30):
        raise RuntimeError('RESOURCE_WAIT: less than 3 GiB available before first registered page')
    context=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=1,mp_context=context) as pool:
        first=pool.submit(render_one,jobs[0]).result()
    after=C.resources();workers,policy=measured_workers(after,first)
    C.write(out/(attempt+'.json'),dict(first_actual_registered_page=first,resources_after_first=after,
        policy=policy,pilot_is_not_extra=True,registered_total=56,remaining=55))
    result={(first['request_id'],first['stage']):first}
    print(json.dumps(dict(rendered_or_reused=1,registered=56,workers_next=workers)),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
        pending=iter(jobs[1:]);active={}
        for _ in range(workers):
            job=next(pending,None)
            if job is not None:active[pool.submit(render_one,job)]=job
        while active:
            completed,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            # Resolve every completed future before dispatching any replacement.
            # A failure therefore stops new pages; existing workers finish their owned page.
            returned=[future.result() for future in completed]
            for future,rec in zip(completed,returned):
                del active[future];result[rec['request_id'],rec['stage']]=rec
            print(json.dumps(dict(rendered_or_reused=len(result),registered=56)),flush=True)
            for _ in completed:
                job=next(pending,None)
                if job is not None:active[pool.submit(render_one,job)]=job
    images=[];template=[]
    # Aggregate in registered order, never completion order. Global files are parent-only.
    for q,stage,*_ in jobs:
        rec=result[q['request_id'],stage];images.append({k:v for k,v in rec.items() if k!='reused'})
        for label in LABELS:template.append(dict(request_id=q['request_id'],stage=stage,label=label,reviewer='',review_status='PENDING',whole_body_complete='',C_adoption='',fragments_or_folded_layers='',visible_join='',notes='',image_path=rec['path'],image_sha256=rec['image_sha256']))
    C.csvwrite(out/'review_template_NOT_REVIEWED.csv',template)
    page='<meta charset="utf-8"><h1>SEAM A/B/C: full fixed panel</h1><p>Four TRAIN C and three historical DEV C. All requests retained. RAW/POST separated.</p>'
    page+=''.join('<h2>'+html.escape(x['request_id']+' '+x['stage'])+'</h2><a href="'+Path(x['path']).name+'"><img style="width:100%" loading="lazy" src="'+Path(x['path']).name+'"></a>' for x in images)
    (out/'index.html').write_text(page,encoding='utf-8')
    C.write(out/'manifest.json',dict(status='RENDERED_PENDING_ACTUAL_REVIEW',images=images,new_model_calls=0,geometry_edits=0))
    C.write(done,dict(status='COMPLETE_PENDING_ACTUAL_REVIEW',definition_sha256=definition_sha,
        pages=56,new_renders=sum(not x['reused'] for x in result.values()),workers_after_first=workers,
        threads_per_worker=1,first_actual_registered_page=first,dispatch_record=str(out/(attempt+'.json')),
        wall_seconds=time.perf_counter()-started,new_model_calls=0,geometry_edits=0,
        timing_interpretation='Measured concurrent render wall time; not an isolated per-page benchmark',
        artifacts={str(p):C.sha(p) for p in out.iterdir() if p.is_file() and p!=done}))

if __name__=='__main__':main()
