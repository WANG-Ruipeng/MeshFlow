"""CPU-only 44-request old/new comparison using inherited cameras/sections.

prepare writes a manifest of exact planned paths; run requires all producer
receipts and meshes. No model, geometry transformation, scoring or GT display.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse, concurrent.futures, html, io, json, multiprocessing, resource, time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'r2/src'))
import numpy as np
from r2.pilots.seam_abc_v1 import eval_common as C
from r2.pilots.regular_patch_control_v1 import depth_gallery as D
from r2.tools.cref_layer_audit_v1 import render as H
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'

OUT=ROOT/'experiments/r2_chair_multipart_seam_v1'
KIND='S2000__MULTIPART_S1500_same_pipeline_roles_v2'
VERSION='chair_multipart_fixed_original_views_role_colors_v2'
BLUE=np.array([22,134,192],float)
GRAY=np.array([185,191,197],float)
ORANGE=np.array([235,142,37],float)
GREEN=np.array([65,164,108],float)

def old_definitions():
    return dict(SEAT_FULL=ROOT/'experiments/r2_hinge_opposite_v1/gallery/definition.json',
                BACK_FULL=ROOT/'experiments/r2_chair_back_transfer_v1/gallery/S2000_BACK_FULL/definition.json')

def paths(q):
    rid=q['request_id'];part=q['part_type'];seat=part=='SEAT_FULL'
    old=(ROOT/'experiments/r2_seam_quality_wave1_wave2_v1/evaluation/PLAIN_S2000' if seat
         else ROOT/'experiments/r2_chair_back_transfer_v1/evaluation/S2000')/rid
    new=OUT/'evaluation/MULTIPART_S1500'/rid
    oldfinal=ROOT/'experiments/r2_unified_rescue_post_v1/cells'/('CHAIR_SEAT_S2000' if seat else 'CHAIR_BACK_S2000')/rid
    newfinal=OUT/'unified/cells'/part/rid
    cells=[]
    for label,base,final in (('S2000',old,oldfinal),('MULTIPART_S1500',new,newfinal)):
        for stage,mesh,receipt in (
            ('RAW',base/'raw/raw.npz',base/'raw/receipt.json'),
            ('BASE_POST',base/'post/post.npz',base/'post/receipt.json'),
            ('UNIFIED_FINAL',final/'FINAL/post.npz',final/'complete.json')):
            cells.append(dict(model=label,stage=stage,label=label+' '+stage,path=str(mesh),receipt=str(receipt),request_id=rid))
    return cells

def prepare():
    dest=OUT/'gallery_manifests'/(KIND+'.json')
    if dest.exists():
        plan=C.read(dest)
        for p,h in plan['protected'].items():
            if C.sha(p)!=h:raise ValueError('Frozen gallery input changed: '+p)
        return plan
    evaluation=OUT/'eval_manifest.json';m=C.read(evaluation);qq=m['requests'];assert len(qq)==44
    source=old_definitions();definitions={part:C.read(p) for part,p in source.items()}
    pages=[];bounds={};depth={}
    for q in sorted(qq,key=lambda q:(q['part_type'],q['split'],q['request_id'])):
        part=q['part_type'];cid=q['condition_id'];historic=definitions[part]
        bounds[cid]=historic['bounds'][cid];depth[cid]=historic['depth_bounds'][cid]
        pages.append(dict(**{k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','K','N','split','part_type')},
            page_id=q['request_id'],cells=paths(q),inference_npz=q['inference_npz'],
            kind='Chair '+part+' | '+q['split'],stage='RAW / BASE / UNIFIED FINAL',
            adaptation_note='BACK FAMILIAR is seen-condition adaptation after new mixed training' if part=='BACK_FULL' and q['split']=='FAMILIAR' else 'Original held-out split retained'))
    protected={str(evaluation):C.sha(evaluation),str(Path(__file__)):C.sha(__file__)}
    protected.update({str(p):C.sha(p) for p in source.values()})
    protected.update({str(Path(mod.__file__)):C.sha(mod.__file__) for mod in (D,H)})
    plan=dict(version=VERSION,pages=pages,bounds=bounds,depth_bounds=depth,protected=protected,
        comparison='Old S2000 and new MULTIPART_S1500; both final columns use identical new unified CPU pipeline',
        layout='Old RAW/BASE/FINAL then new RAW/BASE/FINAL; no optional best rows',
        source_definitions={k:str(v) for k,v in source.items()},
        camera='Exactly inherited azimuth -60 /120 and elevation23, centers and spans. No per-output fit.',
        sections='Exactly inherited front/side central finite planes, bounds and display axes. A cut line endpoint does not establish a physical crack.',
        roles='Blue C; orange retained SEAM; gray BULK; green added BRIDGE. Colors indicate stored roles, not correctness or conflicts.',
        review='PENDING actual nonblind AI_PROXY; images do not certify shared-index topology or intersections.',
        model_calls=0,scoring_calls=0,new_noise=0,geometry_edits=0,GT_display=False)
    C.write(dest,plan);return plan

def authenticate(cell):
    path=C.resolve(cell['path']);receipt=C.resolve(cell['receipt'])
    r=C.read(receipt)
    if r.get('status') not in ('COMPLETE','PASS'):raise ValueError('Incomplete producer '+str(receipt))
    actual=C.sha(path);expected=r.get('raw_sha256' if cell['stage']=='RAW' else 'post_sha256')
    if expected is None:
        values=[v for k,v in r.get('files',{}).items() if C.resolve(k)==path]
        if len(values)!=1:raise ValueError('Producer does not authenticate mesh '+str(path))
        expected=values[0]
    if actual!=expected:raise ValueError('Producer mesh hash mismatch '+str(path))
    rid=r.get('request_id',r.get('identity',{}).get('request_id'))
    if rid is not None and rid!=cell['request_id']:raise ValueError('Request identity differs')
    return {str(path):actual,str(receipt):C.sha(receipt)}

def roles(arr,K):
    tri=np.asarray(arr['output']);n=len(tri)
    if 'face_role' in arr:
        rr=np.asarray(arr['face_role']).astype('<U6')
        if rr.shape!=(n,) or set(rr.tolist())-{'C','SEAM','BULK','BRIDGE'} or not np.all(rr[:K]=='C') or np.any(rr[K:]=='C'):
            raise ValueError('Explicit output face roles inconsistent')
        bridge=np.asarray(arr.get('bridge_mask',rr=='BRIDGE'),bool)
        if not np.array_equal(bridge,rr=='BRIDGE'):raise ValueError('Bridge role/mask mismatch')
        if 'retained_seam_mask' in arr and not np.array_equal(arr['retained_seam_mask'],rr=='SEAM'):
            raise ValueError('SEAM role/mask mismatch')
        return tri,rr
    mask=np.asarray(arr['retained_seam_mask'],bool)
    if mask.shape!=(n,):raise ValueError('Stored SEAM roles do not match output faces')
    bridge=np.asarray(arr.get('bridge_mask',np.zeros(n,bool)),bool)
    if bridge.shape!=(n,) or mask[:K].any() or bridge[:K].any() or (mask&bridge).any():raise ValueError('Stored role identity inconsistent')
    rr=np.full(n,'BULK',dtype='<U6');rr[:K]='C';rr[mask]='SEAM';rr[bridge]='BRIDGE'
    return tri,rr

def role_raster(tri,role,K,basis,center,span):
    # The original z-buffer decides every visible face. Only its final RGB for
    # free roles changes; C/occlusion/depth tie policies are not changed.
    im,owner=D.raster(tri,K,basis,center,span,size=440)
    rgb=np.asarray(im).copy()
    for name,color in (('SEAM',ORANGE),('BRIDGE',GREEN)):
        mask=(owner>=0)&np.isin(owner,np.flatnonzero(role==name))
        if mask.any():
            shade=rgb[mask,0].astype(float)/GRAY[0]
            rgb[mask]=np.clip(shade[:,None]*color,0,255).astype(np.uint8)
    projected=(tri.astype(float)-center)@basis
    edge=span/(2*.86)
    clipped=int((np.abs(projected[:,:,:2])>edge).any(axis=2).sum())
    return rgb,dict(projected_vertices_outside_inherited_frame=clipped,visible_pixels={r:int(np.isin(owner,np.flatnonzero(role==r)).sum()) for r in ('C','SEAM','BULK','BRIDGE')})

def one(job):
    page,out,definition_sha,bounds,depth=job;out=Path(out);png=out/(page['page_id']+'.png');receipt=png.with_suffix('.json')
    if receipt.exists():
        r=C.read(receipt);assert r['definition_sha256']==definition_sha and C.sha(png)==r['image_sha256'];return r
    tick=time.perf_counter();plt=H._mpl();K=int(page['K']);nc=6
    fig=plt.figure(figsize=(5.3*nc,13.5),dpi=110,facecolor='white')
    grid=fig.add_gridspec(4,nc,left=.025,right=.99,bottom=.055,top=.91,wspace=.15,hspace=.23)
    bundle=C.arrays(page['inference_npz']);sourceC=bundle['C_vertices'][bundle['C_faces_original']]
    display=[]
    for col,cell in enumerate(page['cells']):
        arr=C.arrays(cell['path']);tri,role=roles(arr,K);g=dict(C=tri[:K],F=tri[K:],C_original=sourceC,tri=tri)
        facts=dict(label=cell['label'],role_counts={r:int((role==r).sum()) for r in ('C','SEAM','BULK','BRIDGE')},views=[])
        for row,az in enumerate((-60,120)):
            image,scope=role_raster(tri,role,K,D.basis(az,23),np.array(depth['center']),depth['span'])
            ax=fig.add_subplot(grid[row,col]);ax.imshow(image);ax.set_axis_off();ax.set_title(cell['label']+' | '+str(az),fontsize=10)
            facts['views'].append(dict(azimuth=az,**scope))
            if scope['projected_vertices_outside_inherited_frame']:ax.text(.01,.01,'Clipped by inherited frame: see receipt',transform=ax.transAxes,fontsize=7,color='#b34b27')
        for row,view in ((2,'front'),(3,'side')):
            ax=fig.add_subplot(grid[row,col]);seams=np.flatnonzero(role[K:]=='SEAM')
            H.draw_section(ax,g,seams,bounds,C.L,view,1)
            from matplotlib.collections import LineCollection
            axis,project=(2,(0,1)) if view=='front' else (0,(2,1))
            pos=bounds['front_positions' if view=='front' else 'side_positions'][1]
            ff,ids=H.finite_sections(g['F'],axis,pos,1e-12*C.L);bm=role[K:][ids]=='BRIDGE'
            if bm.any():ax.add_collection(LineCollection(ff[bm][:,:,project],colors='#41a46c',linewidths=1.6,zorder=3))
            ax.set_title(cell['label']+' | '+view+f' plane={pos:.5f}',fontsize=8)
        display.append(facts)
    fig.suptitle(page['request_id']+' | '+page['kind']+f" | K={K}, N={page['N']}, noise={page['noise_index']}\n"
        'Old S2000 (left 3) / new MULTIPART (right 3). Blue C | orange SEAM | gray BULK | green new BRIDGE\n'
        'Original views, scale and cut positions retained; same frozen noise. '+page['adaptation_note'],fontsize=12)
    fig.text(.5,.018,'Actual AI_PROXY review pending. Role color is not a quality label. Finite section lines are not proof of cracks; shared edges/intersections need index/geometry evidence. No GT style grading.',ha='center',fontsize=9)
    try:
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=110)
        with png.open('xb') as f:f.write(buffer.getvalue())
    finally:plt.close(fig)
    r=dict(status='PENDING_ACTUAL_VIEW',page_id=page['page_id'],request_id=page['request_id'],condition_id=page['condition_id'],part_type=page['part_type'],split=page['split'],
        path=str(png),image_sha256=C.sha(png),definition_sha256=definition_sha,cells=page['cells'],display=display,
        seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_calls=0,scoring_calls=0)
    C.write(receipt,r);return r

def run(max_workers=2):
    plan=prepare();pages=plan['pages'];out=OUT/'gallery'/KIND
    # Do not render an incomplete subset as if it were the complete 44 panel.
    missing=[p for page in pages for c in page['cells'] for p in (c['path'],c['receipt']) if not C.resolve(p).is_file()]
    if missing:raise FileNotFoundError(json.dumps(dict(status='WAITING_FOR_COMPLETED_CACHES',missing_count=len(missing),first_missing=missing[:8])))
    sources={}
    for page in pages:
        for cell in page['cells']:sources.update(authenticate(cell))
    out.mkdir(parents=True,exist_ok=True);manifest=OUT/'gallery_manifests'/(KIND+'.json')
    definition=dict(version=VERSION,manifest_sha256=C.sha(manifest),protected=plan['protected'],mesh_and_receipt_hashes=sources,
        bounds=plan['bounds'],depth_bounds=plan['depth_bounds'],roles=plan['roles'],model_calls=0,scoring_calls=0,geometry_changes=0,GT_display=False)
    dp=out/'definition.json'
    if dp.exists():assert C.read(dp)==definition
    else:C.write(dp,definition)
    dh=C.sha(dp);done=out/'completion.json'
    if done.exists():
        r=C.read(done);assert r['definition_sha256']==dh
        for im in r['images']:assert C.sha(im['path'])==im['image_sha256']
        return r
    jobs=[(p,str(out),dh,plan['bounds'][p['condition_id']],plan['depth_bounds'][p['condition_id']]) for p in pages]
    start=time.perf_counter();before=C.resources();C.write(out/'resources_before.json',before)
    first=one(jobs[0]);res=C.resources();available=int(next(x.split()[1] for x in res['memory'].splitlines() if x.startswith('MemAvailable:')))*1024
    peak=first['peak_RSS_KiB']*1024;cores=max(1,int(res['cpu_logical']-res['loadavg'][0]-2));ram=max(1,int(max(0,available-(4<<30))/(peak*1.2)))
    workers=max(1,min(int(max_workers),2,len(jobs),cores,ram))
    C.write(out/'dispatch.json',dict(workers=workers,first_peak_RSS_bytes=peak,available_RAM_bytes=available,CPU_budget=cores,
        resources=res,reason='At most2 independent one-thread CPU raster workers; measured RSS/available RAM and CPU; no GPU/model'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        images=[first,*list(pool.map(one,jobs[1:]))]
    text='<meta charset="utf-8"><title>Chair multi-part SEAM comparison</title><h1>S2000 / MULTIPART: all 44 fixed requests</h1><p>Old RAW/BASE/unified FINAL | new RAW/BASE/same unified FINAL. Blue C, orange stored seam, gray bulk, green added bridge. Fixed original views and cuts. Open an image for full resolution. AI_PROXY labels remain separate.</p>'
    for im in images:text+='<h2>'+html.escape(im['part_type']+' '+im['split']+' '+im['request_id'])+'</h2><a href="'+Path(im['path']).name+'"><img loading="lazy" style="width:100%" src="'+Path(im['path']).name+'"></a>'
    (out/'index.html').write_text(text,encoding='utf-8')
    result=dict(status='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW',images=images,definition_sha256=dh,requests=len(images),workers=workers,wall_seconds=time.perf_counter()-start,model_calls=0,GPU_calls=0,new_scores=0)
    C.write(done,result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    result=prepare() if args.action=='prepare' else run(args.workers)
    print(json.dumps(dict(status=result.get('status','PLANNED'),pages=len(result.get('pages',result.get('images',[])))),ensure_ascii=False))
