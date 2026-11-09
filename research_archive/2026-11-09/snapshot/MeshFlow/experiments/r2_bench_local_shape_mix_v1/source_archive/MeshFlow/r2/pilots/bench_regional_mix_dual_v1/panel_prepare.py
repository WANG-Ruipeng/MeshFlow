"""Six pre-output development requests and assisted RAW support regions.

CPU only: no model, target Y, score computation or backend execution.
All original queries, noises, camera/section settings and source arrays stay read-only.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,io,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_regional_mix_dual_v1/panel'
OLD=ROOT/'experiments/r2_bench_four_routes_v1/hybrid'
PARENTS=['a8e987a6a6486b962d6c8e51365a5a87','27eb79ddf26fd7ac7898580cb9d16864','459f90aa2162b1f1d46c340938e2ff1c']
L=9.208138264587333
VIEWED={
 ('a8',0):('2026-10-28 03:15:46 UTC','panel_existing_a8_n0','RAW has broad thin legs; official D0 chain leaves near-line remnants. Central sections miss legs but show loss of seat-strip thickness.'),
 ('a8',1):('2026-10-28 03:15:52 UTC','panel_existing_a8_n1','Both views show RAW leg surfaces versus thin D0 remnants; seat-strip cross sections also shrink. No exact weld certification.'),
 ('27',0):('2026-10-28 03:15:56 UTC','panel_existing_27_n0','RAW lower end supports and arm frames are visible; D0 removes major leg surfaces. Back and curved C survive more clearly; upper frame is outside the selected lower zones.'),
 ('27',1):('2026-10-28 03:16:01 UTC','panel_existing_27_n1','RAW lower legs remain visible in opposite views; D0 leaves thin end remnants. The central side cut does not pass through the legs.'),
 ('45',0):('2026-10-28 03:16:07 UTC','panel_existing_459_n0','RAW has loop-like end supports and a layered/folded underside. D0 removes underside clutter but also major support/back surfaces; cleaning value and structural cost coexist.'),
 ('45',1):('2026-10-28 03:16:22 UTC','panel_existing_459_n1','RAW has thin legs but a gray slatted seat above blue C. D0 thins the gray shell and legs; this local support test is not expected to repair C role or wrong height.'),
}

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(x,ensure_ascii=False,indent=2)+'\n'
    if p.exists():
        if p.read_text(encoding='utf-8')!=text:raise RuntimeError('Refuse overwrite: '+str(p))
    else:
        with p.open('x',encoding='utf-8') as f:f.write(text)
def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def resources():
    import psutil
    r=dict(utc=now(),cpu_cores=os.cpu_count(),cpu_percent=psutil.cpu_percent(.1),RAM_available=psutil.virtual_memory().available,
           loadavg=list(os.getloadavg()),threads=1,workers=1,model_calls=0,score_calls=0)
    for key,args in [('gpu',['--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),
                     ('gpu_processes',['--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])]:
        p=subprocess.run(['nvidia-smi',*args],capture_output=True,text=True);r[key]=p.stdout.strip();r[key+'_returncode']=p.returncode
    return r

def select_regions(raw,k):
    """Fixed by C bounds, same rule for both noises; complete triangles only."""
    c=raw[:k].astype(np.float64);v=raw.astype(np.float64)
    low,high=c.min((0,1)),c.max((0,1));width=float(high[0]-low[0]);cx=float((high[0]+low[0])/2)
    assert width>0
    ycut=float(low[1]-.05*width);xleft=float(cx-.30*width);xright=float(cx+.30*width)
    centers=v.mean(1);low_enough=v[:,:,1].min(1)<ycut;free=np.arange(len(v))>=k
    masks=[free&low_enough&(centers[:,0]<=xleft),free&low_enough&(centers[:,0]>=xright)]
    assert not (masks[0]&masks[1]).any()
    defs=dict(version='assisted_C_bbox_outer20_lower_support_v1',C_bbox_min=low.tolist(),C_bbox_max=high.tolist(),
              x_center=cx,C_width=width,x_left_max=xleft,x_right_min=xright,triangle_min_y_less_than=ycut,
              exact_rule='free AND min(corner_y)<C_min_y-0.05*C_width AND centroid_x<=cx-0.30*C_width (P1) / >=cx+0.30*C_width (P2)',
              scope='Whole RAW triangles; no clipping, recursion, source-Y lookup or result-based edits. Both noises share C-defined thresholds.',
              boundary_limit='A selected triangle may cross a spatial threshold; switching its whole coordinates can break alignment to neighboring D0 triangles. No automatic coherence guarantee.')
    return masks,defs

def render(entry):
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    D,H=G.D,G.H;plt=H._mpl()
    from matplotlib.collections import LineCollection
    q=entry['request'];rid=q['request_id'];k=q['K'];raw=np.load(entry['paths']['raw'])['raw'];masks,defs=select_regions(raw,k)
    path=OUT/'selection_images'/(rid+'.png');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError('Refuse rerender '+str(path))
    fig=plt.figure(figsize=(17,11),dpi=120);gs=fig.add_gridspec(3,3,width_ratios=[1,1,.8],height_ratios=[1.15,1,1],left=.04,right=.98,bottom=.05,top=.91,hspace=.28,wspace=.24)
    colors=[np.array([235,142,37]),np.array([157,83,190])];hexes=['#eb8e25','#9d53be'];depth=q['gallery_depth_bounds'];bounds=q['gallery_bounds']
    for col,az in enumerate((-60,120)):
        B=D.basis(az,23);im,owner=D.raster(raw,k,B,np.array(depth['center']),depth['span'],size=700);rgb=np.asarray(im).copy()
        ax=fig.add_subplot(gs[0,col])
        for mask,color in zip(masks,colors):
            pixels=np.isin(owner,np.flatnonzero(mask));shade=rgb[pixels,0].astype(float)/185
            rgb[pixels]=np.clip(shade[:,None]*color,0,255).astype(np.uint8)
        ax.imshow(rgb);ax.axis('off');ax.set_title('RAW region membership | azimuth '+str(az))
    # Original finite sections stay unchanged. Missing colored segments simply mean this cut misses the legs.
    geom=dict(C=raw[:k],F=raw[k:],C_original=raw[:k],tri=raw)
    for col,view in enumerate(('front','side')):
        ax=fig.add_subplot(gs[1,col]);H.draw_section(ax,geom,np.array([],dtype=int),bounds,L,view,1)
        axis,project=(2,(0,1)) if view=='front' else (0,(2,1));pos=bounds[view+'_positions'][1]
        seg,ids=H.finite_sections(raw[k:],axis,pos,1e-12*L)
        for mask,color in zip(masks,hexes):
            keep=mask[k:][ids]
            if keep.any():ax.add_collection(LineCollection(seg[keep][:,:,project],colors=color,linewidths=2,zorder=5))
        ax.set_title('Inherited '+view+' section = '+format(pos,'.5f'),fontsize=9)
    # Extra all-face front projection is solely an ID-selection diagram, not a replacement evaluation camera.
    for col,(mask,color) in enumerate(zip(masks,hexes)):
        ax=fig.add_subplot(gs[2,col]);ids=np.flatnonzero(mask)
        for fi in range(k):ax.plot(*np.vstack([raw[fi],raw[fi,0]])[:,[0,1]].T,color='#1686c0',lw=.45,alpha=.5)
        for i,fi in enumerate(ids):
            tri=raw[fi];ax.plot(*np.vstack([tri,tri[0]])[:,[0,1]].T,color=color,lw=.75);center=tri.mean(0)
            ax.annotate(str(fi),(center[0],center[1]),xytext=(7+(i%2)*10,(i%5-2)*9),textcoords='offset points',fontsize=6,color=color,arrowprops=dict(arrowstyle='-',lw=.35,color=color))
        if len(ids):
            v=raw[ids].reshape(-1,3);lo=v.min(0);hi=v.max(0);ax.set_xlim(lo[0]-.22,hi[0]+.30);ax.set_ylim(lo[1]-.16,hi[1]+.16)
        ax.set_xlabel('X (model)');ax.set_ylabel('Y (model)');ax.grid(alpha=.12);ax.set_title(f'P{col+1}: selected RAW slot IDs (support zoom)',fontsize=9)
    textax=fig.add_subplot(gs[:,2]);textax.axis('off');text=[]
    for region in entry['regions']:
        ids=region['source_face_ids'];text.append(region['region_id']+' | '+str(len(ids))+' RAW slots\n'+', '.join(map(str,ids)))
    import textwrap
    body='\n\n'.join(textwrap.fill(t,44,replace_whitespace=False) for t in text)
    body+='\n\nC: blue; P1: orange; P2: purple.\nOther free: gray.\n\nThe bottom drawings are ID-only\nzoom supplements. Top cameras and\ncentral section positions are inherited.\n\nSelection uses RAW + C only.\nNo Y, candidate, or scoring consulted.\nNo connectivity correctness claimed.\n\nWhole-face switching can introduce\na mixed-coordinate boundary gap.'
    textax.text(0,1,body,va='top',fontsize=10)
    fig.suptitle(rid+f'\nPre-output assisted regions | N={len(raw)} K={k} | lower left/right support zones',fontsize=12)
    b=io.BytesIO();fig.savefig(b,format='png');plt.close(fig)
    with path.open('xb') as f:f.write(b.getvalue())
    return dict(path=str(path),sha256=sha(path),status='PENDING_ACTUAL_VIEW',inherited_depth=depth,inherited_sections=bounds,
                supplemental_view='Two region-only front projections with RAW slot labels; not used as alternative evaluation cameras')

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);write(OUT/'resources_before.json',resources())
    regpath=OLD/'registration.json';reg=read(regpath);entries=[];protected={str(regpath):sha(regpath)};face_rows=[]
    for parent in PARENTS:
        tasks=sorted((t for t in reg['tasks'] if t['request']['parent_id']==parent),key=lambda t:t['request']['noise_index'])
        assert len(tasks)==2
        prior_definition=None
        for task in tasks:
            q=task['request'];rid=task['request_id'];k=q['K'];rawfile=task['paths']['raw'];z=np.load(rawfile);raw=z['raw'];assert raw.shape==(q['N'],3,3)
            assert np.isfinite(raw).all();assert np.array_equal(z['raw_slot_ids'],np.arange(len(raw)))
            masks,definition=select_regions(raw,k)
            if prior_definition is None:prior_definition=definition
            else:assert prior_definition==definition
            regions=[]
            for idx,mask in enumerate(masks):
                ids=np.flatnonzero(mask).tolist();assert ids and min(ids)>=k
                regions.append(dict(region_id='P'+str(idx+1),source_face_ids=ids,source_id_domain='original_generated_RAW_slot',
                                    definition={**definition,'side':'left' if idx==0 else 'right'},semantic_intent='lower end support; auxiliary nonblind AI_PROXY selection',
                                    target_validity='UNKNOWN: membership is not a clean-GT or physical-support certification'))
            stamp,event,note=VIEWED[(parent[:2],q['noise_index'])]
            image=OLD/'gallery'/('CLEAN_FINAL_POST__'+rid+'.png')
            evidence=dict(path=str(image),sha256=sha(image),actually_viewed=True,utc=stamp,view_event_id=event,
                          reviewer='backend_data',review_type='NONBLIND_AI_PROXY',view_tool='functions.exec/tools.view_image',notes=note,
                          display='Previously completed RAW/M_D0/HYBRID sheet; no new regional candidate exists at selection time')
            relevant={name:p for name,p in task['paths'].items() if name in ('raw','pre','original_base','original_final','input','metadata','query')}
            hashes={p:sha(p) for p in relevant.values()};protected.update(hashes);protected[str(image)]=sha(image)
            selection=OLD/'outputs'/rid/'selection.npz';protected[str(selection)]=sha(selection)
            entry=dict(request_id=rid,request=q,paths=task['paths'],input_hashes=hashes,regions=regions,evidence=[evidence],
                       previous_hybrid_selection_path=str(selection),previous_hybrid_selection_sha256=sha(selection),
                       raw_canonical_coordinate_sha256=array_hash(raw),raw_C_coordinate_sha256=array_hash(raw[:k]),
                       regions_are_pre_output=True,source_Y_loaded=False,raw_faces=len(raw),known_faces=k)
            entry['selection_image']=render(entry);protected[entry['selection_image']['path']]=entry['selection_image']['sha256']
            for fi in range(len(raw)):
                face_rows.append(dict(request_id=rid,raw_slot=fi,is_C=fi<k,region='P1' if masks[0][fi] else 'P2' if masks[1][fi] else '',
                                      centroid_x=float(raw[fi,:,0].mean()),min_y=float(raw[fi,:,1].min())))
            entries.append(entry)
    assert len(entries)==6
    manifest=dict(version='regional_mix_dual_assisted_panel_v1',status='REGIONS_FIXED_PENDING_SELECTION_IMAGE_REVIEW',created_utc=now(),
                  selection_scope='AI_PROXY_ASSISTED_PRE_OUTPUT_DEVELOPMENT_FAULT_COVERAGE',requests=entries,protected=protected,
                  selection_reason='a8 and 27 cover thin support/end-frame surface loss; 459 additionally exposes RAW folds and D0-chain cleanup value. Fixed before any regional candidate.',
                  scope_limits=['3 development parents x 2 already viewed noises; not a representative or independent test set.',
                    'No new model/noise/query/camera/section. No unique source Y used to choose regions.',
                    'af/bcb and other high-N parents are outside this small structural feasibility panel; no high-N scope or latency certification follows. Their past long runtimes are not a result-based selection rule.',
                    'The 459/n1 gray-above-blue organization error can remain despite successful local support restoration.',
                    'Whole-face P1/P2 membership is assisted and deterministic after freeze; it is not the automatic detector result.'],
                  accounting=dict(model_calls=0,denoiser_calls=0,scoring_calls=0,backend_calls=0,new_noise=0,source_Y_arrays_loaded=0),
                  source_code_sha256=sha(__file__))
    path=OUT/'region_membership.csv'
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(face_rows[0]));w.writeheader();w.writerows(face_rows)
    manifest['membership_csv']=dict(path=str(path),sha256=sha(path));write(OUT/'manifest_pending_review.json',manifest)
    print(json.dumps(dict(status=manifest['status'],requests=[dict(request_id=e['request_id'],regions={r['region_id']:len(r['source_face_ids']) for r in e['regions']}) for e in entries]),indent=2))
    return manifest

def freeze():
    m=read(OUT/'manifest_pending_review.json');review=read(OUT/'selection_image_review.json')
    rr={r['request_id']:r for r in review['records']};assert set(rr)=={r['request_id'] for r in m['requests']}
    for e in m['requests']:
        r=rr[e['request_id']];assert r['actually_viewed'] is True and r['image_sha256']==e['selection_image']['sha256'] and r['region_membership_confirmed'] is True
        e['selection_image']['actual_review']=r
    for p,h in m['protected'].items():assert sha(p)==h,p
    m.update(status='FROZEN',frozen_utc=now(),selection_image_review_sha256=sha(OUT/'selection_image_review.json'))
    write(OUT/'manifest.json',m)
    gate=dict(status='PASS',utc=now(),requests=6,parents=3,regions=12,all_C_excluded=True,same_parent_same_spatial_definition=True,
              same_original_queries_and_noise=True,original_assets_unchanged=True,actual_prior_views=6,actual_region_image_views=6,
              raw_slot_domain='original generated RAW indices; never target-Y face IDs',model_calls=0,scoring_calls=0,backend_calls=0,
              manifest_sha256=sha(OUT/'manifest.json'),code_sha256=sha(__file__))
    write(OUT/'COMPLETE.json',gate);print(json.dumps(gate,indent=2));return gate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze']);a=p.parse_args()
    prepare() if a.action=='prepare' else freeze()
