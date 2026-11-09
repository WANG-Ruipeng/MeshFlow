"""Bench Step1 fixed-view/section gallery, reading completed CPU rescue cells.

All96 requests are registered before rendering. Original camera/section bounds
are reused exactly; images do not convert numerical metrics to visual labels.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import concurrent.futures
import html
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
from r2.pilots.seam_followup_v1 import gallery as G
from r2.pilots.seam_abc_v1 import eval_common as C
from r2.pilots.seam_abc_v1.render import measured_workers

ROOT=C.ROOT/'experiments/r2_unified_rescue_post_v1'
BENCH=C.ROOT/'experiments/r2_bench_parts_convergence_v1'
NAME='BENCH_UNIFIED_FIXED96'
OUT=ROOT/'gallery'/NAME
MANIFEST=ROOT/'gallery_manifests'/(NAME+'.json')


def prepare():
    if MANIFEST.exists():
        value=C.read(MANIFEST)
        for p,h in value['protected_definitions'].items():assert C.sha(p)==h,p
        for p,h in value['render_code'].items():assert C.sha(p)==h,p
        return value
    registration=ROOT/'registration.json';reg=C.read(registration)
    source_diag=C.ROOT/'experiments/r2_bench_post_rescue_diagnostic_v1/gallery/RAW_PRE_GEO_POST/definition.json'
    diagnostic=C.read(source_diag)
    protected={str(registration):C.sha(registration),str(source_diag):C.sha(source_diag)}
    definitions={}
    for label in ('SEAM_S15000','SEAM_S21000'):
        number=label.removeprefix('SEAM_S')
        p=BENCH/'gallery'/f'NATIVE_S{number}__SEAM_S{number}_full/definition.json'
        definitions[label]=(p,C.read(p));protected[str(p)]=C.sha(p)
    pages=[]
    for job in reg['jobs']:
        if not job['panel'].startswith('BENCH_'):continue
        q=job['q'];label=job['panel'].removeprefix('BENCH_');cid=q['condition_id'];rid=q['request_id']
        p,d=definitions[label]
        # The three already-discussed S21000 cases keep the exact diagnostic
        # camera and section bounds. Other requests reuse the original full panel.
        if label=='SEAM_S21000' and cid in diagnostic['bounds']:
            p,d=source_diag,diagnostic
        dest=ROOT/'cells'/job['panel']/rid
        cells=[dict(label='OLD_BASE',path=job['old_base']),dict(label='OLD_HINGE015',path=job['old_final']),
               dict(label='GUARD_GEO',path=str(dest/'GUARD_GEO/post.npz')),
               dict(label='UNIFIED_FINAL',path=str(dest/'FINAL/post.npz'))]
        pages.append(dict(**{k:q[k] for k in ('request_id','condition_id','parent_id','K','N','split')},
                          noise_index=q.get('noise_index',q.get('candidate')),panel=job['panel'],checkpoint=label,
                          part_type=q.get('part_type','UNKNOWN'),page_id=label+'__'+rid,
                          stage='OLD_BASE / OLD_HINGE / GUARD_GEO / UNIFIED_FINAL',
                          kind=label+' | '+q.get('part_type','UNKNOWN')+' | fixed historic camera and sections',
                          cells=cells,producer_completion=str(dest/'complete.json'),
                          bounds_source=str(p),bounds_source_sha256=C.sha(p),
                          bounds=d['bounds'][cid],depth_bounds=d['depth_bounds'][cid]))
    if len(pages)!=96 or len({p['page_id'] for p in pages})!=96:raise ValueError('Expected both48-request Bench endpoints')
    pages.sort(key=lambda r:(r['checkpoint'],r['request_id']))
    result=dict(status='FROZEN',pages=pages,protected_definitions=protected,
                render_code={str(Path(mod.__file__)):C.sha(mod.__file__) for mod in (G,G.D,G.H)},
                helper_code={str(Path(__file__)):C.sha(__file__)},
                policy='Exact historic depth bounds and finite sections; two opposing orthographic views (-60,23)/(120,23), front/side central C planes; no regenerated bounds',
                clipping='Fixed frame may clip a new outlier; per-page actual projected-outside-frame counts are recorded, never silently rescaled',
                model_calls=0,new_geometry=0,new_scores=0,metric_based_page_selection=False,
                review='PENDING_ACTUAL_AI_PROXY_VIEW; image generation alone is not a review')
    C.write(MANIFEST,result);return result


def validate_ready(page,registration):
    path=Path(page['producer_completion'])
    if not path.exists():return None
    producer=C.read(path)
    if producer['status']!='COMPLETE' or producer['request_id']!=page['request_id'] or producer['panel']!=page['panel']:
        raise ValueError('Completed producer identity differs')
    value=dict(page);cells=[]
    for cell in page['cells']:
        p=C.resolve(cell['path']);digest=C.sha(p)
        expected=(registration['protected'] if cell['label'].startswith('OLD_') else producer['files']).get(str(p))
        if expected!=digest:raise ValueError('Mesh not authenticated by original/completed producer: '+str(p))
        cells.append(dict(cell,sha256=digest))
    value['cells']=cells;value['producer_completion_sha256']=C.sha(path)
    return value


def render_one(job):
    # The original renderer needs no optional threadpool controller. This module
    # sets all numeric thread env vars before importing numpy in every spawn.
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    page,definition_sha=job
    # Independent worker has explicit output path; no dependency on G.OUT.
    old=OUT/(page['page_id']+'.json')
    if old.exists():
        prior=C.read(old)
        if prior['cells']!=page['cells']:raise ValueError('Rendered source geometry differs')
    result=G.one((page,str(OUT),definition_sha,page['bounds'],page['depth_bounds']))
    identity=OUT/(page['page_id']+'.identity.json')
    if not identity.exists():
        outside=[];db=page['depth_bounds']
        for cell in page['cells']:
            tri=C.arrays(cell['path'])['output'];points=tri.reshape(-1,3)
            for az in (-60,120):
                xy=((points.astype(float)-db['center'])@G.D.basis(az,23))[:,:2]
                outside.append(dict(stage=cell['label'],azimuth=az,
                                    corners_outside_image=int((np.abs(xy)>db['span']*.5/.86).any(axis=1).sum()),total_corners=len(points)))
        C.write(identity,dict(status='RENDERED_PENDING_ACTUAL_VIEW',page_id=page['page_id'],request_id=page['request_id'],
            condition_id=page['condition_id'],checkpoint=page['checkpoint'],part_type=page['part_type'],
            image_path=result['path'],image_sha256=result['image_sha256'],manifest_sha256=C.sha(MANIFEST),
            producer_completion=page['producer_completion'],producer_completion_sha256=page['producer_completion_sha256'],
            bounds_source=page['bounds_source'],bounds_source_sha256=page['bounds_source_sha256'],
            exact_historical_bounds=True,cells=page['cells'],projected_frame_check=outside,
            model_calls=0,new_geometry=0,new_scores=0,actual_viewed=False))
    if 'torch' in sys.modules:raise RuntimeError('CPU render imported neural framework')
    return result


def render_ready():
    manifest=prepare();reg=C.read(ROOT/'registration.json');OUT.mkdir(parents=True,exist_ok=True)
    ready=[p for row in manifest['pages'] if (p:=validate_ready(row,reg)) is not None]
    definition=OUT/'definition.json'
    definition_data=dict(version='bench_unified_historic_camera_v1',manifest_sha256=C.sha(MANIFEST),
                         source_code_sha256=C.sha(__file__),sources=manifest['render_code'],
                         policy=manifest['policy'],geometry_identity='Per-page cells sha256 bound to frozen original or completed Step1 producer',
                         pages=96,GT_display=False)
    if not definition.exists():C.write(definition,definition_data)
    elif C.read(definition)!=definition_data:raise ValueError('Gallery definition changed')
    dh=C.sha(definition);pending=[p for p in ready if not (OUT/(p['page_id']+'.identity.json')).exists()]
    tick=time.perf_counter();workers=0
    if pending:
        before=C.resources();first=render_one((pending[0],dh));snapshot=C.resources();workers,policy=measured_workers(snapshot,first)
        C.write(OUT/f'dispatch_{time.time_ns()}.json',dict(resources_before=before,resources_after_first=snapshot,workers=workers,policy=policy,
                    pending_pages=len(pending),first_page=pending[0]['page_id'],model_calls=0,source_cells_untouched=True))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            list(pool.map(render_one,[(p,dh) for p in pending[1:]]))
    images=[C.read(OUT/(p['page_id']+'.json')) for p in ready]
    for r in images:assert C.sha(r['path'])==r['image_sha256']
    html_page='<meta charset="utf-8"><h1>Bench unified rescue: fixed historical views and sections</h1>'
    html_page+='<p>Four columns: OLD_BASE, old HINGE, GUARD_GEO, UNIFIED_FINAL. Images alone do not certify exact connectivity, manifoldness or absence of intersections.</p>'
    html_page+=''.join('<h2>'+html.escape(r['page_id'])+'</h2><a href="'+Path(r['path']).name+'"><img loading="lazy" style="width:100%" src="'+Path(r['path']).name+'"></a>' for r in images)
    (OUT/'index.html').write_text(html_page,encoding='utf-8')
    result=dict(status='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW' if len(images)==96 else 'PARTIAL_READY_ONLY',
                expected=96,images=images,ready=len(images),waiting=96-len(images),new_rendered=len(pending),workers=workers,
                definition_sha256=dh,manifest_sha256=C.sha(MANIFEST),wall_seconds=time.perf_counter()-tick,
                model_calls=0,new_geometry=0,new_scores=0)
    C.write(OUT/'progress.json',result,replace=(OUT/'progress.json').exists())
    if len(images)==96 and not (OUT/'completion.json').exists():C.write(OUT/'completion.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','render-ready'));args=parser.parse_args()
    result=prepare() if args.action=='prepare' else render_ready()
    print(json.dumps({k:v for k,v in result.items() if k not in ('pages','images','protected_definitions','render_code','helper_code')},ensure_ascii=False,indent=2))
