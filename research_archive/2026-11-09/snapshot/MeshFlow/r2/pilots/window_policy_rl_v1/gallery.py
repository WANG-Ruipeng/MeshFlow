"""Anonymous CPU RAW/POST galleries for the frozen window policy pilot.

No score is read and no semantic label is assigned. The preserved renderer uses
per-pixel depth and finite sections; all model aliases are local to one request.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import concurrent.futures
import csv
import hashlib
import html
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
from . import postprocess as P
D,E=P.bind()
from r2.pilots.terminal_reward_nft_v1 import render_blind as R

VERSION='window_policy_rl_CPU_blind_gallery_v1'
PHASES={
    'calibration': dict(labels=('FULL_DET','S005','S010','S020'), request_key='calibration_requests',
                        count=64, directory='calibration'),
    'final': dict(labels=('FULL_DET','FULL_STOCH','RL_STOCH'), request_key='evaluation_requests',
                 count=44, directory='evaluation'),
}


def phase_spec(phase):
    if phase not in PHASES: raise ValueError('Only calibration and final galleries')
    cfg=PHASES[phase]
    manifest_path=P.OUT/'inputs/manifest.json'
    manifest=D.read(manifest_path)
    qs=manifest[cfg['request_key']]
    if len(qs)!=cfg['count'] or len({q['request_id'] for q in qs})!=cfg['count']:
        raise ValueError('Frozen request panel differs')
    for q in qs: P._safe_name(q['request_id'],'request_id')
    return cfg,manifest,qs


def prepare_blinding(phase):
    """Freeze a hash permutation using identities alone, without reading outputs."""
    P.torch_free()
    cfg,manifest,qs=phase_spec(phase)
    aliases=list('ABCD'[:len(cfg['labels'])])
    rows=[]
    for q in sorted(qs,key=lambda q:q['request_id']):
        rid=q['request_id']
        order=sorted(cfg['labels'],key=lambda label:(
            hashlib.sha256((VERSION+'/'+phase+'/'+rid+'/'+label).encode()).hexdigest(),label))
        rows.append(dict(request_id=rid,aliases=dict(zip(aliases,order))))
    result=dict(status='FROZEN',version=VERSION,phase=phase,rows=rows,
        manifest_sha256=D.sha(P.OUT/'inputs/manifest.json'),
        selection='SHA256 order of frozen phase/request/model identities only; no outputs, metrics, or outcome selection',
        alias_scope='Independent per-request aliases; not fixed global model names',
        distribution='Deterministic hash permutation; no balance guarantee',
        privacy='PRIVATE: do not pass mapping, real-label receipts or real-label paths to reviewers',
        model_calls=0)
    dest=P._checked(P.OUT/cfg['directory']/'private/label_mapping.json')
    D.write(dest,result)
    D.write(P.OUT/cfg['directory']/'blind_design.json',dict(status='FROZEN',version=VERSION,
        phase=phase,requests=len(qs),aliases=aliases,model_blinded=True,condition_visible=True,
        assignment='Frozen identity-only SHA256 permutation independently for each request',
        reviewers_receive='Anonymous images/manifest/template only; no mapping, scores or original-model paths',
        automatic_semantic_labels=False,model_calls=0))
    return result


def _copy(source,dest):
    source=D.resolve(source);dest=P._checked(dest);digest=D.sha(source)
    if dest.exists():
        if D.sha(dest)!=digest:raise ValueError('Anonymous byte copy changed')
        return digest
    dest.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as src,dest.open('xb') as dst:
        for block in iter(lambda:src.read(1<<20),b''):dst.write(block)
        dst.flush();os.fsync(dst.fileno())
    if D.sha(dest)!=digest:raise ValueError('Anonymous copy differs')
    return digest


def _anonymous_cases(phase):
    cfg,manifest,qs=phase_spec(phase)
    mapping=prepare_blinding(phase)
    aliases=list('ABCD'[:len(cfg['labels'])])
    byrid={row['request_id']:row['aliases'] for row in mapping['rows']}
    jobs=[dict(stage_dir=str(P.OUT/cfg['directory']/label),label=label,requests=qs) for label in cfg['labels']]
    rows,identity=P.prepare_jobs(jobs,manifest_path=P.OUT/'inputs/manifest.json',
                                 registration_path=P.OUT/'registration.json')
    # First require every completed POST before any anonymous geometry is copied.
    for row in rows:
        record=D.read(Path(row['out_dir'])/'receipt.json')
        if record['status']!='COMPLETE' or record['input_sha256']!=row['request']['input_sha256']:
            raise ValueError('Incomplete or mismatched POST panel')
        P._verify(record['files'])
    cases=[];provenance=[]
    for row in rows:
        q=row['request'];label=row['label'];rid=q['request_id']
        post=D.read(Path(row['out_dir'])/'receipt.json')
        alias=next(a for a,m in byrid[rid].items() if m==label)
        dest=P.OUT/'evaluation/anonymous_inputs'/phase/rid/alias
        rawsha=_copy(row['raw']['raw_path'],dest/'raw.npz')
        postsha=_copy(post['post_path'],dest/'post.npz')
        case={k:q[k] for k in ('request_id','condition_id','parent_id','candidate','split','N','K') if k in q}
        case.update(label=alias,raw_path=str(dest/'raw.npz'),post_path=str(dest/'post.npz'),
                    condition_path=q['condition_source'])
        cases.append(case)
        provenance.append(dict(request_id=rid,label=label,alias=alias,
            original_raw_path=row['raw']['raw_path'],original_post_path=post['post_path'],
            anonymous_raw_path=str(dest/'raw.npz'),anonymous_post_path=str(dest/'post.npz'),
            raw_sha256=rawsha,post_sha256=postsha))
    D.write(P.OUT/cfg['directory']/'private/anonymous_geometry_provenance.json',
        dict(status='COMPLETE',version=VERSION,phase=phase,rows=provenance,
             copies_are_byte_identical=True,identity=identity))
    return cases,aliases


def _load(cases,aliases,phase):
    grouped={};allfiles={}
    anonymous_root=(P.OUT/'evaluation/anonymous_inputs'/phase).resolve()
    for q in cases:
        rid=q['request_id'];alias=q['label'];K=int(q['K']);N=int(q['N'])
        for field in ('raw_path','post_path'):
            if not D.resolve(q[field]).resolve().is_relative_to(anonymous_root):
                raise ValueError('Renderer receives only anonymous geometry copies')
        raw=E.arrays(q['raw_path'])['raw'].reshape(N,3,3)
        postfile=E.arrays(q['post_path']);post=postfile['output']
        source=E.source_metadata(q['condition_path'],raw[:K])
        raw=raw.copy();raw[:K]=source['C_original']
        if not D.same(raw[:K],post[:K]):raise ValueError('C source order/geometry differs')
        if not np.isfinite(raw).all() or not np.isfinite(post).all():raise ValueError('Nonfinite render geometry')
        if not D.same(postfile['vertices'][postfile['faces']],post):raise ValueError('Indexed POST differs')
        files={str(D.resolve(q[k])):D.sha(q[k]) for k in ('raw_path','post_path','condition_path')}
        if alias in grouped.get(rid,{}):raise ValueError('Repeated anonymous cell')
        grouped.setdefault(rid,{})[alias]=dict(case=q,RAW=raw,POST=post,files=files)
        allfiles.update(files)
    if any(set(cells)!=set(aliases) for cells in grouped.values()):raise ValueError('Incomplete anonymous comparison')
    cameras=[R.basis(a,e) for _,a,e in R.VIEWS]
    points={}
    for cells in grouped.values():
        cid=cells[aliases[0]]['case']['condition_id']
        points.setdefault(cid,[]).extend(cell[s].reshape(-1,3) for cell in cells.values() for s in ('RAW','POST'))
    bounds={}
    for cid,arrays in points.items():
        xyz=np.concatenate(arrays);center=(xyz.min(0)+xyz.max(0))/2
        span=max(float(np.ptp((xyz-center)@camera,axis=0)[:2].max()) for camera in cameras)
        if span<=0:raise ValueError('Degenerate condition view bounds')
        bounds[cid]=dict(center=center.tolist(),span=span)
    return grouped,allfiles,bounds


def _page(job):
    P.torch_free();P.bind()
    from r2.pilots.terminal_reward_nft_v1.cpu_worker import _single_thread
    with _single_thread():result=R._render_page(job)
    P.torch_free()
    return result


def render(phase,*,workers=1,size=360,contact_sheets=True):
    P.torch_free()
    if workers not in (1,2):raise ValueError('Resource-admitted CPU workers must be one or two')
    if size!=360:raise ValueError('Frozen per-view render size is 360 pixels')
    cfg,manifest,qs=phase_spec(phase)
    cases,aliases=_anonymous_cases(phase)
    grouped,allfiles,bounds=_load(cases,aliases,phase)
    dest=P._checked(P.OUT/cfg['directory']/'blind_gallery')
    source_paths=[Path(__file__),Path(R.__file__),
        D.ROOT/'r2/pilots/regular_patch_control_v1/depth_gallery.py',
        D.ROOT/'r2/tools/cref_layer_audit_v1/render.py']
    identity=dict(version=VERSION,phase=phase,requests=len(qs),cases=len(cases),aliases=aliases,
        input_hashes=allfiles,source_hashes={str(p):D.sha(p) for p in source_paths},
        manifest_sha256=D.sha(P.OUT/'inputs/manifest.json'),bounds=bounds,views=R.VIEWS,size=size,
        scale='One shared bound per condition across all models, noise requests, RAW and POST',
        model_blinded=True,condition_visible=True,model_calls=0)
    complete=dest/'manifest.json'
    if complete.exists():
        old=D.read(complete)
        if old['identity']!=D.clean(identity):raise ValueError('Completed gallery identity differs')
        P._verify(old['files'])
        return dict(old,cache_verified=True)
    if dest.exists():raise RuntimeError('Unfinished gallery requires explicit engineering recovery')
    D.write(dest/'registration.json',identity)
    tick=time.perf_counter()
    try:
        jobs=[(rid,cells,aliases,bounds[cells[aliases[0]]['case']['condition_id']],str(dest),size)
              for rid,cells in sorted(grouped.items())]
        if workers==1:images=[_page(job) for job in jobs]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                images=list(pool.map(_page,jobs))
        byrid={r['request_id']:r for r in images};reviews=[]
        for q in sorted(cases,key=lambda x:(x['request_id'],x['label'])):
            im=byrid[q['request_id']]
            reviews.append(dict(request_id=q['request_id'],condition_id=q['condition_id'],label=q['label'],
                parent_id=q.get('parent_id',''),candidate=q.get('candidate',''),split=q.get('split',''),
                RAW_adoption='UNKNOWN',POST_adoption='UNKNOWN',POST_body_complete='UNKNOWN',
                review_status='NOT_REVIEWED',reviewer_kind='',reviewer='',blinded=True,
                image_path=im['image_path'],image_sha256=im['image_sha256'],notes='',
                limitations='Three depth-correct views and two finite sections; hidden/subpixel defects and actual topology are not certified.'))
        template=dest/'review_template.csv'
        with template.open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(reviews[0]));writer.writeheader();writer.writerows(reviews)
        sheets=[]
        if contact_sheets:
            for start in range(0,len(images),4):
                group=images[start:start+4];tw=960;th=int(tw*(3*size+2*int(size*.65)+158)/(len(aliases)*2*size))+25
                sheet=Image.new('RGB',(2*tw,2*th),'white');draw=ImageDraw.Draw(sheet)
                for i,record in enumerate(group):
                    with Image.open(record['image_path']) as image:
                        thumb=image.convert('RGB');thumb.thumbnail((tw,th-25),Image.Resampling.LANCZOS)
                    x=i%2*tw;y=i//2*th;sheet.paste(thumb,(x,y+25))
                    draw.text((x+5,y+5),str(start+i)+': '+record['request_id'][-80:],font=R.font(12),fill='black')
                path=dest/'contact_sheets'/('sheet_'+str(start//4).zfill(2)+'.jpg')
                R._save_image(path,sheet)
                sheets.append(dict(path=str(path),sha256=D.sha(path),request_ids=[r['request_id'] for r in group]))
        links=''.join('<h2>'+html.escape(r['request_id'])+'</h2><a href="images/'+Path(r['image_path']).name+'"><img loading="lazy" style="width:100%;max-width:2200px" src="images/'+Path(r['image_path']).name+'"></a>' for r in images)
        with (dest/'index.html').open('x',encoding='utf8') as stream:
            stream.write('<meta charset="utf-8"><h1>Window policy: anonymous actual RAW / POST</h1><p>Blue C; gray generated F. Aliases vary per request. No automatic semantic labels.</p>'+links)
        public=dict(status='COMPLETE',identity=identity,images=images,contact_sheets=sheets,
            review_template=str(template),review_template_sha256=D.sha(template),model_calls=0,GPU_calls=0,
            automatic_visual_labels=0,workers=workers,CPU_threads_per_worker=1,seconds=time.perf_counter()-tick)
        serialized=json.dumps(D.clean(public),ensure_ascii=False)
        if any(label in serialized for label in cfg['labels']):
            raise RuntimeError('Real model label leaked into public gallery')
        P._verify(allfiles);P.torch_free()
        files={r['image_path']:r['image_sha256'] for r in images}
        files.update({r['path']:r['sha256'] for r in sheets})
        files.update({str(template):D.sha(template),str(dest/'index.html'):D.sha(dest/'index.html')})
        public['files']=files
        D.write(complete,public)
        return public
    except BaseException as error:
        D.write(dest/'failure.json',dict(status='STOPPED',error=repr(error),automatic_retry=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=tuple(PHASES),required=True)
    parser.add_argument('--workers',type=int,choices=(1,2),default=1)
    parser.add_argument('--prepare-blinding',action='store_true')
    args=parser.parse_args()
    if args.prepare_blinding:
        result=prepare_blinding(args.phase)
        print(json.dumps(dict(status=result['status'],requests=len(result['rows']),model_calls=0)))
    else:
        result=render(args.phase,workers=args.workers)
        print(json.dumps(dict(status=result['status'],pages=len(result['images']),model_calls=0,
                              seconds=result['seconds'],cache_verified=result.get('cache_verified',False))))


if __name__=='__main__':main()
