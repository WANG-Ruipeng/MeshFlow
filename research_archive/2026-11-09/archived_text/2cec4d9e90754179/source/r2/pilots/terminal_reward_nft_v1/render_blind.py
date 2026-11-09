"""Blinded final RAW/POST CPU depth gallery; models are anonymized upstream.

No metric threshold is turned into a semantic label. All review fields remain
UNKNOWN until the named reviewer actually inspects the frozen image artifacts.
"""
from __future__ import annotations
import concurrent.futures
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from . import evaluation as E
from r2.pilots.regular_patch_control_v1.depth_gallery import basis,raster
from r2.tools.cref_layer_audit_v1.render import finite_sections

VERSION='terminal_reward_nft_CPU_depth_gallery_blind_v1'
VIEWS=(('Front',-90.,0.),('Side',0.,0.),('Top oblique',-60.,75.))


def font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','C:/Windows/Fonts/arial.ttf'):
        if Path(p).exists():return ImageFont.truetype(p,size)
    return ImageFont.load_default()


def _save_image(path,page):
    path=E.resolve(path);path.parent.mkdir(parents=True,exist_ok=True)
    buf=io.BytesIO();page.save(buf,format='JPEG',quality=94,subsampling=0);data=buf.getvalue()
    if path.exists() and path.read_bytes()!=data:raise ValueError('Changed existing gallery image: '+str(path))
    if not path.exists():path.write_bytes(data)


def _section_image(tri,K,view,size,height):
    C=tri[:K];lower=C.reshape(-1,3).min(0).astype(float);upper=C.reshape(-1,3).max(0).astype(float)
    center=(lower+upper)/2;axis=2 if view=='Front section' else 0;dims=[0,1] if axis==2 else [2,1]
    margin=np.asarray([.075,.10,.075])*E.L;lo=lower-margin;hi=upper+margin
    scale=min((size-24)/(hi[dims[0]]-lo[dims[0]]),(height-30)/(hi[dims[1]]-lo[dims[1]]))
    image=Image.new('RGB',(size,height),'white');draw=ImageDraw.Draw(image)
    for mesh,color,width in ((tri[K:],(110,118,126),2),(C,(22,134,192),3)):
        segments,_=finite_sections(mesh,axis,float(center[axis]),eps=1e-12*E.L)
        for segment in segments:
            p=(segment[:,dims]-center[dims])*scale
            xy=[(float(size/2+x),float(height/2-y)) for x,y in p]
            draw.line(xy,fill=color,width=width)
    draw.rectangle((1,1,size-2,height-2),outline=(210,210,210))
    draw.text((6,5),view+' at C bbox center',font=font(12),fill='black')
    return image

def _render_page(job):
    rid,cells,labels,bounds,dest,size=job
    cols=2*len(labels);header=78;footer=80;section_height=int(size*.65)
    page=Image.new('RGB',(cols*size,3*size+2*section_height+header+footer),'white');draw=ImageDraw.Draw(page)
    f,small=font(16),font(12)
    first=cells[labels[0]];q=first['case'];k=int(q['K']);cameras=[basis(a,e) for _,a,e in VIEWS]
    title=(str(q.get('parent_id',q['condition_id']))[:12]+' | '+str(q.get('split',''))+' | K='+str(k)+' N='+str(q['N'])+' | noise '+str(q.get('candidate',q.get('noise_seed',''))))
    draw.text((10,8),title,font=f,fill='black');draw.text((10,30),'BLUE preserved input C | GRAY generated F | actual mesh, CPU depth | MODEL-BLINDED review; condition visible',font=small,fill='black')
    records=[]
    for li,label in enumerate(labels):
        cell=cells[label]
        for si,stage in enumerate(('RAW','POST')):
            col=2*li+si;tri=cell[stage]
            draw.text((col*size+7,54),label+' / '+stage,font=f,fill='black')
            for j,((name,_,_),B) in enumerate(zip(VIEWS,cameras)):
                im,owner=raster(tri,k,B,np.asarray(bounds['center']),bounds['span'],size=size)
                page.paste(im,(col*size,header+j*size));draw.text((col*size+5,header+j*size+4),name,font=small,fill='black')
                records.append(dict(label=label,stage=stage,view=name,visible_C_pixels=int(((owner>=0)&(owner<k)).sum()),visible_F_pixels=int((owner>=k).sum())))
            for j,name in enumerate(('Front section','Side section')):
                sim=_section_image(tri,k,name,size,section_height)
                page.paste(sim,(col*size,header+3*size+j*section_height))
    y=header+3*size+2*section_height
    draw.text((10,y+8),'Same scale per condition across registered candidates/models/stages. Exact depth ties favor C.',font=small,fill='black')
    draw.text((10,y+28),'Hidden/subpixel defects can be missed; visual touching is not legal connectivity. No labels assigned by renderer.',font=small,fill='black')
    draw.text((10,y+48),rid,font=small,fill='black')
    stem=str(q.get('parent_id','case'))[:8]+'_'+hashlib.sha256(rid.encode()).hexdigest()[:14]
    path=Path(dest)/'images'/(stem+'.jpg');_save_image(path,page)
    return dict(request_id=rid,condition_id=q['condition_id'],labels=labels,image_path=str(path),image_sha256=E.sha(path),
        pixels=list(page.size),views=[dict(name=n,azimuth=a,elevation=e) for n,a,e in VIEWS],bounds=bounds,visibility=records,finite_sections='C-only center front/side; original finite_sections helper; blue over gray in cuts; not topology certification',
        source_files={p:h for cell in cells.values() for p,h in cell['files'].items()},review_status='NOT_REVIEWED',automatic_visual_labels=0)


def render_panel(cases,out_dir,*,label_order=None,expected_cases=None,workers=1,size=360,contact_sheets=True):
    """cases: request metadata + label/raw_path/post_path/condition_path.

    Multiple models sharing request_id become paired columns in the same page.
    Training: 32 cases / 32 pages; final: 36 requests x 3 labels / 36 pages.
    Root owns resource measurement and passes the authorized worker count.
    """
    if 'torch' in sys.modules:raise RuntimeError('Render in a separate CPU/model-free process')
    if workers not in (1,2):raise ValueError('Renderer accepts measured one/two-worker dispatch')
    if expected_cases is not None and len(cases)!=expected_cases:raise ValueError('Incomplete frozen panel')
    if set(c['label'] for c in cases)!={'A','B','C'}:raise ValueError('Blind review requires exactly anonymous A/B/C labels')
    safe_fields={'request_id','condition_id','label','parent_id','candidate','split','N','K','raw_path','post_path','condition_path','noise_seed'}
    anonymous_root=(E.OUT/'evaluation/anonymous_inputs').resolve()
    for c in cases:
        if set(c)-safe_fields:raise ValueError('Unexpected metadata may unblind reviewer: '+str(sorted(set(c)-safe_fields)))
        for field in ('raw_path','post_path'):
            if not E.resolve(c[field]).resolve().is_relative_to(anonymous_root):raise ValueError('Use upstream anonymous mesh copies for blinded rendering')
    tick=time.perf_counter();dest=E.resolve(out_dir);grouped={};allfiles={};labels=label_order or list(dict.fromkeys(c['label'] for c in cases))
    for case in cases:
        q=dict(case);rid=q['request_id'];label=q['label'];k=int(q['K']);n=int(q['N'])
        if label not in labels:raise ValueError('Unregistered render label')
        if label in grouped.get(rid,{}):raise ValueError('Duplicate render case')
        raw=E.arrays(q['raw_path'])['raw'].reshape(n,3,3);postfile=E.arrays(q['post_path']);post=postfile['output']
        source=E.source_metadata(q['condition_path'],raw[:k]);raw=raw.copy();raw[:k]=source['C_original']
        if post[:k].dtype!=raw[:k].dtype or post[:k].tobytes()!=raw[:k].tobytes():raise ValueError('C geometry/order differs across stages')
        if not np.isfinite(raw).all() or not np.isfinite(post).all():raise ValueError('Nonfinite render input')
        if not np.array_equal(postfile['vertices'][postfile['faces']],post):raise ValueError('POST indexed geometry mismatch')
        files={str(E.resolve(q[p])):E.sha(q[p]) for p in ('raw_path','post_path','condition_path')};allfiles.update(files)
        grouped.setdefault(rid,{})[label]=dict(case=q,RAW=raw,POST=post,files=files)
    if any(set(g)!=set(labels) for g in grouped.values()):raise ValueError('Each request needs every registered comparison label')
    cameras=[basis(a,e) for _,a,e in VIEWS];conditions={}
    for rid,g in grouped.items():
        cid=g[labels[0]]['case']['condition_id'];conditions.setdefault(cid,[]).extend([c[s].reshape(-1,3) for c in g.values() for s in ('RAW','POST')])
    bounds={}
    for cid,points in conditions.items():
        points=np.concatenate(points);center=(points.min(0)+points.max(0))/2
        span=max(float(np.ptp((points-center)@B,axis=0)[:2].max()) for B in cameras)
        if span<=0:raise ValueError('Degenerate render extent')
        bounds[cid]=dict(center=center.tolist(),span=span)
    identity=dict(version=VERSION,cases=len(cases),requests=len(grouped),labels=labels,input_hashes=allfiles,
        source_hashes={str(Path(__file__)):E.sha(__file__),str(Path(__file__).with_name('render.py')):E.sha(Path(__file__).with_name('render.py')),str(E.ROOT/'r2/pilots/regular_patch_control_v1/depth_gallery.py'):E.sha(E.ROOT/'r2/pilots/regular_patch_control_v1/depth_gallery.py'),str(E.ROOT/'r2/tools/cref_layer_audit_v1/render.py'):E.sha(E.ROOT/'r2/tools/cref_layer_audit_v1/render.py')},
        blinded=True,review_scope='Model identity hidden; source condition and noise identities visible; mapping is not supplied to renderer',views=VIEWS,size=size,scale='same per condition across all noise/model/stage meshes',model_calls=0)
    complete=dest/'manifest.json'
    if complete.exists():
        old=E.read(complete)
        if old['identity']!=E.clean(identity):raise ValueError('Gallery identity changed')
        for r in old['images']:E.verify({r['image_path']:r['image_sha256']})
        return old
    E.write(dest/'registration.json',identity)
    jobs=[(rid,g,labels,bounds[g[labels[0]]['case']['condition_id']],str(dest),size) for rid,g in sorted(grouped.items())]
    if workers==1:receipts=[_render_page(j) for j in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:receipts=list(pool.map(_render_page,jobs))
    reviews=[]
    byrid={r['request_id']:r for r in receipts}
    for q in cases:
        rr=byrid[q['request_id']]
        reviews.append(dict(request_id=q['request_id'],condition_id=q['condition_id'],label=q['label'],
            parent_id=q.get('parent_id',''),candidate=q.get('candidate',''),split=q.get('split',''),
            RAW_adoption='UNKNOWN',POST_adoption='UNKNOWN',POST_body_complete='UNKNOWN',
            review_status='NOT_REVIEWED',reviewer_kind='',reviewer='',blinded=True,
            image_path=rr['image_path'],image_sha256=rr['image_sha256'],notes='',limitations='Three views; hidden and subpixel issues can be missed; visual contact does not certify topology.'))
    template=dest/'review_template.csv'
    if template.exists():raise ValueError('Blank template already exists without completion; explicit recovery required')
    with template.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(reviews[0]));writer.writeheader();writer.writerows(reviews)
    sheets=[]
    if contact_sheets:
        for start in range(0,len(receipts),4):
            group=receipts[start:start+4];tw=720 if len(labels)>1 else 450;th=int(tw*(3*size+2*int(size*.65)+158)/(len(labels)*2*size))+25
            im=Image.new('RGB',(tw*2,th*2),'white');d=ImageDraw.Draw(im)
            for j,r in enumerate(group):
                small=Image.open(r['image_path']).convert('RGB');small.thumbnail((tw,th-25),Image.Resampling.LANCZOS)
                x=j%2*tw;y=j//2*th;im.paste(small,(x,y+25));d.text((x+5,y+5),str(start+j)+': '+r['request_id'][-70:],font=font(12),fill='black')
            p=dest/'contact_sheets'/('sheet_'+str(start//4).zfill(2)+'.jpg');_save_image(p,im)
            sheets.append(dict(path=str(p),sha256=E.sha(p),request_ids=[r['request_id'] for r in group]))
    links=''.join('<h2>'+html.escape(r['request_id'])+'</h2><a href="images/'+Path(r['image_path']).name+'"><img loading="lazy" style="width:100%;max-width:1800px" src="images/'+Path(r['image_path']).name+'"></a>' for r in receipts)
    (dest/'index.html').write_text('<meta charset="utf-8"><h1>Terminal reward pilot: blinded actual RAW / POST</h1><p>Blue C, gray F; every registered sample shown. Reviews remain separate and must be made from actual viewing.</p>'+links,encoding='utf8')
    E.verify(allfiles)
    result=dict(status='COMPLETE',identity=identity,images=receipts,contact_sheets=sheets,review_template=str(template),
        review_template_sha256=E.sha(template),model_calls=0,GPU_calls=0,automatic_visual_labels=0,workers=workers,seconds=time.perf_counter()-tick)
    E.write(complete,result);return result
