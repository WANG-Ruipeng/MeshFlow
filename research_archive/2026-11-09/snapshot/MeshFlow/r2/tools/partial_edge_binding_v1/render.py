"""CPU depth-only paired saved-mesh figures; never assigns visual labels."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import html
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .assets import ROOT, OUT, arrays, resolve, sha
from r2.pilots.regular_patch_control_v1.depth_gallery import basis, raster

VERSION='partial_binding_cpu_paired_depth_v1'
VIEWS=(('Front',-90,0),('Side',0,0),('Top oblique',-60,75))
SIZE=460


def _font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','C:/Windows/Fonts/arial.ttf'):
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()


def _tri(a):
    if 'output' in a: return a['output'].reshape(-1,3,3)
    return a['vertices'][a['faces']]


def render_case(row, output_npz, outjpg, label=None):
    """Render one original POST vs finalized modified POST, 3 identical views.

    output_npz accepts output or vertices+faces, with source-restored C first K.
    Returns saved-image receipt; no automatic visual adoption decisions.
    """
    before=arrays(row['post_path']); after=arrays(output_npz)
    meshes=(_tri(before),_tri(after)); k=int(row['K'])
    if meshes[0][:k].dtype != meshes[1][:k].dtype or meshes[0][:k].tobytes()!=meshes[1][:k].tobytes():
        raise ValueError('Renderer requires unchanged source-order C')
    points=np.concatenate([x.reshape(-1,3) for x in meshes])
    center=(points.min(0)+points.max(0))/2
    camera=[basis(az,el) for _,az,el in VIEWS]
    span=max(float(np.ptp((points-center)@B,axis=0)[:2].max()) for B in camera)
    page=Image.new('RGB',(2*SIZE,3*SIZE+145),'white');draw=ImageDraw.Draw(page)
    font,small=_font(18),_font(13)
    label=label or resolve(output_npz).parent.name
    draw.text((12,8),row['short_id']+' | '+row['split']+' | blue: preserved C; gray: F',font=font,fill='black')
    records=[]
    for col,(title,tri) in enumerate(zip(('Original FULL_S6000 POST',label),meshes)):
        draw.text((col*SIZE+12,40),title,font=font,fill='black')
        for j,((name,_,_),B) in enumerate(zip(VIEWS,camera)):
            im,owner=raster(tri,k,B,center,span,size=SIZE)
            page.paste(im,(col*SIZE,65+j*SIZE))
            draw.text((col*SIZE+7,69+j*SIZE),name,font=small,fill='black')
            records.append(dict(stage='baseline' if col==0 else 'modified',view=name,
                visible_C_pixels=int(((owner>=0)&(owner<k)).sum()),visible_F_pixels=int((owner>=k).sum())))
    draw.text((10,3*SIZE+77),'Same scale, cameras and depth rendering; exact depth ties favor C.',font=small,fill='black')
    draw.text((10,3*SIZE+98),'Images cannot certify topology or absence of intersections. Labels require actual review.',font=small,fill='black')
    path=resolve(outjpg);path.parent.mkdir(parents=True,exist_ok=True)
    buf=io.BytesIO();page.save(buf,format='JPEG',quality=92,subsampling=0)
    payload=buf.getvalue()
    if path.exists() and path.read_bytes()!=payload:
        raise ValueError('Refusing changed render overwrite: '+str(path))
    if not path.exists():path.write_bytes(payload)
    receipt=dict(version=VERSION,request_id=row['request_id'],short_id=row['short_id'],label=label,
        path=str(path),sha256=sha(path),baseline_path=row['post_path'],baseline_sha256=sha(row['post_path']),
        modified_path=str(resolve(output_npz)),modified_sha256=sha(output_npz),center=center.tolist(),span=span,
        views=[dict(name=n,azimuth=a,elevation=e) for n,a,e in VIEWS],pixels=list(page.size),visibility=records,
        model_calls=0,scoring_calls=0,review_status='NOT_REVIEWED',reviewer_kind=None,
        limitations='Three CPU orthographic views; hidden or subpixel defects may be missed. No automated adoption judgments.')
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    return receipt


def write_gallery(receipts, out_dir=None):
    dest=resolve(out_dir or OUT/'gallery');dest.mkdir(parents=True,exist_ok=True)
    blocks=[]
    for r in receipts:
        rel=os.path.relpath(resolve(r['path']),dest).replace('\\','/')
        blocks.append('<h2>'+html.escape(r['short_id']+' | '+r['label'])+'</h2><a href="'+html.escape(rel)+'"><img loading="lazy" style="width: min(100%, 1000px)" src="'+html.escape(rel)+'"></a>')
    (dest/'index.html').write_text('<meta charset="utf-8"><h1>Constrained partial edge binding: paired CPU figures</h1><p>Selected by recorded conversion, regression and cost criteria; all numerical rows remain in the full tables.</p>'+''.join(blocks),encoding='utf8')
    result=dict(version=VERSION,status='COMPLETE',images=receipts,model_calls=0,automatic_visual_labels=0,
        renderer_sha256=sha(__file__),raster_source_sha256=sha(ROOT/'r2/pilots/regular_patch_control_v1/depth_gallery.py'))
    (dest/'manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result
