"""Depth-correct paired learning-curve gallery of frozen saved meshes only."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse, io, html, json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from . import common as C
from .evaluate import LABELS,TRAIN_LABELS,REFERENCES,registration,verify
from r2.pilots.regular_patch_control_v1.depth_gallery import basis,raster

VERSION='full_seat_learning_curve_cpu_depth_gallery_v1'
SIZE=340


def run(labels=LABELS):
    labels=tuple(labels)
    if not labels or len(set(labels))!=len(labels) or any(x not in LABELS for x in labels):raise ValueError('Only registered model labels')
    r=registration();refs={};mesh={};index={}
    for label in labels:
        for stage in ('RAW','POST'):
            p=C.OUT/'evaluation/indices'/label/(stage.lower()+'_index.json')
            value=C.read(p)
            if value['status']!='COMPLETE' or len(value['rows'])!=24:raise ValueError('Complete fixed panel required before rendering')
            refs[str(p)]=C.sha(p)
            for rec in value['rows']:
                path=rec[stage.lower()+'_path'];digest=rec[stage.lower()+'_sha256'];verify({path:digest});refs[path]=digest
                a=C.arrays(path);tri=a['raw'] if stage=='RAW' else a['output'];tri=tri.reshape(-1,3,3)
                mesh[rec['request_id'],label,stage]=tri;index[rec['request_id'],label,stage]=rec
    B=[basis(-60,23),basis(120,23)];bounds={}
    for cid in {q['condition_id'] for q in r['requests']}:
        points=np.concatenate([mesh[q['request_id'],label,stage].reshape(-1,3) for q in r['requests'] if q['condition_id']==cid for label in labels for stage in ('RAW','POST')])
        center=(points.min(0)+points.max(0))/2;span=max(float(np.ptp((points-center)@b,axis=0)[:2].max()) for b in B)
        bounds[cid]=dict(center=center.tolist(),span=span)
    name='gallery' if labels==LABELS else 'gallery_'+'_'.join(labels)
    dest=C.OUT/'evaluation'/name;code={str(Path(__file__)):C.sha(__file__),str(C.ROOT/'r2/pilots/regular_patch_control_v1/depth_gallery.py'):C.sha(C.ROOT/'r2/pilots/regular_patch_control_v1/depth_gallery.py')}
    manifest=dest/'manifest.json'
    if manifest.exists():
        old=C.read(manifest)
        if tuple(old['labels'])!=labels or old['source_hashes']!=code or old['input_hashes']!=refs:raise ValueError('Gallery identity changed')
        for im in old['images']:verify({im['path']:im['sha256']})
        return old
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    pages=[]
    for q in r['requests']:
        page=Image.new('RGB',(SIZE*len(labels),SIZE*4+130),'white');draw=ImageDraw.Draw(page);b=bounds[q['condition_id']]
        draw.text((12,8),q['parent_id'][:8]+' FULL K='+str(q['K'])+' n'+str(q['candidate'])+' | same frozen C/noise; blue C, gray F',font=font,fill='black')
        records=[]
        for col,label in enumerate(labels):
            draw.text((col*SIZE+12,40),label,font=font,fill='black')
            for row,(stage,view) in enumerate((('RAW',0),('RAW',1),('POST',0),('POST',1))):
                im,owner=raster(mesh[q['request_id'],label,stage],int(q['K']),B[view],np.asarray(b['center']),b['span'],size=SIZE)
                page.paste(im,(col*SIZE,65+row*SIZE));draw.text((col*SIZE+5,68+row*SIZE),stage+' v'+str(view),font=small,fill='black')
                records.append(dict(label=label,stage=stage,view=view,visible_C_pixels=int(((owner>=0)&(owner<q['K'])).sum()),visible_F_pixels=int((owner>=q['K']).sum())))
        draw.text((10,SIZE*4+77),'CPU barycentric depth; same scale across all registered noises/models/stages of each C. Exact depth ties favor C.',font=small,fill='black')
        draw.text((10,SIZE*4+99),'Visibility is not topology or adoption certification. Historical models are frozen context references; checkpoints are fixed, not selected.',font=small,fill='black')
        p=dest/(q['parent_id'][:8]+'_n'+str(q['candidate'])+'.jpg');buf=io.BytesIO();page.save(buf,format='JPEG',quality=91,subsampling=0);C._write_bytes(p,buf.getvalue())
        pages.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],candidate=q['candidate'],path=str(p),sha256=C.sha(p),pixels=list(page.size),columns=records))
        print(json.dumps(dict(rendered=len(pages),registered=24)),flush=True)
    indexhtml='<meta charset="utf-8"><h1>FULL seat learning curve: fixed 24-request panel</h1><p>All saved RAW and single-D0 POST. No model calls, new samples or geometry edits. Blue visibility does not establish connectivity.</p>'
    indexhtml+=''.join('<h2>'+html.escape(x['parent_id'][:8]+'_n'+str(x['candidate']))+'</h2><a href="'+Path(x['path']).name+'"><img style="width:100%" src="'+Path(x['path']).name+'" loading="lazy"></a>' for x in pages)
    C._write_bytes(dest/'index.html',indexhtml.encode())
    result=dict(status='COMPLETE',version=VERSION,labels=labels,registration_sha256=C.sha(C.OUT/'registration.json'),
        source_hashes=code,input_hashes=refs,bounds=bounds,views=[x.tolist() for x in B],images=pages,new_geometry=0,model_calls=0,
        limitation='Two opposite orthographic views can miss subpixel gaps, hidden extra surfaces, and topological defects. No new adoption labels produced.')
    C.write(manifest,result);C.event('LEARNING_CURVE_GALLERY_COMPLETE',labels=labels,pages=24,model_calls=0);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',nargs='+',choices=LABELS,default=list(LABELS));a=p.parse_args();r=run(a.labels)
    print(json.dumps(dict(status=r['status'],pages=len(r['images']),labels=r['labels'])),flush=True)

if __name__=='__main__':main()
