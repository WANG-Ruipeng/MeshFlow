"""CPU-only DEV galleries from immutable saved meshes and metric vectors.

CONFIRM is deliberately refused. Rendering never samples/repairs a mesh,
evaluates a metric or calls a model. AI observations remain PENDING.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import time

for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LP_NUM_THREADS'):
    os.environ.setdefault(_key,'2')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
os.environ.setdefault('GALLIUM_DRIVER','llvmpipe')
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from . import assets

RENDERER = assets.ROOT/'native_t1/runs/chair_two_triangle_control_v1/scripts/raw_renderer.py'
VERSION = 'chair_control_feedback_raw_gallery_v1'
VIEWS = [('front_3q',45.,22.),('back_3q',225.,22.),('front',0.,8.),('side',90.,8.),('wire',45.,22.)]
_renderer_module = None


def _renderer():
    global _renderer_module
    if _renderer_module is None:
        spec=importlib.util.spec_from_file_location('chair_feedback_private_renderer',RENDERER)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        _renderer_module=module
    return _renderer_module


def _write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def _hash(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _font(size,bold=False):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',size)


def _missing(path,size,text='NOT READY'):
    image=Image.new('RGB',(size,size),'#F4F5F7');d=ImageDraw.Draw(image)
    d.text((20,size//2-15),text,font=_font(20,True),fill='#69727C');image.save(path)


def _raw(out,label,request):
    p=Path(out)/'runtime'/label/request['id']/'raw.npz'
    result=p.with_name('result.json')
    if p.exists() and result.exists():
        r=assets.read_json(result)
        if r.get('status') not in ('PASS','DONE','SAMPLED','CACHED'):return None
        if r.get('raw_sha256') and assets.digest(p)!=r['raw_sha256']:raise ValueError('RAW result hash differs')
        return p
    if label=='START' and request.get('raw_cache'):
        r=request['raw_cache'];p=assets.resolve(r['path'])
        if r['status']!='PASS' or assets.digest(p)!=r['sha256']:raise ValueError('START cache identity changed')
        return p
    return None


def _render(C,F,path,camera,size,wire):
    r=_renderer();original=r._actor
    if wire:
        def wire_actor(triangles,color,edges=False):
            actor,receipt=original(triangles,color,edges=False)
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetLineWidth(1.)
            receipt.update(representation='wireframe',line_width=1.)
            return actor,receipt
        r._actor=wire_actor
    try:
        receipt=r.render_panel(C,F,path,camera,size=(size,size),mode='output' if len(F) else 'context')
    finally:
        r._actor=original
    receipt['style']=dict(receipt['style'],representation='wireframe' if wire else 'surface')
    receipt['wireframe_display_only']=bool(wire)
    return receipt


def _layout(path,title,columns,rows,row_names,size,subtitle):
    left=135;gap=8;top=126;bottom=45
    w=left+len(columns)*(size+gap)+12;h=top+len(rows)*(size+gap)+bottom
    im=Image.new('RGB',(w,h),'white');d=ImageDraw.Draw(im)
    d.text((18,12),title,font=_font(25,True),fill='#202833')
    d.text((18,48),subtitle,font=_font(16),fill='#586574')
    boxes=[]
    for j,name in enumerate(columns):
        name=name if len(name)<32 else name[:29]+'...'
        d.text((left+j*(size+gap)+8,91),name,font=_font(19,True),fill='#263341')
    for i,row in enumerate(rows):
        y=top+i*(size+gap)
        for line_i,line in enumerate(row_names[i].split('\n')):
            d.text((12,y+size//2-24+line_i*22),line,font=_font(16,line_i==0),fill='#344354')
        for j,p in enumerate(row):
            x=left+j*(size+gap)
            with Image.open(p) as panel:
                assert panel.size==(size,size)
                im.paste(panel.convert('RGB'),(x,y))
            boxes.append(dict(row=i,column=j,path=str(p),xyxy=[x,y,x+size,y+size]))
    d.text((18,h-31),'Blue: exact C. Orange: free, including GT free. Unmodified geometry; no crop, repair, welding or smoothing.',font=_font(15),fill='#586574')
    im.save(path,dpi=(150,150))
    return dict(path=str(path),sha256=assets.digest(path),size=list(im.size),panel_boxes=boxes,panels_uncropped=True)


def _metric_source(out,label,request):
    folder=Path(out)/'evaluation'/label/'DEV'/'scoring'
    p=folder/'control'/(request['id']+'.npz');j=folder/'joined'/(request['id']+'.json')
    ref=Path(out)/'references'/request['condition_id']/'control.npz'
    if not p.exists() or not j.exists() or not ref.exists():return None
    row=assets.read_json(j)['metrics']
    if row.get('control_vectors_sha256')!=assets.digest(p):raise ValueError('Saved control vectors hash differs')
    return dict(vectors=p,reference=ref,joined=j,metrics=row)


def _overlay(background,path,camera,source,kind,size):
    v=assets.arrays(source['vectors']);ref=assets.arrays(source['reference'])
    if kind=='S':
        points=ref['gamma_points'];hit=v['seam_covered_0.01'].astype(bool);unknown=v['seam_uncertain'].astype(bool)
        classes=[('covered',hit,'#14984D'),('uncovered',~hit & ~unknown,'#D93732'),('uncertain',unknown,'#D2A207')]
    elif kind=='I':
        points=ref['C_points'];cross=v['conflict_cross'].astype(bool);dup=v['conflict_dup'].astype(bool);unk=v['conflict_unknown'].astype(bool)
        classes=[('clear',~(cross|dup|unk),'#8498A7'),('cross',cross,'#E33035'),('duplicate',dup,'#B740A1'),('unknown',unk,'#D2A207')]
    else:
        points=ref['C_points'];eligible=v['exposure_reference_exposed'].astype(bool)
        classes=[('not_reference_exposed',~eligible,'#B8BEC5'),('retained',v['exposure_retained_lower'].astype(bool),'#14984D'),
            ('occluded',v['exposure_occluded'].astype(bool),'#D93732'),('output_unknown',v['exposure_uncertain'].astype(bool),'#D2A207'),
            ('reference_unknown',v['exposure_reference_uncertain'].astype(bool),'#894DA8')]
    # Diagnostic x-ray point projection, explicitly separate from opaque RAW
    # mesh views. Coordinates/masks are the exact saved metric arrays.
    centered=points.astype(np.float64)-np.asarray(camera['focal_point'])
    u=.5+centered@np.asarray(camera['right'])/(2*camera['parallel_scale']*camera['aspect'])
    w=.5-centered@np.asarray(camera['view_up'])/(2*camera['parallel_scale'])
    assert np.all((u>=0)&(u<=1)&(w>=0)&(w<=1)),'Metric source point outside shared geometry frame'
    with Image.open(background) as p:im=p.convert('RGB')
    d=ImageDraw.Draw(im,'RGBA');counts={}
    for name,mask,color in classes:
        assert mask.shape==(len(points),)
        counts[name]=int(mask.sum())
        for x,y in np.column_stack([u[mask],w[mask]])*size:
            d.ellipse((x-1.4,y-1.4,x+1.4,y+1.4),fill=color)
    im.save(path)
    return dict(path=str(path),sha256=assets.digest(path),kind=kind,points=len(points),counts=counts,
        class_draw_order=[c[0] for c in classes],classification_masks_may_overlap=True,
        display='Exact saved sample-point projection; x-ray markers do not use depth testing; unchanged opaque mesh background',
        background=str(background),background_sha256=assets.digest(background),camera=camera,
        reference=str(source['reference']),reference_sha256=assets.digest(source['reference']),
        vectors=str(source['vectors']),vectors_sha256=assets.digest(source['vectors']),
        joined=str(source['joined']),joined_sha256=assets.digest(source['joined']),
        source_point_sha256=assets.array_hash(points),new_geometric_queries=0,metric_recomputed=False)


def generate(out=assets.OUT,labels=('START',),split='DEV',panel_size=400):
    if split!='DEV':raise ValueError('This visualization entry point refuses CONFIRM and any non-DEV pool')
    out=assets.resolve(out);labels=list(dict.fromkeys(labels))
    if not labels or any('/' in x or '\\' in x or x in ('.','..') for x in labels):raise ValueError('Invalid labels')
    if not 240<=panel_size<=900:raise ValueError('Registered readable panel size range 240..900')
    started=time.perf_counter();manifest_path=out/'manifests/DEV.json'
    requests=assets.read_json(manifest_path)['rows']
    assert len(requests)==16 and len({r['uid'] for r in requests})==8
    sources={str(manifest_path):assets.digest(manifest_path),str(RENDERER):assets.digest(RENDERER),str(Path(__file__)):assets.digest(__file__)}
    data={};available={};metric_sources={};conditions={}
    for request in requests:
        cid=request['condition_id'];p=Path(request['condition_npz'])
        assert assets.digest(p)==request['condition_sha256']
        sources[str(p)]=request['condition_sha256'];conditions[cid]=assets.arrays(p)
        C=conditions[cid]['C'];n,k=request['N'],request['K']
        for label in labels:
            key=(request['id'],label);p=_raw(out,label,request);available[key]=str(p) if p else None
            if p is not None:
                a=assets.arrays(p);raw=a.get('output',a.get('raw')).reshape(-1,3,3)
                assert raw.dtype==np.float32 and raw.shape==(n,3,3) and np.isfinite(raw).all()
                assert assets.bits(raw[:k],C),'C mismatch cannot be hidden in a gallery'
                data[key]=raw;sources[str(p)]=assets.digest(p)
            metric_sources[key]=_metric_source(out,label,request)
            if metric_sources[key]:
                for f in ('vectors','reference','joined'):
                    p=metric_sources[key][f];sources[str(p)]=assets.digest(p)
    identity=dict(version=VERSION,labels=labels,split=split,panel_size=panel_size,source_files=sources,
                  availability={k[0]+'|'+k[1]:v for k,v in available.items()},views=VIEWS)
    set_id='DEV_'+'_'.join(labels)+'_'+_hash(identity)[:12]
    dest=out/'visuals'/set_id;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists():
        saved=assets.read_json(dest/'manifest.json')
        assert saved['identity']==identity
        return saved
    panels=dest/'panels';panels.mkdir(exist_ok=True)
    r=_renderer();cache={};receipts=[];layouts=[];overlays=[];cameras={};byparent=defaultdict(list);contacts=[]
    for request in requests:byparent[request['uid']].append(request)
    empty=np.empty((0,3,3),np.float32)
    def panel(C,F,camera,wire=False):
        key=_hash(dict(C=assets.array_hash(C),F=assets.array_hash(F),camera=camera,wire=wire,size=panel_size))
        if key not in cache:
            p=panels/(key+'.png');j=p.with_suffix('.json')
            if j.exists():
                rec=assets.read_json(j);assert assets.digest(p)==rec['png_sha256'] and rec['camera']==camera
            else:
                rec=_render(C,F,p,camera,panel_size,wire);assets.freeze_json(j,rec)
            cache[key]=p;receipts.append(dict(path=str(j),sha256=assets.digest(j),render=rec))
        return cache[key]
    absent=panels/'not_ready.png'
    if not absent.exists():_missing(absent,panel_size)
    try:
        for uid,rows in byparent.items():
            rows=sorted(rows,key=lambda x:x['candidate']);condition=conditions[rows[0]['condition_id']];C=condition['C'];GT=condition['GT_free']
            union=[C,GT]+[data[(q['id'],label)] for q in rows for label in labels if (q['id'],label) in data]
            camera={name:r.frame_for(union,az,el,1.) for name,az,el in VIEWS};cameras[uid]=camera
            table=[];names=[];parent_overlays=[];overlay_names=[]
            for request in rows:
                main=[]
                for view,_,_ in VIEWS:
                    cam=camera[view];row=[panel(C,empty,cam,view=='wire')]
                    for label in labels:
                        raw=data.get((request['id'],label))
                        row.append(panel(C,raw[len(C):],cam,view=='wire') if raw is not None else absent)
                    row.append(panel(C,GT,cam,view=='wire'));table.append(row);names.append('noise '+str(request['candidate'])+'\n'+view)
                    if view=='front_3q':main=row
                contacts.append(dict(uid=uid,candidate=request['candidate'],paths=[str(p) for p in main]))
                for kind in ('S','I','E'):
                    o=[]
                    for li,label in enumerate(labels):
                        source=metric_sources[(request['id'],label)]
                        if source is None or (request['id'],label) not in data:o.append(absent);continue
                        p=panels/(request['id']+'__'+label+'__'+kind+'.png');j=p.with_suffix('.json')
                        if j.exists():
                            receipt=assets.read_json(j);assert assets.digest(p)==receipt['sha256']
                        else:
                            receipt=_overlay(main[li+1],p,camera['front_3q'],source,kind,panel_size);assets.freeze_json(j,receipt)
                        overlays.append(receipt);o.append(p)
                    parent_overlays.append(o);overlay_names.append('noise '+str(request['candidate'])+'\n'+kind)
            title='DEV '+uid+' | RAW comparison'
            layout=_layout(dest/(uid+'.png'),title,['Input C']+labels+['GT: C + free'],table,names,panel_size,
                'Two fixed noises; four orthographic views + wireframe. Same per-parent cameras and scale for every column/noise.')
            layouts.append(layout)
            if any(metric_sources[(q['id'],label)] for q in rows for label in labels):
                layouts.append(_layout(dest/(uid+'_feedback.png'),'DEV '+uid+' | saved feedback points',labels,
                    parent_overlays,overlay_names,panel_size,
                    'S green=covered/red=miss; I red=cross/purple=duplicate; E green=retained/red=occluded; yellow/purple=unknown. X-ray markers.'))
    finally:r.close_renderer()
    # Contact sheet uses all 16 fixed requests, uniformly downsampled, never
    # selected by scores. Per-parent pages retain full panel resolution.
    thumb=190;gap=6;block_cols=len(labels)+2;blockw=block_cols*(thumb+gap)+12;rowh=thumb+42
    sheet=Image.new('RGB',(2*blockw,110+8*rowh),'white');d=ImageDraw.Draw(sheet)
    d.text((14,12),'DEV all 8 parents x 2 fixed noises',font=_font(26,True),fill='#202833')
    d.text((14,50),' | '.join(['Input C']+labels+['GT C+free'])+'   - full-resolution pages in index.html',font=_font(17),fill='#586574')
    for i,item in enumerate(contacts):
        group=i//8;local=i%8;x0=group*blockw;y0=110+local*rowh
        d.text((x0+5,y0),item['uid'][:12]+'  noise '+str(item['candidate']),font=_font(16,True),fill='#344354')
        for j,p in enumerate(item['paths']):
            with Image.open(p) as image:sheet.paste(image.resize((thumb,thumb),Image.Resampling.LANCZOS),(x0+j*(thumb+gap),y0+29))
    cp=dest/'contact_sheet.png';sheet.save(cp,dpi=(150,150))
    body=['<!doctype html><meta charset="utf-8"><title>DEV raw gallery</title>',
        '<style>body{font:16px sans-serif;margin:24px;color:#223}img{max-width:100%;height:auto}a{color:#167}</style>',
        '<h1>DEV raw meshes and frozen feedback samples</h1>',
        '<p>AI_VISUAL_REVIEW: PENDING. Human study: NOT_RUN. Blue is exact C; orange is free, including GT free. No mesh edits.</p>',
        '<p>All eight parents and both registered noises are shown. Four views and wireframe share each parent camera across methods and noises. Feedback markers project saved points through the same camera with x-ray visibility; they do not recompute visibility or metrics.</p>',
        '<img src="contact_sheet.png">']
    for item in layouts:
        name=Path(item['path']).name;body.append('<p><a href="'+html.escape(name)+'">'+html.escape(name)+'</a></p>')
    (dest/'index.html').write_text('\n'.join(body),encoding='utf-8')
    assert all(assets.digest(p)==sha for p,sha in sources.items()),'A source changed during rendering'
    status='COMPLETE' if all(available.values()) else 'PARTIAL'
    result=dict(version=VERSION,status=status,identity=identity,labels=labels,split=split,requests=16,parents=8,
        AI_VISUAL_REVIEW='PENDING',human_usability='NOT_RUN',source_files=sources,cameras=cameras,
        contact_sheet=str(cp),contact_sheet_sha256=assets.digest(cp),index=str(dest/'index.html'),
        parent_pages=layouts,render_receipts=receipts,overlay_receipts=overlays,
        unique_cached_mesh_panels=len(receipts),overlay_panels=len(overlays),expected_overlay_panels=16*len(labels)*3,
        feedback_overlay_status='COMPLETE' if len(overlays)==16*len(labels)*3 else 'PENDING_SCORES',
        mesh_render_seconds=sum(x['render']['seconds']['total'] for x in receipts),wall_seconds=time.perf_counter()-started,
        all_geometry_unchanged=all(x['render']['input_arrays_unchanged'] for x in receipts),
        all_vertices_inside=all(x['render']['all_vertices_inside'] for x in receipts),
        per_parent_camera_shared=True,model_calls=0,new_metric_calls=0,geometry_repair_calls=0,
        software_GL=True,CPU_threads=2,GL_workers=1)
    assets.freeze_json(dest/'manifest.json',result)
    _write(out/'visuals/latest.json',dict(manifest=str(dest/'manifest.json'),sha256=assets.digest(dest/'manifest.json'),
        status=status,AI_VISUAL_REVIEW='PENDING',contact_sheet=str(cp),index=str(dest/'index.html')))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(assets.OUT));p.add_argument('--labels',nargs='+',default=['START'])
    p.add_argument('--split',choices=['DEV'],default='DEV');p.add_argument('--panel-size',type=int,default=400)
    a=p.parse_args();result=generate(a.out,a.labels,a.split,a.panel_size)
    print(json.dumps({k:result[k] for k in ('status','contact_sheet','index','unique_cached_mesh_panels','overlay_panels','wall_seconds')}))
