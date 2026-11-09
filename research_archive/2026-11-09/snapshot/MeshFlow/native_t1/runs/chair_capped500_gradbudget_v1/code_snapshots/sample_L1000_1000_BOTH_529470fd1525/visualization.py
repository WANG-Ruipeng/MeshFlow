"""Full OLD/FRESH CPU mesh galleries; no scoring, mesh edits, or generation.

Every gallery camera encloses all currently registered available models and all
four noises for its parent. Final camera status requires all 288 raw outputs.
Earlier gallery snapshots remain distinct and are explicitly provisional.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import time
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LP_NUM_THREADS'):
    os.environ.setdefault(_key,'2')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
os.environ.setdefault('GALLIUM_DRIVER','llvmpipe')
import numpy as np
from PIL import Image,ImageDraw
from . import evaluation as ev
from experiments.mf_h1_local0.chair_control_feedback_auto_v1 import visualization as renderer

VERSION='chair_capped500_gradbudget_gallery_v1'
VIEWS=renderer.VIEWS


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _raw(out,label,request):
    result_path,result,path=ev.result_for(out,label,request)
    if result and result.get('status') in ('PASS','DONE','SAMPLED','CACHED'):
        if path.exists():
            if ev.digest(path)!=result.get('raw_sha256'):raise ValueError('RAW/result hash mismatch')
            return path
    pointer=request.get('historical_cache',{}).get(label) if request['split']=='OLD' else None
    if pointer:
        path=ev.resolve(pointer['raw_path'])
        if ev.digest(path)!=pointer['raw_sha256']:raise ValueError('Historical mesh hash mismatch')
        return path
    return None


def _geometry(path,condition):
    a=ev.arrays(path);value=a.get('output',a.get('raw'));n,k=int(condition['N']),int(condition['K'])
    if value is None:raise ValueError('RAW missing output')
    value=value.reshape(-1,3,3)
    if value.shape!=(n,3,3) or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('Invalid raw cannot be hidden in renderer')
    if not ev.old._same(value[:k],condition['C']):raise ValueError('Known C changed')
    return value


def prepare_cameras(out=ev.OUT):
    """No renderer context is initialized until actual mesh rendering."""
    out=ev.assert_out(out);requests=ev.manifest(out,'ALL');data={};sources={};available=0
    for q in requests:
        cid=q['condition_id'];path=ev.resolve(q['condition_npz'])
        if ev.digest(path)!=q['condition_sha256']:raise ValueError('Condition changed')
        condition=ev.arrays(path);sources[str(path)]=q['condition_sha256']
        parent=data.setdefault(q['uid'],[condition['C'],condition['GT_free']])
        for label in ev.LABELS:
            path=_raw(out,label,q)
            if path is not None:
                parent.append(_geometry(path,condition));sources[str(path)]=ev.digest(path);available+=1
    r=renderer._renderer()
    cameras={uid:{name:r.frame_for(values,az,el,1.) for name,az,el in VIEWS} for uid,values in data.items()}
    record=dict(version=VERSION,scope='All nine registered labels and all four OLD/FRESH noises; per-parent common frame',
        status='FINAL_COMPLETE' if available==288 else 'PROVISIONAL_AVAILABLE_GEOMETRY',available_outputs=available,expected_outputs=288,
        cameras=cameras,source_files=sources,renderer_sha256=ev.digest(renderer.RENDERER),views=VIEWS)
    name='cameras_'+_hash(record)[:16]+'.json';path=out/'visuals/cameras'/name
    ev.freeze_json(path,record)
    return record,path


def _metric_source(out,label,q):
    joined=Path(out)/'evaluation'/label/q['split']/'scoring/joined'/(q['request_id']+'.json')
    if not joined.exists():return None
    row=ev.read_json(joined)['metrics'];p=row.get('control_vectors_path')
    if not p:return None
    vectors=ev.resolve(p)
    if ev.digest(vectors)!=row['control_vectors_sha256']:raise ValueError('Control vectors changed')
    ref=Path(out)/'references'/q['condition_id']/'control.npz'
    if not ref.exists():return None
    return dict(vectors=vectors,reference=ref,joined=joined,metrics=row)


def generate(out=ev.OUT,labels=ev.LABELS,panel='OLD',panel_size=320):
    out=ev.assert_out(out);labels=list(dict.fromkeys(labels))
    if not labels or any(label not in ev.LABELS for label in labels):raise ValueError('Unknown gallery label')
    if panel not in ev.PANELS or not 240<=panel_size<=600:raise ValueError('Invalid panel/size')
    tick=time.perf_counter();requests=ev.manifest(out,panel);camera_manifest,camera_path=prepare_cameras(out)
    sources={str(camera_path):ev.digest(camera_path),str(Path(__file__)):ev.digest(__file__),str(renderer.RENDERER):ev.digest(renderer.RENDERER)}
    data={};conditions={};metrics={};availability={};byparent=defaultdict(list)
    for q in requests:
        cid=q['condition_id'];cp=ev.resolve(q['condition_npz']);conditions[cid]=ev.arrays(cp);sources[str(cp)]=q['condition_sha256'];byparent[q['uid']].append(q)
        for label in labels:
            key=(q['request_id'],label);path=_raw(out,label,q);availability['|'.join(key)]=str(path) if path else None
            if path:
                data[key]=_geometry(path,conditions[cid]);sources[str(path)]=ev.digest(path)
            metrics[key]=_metric_source(out,label,q)
            if metrics[key]:
                for field in ('vectors','reference','joined'):
                    p=metrics[key][field];sources[str(p)]=ev.digest(p)
    identity=dict(version=VERSION,panel=panel,labels=labels,panel_size=panel_size,camera_manifest=str(camera_path),
        camera_sha256=ev.digest(camera_path),source_files=sources,availability=availability,views=VIEWS)
    dest=out/'visuals'/f'{panel}_{"_".join(labels)}_{_hash(identity)[:12]}';dest.mkdir(parents=True,exist_ok=True)
    manifest_path=dest/'manifest.json'
    if manifest_path.exists():
        saved=ev.read_json(manifest_path)
        if saved['identity']!=json.loads(json.dumps(identity)):raise ValueError('Completed gallery identity changed')
        for row in saved['parent_pages']:
            if ev.digest(ev.resolve(row['path']))!=row['sha256']:raise ValueError('Gallery image changed')
        return saved
    panels=dest/'panels';panels.mkdir(exist_ok=True);empty=np.empty((0,3,3),np.float32)
    missing=panels/'not_ready.png'
    if not missing.exists():renderer._missing(missing,panel_size)
    mesh_cache={};receipts=[];layouts=[];overlays=[];contacts=[]
    def draw(C,F,camera,wire=False):
        key=_hash(dict(C=ev.old.assets.array_hash(C),F=ev.old.assets.array_hash(F),camera=camera,wire=wire,size=panel_size))
        if key not in mesh_cache:
            path=panels/(key+'.png');meta=path.with_suffix('.json')
            if meta.exists():
                value=ev.read_json(meta)
                if ev.digest(path)!=value['png_sha256']:raise ValueError('Panel image changed')
            else:
                value=renderer._render(C,F,path,camera,panel_size,wire);ev.freeze_json(meta,value)
            mesh_cache[key]=path;receipts.append(dict(path=str(meta),sha256=ev.digest(meta),render=value))
        return mesh_cache[key]
    try:
        for uid,group in byparent.items():
            group=sorted(group,key=lambda q:q['candidate']);c=conditions[group[0]['condition_id']];C,GT=c['C'],c['GT_free']
            camera=camera_manifest['cameras'][uid];table=[];names=[];otable=[];onames=[]
            for q in group:
                main=None
                for view,_,_ in VIEWS:
                    row=[draw(C,empty,camera[view],view=='wire')]
                    for label in labels:
                        raw=data.get((q['request_id'],label))
                        row.append(draw(C,raw[len(C):],camera[view],view=='wire') if raw is not None else missing)
                    row.append(draw(C,GT,camera[view],view=='wire'));table.append(row);names.append(f'n{q["candidate"]}\n{view}')
                    if view=='front_3q':main=row
                contacts.append(dict(uid=uid,candidate=q['candidate'],paths=[str(p) for p in main]))
                for kind in ('S','I','E'):
                    row=[]
                    for index,label in enumerate(labels):
                        source=metrics[(q['request_id'],label)]
                        if source is None or (q['request_id'],label) not in data:row.append(missing);continue
                        path=panels/(q['request_id']+'__'+label+'__'+kind+'.png');meta=path.with_suffix('.json')
                        if meta.exists():
                            value=ev.read_json(meta)
                            if ev.digest(path)!=value['sha256']:raise ValueError('Overlay changed')
                        else:
                            value=renderer._overlay(main[index+1],path,camera['front_3q'],source,kind,panel_size);ev.freeze_json(meta,value)
                        overlays.append(value);row.append(path)
                    otable.append(row);onames.append(f'n{q["candidate"]}\n{kind}')
            layouts.append(renderer._layout(dest/(uid+'.png'),panel+' '+uid+' | RAW comparison',
                ['Input C']+labels+['GT C+free'],table,names,panel_size,
                'Four views + wire; one parent frame across all registered OLD/FRESH methods/noises. '+camera_manifest['status']))
            if any(metrics[(q['request_id'],label)] for q in group for label in labels):
                layouts.append(renderer._layout(dest/(uid+'_feedback.png'),panel+' '+uid+' | existing S/I/E samples',labels,otable,onames,panel_size,
                    'S green=covered/red=miss; I red=cross/purple=dup; E green=retained/red=occluded; uncertainty yellow/purple. X-ray points.'))
    finally:renderer._renderer().close_renderer()
    thumb=130;gap=6;cols=len(labels)+2;bw=cols*(thumb+gap)+12;rh=thumb+36
    sheet=Image.new('RGB',(2*bw,110+8*rh),'white');pen=ImageDraw.Draw(sheet)
    pen.text((12,10),panel+' | all 8 parents x 2 fixed noises',font=renderer._font(24,True),fill='#202833')
    pen.text((12,46),' | '.join(['C']+labels+['GT'])+' | full resolution in index.html',font=renderer._font(14),fill='#586574')
    for index,row in enumerate(contacts):
        x0=(index//8)*bw;y0=110+(index%8)*rh
        pen.text((x0+4,y0),row['uid'][:12]+' n'+str(row['candidate']),font=renderer._font(14,True),fill='#344354')
        for j,path in enumerate(row['paths']):
            with Image.open(path) as image:sheet.paste(image.resize((thumb,thumb),Image.Resampling.LANCZOS),(x0+j*(thumb+gap),y0+26))
    contact=dest/'contact_sheet.png';sheet.save(contact,dpi=(150,150))
    body=['<!doctype html><meta charset="utf-8"><title>Gradient budget meshes</title>',
        '<style>body{font:16px sans-serif;margin:24px;color:#223}img{max-width:100%;height:auto}a{color:#167}</style>',
        '<h1>'+panel+' full raw comparison</h1>',
        '<p>AI_VISUAL_REVIEW: PENDING; human usability NOT_RUN. Blue=exact C; orange=free, including GT. No geometry modifications.</p>',
        '<p>All eight parents and both registered noises. Common cameras also include other-panel and other-method available meshes. Camera status: '+camera_manifest['status']+'. X-ray overlays use saved classification arrays, no new scoring.</p>',
        '<img src="contact_sheet.png">']
    for row in layouts:
        name=Path(row['path']).name;body.append('<p><a href="'+html.escape(name)+'">'+html.escape(name)+'</a></p>')
    (dest/'index.html').write_text('\n'.join(body),encoding='utf-8')
    if not all(ev.digest(p)==sha for p,sha in sources.items()):raise ValueError('Source changed during rendering')
    record=dict(version=VERSION,status='COMPLETE' if all(availability.values()) else 'PARTIAL',identity=identity,
        labels=labels,panel=panel,requests=16,parents=8,request_ids=[q['request_id'] for q in requests],
        camera_status=camera_manifest['status'],camera_manifest=str(camera_path),contact_sheet=str(contact),contact_sheet_sha256=ev.digest(contact),
        index=str(dest/'index.html'),parent_pages=layouts,render_receipts=receipts,overlay_receipts=overlays,
        unique_mesh_panels=len(receipts),overlay_panels=len(overlays),expected_overlay_panels=16*len(labels)*3,
        AI_VISUAL_REVIEW='PENDING',human_usability='NOT_RUN',model_calls=0,new_metric_calls=0,geometry_repair_calls=0,
        all_vertices_inside=all(r['render']['all_vertices_inside'] for r in receipts),all_geometry_unchanged=all(r['render']['input_arrays_unchanged'] for r in receipts),
        wall_seconds=time.perf_counter()-tick,mesh_render_seconds=sum(r['render']['seconds']['total'] for r in receipts),
        software_GL=True,CPU_threads=2,GL_workers=1)
    ev.freeze_json(manifest_path,record)
    ev.write_json(out/'visuals'/('latest_'+panel+'.json'),dict(manifest=str(manifest_path),sha256=ev.digest(manifest_path),
        status=record['status'],camera_status=record['camera_status'],contact_sheet=str(contact),index=str(dest/'index.html')))
    return record


def run(out=ev.OUT,label=None,panel='OLD',*,labels=None,panel_size=320,workers=1):
    if workers!=1:raise ValueError('This renderer uses one measured software-GL worker; parallelize disjoint invocations externally if authorized')
    return generate(out,labels or ([label] if label else ev.LABELS),panel,panel_size)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ev.OUT));p.add_argument('--labels',nargs='+',choices=ev.LABELS,default=list(ev.LABELS))
    p.add_argument('--panel',choices=ev.PANELS,default='OLD');p.add_argument('--panel-size',type=int,default=320);a=p.parse_args()
    value=generate(a.out,a.labels,a.panel,a.panel_size)
    print(json.dumps({k:value[k] for k in ('status','camera_status','contact_sheet','index','wall_seconds')}))
