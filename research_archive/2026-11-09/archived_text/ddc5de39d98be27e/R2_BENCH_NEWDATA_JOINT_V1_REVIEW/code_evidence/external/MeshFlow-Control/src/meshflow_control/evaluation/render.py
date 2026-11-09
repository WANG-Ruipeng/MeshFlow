"""Uniform CPU mesh pages. Raw triangles are drawn without any repair."""
from pathlib import Path
import hashlib,time
import numpy as np
from .io import *
VIEWS=(('front_3q',20,-60),('back_3q',20,120),('front',0,-90),('side',0,0),('wire',20,-60))
P1_LABELS=('START','A500','B500','C500','A1000','B1000','C1000')
P2_LABELS=('START','A1000','B1000','C1000')
def _triangles(row):
    a=arrays(row['raw_path'])
    t=np.asarray(a.get('raw',a.get('output')),np.float32).reshape(-1,3,3)
    if row.get('raw_sha256') and digest(row['raw_path'])!=row['raw_sha256']:raise ValueError('Render RAW identity changed')
    return t
def _page(path,columns,bounds,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
    labels=[x[0] for x in columns];fig=plt.figure(figsize=(15,len(columns)*2.9),dpi=110,facecolor='white')
    lo,hi=np.asarray(bounds[0]),np.asarray(bounds[1]);center=(lo+hi)/2;span=max(float(np.max(hi-lo)),.01)*1.08
    for i,(label,C,F) in enumerate(columns):
        for j,(view,elev,azim) in enumerate(VIEWS):
            ax=fig.add_subplot(len(columns),len(VIEWS),i*len(VIEWS)+j+1,projection='3d')
            tri=np.concatenate([F,C]);colors=['#e7994c']*len(F)+['#3a82c5']*len(C)
            displayed=np.asarray(tri)[:,:,(0,2,1)]
            if view=='wire':
                edges=displayed[:,((0,1),(1,2),(2,0)),:].reshape(-1,2,3)
                artist=Line3DCollection(edges,colors=np.repeat(colors,3),linewidths=.38)
            else:
                artist=Poly3DCollection(displayed,facecolors=colors,edgecolors='#444444',linewidths=.10,alpha=1)
            ax.add_collection3d(artist)
            cc=center[(0,2,1),];ax.set_xlim(cc[0]-span/2,cc[0]+span/2);ax.set_ylim(cc[1]-span/2,cc[1]+span/2);ax.set_zlim(cc[2]-span/2,cc[2]+span/2)
            ax.set_box_aspect((1,1,1));ax.view_init(elev=elev,azim=azim);ax.set_axis_off()
            ax.set_title(label+(' / '+view),fontsize=9,pad=0)
    fig.suptitle(title,fontsize=13,y=.99);fig.subplots_adjust(left=0,right=1,top=1-.65/(len(columns)*2.9),bottom=0,wspace=0,hspace=.02)
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,facecolor='white');plt.close(fig)
def render(manifest,out,allow_partial=False):
    rows=load_manifest(manifest);out=resolve(out);out.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
    if not rows:raise ValueError('No output manifest')
    grouped={}
    for r in rows:grouped.setdefault(r['condition_id'],[]).append(r)
    pages=[];camera={};missing=[]
    for cid,rr in sorted(grouped.items()):
        panel=rr[0]['panel'];labels=P1_LABELS if panel=='P1' else P2_LABELS
        candidates=(0,1,4,5) if panel=='P1' else (0,1,2,3)
        by={(r['label'],r['candidate']):r for r in rr}
        need=[(l,n) for l in labels for n in candidates if (l,n) not in by]
        missing += [dict(condition_id=cid,label=l,candidate=n) for l,n in need]
        if need and not allow_partial:continue
        meta=read_json(rr[0]['reference_path']);c=arrays(meta['condition_path']);C=c['C']
        loaded={key:_triangles(r) for key,r in by.items()}
        bounds_geometry=[C]+list(loaded.values())+([c['GT_full']] if panel=='P1' else [])
        allpoints=np.concatenate([t.reshape(-1,3) for t in bounds_geometry]);bounds=(allpoints.min(0),allpoints.max(0))
        camera[cid]=dict(bounds=clean(bounds),views=VIEWS,geometry_scope='All fixed outputs plus source reference where applicable; no per-output centering/scaling/cropping',
            source_hashes={l+'__n'+str(n):digest(r['raw_path']) for (l,n),r in by.items()},status='FINAL_COMPLETE' if not need else 'SNAPSHOT')
        for n in candidates:
            selected=[(l,loaded[(l,n)][:rr[0]['K']],loaded[(l,n)][rr[0]['K']:]) for l in labels if (l,n) in loaded]
            if not selected:continue
            cols=[('Input C',C,np.empty((0,3,3),np.float32))]+selected
            p=out/'blind'/f'{cid}__n{n}.png'
            _page(p,cols,bounds,f'{panel} {cid} n{n} | C blue / free orange | source reference hidden')
            page=dict(condition_id=cid,panel=panel,candidate=n,blind_page=str(p),blind_sha256=digest(p),labels=[v[0] for v in selected],
                view_scope=[v[0] for v in VIEWS],AI_VISUAL_REVIEW='PENDING',human_usability='NOT_RUN')
            if panel=='P1':
                p=out/'reference'/f'{cid}__n{n}.png';_page(p,cols+[('Source reference',C,c['GT_free'])],bounds,f'{panel} {cid} n{n} | one source completion, not the only valid target')
                page.update(reference_page=str(p),reference_sha256=digest(p))
            pages.append(page)
    status='FINAL_COMPLETE' if not missing else 'PARTIAL'
    if missing and not allow_partial and not pages:status='BLOCKED_INCOMPLETE_FIXED_GALLERY'
    write_json(out/'camera_manifest.json',dict(status=status,conditions=camera))
    result=dict(status=status,pages=pages,missing=missing,source_manifest=str(manifest) if isinstance(manifest,(str,Path)) else 'provided dict',
        wall_seconds=time.perf_counter()-tick,workers=1,model_calls=0,mesh_modified=False,AI_VISUAL_REVIEW='PENDING',human_usability='NOT_RUN')
    write_json(out/'gallery_manifest.json',result)
    def link(p):return Path(p).relative_to(out).as_posix()
    lines=['<!doctype html><meta charset="utf-8"><title>Native context readout gallery</title>',
        '<h1>Complete raw mesh gallery</h1><p>C blue; free orange. Four views plus wire. No mesh repair. Rendering is not visual review.</p>',
        '<h2>1. Review without source reference</h2>']
    for p in pages:lines.append(f'<p><a href="{link(p["blind_page"])}">{p["condition_id"]} n{p["candidate"]}</a></p>')
    lines.append('<h2>2. Source reconstruction reference, P1 only</h2>')
    for p in pages:
        if 'reference_page' in p:lines.append(f'<p><a href="{link(p["reference_page"])}">{p["condition_id"]} n{p["candidate"]}</a></p>')
    (out/'INDEX.html').write_text('\n'.join(lines),encoding='utf-8')
    return result
