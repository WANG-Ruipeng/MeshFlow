"""Depth-correct CPU gallery plus fixed finite sections, without visual labels."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import html
import io
import json
import sys
from pathlib import Path
import numpy as np
from . import common as C
from . import evaluate as E
from r2.pilots.regular_patch_control_v1 import depth_gallery as D
from r2.tools.cref_layer_audit_v1 import render as H

VERSION='seam_slot_c200_depth_finite_sections_v1'
COLUMNS=(('FULL4_C200','RAW'),('SEAM_SLOT_C200','RAW'),('FULL4_C200','POST'),('SEAM_SLOT_C200','POST'))
REVIEW_FIELDS=('C_adoption','C_floating','C_replaced_by_other_surface','C_wrong_layer',
               'whole_body_complete','support_missing','fragments_or_folded_layers','visible_join')


def write_bytes(path,payload):
    path=C.resolve(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=payload:raise ValueError('Refuse change frozen figure '+str(path))
    else:
        with path.open('xb') as f:f.write(payload)


def run(root):
    root=C.resolve(root);m=E.manifest(root);dest=root/'evaluation/gallery'
    definition=dict(version=VERSION,columns=[dict(label=a,stage=s) for a,s in COLUMNS],
        rows=['depth_correct_whole_minus60','depth_correct_whole_plus120','finite_front_central','finite_side_central'],
        same_physical_scale='Union of all four noises, arms and RAW/POST within each condition for whole views; C-only bounds for local sections.',
        sections='C-only bbox midpoint Z for front and X for side, original FP64 finite intersections eps=1e-12*L. No infinite plane extrapolation.',
        colors=dict(C='blue',free='gray',source_winding_normal='purple'),nonblind=True,
        limitations='Two central cuts can miss off-plane interfaces. Exact depth ties favor earlier C; sections draw C above free. Neither visual contact nor source winding certifies legal connectivity, outwardness, or adoption.',
        new_geometry=0,model_calls=0,scoring_calls=0,actual_reviews=0,source_GT_display=False)
    hashes={str(p):C.sha(p) for p in (Path(__file__),Path(D.__file__),Path(H.__file__))}
    refs={};cells={};views=[D.basis(-60,23),D.basis(120,23)]
    for q in m['requests']:
        inp=C.arrays(q['inference_npz']);original=inp['C_vertices'][inp['C_faces_original']]
        refs[q['inference_npz']]=C.sha(q['inference_npz'])
        for arm,stage in COLUMNS:
            p,rp=E.artifact(root,arm,q['request_id'],stage);a=C.arrays(p)
            tri=a['output'].reshape(-1,3,3)
            if not np.array_equal(tri[:q['K']],original):raise ValueError('C differs in figure')
            refs[str(p)]=C.sha(p);refs[str(rp)]=C.sha(rp)
            cells[q['request_id'],arm,stage]=dict(C=tri[:q['K']],F=tri[q['K']:],C_original=original,tri=tri,path=str(p),sha256=C.sha(p))
    bounds={};depth_bounds={}
    for cid in {q['condition_id'] for q in m['requests']}:
        gg=[cells[q['request_id'],arm,stage] for q in m['requests'] if q['condition_id']==cid for arm,stage in COLUMNS]
        bounds[cid]=H.bounds_for_condition(gg,None,C.L)
        points=np.concatenate([g['tri'].reshape(-1,3) for g in gg]);center=(points.min(0)+points.max(0))/2
        span=max(np.ptp((points-center)@B,axis=0)[:2].max() for B in views)
        depth_bounds[cid]=dict(center=center.tolist(),span=float(span))
    C.write(dest/'definition.json',definition);C.write(dest/'bounds.json',dict(sections=bounds,depth=depth_bounds))
    plt=H._mpl();images=[];blank=[]
    for q in m['requests']:
        path=dest/(q['request_id']+'.png');rp=path.with_suffix('.json')
        identity=dict(version=VERSION,request_id=q['request_id'],manifest_sha256=C.sha(root/'eval_manifest.json'),
            source_hashes=hashes,geometry=[dict(label=arm,stage=stage,path=cells[q['request_id'],arm,stage]['path'],
                sha256=cells[q['request_id'],arm,stage]['sha256']) for arm,stage in COLUMNS],
            bounds=bounds[q['condition_id']],depth_bounds=depth_bounds[q['condition_id']])
        if rp.exists():
            record=C.read(rp)
            if record['identity']!=identity or C.sha(path)!=record['image_sha256']:raise ValueError('Frozen figure changed')
        else:
            if path.exists():raise RuntimeError('Unreceipted image requires explicit recovery')
            fig=plt.figure(figsize=(16.8,16.4),dpi=145,facecolor='white')
            grid=fig.add_gridspec(4,4,left=.035,right=.988,bottom=.07,top=.94,wspace=.20,hspace=.30)
            visibility=[];db=depth_bounds[q['condition_id']]
            for column,(arm,stage) in enumerate(COLUMNS):
                g=cells[q['request_id'],arm,stage]
                for row,B in enumerate(views):
                    im,owner=D.raster(g['tri'],q['K'],B,np.array(db['center']),db['span'],size=480)
                    ax=fig.add_subplot(grid[row,column]);ax.imshow(im);ax.set_axis_off()
                    ax.set_title(arm+' | '+stage+(' | -60 deg' if row==0 else ' | +120 deg'),fontsize=9)
                    visibility.append(dict(label=arm,stage=stage,view=row,visible_C_pixels=int(((owner>=0)&(owner<q['K'])).sum()),visible_F_pixels=int((owner>=q['K']).sum())))
                for row,view in ((2,'front'),(3,'side')):
                    ax=fig.add_subplot(grid[row,column]);H.draw_section(ax,g,[],bounds[q['condition_id']],C.L,view,1)
                    ax.set_title(stage+' | '+ax.get_title(),fontsize=8)
            fig.suptitle(q['parent_id'][:8]+' | SEAT_FULL K='+str(q['K'])+' N='+str(q['N'])+' | fresh n'+str(q['noise_index'])+'\nBlue preserved C / gray free; nonblind paired geometry',fontsize=13,y=.979)
            fig.text(.5,.035,'Depth-correct whole views and fixed finite central cuts; same physical scale within C across all noises/arms/stages.',ha='center',fontsize=9)
            fig.text(.5,.020,'Two cuts can miss off-plane structures. Purple source winding is not certified outward. No automatic visual labels.',ha='center',fontsize=8)
            buf=io.BytesIO();fig.savefig(buf,format='png',dpi=145,facecolor='white');plt.close(fig);write_bytes(path,buf.getvalue())
            record=dict(status='RENDERED_NOT_REVIEWED',identity=identity,image_sha256=C.sha(path),visibility=visibility)
            C.write(rp,record)
        images.append(dict(request_id=q['request_id'],path=str(path),sha256=record['image_sha256'],receipt_sha256=C.sha(rp)))
        for arm,stage in COLUMNS:
            blank.append(dict(request_id=q['request_id'],condition_id=q['condition_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],
                label=arm,stage=stage,reviewer='',review_status='PENDING_ACTUAL_VIEW',**{f:'' for f in REVIEW_FIELDS},
                notes='',reviewed_image_path=str(path),reviewed_image_sha256=record['image_sha256']))
        print(json.dumps(dict(rendered=len(images),registered=16)),flush=True)
    C.csvwrite(dest/'review_template_NOT_REVIEWED.csv',blank)
    text='<meta charset="utf-8"><title>Seam slot paired endpoint gallery</title><h1>Four TRAIN C; fresh noise, two arms</h1><p>Images only. AI_PROXY judgments require actual review.</p>'
    text+=''.join('<h2>'+html.escape(x['request_id'])+'</h2><a href="'+Path(x['path']).name+'"><img loading="lazy" style="width:100%" src="'+Path(x['path']).name+'"></a>' for x in images)
    write_bytes(dest/'index.html',text.encode())
    E.verify(refs);E.verify(hashes)
    if 'torch' in sys.modules:raise RuntimeError('Renderer imported framework')
    result=dict(status='COMPLETE_RENDERING_PENDING_ACTUAL_REVIEW',images=images,pages=16,geometry_cells=64,
        definition=definition,source_hashes=hashes,input_hashes=refs,model_calls=0,actual_reviews=0)
    C.write(dest/'manifest.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(C.OUT));a=p.parse_args()
    r=run(a.root);print(json.dumps(dict(status=r['status'],pages=r['pages'])),flush=True)


if __name__=='__main__':main()
