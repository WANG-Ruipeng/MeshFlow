"""CPU scientific render of every fixed capacity intervention, no model calls."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
import numpy as np
from r2.pilots.seam_slot_c200_v1 import common as C
from r2.pilots.regular_patch_control_v1 import depth_gallery as D
from r2.tools.cref_layer_audit_v1 import render as H

OUT=C.ROOT/'experiments/r2_seam_geometry_guard_v1'

def run():
    dest=OUT/'gallery'; dest.mkdir(exist_ok=True)
    done=dest/'completion.json'
    if done.exists():
        r=C.read(done)
        for p,h in r['files'].items():assert C.sha(p)==h,p
        return r
    assert C.read(OUT/'capacity/completion.json')['requests']==16
    m=C.read(C.OUT/'eval_manifest.json'); rows=[];plt=H._mpl()
    Bs=[D.basis(-60,23),D.basis(120,23)]
    for cond in m['conditions']:
        qq=[q for q in m['requests'] if q['condition_id']==cond['condition_id']]; meshes={};paths={}
        for q in qq:
            for label,base in (('BASELINE',C.OUT),('CAPACITY',OUT/'capacity')):
                p=base/'evaluation/SEAM_SLOT_C200'/q['request_id']/'post/post.npz'; meshes[q['request_id'],label]=C.arrays(p)['output'];paths[str(p)]=C.sha(p)
        points=np.concatenate([x.reshape(-1,3) for x in meshes.values()]);center=(points.min(0)+points.max(0))/2
        span=max(np.ptp((points-center)@b,axis=0)[:2].max() for b in Bs)
        fig,axes=plt.subplots(4,4,figsize=(13,13),dpi=120)
        for i,q in enumerate(qq):
            for j,(label,view) in enumerate((('BASELINE',0),('CAPACITY',0),('BASELINE',1),('CAPACITY',1))):
                tri=meshes[q['request_id'],label]
                image,owner=D.raster(tri,q['K'],Bs[view],center,span,size=340)
                axes[i,j].imshow(image);axes[i,j].set_axis_off();axes[i,j].set_title(f'n{q["noise_index"]} {label} | view{view+1}',fontsize=9)
        fig.suptitle(cond['parent_id'][:8]+' | frozen RAW + single D0 | original vs capacity GEO\nBlue: original C; gray: free. Same scale; exact contact cannot be certified from images.',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.95]);p=dest/(cond['parent_id'][:8]+'.png')
        if p.exists():raise RuntimeError('Unreceipted figure')
        fig.savefig(p);plt.close(fig)
        rows.append(dict(parent_id=cond['parent_id'],requests=[q['request_id'] for q in qq],image=str(p),sha256=C.sha(p),mesh_hashes=paths,review_status='PENDING_ACTUAL_AI_PROXY_VIEW'))
    html='<meta charset="utf-8"><h1>Capacity guard: all sixteen paired POST</h1><p>Scientific CPU render; no new model call. Same scale within each C, two views.</p>'
    for r in rows:html+=f'<h2>{r["parent_id"]}</h2><img style="width:100%" src="{Path(r["image"]).name}">'
    (dest/'index.html').write_text(html,encoding='utf-8')
    C.write(dest/'completion.json',dict(status='RENDERED_PENDING_REVIEW',rows=rows,files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()},model_calls=0))
    return dict(status='RENDERED',parents=4,requests=16)

if __name__=='__main__':print(json.dumps(run()))
