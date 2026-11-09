"""CPU render all six requests with changed pre-GEO tips, fixed paired scale."""
from __future__ import annotations
from . import run as A
from r2.pilots.regular_patch_control_v1 import depth_gallery as D
from r2.tools.cref_layer_audit_v1 import render as H
from r2.pilots.seam_slot_c200_v1 import generate as G
from pathlib import Path
import numpy as np


def main():
    target=A.OUT/'gallery';target.mkdir(exist_ok=True)
    if (target/'render_receipt.json').exists():
        rec=A.C.read(target/'render_receipt.json')
        for p,h in rec['files'].items():assert A.C.sha(p)==h,p
        return
    requests=A.read(A.OLD/'eval_manifest.json')['requests'];changed=A.rows(A.OUT/'connectivity_pairing.csv')
    ids={r['request_id'] for r in changed if int(r['moved_corners'])>0};assert len(ids)==6
    plt=H._mpl();basis=[D.basis(-60,23),D.basis(120,23)];figures=[];identity=[]
    for q in requests:
        arrays=[]
        for root in (A.OLD,A.OUT):arrays.append(A.arrays(root/'evaluation'/A.ARM/q['request_id']/'post/post.npz'))
        identical=arrays[0].keys()==arrays[1].keys() and all(G.same(arrays[0][k],arrays[1][k]) for k in arrays[0])
        identity.append(dict(request_id=q['request_id'],pre_changed=q['request_id'] in ids,
                             all_POST_arrays_identical=identical,final_triangle_array_identical=G.same(arrays[0]['output'],arrays[1]['output'])))
        if q['request_id'] not in ids:continue
        tri=[a['output'] for a in arrays];points=np.concatenate([a.reshape(-1,3) for a in tri])
        center=(points.min(0)+points.max(0))/2;span=max(np.ptp((points-center)@b,axis=0)[:2].max() for b in basis)
        fig,axes=plt.subplots(2,2,figsize=(10,10),dpi=115)
        for view in range(2):
            for col,label in enumerate(('Original D0 / GEO','Reject D0 normal compression')):
                im,_=D.raster(tri[col],q['K'],basis[view],center,span,size=500)
                axes[view,col].imshow(im);axes[view,col].set_axis_off();axes[view,col].set_title(label+' | view '+str(view+1))
        fig.suptitle(q['request_id']+' | fixed C / occupancy / RAW / single D0 proposal\nBlue: preserved C; gray: free. Same scale; hidden overlap is not certified by views.',fontsize=11)
        fig.tight_layout(rect=[0,0,1,.95]);path=target/(q['request_id']+'.png')
        if path.exists():raise RuntimeError('Unreceipted figure '+str(path))
        fig.savefig(path);plt.close(fig)
        figures.append(dict(request_id=q['request_id'],image=str(path),sha256=A.C.sha(path)))
    A.C.csvwrite(A.OUT/'post_geometry_identity.csv',identity)
    html='<meta charset="utf-8"><h1>Anti-flatten: all six changed proposals</h1><p>Same physical scale within request; two views. No model calls.</p>'
    for item in figures:html+='<h2>'+item['request_id']+'</h2><img style="width:100%" src="'+Path(item['image']).name+'">'
    (target/'index.html').write_text(html,encoding='utf-8')
    A.C.write(target/'render_receipt.json',dict(status='RENDERED_PENDING_ACTUAL_REVIEW',rows=figures,
        identity=identity,model_calls=0,quality_scans=0,files={str(p):A.C.sha(p) for p in target.iterdir() if p.is_file()}))
    print('RENDERED',len(figures))


if __name__=='__main__':main()
