"""Two explicitly selected CPU inspection figures; no model or new scores."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import io
from pathlib import Path
import numpy as np
from r2.pilots.regular_patch_control_v1 import depth_gallery as D
from r2.tools.cref_layer_audit_v1 import render as H
from r2.pilots.seam_slot_c200_v1 import common as C
from .run import OLD,CAP,OUT,ARM


def main():
    dest=OUT/'gallery';dest.mkdir(exist_ok=True)
    selection={'73b369b9_n3':'ONLY_CHANGED_REQUEST','3f741759_n2':'PREVIOUS_I_TAIL_UNCHANGED_CONTROL'}
    requests={q['request_id']:q for q in C.read(OLD/'eval_manifest.json')['requests']}
    plt=H._mpl();rows=[]
    for rid,why in selection.items():
        q=requests[rid];meshes=[];refs={}
        for label,root in (('ORIGINAL_GEO',OLD),('CAPACITY',CAP),('TIP_REDIRECT',OUT)):
            p=root/'evaluation'/ARM/rid/'post/post.npz';a=C.arrays(p);refs[str(p)]=C.sha(p)
            tri=a['output'];meshes.append((label,dict(tri=tri,C=tri[:q['K']],F=tri[q['K']:],C_original=tri[:q['K']])))
        bounds=H.bounds_for_condition([g for _,g in meshes],None,C.L)
        pts=np.concatenate([g['tri'].reshape(-1,3) for _,g in meshes]);center=(pts.min(0)+pts.max(0))/2
        views=[D.basis(-60),D.basis(120)];span=max(np.ptp((pts-center)@v,axis=0)[:2].max() for v in views)
        fig=plt.figure(figsize=(12,14),dpi=125,facecolor='white')
        grid=fig.add_gridspec(4,3,left=.035,right=.98,top=.94,bottom=.06,hspace=.28,wspace=.18)
        for col,(label,g) in enumerate(meshes):
            for row,view in enumerate(views):
                im,_=D.raster(g['tri'],q['K'],view,center,span,size=440)
                ax=fig.add_subplot(grid[row,col]);ax.imshow(im);ax.axis('off');ax.set_title(label+(' / -60' if row==0 else ' / +120'),fontsize=9)
            for row,view in ((2,'front'),(3,'side')):
                ax=fig.add_subplot(grid[row,col]);H.draw_section(ax,g,[],bounds,C.L,view,1);ax.set_title(label+' / '+view,fontsize=8)
        fig.suptitle(rid+' | '+why+'\nBlue C; gray free. Same scale/cuts; nonblind CPU comparison.',fontsize=12)
        fig.text(.5,.025,'Whole views and two finite central cuts may miss small/off-plane changes; no visual claim of watertightness.',ha='center',fontsize=8)
        path=dest/(rid+'.png');buf=io.BytesIO();fig.savefig(buf,format='png');plt.close(fig)
        if path.exists():raise ValueError('Refuse overwrite figure')
        path.write_bytes(buf.getvalue())
        rows.append(dict(request_id=rid,selection_reason=why,image_path=str(path),image_sha256=C.sha(path),source_hashes=refs,status='RENDERED_NOT_REVIEWED'))
    C.write(dest/'manifest.json',dict(model_calls=0,scoring_calls=0,new_geometry=0,rows=rows))
    (dest/'index.html').write_text('<meta charset="utf-8"><h1>Seam tip redirection: two fixed inspection cases</h1>'+''.join('<h2>'+r['request_id']+'</h2><img style="width:100%" src="'+Path(r['image_path']).name+'">' for r in rows),encoding='utf-8')
    print('RENDERED',len(rows),flush=True)


if __name__=='__main__':main()
