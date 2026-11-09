"""Fixed cached geometry contact sheets; no model, repair, or automatic ratings."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import io,json
from pathlib import Path
import numpy as np
from . import run as R
from r2.pilots.regular_patch_control_v1 import depth_gallery as D
from r2.tools.cref_layer_audit_v1 import render as H
from r2.pilots.seam_slot_c200_v1 import common as C


def run():
    reg=R.registration();qs=reg['requests'];dest=R.OUT/'gallery';dest.mkdir(exist_ok=True);plt=H._mpl();pages=[];images=[]
    labels=('RAW','UNCONSTRAINED','RIGID','HARD_RESTORE','POST')
    data={}
    for q in qs:
        p=R.paths(q);rid=q['request_id'];raw=C.arrays(p['raw']);prop=C.arrays(p['proposal'])['official_unconstrained_proposal'];hard=C.arrays(p['hard'])['output']
        rigid=C.arrays(R.OUT/'outputs'/rid/'rigid.npz')['output'];post=C.arrays(p['post'])['output']
        data[rid]=dict(zip(labels,(raw['output'],prop,rigid,hard,post)))
    for cond in reg['conditions']:
        qq=sorted([q for q in qs if q['condition_id']==cond['condition_id']],key=lambda q:q['noise_index'])
        points=np.concatenate([data[q['request_id']][s].reshape(-1,3) for q in qq for s in labels]);center=(points.min(0)+points.max(0))/2
        bases=[D.basis(-60,23),D.basis(120,23)];span=max(np.ptp((points-center)@b,axis=0)[:2].max() for b in bases)
        for vi,b in enumerate(bases):
            path=dest/(cond['parent_id'][:8]+'_view'+str(vi+1)+'.png')
            if not path.exists():
                fig,axes=plt.subplots(4,5,figsize=(15,11.5),dpi=110)
                for row,q in enumerate(qq):
                    for col,s in enumerate(labels):
                        im,_=D.raster(data[q['request_id']][s],q['K'],b,center,float(span),size=340)
                        axes[row,col].imshow(im);axes[row,col].set_axis_off();axes[row,col].set_title('n'+str(q['noise_index'])+' '+s,fontsize=8)
                fig.suptitle(cond['parent_id'][:8]+' | cached D0 and restoration | view '+str(vi+1),fontsize=12)
                fig.text(.5,.015,'Blue = actual C-side triangles in each stage (UNCONSTRAINED/RIGID do NOT preserve user C). Gray = free. No new GEO. Same scale.',ha='center',fontsize=9)
                fig.tight_layout(rect=[0,.035,1,.96]);buf=io.BytesIO();fig.savefig(buf,format='png');plt.close(fig)
                with path.open('xb') as f:f.write(buf.getvalue())
            images.append(dict(parent_id=cond['parent_id'],view=vi+1,path=str(path),sha256=C.sha(path)))
            pages.append('<figure><img style="width:100%;max-width:1650px" src="'+path.name+'"><figcaption>'+cond['parent_id']+' view'+str(vi+1)+'</figcaption></figure>')
    html='<!doctype html><meta charset="utf-8"><title>D0 restoration diagnostic</title><h1>Actual cached geometry; no extra model calls</h1><p>Blue C is moved in UNCONSTRAINED and RIGID. Display cannot certify exact seams. AI_PROXY review recorded separately.</p>'+''.join(pages)
    path=dest/'index.html'
    if path.exists() and path.read_text()!=html:raise ValueError('Gallery changed')
    if not path.exists():path.write_text(html,encoding='utf8')
    result=dict(status='RENDERED_NOT_REVIEWED',images=images,model_calls=0,new_GEO=0,source_hashes={str(p):C.sha(p) for p in (Path(__file__),Path(D.__file__),Path(H.__file__))})
    C.write(dest/'manifest.json',result);print(json.dumps(result,indent=2));return result

if __name__=='__main__':run()
