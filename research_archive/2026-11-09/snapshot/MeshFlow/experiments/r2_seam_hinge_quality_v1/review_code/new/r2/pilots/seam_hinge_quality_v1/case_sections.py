"""Fixed spatial windows across methods; never pretend a slot is semantic GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evaluation import C,ROOT,OUT,ARMS,requests,mesh_path
from r2.tools.seam_s2000_section_audit_v1.render import finite_sections

CASES=(('608b46b0_n3','front',(-1.35,-.6),(-.16,.075)),
       ('3f741759_n2','side',(-1.4,1.4),(-.7,1.02)))

def run(hinge=False):
    dest=OUT/'sections/fixed_cases';dest.mkdir(parents=True,exist_ok=True)
    protocol=C.read(ROOT/'experiments/r2_seam_s2000_section_audit_v1/protocol.json');qs={q['request_id']:q for q in requests()}
    method_labels=('PLAIN_S2000',*ARMS);stages=('POST','HINGE015','HINGE025') if hinge else ('RAW','D0','RESTORED','POST')
    images=[]
    for rid,view,xlim,ylim in CASES:
        suffix='_hinge' if hinge else '_stages';p=dest/(rid+suffix+'.png');rp=p.with_suffix('.json')
        if rp.exists():
            r=C.read(rp);assert C.sha(p)==r['image_sha256'];images.append(r);continue
        q=qs[rid];axis=0 if view=='side' else 2;project=[2,1] if view=='side' else [0,1]
        position=protocol['bounds'][q['condition_id']][view+'_positions'][1];sources={}
        fig,axes=plt.subplots(3,len(stages),figsize=(4*len(stages),10),dpi=150)
        for row,label in enumerate(method_labels):
            for col,stage in enumerate(stages):
                path=OUT/'hinge/cells'/(label+'_'+stage)/rid/'post.npz' if stage.startswith('HINGE') else mesh_path(label,rid,stage)
                sources[str(path)]=C.sha(path);a=C.arrays(path)
                tri=a['raw' if stage=='RAW' else 'official_unconstrained_proposal' if stage=='D0' else 'output']
                slots=a['source_face_ids'] if stage in ('POST','HINGE015','HINGE025') else np.arange(len(tri))
                seg,owners,_=finite_sections(tri,axis,position,1e-11);ax=axes[row,col]
                for s,owner in zip(seg,owners):
                    slot=int(slots[owner]);color='#1686c0' if slot<q['K'] else '#777777'
                    # Only historic-model identities retain the previous highlight.
                    highlight={77:'#c62828',113:'#e58b12'} if rid.startswith('608') else {21:'#c62828'}
                    if label=='PLAIN_S2000' and slot in highlight:color=highlight[slot]
                    pts=s[:,project];ax.plot(pts[:,0],pts[:,1],c=color,lw=2 if label=='PLAIN_S2000' and slot in highlight else 1.)
                ax.set(xlim=xlim,ylim=ylim);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
                ax.set_title(stage+(' (C moved)' if stage=='D0' else ''),fontsize=10)
                if col==0:ax.set_ylabel(label+'\nY model units',fontsize=9)
                ax.set_xlabel(('Z' if view=='side' else 'X')+' model units',fontsize=9);ax.tick_params(labelsize=8)
        fig.suptitle(rid+' | '+view+f' exact plane = {position:.9f} | identical historic spatial window',fontsize=13)
        fig.text(.5,.045,'Blue C / gray free. Old baseline tracked slots highlighted only on baseline: no assumed semantic correspondence across trained models.',ha='center',fontsize=9)
        fig.text(.5,.02,'This is a declared fixed local crop; same plane/axes for every cell. Visible gaps alone do not certify whole-mesh disconnection.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.07,1,.95))
        with p.open('xb') as f:fig.savefig(f,format='png')
        plt.close(fig)
        r=dict(status='PENDING_ACTUAL_REVIEW',request_id=rid,view=view,plane=position,xlim=xlim,ylim=ylim,labels=method_labels,stages=stages,
            path=str(p),image_sha256=C.sha(p),sources=sources,code_sha256=C.sha(__file__),model_calls=0)
        C.write(rp,r);images.append(r)
    return images

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--hinge',action='store_true');a=p.parse_args();r=run(a.hinge);print([x['path'] for x in r],flush=True)
