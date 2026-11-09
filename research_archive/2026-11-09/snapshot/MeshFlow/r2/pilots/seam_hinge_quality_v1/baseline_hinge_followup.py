"""CPU-only baseline hinge diagnostics and the two historic fixed section windows."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,time,resource
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evaluation import C,ROOT,OUT,requests,mesh_path,score_path
from . import topology as T
from .case_sections import CASES
from r2.tools.seam_s2000_section_audit_v1.render import finite_sections


def run():
    start=time.perf_counter();dest=OUT/'topology/baseline_hinge_followup_v2'
    if (dest/'completion.json').exists():
        old=C.read(dest/'completion.json')
        for p,h in old['files'].items():assert C.sha(p)==h
        return old
    C.resources(dest/'resources.json');qs={q['request_id']:q for q in requests()}
    records=T.registry(['PLAIN_S2000'],False)
    records=[r for r in records if r['stage']=='POST']
    records+=C.read(OUT/'hinge/aggregate/PLAIN_S2000/registry.json')['records']
    assert len(records)==84
    rows=[];faces=[];protected={};files={}
    for r in records:
        q=qs[r['request_id']];p=C.resolve(r['mesh_path']);ep=C.resolve(r['edges_path']);sp=C.resolve(r['score_path'])
        score=C.read(sp);receipt=score if 'files' in score else C.read(sp.parent/'complete.json')
        for f,h in receipt['files'].items():assert C.sha(f)==h
        for f in (p,ep,sp):protected[str(f)]=C.sha(f)
        m,ff=T.measure(C.arrays(p),q['K'],T.edges_file(ep),C.L)
        common=dict(label='PLAIN_S2000',stage=r['stage'],**{k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','split','K','N')})
        rows.append(dict(**common,**m));faces.extend(dict(**common,**f) for f in ff)
    for name,data in [('per_output',rows),('per_face_connectivity',faces)]:
        p=dest/(name+'.csv');C.csvwrite(p,data);files[str(p)]=C.sha(p)
    summary=[];metric_names=[k for k in rows[0] if k.startswith('hinge_diag_')]
    for stage in ('POST','POST_HINGE015','POST_HINGE025'):
        for split in ('TRAIN','DEV','ALL'):
            rr=[r for r in rows if r['stage']==stage and (split=='ALL' or r['split']==split)]
            summary.append(dict(stage=stage,split=split,n=len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in metric_names}))
    p=dest/'summary.csv';C.csvwrite(p,summary);files[str(p)]=C.sha(p)
    protocol=C.read(ROOT/'experiments/r2_seam_s2000_section_audit_v1/protocol.json')
    images=[]
    for rid,view,xlim,ylim in CASES:
        q=qs[rid];axis=0 if view=='side' else 2;project=[2,1] if view=='side' else [0,1]
        position=protocol['bounds'][q['condition_id']][view+'_positions'][1]
        fig,axes=plt.subplots(1,3,figsize=(15,5),dpi=160)
        for ax,stage in zip(axes,('POST','HINGE015','HINGE025')):
            p=mesh_path('PLAIN_S2000',rid,'POST') if stage=='POST' else OUT/'hinge/cells'/('PLAIN_S2000_'+stage)/rid/'post.npz'
            a=C.arrays(p);protected[str(p)]=C.sha(p);tri=a['output'];slots=a['source_face_ids']
            seg,owners,_=finite_sections(tri,axis,position,1e-11)
            for s,owner in zip(seg,owners):
                slot=int(slots[owner]);color='#1686c0' if slot<q['K'] else '#777777'
                highlight={77:'#c62828',113:'#e58b12'} if rid.startswith('608') else {21:'#c62828'}
                if slot in highlight:color=highlight[slot]
                pts=s[:,project];ax.plot(pts[:,0],pts[:,1],c=color,lw=2 if slot in highlight else 1.)
            ax.set(xlim=xlim,ylim=ylim);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
            ax.set_title(stage);ax.set_xlabel(('Z' if view=='side' else 'X')+' model units');ax.set_ylabel('Y model units')
        fig.suptitle(rid+' | '+view+f' exact plane = {position:.9f} | PLAIN_S2000 and CPU HINGE only')
        fig.text(.5,.055,'Same historic fixed window; red historic slot77 (608) / slot21 (3f), orange slot113 (608). Blue C, gray other free.',ha='center',fontsize=9)
        fig.text(.5,.025,'A short section segment is not proof of a floating face; inspect the full indexed connectivity ledger separately.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.09,1,.93));p=dest/(rid+'_fixed_hinge.png')
        with p.open('xb') as f:fig.savefig(f,format='png')
        plt.close(fig);files[str(p)]=C.sha(p)
        images.append(dict(request_id=rid,path=str(p),plane=position,view=view,xlim=xlim,ylim=ylim,image_sha256=C.sha(p),status='PENDING_ACTUAL_REVIEW'))
    p=dest/'image_manifest.json';C.write(p,images);files[str(p)]=C.sha(p)
    for p,h in protected.items():assert C.sha(p)==h
    result=dict(status='CPU_COMPLETE_IMAGES_PENDING_REVIEW',cells=84,face_records=len(faces),files=files,
        protected=protected,code_sha256=C.sha(__file__),topology_sha256=C.sha(T.__file__),version=T.VERSION,
        seconds=time.perf_counter()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        model_calls=0,new_quality_scans=0)
    C.write(dest/'completion.json',result);return result

if __name__=='__main__':
    r=run();print(json.dumps({k:v for k,v in r.items() if k not in ('files','protected')}),flush=True)
