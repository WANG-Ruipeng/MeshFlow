"""Saved geometry exterior + identical historic front/side finite sections."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,html,time
from pathlib import Path
from .evaluation import C,ROOT,OUT,ARMS,HIST,requests,mesh_path,dispatch_cpu
from r2.pilots.seam_followup_v1 import gallery as old

def run(kind):
    dest=OUT/'gallery'/kind;done=dest/'completion.json'
    if done.exists():
        r=C.read(done)
        for p in r['images']:assert C.sha(p['path'])==p['image_sha256']
        return r
    pages=[]
    for q in requests():
        if kind in ('RAW','POST','BOUND015'):
            cells=[dict(label=l,path=str(mesh_path(l,q['request_id'],kind))) for l in ('PLAIN_S2000',*ARMS)]
        else:
            label=kind.removeprefix('HINGE_');assert label in ('PLAIN_S2000',*ARMS)
            cells=[dict(label=label+'_POST',path=str(mesh_path(label,q['request_id'],'POST')))]
            cells += [dict(label=label+'_HINGE'+x,path=str(OUT/'hinge/cells'/(label+'_HINGE'+x)/q['request_id']/'post.npz')) for x in ('015','025')]
        pages.append(dict(**{k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','K','N','split')},
            page_id=q['request_id'],kind=kind,stage=kind,cells=cells))
    original=HIST/'gallery/wave1_POST/definition.json';hist=C.read(original)
    definition=dict(version='seam_hinge_quality_historical_camera_v1',code_sha256=C.sha(__file__),
        original_source_path=str(original),original_source_sha256=C.sha(original),
        bounds=hist['bounds'],depth_bounds=hist['depth_bounds'],
        mesh_hashes={c['path']:C.sha(c['path']) for p in pages for c in p['cells']},
        scale='Exact historic centers, scales and central finite sections; no output-specific recentering',
        slot_semantics='Same RAW slot within one generated output only; no cross-model source-face correspondence claim',
        renderer_source={str(Path(old.__file__)):C.sha(old.__file__)})
    dp=dest/'definition.json';C.write(dp,definition);C.write(dest/'manifest.json',dict(pages=pages))
    jobs=[(p,str(dest),C.sha(dp),hist['bounds'][p['condition_id']],hist['depth_bounds'][p['condition_id']]) for p in pages]
    images,n,seconds=dispatch_cpu(jobs,old.one,dest,workers=2)
    (dest/'index.html').write_text('<meta charset="utf-8"><h1>'+html.escape(kind)+'</h1><p>Exact historical exterior cameras and central finite section planes. AI_PROXY review separate.</p>'+''.join(
        '<h2>'+r['request_id']+'</h2><a href="'+r['page_id']+'.png"><img loading="lazy" style="width:100%" src="'+r['page_id']+'.png"></a>' for r in images),encoding='utf-8')
    r=dict(status='RENDER_COMPLETE_PENDING_ACTUAL_REVIEW',images=images,workers=n,wall_seconds=seconds,definition_sha256=C.sha(dp))
    C.write(done,r);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['RAW','POST','BOUND015',*['HINGE_'+x for x in ('PLAIN_S2000',*ARMS)]]);a=p.parse_args()
    r=run(a.kind);print(r['status'],len(r['images']),flush=True)
