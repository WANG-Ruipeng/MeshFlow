"""All sixteen saved requests, same-scale surfaces and C-centered sections."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from . import evaluate as V
from r2.pilots.seam_followup_v1 import gallery as G


def run(root=V.ROOT):
    root=V.C.resolve(root);qq=V.requests(root);pages=[]
    stages=('RAW','BASE_POST','HINGE_OPP015','FINAL_POST')
    for q in qq:
        cells=[]
        for stage in stages:
            mesh,_=V.artifact(root,q,stage);cells.append(dict(label=stage,path=str(mesh)))
        pages.append(dict(**{k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','K','N','split')},
            page_id=q['request_id'],stage=' / '.join(stages),kind='Chair BACK_FULL | frozen S2000 + LONG C-only layout',cells=cells))
    manifest=dict(pages=pages,model_calls=0,metric_based_selection=False,
        section_policy='Original renderer: two opposing surfaces and front/side finite sections centered on C; shared scale across all stages/noises of a condition',
        review_policy='Pending actual nonblind AI_PROXY; do not infer labels from metrics or unique-source style')
    kind='S2000_BACK_FULL';V.C.write(root/'gallery_manifests'/(kind+'.json'),manifest)
    G.OUT=root;G.run(kind)
    return root/'gallery'/kind/'completion.json'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(V.ROOT));a=p.parse_args();print(run(a.root))
