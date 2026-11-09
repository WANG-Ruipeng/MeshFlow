"""CPU fixed-scale complete-panel comparison galleries; no synthesized images."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse
from r2.pilots.seam_abc_v1 import eval_common as C
from r2.pilots.seam_followup_v1 import gallery as old
OUT=C.ROOT/'experiments/r2_seam_pool47_couple_v1'
BASE=C.ROOT/'experiments/r2_seam_followup_v1/layout'

def manifest(kind):
    qset=C.read(OUT/'protocol.json')['evaluation']['requests'];pages=[]
    for q in qset:
        common={k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','K','N','split')}
        if kind=='cpu_binding':
            cells=[dict(label='B_LONG_LAYOUT',path=str(BASE/'evaluation/B_LONG_LAYOUT'/q['request_id']/'post/post.npz'))]
            for cap in ('015','020'):
                label='B_LONG_LAYOUT_BIND_'+cap
                cells.append(dict(label=label,path=str(OUT/'cpu_binding/cells'/label/q['request_id']/'post.npz')))
            pages.append(dict(**common,page_id=q['request_id']+'_POST',stage='POST',kind='B fixed LONG layout, CPU cluster binding',cells=cells))
        else:
            for stage in ('RAW','POST'):
                cells=[dict(label='B_STEP0',path=str(BASE/'evaluation/B_LONG_LAYOUT'/q['request_id']/stage.lower()/(stage.lower()+'.npz')))]
                for arm in ('PLAIN3000','COUPLE3000'):
                    cells.append(dict(label=arm,path=str(OUT/'evaluation'/arm/q['request_id']/stage.lower()/(stage.lower()+'.npz'))))
                pages.append(dict(**common,page_id=q['request_id']+'_'+stage,stage=stage,kind='Same B start and LONG layout, paired 3000 update endpoints',cells=cells))
    dest=OUT/'gallery_manifests'/(kind+'.json')
    C.write(dest,dict(pages=pages,model_calls=0,metric_based_selection=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['cpu_binding','training','endpoint_binding']);a=p.parse_args()
    if a.kind=='endpoint_binding':
        from .results import gallery_manifests
        gallery_manifests()
    else:manifest(a.kind)
    old.OUT=OUT;old.run(a.kind)
if __name__=='__main__':main()
