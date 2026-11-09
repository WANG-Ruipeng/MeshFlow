"""Depth-correct Bench comparisons with fixed C-position finite sections."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, json
from pathlib import Path
from r2.pilots.seam_followup_v1 import gallery as renderer

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_control_transfer_v1'

def run(root=OUT):
    root=Path(root)
    manifest=json.loads((root/'evaluation_inputs/manifest.json').read_text(encoding='utf-8-sig'))
    pages=[]
    for q in manifest['requests']:
        cells=[]
        for arm in ('PLAIN','SEAM'):
            for stage in ('raw','post'):
                path=root/'evaluation'/arm/q['request_id']/stage/(stage+'.npz')
                if not path.exists():raise FileNotFoundError(path)
                cells.append(dict(label=arm+'_'+stage.upper(),path=str(path)))
        common={k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','K','N','split')}
        pages.append(dict(**common,page_id=q['request_id'],stage='RAW / POST',
            kind='Bench: same official EMA, paired 3000-step adaptation',cells=cells))
    path=root/'gallery_manifests'/'paired.json';path.parent.mkdir(parents=True,exist_ok=True)
    value=dict(pages=pages,model_calls=0,metric_based_selection=False,
        section_policy='Central front and side finite sections at the C bounding-box center; same scale per C across both noises and all four cells.')
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Frozen gallery manifest changed')
    else:path.write_text(json.dumps(value,indent=2),encoding='utf-8')
    renderer.OUT=root
    renderer.run('paired')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(OUT));a=p.parse_args();run(a.root)
