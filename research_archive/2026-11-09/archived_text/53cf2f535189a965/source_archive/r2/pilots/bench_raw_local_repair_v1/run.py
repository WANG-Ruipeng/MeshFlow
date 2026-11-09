"""Recoverable CPU-only entry point; no implicit model dispatch or training."""
from __future__ import annotations
import argparse, json, os, sys
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from . import common as C


def targets_ready():
    from . import assets as A
    m=A.load_manifest(True);end=C.OUT/'targets/COMPLETE.json'
    completed=C.read(end);C.verify(completed['files']);C.verify(completed['source_hashes'])
    if completed['status']!='COMPLETE' or completed['requests']!=12:
        raise ValueError('All fixed requests must have completed target construction')
    protected={str(end):C.sha(end)};checks=[]
    for task in m['tasks']:
        rid=task['request_id'];p=C.OUT/'targets'/rid/'target.npz'
        a=C.arrays(p);raw=C.arrays(task['assets']['RAW']['path'])['raw']
        x=a['target'];s=a['corner_state'];k,n=task['K'],task['N']
        assert x.dtype==np.float32 and x.shape==(n,3,3) and np.isfinite(x).all()
        assert np.array_equal(a['raw'],raw) and np.array_equal(x[:k],raw[:k])
        assert s.shape==(n,3) and set(np.unique(s))<={0,1,2,3}
        assert np.all(s[:k]==3) and not np.any(s[k:]==3)
        assert np.array_equal(a['loss_mask'],(s==1)|(s==2))
        assert np.array_equal(x[(s==0)|(s==1)],raw[(s==0)|(s==1)])
        delta=np.linalg.norm(x.astype(float)-raw.astype(float),axis=-1)
        assert np.all(delta[s==2]>0) and delta.max()<=.005*C.L+1e-7
        assert np.array_equal(a['raw_slot_ids'],np.arange(n))
        assert np.array_equal(a['source_corner_ids'],np.tile(np.arange(3),(n,1)))
        assert np.array_equal(a['input_export_source_corner_ids'],C.arrays(task['assets']['RAW']['path'])['source_corner_ids'])
        checks.append(dict(request_id=rid,status='PASS',EDIT_corners=int((s==2).sum()),
                           KEEP_corners=int((s==1).sum()),UNKNOWN_corners=int((s==0).sum()),
                           maximum_displacement_L=float(delta.max()/C.L)))
        protected[str(p)]=C.sha(p)
    result=dict(status='READY',meaning='Ready for CPU processing, NOT a training approval or clean-GT certification',
        requests=12,checks=checks,files=protected,model_calls=0)
    p=C.OUT/'targets_READY.json'
    if p.exists():
        if C.read(p)!=result:raise RuntimeError('Target readiness identity changed')
    else:C.write(p,result)
    return result


def verify():
    from . import assets as A
    A.load_manifest(True);targets_ready()
    from . import evaluate as E
    E.verify_complete('all')
    stages={}
    for name,path in [('gallery','gallery/COMPLETE.json'),('analysis','analysis/COMPLETE.json'),
                      ('review','visual_review/COMPLETE.json'),('sealed','COMPLETE.json')]:
        p=C.OUT/path
        if not p.exists():stages[name]='PENDING';continue
        r=C.read(p);C.verify(r.get('files',{}));stages[name]=r['status']
    assert 'torch' not in sys.modules
    return dict(status='VERIFIED_AVAILABLE_COMPLETED_STAGES',stages=stages,model_calls=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['assets','targets','ready','evaluate','render','analysis','verify','all'])
    p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if a.stage in ('assets','all'):
        from . import assets;assets.build_manifest()
    if a.stage in ('targets','all'):
        from . import targets;targets.run()
    if a.stage in ('ready','all'):targets_ready()
    if a.stage in ('evaluate','all'):
        from . import evaluate;evaluate.run(workers=a.workers,mode='all')
    if a.stage in ('render','all'):
        from . import render;render.run()
    if a.stage in ('analysis','all'):
        from . import analysis;analysis.run()
    if a.stage=='verify':print(json.dumps(verify(),ensure_ascii=False))
    elif a.stage=='all':
        print('CPU stages complete. Actual image review and data gate must be recorded separately; no model is dispatched by this entry point.')

if __name__=='__main__':main()
