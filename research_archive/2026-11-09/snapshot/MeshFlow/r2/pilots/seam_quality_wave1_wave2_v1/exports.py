"""CPU-only fixed Wave1 exports; no model construction, optimizer mixing or forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json
from .protocol import OUT,WAVE1,freeze
from r2.pilots.seam_abc_v1 import generate as G

def export():
    reg=freeze()
    import torch
    from meshflow_control.checkpoints import state_hash
    folder=OUT/'inference';folder.mkdir(exist_ok=True)
    audit=OUT/'wave1_exports.json'
    if audit.exists():
        r=G.read(audit);G.verify_files(r['files']);return r
    files={};records=[]
    def load(label):
        path=Path(reg['sources'][label]);payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if payload['EMA'] is not False or state_hash(payload['generator_state'])!=payload['generator_state_sha256']:
            raise ValueError('Live generator source mismatch')
        if len(payload['generator_state'])!=194:raise ValueError('Expected194 state tensors')
        return payload
    for arm,source,step in ((WAVE1[0],'PLAIN1000',1000),(WAVE1[1],'PLAIN2000',2000),(WAVE1[2],'B_POOL47',None)):
        path=folder/(arm+'_generator.pt')
        if path.with_suffix('.json').exists():
            meta=G.read(path.with_suffix('.json'))
            if G.sha(path)!=meta['sha256']:raise ValueError('Existing export differs')
            records.append(meta);files[str(path)]=meta['sha256'];continue
        if path.exists():raise RuntimeError('Incomplete export requires explicit engineering recovery')
        p=load(source);state=p['generator_state'];checks=[]
        if step is not None:
            assert p['global_step']==13000+step and p['local_step']==step
            provenance=[dict(path=reg['sources'][source],sha256=reg['protected_files'][reg['sources'][source]])]
        else:
            q=load('PLAIN3000');other=q['generator_state']
            if list(state)!=list(other) or p['model_config']!=q['model_config']:raise ValueError('Interpolation architecture/key order mismatch')
            mixed={}
            for name,a in state.items():
                b=other[name]
                if a.shape!=b.shape or a.dtype!=b.dtype or a.dtype!=torch.float32:raise ValueError('Expected shape/dtype matched FP32 state')
                v=.5*a+.5*b
                if not torch.isfinite(v).all():raise ValueError('Nonfinite interpolation')
                mixed[name]=v;checks.append(dict(name=name,shape=list(a.shape),dtype=str(a.dtype)))
            state=mixed
            provenance=[dict(path=reg['sources'][key],sha256=reg['protected_files'][reg['sources'][key]]) for key in ('B_POOL47','PLAIN3000')]
            G.write(OUT/'wave1_interpolation_tensor_audit.json',dict(tensors=checks,alpha=.5,formula='0.5*B+0.5*PLAIN3000, FP32',optimizer_mixed=False))
            del q,other,mixed
        payload=dict(schema='r2_seam_quality_wave1_generator_v1',experiment='R2_SEAM_QUALITY_WAVE1_WAVE2_V1',
            arm=arm,local_step=step,global_step=(13000+step if step else None),seam_total_step=(1000+step if step else None),
            generator_state=state,generator_state_sha256=state_hash(state),model_config=p['model_config'],EMA=False,
            provenance=provenance,interpolation_alpha=(.5 if step is None else None))
        with path.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        meta=dict(path=str(path),sha256=G.sha(path),bytes=path.stat().st_size,arm=arm,
            local_step=step,global_step=payload['global_step'],generator_state_sha256=payload['generator_state_sha256'],provenance=provenance)
        G.write(path.with_suffix('.json'),meta);files[str(path)]=meta['sha256'];records.append(meta)
        print(json.dumps(dict(arm=arm,sha256=meta['sha256'])),flush=True)
        del payload,p,state
    r=dict(status='COMPLETE',records=records,files=files,model_calls=0,optimizer_updates=0)
    G.write(audit,r);return r

if __name__=='__main__':
    r=export();print(json.dumps(dict(status=r['status'],exports=len(r['records']))))
