"""CPU direct delta-W action on saved identical COMMON activations, no forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import torch,numpy as np

def main():
    out=OUT/'weights/functional';done=out/'complete.json'
    if done.exists():verify(read(done)['files']);return
    torch.set_num_threads(2);reg=manifest();tick=time.perf_counter();paths=sorted((OUT/'execution/COMMON/fixed').glob('*/activations_*.npz'))
    assert len(paths)==40,'Require40 fixed-state activation banks'
    states={label:torch.load(p,map_location='cpu',weights_only=True,mmap=True)['generator_state'] for label,p in reg['checkpoints'].items()};base=states['COMMON'];rows=[]
    save(out/'resources.json',resources())
    for path in paths:
        hbank=arrays(path)
        for name,values in hbank.items():
            key=name+'.weight';h=torch.from_numpy(values);w0=base[key].float();b0=base.get(name+'.bias');y0=h@w0.T
            if b0 is not None:y0+=b0
            for label in ('A','COVERAGE','USER'):
                dy=h@(states[label][key].float()-w0).T
                if b0 is not None:dy+=states[label][name+'.bias']-b0
                rows.append(dict(label=label,module=name,request_id=path.parent.name,t=float(path.stem.split('_t')[1]),rows=len(h),
                    h_RMS=float(h.square().mean().sqrt()),base_output_RMS=float(y0.square().mean().sqrt()),delta_output_RMS=float(dy.square().mean().sqrt()),
                    delta_over_base=float(dy.norm()/y0.norm()) if float(y0.norm()) else None,
                    input_scope='same COMMON activations, deterministic stride max64; mixed roles; direct local action including bias, not propagated attribution'))
    csvwrite(out/'delta_W_h.csv',rows)
    save(done,dict(status='COMPLETE',activation_banks=len(paths),rows=len(rows),seconds=time.perf_counter()-tick,model_calls=0,
        files={str(out/'delta_W_h.csv'):sha(out/'delta_W_h.csv')},activation_hashes={str(p):sha(p) for p in paths},
        caveat='Sampled real h; no per-role CKA or causal module localization claimed'))
    print('FUNCTIONAL_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
