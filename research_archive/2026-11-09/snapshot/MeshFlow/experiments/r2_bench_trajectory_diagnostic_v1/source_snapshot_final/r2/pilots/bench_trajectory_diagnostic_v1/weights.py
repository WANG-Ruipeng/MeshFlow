import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import torch,numpy as np,collections

def group(k):
    if k.startswith('backbone.layers.'):
        p=k.split('.');return '.'.join(p[:4])
    return k.rsplit('.',1)[0]

def main():
    torch.set_num_threads(2);out=OUT/'weights';out.mkdir(exist_ok=True)
    reg=manifest();tick=time.perf_counter();save(out/'resources.json',resources())
    states={label:torch.load(p,map_location='cpu',weights_only=True,mmap=True)['generator_state'] for label,p in reg['checkpoints'].items()}
    base=states['COMMON'];rows=[];cosines=[];spectra=[];module=collections.defaultdict(lambda:collections.Counter())
    for key,w0 in base.items():
        if not w0.is_floating_point():continue
        w0=w0.float();d={label:states[label][key].float()-w0 for label in ('A','COVERAGE','USER')}
        norm0=float(w0.norm());n=w0.numel()
        for label,delta in d.items():
            dn=float(delta.norm());g=group(key)
            row=dict(label=label,parameter=key,module=g,count=n,shape=str(list(w0.shape)),base_norm=norm0,delta_norm=dn,delta_RMS=dn/n**.5,
                relative_norm=dn/norm0 if norm0>1e-8 else None,zero_or_tiny_base=norm0<=1e-8)
            rows.append(row);module[(label,g)].update(count=n,base_sq=norm0**2,delta_sq=dn**2)
            # Exact SVD on all context matrices and predeclared early/middle/late attention matrices.
            chosen=w0.ndim==2 and (key.startswith('context_encoder') or (any(f'backbone.layers.{i}.' in key for i in (0,5,11)) and '.attn.' in key))
            if chosen:
                sv=torch.linalg.svdvals(delta);energy=sv.square();total=float(energy.sum());cum=torch.cumsum(energy,0)/(total or 1)
                spectra.append(dict(label=label,parameter=key,exact=True,stable_rank=total/float(energy[0]) if float(energy[0]) else 0,
                    top1_energy=float(cum[0]) if total else 0,top8_energy=float(cum[min(7,len(cum)-1)]) if total else 0,
                    rank90=int(torch.searchsorted(cum,torch.tensor(.9)).item()+1) if total else 0,
                    singular_values_json=json.dumps(sv.tolist())))
        for a,b in [('A','COVERAGE'),('A','USER'),('COVERAGE','USER')]:
            den=float(d[a].norm()*d[b].norm());cosines.append(dict(parameter=key,module=group(key),a=a,b=b,cosine=float((d[a]*d[b]).sum())/den if den else None))
    csvwrite(out/'per_parameter.csv',rows);csvwrite(out/'delta_cosine.csv',cosines);csvwrite(out/'exact_selected_spectra.csv',spectra)
    csvwrite(out/'per_module.csv',[dict(label=label,module=g,**v,delta_RMS=(v['delta_sq']/v['count'])**.5,relative_norm=(v['delta_sq']/v['base_sq'])**.5 if v['base_sq']>1e-16 else None) for (label,g),v in module.items()])
    # Activation files are produced by COMMON fixed-state forwards, never a new model call.
    activation=[];paths=sorted((OUT/'execution/COMMON/fixed').glob('*/activations_*.npz'))
    activation_status='COMPLETE' if len(paths)==40 else 'BLOCKED_GPU_COMMON_FIXED_STATE_NOT_AVAILABLE'
    for path in paths:
        hbank=arrays(path)
        for name,values in hbank.items():
            key=name+'.weight';h=torch.from_numpy(values);w0=base[key].float();b0=base.get(name+'.bias');y0=h@w0.T
            if b0 is not None:y0+=b0
            for label in ('A','COVERAGE','USER'):
                delta=states[label][key].float()-w0;dy=h@delta.T
                if b0 is not None:dy+=states[label][name+'.bias']-b0
                activation.append(dict(label=label,module=name,request_id=path.parent.name,t=float(path.stem.split('_t')[1]),rows=len(h),
                    h_RMS=float(h.square().mean().sqrt()),base_output_RMS=float(y0.square().mean().sqrt()),delta_output_RMS=float(dy.square().mean().sqrt()),
                    delta_over_base=float(dy.norm()/y0.norm()) if float(y0.norm()) else None,
                    input_scope='same COMMON fixed-state activations, deterministic stride max64; mixes roles; not full-network causal attribution'))
    csvwrite(out/'delta_W_h.csv',activation)
    save(out/'complete.json',dict(status='CPU_WEIGHT_PROFILE_COMPLETE',activation_status=activation_status,seconds=time.perf_counter()-tick,parameters=len(rows),exact_selected_SVDs=len(spectra),activation_rows=len(activation),GPU_calls=0,
        caveats='SVD only declared selected matrices; remaining ranks NA. DeltaWh is direct local effect on common h, not total propagated effect. Rank is not shape-manifold dimension.'))
    print('WEIGHTS_COMPLETE',time.perf_counter()-tick,flush=True)
if __name__=='__main__':main()
