"""CPU-only checks for C source recovery and saved RAW artifact schema."""
import numpy as np
import torch
from . import train as T
from .sample import raw_arrays
from meshflow_control.seam.slot_data import inference_layout,boundary_from_C

def main():
    manifest=T.read(T.DEFAULT_ROOT/'evaluation_inputs/manifest.json');rows=[]
    seen=set()
    for q in manifest['requests']:
        if q['condition_id'] in seen:continue
        seen.add(q['condition_id']);b=T.arrays(q['inference_npz']);s=T.arrays(q['source_npz']);n=int(b['N']);k=len(b['C_faces_original'])
        edges=boundary_from_C(b['C_vertices'],b['C_faces_original'])
        for active in (False,True):
            layout=inference_layout(b['C_vertices'],b['C_faces_original'],b['C_faces_model'],n,np.full(len(edges),active,bool))
            # A non-generated fixed Gaussian state is enough to check identities.
            initial=np.zeros((n,3,3),np.float32);initial[k:]=T.arrays(q['noise_npz'])['z'];fixed=layout['anchor_ids']>=0
            initial[fixed]=layout['C_vertices'][layout['anchor_ids'][fixed]]
            result=raw_arrays(initial,layout,np.zeros(len(edges),np.float32),s)
            assert np.array_equal(result['C'],s['C_original'])
            assert np.array_equal(result['vertices'][result['faces']],result['output'])
            assert int(result['known_faces'])==k and int(result['input_faces'])==n
            rows.append(dict(condition_id=q['condition_id'],all_boundary_active=active,N=n,K=k,pass_=True))
    result=dict(status='PASS',rows=rows,generator_forwards=0,head_forwards=0,CUDA_initialized=torch.cuda.is_initialized())
    assert not result['CUDA_initialized'];T.write(T.DEFAULT_ROOT/'cpu_sampling_contract.json',result);print('CPU schema contracts PASS',len(rows))
if __name__=='__main__':main()
