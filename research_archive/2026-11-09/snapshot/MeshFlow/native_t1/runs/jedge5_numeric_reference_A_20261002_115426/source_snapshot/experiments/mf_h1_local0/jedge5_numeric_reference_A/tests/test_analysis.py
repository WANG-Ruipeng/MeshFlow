"""Small CPU-only tests for the endpoint comparison conventions."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from ..analysis import endpoint_difference
from ..assets import write_json


def run():
    tick=time.perf_counter()
    a=np.empty((112,3,3),np.float32)
    base=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],np.float32)
    for i in range(112):a[i]=base+np.array([0.,0.,i*.01],np.float32)
    identity=endpoint_difference(a,a.copy(),3.)
    assert identity['coordinate_RMS']==0
    assert identity['surface']['symmetric_rms']<1e-12
    b=a.copy();b[12:,:,0]+=.01
    shift=endpoint_difference(a,b,3.)
    assert abs(shift['coordinate_RMS']-.01/np.sqrt(3))<1e-8
    assert abs(shift['vertex_RMS']-.01)<1e-8
    assert shift['surface']['symmetric_rms']>0
    assert a[12:,:,[0,2,1]].shape==(100,3,3)
    b[0,0,0]+=.01
    try:endpoint_difference(a,b,3.)
    except ValueError:known_rejected=True
    else:known_rejected=False
    assert known_rejected
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS',analytic_endpoint_cases=3,C_identity_failure_case=True,
        identical_surface_rms=identity['surface']['symmetric_rms'],
        translation_coordinate_rms=shift['coordinate_RMS'],
        model_forwards=0,backward_calls=0,OT_calls=0,CUDA_initialized=False,
        seconds=time.perf_counter()-tick)
    write_json(Path(__file__).with_suffix('.json'),result)
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':run()
