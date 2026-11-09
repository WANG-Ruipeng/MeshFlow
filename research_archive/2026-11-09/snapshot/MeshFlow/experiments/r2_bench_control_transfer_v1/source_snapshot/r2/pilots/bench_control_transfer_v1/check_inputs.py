"""CPU algebra checks of the registered first batch, without model forwards."""
import time
import numpy as np
import torch
from . import train as T

def run():
    root=T.DEFAULT_ROOT;reg=T.checked(root);entry=T.read(root/'input_schedule.json')['steps'][0]
    dest=root/'cpu_input_checks';dest.mkdir(exist_ok=True);out=[];started=time.perf_counter()
    cases=[T.arrays(r['case_path']) for r in reg['records']]
    for arm in T.ARMS:
        for mi,row in enumerate(entry['samples']):
            s=T.sample_for_training(cases[row['index']],row,arm);n=int(s['N']);k=int(s['K']);fixed=s['anchor_ids']>=0
            assert np.array_equal(np.sort(s['source_face_ids']),np.arange(n))
            assert not s['u'][fixed].any()
            assert int(s['free_coordinate_count'])==int((~fixed).sum())*3
            path=dest/f'{arm}_step0001_m{mi}.npz';np.savez_compressed(path,**s)
            out.append(dict(arm=arm,micro=mi,N=n,K=k,free_coordinates=int(s['free_coordinate_count']),sha256=T.sha(path)))
    result=dict(status='PASS',stage='CPU input algebra diagnostic snapshots; training recomputes the same registered OT',
        logical_diagnostic_OT_calls=len(out),rows=out,seconds=time.perf_counter()-started,cuda_initialized=torch.cuda.is_initialized(),generator_forwards=0,optimizer_updates=0)
    assert not result['cuda_initialized'];T.write(dest/'receipt.json',result);print({k:v for k,v in result.items() if k!='rows'})
if __name__=='__main__':run()
