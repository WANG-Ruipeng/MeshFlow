"""CPU syntax/input audit; imports no neural framework."""
import argparse,ast
from pathlib import Path
import numpy as np
from .common import *
def run(manifest=None):
    files=list(Path(__file__).parent.glob('*.py'))
    for p in files:ast.parse(p.read_text('utf-8'))
    count=0
    if manifest:
        m=read(manifest)
        for row in m['rows']:
            path=resolve(row['trace_path']);assert sha(path)==row['trace_sha256']
            with np.load(path,allow_pickle=False) as z:
                n=int(z['N']);assert z['x_by_time'].shape in ((5,n,9),(5,n,3,3))
                assert z['anchor_ids'].shape==(n,3)
                for name in ('valid_mask','known_mask'):
                    assert z[name].dtype==np.bool_ and z[name].shape==(n,)
                assert z['x_by_time'].dtype==np.float32 and np.isfinite(z['x_by_time']).all()
                assert np.allclose(z['times'],[.1,.3,.5,.7,.9],atol=1e-7)
            count+=1
        assert count==16
    result=dict(status='PASS',syntax_files=len(files),trace_arrays=count,model_calls=0)
    print(result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest');run(p.parse_args().manifest)
