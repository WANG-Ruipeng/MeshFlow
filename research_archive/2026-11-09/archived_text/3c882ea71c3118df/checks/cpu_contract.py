from pathlib import Path
import sys,types,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from r2.pilots.bench_loop_motif_mini_v1 import run as X
import torch
assert not torch.cuda.is_initialized()
def fixture():
    return types.SimpleNamespace(layers=[types.SimpleNamespace(forward=lambda x,c,mask=None:x+.01*(x*x+c)) for _ in range(12)])
x=torch.tensor([.2,.5],requires_grad=True);c=torch.tensor([.1,.2]);m=fixture()
y=x
for l in m.layers:y=l.forward(y,c)
g=torch.autograd.grad(y.sum(),x)[0]
X.install_loop(m,1);z=x
for l in m.layers:z=l.forward(z,c)
gz=torch.autograd.grad(z.sum(),x)[0]
assert torch.equal(y,z) and torch.equal(g,gz) and m._layer_call_order==list(range(12))
X.install_loop(m,2);z=x
for l in m.layers:z=l.forward(z,c)
gz=torch.autograd.grad(z.sum(),x)[0]
assert m._layer_call_order==[0,1,2,3,4,4,5,5,6,6,7,7,8,9,10,11]
assert torch.isfinite(gz).all() and not torch.equal(y,z)
assert not torch.cuda.is_initialized()
X.write(Path(__file__).with_name('cpu_contract.json'),dict(status='PASS',source_sha256=X.sha(__file__),
    wrapper_sha256=X.sha(X.__file__),analytic_only=True,real_model_calls=0,CUDA_initialized=False,
    checks=['r1 exact function','r1 exact input derivative','r2 full layer order','r2 differentiable changed function']))
print('CPU contract PASS')
