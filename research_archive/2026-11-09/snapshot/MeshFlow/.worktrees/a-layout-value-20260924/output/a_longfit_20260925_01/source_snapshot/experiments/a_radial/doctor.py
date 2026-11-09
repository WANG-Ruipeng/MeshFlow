from pathlib import Path
import argparse,json,sys,platform,importlib.util
import torch
from .io_utils import write_json
from .verify_repo import verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--report',required=True);a=p.parse_args()
    r={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'compiled_arch_list':torch.cuda.get_arch_list() if torch.cuda.is_available() else [],'optional_packages':{n:importlib.util.find_spec(n) is not None for n in ['vtk','trimesh','scipy','numba','timm','torchdiffeq','accelerate','gradio','pymeshlab','flash_attn']}}
    try:r['repo']=verify(a.repo)
    except Exception as e:r['repo_error']=repr(e)
    if torch.cuda.is_available():
        r['gpu_name']=torch.cuda.get_device_name(0);r['capability']=torch.cuda.get_device_capability(0)
        try:
            x=torch.randn(256,256,device='cuda',requires_grad=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=(x@x).square().mean()
            loss.backward();torch.cuda.synchronize();r['matmul_backward_bf16_finite']=bool(torch.isfinite(x.grad).all())
        except Exception as e:r['gpu_smoke_error']=repr(e)
    r['notes']=['No automatic driver/toolchain change. Blackwell needs a validated CUDA/PyTorch build; upstream torch2.4/cu124 README is not a prescription for this GPU.', 'FlashAttention is optional only when pinned SDPA fallback passes parity/mask tests. Backend changes must be recorded.', 'This doctor is not a full official-model gradient/OT test.']
    write_json(a.report,r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
