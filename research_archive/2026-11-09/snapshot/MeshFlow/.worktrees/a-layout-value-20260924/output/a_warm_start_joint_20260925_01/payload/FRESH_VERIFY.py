from pathlib import Path
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'source'))
from experiments.a_radial.system_probe import warm_start_package as p
p.torch.set_num_threads(4)
print(json.dumps(p.portable_readback(root)),flush=True)
