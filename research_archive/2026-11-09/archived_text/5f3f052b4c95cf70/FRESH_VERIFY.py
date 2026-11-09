from pathlib import Path
import sys,os,json
os.environ['CUDA_VISIBLE_DEVICES']=''
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'source'))
from experiments.a_radial.layout_margin_single import package_results as p
p.torch.set_num_threads(4)
print(json.dumps(p.portable(root)),flush=True)
