from pathlib import Path
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'source'))
from experiments.a_radial.layout_teacher_margin_control import package_results as p
assert Path(p.__file__).resolve().is_relative_to(root/'source')
print(json.dumps(p.portable_readback(root)),flush=True)
