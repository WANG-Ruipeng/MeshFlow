import sys,os,ast
sys.path.insert(0,'/mnt/e/MeshFlow')
os.chdir('/mnt/e/MeshFlow')
from pathlib import Path
import torch
from experiments.mf_h1_local0.control_mvp48_budget_single.reporting import run
from experiments.mf_h1_local0.control_mvp48_budget_single.common import atomic_json
source=Path('native_t1/runs/control_mvp48_v1');out=Path('native_t1/runs/control_mvp48_budget_single_v1')
for name in ('evaluation.py','reporting.py'):ast.parse((Path('experiments/mf_h1_local0/control_mvp48_budget_single')/name).read_text())
result=run(source,out,False)
assert result['old_holdout_raw_reused']==96,result['old_holdout_raw_reused']
assert result['evaluation_generations']==0
assert not torch.cuda.is_initialized()
atomic_json(out/'registration/reporting_cpu_test.json',dict(status='PASS',old_raw_reused=96,new_generations=0,CUDA_initialized=False,model_forwards=0))
print('REPORTING_CPU_PASS',result['old_holdout_raw_reused'])
