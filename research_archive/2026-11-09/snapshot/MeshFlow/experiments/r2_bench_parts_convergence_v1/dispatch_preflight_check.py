"""Pure CPU structural tests; no Driver construction, child process or GPU."""
import ast,json,sys,time
from pathlib import Path
ROOT=Path('/mnt/e/MeshFlow');sys.path.insert(0,str(ROOT))
from r2.pilots.bench_parts_convergence_v1 import run_all as R
source=Path(R.__file__);compile(source.read_text('utf-8'),str(source),'exec')
assert 'torch' not in sys.modules
assert R.plan()['execution_enabled'] is False
class Fake:
    complete=None
    def __init__(self,endpoint):self.endpoint=endpoint;self.calls=[]
    def train_target(self,arms,target):self.calls.append(('train',list(arms),target))
    def evaluation(self,step,arms,subset):self.calls.append(('eval',step,list(arms),subset))
    def decision(self,step):return dict(stop_training=step==self.endpoint)
    def final_cpu(self,endpoint,decision):return dict(endpoint=endpoint,calls=self.calls)
results=[]
for endpoint in (15000,16000,17000,21000):
    result=R.Driver.run(Fake(endpoint));calls=result['calls']
    common=[r for r in calls if r[0]=='eval' and r[2]==['COMMON']]
    assert common==[('eval',2000,['COMMON'],'process'),('eval',6000,['COMMON'],'process'),('eval',10000,['COMMON'],'process'),('eval',12000,['COMMON'],'full')]
    paired=[r for r in calls if r[0]=='eval' and r[2]==['PLAIN','SEAM']]
    assert sum(r[1]==endpoint and r[3]=='full' for r in paired)==1
    assert not any(r[1]==endpoint and r[3]=='process' for r in paired)
    assert max(r[2] for r in calls if r[0]=='train')==endpoint
    assert len({(r[1],r[3]) for r in paired})==len(paired)
    results.append(dict(endpoint=endpoint,evaluation_schedule=[list(r) for r in calls if r[0]=='eval'],training_waves=sum(r[0]=='train' for r in calls)))
text=source.read_text('utf-8')
assert "if not target.exists():" in text and "--checkpoint-content-audit" in text
assert "self.force_serial['training']=True" in text
assert "first_25_isolated_update_rate" in text and 'combined_overlap_rate' in text
assert "gallery=self.root/'gallery'/key/'index.html'" in text
assert "'--estimated-peak-mib'" in text
assert 'torch' not in sys.modules
output=ROOT/'experiments/r2_bench_parts_convergence_v1/dispatch_preflight.json'
R.write(output,dict(status='PASS_STATIC_AND_SIMULATED_PLAN',code_sha256=R.sha(source),script_sha256=R.sha(__file__),
    simulations=results,default_no_dispatch=True,Torch_imported=False,subprocesses_started=0,GPU_calls=0,
    checks=['stage budget parity','minimum and cap endpoints','actual final full replaces same-step process',
            'no duplicate endpoint evaluation','matching sample-child/controller VRAM estimate',
            'real first25/overlap throughput policy implemented','gallery path matches producer',
            'named recovery checkpoint compared by CPU content rather than serialization SHA'],
    limitations=['No real child GPU dispatch executed by preflight.','Future checkpoint audit runs only after real finalized weights exist.']))
print(json.dumps(dict(status='PASS_STATIC_AND_SIMULATED_PLAN',code_sha256=R.sha(source),output=str(output)),indent=2))
