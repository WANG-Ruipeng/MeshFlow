"""Synthetic driver fault injection only. No model inference or mesh extraction."""
import contextlib,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
CODE=ROOT.parents[1]/'experiments/qem_set/code'
sys.path.insert(0,str(CODE))
source=CODE/'sample_fm_pilot.py';snapshot=ROOT/'sample_fm_pilot_snapshot.py';shutil.copy2(source,snapshot)
spec=importlib.util.spec_from_file_location('fm_driver_under_test',snapshot)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
work=ROOT/'mock_run';work.mkdir();(work/'samples').mkdir()
s.ROOT=work
np.savez_compressed(work/'data.npz',states=np.zeros((24,256,22)),mean=np.zeros(22),scale=np.ones(22),object_names=np.array(['target%02d'%i for i in range(24)]))
for arm in ['direct','transport_residual']:
    (work/'training'/arm).mkdir(parents=True)
    (work/'training'/arm/'final.pt').write_bytes(b'MOCK ONLY')
class DummyModel:
    def __init__(self,**kw):pass
    def to(self,*args,**kw):return self
    def load_state_dict(self,*args,**kw):pass
    def eval(self):return self
s.DeepSetsVelocity=DummyModel
s.torch.load=lambda *a,**kw:dict(model_state_dict={},status='COMPLETED',completed_steps=3000)
original_tensor=torch.as_tensor;original_full=torch.full
s.torch.as_tensor=lambda *a,**kw:original_tensor(*a,**{**kw,'device':'cpu'})
s.torch.full=lambda *a,**kw:original_full(*a,**{**kw,'device':'cpu'})
s.sync=lambda:None
calls={'direct':-1,'transport_residual':-1,'decode':0,'extract':0,'geometry':0,'metrics':0}
def velocity(model,z,t,mean,scale,arm):
    k=round(float(t.item())*64)
    if k==0:calls[arm]+=1
    if arm=='direct' and calls[arm]==0 and k==2:raise FloatingPointError('MOCK_ROLLOUT_THROW')
    value=torch.zeros_like(z)
    if arm=='direct' and calls[arm]==1 and k==3:value[:]=float('inf')
    return value,{}
s.total_velocity=velocity
original_decode=s.decode_state
def decode(*a,**kw):
    calls['decode']+=1
    if calls['decode']==1:raise ValueError('MOCK_DECODE_THROW')
    return original_decode(*a,**kw)
s.decode_state=decode
v=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]])
f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype=int)
class MockExtractorError(RuntimeError):
    def __init__(self):
        super().__init__('MOCK_EXTRACTION_THROW')
        self.audit={'status':'MOCK_EXTRACTION_FAILED'}
        self.trace_arrays={'mock_failure_trace':np.array([1])}
def extract(params):
    calls['extract']+=1
    if calls['extract']==1:raise MockExtractorError()
    return v,f,np.arange(4),{'mock_success_trace':np.array([1])},{'status':'EXTRACTED','graph_capacity_stats':{'nonfinite_count':0,'negative_count':0}}
s.extract=extract
s.site_cd=lambda a,b:float(np.sum((a.mean(axis=0)-b.mean(axis=0))**2))
def geometry(*a):
    calls['geometry']+=1
    return dict(vertices=4,faces=4,closed_oriented_2manifold=calls['geometry']!=2,beta0=1,beta1=0,beta2=1,duplicate_faces=0)
s.mesh_audit=geometry
s.strict_links=lambda f:dict(strict_all_vertex_links_simple_cycles=True)
s.exact_mesh=lambda *a:dict(exact_intersection_pairs=0,exact_degenerate_faces=0)
s.load_mesh=lambda *a:(v,f)
s.sample=lambda *a,**kw:(v,np.ones_like(v),None)
def metrics(*a,**kw):
    calls['metrics']+=1
    if calls['metrics']==1:raise ValueError('MOCK_METRICS_THROW')
    return dict(mock_only=True)
s.metrics=metrics
with (ROOT/'MOCK_STDOUT.log').open('w') as log,contextlib.redirect_stdout(log):s.main()
rows=json.loads((work/'LEDGER.json').read_text());summary=json.loads((work/'SUMMARY_FM.json').read_text())
checks={
    '64_rows':len(rows)==64,
    'unique_full_slots':{(r['arm'],r['sample_id']) for r in rows}=={(a,i) for a in ['direct','transport_residual'] for i in range(32)},
    'all_64_rollouts_attempted':calls['direct']==31 and calls['transport_residual']==31,
    'rollout_throw_NFE_preserved':rows[0]['NFE']==3 and rows[0]['rollout_failure'] is not None,
    'nonfinite_Euler_failure_preserved':rows[1]['NFE']==4 and rows[1]['rollout_failure'] is not None,
    'five_failure_stages_preserved':all(not r['valid'] and r['status']!='EXTRACTED' for r in rows[:5]),
    'invalid_geometry_preserved':rows[5]['status']=='EXTRACTED' and rows[5]['valid'] is False,
    '58_valid_outcomes':sum(r['valid'] for r in rows)==58,
    '32_requested_denominators':all(x['requested_samples']==32 for x in summary['by_arm'].values()),
    'failure_trace_saved':(work/'samples/direct_003/extractor_trace.npz').is_file(),
    'all_60_successful_extraction_traces_saved':len(list((work/'samples').glob('*/extractor_trace.npz')))==61,
    'failed_rollout_states_saved':len(list((work/'samples').glob('*/failed_rollout_state.npz')))==2,
    'JSON_summary_parses':True,
}
result=dict(passed=all(checks.values()),checks=checks,calls=calls,ledger_rows=len(rows),valid=sum(r['valid'] for r in rows),
    pipeline_completed=sum(r['status']=='EXTRACTED' for r in rows),actual_mock_extraction_completed=calls['extract']-1,
    source=dict(path=str(source),snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest()),
    scope='CPU-only synthetic driver fault injection. Dummy model, zero velocity, mocked mesh extraction/audits/metrics. No real sampling or geometry conclusions.')
(ROOT/'VALIDATION.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(0 if result['passed'] else 1)