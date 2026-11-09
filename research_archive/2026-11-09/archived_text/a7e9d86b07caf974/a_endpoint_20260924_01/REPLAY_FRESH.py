"""Fresh extracted original CPU replay, using extracted artifacts only."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--report',required=True);a=p.parse_args()
run=Path(a.run).resolve();sys.path.insert(0,str(run/'source_snapshot'))
import numpy as np
import torch
from experiments.a_radial import exporter
from experiments.a_radial.layout_student import deployment
from experiments.a_radial.layout_student.train import model_sha256
from experiments.a_radial.legacy.networks import DomainVelocity
from experiments.a_radial.io_utils import array_hash,file_hash
assert Path(exporter.__file__).resolve().is_relative_to(run/'source_snapshot')
assert Path(deployment.__file__).resolve().is_relative_to(run/'source_snapshot')
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
protocol=json.loads((run/'RUN_PROTOCOL.json').read_text());uid=sorted(protocol['uids'])[0]
checkpoint=run/'training/c2/seed_10/final.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert state['update']==1000 and state['is_predeclared_final'] and not state['diagnostic_only']
assert state['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json')
assert state['model_sha256']==model_sha256(state['model'])
initial=run/protocol['paths']['initial_checkpoints']['10']
assert file_hash(initial)==state['initial_checkpoint_sha256']==protocol['hashes']['initial_checkpoint_sha256']['10']
model=DomainVelocity().cpu();model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
with np.load(run/'data'/f'{uid}.npz',allow_pickle=False) as z:
    y=z['logr'];base=z['base_u'];faces=z['faces']
support=exporter.RadialSupport(run/'source_snapshot/experiments/a_radial/assets/support.npz')
result=deployment.deploy(model,y,support,base,faces)
rows=json.loads((run/'attempts'/uid/'c2_10/result.json').read_text())['rows'];checks={}
for policy,prefix in [('raw','raw'),('quality_fallback','delivered')]:
    row=next(r for r in rows if r['policy']==policy)
    mesh=result[prefix+'_mesh'];audit=result[prefix+'_audit'];quality=result[prefix+'_quality']
    assert array_hash(mesh['vertices'],faces)==row['mesh_hash']
    assert array_hash(mesh['vertices'],mesh['faces'])==row['stored_array_hash']
    assert file_hash(run/row['mesh_file'])==row['mesh_npz_sha256']
    with np.load(run/row['mesh_file'],allow_pickle=False) as saved:
        for key in ('vertices','faces','u','r'):
            assert np.array_equal(mesh[key],saved[key]),(policy,key)
        if 'alpha' in mesh:assert np.array_equal(mesh['alpha'],saved['alpha'])
    radius=support.query(y,mesh['u'])
    assert np.array_equal(mesh['r'],radius)
    assert np.array_equal(mesh['vertices'],mesh['u']*radius[:,None])
    gate=deployment.qualification(quality,audit,result['floor'])
    assert gate['qualified']==row['qualified'] and gate['failure_reasons']==row['failure_reasons']
    assert audit['exact_requested'] and audit['exact_executed']
    for key in ('valid','selfintersection_count'):assert audit[key]==row['audit'][key]
    for key in ('min_triangle_angle_deg','triangle_min_angle_p05_deg'):assert abs(quality[key]-row['quality'][key])<=1e-10
    checks[policy]={'mesh_hash':row['mesh_hash'],'stored_array_hash':row['stored_array_hash'],
        'mesh_npz_sha256':row['mesh_npz_sha256'],'geometry_and_accepted_query_bitwise_equal':True,
        'audit_and_qualification_match':True,'qualified':gate['qualified']}
assert result['delivery']['fallback_to_fixed']==next(r for r in rows if r['policy']=='quality_fallback')['fallback_to_fixed']
report={'status':'PASS','model':'c2_10','uid':uid,'updates':1000,'device':'cpu',
    'loaded_from_fresh_extracted_source':True,'teacher_label_read_for_prediction':False,
    'initial_checkpoint_sha256_verified':True,'checkpoint_sha256':file_hash(checkpoint),
    'checks':checks,'fallback_to_fixed':result['delivery']['fallback_to_fixed']}
with Path(a.report).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report),flush=True)
