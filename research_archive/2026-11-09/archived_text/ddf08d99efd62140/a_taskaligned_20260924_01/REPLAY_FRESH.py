"""Replay only extracted code, input, and the predeclared final student."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--run',required=True,type=Path)
p.add_argument('--report',required=True,type=Path)
a=p.parse_args();run=a.run.resolve()
sys.path.insert(0,str(run/'source_snapshot'))
import numpy as np
import torch
from experiments.a_radial import exporter
from experiments.a_radial.layout_student import deployment
from experiments.a_radial.layout_student.train import model_sha256
from experiments.a_radial.layout_student.integrity import validate_integrity
from experiments.a_radial.legacy.networks import DomainVelocity
from experiments.a_radial.io_utils import array_hash,file_hash
assert Path(exporter.__file__).resolve().is_relative_to(run/'source_snapshot')
assert Path(deployment.__file__).resolve().is_relative_to(run/'source_snapshot')
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
protocol=json.loads((run/'RUN_PROTOCOL.json').read_text())
integrity=validate_integrity(run)
integrity_rows={(r['uid'],r['model'],r['policy']):r for r in integrity['rows']}
uid=sorted(protocol['uids'])[0]
checkpoint=run/'training/seed_10/final.pt'
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert state['update']==3000 and state['is_predeclared_final']
assert state['model_sha256']==model_sha256(state['model'])
model=DomainVelocity().cpu();model.load_state_dict(state['model'],strict=True)
model.eval().requires_grad_(False)
with np.load(run/'data'/f'{uid}.npz',allow_pickle=False) as z:
    y=z['logr'];base=z['base_u'];faces=z['faces']
support=exporter.RadialSupport(run/'source_snapshot/experiments/a_radial/assets/support.npz')
result=deployment.deploy(model,y,support,base,faces)
rows=json.loads((run/'attempts'/uid/'student_10/result.json').read_text())['rows']
checks={}
for policy,prefix in [('raw','raw'),('quality_fallback','delivered')]:
    row=next(x for x in rows if x['policy']==policy)
    mesh=result[prefix+'_mesh'];audit=result[prefix+'_audit'];quality=result[prefix+'_quality']
    stored=integrity_rows[(uid,'student_10',policy)]
    observed_hash=array_hash(mesh['vertices'],faces)
    observed_stored_hash=array_hash(mesh['vertices'],mesh['faces'])
    assert np.array_equal(mesh['faces'],faces)
    assert observed_hash==row['mesh_hash']==stored['mesh_hash'],(policy,observed_hash,row['mesh_hash'])
    assert observed_stored_hash==stored['stored_array_hash']
    assert file_hash(run/row['mesh_file'])==stored['mesh_npz_sha256']
    q=deployment.qualification(quality,audit,result['floor'])
    assert q['qualified']==row['qualified'] and q['failure_reasons']==row['failure_reasons']
    assert audit['valid']==row['audit']['valid']
    assert audit['exact_executed'] and audit['exact_requested']
    assert audit['selfintersection_count']==row['audit']['selfintersection_count']
    with np.load(run/row['mesh_file'],allow_pickle=False) as saved:
        assert np.array_equal(mesh['vertices'],saved['vertices'])
        assert np.array_equal(mesh['faces'],saved['faces'])
        assert np.array_equal(mesh['u'],saved['u'])
    checks[policy]={'mesh_hash':observed_hash,'stored_array_hash':observed_stored_hash,
        'logical_and_stored_hashes_match':True,'mesh_npz_sha256_matches_integrity_sidecar':True,'matches_saved_hash':True,
        'vertices_faces_directions_bitwise_equal':True,'qualified':q['qualified'],
        'exact_endpoint_audit_matches':True}
assert result['delivery']['fallback_to_fixed']==next(x for x in rows if x['policy']=='quality_fallback')['fallback_to_fixed']
report={'status':'PASS','uid':uid,'model':'student_10','training_updates':3000,
    'device':'cpu','guard_steps':32,'network_dtype':'float32','guard_geometry_dtype':'float64',
    'checkpoint_sha256':file_hash(checkpoint),'model_sha256':state['model_sha256'],
    'code_loaded_from_fresh_extraction':True,'deployment_module':str(Path(deployment.__file__).resolve()),
    'teacher_label_read_for_prediction':False,'checks':checks,
    'fallback_to_fixed':result['delivery']['fallback_to_fixed']}
with a.report.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report),flush=True)
