import json, os, time
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
import torch
from experiments.a_radial.io_utils import file_hash,array_hash,save_npz,write_json
from experiments.a_radial.legacy.networks import DomainVelocity
from experiments.a_radial.exporter import RadialSupport
from experiments.a_radial.layout_student.deployment import predict_raw
from experiments.a_radial.time_quality_followup.conditioned_deployment import predict_raw_with_condition

root=Path.cwd()
out=root/'output/a_condition_wrapper_gate_20260925_01'
source=root/'output/a_time_quality16_20260925_01'
p=json.loads((source/'RUN_PROTOCOL.json').read_text())
b=p['model_bindings']['early_v1_10']
assert file_hash(b['path'])==b['file_sha256']
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
net=DomainVelocity().cpu().eval().requires_grad_(False)
net.load_state_dict(torch.load(b['path'],map_location='cpu',weights_only=True)['model'])
support=RadialSupport();results=[];datasets={};condition_paths={};start=time.perf_counter()
for uid in p['uids']:
    binding=p['data'][uid]
    assert file_hash(binding['path'])==binding['sha256']
    with np.load(binding['path'],allow_pickle=False) as z:d={k:np.array(z[k]) for k in z.files}
    datasets[uid]=d;cp=out/'conditions'/f'{uid}.npz';save_npz(cp,logr=d['logr']);condition_paths[uid]=cp
    kwargs=dict(target_logr=d['logr'],target_base=d['base_u'],target_faces=d['faces'],target_uid=uid,
        target_field_hash=array_hash(d['logr']),support=support,condition_path=cp,condition_uid=uid,
        condition_file_sha256=file_hash(cp),condition_field_hash=array_hash(d['logr']),expected_model_sha256=b['model_sha256'])
    old,meta0=predict_raw(net,d['logr'],support,d['base_u'],d['faces'])
    new,meta1=predict_raw_with_condition(net,**kwargs)
    task=json.loads((source/'cases'/uid/'early_v1_10/repeat_0/TASK.json').read_text())
    ap=source/'cases'/uid/'early_v1_10/repeat_0'/task['arrays_file']
    assert file_hash(ap)==task['arrays_file_sha256']
    with np.load(ap,allow_pickle=False) as stored:
        for key in ('u','r','vertices','faces','alpha'):
            assert old[key].dtype==new[key].dtype and np.array_equal(old[key],new[key]),(uid,key,'legacy parity')
            assert new[key].dtype==stored['raw_'+key].dtype and np.array_equal(new[key],stored['raw_'+key]),(uid,key,'saved parity')
    assert meta0['proxy_points_hash']==meta1['proxy_points_hash']
    results.append({'uid':uid,'status':'PASS','condition_file_sha256':file_hash(cp),'target_field_hash':array_hash(d['logr']),
        'exact_keys':['u','r','vertices','faces','alpha'],'legacy_and_previously_saved_raw_equal':True})
uid,donor=p['uids'][:2];d=datasets[uid];dy=datasets[donor]['logr'];cp=condition_paths[donor]
wrong,wm=predict_raw_with_condition(net,target_logr=d['logr'],target_base=d['base_u'],target_faces=d['faces'],
    target_uid=uid,target_field_hash=array_hash(d['logr']),support=support,condition_path=cp,condition_uid=donor,
    condition_file_sha256=file_hash(cp),condition_field_hash=array_hash(dy),expected_model_sha256=b['model_sha256'])
tr=support.query(d['logr'],wrong['u']);dr=support.query(dy,wrong['u'])
assert np.array_equal(tr,wrong['r']) and np.array_equal(wrong['vertices'],wrong['u']*tr[:,None])
assert not np.array_equal(dr,wrong['r'])
assert wm['target_uid']==uid and wm['condition_uid']==donor
assert not torch.cuda.is_initialized()
value={'status':'PASS','scope':'Implementation gate on already-seen development objects, not new-parent validation or efficacy result',
    'correct_condition_exact_parity_count':len(results),'results':results,'real_mismatched_target_radius_check':{'target':uid,'donor':donor,
    'target_query_exact':True,'donor_radius_negative_control_rejected':True},'model_tensor_sha256':b['model_sha256'],
    'source_protocol_sha256':file_hash(source/'RUN_PROTOCOL.json'),'wrapper_sha256':file_hash(root/'experiments/a_radial/time_quality_followup/conditioned_deployment.py'),
    'script_sha256':file_hash(__file__),'wall_seconds':time.perf_counter()-start,'cuda_initialized':False,'training_updates':0}
write_json(out/'REAL_PARITY.json',value)
print(json.dumps({'status':value['status'],'exact_parity_objects':len(results),'wrong_radius_control':'PASS','wall_seconds':value['wall_seconds']}))