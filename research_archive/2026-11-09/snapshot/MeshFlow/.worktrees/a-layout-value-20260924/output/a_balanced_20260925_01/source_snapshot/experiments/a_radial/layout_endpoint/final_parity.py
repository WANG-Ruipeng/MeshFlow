"""Post-training CPU deployment versus differentiable CPU/CUDA mirrors.

All six final checkpoints and all 16 objects are retained. This diagnostic never
selects weights, changes mesh outputs, or substitutes GPU qualification in tables.
"""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from ..exporter import RadialSupport
from ..io_utils import file_hash,write_json
from ..layout_student.deployment import named_quality,qualification
from ..layout_value.geometry import audit_mesh
from ..layout_value.optimizer import TorchRadialField
from .rollout import rollout
from .evaluate import load_network


def run_checks(run):
    run=Path(run);dest=run/'FINAL_DEPLOYMENT_PARITY.json'
    if dest.exists():raise FileExistsError(dest)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=json.loads((run/'RUN_PROTOCOL.json').read_text())
    ledger=json.loads((run/'results.json').read_text())
    rows={(r['uid'],r['model'],r['policy']):r for r in ledger['rows']}
    support=RadialSupport();records=[];begin=time.perf_counter()
    for arm in ['c0','c1','c2']:
        for seed in [10,11]:
            name=f'{arm}_{seed}';model,identity=load_network(run,name,protocol)
            for uid in sorted(protocol['uids']):
                row=rows.get((uid,name,'raw'))
                if row is None:
                    for device in ['cpu','cuda']:
                        records.append({'uid':uid,'model':name,'device':device,'status':'MISSING_CPU_RESULT'})
                    continue
                with np.load(run/'data'/f'{uid}.npz',allow_pickle=False) as z:
                    base=z['base_u'];features=z['point_features'];faces=z['faces'];y=z['logr']
                with np.load(run/row['mesh_file'],allow_pickle=False) as z:
                    cpu={key:np.array(z[key]) for key in z.files}
                for device in ['cpu','cuda']:
                    model.to(device)
                    with torch.no_grad():
                        pred=rollout(model,torch.as_tensor(base[None],device=device),
                                     torch.as_tensor(features[None],device=device),faces)
                        field=TorchRadialField(y,device=device,support=support)
                        rad=field.query(pred['u'][0]);v=(pred['u'][0]*rad[:,None]).cpu().numpy()
                    u=pred['u'][0].cpu().numpy();r=rad.cpu().numpy()
                    quality=named_quality(v,faces);audit=audit_mesh(v,faces,exact=True)
                    qualified=qualification(quality,audit,row['quality_floor'])
                    delta_u=float(np.max(np.abs(u-cpu['u'])))
                    alpha_difference=int(np.sum(pred['alphas'][0]!=cpu['alpha']))
                    gate_changed=qualified['qualified']!=row['qualified']
                    reasons_changed=qualified['failure_reasons']!=row['failure_reasons']
                    records.append({'uid':uid,'model':name,'device':device,'status':'MEASURED',
                        'checkpoint_sha256':identity['file_sha256'],'cpu_mesh_file_sha256':file_hash(run/row['mesh_file']),
                        'direction_max_abs_difference':delta_u,
                        'radius_max_abs_difference':float(np.max(np.abs(r-cpu['r']))),
                        'vertex_max_abs_difference':float(np.max(np.abs(v-cpu['vertices']))),
                        'guard_alpha_disagreements':alpha_difference,
                        'angle_difference_deg':{k:quality[k]-row['quality'][k] for k in ['min_triangle_angle_deg','triangle_min_angle_p05_deg']},
                        'cpu_qualified':row['qualified'],'mirror_qualified':qualified['qualified'],
                        'qualification_changed':gate_changed,'failure_reasons_changed':reasons_changed,
                        'cpu_failure_reasons':row['failure_reasons'],'mirror_failure_reasons':qualified['failure_reasons'],
                        'mirror_exact_audit_valid':audit['valid'],
                        'numerical_parity_within_direction_atol_1e_minus_6_and_same_alpha_gate':delta_u<=1e-6 and alpha_difference==0 and not gate_changed})
            print(json.dumps({'parity_model_complete':name}),flush=True)
    stats={}
    for device in ['cpu','cuda']:
        rr=[r for r in records if r.get('device')==device and r['status']=='MEASURED']
        stats[device]={'recorded':len(rr),'expected':96,'missing_cases':96-len(rr),
            'direction_tolerance_exceeded_cases':sum(r['direction_max_abs_difference']>1e-6 for r in rr),
            'per_row_parity_failure_count':sum(not r['numerical_parity_within_direction_atol_1e_minus_6_and_same_alpha_gate'] for r in rr),
            'direction_max_abs_difference':max((r['direction_max_abs_difference'] for r in rr),default=None),
            'radius_max_abs_difference':max((r['radius_max_abs_difference'] for r in rr),default=None),
            'angle_max_abs_difference_deg':max((abs(v) for r in rr for v in r['angle_difference_deg'].values()),default=None),
            'alpha_disagreement_cases':sum(r['guard_alpha_disagreements']>0 for r in rr),
            'qualification_changed_cases':sum(r['qualification_changed'] for r in rr),
            'failure_reasons_changed_cases':sum(r['failure_reasons_changed'] for r in rr),
            'mirror_invalid_count':sum(not r['mirror_exact_audit_valid'] for r in rr)}
    result={'status':'NUMERICAL_DEPLOYMENT_GAP' if any(s['per_row_parity_failure_count'] or s['recorded']!=96 for s in stats.values()) else 'MEASURED_WITHIN_DIRECTION_TOLERANCE_SAME_ALPHA_AND_QUALIFICATION',
        'scope':'POSTHOC_ALL_FIXED_FINALS_NO_CHECKPOINT_SELECTION','main_table_uses_original_CPU_only':True,
        'cuda_bitwise_equal_claim':False,'statistics':stats,'records':records,
        'seconds':time.perf_counter()-begin,'results_sha256':file_hash(run/'results.json'),
        'rollout_source_sha256':file_hash(Path(__file__).with_name('rollout.py')),'source_sha256':file_hash(__file__)}
    write_json(dest,result);print(json.dumps({'status':result['status'],'statistics':stats}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run_checks(a.run)