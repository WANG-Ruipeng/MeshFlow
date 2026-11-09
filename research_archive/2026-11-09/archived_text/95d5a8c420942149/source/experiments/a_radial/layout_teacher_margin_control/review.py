"""Independent paired readback for teacher-start64 original-vs-strict floor."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from ..io_utils import array_hash,file_hash,write_json
from ..exporter import RadialSupport
from ..layout_student.deployment import named_quality,qualification
from ..layout_student.integrity import mesh_identity,safe_path
from ..legacy.chart import dets
from ..layout_teacher_margin.review import reconstruct

EXPECTED_PROTOCOL='094895e034968c910ce69a7e128318e9eb6f0ce62065210150096779e24a267e'
KEYS=('min_triangle_angle_deg','triangle_min_angle_p05_deg')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def require(ok,message):
    if not ok:raise ValueError(message)


def mesh_read(run,path,record,data):
    identity=mesh_identity(path,data['faces']);require(all(record[k]==v for k,v in identity.items()),'Typed geometry identity differs')
    with np.load(path,allow_pickle=False) as z:u=np.array(z['u']);v=np.array(z['vertices']);radius=np.array(z['r'])
    support=RadialSupport();fresh=support.query(data['logr'],u)
    require(np.array_equal(radius,fresh) and np.array_equal(v,u*fresh[:,None]),'Accepted radius not fresh')
    q=named_quality(v,data['faces']);require(q==record['quality'],'Actual quality changed')
    return u,q


def audit(run):
    started=time.perf_counter();run=Path(run).resolve()
    require(not torch.cuda.is_initialized(),'CPU-only paired review')
    require(file_hash(run/'RUN_PROTOCOL.json')==EXPECTED_PROTOCOL,'Unexpected frozen causal protocol')
    p=read(run/'RUN_PROTOCOL.json');uids=p['uids'];result=read(run/'RESULTS.json');summary=read(run/'SUMMARY.json')
    noise=read(run/'SENSITIVITY.json');source_result=read(run/'source_evidence/RESULTS.json');source_noise=read(run/'source_evidence/SENSITIVITY.json')
    require(result['protocol_sha256']==summary['protocol_sha256']==EXPECTED_PROTOCOL,'Protocol reference differs')
    require(summary['results_sha256']==file_hash(run/'RESULTS.json') and summary['sensitivity_sha256']==file_hash(run/'SENSITIVITY.json'),'Summary identity differs')
    bound={**p['bound_source_sha256'],**p['source_sha256']}
    for path,digest in bound.items():require(file_hash(path)==digest,'Frozen source changed: '+path)
    require(file_hash(run/'DATA_MANIFEST.json')==p['data_manifest_sha256'],'Data manifest changed')
    files=read(run/'DATA_MANIFEST.json')['files']
    for relative,digest in files.items():require(file_hash(safe_path(run,relative))==digest,'Copied source changed: '+relative)
    rows={r['uid']:r for r in result['rows']};tasks={t['uid']:t for t in result['tasks']}
    require(len(rows)==len(result['rows'])==len(tasks)==len(result['tasks'])==4 and set(rows)==set(tasks)==set(uids),'Four fixed calls/rows incomplete')
    strict_tasks={t['uid']:t for t in source_result['tasks'] if t['initializer']=='teacher'}
    new_noise={(r['uid'],r['trial'],r['fraction']):r for r in noise['rows']}
    old_noise={(r['uid'],r['trial'],r['fraction']):r for r in source_noise['rows'] if r['endpoint']=='opt_teacher'}
    expected={(uid,t,f) for uid in uids for t in range(16) for f in p['fractions']}
    require(len(noise['rows'])==len(new_noise)==448 and set(new_noise)==set(old_noise)==expected,'Paired448 noise slots differ')
    support=RadialSupport();pairs=[];noise_summary=[];max_angle_error=0.;max_mse_error=0.;reconstructed=0
    for uid in uids:
        with np.load(run/'data'/f'{uid}.npz',allow_pickle=False) as z:data={k:np.array(z[k]) for k in z.files}
        fixed=named_quality(data['base_u']*support.query(data['logr'],data['base_u'])[:,None],data['faces'])
        strict_floor={k:fixed[k]+.1 for k in KEYS};original_floor={k:fixed[k] for k in KEYS}
        row=rows[uid];task=tasks[uid]
        require(row==read(run/'endpoints'/uid/'teacher_start_original_floor'/'RESULT.json'),'Control endpoint ledger differs')
        require(task==read(run/'attempts'/uid/'TASK.json'),'Control task ledger differs')
        strict_record=read(safe_path(run,p['reference_rows'][uid]['opt_teacher']['row']))
        strict_u,strict_q=mesh_read(run,safe_path(run,p['reference_rows'][uid]['opt_teacher']['mesh']),strict_record,data)
        reference_teacher=read(safe_path(run,p['reference_rows'][uid]['teacher']['row']))
        old_u,_=mesh_read(run,safe_path(run,p['reference_rows'][uid]['teacher']['mesh']),reference_teacher,data)
        require(np.array_equal(old_u,data['teacher_u']),'Original teacher start changed')
        require(strict_record['qualified_original']==qualification(strict_q,strict_record['audit'],fixed)
            and strict_record['qualified_strict']==qualification(strict_q,strict_record['audit'],strict_floor),'Strict endpoint gates differ')
        control_u=None;pair={'uid':uid,'control_status':row['status'],'control_restore_failed':task.get('restore_failed'),
            'strict_margins_vs_fixed_deg':strict_record['actual_margin_vs_fixed_deg'],'strict_qualified':strict_record['qualified_strict']['qualified']}
        if 'stats' in task:
            stats=task['stats'];other=strict_tasks[uid]['stats'];disk=read(run/'attempts'/uid/'OPTIMIZER.json')
            require(stats==disk['stats'],'Optimizer stats changed')
            settings=('max_steps','max_seconds','device','dtype','training_seed','training_reference_samples',
                'reverse_quadrature_points_per_face','learning_rate','max_tangent_step_norm','line_search_trials','quality_tolerance_deg','selection')
            require(all(stats[k]==other[k] for k in settings),'A non-floor optimizer setting differs')
            require(stats['max_steps']==64 and stats['device']=='cpu' and stats['training_seed']==1493501+p['all16_sorted_uids'].index(uid),'Budget/device/seed mismatch')
            require(stats['quality_floor']=={'min_angle_deg':fixed[KEYS[0]],'angle_q05_deg':fixed[KEYS[1]]},'Control secretly retained strict floor')
            require(other['quality_floor']=={'min_angle_deg':strict_floor[KEYS[0]],'angle_q05_deg':strict_floor[KEYS[1]]},'Paired strict floor differs')
            arrays=run/'attempts'/uid/'OPTIMIZER_ARRAYS.npz';require(file_hash(arrays)==disk['arrays_sha256'],'Control optimizer arrays changed')
            with np.load(arrays,allow_pickle=False) as z:
                require(np.array_equal(z['snapshot_0_u'],data['teacher_u']),'Control start is not frozen teacher')
                optimizer_final=np.array(z['final_u'])
            require(task['restore_failed']==(stats['best_feasible_iteration'] is None),'Restoration failure hidden')
            pair.update(control_actual_steps=stats['iterations'],strict_actual_steps=other['iterations'],
                optimizer_settings_equal_except_floor=True,optimizer_call_seconds=task['optimizer_call_seconds'])
        if 'mesh_file' in row:
            control_u,control_q=mesh_read(run,safe_path(run,row['mesh_file']),row,data)
            require(np.array_equal(optimizer_final,control_u),'Control final endpoint selection changed')
            require(row['fixed_floor']==original_floor and row['strict_floor']==strict_floor,'Evaluation gate changed')
            require(row['qualified_original']==qualification(control_q,row['audit'],fixed)
                and row['qualified_strict']==qualification(control_q,row['audit'],strict_floor),'Control gates differ')
            pair.update(control_margins_vs_fixed_deg=row['actual_margin_vs_fixed_deg'],
                control_original_qualified=row['qualified_original']['qualified'],control_strict_qualified=row['qualified_strict']['qualified'])
        if row['status']=='success':
            seeds=[base+p['all16_sorted_uids'].index(uid) for base in (1593501,1693501)]
            require([x['seed'] for x in row['metrics']]==[x['seed'] for x in strict_record['metrics']]==seeds,'Paired MC differs')
            require(row['GT_reference_hash']==strict_record['GT_reference_hash']==array_hash(data['ref_vertices'],data['ref_faces'])
                and row['mainfield_hash']==strict_record['mainfield_hash']==array_hash(data['logr']),'Paired GT/mainfield differs')
            control=np.array([x['rms_bbox_percent'] for x in row['metrics']]);strict=np.array([x['rms_bbox_percent'] for x in strict_record['metrics']])
            original=np.array([x['rms_bbox_percent'] for x in reference_teacher['metrics']])
            pair.update(control_RMS_bbox_percent=control.tolist(),strict_RMS_bbox_percent=strict.tolist(),original_teacher_RMS_bbox_percent=original.tolist(),
                strict_gain_vs_control_percent=(100*(1-strict/control)).tolist(),control_gain_vs_original_teacher_percent=(100*(1-control/original)).tolist())
        energy=float(np.square(data['teacher_u']-data['base_u']).mean());pair['zero_teacher_energy']=energy==0
        for trial in range(16):
            seed=1393501+16*p['all_train_uids'].index(uid)+trial;ambient=np.random.default_rng(seed).normal(size=(258,3))
            for fraction in p['fractions']:
                new=new_noise[(uid,trial,fraction)];old=old_noise[(uid,trial,fraction)]
                require(new['seed']==old['seed']==seed and new['original_teacher_energy']==old['original_teacher_energy']==energy,'Paired noise randomization differs')
                require(new['nominal_tangent_component_RMS']==old['nominal_tangent_component_RMS']==fraction*np.sqrt(energy),'Paired absolute noise scale differs')
                require(new['paired_strict_angular_pass']==old['angular_pass'] and new['paired_strict_status']==old['status'],'Stored paired angular result differs')
                require(new['informative_perturbation']==old['informative_perturbation']==(fraction>0 and energy>0),'Degenerate noise treated as evidence')
                for center,record in ((control_u,new),(strict_u,old)):
                    if center is None or record['status']!='success':continue
                    u=reconstruct(center,ambient,energy,fraction);q=named_quality(u*support.query(data['logr'],u)[:,None],data['faces'])
                    margins={k:q[k]-fixed[k] for k in KEYS}
                    require(all(v>=-1e-6 for v in margins.values())==record['angular_pass'],'Reconstructed paired angular pass differs')
                    require(bool(dets(u,data['faces']).min()>0)==record['direction_determinants_positive'],'Direction determinant differs')
                    angle_error=max(abs(margins[k]-record['margins_vs_fixed_deg'][k]) for k in KEYS)
                    mse_error=abs(float(np.square(u-center).mean())-record['actual_direction_MSE_to_endpoint'])
                    require(angle_error<=1e-10 and mse_error<=1e-18,'Independent paired noise reconstruction differs')
                    max_angle_error=max(max_angle_error,angle_error);max_mse_error=max(max_mse_error,mse_error);reconstructed+=1
        for fraction in p['fractions']:
            nn=[new_noise[(uid,t,fraction)] for t in range(16)];oo=[old_noise[(uid,t,fraction)] for t in range(16)]
            noise_summary.append({'uid':uid,'fraction':fraction,'planned':16,
                'control_angular_pass':sum(r['angular_pass'] for r in nn),'strict_angular_pass':sum(r['angular_pass'] for r in oo),
                'control_success':sum(r['status']=='success' for r in nn),'strict_success':sum(r['status']=='success' for r in oo),
                'strict_only_pass':sum(o['angular_pass'] and not n['angular_pass'] for n,o in zip(nn,oo)),
                'control_only_pass':sum(n['angular_pass'] and not o['angular_pass'] for n,o in zip(nn,oo)),
                'informative_perturbation':fraction>0 and energy>0})
        pairs.append(pair)
    for path,digest in bound.items():require(file_hash(path)==digest,'Frozen source changed during paired review: '+path)
    return {'status':'PASS','protocol_sha256':EXPECTED_PROTOCOL,'results_sha256':file_hash(run/'RESULTS.json'),
        'sensitivity_sha256':file_hash(run/'SENSITIVITY.json'),'review_source_sha256':file_hash(__file__),
        'independent_reconstruction_source_sha256':file_hash(Path(__file__).parents[1]/'layout_teacher_margin/review.py'),
        'frozen_bindings_verified':len(bound),'copied_input_files':len(files),'planned_control_calls':4,'planned_paired_noise_slots':448,
        'reconstructed_arm_noise_records':reconstructed,'max_reconstructed_angle_error_deg':max_angle_error,
        'max_reconstructed_direction_mse_error':max_mse_error,'pairs':pairs,'paired_sensitivity':noise_summary,
        'only_floor_policy_differs_in_registered_original_optimizer':True,'no_new_exact_intersection_audit_of_noise':True,
        'no_new_GT_MC_recomputation':True,'no_old_run_or_label_modifications':True,
        'scope':'local floor-policy intervention on four seen parents; not learned network/generalization advantage',
        'elapsed_review_seconds':time.perf_counter()-started}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);args=parser.parse_args()
    report=audit(args.run);write_json(Path(args.run)/'REVIEW.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs','paired_sensitivity')},indent=2),flush=True)


if __name__=='__main__':main()
