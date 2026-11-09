"""Independent readback of the fixed stricter-teacher experiment; never optimizes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from ..exporter import RadialSupport
from ..io_utils import array_hash, file_hash, write_json
from ..layout_student.deployment import named_quality, qualification
from ..layout_student.integrity import mesh_identity, safe_path
from ..legacy.chart import dets

EXPECTED_PROTOCOL_SHA256 = '56556db13672b81c9ae22e5d40db0bb7f72bcae992230c4ec7eae3eff79851aa'
UIDS = ('7f1be3e7b6adeeae9bb399e812556da2','2379915c82c2b2d02530f02db7e9157e',
        '8c919e992db7cb081e5b68ee10358ea3','736820a85063a4371e5b68ee10358ea3')
NAMES = ('fixed','teacher','opt_fixed','opt_teacher')
SENSITIVE_NAMES = ('teacher','opt_fixed','opt_teacher')
KEYS = ('min_triangle_angle_deg','triangle_min_angle_p05_deg')
FRACTIONS = (0.,.0001,.001,.01,.1,.5,1.)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def require(value, message):
    if not value: raise ValueError(message)


def reconstruct(center, ambient, energy, fraction):
    """Independent direct formula; does not call experiment.perturb."""
    if not fraction or not energy: return center.copy()
    tangent = ambient - (ambient * center).sum(axis=1)[:,None] * center
    tangent /= np.sqrt((tangent * tangent).mean())
    displaced = center + fraction * np.sqrt(energy) * tangent
    return displaced / np.sqrt((displaced * displaced).sum(axis=1))[:,None]


def audit(run):
    started=time.perf_counter();run=Path(run).resolve()
    require(not torch.cuda.is_initialized(),'CPU-only independent review')
    require(file_hash(run/'RUN_PROTOCOL.json')==EXPECTED_PROTOCOL_SHA256,'Unexpected predeclared protocol')
    p=read(run/'RUN_PROTOCOL.json');require(p['uids']==list(UIDS),'UIDs changed')
    require(p['margin_deg']==.1 and p['max_steps']==64 and p['max_seconds']==3600,'Budgets changed')
    require(p['threads']==4 and p['interop_threads']==1,'CPU4 execution contract changed')
    require(p['expected_tasks']==8 and p['expected_endpoint_rows']==16 and p['sensitivity']['planned_slots']==1344,'Denominator changed')
    require(file_hash(run/'DATA_MANIFEST.json')==p['data_manifest_sha256'],'Data manifest changed')
    bound={**p['bound_source_sha256'],**p['source_sha256']}
    for path,digest in bound.items(): require(file_hash(path)==digest,'Frozen source changed: '+path)
    for relative,digest in read(run/'DATA_MANIFEST.json')['files'].items():
        require(file_hash(safe_path(run,relative))==digest,'Prepared input changed: '+relative)
    results=read(run/'RESULTS.json');sensitivity=read(run/'SENSITIVITY.json');summary=read(run/'SUMMARY.json')
    require(results['protocol_sha256']==sensitivity['protocol_sha256']==EXPECTED_PROTOCOL_SHA256,'Ledger protocol differs')
    require(summary['results_sha256']==file_hash(run/'RESULTS.json') and summary['sensitivity_sha256']==file_hash(run/'SENSITIVITY.json'),'Summary binding changed')
    rows=results['rows'];lookup={(r['uid'],r['endpoint']):r for r in rows}
    require(len(rows)==len(lookup)==16 and set(lookup)=={(u,n) for u in UIDS for n in NAMES},'Endpoint slots incomplete')
    tasks={(t['uid'],t['initializer']):t for t in results['tasks']}
    require(len(results['tasks'])==len(tasks)==8 and set(tasks)=={(u,n) for u in UIDS for n in ('fixed','teacher')},'Task slots incomplete')
    expected_noise={(u,n,t,f) for u in UIDS for n in SENSITIVE_NAMES for t in range(16) for f in FRACTIONS}
    noise_lookup={(r['uid'],r['endpoint'],r['trial'],r['fraction']):r for r in sensitivity['rows']}
    require(len(sensitivity['rows'])==len(noise_lookup)==1344 and set(noise_lookup)==expected_noise,'Noise slots incomplete')
    support=RadialSupport();centers={};floor_by_uid={};data_by_uid={};endpoint_checks=[];task_checks=[]
    max_angle_error=0.;max_mse_error=0.;noise_verified=0;missing_noise=0
    for uid in UIDS:
        path=run/'data'/f'{uid}.npz'
        with np.load(path,allow_pickle=False) as z: data={k:np.array(z[k]) for k in z.files}
        data_by_uid[uid]=data
        base=data['base_u'];faces=data['faces'];y=data['logr']
        fq=named_quality(base*support.query(y,base)[:,None],faces);floor_by_uid[uid]=fq
        strict={k:fq[k]+.1 for k in KEYS}
        energy=float(np.square(data['teacher_u']-base).mean())
        require(energy==p['parent_evidence'][uid]['original_teacher_energy'],'Original noise energy changed')
        for name in NAMES:
            row=lookup[(uid,name)];disk=read(run/'endpoints'/uid/name/'RESULT.json')
            require(row==disk,'Endpoint ledger and file disagree')
            checked={'uid':uid,'endpoint':name,'status':row['status'],'saved_mesh':bool(row.get('mesh_file'))}
            if 'mesh_file' in row:
                mesh=safe_path(run,row['mesh_file']);identity=mesh_identity(mesh,faces)
                require(all(row[k]==v for k,v in identity.items()),'Typed mesh identity differs')
                with np.load(mesh,allow_pickle=False) as z:
                    u=np.array(z['u']);r=support.query(y,u);v=u*r[:,None]
                    require(np.array_equal(z['r'],r) and np.array_equal(z['vertices'],v),'Accepted radius differs')
                centers[(uid,name)]=u
                if name=='fixed': require(np.array_equal(u,base),'Fixed was changed')
                if name=='teacher': require(np.array_equal(u,data['teacher_u']),'Frozen teacher was changed')
                q=named_quality(v,faces)
                if 'quality' in row:
                    require(q==row['quality'],'Actual saved quality differs')
                    require(row['fixed_floor']=={k:fq[k] for k in KEYS} and row['strict_floor']==strict,'Saved threshold is not Fixed plus0.1')
                    require(row['qualified_original']==qualification(q,row['audit'],fq)
                        and row['qualified_strict']==qualification(q,row['audit'],strict),'Qualification differs')
                if row['status']=='success':
                    seeds=[base_seed+p['all16_sorted_uids'].index(uid) for base_seed in (1593501,1693501)]
                    require([r['seed'] for r in row['metrics']]==seeds,'Posthoc MC seed changed')
                    require(row['GT_reference_hash']==array_hash(data['ref_vertices'],data['ref_faces']),'GT reference changed')
                    require(row['data_file_sha256']==file_hash(path) and row['mainfield_hash']==array_hash(y),'Field identity differs')
                checked.update(accepted_radius_exact=True,quality_exact='quality' in row,logical_storage_file_hashes_verified=True)
            endpoint_checks.append(checked)
        for initial in ('fixed','teacher'):
            task=tasks[(uid,initial)];folder=run/'attempts'/uid/initial
            require(task==read(folder/'TASK.json'),'Task ledger/file differs')
            require(task['seed']==1493501+p['all16_sorted_uids'].index(uid),'Optimizer seed mismatch')
            item={'uid':uid,'initializer':initial,'status':task['status']}
            if 'stats' in task:
                ledger=read(folder/'OPTIMIZER.json');stats=ledger['stats']
                require(stats==task['stats'],'Optimizer stats disagree')
                require(stats['quality_floor']=={'min_angle_deg':strict[KEYS[0]],'angle_q05_deg':strict[KEYS[1]]},'Optimizer threshold changed')
                require(stats['max_steps']==64 and stats['max_seconds']==3600 and stats['device']=='cpu'
                    and stats['dtype']=='float64' and stats['quality_tolerance_deg']==1e-6,'Optimizer execution contract differs')
                require(stats['training_seed']==task['seed'] and stats['training_reference_samples']==2048
                    and stats['reverse_quadrature_points_per_face']==3,'Internal reference sampling differs')
                require(stats['selection']=='minimum_training_loss_among_accepted_feasible_iterates','Output selection changed')
                require(file_hash(folder/'OPTIMIZER_ARRAYS.npz')==ledger['arrays_sha256'],'Optimizer arrays changed')
                with np.load(folder/'OPTIMIZER_ARRAYS.npz',allow_pickle=False) as z:
                    expected=data['base_u'] if initial=='fixed' else data['teacher_u']
                    require(np.array_equal(z['snapshot_0_u'],expected),'Wrong original start')
                    final=np.array(z['final_u']);radius=support.query(y,final)
                    require(np.array_equal(z['final_vertices'],final*radius[:,None]),'Optimizer chose stale radius')
                    if (uid,'opt_'+initial) in centers:
                        require(np.array_equal(final,centers[(uid,'opt_'+initial)]),'Final endpoint selection changed')
                require(len(ledger['trace'])==stats['iterations']+1 and stats['iterations']<=64,'Actual iteration ledger incomplete')
                require(task['restore_failed']==(stats['best_feasible_iteration'] is None),'Restoration failure hidden')
                item.update(actual_steps=stats['iterations'],restoration_failed=task['restore_failed'],selection_and_radius_verified=True,
                    optimizer_seconds=task['optimizer_call_seconds'])
            task_checks.append(item)
        for name in SENSITIVE_NAMES:
            center=centers.get((uid,name))
            for trial in range(16):
                seed=1393501+16*p['all_train_uids'].index(uid)+trial
                ambient=np.random.default_rng(seed).normal(size=(258,3))
                for fraction in FRACTIONS:
                    record=noise_lookup[(uid,name,trial,fraction)]
                    require(record['seed']==seed and record['original_teacher_energy']==energy,'Noise seed or common energy differs')
                    require(record['nominal_tangent_component_RMS']==fraction*np.sqrt(energy),'Noise scale is not shared original energy')
                    require(record['zero_teacher_noise_degenerate']==(energy==0)
                        and record['informative_perturbation']==(energy>0 and fraction>0),'Zero-noise informative flag wrong')
                    if center is None or record['status']!='success':
                        missing_noise+=1;continue
                    u=reconstruct(center,ambient,energy,fraction);q=named_quality(u*support.query(y,u)[:,None],faces)
                    margins={k:q[k]-fq[k] for k in KEYS};decision=all(v>=-1e-6 for v in margins.values())
                    require(decision==record['angular_pass'],'Independent angle pass decision differs')
                    require(bool(dets(u,faces).min()>0)==record['direction_determinants_positive'],'Direction determinant differs')
                    err=max(abs(margins[k]-record['margins_vs_fixed_deg'][k]) for k in KEYS)
                    mse_error=abs(float(np.square(u-center).mean())-record['actual_direction_MSE_to_endpoint'])
                    max_angle_error=max(max_angle_error,err);max_mse_error=max(max_mse_error,mse_error)
                    require(err<=1e-10 and mse_error<=1e-18,'Independent noise reconstruction changed')
                    if energy==0: require(np.array_equal(u,center),'Stationary control moved')
                    noise_verified+=1
    for row in sensitivity['per_parent']:
        informative=row['informative_perturbation']
        require((row['angular_pass_fraction_informative'] is not None)==informative,'Degenerate noise has informative pass ratio')
    for path,digest in bound.items(): require(file_hash(path)==digest,'Source changed during independent review: '+path)
    return {'status':'PASS','protocol_sha256':EXPECTED_PROTOCOL_SHA256,'results_sha256':file_hash(run/'RESULTS.json'),
        'sensitivity_sha256':file_hash(run/'SENSITIVITY.json'),'source_sha256':file_hash(__file__),
        'source_bindings_unchanged':len(bound),'copied_input_files':len(read(run/'DATA_MANIFEST.json')['files']),
        'endpoint_slots':16,'task_slots':8,'sensitivity_slots':1344,'endpoint_checks':endpoint_checks,'task_checks':task_checks,
        'independently_reconstructed_noise_success_slots':noise_verified,'missing_or_error_noise_slots_retained':missing_noise,
        'max_reconstructed_angle_error_deg':max_angle_error,'max_reconstructed_direction_mse_error':max_mse_error,
        'same_original_teacher_noise_scale_across_all_endpoints':True,'zero_noise_not_a_robustness_result':True,
        'optimizer_gets_only_logr_initial_u_faces_floor_and_fixed_budget':True,
        'GT_posthoc_only_evidence':'Frozen source call passes only logr, selected start, faces and fixed quality floor; MC pairing/data hashes independently checked; source lightweight test verifies argument isolation.',
        'no_new_exact_intersection_certificate_for_noise':True,'no_new_GT_MC_recomputation':True,
        'source_exact_endpoint_audits_preserved_not_reexecuted':True,'old_training_labels_unchanged':True,
        'timing_scope':'CPU4 concurrent with other CPU/GPU experiments; not isolated timing',
        'elapsed_independent_audit_seconds':time.perf_counter()-started}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);args=parser.parse_args()
    result=audit(args.run);write_json(Path(args.run)/'REVIEW.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('endpoint_checks','task_checks')},indent=2),flush=True)


if __name__=='__main__': main()
