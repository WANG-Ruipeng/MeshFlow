"""Teacher-start64 control: only the optimization quality floor changes to Fixed."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil,time,traceback
import numpy as np
import torch
from ..io_utils import array_hash,file_hash,new_dir,write_json
from ..exporter import RadialSupport
from ..layout_value.optimizer import optimize_layout
from ..layout_student.deployment import named_quality
from ..layout_student.integrity import safe_path
from ..legacy.chart import dets
from ..layout_teacher_margin import experiment as frozen

UIDS=frozen.UIDS
NAME='teacher_start_original_floor'
SOURCE_PROTOCOL='56556db13672b81c9ae22e5d40db0bb7f72bcae992230c4ec7eae3eff79851aa'


def require(ok,message):
    if not ok: raise ValueError(message)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def floor_for_optimizer(fixed_quality):
    return {'min_angle_deg':float(fixed_quality['min_triangle_angle_deg']),
            'angle_q05_deg':float(fixed_quality['triangle_min_angle_p05_deg'])}


def source_hashes():
    return {**frozen.source_files(),str(Path(__file__).resolve()):file_hash(__file__),
        str(Path(__file__).with_name('__init__.py').resolve()):file_hash(Path(__file__).with_name('__init__.py'))}


def assert_contract(p):
    require(p['schema']=='A-TEACHER-MARGIN-CONTROL-v1' and p['uids']==list(UIDS),'Wrong fixed study')
    require(p['initializer']=='frozen_teacher_only' and p['optimization_margin_deg']==0.0
        and p['paired_reference_margin_deg']==0.1,'Only the floor may differ')
    require(p['max_steps']==64 and p['max_seconds']==3600 and p['snapshot_steps']==[0,16,64]
        and p['device']=='cpu' and p['threads']==4 and p['interop_threads']==1,'Budget/device changed')
    require(p['expected_tasks']==4 and p['expected_new_endpoint_rows']==4
        and p['expected_paired_UIDs']==4 and p['noise_slots']==448,'Planned denominator changed')
    require(p['optimizer_seed_base']==1493501 and p['evaluation_seed_bases']==[1593501,1693501]
        and p['noise_seed_base']==1393501 and p['fractions']==list(frozen.FRACTIONS)
        and p['noise_trials']==16 and p['points_each_direction']==8192,'Original randomization changed')
    require(p['source_protocol_sha256']==SOURCE_PROTOCOL and p['no_GT_selection'] and p['no_old_label_updates']
        and p['no_network_training'] and p['no_fixed_fallback'],'Source/scope changed')


def prepare(source,out):
    source=Path(source).resolve();out=Path(out).resolve()
    require(not out.exists() and not torch.cuda.is_initialized(),'Fresh CPU-only preparation required')
    source_p=frozen.verify_protocol(source)
    require(file_hash(source/'RUN_PROTOCOL.json')==SOURCE_PROTOCOL,'Wrong strict experiment')
    review=read(source/'REVIEW.json')
    require(review['status']=='PASS' and review['results_sha256']==file_hash(source/'RESULTS.json')
        and review['sensitivity_sha256']==file_hash(source/'SENSITIVITY.json'),'Strict readback missing or changed')
    result=read(source/'RESULTS.json');source_rows={(r['uid'],r['endpoint']):r for r in result['rows']}
    bound={**source_p['bound_source_sha256'],**source_p['source_sha256']}
    copies={f'data/{uid}.npz':source/'data'/f'{uid}.npz' for uid in UIDS}
    copies.update({f'source_evidence/{name}':source/name for name in ['RUN_PROTOCOL.json','DATA_MANIFEST.json','RESULTS.json','SUMMARY.json','SENSITIVITY.json','REVIEW.json']})
    references={};support=RadialSupport();floors={}
    for uid in UIDS:
        data=frozen.read_data(source/'data'/f'{uid}.npz')
        q=named_quality(data['base_u']*support.query(data['logr'],data['base_u'])[:,None],data['faces'])
        floors[uid]={key:q[key] for key in frozen.ANGLES};references[uid]={}
        for name in ('fixed','teacher','opt_teacher'):
            row=source_rows[(uid,name)];row_path=source/'endpoints'/uid/name/'RESULT.json';mesh_path=safe_path(source,row['mesh_file'])
            require(row==read(row_path) and row['status']=='success' and file_hash(mesh_path)==row['mesh_npz_sha256'],
                'Frozen reference failed identity: '+uid+'/'+name)
            if name=='opt_teacher':
                task=next(t for t in result['tasks'] if t['uid']==uid and t['initializer']=='teacher')
                require(task['stats']['iterations']==64 and task['stats']['max_steps']==64
                    and task['stats']['training_seed']==1493501+source_p['all16_sorted_uids'].index(uid),'Strict matching budget/seed differs')
            rel=f'references/{uid}/{name}';copies[rel+'/RESULT.json']=row_path;copies[rel+'/mesh.npz']=mesh_path
            references[uid][name]={'row':rel+'/RESULT.json','mesh':rel+'/mesh.npz',
                'original_row_sha256':file_hash(row_path),'original_mesh_sha256':file_hash(mesh_path)}
    for path in copies.values(): bound[str(path)]=file_hash(path)
    impl=source_hashes();out=new_dir(out);files={}
    for relative,source_path in copies.items():
        dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source_path,dest)
        files[relative]=file_hash(dest);require(files[relative]==bound[str(source_path)],'Copied input changed')
    write_json(out/'DATA_MANIFEST.json',{'files':files})
    p={'schema':'A-TEACHER-MARGIN-CONTROL-v1','uids':list(UIDS),'all16_sorted_uids':source_p['all16_sorted_uids'],
        'all_train_uids':source_p['all_train_uids'],'initializer':'frozen_teacher_only','optimization_margin_deg':0.0,
        'paired_reference_margin_deg':0.1,'max_steps':64,'max_seconds':3600,'snapshot_steps':[0,16,64],
        'device':'cpu','threads':4,'interop_threads':1,'expected_tasks':4,'expected_new_endpoint_rows':4,
        'expected_paired_UIDs':4,'reference_endpoint_rows':12,'optimizer_seed_base':1493501,
        'evaluation_seed_bases':[1593501,1693501],'points_each_direction':8192,'noise_seed_base':1393501,
        'fractions':list(frozen.FRACTIONS),'noise_trials':16,'noise_slots':448,
        'source_run':str(source),'source_protocol_sha256':SOURCE_PROTOCOL,'reference_rows':references,'fixed_floors':floors,
        'parent_evidence':source_p['parent_evidence'],'source_sha256':impl,'bound_source_sha256':bound,
        'data_manifest_sha256':file_hash(out/'DATA_MANIFEST.json'),
        'causal_difference':'optimization floor only: original Fixed instead of Fixed+0.1deg; all other call arguments same',
        'output_selection':'unchanged optimizer best accepted feasible internal loss; failed restoration retained',
        'quality_reporting':'both original Fixed and Fixed+0.1 thresholds reported; neither uses GT',
        'noise':'same original-teacher energy, same ambient Gaussian and fraction grid as paired strict endpoint',
        'zero_teacher':'zero amplitude, not informative robustness',
        'GT_reference':'original GT triangles only in posthoc endpoint MC, same paired seeds; no selection',
        'inherited_reference_MC_not_new_independent_replicates':True,'no_GT_selection':True,
        'no_old_label_updates':True,'no_network_training':True,'no_fixed_fallback':True,
        'scope':'four seen training parents; local optimizer floor intervention, not learned generalization',
        'timing_scope':'CPU4 potentially concurrent with other experiments; not isolated benchmark',
        'cost_estimate_optimizer_seconds_from_completed_strict_teacher_arm':sum(next(t for t in result['tasks']
            if t['uid']==uid and t['initializer']=='teacher')['optimizer_call_seconds'] for uid in UIDS)}
    assert_contract(p);write_json(out/'RUN_PROTOCOL.json',p);verify(out);return p


def verify(run):
    run=Path(run).resolve();p=read(run/'RUN_PROTOCOL.json');assert_contract(p)
    require(file_hash(run/'DATA_MANIFEST.json')==p['data_manifest_sha256'],'Control input manifest changed')
    for relative,digest in read(run/'DATA_MANIFEST.json')['files'].items():
        require(file_hash(safe_path(run,relative))==digest,'Copied control input changed: '+relative)
    for path,digest in {**p['bound_source_sha256'],**p['source_sha256']}.items():
        require(file_hash(path)==digest,'Frozen control/source changed: '+path)
    return p


def task(run,p,uid,data,support,quality):
    folder=new_dir(run/'attempts'/uid);info={'uid':uid,'status':'error','initializer':'teacher',
        'optimizer_floor':floor_for_optimizer(quality),'optimization_margin_deg':0.,
        'seed':1493501+p['all16_sorted_uids'].index(uid),'no_fixed_fallback':True};row=None
    try:
        started=time.perf_counter()
        result=optimize_layout(data['logr'],data['teacher_u'],data['faces'],floor_for_optimizer(quality),
            device='cpu',max_steps=64,max_seconds=3600,seed=info['seed'],snapshot_steps=(0,16,64))
        info['optimizer_call_seconds']=time.perf_counter()-started
        frozen.serialize_optimizer(result,folder);requested,extras=frozen.snapshot_records(result,(0,16,64))
        info.update(status='optimizer_call_completed',stats=result['stats'],restore_failed=result['stats']['best_feasible_iteration'] is None,
            requested_snapshots=[{k:v for k,v in x.items() if k!='snapshot'} for x in requested],
            extra_terminal_snapshots=[x['iteration'] for x in extras])
        row=frozen.endpoint_row(run,p,uid,NAME,result['u'],data,support,quality,result['vertices'])
    except Exception as exc:info.update(error=repr(exc),traceback=traceback.format_exc())
    if row is None:row=frozen.unavailable_row(run,uid,NAME,info.get('error','No returned endpoint'))
    info['endpoint_record']=row;write_json(folder/'TASK.json',info);return info,row


def neighborhood(run,p,rows):
    support=RadialSupport();output=[];lookup={row['uid']:row for row in rows}
    original_noise=read(run/'source_evidence/SENSITIVITY.json')['rows']
    strict_lookup={(r['uid'],r['trial'],r['fraction']):r for r in original_noise if r['endpoint']=='opt_teacher'}
    for uid in UIDS:
        data=frozen.read_data(run/'data'/f'{uid}.npz');record=lookup[uid];center=None;error=None
        try:
            require('mesh_file' in record,'Control endpoint unavailable');path=safe_path(run,record['mesh_file'])
            require(file_hash(path)==record['mesh_npz_sha256'],'Control endpoint changed')
            with np.load(path,allow_pickle=False) as z:center=np.array(z['u'])
        except Exception as exc:error=repr(exc)
        energy=float(np.square(data['teacher_u']-data['base_u']).mean());floor=p['fixed_floors'][uid]
        for trial in range(16):
            seed=1393501+16*p['all_train_uids'].index(uid)+trial;ambient=np.random.default_rng(seed).normal(size=(258,3))
            for fraction in frozen.FRACTIONS:
                strict=strict_lookup[(uid,trial,fraction)]
                require(strict['seed']==seed and strict['original_teacher_energy']==energy,'Paired strict randomization differs')
                row={'uid':uid,'trial':trial,'fraction':fraction,'seed':seed,'status':'unavailable' if center is None else 'error',
                    'original_teacher_energy':energy,'nominal_tangent_component_RMS':fraction*np.sqrt(energy),
                    'informative_perturbation':energy>0 and fraction>0,'zero_teacher_noise_degenerate':energy==0,
                    'angular_pass':False,'paired_strict_angular_pass':strict['angular_pass'],'paired_strict_status':strict['status']}
                if center is None:row['error']=error
                else:
                    try:
                        u=frozen.perturb(center,ambient,energy,fraction);q=named_quality(u*support.query(data['logr'],u)[:,None],data['faces'])
                        margin={k:q[k]-floor[k] for k in frozen.ANGLES}
                        row.update(status='success',angular_pass=all(v>=-1e-6 for v in margin.values()),margins_vs_fixed_deg=margin,
                            direction_determinants_positive=bool(dets(u,data['faces']).min()>0),
                            actual_direction_MSE_to_endpoint=float(np.square(u-center).mean()))
                    except Exception as exc:row.update(error=repr(exc))
                output.append(row)
    require(len(output)==448 and len({(r['uid'],r['trial'],r['fraction']) for r in output})==448,'Noise denominator changed')
    write_json(run/'SENSITIVITY.json',{'rows':output,'planned':448,'same_random_noise_and_original_energy_as_strict':True,
        'zero_energy_is_not_robustness':True,'noise_not_learned_error_no_exact_intersection_audit':True})
    return output


def run(run):
    run=Path(run).resolve();p=verify(run)
    require(not torch.cuda.is_initialized(),'CPU only')
    require(not any((run/n).exists() for n in ['attempts','endpoints','RESULTS.json','SENSITIVITY.json']),'No overwrite or retries')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);support=RadialSupport();tasks=[];rows=[]
    for uid in UIDS:
        data=frozen.read_data(run/'data'/f'{uid}.npz')
        q=named_quality(data['base_u']*support.query(data['logr'],data['base_u'])[:,None],data['faces'])
        require({key:q[key] for key in frozen.ANGLES}==p['fixed_floors'][uid],'Actual Fixed floor changed')
        info,row=task(run,p,uid,data,support,q);tasks.append(info);rows.append(row)
        print(json.dumps({'uid':uid,'status':info['status'],'actual_steps':info.get('stats',{}).get('iterations')}),flush=True)
    verify(run);require(len(rows)==4 and {row['uid'] for row in rows}==set(UIDS),'Missing control endpoint')
    write_json(run/'RESULTS.json',{'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),'rows':rows,'tasks':tasks,'no_dropped_slots':True})
    noise=neighborhood(run,p,rows);paired=[]
    for row,info in zip(rows,tasks):
        refs={name:read(safe_path(run,paths['row'])) for name,paths in p['reference_rows'][row['uid']].items()}
        paired.append({'uid':row['uid'],'status':row['status'],'actual_steps':info.get('stats',{}).get('iterations'),
            'restoration_failed':info.get('restore_failed'),'control_margins_vs_fixed_deg':row.get('actual_margin_vs_fixed_deg'),
            'strict_margins_vs_fixed_deg':refs['opt_teacher']['actual_margin_vs_fixed_deg'],
            'control_gain_vs_strict_percent':frozen.paired_gain(row,refs['opt_teacher']),
            'control_gain_vs_original_teacher_percent':frozen.paired_gain(row,refs['teacher']),
            'control_original_qualified':row.get('qualified_original',{}).get('qualified'),
            'control_strict_qualified':row.get('qualified_strict',{}).get('qualified'),
            'optimizer_call_seconds':info.get('optimizer_call_seconds')})
    report={'planned_calls':4,'observed_calls':len(tasks),'planned_endpoint_rows':4,'successful_endpoint_rows':sum(r['status']=='success' for r in rows),
        'noise_slots':448,'noise_success':sum(r['status']=='success' for r in noise),'paired':paired,
        'only_floor_differs_between_matched_teacher64_arms':True,'no_GT_selection_or_old_label_updates':True,
        'no_global_optimality_or_generalization_claim':True,'timing_not_isolated':True,
        'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),'results_sha256':file_hash(run/'RESULTS.json'),
        'sensitivity_sha256':file_hash(run/'SENSITIVITY.json')}
    verify(run);write_json(run/'SUMMARY.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--source-run',required=True);prep.add_argument('--out',required=True)
    execute=sub.add_parser('run');execute.add_argument('--run',required=True);args=parser.parse_args()
    if args.command=='prepare':
        p=prepare(args.source_run,args.out)
        print(json.dumps({'prepared':str(Path(args.out).resolve()),'protocol_sha256':file_hash(Path(args.out)/'RUN_PROTOCOL.json'),
            'tasks':4,'new_endpoint_rows':4,'inherited_reference_rows':12,'paired_noise_slots':448,
            'optimizer_cost_projection_seconds':p['cost_estimate_optimizer_seconds_from_completed_strict_teacher_arm']},indent=2),flush=True)
    else:print(json.dumps(run(args.run),indent=2),flush=True)


if __name__=='__main__':main()
