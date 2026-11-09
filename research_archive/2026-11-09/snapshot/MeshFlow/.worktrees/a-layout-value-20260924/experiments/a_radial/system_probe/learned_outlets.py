"""New raw learned-A outlets on the immutable original AE3200/FM8000 fields.

This is an exploratory outlet extension. It never replaces the original A or
chooses among seeds, checkpoints, threads, samples, or a Fixed fallback.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from ..exporter import RadialSupport
from ..legacy.networks import DomainVelocity
from ..io_utils import array_hash,file_hash,new_dir,write_json,save_npz
from ..layout_student.deployment import predict_raw
from ..layout_student.train import model_sha256
from ..layout_value.geometry import audit_mesh,evaluate_mesh
from .ours_metrics import (SEEDS,expected_ids,read_json,require,inside,verify_protocol as verify_generation,
    seed_inputs,load_sample,load_mesh,audit_inputs,quality_tails,common_gate,paired_metrics,POINTS,SEED_BASES)
from .warm_start import validate_final

ARMS=('longfit','balanced')


def expected_slots():return {(seed,sid,arm) for seed in SEEDS for sid in expected_ids() for arm in ARMS}


def source_files():
    root=Path(__file__).resolve().parents[1]
    names=['system_probe/learned_outlets.py','system_probe/ours_metrics.py','system_probe/warm_start.py',
        'layout_student/deployment.py','layout_student/train.py','layout_value/geometry.py','exporter.py','models.py',
        'legacy/deform.py','legacy/chart.py','legacy/networks.py','legacy/common.py','legacy/exact_accel.py',
        'legacy/geometry_task.py','mesh_ops.py','io_utils.py','assets/support.npz']
    return {str(root/name):file_hash(root/name) for name in names}


def baseline_map(rows):
    keys=[(r['seed'],r['sample_id'],r['arm']) for r in rows]
    expected={(seed,sid,arm) for seed in SEEDS for sid in expected_ids() for arm in ('fixed','a','qem')}
    require(len(keys)==384 and len(set(keys))==384 and set(keys)==expected,'Original baseline metric slots missing or duplicated')
    return dict(zip(keys,rows))


def validate_baseline_row(row,seed,sid,nodes_hash,reference_hash,mesh_sha):
    index=SEEDS.index(seed)*64+int(sid)
    require(row['metric_status']=='ok' and row['seed']==seed and row['sample_id']==sid,'Unreadable or mismatched baseline row')
    require(row['source_nodes_sha256']==nodes_hash and row['reference_surface_hash']==reference_hash,'Baseline uses another mainfield')
    require(row['mesh_npz_sha256']==mesh_sha,'Baseline metric mesh identity differs')
    require(len(row['surface_metrics'])==2 and [m['seed'] for m in row['surface_metrics']]==[b+index for b in SEED_BASES],
            'Baseline uses different MC seeds')
    require(all(m['points_per_direction']==8192 and np.isfinite(m['rms_bbox_percent']) for m in row['surface_metrics']),
            'Baseline point count or metric differs')


def prepare(generation_run,long_run,balanced_run,out):
    require(not torch.cuda.is_initialized(),'CPU-only preparation')
    generation=Path(generation_run).resolve();gp=read_json(generation/'RUN_PROTOCOL.json');verify_generation(generation,gp)
    original_metrics=generation/'metrics/RESULTS.json';metrics=read_json(original_metrics);baseline=baseline_map(metrics['rows'])
    require(metrics['metric_success_count']==384 and not metrics['provenance_errors'],'Original metrics incomplete')
    audit=audit_inputs(generation,gp);require(audit['status']=='PASS','Original generation/export input audit failed')
    sources={'longfit':Path(long_run).resolve(),'balanced':Path(balanced_run).resolve()};source_protocols={};declarations={}
    bound=source_files();bound[str(generation/'RUN_PROTOCOL.json')]=file_hash(generation/'RUN_PROTOCOL.json')
    bound[str(original_metrics)]=file_hash(original_metrics)
    metadata=read_json(Path(gp['data'])/'DATASET.json');generation_uids=sorted(x['uid'] for x in metadata['rows'])
    for arm,source in sources.items():
        path=source/'RUN_PROTOCOL.json';p=read_json(path);source_protocols[arm]={'path':str(path),'sha256':file_hash(path)}
        require(sorted(p['uids'])==generation_uids and len(p['train_uids'])==12,'Different source object pool/split')
        bound[str(path)]=file_hash(path)
        for seed in SEEDS:
            declarations[f'{arm}_{seed}']={'source_run':str(source),'path':str(source/'training'/f'joint_{seed}'/'final.pt'),
                'kind':arm,'seed':seed,'update':240000,'job':f'joint_{seed}',
                'expected_arm':'c0' if arm=='longfit' else 'c0_parent_weighted'}
    require(read_json(source_protocols['longfit']['path'])['train_uids']==read_json(source_protocols['balanced']['path'])['train_uids'],
            'Learned initializers have different training split')
    support=RadialSupport();seed_bindings={}
    for seed in SEEDS:
        inputs=seed_inputs(generation,seed,gp);samples={}
        for rel in inputs['checkpoint_paths'].values():bound[str(generation/rel)]=file_hash(generation/rel)
        bound[str(inputs['sample_dir']/'SAMPLES.json')]=inputs['sample_manifest_sha256']
        for arm in ('fixed','a','qem'):
            path=inputs['export_dir']/f'{arm}_MANIFEST.json';bound[str(path)]=inputs['outlet_manifest_sha256'][arm]
        for sid in expected_ids():
            arrays,identity=load_sample(inputs,sid);record=inputs['sample_records'][sid]
            path=inside(inputs['sample_dir'],record['path']);bound[str(path)]=identity['sample_npz_sha256']
            reference_hash=array_hash(*support.mesh(arrays['nodes']))
            for oldarm in ('fixed','a','qem'):
                export=inputs['outlet_records'][oldarm][sid];mesh=inside(inputs['export_dir'],export['mesh_path'])
                bound[str(mesh)]=export['mesh_sha256']
                validate_baseline_row(baseline[(seed,sid,oldarm)],seed,sid,record['nodes_hash'],reference_hash,export['mesh_sha256'])
            samples[sid]={**identity,'path':str(path),'array_hash':record['array_hash'],'nodes_hash':record['nodes_hash'],
                'reference_surface_hash':reference_hash}
        seed_bindings[str(seed)]={'sample_manifest_sha256':inputs['sample_manifest_sha256'],
            'AE_FM_checkpoint_sha256':inputs['checkpoint_sha256'],'samples':samples}
    for name,digest in gp['data_hashes'].items():bound[str(Path(gp['data'])/name)]=digest
    out=new_dir(out);write_json(out/'INPUT_AUDIT.json',audit);shutil.copyfile(original_metrics,out/'REFERENCE_METRICS.json')
    protocol={'scope':'EXPLORATORY_RAW_LEARNED_OUTLETS_ON_ORIGINAL_AE3200_FM8000_FIELDS',
        'generation_run':str(generation),'generation_protocol_sha256':file_hash(generation/'RUN_PROTOCOL.json'),
        'original_metrics_sha256':file_hash(original_metrics),'reference_metrics_copy_sha256':file_hash(out/'REFERENCE_METRICS.json'),
        'source_protocols':source_protocols,'model_declarations':declarations,'model_registration':'all four fixed declared240k final paths verified together; no best selection',
        'seeds':list(SEEDS),'sample_ids':expected_ids(),'arms':list(ARMS),'expected_samples':128,'expected_attempts':256,
        'seed_bindings':seed_bindings,'seed_pairing':'AE/FM seed10 uses A seed10; seed11 uses A seed11; no cross-seed choice',
        'primary_CPU_threads':1,'primary_interop_threads':1,'prior_GT_A_deployment_CPU_threads':4,
        'thread_scope':'This generated-input outlet contract uses original system-export threads1; prior GT study used threads4. No selection across environments.',
        'steps':32,'proxy_points':256,'final_accepted_direction_radius_requery':True,'mainfield_nodes':2050,
        'network_inputs':'original proxy256 from saved generated nodes only; no GT, teacher directions, or new latent sampling',
        'points_each_direction':POINTS,'MC_seed_bases':list(SEED_BASES),'MC_seed_rule':'base + seed_index*64 + integer(sample_id)',
        'reference_surface':'same saved generated nodes via RadialSupport.mesh; identical to original ours_metrics',
        'quality_gate':'same actual Fixed minimum angle and per-face-min p05, tolerance1e-6 degrees, exact endpoint valid and actual512',
        'raw_only':True,'Fixed_fallback':False,'original_system_rows_replaced':False,'heldout_class_generalization':False,
        'formal_timing_available':False,'bound_files_sha256':bound}
    write_json(out/'RUN_PROTOCOL.json',protocol);write_json(out/'STATUS.json',{'status':'PREPARED_AWAITING_ALL_FOUR_FINAL_MODEL_REGISTRATION','new_outputs':0})
    return protocol


def verify_contract(run):
    run=Path(run);p=read_json(run/'RUN_PROTOCOL.json')
    require(p['seeds']==list(SEEDS) and p['sample_ids']==expected_ids() and p['arms']==list(ARMS),'Fixed outlet slots changed')
    require(p['primary_CPU_threads']==1 and p['primary_interop_threads']==1 and p['steps']==32 and p['proxy_points']==256,
            'Generated-output deployment contract changed')
    require(p['MC_seed_bases']==list(SEED_BASES) and p['points_each_direction']==8192,'Shared evaluation sampling changed')
    require(p['raw_only'] is True and p['Fixed_fallback'] is False,'Raw-only policy changed')
    require(file_hash(run/'REFERENCE_METRICS.json')==p['reference_metrics_copy_sha256']==p['original_metrics_sha256'],'Reference metrics snapshot changed')
    for path,digest in p['bound_files_sha256'].items():require(file_hash(path)==digest,'Bound original source or input changed: '+path)
    return p


def register_models(run):
    run=Path(run).resolve();p=verify_contract(run);require(not (run/'MODEL_REGISTRY.json').exists(),'Model registry is immutable once written')
    entries={}
    for name,declaration in p['model_declarations'].items():
        protocol,identity=validate_final(declaration['source_run'],declaration['kind'],declaration['seed'])
        require(identity['path']==declaration['path'] and identity['arm']==declaration['expected_arm'],'Initializer path/arm was substituted')
        require(identity['protocol_sha256']==p['source_protocols'][declaration['kind']]['sha256'],'Source protocol changed before binding final')
        entries[name]=identity
    require(set(entries)=={f'{a}_{s}' for a in ARMS for s in SEEDS},'Incomplete all-four final registration')
    value={'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),'models':entries,'fixed_final_240k_only':True,
        'no_quality_or_RMS_selection':True,'source_models_modified':False}
    write_json(run/'MODEL_REGISTRY.json',value)
    write_json(run/'STATUS.json',{'status':'ALL_FOUR_FINAL_MODELS_REGISTERED_EVALUATION_PENDING','new_outputs':0},replace=True)
    return value


def load_models(run,p):
    registry=read_json(run/'MODEL_REGISTRY.json');require(registry['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json'),'Wrong model registry')
    require(set(registry['models'])==set(p['model_declarations']),'Model registry incomplete')
    models={};errors={}
    for name,identity in registry['models'].items():
        try:
            declaration=p['model_declarations'][name];require(identity['path']==declaration['path'],'Model path changed')
            require(file_hash(identity['path'])==identity['file_sha256'],'Registered final checkpoint changed')
            state=torch.load(identity['path'],map_location='cpu',weights_only=False);cfg=state['config']
            require(state['update']==240000 and state['is_predeclared_final'] and cfg['seed']==declaration['seed']
                and cfg['job_id']==declaration['job'] and cfg['arm']==declaration['expected_arm'],'Wrong fixed checkpoint identity')
            require(state['protocol_sha256']==identity['protocol_sha256']==p['source_protocols'][declaration['kind']]['sha256'],'Model protocol differs')
            require(model_sha256(state['model'])==state['model_sha256']==identity['model_sha256'],'Model tensor identity differs')
            model=DomainVelocity().cpu();model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
            require(sum(v.numel() for v in model.parameters())==29907,'Architecture changed')
            models[name]=model;del state;gc.collect()
        except Exception as exc:errors[name]={'error':repr(exc),'traceback':traceback.format_exc()}
    return models,errors,registry


def moments(values):
    if not values:return None
    a=np.asarray(values,np.float64)
    return {'n':len(a),'mean':float(a.mean()),'min':float(a.min()),'p05':float(np.quantile(a,.05)),
        'p25':float(np.quantile(a,.25)),'median':float(np.median(a)),'p75':float(np.quantile(a,.75)),
        'p95':float(np.quantile(a,.95)),'max':float(a.max())}


def verify_raw_geometry(mesh,nodes,support,faces):
    """Validate the actual saved raw candidate without repairing its geometry."""
    radius=support.query(nodes,mesh['u'])
    require(np.array_equal(mesh['r'],radius) and np.array_equal(mesh['vertices'],mesh['u']*radius[:,None]),
            'Stale accepted radius')
    require(np.array_equal(mesh['faces'],faces),'Learned outlet connectivity changed')


def summarize(rows,baseline):
    keys=[(r['seed'],r['sample_id'],r['arm']) for r in rows]
    require(len(keys)==256 and len(set(keys))==256 and set(keys)==expected_slots(),'Missing or duplicated new outlet attempts')
    groups=[]
    for seed in ('pooled',10,11):
        for arm in ARMS:
            selected=[r for r in rows if r['arm']==arm and (seed=='pooled' or r['seed']==seed)]
            paired=[r for r in selected if r.get('paired_available')];good=[r for r in selected if r['metric_status']=='ok']
            for r in paired:require(r['source_nodes_sha256']==baseline[(r['seed'],r['sample_id'],'fixed')]['source_nodes_sha256'],'Pair no longer uses common mainfield')
            groups.append({'seed':seed,'arm':arm,'planned_samples':128 if seed=='pooled' else 64,'attempts':len(selected),
                'metric_success_count':len(good),'metric_failed_count':len(selected)-len(good),
                'exact_valid_count':sum(r.get('audit',{}).get('valid') is True for r in selected),
                'qualified_count':sum(r.get('qualified') is True for r in selected),'paired_count':len(paired),
                'gate_unavailable_count':sum(not r.get('gate_available',False) for r in selected),
                'actual_face_count_histogram':{str(n):sum(r.get('quality',{}).get('face_count')==n for r in selected)
                    for n in sorted({r['quality']['face_count'] for r in selected if 'quality' in r})},
                'improved_both_count':sum(r.get('improved_both',False) for r in paired),
                'qualified_ge2pct_both_count':sum(r.get('qualified_ge2pct_both',False) for r in paired),
                'paired_gain_percent_distribution_two_measurements':[moments([r['rms_gain_percent_two_measurements'][k] for r in paired]) for k in (0,1)],
                'RMS_bbox_percent_distribution_two_measurements':[moments([r['surface_metrics'][k]['rms_bbox_percent'] for r in good]) for k in (0,1)],
                'worse_either_measurement_count':sum(any(x<0 for x in r['rms_gain_percent_two_measurements']) for r in paired),
                'worst5_by_first_measurement':[{'seed':r['seed'],'sample_id':r['sample_id'],'gains':r['rms_gain_percent_two_measurements'],
                    'qualified':r.get('qualified')} for r in sorted(paired,key=lambda x:x['rms_gain_percent_two_measurements'][0])[:5]],
                'quality_min_angle_margin_distribution':moments([r['min_angle_margin_deg'] for r in good if 'min_angle_margin_deg' in r]),
                'quality_p05_margin_distribution':moments([r['p05_angle_margin_deg'] for r in good if 'p05_angle_margin_deg' in r])})
    return {'groups':groups,'attempts':256,'failed_slots':[[r['seed'],r['sample_id'],r['arm']] for r in rows if r['metric_status']!='ok'],
        'raw_only':True,'no_Fixed_fallback':True,'primary_CPU_threads':1,'prior_GT_CPU_threads':4,
        'paired_gains_are_per_sample_ratios_not_ratio_of_medians':True,'two_MC_measurements_not_independent_training_runs':True,
        'original_system_table_replaced':False,'heldout_class_generalization':False,'external_baseline_victory_claim':False}


def run(run):
    run=Path(run).resolve();p=verify_contract(run);require(not (run/'RESULTS.json').exists(),'Do not overwrite new outlet results')
    require(not torch.cuda.is_initialized(),'CPU-only deployment');torch.set_num_threads(1);torch.set_num_interop_threads(1)
    models,model_errors,registry=load_models(run,p);baseline=baseline_map(read_json(run/'REFERENCE_METRICS.json')['rows'])
    generation=Path(p['generation_run']);gp=read_json(generation/'RUN_PROTOCOL.json');support=RadialSupport();rows=[]
    for seed in SEEDS:
        seed_error=None
        try:inputs=seed_inputs(generation,seed,gp)
        except Exception as exc:seed_error=repr(exc)
        for sid in expected_ids():
            sample_error=seed_error;arrays=None;sample_identity={};base=None;faces=None
            try:
                if seed_error:raise RuntimeError(seed_error)
                arrays,sample_identity=load_sample(inputs,sid);old=inputs['outlet_records']['fixed'][sid]
                _,faces,_=load_mesh(inputs,old,'fixed',arrays['nodes'],support)
                with np.load(inside(inputs['export_dir'],old['mesh_path']),allow_pickle=False) as archive:base=archive['u'].copy()
                rv,rf=support.mesh(arrays['nodes'])
            except Exception as exc:sample_error=repr(exc)
            for arm in ARMS:
                name=f'{arm}_{seed}';folder=run/'attempts'/f'seed{seed}'/sid/arm;folder.mkdir(parents=True,exist_ok=False)
                row={'seed':seed,'sample_id':sid,'arm':arm,'model_name':name,'model_identity':registry['models'][name],
                    'metric_status':'failed','qualified':None,'gate_available':False,'raw_only':True,'fallback_to_Fixed':False,
                    'generation_seed_equals_A_seed':True,'primary_CPU_threads':1,'prior_GT_A_deployment_CPU_threads':4,
                    'AE_FM_checkpoint_sha256':p['seed_bindings'][str(seed)]['AE_FM_checkpoint_sha256'],**sample_identity}
                try:
                    if sample_error:raise RuntimeError(sample_error)
                    if name in model_errors:raise RuntimeError(model_errors[name]['error'])
                    mesh,meta=predict_raw(models[name],arrays['nodes'],support,base,faces)
                    path=folder/'mesh.npz';save_npz(path,**mesh)
                    row.update(mesh_file=str(path.relative_to(run)),mesh_npz_sha256=file_hash(path),
                        stored_array_hash=array_hash(mesh['vertices'],mesh['faces']),stored_vertices_dtype=str(mesh['vertices'].dtype),
                        stored_faces_dtype=str(mesh['faces'].dtype),canonical_geometry_hash=array_hash(np.asarray(mesh['vertices'],np.float64),np.asarray(mesh['faces'],np.int64)),
                        geometry_hash_schema='canonical float64 vertices/int64 faces; stored typed hash recorded separately',
                        deployment_metadata_not_formal_timing=meta)
                    verify_raw_geometry(mesh,arrays['nodes'],support,faces)
                    row['final_accepted_radius_requery_exact']=True
                    row['quality']=quality_tails(mesh['vertices'],mesh['faces']);row['audit']=audit_mesh(mesh['vertices'],mesh['faces'],exact=True)
                    fixed=baseline[(seed,sid,'fixed')];row.update(common_gate(row['quality'],row['audit'],fixed['quality'],fixed['audit']))
                    index=SEEDS.index(seed)*64+int(sid);row['reference_surface_hash']=array_hash(rv,rf)
                    row['surface_metrics']=[evaluate_mesh(mesh['vertices'],mesh['faces'],rv,rf,points=8192,seed=b+index) for b in SEED_BASES]
                    row['metric_status']='ok'
                except Exception as exc:row.update(metric_error=repr(exc),metric_traceback=traceback.format_exc())
                row.update(paired_metrics(row,baseline[(seed,sid,'fixed')]))
                row['comparison_to_original_A']=paired_metrics(row,baseline[(seed,sid,'a')])
                write_json(folder/'RESULT.json',row);rows.append(row)
            if (int(sid)+1)%16==0:print(json.dumps({'seed':seed,'samples':int(sid)+1,'attempts':len(rows),'failures':sum(r['metric_status']!='ok' for r in rows)}),flush=True)
    verify_contract(run)
    for identity in registry['models'].values():require(file_hash(identity['path'])==identity['file_sha256'],'Model changed during evaluation')
    result={'rows':rows,'attempt_count':256,'model_errors':model_errors,'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),
        'model_registry_sha256':file_hash(run/'MODEL_REGISTRY.json'),'production_outputs_unchanged':True}
    write_json(run/'RESULTS.json',result)
    for arm in ARMS:write_json(run/f'{arm}_MANIFEST.json',{'row':'exploratory_raw_learned_A_'+arm,'requested_samples':128,
        'records':[r for r in rows if r['arm']==arm],'raw_only':True,'original_system_table_replaced':False})
    value=summarize(rows,baseline);write_json(run/'SUMMARY.json',value)
    lines=['任务对齐A接回原AE/FM主场：新增raw出口探索','',
        'AE3200/FM8000原128次采样及原三出口不变；每生成种子只配对应种子的longfit240k与balanced240k，共256尝试。',
        '主流程使用原生成系统出口的CPU单线程；此前GT上的A部署使用4线程。此处不挑线程、不挑种子、不挑样本。',
        '网络仍只看同一2050节点主场采出的proxy256；原CPU32步，最终接受方向重查半径，禁止Fixed回退。',
        '误差参考、8192点与两组MC种子完全沿用原ours_metrics；相对同场实际Fixed检查角度质量、512面和精确有效性。','',
        '|种子|新出口|计算成功/计划|精确有效|质量合格|RMS逐样本改善中位1/2 %|改善均值1/2 %|合格且双次≥2%|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for g in value['groups']:
        d=g['paired_gain_percent_distribution_two_measurements']
        median=' / '.join('-' if x is None else f"{x['median']:.3f}" for x in d);mean=' / '.join('-' if x is None else f"{x['mean']:.3f}" for x in d)
        lines.append(f"|{g['seed']}|{g['arm']}|{g['metric_success_count']}/{g['planned_samples']}|{g['exact_valid_count']}|{g['qualified_count']}|{median}|{mean}|{g['qualified_ge2pct_both_count']}|")
    lines+=['','SUMMARY保留分位数、均值、最坏5例、失败分母；原A/QEM/Fixed完整已核验指标在REFERENCE_METRICS.json。',
        '这是新增出口探索，不替换原五行或8k系统结果；没有类别泛化或外部baseline胜出资格。未提供正式端到端计时。']
    (run/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(run/'STATUS.json',{'status':'EVALUATION_COMPLETE_FAILURES_RETAINED','attempts':256,'failures':len(value['failed_slots'])},replace=True)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--generation-run',required=True);p.add_argument('--long-run',required=True)
    p.add_argument('--balanced-run',required=True);p.add_argument('--out',required=True)
    for command in ('register','run'):
        q=sub.add_parser(command);q.add_argument('--run',required=True)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.generation_run,a.long_run,a.balanced_run,a.out)
    elif a.command=='register':register_models(a.run)
    else:run(a.run)


if __name__=='__main__':main()

