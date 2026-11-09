"""Explicit learned-initialization + unchanged layout optimizer CPU diagnostic.

This is never a pure-network result. Raw, requested snapshots, terminal snapshots,
and the optimizer's own best-feasible output have separate immutable records.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from ..exporter import RadialSupport
from ..legacy.geometry_task import fibonacci,hull_faces
from ..legacy.networks import DomainVelocity
from ..io_utils import array_hash,file_hash,new_dir,write_json,save_npz
from ..layout_value.optimizer import optimize_layout
from ..layout_value.geometry import mesh_quality,audit_mesh,evaluate_mesh
from ..layout_student.deployment import make_fixed,predict_raw,named_quality,qualification
from ..layout_student.integrity import mesh_identity
from ..layout_student.train import model_sha256
from ..layout_student.evaluate import metric_cache

UIDS=('7f1be3e7b6adeeae9bb399e812556da2','2379915c82c2b2d02530f02db7e9157e',
      '8c919e992db7cb081e5b68ee10358ea3','736820a85063a4371e5b68ee10358ea3')
MODELS=('fixed','longfit_10','longfit_11','balanced_10','balanced_11')
SNAPSHOTS=(0,1,4,16)
MAX_STEPS=16
MAX_SECONDS=3600
SEED_BASES=(1593501,1693501)
OPT_SEED_BASE=1493501
POINTS=8192
ROW_KINDS=('raw','snapshot_0','snapshot_1','snapshot_4','snapshot_16','optimizer_final')


def require(ok,message):
    if not ok:raise ValueError(message)


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_files():
    root=Path(__file__).resolve().parents[1]
    names=['system_probe/warm_start.py','layout_value/optimizer.py','layout_value/geometry.py',
        'layout_student/deployment.py','layout_student/integrity.py','layout_student/evaluate.py',
        'layout_student/train.py','legacy/deform.py','legacy/chart.py','legacy/networks.py',
        'legacy/common.py','legacy/exact_accel.py','legacy/geometry_task.py','mesh_ops.py','exporter.py','io_utils.py','assets/support.npz']
    return {str(root/name):file_hash(root/name) for name in names}


def validate_final(source,kind,seed):
    source=Path(source).resolve();p=read_json(source/'RUN_PROTOCOL.json');name=f'joint_{seed}'
    path=source/'training'/name/'final.pt';before=file_hash(path)
    state=torch.load(path,map_location='cpu',weights_only=False);cfg=state['config']
    report_path=path.with_name('TRAIN_REPORT.json');report=read_json(report_path)
    arm='c0' if kind=='longfit' else 'c0_parent_weighted'
    require(state['update']==cfg['total_updates']==report['update']==240000 and state['is_predeclared_final'] is True,
            'Only fixed240k final checkpoints may initialize')
    require(state['job_id']==cfg['job']==cfg['job_id']==name and cfg['seed']==seed and cfg['arm']==arm,
            'Initializer job/seed/objective differs')
    require(state['protocol_sha256']==cfg['protocol_sha256']==report['protocol_sha256']==file_hash(source/'RUN_PROTOCOL.json'),
            'Initializer protocol binding differs')
    require(report['status']=='COMPLETE' and report['checkpoint_sha256']==before,'Source final training incomplete or changed')
    require(cfg['active_training_uids']==p['train_uids'] and cfg['parameter_count']==29907 and not cfg['network_extra_inputs'],
            'Initializer architecture/parents/input contract changed')
    require(state['training_labels_sha256']==p['hashes']['training_labels_sha256'],'Initializer data differs')
    mh=model_sha256(state['model']);require(mh==state['model_sha256']==report['model_sha256'],'Initializer tensor hash differs')
    require(file_hash(path)==before,'Initializer changed while read')
    return p,{'path':str(path),'file_sha256':before,'model_sha256':mh,'update':240000,'arm':arm,'seed':seed,
        'job':name,'protocol_path':str(source/'RUN_PROTOCOL.json'),'protocol_sha256':file_hash(source/'RUN_PROTOCOL.json'),
        'training_report_path':str(report_path),'training_report_sha256':file_hash(report_path)}


def prepare(long_run,balanced_run,out):
    require(not torch.cuda.is_initialized(),'CPU-only preparation')
    sources={'longfit':Path(long_run).resolve(),'balanced':Path(balanced_run).resolve()};models={};protocols={}
    bound=source_files()
    for kind,source in sources.items():
        for seed in (10,11):
            p,identity=validate_final(source,kind,seed);models[f'{kind}_{seed}']=identity;protocols[kind]=p
            for name,digest in [('path','file_sha256'),('protocol_path','protocol_sha256'),('training_report_path','training_report_sha256')]:
                bound[identity[name]]=identity[digest]
    longp,balancedp=protocols['longfit'],protocols['balanced']
    require(longp['uids']==balancedp['uids'] and longp['train_uids']==balancedp['train_uids'],'Parent split mismatch')
    require(len(longp['uids'])==16 and set(UIDS)<=set(longp['train_uids']),'Fixed four UID set is not the registered training subset')
    require(longp['hashes']['training_labels_sha256']==balancedp['hashes']['training_labels_sha256'],'Training archive differs')
    data={}
    for kind,source in sources.items():
        manifest_path=source/'DATA_MANIFEST.json';require(file_hash(manifest_path)==protocols[kind]['hashes']['data_manifest_sha256'],'Source manifest changed')
        manifest=read_json(manifest_path)['files'];bound[str(manifest_path)]=file_hash(manifest_path)
        for uid in UIDS:
            path=source/'data'/f'{uid}.npz';digest=file_hash(path)
            require(digest==manifest[f'data/{uid}.npz'],'Selected data file changed')
            bound[str(path)]=digest
            if kind=='longfit':data[uid]={'path':str(path),'sha256':digest}
            else:require(digest==data[uid]['sha256'],'Initializers see different mainfields/data')
    # No output directory is created until all four fixed final checkpoints exist and verify.
    out=new_dir(out);p={'scope':'FOUR_TRAIN_PARENT_INITIALIZATION_PLUS_ORIGINAL_OPTIMIZER_DIAGNOSTIC_NOT_PURE_A',
        'uids':list(UIDS),'all16_sorted_uids':sorted(longp['uids']),'models':list(MODELS),'model_bindings':models,'data':data,
        'expected_tasks':20,'row_kinds':list(ROW_KINDS),'expected_primary_rows':120,
        'max_steps':MAX_STEPS,'max_seconds':MAX_SECONDS,'snapshot_steps':list(SNAPSHOTS),'device':'cpu','threads':1,
        'optimizer_seed_base':OPT_SEED_BASE,'optimizer_seed_rule':'1493501 + original sorted16 UID index; same within each UID across all initializers',
        'evaluation_seed_bases':list(SEED_BASES),'points_each_direction':POINTS,
        'surface_reference':'original GT triangles from frozen longfit ref_vertices/ref_faces; offline evaluation only',
        'optimizer_surface_reference':'same saved nodal mainfield via original RadialSupport.mesh; optimizer never receives GT triangles',
        'network_inputs':'original proxy256 from same saved field only; no teacher or GT geometry',
        'optimizer_implementation':'unmodified layout_value.optimizer.optimize_layout',
        'optimizer_internal_objective':'original 2048 fixed reference MC plus three-point output-face quadrature; distinct evaluation seeds',
        'quality_floor':'actual Fixed min angle and per-triangle min-angle p05 for same field; unchanged1e-6 degree tolerance',
        'output_selection':'original minimum internal loss among accepted feasible iterates; if none original last iterate remains infeasible',
        'Fixed_fallback_inserted':False,'RMS_or_check_used_for_selection':False,'checkpoint_selection':'all four fixed240k finals; no best model selection',
        'max_seconds_is_safety_cap_not_equal_measured_runtime':True,'termination_and_missing_snapshots_retained':True,
        'source_protocols':{k:str(v/'RUN_PROTOCOL.json') for k,v in sources.items()},'bound_files_sha256':bound}
    write_json(out/'RUN_PROTOCOL.json',p);return p


def verify_protocol(run):
    p=read_json(Path(run)/'RUN_PROTOCOL.json')
    require(p['uids']==list(UIDS) and p['models']==list(MODELS),'Predeclared task set changed')
    require(p['max_steps']==16 and p['max_seconds']==3600 and p['snapshot_steps']==list(SNAPSHOTS),'Optimization budget changed')
    require(p['evaluation_seed_bases']==list(SEED_BASES) and p['points_each_direction']==8192,'Evaluation sampling changed')
    for path,digest in p['bound_files_sha256'].items():require(file_hash(path)==digest,'Bound source/data/checkpoint changed: '+path)
    return p


def load_data(p,uid):
    record=p['data'][uid];require(file_hash(record['path'])==record['sha256'],'Data changed')
    with np.load(record['path'],allow_pickle=False) as archive:
        # GT arrays are passed only to the offline metric function below.
        # predict_raw and optimize_layout receive only logr/base_u/template faces.
        d={key:np.array(archive[key]) for key in ('logr','base_u','faces','ref_vertices','ref_faces')}
    require(d['logr'].shape==(2050,) and d['base_u'].shape==(258,3) and d['faces'].shape==(512,3)
            and d['faces'].dtype==np.int32,'Mainfield/template schema changed')
    require(all(np.isfinite(x).all() for x in d.values()),'Nonfinite source array')
    return d


def load_model(p,name):
    if name=='fixed':return None
    identity=p['model_bindings'][name];require(file_hash(identity['path'])==identity['file_sha256'],'Initializer changed')
    state=torch.load(identity['path'],map_location='cpu',weights_only=False)
    require(state['update']==240000 and state['model_sha256']==model_sha256(state['model'])==identity['model_sha256'],
            'Initializer no longer matches fixed final')
    model=DomainVelocity().cpu();model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
    return model


def snapshot_records(result,requested=SNAPSHOTS):
    stats=result['stats'];trace=result['trace'];snapshots=result['snapshots']
    require(len(trace)==stats['iterations']+1 and [r['iteration'] for r in trace]==list(range(stats['iterations']+1)),
            'Optimizer trace is incomplete or reordered')
    ids=[s['iteration'] for s in snapshots];require(len(ids)==len(set(ids)),'Duplicate optimizer snapshots')
    require(all(0<=i<=stats['iterations'] for i in ids),'Snapshot exceeds actual executed steps')
    lookup={s['iteration']:s for s in snapshots};rows=[]
    for iteration in requested:
        if iteration in lookup:rows.append({'requested_iteration':iteration,'status':'available','snapshot':lookup[iteration]})
        else:rows.append({'requested_iteration':iteration,'status':'terminated_before_snapshot' if iteration>stats['iterations'] else 'missing_snapshot',
            'actual_last_iteration':stats['iterations'],'stop_reason':stats['stop_reason']})
    extras=[s for s in snapshots if s['iteration'] not in requested]
    return rows,extras


def serialize_optimizer(result,folder):
    arrays={'final_u':result['u'],'final_vertices':result['vertices'],'faces':result['faces']}
    metadata=[]
    for snapshot in result['snapshots']:
        n=snapshot['iteration'];arrays[f'snapshot_{n}_u']=snapshot['u'];arrays[f'snapshot_{n}_vertices']=snapshot['vertices']
        metadata.append({k:v for k,v in snapshot.items() if k not in ('u','vertices','faces')})
    save_npz(folder/'OPTIMIZER_ARRAYS.npz',**arrays)
    ledger={'stats':result['stats'],'trace':result['trace'],'attempts':result['attempts'],'snapshots':metadata,
        'arrays_sha256':file_hash(folder/'OPTIMIZER_ARRAYS.npz'),'selection_unchanged':True,'no_fixed_fallback':True}
    write_json(folder/'OPTIMIZER.json',ledger);return ledger


def evaluate_u(run,p,uid,model,kind,u,d,support,floor,claimed_vertices=None,extras=None):
    row={'uid':uid,'model':model,'kind':kind,'status':'failed','pure_network_result':kind=='raw',
        'mainfield_hash':array_hash(d['logr']),'no_fixed_fallback':True,**(extras or {})}
    try:
        u=np.asarray(u,np.float64);require(u.shape==(258,3) and np.isfinite(u).all()
            and np.allclose(np.linalg.norm(u,axis=1),1,rtol=0,atol=1e-9),'Invalid accepted direction array')
        r=support.query(d['logr'],u);mesh={'u':u,'r':r,'vertices':u*r[:,None],'faces':d['faces'].astype(np.int64)}
        if claimed_vertices is not None:require(np.array_equal(mesh['vertices'],claimed_vertices),'Returned radius stale or geometry differs from final requery')
        path=run/'attempts'/uid/model/kind/'mesh.npz';save_npz(path,**mesh)
        row.update(mesh_file=str(path.relative_to(run)),**mesh_identity(path,d['faces']))
        before=time.perf_counter();q=named_quality(mesh['vertices'],mesh['faces']);audit=audit_mesh(mesh['vertices'],mesh['faces'],exact=True)
        row['quality_and_exact_seconds']=time.perf_counter()-before
        row.update(quality=q,audit=audit,**qualification(q,audit,floor),final_radius_requery_exact=True)
        rv,rf=d['ref_vertices'],d['ref_faces'];index=p['all16_sorted_uids'].index(uid)
        row['surface_reference']='original_GT_triangles_offline_only';row['GT_reference_hash']=array_hash(rv,rf)
        seeds=[base+index for base in SEED_BASES]
        begin=time.perf_counter();cached=metric_cache(run,uid,mesh['vertices'],mesh['faces'],rv,rf,seeds,POINTS)
        row['surface_evaluation_seconds']=time.perf_counter()-begin;row['metrics']=[x['metrics'] for x in cached]
        row['status']='success'
    except Exception as exc:row.update(error=repr(exc),traceback=traceback.format_exc())
    write_json(run/'attempts'/uid/model/kind/'RESULT.json',row);return row


def task(run,p,uid,name,model,support):
    folder=run/'attempts'/uid/name;folder.mkdir(parents=True,exist_ok=False)
    rows=[];result=None;raw=None;extras=[];info={'uid':uid,'model':name,'status':'failed','rows':rows}
    try:
        d=load_data(p,uid);start=time.perf_counter();fixed,_=make_fixed(support,d['logr'],d['base_u'],d['faces'])
        floor=named_quality(fixed['vertices'],fixed['faces']);fixed_audit=audit_mesh(fixed['vertices'],fixed['faces'],exact=True)
        require(fixed_audit['valid'],'Actual Fixed floor mesh invalid')
        info['fixed_floor_generation_and_check_seconds']=time.perf_counter()-start;info['floor']=floor
        if name=='fixed':raw,generation_seconds=make_fixed(support,d['logr'],d['base_u'],d['faces']);meta={'generation_seconds':generation_seconds}
        else:raw,meta=predict_raw(model,d['logr'],support,d['base_u'],d['faces'])
        info['raw_initialization']=meta
        rows.append(evaluate_u(run,p,uid,name,'raw',raw['u'],d,support,floor,raw['vertices']))
        index=p['all16_sorted_uids'].index(uid);optimizer_floor={'min_angle_deg':floor['min_triangle_angle_deg'],
            'angle_q05_deg':floor['triangle_min_angle_p05_deg']}
        started=time.perf_counter()
        result=optimize_layout(d['logr'],raw['u'],d['faces'],optimizer_floor,device='cpu',max_steps=16,
            max_seconds=3600,seed=OPT_SEED_BASE+index,snapshot_steps=SNAPSHOTS)
        info['optimizer_call_wall_seconds']=time.perf_counter()-started
        serialize_optimizer(result,folder);info['optimizer_stats']=result['stats']
        requested,terminal=snapshot_records(result)
        for entry in requested:
            kind=f"snapshot_{entry['requested_iteration']}"
            if entry['status']=='available':
                snapshot=entry['snapshot'];row=evaluate_u(run,p,uid,name,kind,snapshot['u'],d,support,floor,snapshot['vertices'],
                    {'actual_iteration':snapshot['iteration'],'optimizer_trace_record':{k:v for k,v in snapshot.items() if k not in ('u','vertices','faces')}})
            else:
                row={'uid':uid,'model':name,'kind':kind,**entry,'no_fabricated_budget_completion':True}
                write_json(folder/kind/'RESULT.json',row)
            rows.append(row)
        for snapshot in terminal:
            extras.append(evaluate_u(run,p,uid,name,f"terminal_extra_{snapshot['iteration']}",snapshot['u'],d,support,floor,snapshot['vertices'],
                {'actual_iteration':snapshot['iteration'],'outside_requested_snapshot_grid':True}))
        final=evaluate_u(run,p,uid,name,'optimizer_final',result['u'],d,support,floor,result['vertices'],
            {'selected_by':'original_optimizer_best_feasible_rule','best_feasible_iteration':result['stats']['best_feasible_iteration'],
             'returned_last_infeasible_when_no_feasible':result['stats']['best_feasible_iteration'] is None})
        rows.append(final);info['status']='completed_optimizer_call'
        info['initialization_plus_optimizer_seconds']=meta['generation_seconds']+info['optimizer_call_wall_seconds']
        info['initialization_floor_optimizer_finalcheck_seconds']=info['initialization_plus_optimizer_seconds']+info['fixed_floor_generation_and_check_seconds']+final.get('quality_and_exact_seconds',0)
        info['evaluation_RMS_and_disk_excluded_from_above_online_cost']=True
    except Exception as exc:info.update(error=repr(exc),traceback=traceback.format_exc(),optimizer_actual_steps_unknown=result is None)
    recorded={r['kind'] for r in rows}
    for kind in ROW_KINDS:
        if kind not in recorded:
            row={'uid':uid,'model':name,'kind':kind,'status':'unavailable_runtime_failure','error':info.get('error'),
                'no_fabricated_budget_completion':True};write_json(folder/kind/'RESULT.json',row);rows.append(row)
    info['extra_terminal_snapshots']=extras;write_json(folder/'TASK.json',info);return info


def gain(row,reference):
    if row.get('status')!='success' or reference.get('status')!='success':return None
    require(row['uid']==reference['uid'] and row['mainfield_hash']==reference['mainfield_hash'] and row['GT_reference_hash']==reference['GT_reference_hash'],'Different mainfield or GT reference pair')
    require([x['seed'] for x in row['metrics']]==[x['seed'] for x in reference['metrics']],'Different evaluation seed pair')
    a=np.array([x['rms_bbox_percent'] for x in row['metrics']]);b=np.array([x['rms_bbox_percent'] for x in reference['metrics']])
    return None if (b<=0).any() else (100*(1-a/b)).tolist()


def summary(attempts):
    require(len(attempts)==20 and {(a['uid'],a['model']) for a in attempts}=={(u,m) for u in UIDS for m in MODELS},'Lost or duplicated task')
    rows=[r for a in attempts for r in a['rows']];keys=[(r['uid'],r['model'],r['kind']) for r in rows]
    require(len(keys)==120 and len(set(keys))==120 and set(keys)=={(u,m,k) for u in UIDS for m in MODELS for k in ROW_KINDS},'Lost requested row slot')
    lookup={(r['uid'],r['model'],r['kind']):r for r in rows};groups=[]
    for name in MODELS:
        paired=[]
        for uid in UIDS:
            raw=lookup[(uid,name,'raw')];final=lookup[(uid,name,'optimizer_final')]
            item={'uid':uid,'raw_qualified':raw.get('qualified'), 'final_qualified':final.get('qualified'),
                'final_vs_Fixed_raw_gain_percent':gain(final,lookup[(uid,'fixed','raw')]),
                'final_vs_Fixed_optimized_gain_percent':gain(final,lookup[(uid,'fixed','optimizer_final')])}
            a=next(x for x in attempts if x['uid']==uid and x['model']==name)
            item.update(optimizer_stats=a.get('optimizer_stats'),
                initialization_plus_optimizer_seconds=a.get('initialization_plus_optimizer_seconds'),
                initialization_floor_optimizer_finalcheck_seconds=a.get('initialization_floor_optimizer_finalcheck_seconds'))
            paired.append(item)
        groups.append({'model':name,'planned_parents':4,'raw_qualified_count':sum(x['raw_qualified'] is True for x in paired),
            'optimizer_final_qualified_count':sum(x['final_qualified'] is True for x in paired),'parents':paired})
    return {'groups':groups,'task_count':20,'primary_row_count':120,'missing_or_failed_primary_rows':sum(r.get('status')!='success' for r in rows),
        'pure_A_claim_permitted':False,'four_train_parents_only':True,'no_generalization_claim':True,
        'optimizer_internal_first_feasible_is_numerical_not_exact_audit_certificate':True,
        'no_Fixed_fallback_or_MC_based_selection':True}


def run(run):
    run=Path(run).resolve();p=verify_protocol(run);require(not (run/'RESULTS.json').exists(),'Do not overwrite an existing study')
    require(not torch.cuda.is_initialized(),'CPU-only warm-start study');torch.set_num_threads(1)
    support=RadialSupport();models={};errors={};attempts=[]
    for name in MODELS:
        try:models[name]=load_model(p,name)
        except Exception as exc:errors[name]=repr(exc)
    for uid in UIDS:
        for name in MODELS:
            if name in errors:
                folder=run/'attempts'/uid/name;folder.mkdir(parents=True,exist_ok=False)
                a={'uid':uid,'model':name,'status':'failed','error':errors[name],'rows':[
                    {'uid':uid,'model':name,'kind':kind,'status':'unavailable_model_load_failure','error':errors[name]} for kind in ROW_KINDS]}
                write_json(folder/'TASK.json',a)
            else:a=task(run,p,uid,name,models[name],support)
            attempts.append(a);print(json.dumps({'uid':uid,'model':name,'status':a['status'],'actual_steps':a.get('optimizer_stats',{}).get('iterations')}),flush=True)
    verify_protocol(run);result={'attempts':attempts,'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),'source_inputs_unchanged':True}
    write_json(run/'RESULTS.json',result);s=summary(attempts);write_json(run/'SUMMARY.json',s)
    lines=['固定四训练对象：初始化＋原优化器诊断','',
        '所有方法同16步上限、同UID内部种子、相同Fixed角度floor；3600秒仅安全上限。原始网络输出和原优化器结果分别保留。',
        '最终输出按原优化器best-feasible训练目标规则；没有Fixed回退，也没有用独立RMS选快照。',
        '缺失的1/4/16步快照明确记为提前终止或缺失；实际终止快照另存，不能伪称完成16步。',
        '该方法是显式学习初始化＋优化器，不能称纯A。误差参考为同一主场的完整三角面，两次8192点评测。','',
        '|初始化|raw合格/4|优化后合格/4|','|---|---:|---:|']
    for group in s['groups']:lines.append(f"|{group['model']}|{group['raw_qualified_count']}|{group['optimizer_final_qualified_count']}|")
    lines+=['','逐对象恢复步数、first-feasible时间、接受/拒绝提案、best-feasible位置与成本见SUMMARY.json及各OPTIMIZER.json。',
        'first-feasible是优化器数值角度约束判定；独立精确几何认证只在保存输出上执行。只覆盖4个指定训练对象。']
    (run/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return s


def smoke(out):
    require(not torch.cuda.is_initialized(),'CPU-only synthetic smoke');torch.set_num_threads(1);out=new_dir(out)
    source_before=source_files();support=RadialSupport();u=fibonacci(258);faces=hull_faces(u).astype(np.int32)
    radius=1/np.sqrt((support.u**2/np.array([1.5,.8,1.])**2).sum(1));y=np.log(radius)
    fixed=u*support.query(y,u)[:,None];floor=mesh_quality(fixed,faces)
    start=time.perf_counter();result=optimize_layout(y,u,faces,floor,device='cpu',max_steps=4,max_seconds=3600,
        seed=1493499,snapshot_steps=(0,1,4));optimizer_seconds=time.perf_counter()-start
    serialize_optimizer(result,out);snapshots,extra=snapshot_records(result,(0,1,4))
    require(result['stats']['iterations']==4,'Synthetic smoke did not complete4 actual steps')
    require(np.array_equal(result['vertices'],result['u']*support.query(y,result['u'])[:,None]),'Smoke final radius stale')
    start=time.perf_counter();audit=audit_mesh(result['vertices'],faces,exact=True);audit_seconds=time.perf_counter()-start
    start=time.perf_counter();rv=u/np.sqrt((u**2/np.array([1.5,.8,1.])**2).sum(1))[:,None];rf=faces
    metrics=[evaluate_mesh(result['vertices'],faces,rv,rf,points=8192,seed=seed) for seed in (1593499,1693499)]
    metric_seconds=time.perf_counter()-start
    require(source_files()==source_before,'Frozen optimizer/source was modified')
    save_npz(out/'SYNTHETIC_INPUT.npz',logr=y,base_u=u,faces=faces)
    report={'scope':'SYNTHETIC_CPU4_COST_SMOKE_ONLY_NOT_FORMAL_METHOD_RESULT','actual_updates':result['stats']['iterations'],
        'optimizer_seconds':optimizer_seconds,'one_exact_audit_seconds':audit_seconds,'one_two8192_metric_seconds':metric_seconds,
        'stats':result['stats'],'audit':audit,'metrics':metrics,'snapshots':[{'iteration':x['requested_iteration'],'status':x['status']} for x in snapshots],
        'rough_opt_only20tasks16steps_seconds':optimizer_seconds*20*16/4,
        'rough120uncached_output_metrics_seconds':metric_seconds*120,
        'cost_projection_not_formal_timing':True,'synthetic_shape_not_any_registered_parent':True,'synthetic_metric_reference':'512-face analytic ellipsoid radius samples; optimizer objective remains2050-node field',
        'no_GPU_context_created':not torch.cuda.is_initialized(),'production_artifacts_opened':False,'source_sha256':source_before}
    write_json(out/'SMOKE.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('stats','audit','metrics','source_sha256')},indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--long-run',required=True);p.add_argument('--balanced-run',required=True);p.add_argument('--out',required=True)
    r=sub.add_parser('run');r.add_argument('--run',required=True)
    s=sub.add_parser('smoke');s.add_argument('--out',required=True);a=parser.parse_args()
    if a.command=='prepare':prepare(a.long_run,a.balanced_run,a.out)
    elif a.command=='run':run(a.run)
    else:smoke(a.out)


if __name__=='__main__':main()
