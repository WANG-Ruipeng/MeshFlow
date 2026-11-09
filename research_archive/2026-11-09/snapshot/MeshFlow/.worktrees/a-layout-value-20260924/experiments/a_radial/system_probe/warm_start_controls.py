"""Two fixed initializer controls for the completed 16-step warm-start study.

Original warm-start source, optimizer, and 120 saved rows remain immutable.
An extra deterministic raw rollout exists only to verify replay identity.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import copy
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from ..exporter import ASSETS,RadialSupport
from ..legacy.networks import DomainVelocity
from ..io_utils import array_hash,file_hash,new_dir,save_npz,write_json
from ..layout_student.train import model_sha256
from . import warm_start as ws

NEW_MODELS=('original_A_10','early_v1_10')
EARLY_V1_HASH='7c02c7926d11c76f22d520a024fa7fac5d4c08c69050cb3745c226a4e6c1b760'
ALL_MODELS=tuple(ws.MODELS)+NEW_MODELS
REFERENCE='same frozen 2050-support-node / 4096-face radial mainfield complete triangular surface'


def verify_attempt_slots(attempts,models):
    expected={(uid,name) for uid in ws.UIDS for name in models}
    keys=[(a['uid'],a['model']) for a in attempts]
    ws.require(len(keys)==len(set(keys)) and set(keys)==expected,'Lost or duplicate registered task')
    ws.require(all(row['uid']==a['uid'] and row['model']==a['model'] for a in attempts for row in a['rows']),
               'Row belongs to another task')
    rows=[row for attempt in attempts for row in attempt['rows']]
    keys=[(r['uid'],r['model'],r['kind']) for r in rows]
    planned={(uid,name,kind) for uid,name in expected for kind in ws.ROW_KINDS}
    ws.require(len(keys)==len(set(keys)) and set(keys)==planned,'Lost or duplicate requested row slot')
    return rows


def verify_parent_state(p,results):
    ws.require(p['uids']==list(ws.UIDS) and p['models']==list(ws.MODELS),'Parent study model/UID set changed')
    for key,value in [('max_steps',16),('max_seconds',3600),('snapshot_steps',[0,1,4,16]),
        ('threads',4),('interop_threads',1),('optimizer_seed_base',1493501),
        ('evaluation_seed_bases',[1593501,1693501]),('points_each_direction',8192)]:
        ws.require(p[key]==value,'Parent fixed setting changed: '+key)
    return verify_attempt_slots(results['attempts'],ws.MODELS)


def validate_early_metadata(state):
    ws.require(state.get('source_model_only') is True and state.get('source_stored_update')==3000,
               'Early initializer must be the fixed3000 model-only source')
    ws.require(state.get('model_sha256')==EARLY_V1_HASH,'Early initializer is not the registered shared v1 seed10 model')
    ws.require(not set(state)&{'optimizer','scheduler','rng','sampler'},'Early initializer contains training state')
    ws.require(state.get('optimizer_rng_scheduler_sampler_from_source_omitted') is True,'Early source omission contract changed')


def inspect_initializer(path,name):
    path=Path(path).resolve();before=file_hash(path)
    value=torch.load(path,map_location='cpu',weights_only=True)
    if name=='original_A_10':
        state=value;update=None;source='unchanged original assets/original_A_10.pt'
    elif name=='early_v1_10':
        validate_early_metadata(value);state=value['model'];update=3000;source='same v1 seed10 initializer shared by the two sealed single-object diagnostics'
    else:raise ValueError('Unregistered initializer')
    model=DomainVelocity().cpu().float();model.load_state_dict(state,strict=True)
    ws.require(sum(x.numel() for x in model.parameters())==29907,'Initializer architecture changed')
    ws.require(all(x.dtype==torch.float32 for x in state.values()),'Initializer parameter dtype changed')
    digest=model_sha256(state)
    if name=='early_v1_10':ws.require(digest==EARLY_V1_HASH,'Early model tensors differ')
    ws.require(file_hash(path)==before,'Initializer changed while read')
    return {'path':str(path),'file_sha256':before,'model_sha256':digest,'seed':10,
            'source_stored_update':update,'model_only_initialization':True,'parameter_count':29907,'source':source}


def bound_inventory(root):
    root=Path(root).resolve()
    return {str(path):file_hash(path) for path in sorted(root.rglob('*')) if path.is_file()}


def verify_bound(files):
    for path,digest in files.items():ws.require(file_hash(path)==digest,'Frozen bound file changed: '+path)


def prepare(parent_run,early_checkpoint,out):
    ws.require(not torch.cuda.is_initialized(),'Preparation must be CPU-only')
    parent=Path(parent_run).resolve();oldp=ws.verify_protocol(parent);results=ws.read_json(parent/'RESULTS.json')
    rows=verify_parent_state(oldp,results)
    ws.require(results['protocol_sha256']==file_hash(parent/'RUN_PROTOCOL.json'),'Parent results/protocol mismatch')
    parent_files=bound_inventory(parent);support=RadialSupport();field_checks=[]
    for uid in ws.UIDS:
        data=ws.load_data(oldp,uid);v,f=support.mesh(data['logr'])
        ws.require(np.array_equal(v,data['ref_vertices']) and np.array_equal(f,data['ref_faces']),
                   'Reference is not the same frozen mainfield surface')
        field_checks.append({'uid':uid,'mainfield_hash':array_hash(data['logr']),
            'reference_hash':array_hash(data['ref_vertices'],data['ref_faces']),'reference_vertices':len(v),'reference_faces':len(f)})
    for row in rows:
        if row.get('status')!='success':continue
        path=(parent/row['mesh_file']).resolve();ws.require(path.is_relative_to(parent),'Parent mesh path escape')
        ws.require(file_hash(path)==row['mesh_npz_sha256'],'Parent saved mesh changed')
        with np.load(path,allow_pickle=False) as z:
            ws.require(array_hash(z['vertices'],z['faces'])==row['stored_array_hash'],'Parent stored arrays changed')
    bindings={name:inspect_initializer(path,name) for name,path in [
        ('original_A_10',ASSETS/'original_A_10.pt'),('early_v1_10',early_checkpoint)]}
    bound={**oldp['bound_files_sha256'],**ws.source_files(),**parent_files}
    for value in bindings.values():bound[value['path']]=value['file_sha256']
    for name in ['warm_start_controls.py','test_warm_start_controls.py']:
        path=Path(__file__).with_name(name);bound[str(path)]=file_hash(path)
    verify_bound(bound);out=new_dir(out)
    write_json(out/'PARENT_FILE_MANIFEST.json',{'root':str(parent),'files':parent_files})
    p={'scope':'TWO_FIXED_INITIALIZER_CONTROLS_PLUS_UNCHANGED_OPTIMIZER_NOT_PURE_A',
       'parent_run':str(parent),'parent_protocol_sha256':file_hash(parent/'RUN_PROTOCOL.json'),
       'parent_results_sha256':file_hash(parent/'RESULTS.json'),'parent_existing_tasks':20,'parent_existing_rows':120,
       'parent_file_manifest_sha256':file_hash(out/'PARENT_FILE_MANIFEST.json'),'parent_outputs_recomputed':False,
       'uids':list(ws.UIDS),'all16_sorted_uids':oldp['all16_sorted_uids'],'models':list(NEW_MODELS),
       'combined_models':list(ALL_MODELS),'model_bindings':bindings,'data':oldp['data'],
       'expected_new_tasks':8,'expected_new_rows':48,'expected_combined_tasks':28,'expected_combined_rows':168,
       'row_kinds':list(ws.ROW_KINDS),'max_steps':16,'max_seconds':3600,'snapshot_steps':[0,1,4,16],
       'device':'cpu','threads':4,'interop_threads':1,'optimizer_seed_base':1493501,
       'evaluation_seed_bases':[1593501,1693501],'points_each_direction':8192,
       'same_per_UID_optimizer_and_MC_seeds_as_parent':True,'surface_reference':REFERENCE,
       'historical_GT_reference_hash_key_means_same_radial_surface':True,
       'optimizer_surface_reference':REFERENCE,'MC_independence_is_sampling_not_different_geometry':True,
       'quality_floor':oldp['quality_floor'],'output_selection':oldp['output_selection'],
       'optimizer_implementation':'unmodified warm_start.task -> layout_value.optimizer.optimize_layout',
       'extra_raw_identity_rollout':{'per_new_task':1,'purpose':'save raw then verify exact CPU4 replay by unchanged ws.task',
           'same_fixed_model_input_and_32steps':True,'additional_optimizer_calls':0,'counts_as_candidate_search':False,
           'setup_validation_seconds_reported_separately':True,'excluded_from_primary_online_cost':True},
       'no_Fixed_fallback':True,'no_MC_or_check_selection':True,'no_retry_or_new_random_initialization':True,
       'scientific_scope':'Only whether the four trained fixed initializers outperform original_A10 or early_v1_seed10 after the same optimizer. Does not exclude all arbitrary-initializer/random-perturbation mechanisms.',
       'same_training_seed_claim_for_seed11_vs_seed10_controls':False,'four_previously_seen_train_parents_only':True,
       'source_reference_value_checks':field_checks,'bound_files_sha256':bound}
    write_json(out/'RUN_PROTOCOL.json',p);return p


def verify_protocol(run):
    run=Path(run).resolve();p=ws.read_json(run/'RUN_PROTOCOL.json')
    ws.require(p['uids']==list(ws.UIDS) and p['models']==list(NEW_MODELS),'New model/UID set changed')
    for key,value in [('max_steps',16),('max_seconds',3600),('snapshot_steps',[0,1,4,16]),('threads',4),
        ('interop_threads',1),('optimizer_seed_base',1493501),('evaluation_seed_bases',[1593501,1693501]),
        ('points_each_direction',8192),('expected_new_tasks',8),('expected_new_rows',48)]:
        ws.require(p[key]==value,'Fixed control setting changed: '+key)
    ws.require(file_hash(run/'PARENT_FILE_MANIFEST.json')==p['parent_file_manifest_sha256'],'Parent inventory changed')
    verify_bound(p['bound_files_sha256'])
    parent=Path(p['parent_run']);old=ws.read_json(parent/'RESULTS.json');oldp=ws.read_json(parent/'RUN_PROTOCOL.json')
    ws.require(file_hash(parent/'RESULTS.json')==p['parent_results_sha256']
        and file_hash(parent/'RUN_PROTOCOL.json')==p['parent_protocol_sha256'],'Parent ledger binding changed')
    verify_parent_state(oldp,old);return p,old


def load_model(p,name):
    identity=p['model_bindings'][name];ws.require(inspect_initializer(identity['path'],name)==identity,'Initializer identity differs')
    value=torch.load(identity['path'],map_location='cpu',weights_only=True)
    state=value if name=='original_A_10' else value['model']
    model=DomainVelocity().cpu().float();model.load_state_dict(state,strict=True);model.eval().requires_grad_(False)
    return model


def failed_task(run,uid,name,error):
    folder=run/'attempts'/uid/name;folder.mkdir(parents=True,exist_ok=False)
    rows=[{'uid':uid,'model':name,'kind':kind,'status':'unavailable_initializer_identity_replay_failure',
           'error':repr(error),'no_fabricated_budget_completion':True} for kind in ws.ROW_KINDS]
    for row in rows:write_json(folder/row['kind']/'RESULT.json',row)
    attempt={'uid':uid,'model':name,'status':'failed','rows':rows,'error':repr(error),
             'traceback':traceback.format_exc(),'optimizer_call_executed':False,'optimizer_actual_steps':0}
    write_json(folder/'TASK.json',attempt);return attempt


def summarize_combined(old_attempts,new_attempts):
    verify_attempt_slots(old_attempts,ws.MODELS);verify_attempt_slots(new_attempts,NEW_MODELS)
    attempts=old_attempts+new_attempts;rows=verify_attempt_slots(attempts,ALL_MODELS)
    by={(r['uid'],r['model'],r['kind']):r for r in rows};groups=[];paired=[]
    def joint(row,gains):return bool(row.get('status')=='success' and row.get('qualified') is True
        and gains is not None and min(gains)>=2.)
    for name in ALL_MODELS:
        parents=[]
        for uid in ws.UIDS:
            raw=by[uid,name,'raw'];final=by[uid,name,'optimizer_final']
            raw_gain=ws.gain(final,by[uid,'fixed','raw']);opt_gain=ws.gain(final,by[uid,'fixed','optimizer_final'])
            attempt=next(a for a in attempts if (a['uid'],a['model'])==(uid,name))
            parents.append({'uid':uid,'raw_qualified':raw.get('qualified') is True,
                'final_qualified':final.get('qualified') is True,'final_status':final['status'],
                'final_vs_Fixed_raw_gain_percent':raw_gain,'final_vs_Fixed_optimized_gain_percent':opt_gain,
                'joint_vs_Fixed_raw':joint(final,raw_gain),'joint_vs_Fixed_opt16':joint(final,opt_gain),
                'optimizer_stats':attempt.get('optimizer_stats'),
                'initialization_plus_optimizer_seconds':attempt.get('initialization_plus_optimizer_seconds'),
                'extra_raw_identity_setup_seconds':attempt.get('extra_raw_identity_setup_seconds')})
        groups.append({'model':name,'planned_parents':4,'raw_qualified_count':sum(x['raw_qualified'] for x in parents),
            'optimizer_final_qualified_count':sum(x['final_qualified'] for x in parents),
            'joint_vs_Fixed_raw':sum(x['joint_vs_Fixed_raw'] for x in parents),
            'joint_vs_Fixed_opt16':sum(x['joint_vs_Fixed_opt16'] for x in parents),'parents':parents})
    for learned in ws.MODELS[1:]:
        for control in NEW_MODELS:
            for uid in ws.UIDS:
                a=by[uid,learned,'optimizer_final'];b=by[uid,control,'optimizer_final'];gains=ws.gain(a,b)
                paired.append({'uid':uid,'learned':learned,'control':control,'gain_percent_two_MC':gains,
                    'learned_qualified':a.get('qualified') is True,'control_qualified':b.get('qualified') is True,
                    'learned_joint_ge2pct':joint(a,gains),'both_qualified_joint_ge2pct':joint(a,gains) and b.get('qualified') is True,
                    'pair_available':gains is not None,'same_training_seed':learned.endswith('_10')})
    return {'groups':groups,'task_count':28,'primary_row_count':168,'new_task_count':8,'new_primary_row_count':48,
        'missing_or_failed_primary_rows':sum(r.get('status')!='success' for r in rows),
        'new_missing_or_failed_rows':sum(r.get('status')!='success' for a in new_attempts for r in a['rows']),
        'paired_long_vs_new_controls':paired,'all_denominators_fixed':True,'pure_A_claim_permitted':False,
        'surface_reference':REFERENCE,'no_MC_or_check_selection':True,'no_generalization_claim':True,
        'arbitrary_initializers_or_random_perturbation_mechanism_excluded':False,
        'parent_outputs_recomputed':False,'extra_identity_raw_not_primary_online_cost':True}


def write_report(run,summary,old_attempts,new_attempts):
    lines=['固定两初态补充对照：原A10、早期v1 seed10，各加同一16步优化器','',
        '原warm研究20个任务/120条结果只读复用。本轮仅新增8个任务/48条预定输出，保留失败与实际步数。',
        '参考均为同一冻结2050节点/4096面径向主场完整曲面，两次MC采样独立；不是另一套原资产GT。',
        '额外一次raw运行只生成身份核验证据；随后原ws.task精确重放。该setup计时单列，主要online成本沿用一次raw+优化器。','',
        '|初始化|raw合格/4|优化后合格/4|相对Fixed优化16步合格且双MC≥2%/4|',
        '|---|---:|---:|---:|']
    for g in summary['groups']:lines.append(f"|{g['model']}|{g['raw_qualified_count']}|{g['optimizer_final_qualified_count']}|{g['joint_vs_Fixed_opt16']}|")
    lines+=['','|长训初始化 / 新控制|两者合格且长训双MC≥2% / 4|可配对 / 4|','|---|---:|---:|']
    for learned in ws.MODELS[1:]:
        for control in NEW_MODELS:
            rr=[r for r in summary['paired_long_vs_new_controls'] if r['learned']==learned and r['control']==control]
            lines.append(f"|{learned} / {control}|{sum(r['both_qualified_joint_ge2pct'] for r in rr)}|{sum(r['pair_available'] for r in rr)}|")
    lines+=['','本补充只检验这些固定长训权重是否优于现成原A或早期v1初始化；不能排除任意初始化、随机扰动或restoration机制的全部解释。',
        '均为4个已见训练对象；seed11模型对seed10控制不属于同训练随机种子的严格配对。结果是初始化+优化器，不能当作纯A。']
    (run/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    by={(r['uid'],r['model'],r['kind']):r for a in old_attempts+new_attempts for r in a['rows']}
    fig,axes=plt.subplots(2,2,figsize=(13.5,9));colors=['#555555','#4477aa','#66ccee','#cc6677','#aa3377','#228833','#aa8800']
    for ax,uid in zip(axes.flat,ws.UIDS):
        base=by[uid,'fixed','optimizer_final'];den=np.array([m['rms_bbox_percent'] for m in base['metrics']])
        for index,(name,color) in enumerate(zip(ALL_MODELS,colors)):
            row=by[uid,name,'optimizer_final']
            if row.get('status')!='success':ax.text(index,1,'failed',rotation=90,ha='center',color='crimson');continue
            values=np.array([m['rms_bbox_percent'] for m in row['metrics']])/den
            ax.plot([index,index],values,color=color,lw=2)
            ax.scatter([index-.05,index+.05],values,color=color,marker='o' if row.get('qualified') else 'x',s=40)
        ax.axhline(1,color='#999999',ls='--');ax.set_title(uid[:8]+' (seen train parent)')
        ax.set_xticks(range(7),['Fixed','Long10','Long11','Balanced10','Balanced11','OriginalA10','EarlyV1'],rotation=35,ha='right')
        ax.set_ylabel('Returned same-field RMS / Fixed opt16 RMS');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Fixed initializer controls + the same 16-step optimizer; all four parents')
    fig.text(.5,.01,'Two points: two shared MC draws, not confidence intervals. Circle: qualified; cross: unqualified. Failures retained. No pure-A or generalization claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.95])
    for ext in ['png','pdf']:fig.savefig(run/f'initializer_controls.{ext}',dpi=160)
    plt.close(fig)


def run_study(run):
    run=Path(run).resolve();p,old=verify_protocol(run)
    ws.require(not torch.cuda.is_initialized(),'CPU-only controls');torch.set_num_threads(4)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    write_json(run/'START.json',{'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),'no_automatic_retry':True,
        'old_tasks_recomputed':0,'new_optimizer_call_budget':8,'max_steps_each':16})
    support=RadialSupport();models={name:load_model(p,name) for name in NEW_MODELS};attempts=[]
    try:
        for uid in ws.UIDS:
            for name in NEW_MODELS:
                setup=time.perf_counter();reference_folder=run/'raw_identity_reference'/uid/name
                reference_folder.mkdir(parents=True,exist_ok=False)
                try:
                    d=ws.load_data(p,uid);raw,meta=ws.predict_raw(models[name],d['logr'],support,d['base_u'],d['faces'])
                    path=reference_folder/'raw.npz';save_npz(path,**raw)
                    record={'path':str(path),'sha256':file_hash(path)}
                    setup_seconds=time.perf_counter()-setup
                    write_json(reference_folder/'IDENTITY.json',{'uid':uid,'model':name,'model_identity':p['model_bindings'][name],
                        'raw_file_sha256':record['sha256'],'raw_generation':meta,'setup_validation_seconds':setup_seconds,
                        'purpose':'fixed-input replay identity only; no candidate selection and no optimization'})
                except Exception as error:
                    attempt=failed_task(run,uid,name,error);attempts.append(attempt)
                    print(json.dumps({'uid':uid,'model':name,'status':attempt['status'],'optimizer_calls':0}),flush=True)
                    continue
                runtime=copy.deepcopy(p);runtime['raw_initializer_references']={name:{uid:record}}
                attempt=ws.task(run,runtime,uid,name,models[name],support)
                # Keep untouched original TASK/RESULT records; supplement setup timing outside them.
                attempt={**attempt,'extra_raw_identity_setup_seconds':setup_seconds,'extra_raw_identity_record':str(reference_folder/'IDENTITY.json')}
                attempts.append(attempt)
                print(json.dumps({'uid':uid,'model':name,'status':attempt['status'],
                    'actual_steps':attempt.get('optimizer_stats',{}).get('iterations')}),flush=True)
        verify_attempt_slots(attempts,NEW_MODELS);verify_protocol(run)
        result={'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),'attempts':attempts,
            'parent_results_path':str(Path(p['parent_run'])/'RESULTS.json'),'parent_results_sha256':p['parent_results_sha256'],
            'parent_outputs_recomputed':False,'surface_reference':REFERENCE,'historical_GT_key_is_same_mainfield':True,
            'source_inputs_unchanged':True,'new_tasks':8,'new_rows':48}
        write_json(run/'RESULTS.json',result)
        summary=summarize_combined(old['attempts'],attempts);write_json(run/'SUMMARY.json',summary)
        write_report(run,summary,old['attempts'],attempts)
        write_json(run/'COMPLETE.json',{'status':'COMPLETE','new_tasks':8,'new_rows':48,
            'failed_new_rows':summary['new_missing_or_failed_rows'],'results_sha256':file_hash(run/'RESULTS.json'),
            'summary_sha256':file_hash(run/'SUMMARY.json'),'parent_files_unchanged':True,
            'cuda_context_initialized':torch.cuda.is_initialized()})
        return summary
    except Exception as error:
        write_json(run/'FAILED.json',{'error':repr(error),'traceback':traceback.format_exc(),
            'tasks_recorded':len(attempts),'no_automatic_retry':True});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--parent-run',required=True);p.add_argument('--early-checkpoint',required=True);p.add_argument('--out',required=True)
    r=sub.add_parser('run');r.add_argument('--run',required=True);args=parser.parse_args()
    if args.command=='prepare':value=prepare(args.parent_run,args.early_checkpoint,args.out);print(json.dumps({'new_tasks':value['expected_new_tasks']}),flush=True)
    else:value=run_study(args.run);print(json.dumps({'groups':value['groups'],'new_failed_rows':value['new_missing_or_failed_rows']}),flush=True)


if __name__=='__main__':main()
