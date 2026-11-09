"""Deterministic tables and figures from actual completed work only."""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import numpy as np
from .common import atomic_json,read_json,digest
from .evaluation import CURVE_MODELS,SWAP_MODELS,FIT_MODELS,PHASE_KEYS,resolve,arrays,score_record

PRIMARY = ['remainder_symmetric_RMS_bbox_pct','fixed_R12_coverage_within_1pct_fraction','boundary_RMS_bbox_pct','Gamma_to_free_edges_RMS_bbox_pct','CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction','free_area_over_GT_free_area']
NA = 'NA'

def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:
        fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for row in rows:
            w.writerow({k:NA if v is None else v for k,v in row.items()})

def finite(x):
    return isinstance(x,(int,float,np.integer,np.floating)) and math.isfinite(x)

def mean(values):
    vals=[float(x) for x in values if finite(x)]
    return float(np.mean(vals)) if vals else None

def _step(model):
    return 0 if model=='Native0' else int(model.rsplit('_',1)[1])

def _arm(model):
    return 'Native0' if model=='Native0' else model.rsplit('_',1)[0]

def collect(out,panels):
    from experiments.mf_h1_local0.control_mvp48.eval_scoring import METRIC_SPECS
    rows=[];teacher=[];ledgers=[]
    for phase,models in [('curve',CURVE_MODELS),('swap',SWAP_MODELS),('fit',FIT_MODELS),('holdout',[f'{a}_5000' for a in ('T1','Geo','Full')])]:
        key,tkey=PHASE_KEYS[phase]
        for model in models:
            path=out/'evaluation'/phase/model/'ledger.json'
            ledger=read_json(path) if path.exists() else {}
            if ledger:ledgers.append(ledger)
            indexed={x['case_id']:x for x in ledger.get('rows',[])}
            for r in panels[key]:
                found=indexed.get(r['case_id'],{})
                row=dict(phase=phase,model_id=model,arm=_arm(model),step=_step(model),case_id=r['case_id'],
                    split=r.get('split','holdout' if phase=='holdout' else None),uid=r.get('uid',r.get('parent_id')),
                    patch_id=r.get('patch_id',r.get('condition_id')),noise_index=r.get('noise_index',r.get('noise_id')),
                    C_area_fraction=r.get('C_area_fraction'),status=found.get('status',ledger.get('status','NOT_RUN')),
                    raw_path=found.get('raw_path'),raw_sha256=found.get('raw_sha256'),score_path=found.get('score_path'),recipient_case_id=r.get('recipient_case_id'),
                    donor_uid=r.get('donor_uid'))
                row.update({metric:None for metric in METRIC_SPECS})
                if found.get('score_path'):
                    p=resolve(out,found['score_path'])
                    if digest(p)!=found['score_sha256']:raise ValueError('Score identity changed')
                    score=read_json(p);row.update(score.get('metrics',{}));row['score_status']=score['status']
                rows.append(row)
            tidx={x['case_id']:x for x in ledger.get('teacher_rows',[])}
            for r in panels[tkey] if tkey else []:
                found=tidx.get(r.get('teacher_id',r['case_id']),{})
                teacher.append(dict(phase=phase,model_id=model,arm=_arm(model),step=_step(model),case_id=r['case_id'],
                    split=r['split'],uid=r['uid'],patch_id=r['patch_id'],t=r['t'],status=found.get('status','NOT_RUN'),
                    FM_error=found.get('FM_error'),D_edge=found.get('D_edge'),L5=found.get('L5'),
                    Yhat_free_area=found.get('geometry',{}).get('Yhat_free',{}).get('total_area'),nfe=found.get('nfe')))
    return rows,teacher,ledgers

def historical_holdout(source,out,panels):
    """Read-only old raw/score reuse, requiring input/checkpoint/scorer identity."""
    from experiments.mf_h1_local0.control_mvp48.eval_scoring import METRIC_SPECS,source_identity
    old_path=source/'evaluation/gpu_run.json'
    if not old_path.exists():return []
    old=read_json(old_path)
    prep=read_json(source/'registration/eval_protocol.json')
    old_sources=prep['scorer_identity']
    if not old_sources:
        fixture=source/'registration/eval_scoring_fixtures.json'
        old_sources=read_json(fixture).get('source_identity',{}) if fixture.exists() else {}
    scorer_matches=old_sources==source_identity()
    indexed={(r.get('case_id'),r.get('model')):r for r in old.get('rows',[])}
    result=[]
    handoff=read_json(source/'registration/train_handoff.json')
    verified_checkpoints={arm:digest(source/'checkpoints'/f'{arm}_step1000.pt')==next(v['sha256'] for v in handoff['completed_checkpoints'] if v['variant']==arm and v['logical_step']==1000) for arm in ('T1','Geo','Full')}
    for r in panels['holdout']:
        for arm in ('T1','Geo','Full'):
            previous=indexed.get((r.get('source_case_id',r['case_id']),arm),{})
            row=dict(phase='holdout',model_id=arm+'_1000',arm=arm,step=1000,case_id=r['case_id'],split='holdout',
                uid=r['uid'],patch_id=r['patch_id'],noise_index=r['noise_index'],status='NOT_COMPARABLE',
                reuse_generation_count=0,raw_path=None,raw_sha256=None)
            row.update({k:None for k in METRIC_SPECS})
            raw=source/previous.get('raw_path','missing')
            compatible=(scorer_matches and previous.get('status')=='PASS' and raw.is_file()
                and digest(raw)==previous.get('raw_sha256') and previous.get('input_file_sha256')==r['input_sha256'])
            if compatible:
                compatible=verified_checkpoints[arm]
            if compatible:
                scorepath=source/'evaluation/scores'/(previous['request_id']+'.json')
                if scorepath.exists():
                    score=read_json(scorepath)
                    if score.get('raw_sha256')!=previous['raw_sha256']:raise ValueError('Historical score/raw identity mismatch')
                    metrics=score['metrics']
                else:
                    # CPU only on actual raw; never regenerate old endpoints.
                    score=score_record(out,r,arrays(raw)['raw'],phase='holdout');metrics=score['metrics']
                    atomic_json(out/'evaluation/holdout_reused_scores'/(previous['request_id']+'.json'),score)
                row.update(metrics);row.update(status='REUSED',score_status=score.get('scoring_status',score.get('status','UNKNOWN')),raw_path=str(raw),raw_sha256=previous['raw_sha256'],scorer_identity_match=True)
            else:
                row['reason']='Raw/input/checkpoint/scorer identity missing or mismatched; no old generation replay'
            result.append(row)
    return result

def aggregate(rows,metrics):
    per_parent=[];curves=[]
    groups={}
    for row in rows:
        key=(row['phase'],row['model_id'],row['split'],row['uid'])
        groups.setdefault(key,[]).append(row)
    for (phase,model,split,uid),members in sorted(groups.items(),key=lambda p:str(p[0])):
        row=dict(phase=phase,model_id=model,arm=_arm(model),step=_step(model),split=split,uid=uid,
            requested_cases=len(members),completed_cases=sum(r['status'] in ('PASS','REUSED') for r in members),score_valid_cases=sum(r.get('score_status')=='PASS' for r in members))
        for m in metrics:
            row[m]=mean([r.get(m) for r in members if r.get('score_status')=='PASS']);row[m+'_valid_cases']=sum(finite(r.get(m)) and r.get('score_status')=='PASS' for r in members)
        per_parent.append(row)
    grouped={}
    for row in per_parent:grouped.setdefault((row['phase'],row['model_id'],row['split']),[]).append(row)
    for (phase,model,split),members in sorted(grouped.items(),key=lambda p:str(p[0])):
        row=dict(phase=phase,model_id=model,arm=_arm(model),step=_step(model),split=split,
            requested_parents=len(members),valid_parents=sum(r['score_valid_cases']>0 for r in members),
            requested_cases=sum(r['requested_cases'] for r in members),completed_cases=sum(r['completed_cases'] for r in members),score_valid_cases=sum(r['score_valid_cases'] for r in members))
        for m in metrics:
            row[m]=mean([r.get(m) for r in members]);row[m+'_valid_parents']=sum(finite(r.get(m)) for r in members)
        curves.append(row)
    return per_parent,curves

def paired(rows,kind):
    result=[]
    if kind=='control':
        base={(r['model_id'],r['case_id']):r for r in rows if r['phase']=='curve'}
        for wrong in [r for r in rows if r['phase']=='swap']:
            correct=base.get((wrong['model_id'],wrong['recipient_case_id']),{})
            item=dict(model_id=wrong['model_id'],step=wrong['step'],uid=wrong['uid'],case_id=wrong['case_id'],donor_uid=wrong['donor_uid'],
                status='PASS' if correct.get('status')=='PASS' and wrong['status']=='PASS' else 'NOT_COMPARABLE',
                absolute_quality_scope='recipient own free GT; wrong condition is a diagnostic')
            for m in PRIMARY:
                a,b=wrong.get(m),correct.get(m)
                item[m+'_wrong']=a;item[m+'_correct']=b;item[m+'_wrong_minus_correct']=a-b if finite(a) and finite(b) else None
            result.append(item)
    else:
        # Pair each parent at the same budget before averaging parents.
        index={(r['model_id'],r['split'],r['uid']):r for r in rows if r['phase']=='curve'}
        for step in (1000,2000,3500,5000):
            for arm,ref in (('Geo','T1'),('Full','Geo'),('Full','T1')):
                for (model,split,uid),a in sorted(index.items(),key=lambda p:str(p[0])):
                    if model!=f'{arm}_{step}':continue
                    b=index.get((f'{ref}_{step}',split,uid),{})
                    item=dict(comparison=arm+'-'+ref,step=step,split=split,uid=uid)
                    for m in PRIMARY:
                        x,y=a.get(m),b.get(m);item[m]=x-y if finite(x) and finite(y) else None
                    result.append(item)
    return result

def fmt(value):
    return f'{float(value):.6g}' if finite(value) else NA

def figures(out,panels,rows,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    folder=out/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,metric in zip(axes.flat,PRIMARY[:6]):
        for split in ('train','validation'):
            for arm,color in [('T1','#3974bc'),('Geo','#dc8a2c'),('Full','#469b62')]:
                data=sorted([r for r in curves if r['phase']=='curve' and r['split']==split and (r['arm']==arm or (arm in ('Geo','Full') and r['model_id']=='Geo_shared_500')) and finite(r.get(metric))],key=lambda r:r['step'])
                if data:ax.plot([r['step'] for r in data],[r[metric] for r in data],marker='o',color=color,linestyle='-' if split=='train' else '--',label=arm+' '+split)
        native=[r[metric] for r in curves if r['phase']=='curve' and r['model_id']=='Native0' and finite(r.get(metric))]
        for value in native:ax.axhline(value,color='#666666',linewidth=.8,linestyle=':',label='Native0 reference')
        ax.set_title(metric,fontsize=8);ax.set_xlabel('MVP48 conditional updates');ax.grid(alpha=.2)
    if any(ax.lines for ax in axes.flat):axes.flat[0].legend(fontsize=6)
    fig.tight_layout();fig.savefig(folder/'train_validation_curves.png',dpi=150);plt.close(fig)
    index={(r['model_id'],r['case_id']):r for r in rows if r['phase']=='curve'}
    images=[]
    for record in panels['generation']:
        if record['noise_index']!=0:continue
        bank=arrays(resolve(out,record['input_path']));C=bank['C'].reshape(-1,3,3);full=bank['full_target'].reshape(-1,3,3)
        geometries=[('GT',full),('C',C)]
        for model in ('Native0','T1_1000','T1_5000','Geo_5000','Full_5000'):
            row=index.get((model,record['case_id']),{})
            raw=arrays(resolve(out,row['raw_path']))['raw'].reshape(-1,3,3) if row.get('raw_path') and row['status']=='PASS' else None
            geometries.append((model,raw))
        if not any(g is not None for _,g in geometries[2:]):continue
        pointsets=[g.reshape(-1,3) for _,g in geometries if g is not None]
        points=np.concatenate(pointsets);low=points.min(0);high=points.max(0);center=(low+high)/2;radius=max(high-low)*.55
        fig=plt.figure(figsize=(3*len(geometries),9))
        reference=arrays(resolve(out,record['ref_npz']))
        queries=reference['coverage_points']
        coverage_distances={}
        for model in ('Native0','T1_1000','T1_5000','Geo_5000','Full_5000'):
            r=index.get((model,record['case_id']),{})
            if r.get('score_path'):
                score=read_json(resolve(out,r['score_path']))
                coverage_distances[model]=score.get('detail',{}).get('fixed_GT_coverage',{}).get('per_point_distances')
        for j,(label,geo) in enumerate(geometries):
            for rownum in (0,1,2):
                ax=fig.add_subplot(3,len(geometries),rownum*len(geometries)+j+1,projection='3d');ax.set_title(label+(' full' if rownum==0 else ' free' if rownum==1 else ' GT coverage'),fontsize=9)
                if geo is None:
                    ax.text2D(.2,.5,'NOT_RUN',transform=ax.transAxes)
                else:
                    selected=geo if rownum==0 or label=='C' else geo[int(bank['K']):] if label!='GT' else bank['gt_free'].reshape(-1,3,3)
                    view=selected[:,:,[0,2,1]]
                    if rownum!=2:ax.add_collection3d(Poly3DCollection(view,facecolors='#7195b8',edgecolors='#344b61',linewidths=.15,alpha=.65))
                    else:
                        distances=coverage_distances.get(label)
                        colors=np.where(np.asarray(distances)<=.01*float(bank['L']),'#29964a','#c84135') if distances is not None else '#aaaaaa'
                        ax.scatter(queries[:,0],queries[:,2],queries[:,1],c=colors,s=3)
                        if distances is None:ax.text2D(.03,.02,'reference only',transform=ax.transAxes,fontsize=6)
                ax.view_init(20,-55);ax.set_proj_type('ortho')
                for setter,dim in ((ax.set_xlim,0),(ax.set_ylim,2),(ax.set_zlim,1)):setter(center[dim]-radius,center[dim]+radius)
                ax.set_box_aspect((1,1,1));ax.set_axis_off()
        fig.tight_layout();name=record['case_id']+'.png';fig.savefig(folder/name,dpi=110);plt.close(fig);images.append(name)
    atomic_json(folder/'manifest.json',dict(curves='train_validation_curves.png',fixed_panels=images,scope='All registered noise0 cases; fixed views and common per-case extents; missing outputs labelled NOT_RUN'))

def run(source,out,make_figures=True):
    source,out=Path(source),Path(out)
    if not (out/'panels.json').exists():
        return dict(status='NOT_RUN',reason='No frozen panels')
    panels=read_json(out/'panels.json')
    from experiments.mf_h1_local0.control_mvp48.eval_scoring import METRIC_SPECS
    rows,teacher,ledgers=collect(out,panels)
    rows+=historical_holdout(source,out,panels)
    parents,curves=aggregate(rows,list(METRIC_SPECS))
    controls=paired(rows,'control');increments=paired(parents,'increments')
    native={(r['split'],r['uid']):r for r in parents if r['phase']=='curve' and r['model_id']=='Native0'}
    relative=[]
    for r in parents:
        if r['phase']!='curve' or r['model_id']=='Native0':continue
        b=native.get((r['split'],r['uid']),{})
        item={k:r[k] for k in ('model_id','arm','step','split','uid')}
        for metric in METRIC_SPECS:
            x,y=r.get(metric),b.get(metric);item[metric+'_minus_Native0']=x-y if finite(x) and finite(y) else None
        relative.append(item)
    write_csv(out/'relative_initialization.csv',relative)
    rowindex={(r['model_id'],r['case_id']):r for r in rows if r['phase']=='curve'}
    swapindex={(r['model_id'],r['case_id']):r for r in rows if r['phase']=='swap'}
    for item in controls:
        wrong=swapindex[(item['model_id'],item['case_id'])]
        correct=rowindex.get((item['model_id'],wrong['recipient_case_id']),{})
        if item['status']=='PASS':
            a=arrays(resolve(out,wrong['raw_path']));b=arrays(resolve(out,correct['raw_path']))
            k=len(a['C']);item['free_output_change_RMS_coordinates']=float(np.sqrt(np.square(a['raw'][k:]-b['raw'][k:]).mean()))
        else:item['free_output_change_RMS_coordinates']=None
    baseline={r['case_id']:r for r in controls if r['model_id']=='Native0'}
    for r in controls:
        b=baseline.get(r['case_id'],{})
        for metric in PRIMARY:
            key=metric+'_wrong_minus_correct';a,z=r.get(key),b.get(key)
            r[metric+'_advantage_change_from_Native0']=a-z if finite(a) and finite(z) else None
    hold=[]
    oldindex={(r['arm'],r['case_id']):r for r in rows if r['phase']=='holdout' and r['step']==1000}
    for row in [r for r in rows if r['phase']=='holdout' and r['step']==5000]:
        old=oldindex.get((row['arm'],row['case_id']),{})
        rec=dict(row)
        rec['comparison_status']='PASS' if row['status']=='PASS' and old.get('status')=='REUSED' else 'NOT_COMPARABLE'
        for metric in METRIC_SPECS:
            value,previous=row.get(metric),old.get(metric)
            rec[metric+'_1000']=previous;rec[metric+'_5000_minus_1000']=value-previous if finite(value) and finite(previous) else None
        hold.append(rec)
    write_csv(out/'per_case.csv',rows);write_csv(out/'per_parent.csv',parents)
    write_csv(out/'train_validation_learning_curves.csv',[r for r in curves if r['phase']=='curve'])
    write_csv(out/'teacher_curves.csv',teacher)
    write_csv(out/'control_swap.csv',controls)
    write_csv(out/'matched_budget_increments.csv',increments)
    write_csv(out/'fit2_probe.csv',[r for r in rows if r['phase']=='fit'])
    write_csv(out/'holdout_regression.csv',hold)
    increment_summary=[]
    for comparison in ('Geo-T1','Full-Geo','Full-T1'):
        for step in (1000,2000,3500,5000):
            for split in ('train','validation'):
                members=[r for r in increments if (r['comparison'],r['step'],r['split'])==(comparison,step,split)]
                row=dict(comparison=comparison,step=step,split=split,requested_parents=len(members))
                for metric in PRIMARY:
                    row[metric]=mean([r[metric] for r in members]);row[metric+'_valid_parents']=sum(finite(r[metric]) for r in members)
                increment_summary.append(row)
    write_csv(out/'matched_budget_increment_summary.csv',increment_summary)
    teacher_summary=[]
    for model in CURVE_MODELS+['T1_fit2_probe_500','T1_fit2_probe_1000']:
        for phase in ('curve','fit'):
            for split in ('train','validation'):
                for t in (.2,.5,.8):
                    members=[r for r in teacher if (r['model_id'],r['phase'],r['split'],r['t'])==(model,phase,split,t)]
                    if not members:continue
                    row=dict(model_id=model,phase=phase,split=split,t=t,requested_parents=len(members))
                    for metric in ('FM_error','D_edge','L5','Yhat_free_area'):
                        row[metric]=mean([r[metric] for r in members]);row[metric+'_valid_parents']=sum(finite(r[metric]) for r in members)
                    teacher_summary.append(row)
    write_csv(out/'teacher_curve_summary.csv',teacher_summary)

    training=[read_json(path) for path in sorted((out/'training').glob('*_status.json'))]
    main_updates=sum(r.get('counts',{}).get('optimizer_updates',0) for r in training if r.get('arm')!='T1_fit2_probe')
    probe_updates=sum(r.get('counts',{}).get('optimizer_updates',0) for r in training if r.get('arm')=='T1_fit2_probe')
    # Controller/worker ledger is authoritative; schema-dependent fields are
    # surfaced verbatim, never inferred from checkpoint filenames.
    accounting=dict(main_optimizer_updates=main_updates,probe_optimizer_updates=probe_updates,total_optimizer_updates=main_updates+probe_updates,evaluation_generations=sum(x.get('sampling_counts',{}).get('complete_rollouts',0) for x in ledgers),
        evaluation_generation_attempts=sum(x.get('sampling_counts',{}).get('rollout_attempts',0) for x in ledgers),
        teacher_forwards=sum(x.get('teacher_counts',{}).get('backbone_forward_attempts',0) for x in ledgers),
        generation_forwards=sum(x.get('sampling_counts',{}).get('backbone_forward_attempts',0) for x in ledgers),
        old_holdout_raw_reused=sum(r.get('status')=='REUSED' for r in rows if r['phase']=='holdout' and r['step']==1000),
        failed_or_interrupted_requests=sum(r.get('status') in ('FAIL','RUNNING','STOPPED') for l in ledgers for r in l.get('rows',[])+l.get('teacher_rows',[])),
        training_ledgers=training)
    if (out/'run_state.json').exists():accounting['run_state']=read_json(out/'run_state.json')
    atomic_json(out/'report_accounting.json',accounting)
    if make_figures:figures(out,panels,rows,curves)
    text=['# Control MVP48 全流程证据报告','',
        f"本轮主训练实际新增 {main_updates}/12000 更新，fit2 新增 {probe_updates}/1000 更新；实际完成完整生成 {accounting['evaluation_generations']}，实际固定 teacher 前向 {accounting['teacher_forwards']}；旧留出 raw 严格复用 {accounting['old_holdout_raw_reused']}。以上来自实际训练/评价账本，未执行项均为 NA/NOT_RUN。",'',
        '固定预算：主训练新增 12000 更新、独立 fit2 新增 1000 更新。所有生成固定原 Euler50，BF16 前向、FP32 状态与积分，C 在 51 状态保持。没有自动扩训、求解器扫描或默认模型变更。','',
        '## 训练与验证：父对象等权','',
        '| 状态 | 面板 | 完成/请求 | 剩余 RMS (% L) | coverage@1%L | Gamma 边 RMS (% L) |',
        '|---|---|---:|---:|---:|---:|']
    for row in [r for r in curves if r['phase']=='curve']:
        text.append(f"| {row['model_id']} | {row['split']} | {row['completed_cases']}/{row['requested_cases']} | {fmt(row.get(PRIMARY[0]))} | {fmt(row.get(PRIMARY[1]))} | {fmt(row.get(PRIMARY[3]))} |")
    text+=['','先对象内平均 C/噪声，再父对象等权。曲线包含 Native 起点、T1/Geo_shared 500 和三臂固定 1000/2000/3500/5000；不存在的历史状态保留 NA，不重训历史。','',
        '## 条件利用、模块增量与拟合探针','',
        'control_swap.csv 逐请求列出 E_wrong−E_correct、正确条件绝对质量和 free 输出变化。替换 C 使用 donor 自身原坐标，保持接收请求 N/K/Z/槽位；分数只对接收请求的原自由 GT/Gamma。非零响应不等于可用控制，正确优于错误也不证明绝对质量足够。','',
        'matched_budget_increments.csv 在同一父对象、同一预算下逐项报告 Geo−T1、Full−Geo、Full−T1。未构造偏向 Full 的加权总分。fit2_probe.csv 与 teacher_curves.csv 的 fit 行记录独立双对象探针；该权重不进入主模型排名。','',
        '## 留出回归与局限','',
        'holdout_regression.csv 使用源 Session C 实际数组，列出固定 5000 对原 1000 的逐请求差值。原 raw 或身份缺失时写 NOT_COMPARABLE，不补跑历史生成。此集合是未参与条件微调、此前已观察的留出回归，不是全新确认集。','',
        '训练/验证生成共同改善才支持当前预算延长的收益；仅 teacher 降低不能代替完整生成。训练改善而验证不改善时需区分迁移；主任务差而 fit2 改善仅支持小任务学习信号。缺失或未完成曲线不作方向结论，不把执行 PASS 当作几何成功。','',
        '历史 BF16 置换 FAIL 保留：10/1008 坐标超旧容差，max_abs=0.064453125。repair_v2 仅按原修订删除精确零面积面与重复副本，未宣称修复水密性。精确拓扑/水密认证 NOT_RUN。','',
        '全部逐 case/父对象表、完整 teacher、资源/失败/执行账本、checkpoint 清单和 code.diff 保留；图件见 figures/manifest.json。']
    (out/'ALL_IN_ONE_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    from .finalize import finalize
    finalize(source,out)
    return accounting

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--no-figures',action='store_true')
    a=p.parse_args();run(a.source,a.out,not a.no_figures)

if __name__=='__main__':main()
