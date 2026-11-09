"""Read-only evidence aggregation, plots and pending report preparation.

No torch, models, scoring or metric redefinition. New timestamped summaries are
append-only. REPORT.md is created once as a pending skeleton, never overwritten.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import csv
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.bench_three_mechanisms_v1 import common as C

OUT=C.OUT
STAGES=('RAW','BASE_POST','CLEAN_FINAL_POST')
CONTRASTS=(('PLAIN200','PATCH200'),('PLAIN200','LOOP200'),('PLAIN267','LOOP200'),('RL_ZERO','RL16'))
KEYS=('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C',
      'ConflictCov_C_interior_weight_fraction','free_minimum_angle_lt5_fraction','free_q_gt10_fraction',
      'free_exact_zero_area_count','free_near_degenerate_count','exact_duplicate_faces','nonmanifold_edges',
      'hinge_diag_isolated_1face_edge_components','hinge_diag_fully_C_disconnected_free_area_fraction',
      'C_F_valid_edge_count','C_F_valid_edge_total_length','graph_valid_edges',
      'opposite_winding_reached_free_area_fraction','actual_free_faces','deleted_free_faces','free_area')


def readlines(path):
    if not path.exists():return []
    return [json.loads(s) for s in path.read_text('utf-8').splitlines() if s.strip()]


def num(x):return isinstance(x,(int,float)) and math.isfinite(x)


def mean(values):
    v=[x for x in values if num(x)]
    return statistics.fmean(v) if v else None


def savecsv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def prepare():
    """Freeze empty actual-view assignments and a plainly pending report."""
    vr=OUT/'visual_review';prior=C.read(C.PRIOR/'visual_review/protocol.json')
    qs=C.read(OUT/'data/eval_manifest.json')['requests']
    if not (vr/'protocol.json').exists():
        expected=[];assignments=[]
        for part,(start,stop) in enumerate(((0,4),(4,8),(8,12)),1):
            aid=f'review_part_{part:02d}';subset=qs[start:stop]
            assignments.append(dict(assignment_id=aid,request_indices=list(range(start,stop)),
                request_ids=[q['request_id'] for q in subset],pages=12,cells=84,
                rule='All seven models and three stages of each request reviewed by the same actual reviewer'))
            for i,q in enumerate(subset,start):
                for stage in STAGES:
                    for label in C.LABELS:
                        expected.append(dict(request_index=i,request_id=q['request_id'],parent_id=q['parent_id'],
                            condition_id=q['condition_id'],noise_index=q['noise_index'],report_group=q['report_group'],
                            label=label,stage=stage,page_id=stage+'__'+q['request_id'],review_assignment=aid))
        protocol=dict(schema='bench_three_mechanisms_actual_review_v1',status='FROZEN',
            frozen_at_utc=datetime.now(timezone.utc).isoformat(),experiment=C.EXPERIMENT,
            criteria_version=prior['criteria_version'],definitions=prior['definitions'],axes=prior['axes'],
            criteria_source_path=prior['criteria_source_path'],criteria_source_sha256=prior['criteria_source_sha256'],
            axes_source_path=prior['axes_source_path'],axes_source_sha256=prior['axes_source_sha256'],
            labels=list(C.LABELS),stages=list(STAGES),requests=12,parents=6,pages=36,cells=252,
            expected_units=expected,manifest_sha256=C.sha(OUT/'data/eval_manifest.json'),
            reviewer_kind='AI_PROXY',blind=False,actual_images_viewed_at_registration=0,ratings_copied=0,
            rules=['Do not read metrics or historical labels to generate ratings.',
                'Allow UNCERTAIN only for actually viewed ambiguous evidence; unseen cells remain null.',
                'Recognizable intentional rods can be support; broken shell remnants are not automatically PRESENT.',
                'Finite-section endpoints alone do not prove gaps; inspect both oblique views.',
                'No unique-source-style grading; shared identity alone is not quality certification.',
                'Three-axis joint means REASONABLE and PLAUSIBLE and PRESENT, not no gaps or no intersections.'])
        C.write(vr/'protocol.json',protocol);C.write(vr/'assignments.json',dict(assignments=assignments))
        for a in assignments:
            rows=[dict(request_id=r['request_id'],label=r['label'],stage=r['stage'],page_id=r['page_id'],
                actually_viewed=False,views_and_sections_checked=False,image_sha256=None,image_receipt_sha256=None,
                tool='view_image',view_event_id=None,viewed_at_utc=None,**{axis:None for axis in prior['axes']},note=None)
                for r in expected if r['review_assignment']==a['assignment_id']]
            C.write(vr/'templates'/(a['assignment_id']+'.json'),dict(schema='bench_three_mechanisms_actual_review_rows_v1',
                status='PENDING_ACTUAL_VIEW',assignment_id=a['assignment_id'],protocol_sha256=C.sha(vr/'protocol.json'),
                reviewer=None,reviewer_kind='AI_PROXY',blind=False,rows=rows))
    instructions='''# 本轮实际图审

按 assignments.json 分三份，每份 4 请求 × 3 阶段 × 7 模型＝84 cells。
真实打开 gallery/<stage>__<request_id>/comparison.png（detail=original）；七行可能显示过小，
可以用本工具 crops 生成原图无缩放裁切，逐组查看。裁切仅帮助看清，不改变相机或截面。
每个 cell 独立填写五轴和中文可见证据，完整三阶段由同一 reviewer 评阅。
完成文件放 completed/review_part_01.json、02、03；不得修改空模板来冒充历史查看。
填写原图与 comparison.json SHA256、真实 view 事件、UTC、actually_viewed 和 views_and_sections_checked。
若使用裁切，额外记录 crop_paths/crop_sha256/crop_view_events；仍保留源原图哈希。
未查看保持 null；不确定标签仅表示真实观察仍无法判断。
不要读取数值评分或旧标签反推图审。AI_PROXY，非盲，非独立人类认证。
'''
    text_once(vr/'ROWS_FORMAT.md',instructions)
    text_once(OUT/'REPORT.md',report_skeleton())
    return dict(status='PREPARED',pages=36,cells=252,ratings_generated=0,model_calls=0,scoring_calls=0)


def text_once(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():path.write_text(text,encoding='utf-8')


def report_skeleton():
    return '''# Bench 三机制固定预算实验报告

**状态：DRAFT / PENDING EVALUATION。当前文件是收口骨架，不包含预判结果。**

## 1. 问题与假说

在同一 EXPOSURE 起点上，分别验证共享两面 PATCH、经过训练的两轮 LOOP、局部质量 RL，
是否改善 Bench 的碎面与几何质量，同时保住 C 采用、主体与支撑。三者是独立快筛，不同时组合。

## 2. 实际方法与对照

- PLAIN200 对照 PATCH200 与 LOOP200；PLAIN267 提供约匹配 Transformer block 计算量的 LOOP 对照。
- LOOP 重复最后四层、双轮 FM 深监督，保持 C 输入；这不是冻结层重复，也不是完整 LiFT 复现。
- PATCH 固定 C-only 两三角共享拓扑；训练/推理共享噪声与速度投影，D0 后恢复同一共享解码合同。
- RL16 与 RL_ZERO 使用同一局部随机过程、mask 和预存噪声；冻结主干，仅更新 rank4 adapter。
- 同30 NATURAL TRAIN父；同12请求、同 C/N/原始 z/查询/相机/截面。不得挑 checkpoint 或挑例。

## 3. 执行与资源

PENDING：填写真实更新、成功/失败/额外调用、时间、显存和并行范围。PyTorch allocated/reserved
不等于进程显存；设备总显存包括用户程序。工程修复与计数见 engineering_events、audits。

## 4. 训练曲线

PENDING：引用 control/finalize.py 生成的 PNG/CSV。PATCH 独立坐标分母、LOOP 双轮目标不同，
不能仅靠跨臂绝对 loss 大小宣布优劣。RL奖励也不能替代真实最终成品评价。

## 5. RAW / BASE_POST / CLEAN_FINAL_POST 完整指标

PENDING：全部177字段与逐请求配对保留，报告关键代价。I 的支持域 NA 必须保留；
共边/共点改善不自动表示非法相交减少。PATCH后端共享恢复是候选流程的一部分。

## 6. 实际五轴图审与固定截面

PENDING_ACTUAL_VIEW：252 cells，AI_PROXY、非盲。合理形状∩合理使用C∩支撑保留
不是无碎面、无重层、无交叉认证。必须另外报告裂缝/悬片与重层标签及不确定性。

## 7. 三个决定

PATCH：PENDING。LOOP：PENDING。RL：PENDING。
先回答各自相对匹配对照是否有稳定改善，再讨论能否保留；不自动合并机制或延长训练。

## 8. 边界与交付

小面板、短预算、已有开发请求；不是普遍泛化认证。不更改默认模型、不自动清理、commit或push。
交付实际权重/状态、固定输入、RAW和POST、完整177字段、配对表、图集、实际图审、账本与恢复入口。
'''


def training_tables():
    summary=[];curves=[];sources={}
    for arm,target in C.UPDATES.items():
        folder=OUT/'training'/arm;statuspath=folder/'status.json';log=folder/'training_log.jsonl'
        status=C.read(statuspath) if statuspath.exists() else {}
        rows=readlines(log)
        if len({r['step'] for r in rows})!=len(rows):raise ValueError('Duplicate training updates: '+arm)
        if rows and [r['step'] for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('Training log gap')
        for r in rows:curves.append({k:v for k,v in r.items() if k!='reference_group_weights'})
        for p in (statuspath,log):
            if p.exists():sources[str(p)]=C.sha(p)
        calls=status.get('calls',{});resources=status.get('resources',{})
        summary.append(dict(arm=arm,status=status.get('status','NOT_RUN'),updates=len(rows),target_updates=target,
            first20_objective=mean([r['objective'] for r in rows[:20]]),last20_objective=mean([r['objective'] for r in rows[-20:]]),
            first20_FM_last=mean([r['FM_last'] for r in rows[:20]]),last20_FM_last=mean([r['FM_last'] for r in rows[-20:]]),
            successful_forwards=calls.get('successful',{}).get('train_forward'),
            attempted_forwards=calls.get('attempted',{}).get('train_forward'),
            successful_backwards=calls.get('successful',{}).get('train_backward'),
            successful_optimizer_updates=calls.get('successful',{}).get('optimizer_update'),
            block_calls_successful_updates=sum(r['block_calls'] for r in rows),
            logged_update_seconds=sum(r['update_seconds'] for r in rows),
            latest_attempt_wall_seconds=status.get('elapsed_seconds'),
            peak_allocated_bytes=max([r['peak_allocated_bytes'] for r in rows],default=None),
            peak_reserved_bytes=max([r['peak_reserved_bytes'] for r in rows],default=None),
            process_peak_mib=resources.get('process_peak_mib'),process_memory_status=resources.get('process_memory_status'),
            total_device_used_peak_mib=resources.get('total_device_used_peak_mib')))
    rlrows=[]
    for p in sorted((OUT/'rl/training').glob('opportunity_*/complete.json')):
        r=C.read(p);sources[str(p)]=C.sha(p)
        rlrows.append(dict(opportunity=r.get('opportunity'),status=r.get('status'),data=json.dumps(r,ensure_ascii=False)))
    return summary,curves,rlrows,sources


def metric_tables():
    p=OUT/'results/completion.json'
    if not p.exists():return [],[],dict(status='PENDING'),{}
    seal=C.read(p);C.verify(seal['files']);rows=C.read(OUT/'results/metric_rows.json')
    if len(rows)!=252:raise ValueError('Incomplete fixed metric panel')
    fields=C.read(OUT/'data/eval_manifest.json')['metric_fields'];summary=[]
    for label in C.LABELS:
        for stage in STAGES:
            subset=[r for r in rows if r['label']==label and r['stage']==stage]
            for group in ['ALL']+sorted({r['report_group'] for r in subset}):
                rr=[r for r in subset if group=='ALL' or r['report_group']==group]
                for metric in fields:
                    parents=defaultdict(lambda:defaultdict(list))
                    for r in rr:parents[r['parent_id']][r['condition_id']].append(r[metric])
                    pm=[mean([mean(v) for v in cases.values()]) for cases in parents.values()]
                    summary.append(dict(label=label,stage=stage,group=group,metric=metric,
                        parent_macro_mean=mean(pm),parents=len(parents),valid_parents=sum(num(x) for x in pm),
                        outputs=len(rr),numeric_outputs=sum(num(r[metric]) for r in rr),NA_outputs=sum(not num(r[metric]) for r in rr)))
    keyed={(r['label'],r['stage'],r['group'],r['metric']):r for r in summary};paired=[]
    for a,b in CONTRASTS:
        for row in summary:
            if row['label']!=a:continue
            br=keyed[b,row['stage'],row['group'],row['metric']];x,y=row['parent_macro_mean'],br['parent_macro_mean']
            paired.append(dict(reference=a,candidate=b,stage=row['stage'],group=row['group'],metric=row['metric'],
                reference_mean=x,candidate_mean=y,delta=y-x if num(x) and num(y) else None,
                reference_NA_outputs=row['NA_outputs'],candidate_NA_outputs=br['NA_outputs']))
    return summary,paired,dict(status='COMPLETE',rows=252,fields=177),{str(p):C.sha(p),str(OUT/'results/metric_rows.json'):C.sha(OUT/'results/metric_rows.json')}


def plots(dest,curves):
    if not curves:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),dpi=150)
    colors={'PLAIN':'#3368ad','PATCH':'#d15a30','LOOP':'#37855b'}
    for arm in C.UPDATES:
        r=[x for x in curves if x['arm']==arm]
        if not r:continue
        for ax,key,title in zip(axes.flat,('objective','FM_last','gradient_norm','update_seconds'),
            ('Training objective (different PATCH/LOOP definitions)','Last-pass FM','Gradient norm before clip','Actual seconds/update (contention included)')):
            ax.plot([x['step'] for x in r],[x[key] for x in r],alpha=.15,lw=.6,color=colors[arm])
            sm=[statistics.fmean(x[key] for x in r[max(0,i-19):i+1]) for i in range(len(r))]
            ax.plot([x['step'] for x in r],sm,label=arm,lw=1.3,color=colors[arm]);ax.set_title(title,fontsize=9);ax.set_xlabel('Completed update');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Bench three mechanisms: actual logs, trailing20 mean')
    fig.tight_layout();fig.savefig(dest/'learning_curves.png');plt.close(fig)


def snapshot(require_complete=False):
    prepare();summary,curves,rlrows,sources=training_tables();metrics,paired,mstate,msources=metric_tables();sources.update(msources)
    visual=OUT/'visual_review/summary_v1/completion.json'
    # A producer COMPLETE, never file presence alone, is required for final closure.
    vstate=C.read(visual).get('status','PENDING') if visual.exists() else 'PENDING_ACTUAL_VIEW'
    rlpath=OUT/'rl/verification.json';rlstatus=C.read(rlpath).get('status') if rlpath.exists() else 'PENDING'
    complete=all(r['status']=='COMPLETE' and r['updates']==r['target_updates'] for r in summary) and mstate['status']=='COMPLETE' and vstate=='COMPLETE' and rlstatus=='PASS'
    if require_complete and not complete:return dict(status='PENDING',training=summary,metrics=mstate,visual=vstate,RL=rlstatus)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');dest=OUT/'report_snapshots'/stamp;dest.mkdir(parents=True)
    savecsv(dest/'training_summary.csv',summary);savecsv(dest/'training_curves.csv',curves);savecsv(dest/'rl_opportunities.csv',rlrows)
    savecsv(dest/'all177_parent_macro.csv',metrics);savecsv(dest/'all177_primary_contrasts.csv',paired)
    savecsv(dest/'key_metric_parent_macro.csv',[r for r in metrics if r['metric'] in KEYS])
    savecsv(dest/'key_metric_contrasts.csv',[r for r in paired if r['metric'] in KEYS]);plots(dest,curves)
    resources=[]
    for base in ('training','evaluation','dispatch','rl'):
        for p in (OUT/base).rglob('*.json'):
            if 'resource' in p.name or p.name=='dispatch.json':
                try:r=C.read(p)
                except (ValueError,OSError):continue
                sources[str(p)]=C.sha(p);resources.append(dict(path=str(p),sha256=C.sha(p),data=r))
    C.write(dest/'resource_evidence.json',dict(records=resources,scope='Measured snapshots; missing PID memory remains NA. Concurrent timings not isolated throughput.'))
    r=dict(status='COMPLETE' if complete else 'PARTIAL',metrics=mstate,visual=vstate,RL=rlstatus,training=summary,
        source_hashes=sources,ratings_generated=0,model_calls=0,scoring_calls=0,
        snapshot=str(dest),files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
    C.write(dest/'snapshot.json',r)
    return {k:v for k,v in r.items() if k not in ('files','source_hashes','training')}


def crops():
    from PIL import Image
    complete=C.read(OUT/'gallery/complete.json');C.verify(complete['files']);records=[]
    for p in sorted((OUT/'gallery').glob('*/comparison.png')):
        with Image.open(p) as im:
            w,h=im.size;top=.11*h;cell=.835*h/(7+6*.29);gap=.29*cell
            for index,(lo,hi) in enumerate(((0,3),(3,5),(5,7)),1):
                y0=max(0,int(top+lo*(cell+gap)-.022*h));y1=min(h,int(top+(hi-1)*(cell+gap)+cell+.012*h))
                dest=OUT/'visual_review/crops'/p.parent.name/f'rows_{lo+1}_{hi}.png';dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():im.crop((0,y0,w,y1)).save(dest)
                records.append(dict(source=str(p),source_sha256=C.sha(p),path=str(dest),sha256=C.sha(dest),
                    box=[0,y0,w,y1],labels=list(C.LABELS[lo:hi]),resized=False))
    C.write(OUT/'visual_review/crops/manifest.json',dict(status='COMPLETE',crops=records,source_pages=36,ratings_generated=0))
    return dict(status='COMPLETE',crops=len(records),ratings_generated=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','snapshot','final','crops'))
    args=parser.parse_args();result=prepare() if args.action=='prepare' else crops() if args.action=='crops' else snapshot(args.action=='final')
    print(json.dumps(result,ensure_ascii=False))
