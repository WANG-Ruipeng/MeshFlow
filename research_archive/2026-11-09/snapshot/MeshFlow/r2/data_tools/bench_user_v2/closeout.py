"""Package existing Bench v2 evidence; no additional geometry/model sampling."""
from __future__ import annotations
from collections import Counter
import csv
import html
import json
from pathlib import Path
from .build import ROOT, OUT, SOURCE, read, resolve, save, sha, csvout


def main():
    m=read(OUT/'manifest.json');s=read(OUT/'data_summary.json')
    vp=read(OUT/'validation/latest.json');v=read(vp['receipt'])
    assert vp['status']==v['status']=='PASS' and v['passed']==len(m['train'])
    assert vp['manifest_sha256']==sha(OUT/'manifest.json')
    assert vp['receipt_sha256']==sha(vp['receipt'])
    original=read(SOURCE)
    protected={str(SOURCE):sha(SOURCE)}
    for row in original['rows']:
        for key,hkey in [('case_path','case_sha256'),('input_path','input_sha256'),
                         ('paired_path','paired_sha256'),('supervision_file','supervision_sha256'),
                         ('source_case_path','source_case_sha256'),('quality_file','quality_sha256'),
                         ('input_obj_path','input_obj_sha256')]:
            if key in row and hkey in row:
                path=str(resolve(row[key]));expected=row[hkey]
                assert sha(path)==expected,path
                protected[path]=expected
    save(OUT/'protected_original_assets_verified.json',dict(status='PASS',expected_hashes_from_frozen_manifest=True,
         case_records=360,DEV_records_unchanged=m['dev']==original['dev'],files=protected))
    oldfile=ROOT/'experiments/r2_bench_data_retirement_20261021/future_pool_v1/selection.csv'
    with oldfile.open(encoding='utf-8-sig',newline='') as f:old={r['condition_id']:r for r in csv.DictReader(f)}
    transitions=[]
    for r in m['conditions']:
        src=old[r['source_condition_id']]
        transitions.append(dict(condition_id=r['condition_id'],source_condition_id=r['source_condition_id'],
            parent_id=r['parent_id'],part_type=r['part_type'] if 'part_type'in r else r['template'],
            old_status=src['future_pool_status'],new_status=r['eligibility'],
            admitted=r['condition_id'] in {a['condition_id'] for a in m['train']},reasons=r['admission_reasons']))
    csvout(OUT/'recovery_transitions.csv',transitions)
    recovered=Counter(r['old_status'] for r in transitions if r['admitted'])
    activeparents={r['parent_id'] for r in m['train']}
    parents=[p for p in m['parents'] if p['parent_id'] in activeparents]
    acceptedstats=dict(s,previous_condition_statuses=dict(recovered),
        genuine_new_parent_IDs=0,existing_rebuilt_parent_IDs=len(parents),
        was_pending_parent_count=sum(p['was_pending_parent'] for p in parents),
        K_range=[min(r['K'] for r in m['train']),max(r['K'] for r in m['train'])],
        N_range=[min(p['N'] for p in parents),max(p['N'] for p in parents)],
        closed_conditions=sum(r['boundary_edges']==0 for r in m['train']),
        open_no_owner_conditions=sum(r['boundary_edges']>0 and not r['source_owner_count'] for r in m['train']),
        owners=sum(r['source_owner_count'] for r in m['train']),aliases=sum(r['source_alias_count'] for r in m['train']),
        original_assets_hash_verified=len(protected),
        CPU_OT_completed=v['new_CPU_OT_completed'],CPU_OT_attempts=v['new_CPU_OT_attempts'],
        CPU_noise_arrays=v['new_CPU_noise_arrays'],validation_wall_seconds=v['wall_seconds'])
    save(OUT/'final_summary.json',acceptedstats)
    codepaths=list((ROOT/'r2/data_tools/bench_user_v2').glob('*.py'))
    codepaths += [ROOT/'r2/data_tools/bench_parts_v1/construct.py',ROOT/'r2/data_tools/bench_parts_v1/quality.py',
                  ROOT/'r2/pilots/bench_parts_convergence_v1/data.py',ROOT/'r2/pilots/seam_abc_v1/slot_data_extension.py']
    save(OUT/'code_hashes.json',{str(p):sha(p) for p in sorted(codepaths)})
    render=read(OUT/'gallery/render_completion.json');audit=read(OUT/'audit/latest.json')
    audit_summary=read(resolve(audit['run'])/'summary.json')
    ready=dict(schema='BENCH_USER_V2_BATCH01_READY',status='DATA_READY_FOR_USER_REVIEW_NO_TRAINING',
        manifest_path=str(OUT/'manifest.json'),manifest_sha256=sha(OUT/'manifest.json'),
        validation_receipt=vp['receipt'],validation_receipt_sha256=sha(vp['receipt']),
        summary_path=str(OUT/'final_summary.json'),summary_sha256=sha(OUT/'final_summary.json'),
        accepted_conditions=len(m['train']),accepted_parents=len(parents),
        training_authorized=False,default_pool_changed=False,
        qualifier='manifest records the pre-CPU admission stage; this receipt certifies the exact same manifest passed all18 CPU packing checks')
    save(OUT/'DATASET_READY.json',ready)
    # New review index references the immutable pre-review renders and actual notes.
    def link(p):return Path(resolve(p)).relative_to(OUT).as_posix()
    esc=html.escape
    blocks=['<!doctype html><meta charset="utf-8"><title>Bench v2 首批实际数据</title>',
       '<style>body{font:17px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;background:#f4f6f8;color:#17212b}article{background:white;padding:22px;margin:24px 0}img{width:100%;height:auto}li{margin:8px 0}a{color:#176ca4}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccc}</style>',
       '<h1>Bench v2：6个父目标，18条已验收条件</h1>',
       '<p>蓝色为真实输入C，灰色为同版本补集Y。零训练、零模型调用。几何验收＋实际AI_PROXY图审＋18/18生产CPU OT打包检查通过。</p>',
       '<p>下方图片保留渲染时的NOT_REVIEWED标注；当前逐项结果以文字和 <a href="visual_review.json">实际图审记录</a> 为准。不是对其余26个父的通过认证。</p>',
       '<p><a href="REPORT.md">完整报告</a> · <a href="accepted_conditions.csv">18条有效清单</a> · <a href="held_conditions.csv">89条暂存清单</a> · <a href="DATASET_READY.json">验收入口</a> · <a href="gallery/review_v1/index_all.html">全部171页候选图册</a></p>']
    for p in parents:
        uid=p['parent_id'];rs=[r for r in m['conditions'] if r['parent_id']==uid]
        blocks += ['<article><h2>'+esc(uid)+' | N '+str(p['N_original'])+' → '+str(p['N'])+'</h2>',
          '<p>'+esc(p['review']['notes'])+'</p>',
          '<p><a href="'+link(Path(p['target_case_path']).parent/'target.obj')+'">完整M.obj</a> · <a href="gallery/review_v1/parents/'+uid+'.png">原/新目标两视角</a></p><ul>']
        for r in rs:
            desc=r['review']['observed_extent']
            yes=r['condition_id'] in {a['condition_id'] for a in m['train']}
            d=resolve(r['case_path']).parent
            blocks.append('<li><b>'+('ACCEPT' if yes else 'HOLD')+' · '+esc(r['template'])+' · K='+str(r['K'])+'</b> '+esc(desc)+
                ' <a href="'+link(d/'C.obj')+'">C.obj</a> / <a href="'+link(d/'Y.obj')+'">Y.obj</a> / <a href="'+link(d/'input.npz')+'">C-only输入</a></li>')
        blocks+=['</ul><img loading="lazy" src="gallery/review_v1/overview/'+uid+'__p01.png"></article>']
    (OUT/'index.html').write_text('\n'.join(blocks),encoding='utf-8')
    table='\n'.join('| '+p['parent_id'][:6]+' | '+str(p['N_original'])+'→'+str(p['N'])+' | '+str(p['zero_faces_removed'])+' | '+str(sum(r['parent_id']==p['parent_id'] for r in m['train']))+' |' for p in parents)
    parttable='\n'.join('| '+k+' | '+str(n)+' | '+str(s['part_parent_counts'][k])+' |' for k,n in s['accepted_parts'].items())
    report=f'''# Bench v2 目标与条件重建：首批实际交付

**已完成限定的首批数据制作：32个父目标提案、107条C/Y候选，最终6个父／18条条件通过完整目标几何检查、实际AI_PROXY图审及生产CPU打包验收。** 不是32个PASS；不把这6个父直接替换原79父训练池。本轮未训练、未调用生成器／D0／占用头、未运行GPU计算，未commit/push。

## 1. 先看实际数据

- [已验收数据图册与OBJ下载](index.html)：6父原/新对照、C单独显示、C+Y、每条范围说明。
- [全部候选图册](gallery/review_v1/index_all.html)：32父、107条件，共171页；未入池项没有隐去。
- [机器验收入口](DATASET_READY.json)、[冻结manifest](manifest.json)、[18条有效条件CSV](accepted_conditions.csv)、[89条暂存条件CSV](held_conditions.csv)。
- [实际图审](visual_review.json)、[恢复／隔离状态转移](recovery_transitions.csv)、[父级清单](parents.csv)。

蓝色是真实来源三角面组成的新版本C，不是模型生成图。每父先选少量有意义部位，所有C/Y从同一目标版本取互斥补集。语义标签只用于制作与记录，不输入模型。原始资产、历史321条件及39 DEV记录均保留。

## 2. 本批到底做了什么

首批上限32父在完整几何结果出来前锁定：先放入原清单有BACK候选的21父，再依据精确缺陷、此前待核验来源和部位覆盖确定其余11父；不依据生成好坏选择。剩余47父只在队列中，未冒充已重建。

父级确定操作：移除有效精确零面积面；相同几何副本归并仅作为单表面提案；按共享边奇偶约束整理可解绕序；低于N128的提案只作保表面的质心细分。没有删除小而非零的面，没有D0清洗、焊点、磨平、细腿加粗或独立归一化。去重的作者意图仍为UNKNOWN；复杂提案未通过就暂存。

完整目标检查沿用已有有限三角形检测器，同时保留非法和未决分类；要求无检测到的非法/未决面对、无非流形边、绕序可解、无有效零面积以及表面来源映射成立。这是本版本检测器与容差下的验收，不是无限精度几何证明，也不保证外向法线、水密或所有近接触都符合用户意图。没有要求全部部件连成单一组件。

**6个最终通过父没有执行去重、细分或翻面，只共删除41张精确零面积面；全部正面积面、原顶点三套坐标和归一化保持。** 独立复核面积差低于1e-12量级。不能把本批说成已经解决双面薄壁单层化或复杂相交修复。

| 父ID前缀 | 原N→有效N | 删除精确零面积 | 通过条件数 |
|---|---:|---:|---:|
{table}

这些都是已有TRAIN来源，新增真实父ID为0；其中{acceptedstats['was_pending_parent_count']}个父此前属于待补审来源。新条件在同一父内不增加独立形态数。原坐标和面身份的旧→新／新→旧映射保存在每父`lineage.json`。

## 3. 条件不是随机抽面

| 输入类别 | 条数 | 独立父数 |
|---|---:|---:|
{parttable}

K范围{acceptedstats['K_range'][0]}–{acceptedstats['K_range'][1]}，N范围{acceptedstats['N_range'][0]}–{acceptedstats['N_range'][1]}。包括规则长板、连续L形座面、门形支架侧表面、圆角框及座背／座体支撑组合。明确区分单翼靠背、上部一条背板、座面前部矩形带与完整座面；没有都改名FULL。部分输入是部件的完整侧表面，并不是闭合三维整腿。

实际图审6个父及20条候选，接受18条，暂存`3e4f4c`的座面区域及座背组合2条：C单独显示仍有不规则端部三角缺口和折面，不能靠整椅上下文把它认证成规整座板。其他26父的87条件因目标几何阻塞未进入语义验收，不伪称全部107条已人工查看。

本批18条全部是开放且有来源owner的条件，闭合C与无owner条件为0；适配器保留对应路径，但本批不宣称实测覆盖它们。`a8e987`仅1条已认证输入，不强凑2条。4个靠背来源相对旧候选的3个是恢复覆盖，但全池收缩为6父，不能把它说成训练总体覆盖扩大。

## 4. 44条待审补齐结果与26个未修好父

44条／15父相同口径检查实际完成：17条通过原两项精确门槛；4条有C/Y整面重合（10对），24条有效Y有精确零面积，27条有Y内部重复，类别可交叠。通过两项不等于整个父干净。已有277条精确审计复用核对，未重复大审计。

32父提案中26个未通过完整目标门槛：13个仅非法有限面对，7个仅非流形边，5个两者兼有，1个两者兼有并存在未决面对。合计19父仍有非法关系、13父有非流形边、1父有未决；这三个数不可相加。逐对证据在每父`geometry.json`，原件未删。

因此这次成功的是统一父目标／C/Y／来源身份／验收流水线及6个可用父的恢复，**不是仅靠去重就修好了大部分Bench**。没有用删掉相交正面积面、拆点掩盖非流形或强行布尔拼接去凑PASS。需要真实几何重建或表达判断的对象保留具体BLOCKED原因；余下47父和程序化扩充留到后续批次。

18条有效条件的旧状态分布为`{dict(recovered)}`。这些恢复均使用新target版本、新图审和新打包验收，不自动恢复先前排除的26条旧候选及3条新候选。

## 5. 实际CPU验收与资产合同

独立验收**18/18 PASS**，18次生产CPU OT尝试全部完成，无重试，耗时{v['wall_seconds']:.3f}秒。为此产生36个验证专用噪声数组／{v['new_CPU_noise_scalars']}个标量；没有生成模型输出，也未预先创建训练噪声或训练日程。

核对case与父目标vertices/faces、canonical及逆映射、C+Y精确还原、owner/bulk每张有效Y恰好一次、alias不重复实例化、所有固定角点零速度、有效FM分母与padding、OT后HYBRID以及共同增强的代数一致性。18条总来源owner={acceptedstats['owners']}、alias={acceptedstats['aliases']}；这是监督来源关系，不是推理可用的隐藏标签。

每条`input.npz`仅含C、索引/角序及N等输入身份；完整目标、owner/alias位于独立监督文件与case。任务仍是给定真实C的开放完成，不用唯一来源款式定义未来生成成功。旧DEV39／8父逐字段保持且与TRAIN父不交叠；没有新泛化测试集，也未认证跨源近重复组。

另按冻结旧manifest检查{len(protected)}个唯一资产的已登记hash，360个case记录全部覆盖。原数据、旧清单、旧评分及权重均未写入。本批后续曝光政策见`sampling_policy.json`：先父对象，再其合法部位/条件，不用全局BACK配额反复放大少数父；未冻结任何训练步数。

## 6. 资源、工程记录与恢复

开始时RTX5080总显存16303MiB，空闲约10GiB；24逻辑CPU、WSL可用RAM约61GiB。全程GPU worker=0，其他用户GPU进程未干预。父级独立CPU构造4worker、每worker单数值线程；渲染1worker实测2线程；验收单worker单线程。审计资源与各阶段记录独立保存，不拿并发墙钟当孤立性能基准。

补审{audit_summary['wall_seconds']:.2f}秒；完整目标32父构造／评分本次15.16秒；渲染171页{render['wall_seconds_reported_by_renderer']:.2f}秒；独立映射复核2.84秒；CPU OT验收{v['wall_seconds']:.2f}秒。条件构造没有独立墙钟计时字段，记NA，不拼造总运行秒数；实际审阅与工程时间另计。渲染首次启动因缺少可选threadpoolctl退出，尚未产图；随后用环境变量限线程的无安装fallback完成，故障保留于`gallery/render_engineering_attempts.json`。无训练/采样重放。

恢复总入口：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.data_tools.bench_user_v2.run verify
```

`build`、`render`、`finalize`、`validate`分阶段可恢复；已有记录先验hash，失败不静默重试。新父或规则变更必须新版本/批次，不能覆盖本批。`DATASET_READY.json`绑定最终18条件manifest及CPU验收收据，原manifest保留验收前阶段名称，避免验收后改hash。

## 7. 本轮收口

这是能直接查看和复算的数据小批次。当前最值得先看的，是[有效图册](index.html)里几种完整板面与支撑输入是否符合你想要的使用方式。**6父规模不足以替代原79父来研究泛化，也没有实验依据说它能改善生成质量。** 后续应按相同标准继续处理剩余来源，并单独处理穿插／非流形的父目标；不因本批PASS稀少而放宽原门槛，也不让这18条被过度重复来冒充扩数据。本次在数据交付处停止。
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    artifact_files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ('artifact_hashes.json',)]
    # Hash this sealed delivery snapshot; later verification logs do not rewrite it.
    save(OUT/'artifact_hashes.json',{str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in sorted(artifact_files)})
    print(json.dumps(acceptedstats,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
