"""Write a reproducible local report from completed tables; no model/scorer calls."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_seam_hinge_quality_v1'
LABELS = ('PLAIN_S2000', 'CLEAN_EDGE_COH', 'SMOOTH_PRIOR')


def readcsv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def run():
    summary_path = OUT / 'results/final/key_results.csv'
    output_path = OUT / 'results/final/per_output.csv'
    ss, rr = readcsv(summary_path), readcsv(output_path)
    assert len(rr) == 504
    lookup = {(r['label'], r['stage'], r['split']): r for r in ss}
    per = {(r['label'], r['stage'], r['request_id']): r for r in rr}
    fmt = lambda row, key, scale=1, digits=2: f'{float(row[key])*scale:.{digits}f}'
    main = []
    for split in ('TRAIN', 'DEV'):
        for label in LABELS:
            r = lookup[label, 'POST', split]
            main.append([split, label, fmt(r, 'FF_illegal_affected_free_fraction', 100),
                fmt(r, 'CF_illegal_affected_free_fraction', 100), fmt(r, 'ConflictCov_C', 100, 3),
                fmt(r, 'free_minimum_angle_lt5_fraction', 100),
                f"{r['bulk_reach30_count']}/{r['n']}", fmt(r, 'nonmanifold_edges')])
    raw = []
    for split in ('TRAIN', 'DEV'):
        raw.append([split] + [fmt(lookup[l, 'RAW', split], 'FF_illegal_affected_free_fraction', 100) for l in LABELS])
    topo = []
    for split in ('TRAIN', 'DEV'):
        for label in LABELS:
            r = lookup[label, 'POST', split]
            topo.append([split, label,
                fmt(r, 'hinge_diag_isolated_1face_edge_components'),
                fmt(r, 'hinge_diag_isolated_1face_unfiltered_edge_components'),
                fmt(r, 'hinge_diag_vertex_attached_only_free_area_fraction', 100),
                fmt(r, 'hinge_diag_vertex_attached_only_unfiltered_free_area_fraction', 100),
                fmt(r, 'hinge_diag_bulk_only_valid_shared_edge_count'),
                fmt(r, 'hinge_diag_open_edge_count'), fmt(r, 'actual_free_faces'), fmt(r, 'deleted_free_faces')])
    hinge = []
    for label in LABELS:
        for split in ('TRAIN', 'DEV'):
            for stage in ('POST', 'POST_HINGE015', 'POST_HINGE025'):
                r = lookup[label, stage, split]
                hinge.append([label, split, stage.replace('POST_', ''),
                    fmt(r, 'FF_illegal_affected_free_fraction', 100), fmt(r, 'ConflictCov_C', 100, 3),
                    f"{r['bulk_reach30_count']}/{r['n']}",
                    fmt(r, 'hinge_diag_isolated_1face_unfiltered_edge_components'),
                    fmt(r, 'hinge_diag_vertex_attached_only_unfiltered_free_area_fraction', 100)])
    bound = []
    for split in ('TRAIN', 'DEV'):
        for label in LABELS:
            r = lookup[label, 'BOUND015', split]
            bound.append([split, label, fmt(r, 'FF_illegal_affected_free_fraction', 100),
                          fmt(r, 'ConflictCov_C', 100, 3), f"{r['bulk_reach30_count']}/{r['n']}"])
    parent = []
    for parent_id in dict.fromkeys(r['parent_id'] for r in rr if r['label'] == 'PLAIN_S2000'):
        row = [parent_id[:8]]
        for label in LABELS:
            group = [r for r in rr if r['label'] == label and r['stage'] == 'POST' and r['parent_id'] == parent_id]
            assert len(group) == 4
            row += [f"{100*sum(float(r['FF_illegal_affected_free_fraction']) for r in group)/4:.2f}",
                    f"{100*sum(float(r['ConflictCov_C']) for r in group)/4:.3f}"]
        parent.append(row)
    wins = []
    for label in LABELS[1:]:
        for split in ('TRAIN', 'DEV'):
            group = [r for r in rr if r['label'] == label and r['stage'] == 'POST' and r['split'] == split]
            for key in ('FF_illegal_affected_free_fraction', 'ConflictCov_C',
                        'hinge_diag_vertex_attached_only_unfiltered_free_area_fraction'):
                changes = [float(r[key]) - float(per['PLAIN_S2000', 'POST', r['request_id']][key]) for r in group]
                wins.append(dict(label=label, split=split, metric=key, n=len(group),
                                 lower=sum(x < -1e-12 for x in changes), equal=sum(abs(x) <= 1e-12 for x in changes),
                                 higher=sum(x > 1e-12 for x in changes), mean_delta=sum(changes)/len(changes)))
    tail = per['CLEAN_EDGE_COH', 'POST', '73b369b9_n1']
    raw_tail = per['CLEAN_EDGE_COH', 'RAW', '73b369b9_n1']
    source_checks = json.loads((OUT/'training_summary/completion.json').read_text())
    weights = {p: h for p, h in source_checks['source_hashes'].items()
               if p.endswith('.pt') and '/r2_seam_hinge_quality_v1/' in p}
    (OUT/'WEIGHTS.json').write_text(json.dumps(dict(status='RETAINED_FINAL', files=weights,
        live_generator_not_EMA=True, final_global_step=15000,
        smooth_requires_registered_inference_noise_transform=True), indent=2)+'\n', encoding='utf-8')
    decision = dict(status='FIXED_BUDGET_COMPLETE_NO_PROMOTION', paired_post_changes=wins,
                    sources={str(p):sha(p) for p in (summary_path, output_path)},
                    no_new_scoring=True, no_new_model_calls=True)
    (OUT/'decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    text = f'''# SEAM 共边监督、平滑先验与铰链合并：固定预算结果

本轮已实际完成两臂各1000步训练、56份RAW、56次单次官方D0，以及168格CPU铰链比较。**两条训练臂均没有形成全面超过S2000的升级；HINGE有有限的无新增非法面对的局部接边收益，但仍未解决DEV主体接续，也没有修好指定的608/n3裂口。** 保留权重与完整证据，不切换默认，不继续延长。

用户要求的同位置截面已包含：每个新臂28张四阶段图，以及`608/n3`、`3f/n2`固定空间窗口的三模型放大图。图审为实际非盲AI_PROXY，不是人工盲评，也不把渲染成功算作已看图。

## 1. 先看同一POST流程下的终点

FF/CF为原非法面对影响的free面比例，I为原`ConflictCov_C`，不是整个物体的相交面积比例。数值是逐输出宏均值；TRAIN16份、DEV12份。`bulk≥30%`只表示由合法C/seam种子经不含C的合法free共边图可达的bulk面积占比，不是视觉可用率。

{table(['分组','模型','FF %↓','CF %↓','I %↓','<5° %↓','bulk≥30%','非流形边/份↓'], main)}

**CLEAN_EDGE_COH**降低了DEV小角率及I，但FF没有下降，DEV接续仍0/12。TRAIN的I出现明显坏尾：`73/n1`从RAW的{fmt(raw_tail,'ConflictCov_C',100,3)}%到POST仍有{fmt(tail,'ConflictCov_C',100,3)}%，其中近重复覆盖{fmt(tail,'ConflictCov_C_dup',100,3)}%；POST保留{int(float(tail['actual_free_faces']))}张free、删除{int(float(tail['deleted_free_faces']))}张。这个尾部在RAW已存在，不能归为GEO新造，也不能被DEV均值下降掩盖。

**SMOOTH_PRIOR**的RAW FF明显降低，却没有转为最终POST质量优势；DEV FF显著升高，非流形边也增加，TRAIN主体接续少于S2000。这里只否定本次从PLAIN_S1000适配1000步、ρ=0.25、固定原OT对应的配方，不否定所有相关先验。

{table(['RAW FF %','S2000','CLEAN_EDGE','SMOOTH'], raw)}

不能从RAW到POST的变化单独认定D0“不适配”：本轮没有分离D0、恢复锚点、焊接、删面的随机对照；但足以说明更低RAW自交不保证更低终态自交。

## 2. 完整逐父分布与断片诊断

{table(['父ID','S2000 FF%','S2000 I%','CLEAN FF%','CLEAN I%','SMOOTH FF%','SMOOTH I%'], parent)}

下表所有分量统计来自完整未剖切的实际索引网格。合法共边图排除了非法/非流形关系；未过滤共边图保留这些关系，用于区别“物理断开”与“连接不合法”。单面分量不设面积阈值。两种共点挂接面积比例的分母均为该输出实际保留free面积。

{table(['分组','模型','合法图单面','实际共边图单面','仅共点/合法图 %','仅共点/实际图 %','bulk合法边','开放边','保留free','删除free'], topo)}

两条训练臂在TRAIN与DEV的实际共边单面分量数、仅靠共点关联C的面积比例均高于S2000；没有兑现本轮“减少只共点断片”的主目标。DEV实际图单面均值从8.08升至9.17/9.75，仅共点挂接面积从82.25%升至87.49%/86.40%。DEV的bulk合法边数量虽增加，但保留free面数也增加，不能把数量增加单独说成表面更完整。

不能把合法图中的孤岛都称为漂浮片：旧`3f/n2 slot21`在实际共边图里属于含C的166面分量，只是通过三面共边而被合法图切断。相反，旧`608/n3 slot77`确是实际共边单面，靠共点关联含C的134面分量。原报告与新口径同时保留。

完整124字段、额外11项独立版本拓扑字段、分母/NA、逐噪声和逐父配对见[结果目录](results/final/)。[配对增减计数](decision.json)完整保留，不只列有利字段。

## 3. HINGE：有限局部收益，未兑现指定裂口与DEV接续

各阈值都独立从原POST开始；没有与BOUND级联。1.5%L为主观察，2.5%L为压力档，不事后择优宣称主结果。

{table(['模型','分组','阶段','FF %','I %','bulk≥30%','实际图单面','仅共点/实际图 %'], hinge)}

基线S2000的1.5%L铰链操作将TRAIN主体可达覆盖13/16→14/16，新增为`73/n3`；DEV仍0/12。基线接受84次簇合并，新增108条合法FF边、消除22对旧非法FF；没有新增非法CF/FF、丢失旧合法CF/FF或删面。2.5%L接受99次，新增129条合法FF边、消除36对旧非法FF，仍无DEV新增覆盖。完整三模型逐操作记录与拒绝漏斗在[hinge](hinge/)。

两个新臂合计112格的331次接受操作也通过独立保存操作重放和终态审计：没有新增非法CF/FF、丢失原合法边、删面或新增UNKNOWN。CLEAN的TRAIN主体覆盖13→14/16、SMOOTH11→12/16，DEV都仍0/12；这没有扭转两臂相对S2000的综合结论。见[新臂审计](hinge/endpoint_audit.json)。

这些约束不是“零几何损伤”认证：中点合并改变面积与形状，基线有一张原本极小的面面积比达到26.85倍；I也可能在非法面对不变时略增。已检查终态及保存操作，未做连续碰撞路径认证。图上多数整体轮廓几乎不变，部分局部开口变小，不能写成普遍修复碎片。

## 4. 原BOUND015单列，不混用保护合同

{table(['分组','模型','BOUND FF %','BOUND I %','bulk≥30%'], bound)}

CLEAN的`2d/n0`在BOUND后达到约41.23%主体可达，是一个条件性的局部正信号，但该次绑定新增5对非法FF。它既不是原POST的天然接续，也不符合HINGE的零新增FF合同。新臂在相同BOUND下的完整代价已保留，不能只把这一例计成无代价DEV突破。

BOUND也缓解了CLEAN的`73/n1`尾部：I从47.314%降至4.199%，CF/FF非法面对数仍为23/6，主体可达仍0；固定截面中的近贴层仍可见。因此保留这个覆盖改善，同时不把它解释为非法面对消失或主体已经接好。BOUND保存了全28请求图，实际额外图审仅覆盖四个明确记录的代价案例，不冒称全部BOUND盲评完成。

## 5. 你指定的截面：RAW → D0提议 → 恢复锚点 → POST

[全部审阅入口](review/index.html)包含RAW/POST三模型对照、两套28请求四阶段图和HINGE对照。以下两张使用上一轮同一平面、同一位置、同一局部裁框：

![608/n3同位置四阶段](sections/fixed_cases/608b46b0_n3_stages.png)

`608/n3`：基线两段座下截线的间隔在RAW已有，D0后扩大并保留到POST。CLEAN仍有原位置缺口。SMOOTH在旧窄Y窗口没有灰线，是下层移到了更低的位置；完整四阶段图可见，不能当作悬片消失。其座下部分截线更连续，但靠背有更明显的大裂口，这是共同存在的视觉取舍。

基线slot77与slot113相邻候选另一端在RAW相距3.5425%L、POST相距5.235%L，均超出本次2.5%L压力阈值；没有强行绑上。slot77仍是实际共边单面；slot113所属分量从31变32面，但仍未与C共边连通。

![3f/n2同位置四阶段](sections/fixed_cases/3f741759_n2_stages.png)

`3f/n2`：SMOOTH上方短线在四阶段均保留；CLEAN该位置短线不见了，却从RAW起就出现更大的座下间隔，POST没有修复。不能只删除不利视窗或只展示上半部。新模型相同slot编号不代表同一个物理面；本轮比较空间窗口，只有原基线沿处理阶段追踪原slot身份。

`73/n1`高I尾部的截面可见C旁的近贴层，完整空间检测定位以原评分及面对记录为准。三维图看起来简洁，不认证没有近重复覆盖。CLEAN的`85/n0`、`8c/n3`也有新增或更宽的底部缺线；这些观察见[实际图审记录](visual/)。

## 6. 训练合同、资源与实际执行

两臂从同一`PLAIN3000_step1000.pt`活生成器及匹配AdamW/RNG恢复，global14000→15000；每臂1000更新、batch8/micro1、8000曝光，47个父，严格重放旧PLAIN第1001—2000步。同预算对照是既有PLAIN_S2000，起点S1000仅作学习曲线参照。无新父、新评测噪声、额外校准或中途择优。

CLEAN同时使用脏bulk FM权重0.2与干净整边辅助λ0.1；10,610条合格源边、1,805,268次曝光边实例端点搬运核验通过。没有删除/修复源目标，缺少同窗口CLEAN-only臂，因此不能单独归功于整边项。SMOOTH仅对bulk的Z/epsilon及推理Gaussian作质心保持、残差缩至0.25，保留原OT对应；这是满秩无界相关Gaussian，边际方差0.375，不是“有界小三角”保证。

[训练曲线](training_summary/learning_curves.png)及[逐100步时间段误差](training_summary/window100_time_bin_unweighted.csv)保存总FM、seam/bulk及辅助项。最后100步总未加权误差PLAIN/CLEAN/SMOOTH约0.10754/0.11161/0.05936；SMOOTH改变目标速度，数值不能用来直接排名几何质量。

按实测资源渐进启动两个独立训练worker；每臂墙钟约1881秒，主要重叠运行。各worker已测CUDA allocated峰值2811.3MiB、reserved3174MiB；采样到的设备总显存峰值9904MiB含两个worker及其他进程，WDDM未给出可信独立进程显存。并行墙钟不作为对旧串行流程的速度基准。CPU评分/绑定与图集按资源分别并行，记录在[resources](resources/)及[阶段资源](training_summary/phase_resources.json)。

实际预算：16000训练前向/反向、2000优化器更新；新生成器推理2800前向（56×Euler50）、D0共56前向。56RAW全部保存后才执行D0。原模型、数据、评分器及旧报告只读，原基线评分复用；新原定义评分112格、BOUND56格、HINGE168格。没有重新抽噪声，没有自动继续训练。

工程记录：第一次RAW启动因虚拟环境指向旧editable包而在模型加载前导入失败，0模型调用；显式设置当前仓库PYTHONPATH后执行原预算。CPU训练汇总的一次慢路径规范化任务停止后作等价字符串快路径修复，0模型调用/额外科学评分。原错误、源码及hash均保留，未静默重试已消耗的GPU工作。

## 7. 交付、边界与下一步决定

- [最终权重身份](WEIGHTS.json)：两臂最终恢复检查点及相同活参数推理导出，global15000，非EMA；SMOOTH需要本轮噪声变换。
- [完整逐输出表](results/final/per_output.csv)、[逐父配对](results/final/paired_per_parent.csv)、[逐噪声配对](results/final/paired_per_noise.csv)、[非法面对分区](results/final/illegal_pairs_by_role.csv)、[删面账本](results/final/deleted_faces.csv)。
- [逐面四类连接图](topology/final_v2/per_face_connectivity.csv)、[执行核验](audit/execution.json)、[独立数值核验](evaluation_audit/completion.json)。
- [恢复入口](RUN.md)、[固定方法与历史路线边界](METHODS.md)、[源码快照及hash](source_snapshot.json)、[本轮新增代码diff](code.diff)。

本次结果支持保留S2000作为当前同预算参照，HINGE作为有边界的可选CPU整理候选；CLEAN_EDGE与SMOOTH均不晋级默认。**如果下一轮只选一个问题，优先检验能否在保住实际共边连接与主体表面的同时降低bulk内部冲突，而不是继续把单点接近、RAW FF下降或更直的截线当作终态成功。** 本轮没有授权额外臂来分离这些机制，故不据此自动加loss、扩大绑定半径、续训或换默认。

DEV只有3个反复观察的开发父，每父4噪声，不能当作新的独立泛化检验。源款式不是唯一正确答案；本报告保留所有28请求及实质代价，没有按结果换父、换噪声或挑检查点。固定任务完成后停止；无清理、commit或push。
'''
    (OUT/'REPORT.md').write_text(text, encoding='utf-8')
    result = dict(status='REPORT_WRITTEN', inputs=decision['sources'],
                  outputs={str(p):sha(p) for p in (OUT/'REPORT.md', OUT/'decision.json', OUT/'WEIGHTS.json')},
                  model_calls=0, new_scoring=0)
    (OUT/'report_receipt.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
