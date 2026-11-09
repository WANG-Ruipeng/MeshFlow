"""Build the experiment report from completed, immutable results and actual review."""
from pathlib import Path
import csv, json, statistics

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'results/report_numbers.json').read_text('utf-8'))
rows=data['primary']; visuals=data['visual']
labels=['EXPOSURE','PLAIN1000','LAYOUT1000','LAYOUT_NULL']
def get(label,stage,g='ALL'):
    return next(r for r in rows if (r['label'],r['stage'],r['group'])==(label,stage,g))
def vis(label,stage):
    return next(r for r in visuals if (r['label'],r['stage'],r['group'])==(label,stage,'ALL'))
def fmt(v,percent=False):
    return 'NA' if v is None else f'{v*(100 if percent else 1):.3f}'
def metric_table(stage,group='ALL'):
    out=['| 方法 | CF% ↓ | FF% ↓ | I% ↓ | 小角面% ↓ | 方向合法可达面积% | 保留free面数 |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for l in labels:
        r=get(l,stage,group)
        values=[fmt(r[k],True) for k in ['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction','opposite_winding_reached_free_area_fraction']]
        out.append('| '+l+' | '+' | '.join(values+[fmt(r['actual_free_faces'])])+' |')
    return '\n'.join(out)
vtable=['| 方法 | RAW三轴联合 | FINAL三轴联合 | RAW合理采用C | FINAL明确保留支撑 | FINAL可见碎片/裂口 |',
        '|---|---:|---:|---:|---:|---:|']
for l in labels:
    a,b=vis(l,'RAW'),vis(l,'CLEAN_FINAL_POST')
    vtable.append(f"| {l} | {a['joint']}/12 | {b['joint']}/12 | {a['C_use__PLAUSIBLE']}/12 | {b['support_retention__PRESENT']}/12 | {b['visible_fragment_or_gap__YES']}/12 |")
with (ROOT/'analysis/final_v1/raw_P_vs_NULL.csv').open(encoding='utf-8',newline='') as f:
    response=list(csv.DictReader(f))
nn=[float(r['symmetric_corner_NN_pctL']) for r in response]
def delta(field, reference, candidate='LAYOUT1000'):
    return 100*(get(candidate,'CLEAN_FINAL_POST')[field]-get(reference,'CLEAN_FINAL_POST')[field])
report=f'''# R2_BENCH_LAYOUT_ORACLE_V1 — 实际实验报告

完成日期：2026-10-30。固定两臂各1000更新、6父12请求；本轮不晋级，不启动C→P预测头。

## 1. 直接结论

**本版“来源粗布局＋小读取分支”没有形成可辨别的条件采用或完整主体提升。** 两臂已真实训练至固定终点，36份新RAW和36次官方单次D0已完成，保留144个方法×阶段×请求评分单元、177个原始数值字段和144条实际AI_PROXY图审。

LAYOUT相对同预算PLAIN的最终CF/FF/I略低，但相对保留的EXPOSURE，FF、小角面和可达面积仍有代价；其最终保留free面数也更少。图审没有出现LAYOUT独有的新联合正例。**不能把局部几何指标变好写成整体升级，也不能直接进入预测布局的第二阶段。**

这不是“所有结构布局都无用”的证据。本次P只是最多两个来源区域联合包围盒，新增分支只有111,424参数、单处注入、与主干同为1e-5学习率，预算1000步。它没有测试完整部件表示、显式连接图或更强的布局使用机制。

## 2. 实际做了什么

两臂从同一保留EXPOSURE live权重与匹配AdamW开始：global25500／Native13500，固定新增1000步至global26500／Native14500。没有从官方权重重启，也没有从另一份EMA拼接优化器。

- **PLAIN1000**：原Native/T1＋role＋Geo普通续训。
- **LAYOUT1000**：相同完整目标、C、N和随机流，新增粗布局P读取；P的来源是完整参考网格，因此属于特权信息诊断。
- **LAYOUT_NULL**：与LAYOUT1000完全相同的终点权重，推理时遮蔽P；不是第三条训练臂。
- **EXPOSURE**：固定旧结果参照，直接复用同请求原缓存，不重采样或重评分。

原free-only FM、OT后HYBRID0.25、batch8/microbatch1、原组FM系数、CLEAN/X_WIDTH配对、世界坐标、BF16前向与FP32参数/积分全部保留。布局分支有独立fresh AdamW，原参数动量没有重置；两个优化器的更新记为一次完整候选更新。

P包含BACK_REGION和SUPPORT_REGION两个槽，每槽是相对C中心的框中心、完整跨度和类别one-hot，共8维。没有可靠区域的槽标UNKNOWN；不把UNKNOWN当“该部件不存在”。SUPPORT可覆盖端架或下部组合，**不是完整部件分解，也不保证是一条腿**。只有一个有效槽时其attention上下文为同一全局区域编码，不能宣称做到了逐角点部件分配。

新增出口零初始化，实际缺失1662/8000实例（20.775%，同一CLEAN/X_WIDTH对同时缺失）。C几何、N及原面集合未改变。完整计算图、初始化、优化器和信息边界见[METHODS.md](METHODS.md)。

## 3. 数据与固定评测

训练仍是原30个独立父：OLD22、FOCUS2、HIGH6；没有替换不合格父或增加训练几何。每臂8000实例：OLD6000、FOCUS1000、HIGH1000。每FOCUS父500次，OLD每父272/274次，HIGH每父166/168次，**不是全父均匀采样**。实际曝光和时间/随机流/alpha/FM系数逐项配对通过。

共33个唯一训练/评价条件有布局记录：26个复用真实同版本来源区域，7个训练父通过实际来源三视图检查补充粗支撑区域。9个条件有可靠BACK，31个有可靠SUPPORT。33条件在alpha=0.9/1.0/1.1下独立重算框和token全部一致。12个推理C与对应布局来源C逐角点精确相同。

| 评价身份 | 父 | 请求 |
|---|---|---:|
| 熟悉TRAIN | a8e987、27eb79、bcb4f5 | 6 |
| 已用开发DEV | 459f90、f495bd | 4 |
| 适配留出HOLDOUT | df6fd3 | 2 |

每父两份既有固定噪声，未补抽；这些标签不承诺从未出现在官方预训练中。来源P可用于本轮诊断，但推理部署时未必可得。来源款式不是唯一正确答案。**本面板只有6父12请求，不能拿其计数与历史38/46请求总数直接作前后比较。**

处理链统一为一次官方Bench D0、GEO_C、来源C绕序恢复与free定向、HINGE_OPP015、ALPHA、HOLE。未加入RBF、多步去噪、新后端或新loss。全部36份新RAW先保存，再开始任何D0。

## 4. 数值结果：完整保留取舍

以下先按父内两噪声平均，再父等权。CF/FF是原非法面对影响的free面比例；I是原**ConflictCov_C**条件查询支持域覆盖，不是全网格自交率Ri。小角为free最小内角<5°比例；可达面积是原方向合法图指标，不等于完整边界封合。RAW是独立角点soup，不能拿其索引连通分量直接定义视觉碎片。

### RAW，全6父

{metric_table('RAW')}

### 单次D0＋GEO_C（BASE_POST），全6父

{metric_table('BASE_POST')}

### 完整后端（CLEAN_FINAL_POST），全6父

{metric_table('CLEAN_FINAL_POST')}

LAYOUT相对PLAIN的FINAL：CF下降约{-delta('CF_illegal_affected_free_fraction','PLAIN1000'):.3f}pp、FF下降{-delta('FF_illegal_affected_free_fraction','PLAIN1000'):.3f}pp、I下降{-delta('ConflictCov_C','PLAIN1000'):.3f}pp，但平均保留free面数从61.333降到59.083，方向可达面积从13.921%降到13.177%。相对EXPOSURE，FINAL FF增加约{delta('FF_illegal_affected_free_fraction','EXPOSURE'):.3f}pp、小角增加{delta('free_minimum_angle_lt5_fraction','EXPOSURE'):.3f}pp、可达面积减少{-delta('opposite_winding_reached_free_area_fraction','EXPOSURE'):.3f}pp。因此不作全面质量提升结论。

### DEV最终结果，两父等权

{metric_table('CLEAN_FINAL_POST','DEV')}

DEV的布局候选FF低于PLAIN/NULL，但仍高于EXPOSURE。留出df6上LAYOUT的FINAL FF为{fmt(get('LAYOUT1000','CLEAN_FINAL_POST','HOLDOUT')['FF_illegal_affected_free_fraction'],True)}%，高于PLAIN的{fmt(get('PLAIN1000','CLEAN_FINAL_POST','HOLDOUT')['FF_illegal_affected_free_fraction'],True)}%和NULL的{fmt(get('LAYOUT_NULL','CLEAN_FINAL_POST','HOLDOUT')['FF_illegal_affected_free_fraction'],True)}%；完整逐父/逐噪声表保留，不用DEV平均掩盖这项代价。所有177字段、有效分母及NA均在[逐输出表](results/per_output.csv)、[逐父表](results/per_parent.csv)和[配对表](results/paired_per_output.csv)，未另改评分定义。

## 5. 图审：没有新增的采用—完整—支撑联合成功

三位代理实际查看36页原图（每页四方法、两斜视和原位置front/side截面），记录144条五轴结果、原图与receipt哈希、真实查看事件；必要时用原像素裁剪辅助。**非盲AI_PROXY，不是独立人类认证。** 不确定保留，不由数值反推标签。

“三轴联合”仅指同一输出同时REASONABLE∩PLAUSIBLE∩PRESENT，不意味着无冲突或无碎片。

{chr(10).join(vtable)}

四方法RAW联合的六个身份完全相同，均来自三个熟悉父。DEV未出现合理采用的新增请求；df6两噪声C用途保留UNCERTAIN。LAYOUT相对PLAIN和同权重NULL，各阶段五轴标签没有变化；细小几何变化仍保留在数值与图页中。

- **a8、27、bcb**：RAW可形成合理座面与支撑，四方法接近；BASE/FINAL均大量损失腿、侧架或端框表面，成为细丝、悬离残片或缺口。P没有恢复这些结构。27/n1的候选有更粗腿残段，但仍断开，不计作支撑恢复。
- **459**：蓝灰座面错层持续；n1新输出RAW虽有较完整板条凳，C仍在灰座面下方。后端继续损伤支撑。
- **f495**：n0灰上板仍高于蓝C，后端弧端过渡破损；n1条板主体可辨，但C仍被灰座体包住，没有明确采用升级。
- **df6**：RAW两端有大支撑，中央灰面破碎，C用途不确定；后端大支撑变成残条。候选没有解决。

在本次真实图审口径下，四方法FINAL联合均为0/12；这不能外推为EXPOSURE在全部历史面板上零成功。EXPOSURE有一条支撑UNCERTAIN，已保留。完整[图集](gallery/index.html)、[五轴记录](results/actual_visual_review_rows.json)、[逐请求转移](results/report_visual_transitions.csv)可复查。

## 6. 新分支到底有没有被使用

不是完全没有数值作用：step1000常规训练前向的布局增量/hidden RMS约0.0001403，记录到381105个hidden标量发生变化，无额外探针前向。

但作用很小。LAYOUT1000对LAYOUT_NULL使用相同权重、相同C/N和同一初始噪声；51个保存状态中C均精确一致。RAW自由角点集合双向最近邻差异，12请求均值为 **{statistics.mean(nn):.6f}%L**，范围 **{min(nn):.6f}—{max(nn):.6f}%L**。这个数仅测响应，不是语义质量指标。slotwise位移有较大尾部，不能自动当作对应结构大幅移动；完整轨迹及集合距离一起保留。

| 训练FM均值 | PLAIN | LAYOUT |
|---|---:|---:|
| 首100步 | 0.099386 | 0.099395 |
| 尾100步 | 0.101263 | 0.101268 |

末100步差仅约+0.000005525，两曲线几乎重合。不同训练窗口包含不同时间与样本，首尾均值不是固定任务的独立收敛认证。结合小响应与无新增图审成功，当前证据是：**这版1000步、小粗框分支没有有效改变主体组织；尚未得到值得训练预测器去模仿的来源P收益。** 不能由此分辨是粗框信息不足、读取方式弱、优化作用小还是预算限制；本轮不追加扫描去补答案。

## 7. 执行、资产与完整性

- 两臂各1000成功更新、8000训练前向、8000反向；零失败、零生产重算。
- 新增36 RAW／1800逻辑向量场调用；同count CFG与Euler50定义。36次官方D0，没有额外采样、VJP或教师调用。
- 新评分108单元＋原缓存36单元；36张比较页、144条实际图审。保存全部177字段。
- 训练实际并行两进程，记录墙钟PLAIN1843.14秒、LAYOUT1919.89秒。整卡观察峰约12971MiB，含用户背景进程；nvidia单进程VRAM不可得，allocator峰不能冒充进程峰。采样/D0重新测量后增量派发；各组采样约53—54秒，D0/GEO约32—33秒，不能当成隔离性能基准。
- 独立CPU训练汇总曾有一次切片笔误，保留partial并修复到v2；无生产模型重跑。另有早期CPU测试导入路径修正，详见engineering_events。未隐瞒为全程无工程问题。
- 原模型、原评分器、旧输出和数据未修改；没有清理、切默认、commit或push。

实际方法代码位于`r2/pilots/bench_layout_oracle_v1/`；[源码归档](SOURCE_ARCHIVE.md)包含实际评分/后端等266个源码文件与哈希，`code.diff`记录独立新增代码。收口控制工具另作补充归档。训练权重及完整AdamW恢复状态在`training/PLAIN/`、`training/LAYOUT/`；[权重清单与实际哈希](artifacts/retained_weights.json)、[执行账本](output/EXECUTION_LEDGER.md)、[恢复与核验入口](RUNBOOK.md)列出资产和实际命令。

## 8. 收口决定

**保留本轮权重、曲线、逐输出/逐父结果与图审作为有边界的负结果；继续保留EXPOSURE为研究参照，不晋级LAYOUT，不启动C→P预测头。**

若以后继续布局方向，优先问题应是“提供粗P后，生成器是否能实质利用它来改变主体关系”，而非先优化预测P。当前试验没有证明完整布局路线无效，也没有建立可供部署或蒸馏的有效布局教师。

本固定任务到此停止。完整性核验结果见[final_verification.json](results/final_verification.json)。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')
print(json.dumps({'status':'WRITTEN','report':str(ROOT/'REPORT.md'),'bytes':(ROOT/'REPORT.md').stat().st_size}))
