"""Read completed evidence only; write the final report and a small artifact receipt."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = 'A_S1500'
CAND = 'COVERAGE_S1500'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def csvread(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |'] +
        ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])

def main():
    review = read(OUT/'visual_review/summary.json')
    assert review['status'] == 'COMPLETE' and review['reviewed_cells'] == 288
    audit_link = read(OUT/'final_verify_latest.json')
    assert audit_link['status'] == 'PASS', audit_link
    receipt = Path(audit_link['receipt'])
    audit = read(receipt)
    assert audit['status'] == 'PASS'
    train = read(OUT/'training/COVERAGE_SEAM/status.json')
    assert train['local_step'] == 1500 and train['status'] == 'COMPLETE'
    met = csvread(OUT/'analysis/key_metrics.csv')
    def m(label, stage, split, metric):
        rs = [r for r in met if (r['label'], r['stage'], r['split'], r['analysis_id']) == (label, stage, split, metric)]
        assert len(rs) == 1, (label, stage, split, metric)
        return float(rs[0]['parent_macro_mean'])
    quality = [
        ('CF', 'CF_illegal_affected_free_fraction'),
        ('FF', 'FF_illegal_affected_free_fraction'),
        ('I（C域）', 'ConflictCov_C'),
        ('小角<5°', 'free_minimum_angle_lt5_fraction')]
    quality_rows = []
    for split in ('TRAIN', 'DEV'):
        for stage in ('RAW', 'BASE_POST', 'CLEAN_FINAL_POST'):
            vals = [f'{m(BASE,stage,split,k)*100:.2f} → {m(CAND,stage,split,k)*100:.2f}' for _, k in quality]
            quality_rows.append([split, stage] + vals)
    quality_table = table(['面板', '阶段', 'CF %', 'FF %', 'I %', '小角面 %'], quality_rows)
    final_rows = []
    for name, key, multiplier in [
        ('保留free面数', 'actual_free_faces', 1),
        ('原free删面数', 'deleted_free_faces', 1),
        ('非法FF面对数', 'FF_illegal_pair_count', 1),
        ('非法CF面对数', 'CF_illegal_pair_count', 1),
        ('非流形边数', 'nonmanifold_edges', 1),
        ('单面边连通分量数', 'hinge_diag_isolated_1face_edge_components', 1),
        ('合法C/F共边数', 'C_F_valid_edge_count', 1),
        ('合法C/F共边长度（model unit）', 'C_F_valid_edge_total_length', 1),
        ('seam→bulk可达面积 %（适用请求）', 'free_only_reached_bulk_area_fraction', 100)]:
        vals = [f'{m(BASE,"CLEAN_FINAL_POST",s,key)*multiplier:.3f} → {m(CAND,"CLEAN_FINAL_POST",s,key)*multiplier:.3f}' for s in ('TRAIN','DEV')]
        final_rows.append([name] + vals)
    structure_table = table(['FINAL指标（每父宏均值）', 'TRAIN A→COVERAGE', 'DEV A→COVERAGE'], final_rows)
    review_rows=[]
    for split in ('TRAIN','DEV'):
        for stage in ('RAW','BASE_POST','CLEAN_FINAL_POST'):
            a,b=[next(r for r in review['summary'] if (r['split'],r['stage'],r['label'])==(split,stage,label)) for label in (BASE,CAND)]
            pref=next(r for r in review['preferences'] if (r['split'],r['stage'])==(split,stage))
            review_rows.append([split,stage,str(a['n']),
                f'{a["C_use__YES"]} → {b["C_use__YES"]}',
                f'{a["whole_shape__COMPLETE"]} → {b["whole_shape__COMPLETE"]}',
                f'{a["layered_or_conflicting__YES"]} → {b["layered_or_conflicting__YES"]}',
                f'{a["visible_fragment__YES"]} → {b["visible_fragment__YES"]}',
                f'{pref["COVERAGE"]} / {pref["A"]} / {pref["TIE"]} / {pref["UNCERTAIN"]}'])
    review_table=table(['面板','阶段','请求数','C采用YES A→新','主体COMPLETE A→新','层叠YES A→新','碎片/破口YES A→新','偏好新/A/平/不定'],review_rows)
    parent_rows=[]
    pp=csvread(OUT/'analysis/paired_parent_metrics.csv')
    ids=sorted({r['parent_id'] for r in pp if r['split']=='DEV'})
    for pid in ids:
        vals=[]
        for _,key in quality:
            r=next(r for r in pp if r['stage']=='CLEAN_FINAL_POST' and r['split']=='DEV' and r['parent_id']==pid and r['analysis_id']==key)
            vals.append(f'{float(r["delta_candidate_minus_reference"])*100:+.2f}')
        parent_rows.append([pid[:6]]+vals)
    parent_table=table(['DEV父','ΔCF pp','ΔFF pp','ΔI pp','Δ小角 pp'],parent_rows)
    rfinal=next(r for r in review['preferences'] if r['split']=='DEV' and r['stage']=='CLEAN_FINAL_POST')
    a_final,b_final=[next(r for r in review['summary'] if (r['split'],r['stage'],r['label'])==('DEV','CLEAN_FINAL_POST',label)) for label in (BASE,CAND)]
    artifacts={}
    rels=['PROTOCOL.md','registration.json','data/manifest.json','data/schedule.json','eval_manifest.json',
        'training/COVERAGE_SEAM/status.json','training/COVERAGE_SEAM/generator_step1500.json',
        'results/per_output.csv','results/paired_per_output.csv','results/paired_per_parent.csv',
        'analysis/key_metrics.csv','analysis/paired_parent_metrics.csv','analysis/training_summary.json',
        'analysis/training_curve.png','preparation/visual_reviews.json','visual_review/per_cell.csv','visual_review/paired_requests.csv',
        'visual_review/summary.json','gallery/index.html','RUNBOOK.md','cpu_gate.json','final_verify_latest.json']
    rels.append('resources_final.json')
    for rel in rels:
        p=OUT/rel
        artifacts[rel]={'bytes':p.stat().st_size,'sha256':sha(p)}
    # These eight pages were actually inspected by the root in addition to the full review.
    spot_specs=[
        ('RAW','9fd5','SEAT_SURFACE','n0','蓝色座面明显高于灰座面；两模型均存在替代层。'),
        ('CLEAN_FINAL_POST','9fd5','SEAT_SURFACE','n0','错层保留，灰支承变细/缺失；未观察到采用修复。'),
        ('RAW','9fd5','LEG_SUPPORT','n0','蓝色U形侧框位于长椅端部，两模型整体均可解释，差异较小。'),
        ('CLEAN_FINAL_POST','459f','SEAT_SURFACE','n0','新模型灰主体比例更接近长椅、靠近蓝C，但灰座面仍位于其下方。'),
        ('RAW','459f','SEAT_SURFACE','n0','新模型改善灰主体比例；蓝C仍未明确成为唯一座面。'),
        ('CLEAN_FINAL_POST','f495','LEG_SUPPORT','n0','蓝色中央立柱高出灰座面，整体相近，C用途仍有歧义。'),
        ('CLEAN_FINAL_POST','2d1e','BACK_SURFACE','n0','新模型主体更像四腿长椅，背部仍见叠层，不能据主体变好认定接续成功。'),
        ('CLEAN_FINAL_POST','5dda','SEAT_SURFACE','n0','蓝木条高于灰座面，两模型均存在分层及支承破碎。')]
    spots=[]
    for stage,prefix,part,noise,note in spot_specs:
        hits=list((OUT/'gallery').glob(f'{stage}__{prefix}*__{part}*__{noise}.png'))
        assert len(hits)==1
        spots.append(dict(image=str(hits[0].relative_to(OUT)),image_sha256=sha(hits[0]),note=note))
    (OUT/'visual_review/root_spotcheck.json').write_text(json.dumps(dict(reviewer='AI_PROXY_ROOT_NONBLIND',
        scope='Additional actual spotchecks; excluded from 288-cell aggregate to avoid duplication',rows=spots),ensure_ascii=False,indent=2),encoding='utf-8')
    def img(stage,prefix,part,noise):
        hits=list((OUT/'gallery').glob(f'{stage}__{prefix}*__{part}*__{noise}.png'))
        assert len(hits)==1
        return 'gallery/'+hits[0].name
    report=f'''# R2_BENCH_COVERAGE_EXPANSION_V1 实际实验报告

日期：2026-10-21。状态：**已完成1500步训练、48请求完整推理、144项新阶段评分、144页实际配对图审及独立核验**。

## 1. 先回答结果

**恢复并扩充Bench条件覆盖带来了局部视觉与数值改善，但本轮未形成足以替换A的综合升级。** 训练池从195条/58父扩到321条/79父；固定1500步后，DEV最终小角面比例45.38%→42.62%，C域冲突覆盖I为8.54%→7.78%。与此同时，CF为6.08%→7.59%，FF为26.97%→27.84%，单面边连通分量15.09→17.13；seam→bulk可达面积没有提高。

完整非盲AI_PROXY图审中，DEV最终32请求偏好COVERAGE/A/相近/不确定为 **{rfinal['COVERAGE']}/{rfinal['A']}/{rfinal['TIE']}/{rfinal['UNCERTAIN']}**；明确C采用为 **{a_final['C_use__YES']}→{b_final['C_use__YES']}**。这些是实际看过两视角与固定截面的判断，不是由接缝指标自动生成的用途标签，也不是人类独立盲评。

**整体视觉有局部正信号，值得和控制收益分开保存：** DEV主体完整22→24，明显碎片/破口16→14，整体偏好也更偏向新模型。与此同时，C采用的不确定项2→5，明确YES减少，不能把灰色主体更好看直接解释成条件控制更好。拓扑单面分量增加与视觉破口减少并不矛盾，它们测量的不是同一件事。

**这没有证明“Bench数据已经够了”，也没有证明“增加覆盖没用”。** 本轮同时改变父数量、部位覆盖和曝光分配，保留了COMMON12000起点、固定推理布局及未清洗的来源目标；每父曝光反而从206–208降为150–152。能得出的结论是：在这些固定条件下，这一版扩池策略没有解决宏观座面替代、错层和主体接续。

## 2. 为什么做这一轮，具体改了什么

之前Bench坏例并非只有毫米级未焊接：蓝色座面上方/下方另长灰座面，蓝色支承插在主体中部，RAW已有宏观分层；D0和几何后处理还可能把细腿清成骨架。最近A、扰动和异步候选没有形成综合升级。只读数据审计进一步发现，旧活动池用“有来源owner接口”筛选任务，把合理闭合零件和无需在来源上接缝的输入也排除了，腿类尤其受影响。

本轮只检验数据覆盖策略，不叠加新损失、参数或后处理。最近历史路线及问题边界在[预注册合同](PROTOCOL.md)、[数据链审计](../r2_bench_data_chain_audit_20261021/REPORT.md)和[数据解释](preparation/DATA_COVERAGE_INTERPRETATION.md)中记录。它不是重新命名旧去重实验，也不是单纯延长训练。

| 组成 | 条件数 | 独立父变化 |
| --- | ---: | --- |
| 旧A活动池，原样保留 | 195 | 58父 |
| 恢复已有真实审核条件 | 82 | 涉及37父，其中6父原先完全被排除 |
| 新构造且实际图审通过 | 44 | 15个本地条件训练池新父 |
| 最终TRAIN | **321** | **79父** |
| 保持隔离的DEV库存 | 39 | 8父，实际评价用其中16条C |

新增44条包含座面12、腿/支承26、靠背4、组合部件2。47个候选中，2条因局部几何问题隔离，1条底面/上表面含义不确定而排除。16个可用未选父中，15个产生合格输入；`4a3f…`未被当前提议器构造出合格区域，保留BLOCKED，未硬造。

恢复82条并非模型首次见到这些任务：原64父已存在于COMMON训练历史。相对旧活动池增加21个父中，6个是恢复，真正相对本地条件训练资产新加入的是15个。

不是所有“座面”条目都表示完整座面：存在单条木条、侧框、顶盖和组合块。部位名称仅用于构造与分组，不输入网络。输入保持来源三角形、K≥2、总N128..256，未去重、修复或重三角化。源父ID不同也不等于已认证几何无近重复；新15父并不等于官方预训练未见父。

冻结12000实例中，旧195条占7370，恢复82条占2350，新44条占2280。腿/支承曝光 **12.63%→26.23%**，座面32.13%→31.33%，靠背7.67%→7.63%。所以新增条目不意味着各部位都得到更多监督。55条闭合C、42条开放但无来源owner的条件走已有普通bulk/FM特例；不凭空创建SEAM。

完整目标质量仍有限：原64父中45父有同ID重复面（超额1334张），33父有190张退化面；新增15父中9父有重复面（超额83张）。本轮没有全量重算Y–Y相交，也没有把反向重复面一概认定为脏面或合法薄壁。**本轮是扩大覆盖，不能称为目标净化。**

## 3. 训练实现、预算与实际执行

起点精确复用A的step0：COMMON累计12000步的live生成器和相同fresh AdamW，不从A1500或旧SEAM终点续训。普通Native/T1＋role＋Geo＋同步SEAM保持不变；新增网络参数0。来源owner只用于训练监督，推理继续冻结旧C-only布局，未用来源oracle。

每更新4个不同父，每父原尺度和XYZ独立U[0.92,1.08]一对实例；batch8/micro1。仍为free-only FM、受限OT后HYBRID0.25；AdamW两组lr=1e-5、betas=(0.9,0.95)、eps=1e-8、wd=0、clip=1。BF16前向、FP32参数/优化器/积分；严格确定性、TF32关闭、SDPA math及CUBLAS工作区合同保持。

实际一次完整执行：**1500更新、12000训练前向、12000反向，local1500/global13500**。训练耗时 **{train['elapsed_seconds']/60:.2f}分钟**。没有额外模型探针、占用头更新或模型失败重试；固定终点评价，没有挑最好checkpoint。每父真实曝光150–152，每条件12–152、中位38。

| 正常前向日志量 | 前100步 | 后100步 | 解释 |
| --- | ---: | ---: | --- |
| 总训练objective | 0.106395 | 0.103107 | −3.09%，不能据此宣布收敛 |
| bulk速度MSE | 0.102605 | 0.100481 | −2.07% |
| seam第三点速度MSE | 0.475922 | 0.306441 | −35.61%，训练状态改善不等于生成采用 |

bulk/seam MSE按窗口累计SSE除以累计有效标量数计算；总objective是各更新记录值的均值。完整窗口在[training_windows.csv](analysis/training_windows.csv)，[训练曲线](analysis/training_curve.png)已实际查看。不同池的A末端seam MSE不能直接当作同任务拟合优劣，因为样本和有效掩码不同。

资源按阶段实测：RTX5080总显存16303MiB，开训可用13648MiB，至少留2048MiB余量。只有一条授权训练臂，使用一个GPU worker；不存在为了并行而新增训练臂。训练全设备已用显存观测峰5949MiB包含其它应用，WDDM未提供可靠本进程显存，不能冒称其为模型进程峰值；PyTorch保留峰约3.13GiB另列。CPU约24逻辑核、61GiB可用RAM。RAW评分与独立D0重叠，CPU评分2worker，图渲染4worker；不把并发墙钟作独立算力基准。

## 4. 固定评价与指标口径

48请求：TRAIN4父×2C×2噪声=16，DEV8父×2C×2噪声=32。保持原C/N、噪声数组、布局、查询、相机尺度、front/side截面位置。48新RAW共2400次Euler50前向，全部RAW保存后单次官方D0共48次。RAW阶段188.32秒、D0阶段63.46秒；RAW CPU评分64.83秒与D0部分重叠，POST CPU评分602.74秒，图渲染14.27秒，人工/AI实际审阅时间另计。

统一流水线：`RAW → 单次D0 → 恢复C及seam锚点 → GEO_C＋来源定向 → BASE_POST → HINGE_OPP015 → ALPHA → HOLE → CLEAN_FINAL_POST`。没有多次D0、RBF或新几何阈值。旧A144行只读复用，新模型144行完整评分；共288行，原177个数值汇总字段及逐输出完整记录保留，不只保留有利指标。

- CF/FF：至少参与一个非法面对的free面占比，**不是整个表面面积百分比**；面对数量另报，防止分母变化掩盖代价。
- I：原固定C域`ConflictCov_C`，**不是全物体自交率**。DEV每模型每阶段30 PASS＋2 LOW_SUPPORT，保留低支持状态，主表沿用原定义；不能把微小差值当作高置信总体改善。
- A-VMR/PMR保留完整表，但共享锚点可让它们很高而主体未接入；不能代替seam→bulk可达。
- 闭合C在TRAIN2份、DEV6份，接口任务为NA。主体接续阈值的有效分母是14和26，不把闭合输入算失败。
- 单面边连通分量是拓扑观察，不能等同全部视觉碎片；大块脱离C的主体也不能一概叫碎片。

按父等权汇总；同一父先平均两噪声/条件。以下箭头均为 **A→COVERAGE**，百分数而非相对百分比。

{quality_table}

RAW的FF接近94%是未整理triangle soup的“受影响面”诊断，不能解读成94%物体面积不可用。RAW的I显著下降也没有自动转化为最终主体接续。后处理大幅改变小角和删面分布，必须分别看。

## 5. 最终几何代价与逐父变化

{structure_table}

**seam→bulk可达≥30%：两模型均TRAIN 0/14、DEV 0/26；其余8份闭合C不适用。** 合法C/F共边数量可以不变，但大主体仍接不进去。全部288行C精确保持为true；这不等于完整物体的原顶点连接或无交认证。

TRAIN的FF比例小降，但非法FF面对数31.75→39.69，原free删面92.5→95.5，不能把比例下降单独包装成自交减少。DEV保留面略增加、删面略减少，同时FF/CF和单面分量增加；这不是简单“多删面才得低I”的单一解释。

{parent_table}

DEV8父中，CF是5父下降3父上升，但坏尾部让总体上升；FF和小角都是4父下降4父上升，I为5降3升。`2d1e` FF下降约20.92pp，`5dda`反而增加14.05pp；`fd355`小角下降约10.01pp，同时FF增加8.55pp、单面分量10.5→29.5。改善与损伤并存，不能靠挑一父或一个均值定胜负。详细差值、有效分母及正负身份在[逐父配对](analysis/paired_parent_metrics.csv)。

## 6. 完整实际图审与截面

三位AI_PROXY按父分工，实际查看全部144页，每页两模型、两3D视角和两固定截面，共288模型记录；每页图hash与冻结rubric绑定。root另实际抽查8页，单独保存，不重复计入。图审非盲，同一请求两模型由同一评阅者比较。UNCERTAIN完整保留，未把它当成NO或成功。

{review_table}

这些数是具体开发面板上的描述，不是总体泛化成功率。该DEV已经在历次研究中反复查看，并非新锁定的独立测试集。层叠与碎片允许同时发生；主体COMPLETE不认证水密，C采用YES也不认证合法共边。

有代表性的具体观察：

- [459f SEAT n0最终图]({img('CLEAN_FINAL_POST','459f','SEAT_SURFACE','n0')})：新模型灰主体的高度/比例更接近蓝色C，但截面仍能看到低一层灰座面；可以记局部视觉改善，不能直接记连接成功。
- [2d1e BACK n0最终图]({img('CLEAN_FINAL_POST','2d1e','BACK_SURFACE','n0')})：灰主体由折裙状转向四腿形态，背部仍有叠层，POST细腿仍有缺口。
- [9fd5 SEAT n0 RAW]({img('RAW','9fd5','SEAT_SURFACE','n0')})和[最终图]({img('CLEAN_FINAL_POST','9fd5','SEAT_SURFACE','n0')})：蓝座面高于另一层灰座面的问题贯穿两阶段，本轮没有把这个宏观错配修好。
- [5dda SEAT n0最终图]({img('CLEAN_FINAL_POST','5dda','SEAT_SURFACE','n0')})：仍有木条分层和支承破碎，I下降不能代替这些观察。
- TRAIN `1dd7 SEAT n0`、`2022 COMBO n1`出现COVERAGE灰座面相对蓝C错层的退化；`122a LEG n1`和`1dd7 SEAT n1`则在POST保住更多支承，属于局部主体保留优势。
- 有些请求RAW支承较完整、POST反而清成细线或断腿；也有RAW被灰条带遮住的蓝板在POST显露。报告保留这些阶段差异，不把后处理改善全算给生成器。

全部页面：[固定截面配对图集](gallery/index.html)。全部评价：[逐模型图审](visual_review/per_cell.csv)、[逐请求偏好及用途转移](visual_review/paired_requests.csv)、[汇总](visual_review/summary.json)。主表中的YES计数只计算明确YES，NO与UNCERTAIN详见完整表。

## 7. 当前能下的研究判断

1. **覆盖筛选确实有偏差，恢复合理输入是有价值的资产修正。** 旧池不应把“无来源owner”直接解释成“不值得训练的用户输入”；本轮已让这些样本在既有普通FM路径合法运行，并真实增加腿/组合件覆盖。
2. **固定1500步下，覆盖扩大没有兑现为稳定的条件采用与整体质量升级。** 有局部主体比例、I、小角的收益，也有错层保留、FF/CF成本、拓扑单面分量增加，主体接续仍无提升。暂不晋级COVERAGE，不改当前默认。
3. **这不能裁决“数据不足”与“模型不行”二选一。** 训练起点已含COMMON12000历史，新阶段只改变1500步；父扩充降低每父曝光；目标含旧重复/退化；冻结布局不变；当前部位类别内部跨度仍大。本轮没有分别隔离这些变量，也未证明收敛。
4. **当前N范围的数据供给已接近用尽。** 本地归档560个Bench源UID中，现成N128..256只有88个；本轮79TRAIN＋8DEV占87个，余1个当前构造器BLOCKED。再往同一库存增加条件，主要是在既有父上增加任务，而不是继续增加大量独立形状。不能把321条输入当321个独立长椅。
5. 保留新数据、来源身份、未纳入原因及本终点作为覆盖对照。若后续继续，优先明确要检验“新覆盖在较早适配阶段是否有效”还是“目标几何质量”；不要从本轮结果自动推出延长步数、添加loss或改默认。本任务到此停止。

## 8. 产物、哈希与恢复

- [推理权重](training/COVERAGE_SEAM/generator_step1500.pt)：521,665,915 bytes，SHA256 `{train['inference_export']['sha256']}`。
- [完整恢复检查点](training/COVERAGE_SEAM/checkpoint_step1500.pt)：1,565,056,363 bytes，含匹配AdamW，SHA256 `{train['checkpoint']['sha256']}`。
- 训练0/500/1500点及滚动安全点保留；正式结果仅step1500。没有自动删除其它checkpoint。
- [数据清单](data/manifest.json)、[日程](data/schedule.json)、[真实曝光日志](training/COVERAGE_SEAM/exposure.jsonl)、[训练调用账本](training/COVERAGE_SEAM/calls.jsonl)。
- [全部逐输出数值](results/per_output.csv)、[逐输出配对](results/paired_per_output.csv)、[逐父完整配对](results/paired_per_parent.csv)、[关键指标及分母](analysis/key_metrics.csv)。
- 新RAW/D0提议及POST在`evaluation/COVERAGE_S1500/`，完整几何级联及评分在`cpu/COVERAGE_S1500/`；来源及旧A只读复用。
- [实现入口](../../r2/pilots/bench_coverage_expansion_v1/resume.py)、[可恢复操作说明](RUNBOOK.md)、[冻结代码和输入hash](registration.json)。已完成入口应识别完成状态，不重训或重生成。
- [独立最终核验]({receipt.relative_to(OUT).as_posix()})：**PASS**，账本、48请求、144新评分、288合并行、144图/288真实图审、输入保护及C保持均核验。核验本身模型前向0、GPU调用0、评分调用0、检查点反序列化0。

预备阶段有渲染目录守卫和记录类型的工程修正，独立保存在`preparation/render_engineering_revision*.json`；不隐瞒为全流程无错误。本次正式训练、生成和评分没有失败的模型重试。没有新增GPU实验、切默认模型、commit或push。
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    artifacts['REPORT.md']={'bytes':(OUT/'REPORT.md').stat().st_size,'sha256':sha(OUT/'REPORT.md')}
    sourcefiles=list((OUT.parents[1]/'r2/pilots/bench_coverage_expansion_v1').glob('*.py'))
    sourcefiles += [OUT/n for n in ('analysis_training.py','analysis_results.py','analysis_review.py','final_verify.py','build_closeout.py')]
    sourcehash={str(p):sha(p) for p in sourcefiles}
    (OUT/'source_hashes_closeout.json').write_text(json.dumps(sourcehash,indent=2),encoding='utf-8')
    result=dict(status='COMPLETE',generated_utc=datetime.now(timezone.utc).isoformat(),
        training_updates=1500,training_forward=12000,training_backward=12000,sampling_forward=2400,
        D0_forward=48,new_requests=48,new_score_rows=144,merged_score_rows=288,
        actual_review_pages=144,actual_review_cells=288,final_audit=str(receipt),final_audit_sha256=sha(receipt),
        report_sha256=sha(OUT/'REPORT.md'),source_hashes_sha256=sha(OUT/'source_hashes_closeout.json'),new_scientific_calls_in_closeout=0,
        inference_export=train['inference_export'],checkpoint=train['checkpoint'],artifacts=artifacts,
        automatic_default_change=False,commit=False,push=False,stop=True)
    (OUT/'completion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','training_updates','new_requests','actual_review_pages','report_sha256')},indent=2))

if __name__=='__main__':
    main()
