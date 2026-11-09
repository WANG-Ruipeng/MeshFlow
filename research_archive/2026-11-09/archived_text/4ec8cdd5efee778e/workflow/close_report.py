"""Seal completed, already-computed evidence; no model or geometry imports."""
from pathlib import Path
import csv, hashlib, json, os, subprocess, statistics
from datetime import datetime, timezone
import psutil

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]

def read(path):
    return json.loads((OUT / path).read_text(encoding='utf-8-sig'))

def rows(path):
    with (OUT / path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def put(path, value):
    with (OUT / path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def table(headers, data):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in data])

def main():
    calls = read('analysis_numeric/final/calls/call_audit.json')
    numbers = read('analysis_numeric/final/metrics/numeric_summary.json')
    verdicts = read('analysis_numeric/final/report_helper/protocol_verdicts.json')
    vdone = read('analysis_numeric/final/visual/complete.json')
    cpu = read('workflow/post_four_incremental/complete.json')
    visual = rows('analysis_numeric/final/visual/actual_review_per_output.csv')
    assert calls['status'] == 'PASS' and calls['actual'] == calls['expected']
    assert calls['actual']['training_updates'] == 0
    assert numbers['stage_rows'] == 706 and numbers['original_fields'] == 177
    assert vdone['reviewed_cells'] == 476 and vdone['pending_cells'] == 0 and vdone['ratings_inferred'] == 0
    assert len(visual) == 476 and all(r['review_status'] == 'ACTUALLY_VIEWED' for r in visual)
    assert cpu['outputs'] == 184 and cpu['final_global_full_RAW_verification']
    assert read('analysis_numeric/final/report_helper/calibration_label_reuse_audit.json')['changed_axes'] == 0

    source_checks = []
    for r in read('SOURCE_HASHES.json')['records']:
        original = Path(r['source'])
        archived = OUT / r['archive']
        oh, ah = sha(original), sha(archived)
        assert oh == ah == r['sha256'], r['source']
        source_checks.append(dict(source=str(original), archive=r['archive'], sha256=ah, status='PASS'))

    scientific_processes = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        cmd = p.info.get('cmdline') or []
        text = ' '.join(cmd)
        if p.pid == os.getpid() or 'python' not in (p.info.get('name') or '').lower():
            continue
        if 'bench_full_prior_motif_guidance_v1' in text or 'final_numeric_ready.py' in text:
            scientific_processes.append(p.info)
    assert not scientific_processes, scientific_processes
    def command(args):
        p = subprocess.run(args, text=True, capture_output=True, check=False)
        return dict(args=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    resources = dict(utc=datetime.now(timezone.utc).isoformat(),
                     GPU=command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),
                     GPU_processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']),
                     CPU_logical=psutil.cpu_count(), RAM_available_bytes=psutil.virtual_memory().available,
                     load_average=os.getloadavg(), experiment_workers=scientific_processes,
                     visibility_note='WDDM can omit host process VRAM; aggregate device memory is not experiment-process memory.')
    put('archive/final_resources.json', resources)
    put('archive/final_small_source_verification.json', dict(status='PASS', records=source_checks, new_model_calls=0))

    main_visual = [r for r in visual if r['main_panel'] == 'True']
    assert len(main_visual) == 460
    methods = ['R','G0','G1','G2','G3']
    def count(method, stage, scope, axis='J', value='1'):
        x = [r for r in main_visual if r['method'] == method and r['stage'] == stage]
        if scope == 'cal': x = [r for r in x if r['panel_role'] == 'calibration']
        if scope == 'confirm': x = [r for r in x if r['panel_role'] == 'confirmation']
        if scope == 'unseen': x = [r for r in x if r['cohort'] in ('CONFIRMATION_DEV','CONFIRMATION_HOLDOUT')]
        return sum(r[axis] == value for r in x)
    jtable = table(['方法','校准 RAW /8','确认 RAW /38','校准 FINAL /8','确认 FINAL /38','全组 RAW /46','全组 FINAL /46'],
                   [[m, count(m,'RAW','cal'),count(m,'RAW','confirm'),count(m,'CLEAN_FINAL_POST','cal'),
                     count(m,'CLEAN_FINAL_POST','confirm'),count(m,'RAW','all'),count(m,'CLEAN_FINAL_POST','all')] for m in methods])
    atable = table(['方法/阶段','整体合理 /38','C采用合理 /38','支撑存在 /38','同输出三项交集 /38'],
                   [[m+'/'+s.replace('CLEAN_FINAL_POST','FINAL'), count(m,s,'confirm','body','REASONABLE'),
                     count(m,s,'confirm','adoption','PLAUSIBLE'),count(m,s,'confirm','support','PRESENT'),count(m,s,'confirm')]
                    for s in ('RAW','CLEAN_FINAL_POST') for m in methods])
    def mean(m,s,k):
        x=[r for r in numbers['primary_parent_means'] if r['method']==m and r['stage']==s and r['metric']==k and r['scope']=='CONFIRMATION_ALL']
        assert len(x)==1 and x[0]['status']=='COMPLETE'
        return x[0]['complete_domain_mean']
    metrics=['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C']
    gtable=table(['方法','RAW CF %','RAW FF %','RAW I %','FINAL CF %','FINAL FF %','FINAL I %'],
                 [[m]+[f'{100*mean(m,s,k):.3f}' for s in ('RAW','CLEAN_FINAL_POST') for k in metrics] for m in methods])
    gates=[r for r in numbers['quality_gates'] if r['scope']=='CONFIRMATION_ALL' and
           ((r['reference']=='G1' and r['stage']=='RAW') or (r['reference']=='R' and r['stage']=='CLEAN_FINAL_POST'))]
    gatetable=table(['配对','指标','父均值变化 pp','新增严重坏尾请求数','原门槛结论'],
                    [[f"G3−{r['reference']} {r['stage'].replace('CLEAN_FINAL_POST','FINAL')}",
                      {'CF_illegal_affected_free_fraction':'CF','FF_illegal_affected_free_fraction':'FF','ConflictCov_C':'I'}[r['metric']],
                      f"{100*r['parent_mean_delta']:+.3f}",len(r['severe_tails']),r['status']] for r in gates])
    paired=table(['比较','范围','候选/参照 J','严格新增','保留','丢失','跨父/质量/综合'],
                 [[r['claim'],r['scope'],f"{r['candidate_J']}/{r['reference_J']}",r['strict_new'],r['retained'],r['lost'],
                   f"{r['J_gate_status']}/{r['quality_gate_status']}/{r['protocol_combined_status']}"] for r in verdicts['verdicts']])

    # Count/area support diagnostics are read from original stage accounting, not rescored.
    retention=rows('analysis/final/stage_retention_and_costs.csv')
    retention_table=[]
    for m in methods:
        rr=[r for r in retention if r['label']==m and r['panel_role']=='confirmation' and r['main_panel']=='True']
        assert len(rr)==38, (m,len(rr))
        for r in rr: assert float(r['actual_free_faces__RAW']) > 0
        retention_table.append([m, f"{statistics.mean(float(r['actual_free_faces__RAW']) for r in rr):.2f}",
                                f"{statistics.mean(float(r['actual_free_faces__CLEAN_FINAL_POST']) for r in rr):.2f}",
                                f"{100*statistics.mean(float(r['free_area__CLEAN_FINAL_POST_over_RAW']) for r in rr):.2f}%"])
    rtable=table(['方法','RAW free面均数','FINAL free面均数','逐请求 FINAL/RAW free面积比的均值'],retention_table)
    guide=rows('analysis_numeric/final/guidance/per_output.csv')
    stable=[]
    for m in methods[1:]:
        rr=[r for r in guide if r['arm']==m and r['panel_role']=='confirmation']
        assert len(rr)==38
        v=statistics.mean(float(r['shadow_terminal_corner_rms']) for r in rr)
        stable.append([m,f'{v:.6f}',f'{100*v/9.208138264587333:.6f}%',
                       sum(int(float(r['clip_steps'])) for r in rr),sum(int(float(r['guidance_steps'])) for r in rr)])
    stable=table(['方法','终态shadow角点RMS（模型单位）','RMS/L','触发限幅步数','引导步数'],stable)

    report=r'''# Bench 冻结完整先验的 motif guidance：实际实验报告

**结论：本版不晋级。全部固定实验已完成，没有得到优于现有 EXPOSURE 的条件完成流程，也没有达到相对逐步固定 C 的机制正信号门槛。**

最直接的结果是：19父、38请求确认集上，G3全状态引导的 RAW 联合正例为 **5/38**，G1逐步固定C为 **7/38**；统一后端后，G3为 **4/38**，EXPOSURE为 **15/38**。G3 FINAL相对EXPOSURE的CF、FF、I父均值分别增加 **2.104、5.098、3.575个百分点**。这不能作为解决碎面、条件采用或跨父泛化的升级。

执行完成不等于方法成功。200份新RAW、184次单次D0、568行新增原177字段评分、476个唯一实际图审单元均完成。参数更新0、输入VJP 4860次。原资产、评分定义与历史记录保留。本轮结束，不追加训练、强度、采样、Chair迁移或后端变体。

## 1. 本轮实际比较了什么

冻结官方Bench生成器EMA，以完整状态 `X=[S,F]` 生成，其中S是与输入C固定对应的影子槽。最终输出用原C替换S，free部分不在这一步被修改。四臂使用同一个官方EMA、同一个N、C、影子噪声和完整free噪声数组。

| 方法 | 内部条件操作 | 用途 |
| --- | --- | --- |
| R | 复用EXPOSURE/global25500既有RAW/BASE/FINAL | 现状流程参照，不补生成、不重新评分 |
| G0 | 完整先验自由生成，结束后覆盖C | 检查单纯拼入C的效果 |
| G1 | 初始及每次Euler更新后固定S=C | 同官方先验的硬条件基准 |
| G2 | 预测终点误差对全状态求导，但只将修正施加S | 同强度机制对照 |
| G3 | 同一误差梯度修正S与F | 本轮候选 |

G0—G3是同权重机制比较；G3与R有网络及训练历史差异，是实用流程比较。没有Native role/Geo、SEAM槽或占用头加入官方先验。实际EquiDiT v3使用3N角点输入嵌入、N个面token参与attention，不能把3N误写成attention序列长度。

固定Euler为50次更新、51状态，t=k/50，h=0.02；count CFG=2.0，count桶N//20，null标签801。官方CLI的“50时间点”对应49区间，本轮独立50-update实现与R更新数对齐，不宣称逐位复现原CLI。BF16前向，FP32参数/状态/积分，严格确定性算法、math SDPA、TF32关闭。

引导仅k=5..49执行：

```text
Yhat = X + (1-t) v_cfg(X,t)
omega² = (1-t)² / (t²+(1-t)²) + 1e-6
E = sum((Yhat[:K]-C)²) / (2 omega²)
g = grad(E,X)
d_full = -h λ (1-t)/t · g
a = min(1, 0.005L / max_corner_norm(d_full))
G3: Xnext = X + h v + a d_full
G2: Xnext = X + h v + mask_S · a d_full
L = 9.208138264587333
```

参数全部冻结，每步仅对当前输入求导、更新后detach；没有反传整条轨迹或activation checkpoint重算。G2先按全状态计算同一个限幅系数再屏蔽F。能量是sum，未偷偷改成按K或面积平均。上述近似不是MeshFlow条件后验精确推导，也不保证拓扑接续。

## 2. 面板、校准与评价边界

固定23父×每父1条C×2份原噪声＝46请求。4父8请求用于校准，余19父38请求用于确认。确认包含12个EXPOSURE训练父、4个历史DEV父、3个适配留出父；后两组共7父14请求单列。它们不是首次观察的全新测试集，官方预训练是否包含这些父仍UNKNOWN。

校准固定6b(K4/N133)、a8(K39/N189)、28b9(K10/N384)、bcb(K2/N520)。G3三档λ=0.1/0.3/1.0均得到J=0/8，明确碎片项同为5；按预登记后续排序，父等权RAW CF+FF较小的 **λ=0.3** 被锁定。其他两档仍留档，未接D0，也未用于确认集选参。G2沿用λ=0.3，没有独立调参。

J为同一输出同时满足：主体REASONABLE、C采用PLAUSIBLE、主要支撑PRESENT。UNCERTAIN/UNKNOWN不算J。原五轴和观察笔记均保留，不能从客观指标反推视觉标签。来源完整Y不作为唯一正确款式评分。

CF/FF按原受影响free面比例统计；I保留原固定2048查询及权重分母。确认CF/FF为19父38请求完整域；I固定16个PASS父32请求，abf/945/af原LOW_SUPPORT不混入主I。校准I有3父6请求PASS，28b9为LOW_SUPPORT，a8只有69/2048内部支持点的限制保留。不按内部点数重新归一化，也没有通过缩小有效域来通过门槛。闭合C的开放边接口字段NA，不冒充必须接续的失败。

## 3. 实际图审：条件采用与完整主体

<!-- J_TABLE -->

确认集逐轴结果如下。单项合理不能代替同输出三项同时成立。

<!-- AXES_TABLE -->

预先要求J净增至少2、至少2父各有净增和严格新增，并通过质量门槛。实际配对：

<!-- PAIRED_TABLE -->

G3 RAW相对G1：保留4例、严格新增1例(df6)、丢失3例，净−2。适配未见14请求虽为2对1，但增益只来自一个父，不够构成跨父正信号。

G3 FINAL相对R：保留3例、严格新增1例(f495)、丢失12例，净−11；其中11例明确任务丢失，1例不确定。原R正例里有5份出现明确主体或支撑损失：617/n1、7b/n0、838/n0、838/n1、abf/n0。不能用少量新例抵消这些损失后只报“能生成”。

G3与G2在确认RAW上保留相同5个J、FINAL保留相同4个J，没有严格新增；这不是几何逐位相同，也不是G2已经被充分调优，而是本次同强度下没有观察到直接free梯度带来的额外任务成功。

### 实际看到的代表情况

- bcb/n0：G0/G2/G3能生成有双基座的灰色长椅，但蓝C夹在厚灰座条内部，整体像椅子不能证明C被采用。bcb/n1则把C留在高靠背中段，灰座面位于下方。
- 28b9/n0：灰色长腿主体已有自己的窄座面，蓝C较宽且悬在其上方。28b9/n1的官方臂后处理后主要剩薄条，主体/支撑受损；R仍有两个箱式支撑，但局部空隙并未因此认证消失。
- a8/n0：R在RAW可见的支撑到BASE已大幅损失；G1 FINAL是本校准局部正例。a8/n1中灰三角面覆盖蓝条面，仍存在替代或重层。
- 正例也保留：G3 RAW的5例为91e/n0、77aa/n0、77aa/n1、df6/n1、f495/n1。图片包同时含91e，避免只展示失败。

全部主RAW/FINAL 460单元＋未选λ的16个RAW单元，共476个唯一单元实际打开评阅；48条校准原记录的240个轴标签核对无改动。BASE有全数值/全图，另实际查看a8/n0和28b9/n1的两页五方法阶段图，不宣称全数BASE已逐例评阅。BASE已经混合D0与GEO影响，仅凭它不能进一步判定哪一步独立造成全部损伤。

评阅者为AI_PROXY。校准隐藏方法名，最终比较显示方法名，不是独立人类盲评。本轮对不变R重新同页评阅，92输出×5轴中有 **97/460个实质标签变化**；另有42项仅可见性词汇归并变化。RAW主体/支撑标签未变，FINAL主体11、支撑22项变化。旧标签与逐轴转移完整保留，这些是评阅漂移，不能记为新模型收益。

## 4. 几何代价及实际分母

下表均为确认集父等权；每父两噪声先取均值，CF/FF 19父、I 16父。越低越好。

<!-- GEOMETRY_TABLE -->

RAW的FF数值很高，表示三角面汤里被非法FF面对涉及的free面占比，不是“90%的所有面组合相交”，也不能与历史POST的约9%直接跨阶段对比。相同阶段的五组均用原评分器及原定义。

<!-- GATE_TABLE -->

固定均值上限为每项+2pp，严重逐请求尾部为CF>+5pp、FF>+10pp、I>+10pp。G3 RAW相对G1的平均FF下降3.378pp，但CF有6份严重坏尾、I有2份，因此仍未过质量门。G3 FINAL相对R三项均值均超过上限；对应严重尾部计数10/20/4，三列可能重叠，不能相加当独立请求数。

保留/删面与面积变化也一并报告：

<!-- RETENTION_TABLE -->

确认集原RAW均ΣF=11172、每请求136..708面；FINAL R为3136面、G1为4489面、G3为4119面。不能把这些总和池化后代替父等权比例，也不能把少删面自动当成表面质量更好。完整逐来源槽去向、删面、单面孤岛、最小角、合法接口、绝对非法面对数与面积见原177字段和阶段账本。面积保留率允许超过100%，是实际移动后的面积比，不是正确性评分。

## 5. 梯度确实进入F，但这次没有把影子C拉到目标

<!-- SHADOW_TABLE -->

G0到G3的确认shadow RMS只从3.491375降到3.488434模型单位，约37.916%L→37.884%L，远没有形成可靠的目标拟合。G1的shadow误差为0是每步覆盖定义造成，不能用它证明完整形状正确。

全部4860个引导步均触发0.5%L最大角点修正限幅。确认G3累计S/F的逐步RMS范数和约0.714375/0.111657；这些是路径修正量，不能当净位移。限幅饱和时，在同一状态与梯度下正的全局λ和能量标量会被统一归一化抵消，因此本轮三档标称λ不是三个有效的未限幅强度；浮点数和轨迹仍可能不同，实际距离记录保留。

源码和实际日志已核查：终态shadow取覆盖C之前的状态；最终覆盖只修改S，F保持。VJP经过预测速度，没有错误detach；G3存在非零F修正。N776的77aa/n0在第5步，G2/G3相同状态的gS/gF L2约13211.98/11879.04，缩放系数1.17936e−4；G3实际F修正RMS为0.0016415，G2为0。不是“输入梯度没算到free”或“先覆盖C把损失抹掉”的实现错误。

这支持关闭**当前固定槽对应＋sum能量＋当前时间系数/限幅**的候选配方；并不证明冻结先验guidance普遍无效，也没有单独证明限幅是唯一失败原因。不能据此自动去扫更大λ、换对应、改范数或追加训练。

## 6. 固定预算与真实执行

| 项目 | 预登记 | 实际 |
| --- | ---: | ---: |
| 新RAW | 200 | 200 |
| 逻辑向量场调用（50/输出） | 10000 | 10000 |
| count/null双batch前向样本等价 | 20000 | 20000 |
| 当前输入VJP | 4860 | 4860 |
| 单次官方D0 / GEO | 184 / 184 | 184 / 184 |
| 新RAW/BASE/FINAL评分行 | 200/184/184 | 200/184/184 |
| R复用评分行 | 138 | 138 |
| 总评分行×原字段 | 706×177 | 706×177 |
| 实际非BASE图审单元 | 476 | 476 |
| 新shadow / 新free噪声 | 46 / 0 | 46 / 0 |
| 参数更新、科学重放 | 0 / 0 | 0 / 0 |

调用begin与returned数量分别核对一致，逐输出50步提交均完整。568是新评分阶段行，**不是568次全部面对重算**：RAW完整扫描200、BASE缓存完整扫描184、FINAL完整重扫0，增量候选几何扫描4184次。缓存继承及增量合法性计算按原后端执行，未额外扩展评分。

全部200 RAW完成后才启动D0，RAW身份403项最终完整核验PASS。所有生成、D0、CPU几何和渲染worker均已结束。没有OOM、非有限状态、参数更新或科学计算重放。

### 资源与并行

RTX5080总16303MiB；WSL 24逻辑核、约64GiB RAM。先测已预算正式任务，再逐个增加worker。WDDM没有可靠逐进程显存，使用整卡已用峰值作保守上界，并保留CUDA allocated/reserved的范围区别；不把它写成训练或单进程显存。

| 阶段 | 实际安排/观测 | 耗时含义 |
| --- | --- | --- |
| 首个G3 N520 | 单worker，整卡峰5110MiB | 正式科学计算约7.55秒，不是独立探针 |
| 首个G3 N776 | 单worker，整卡峰5828MiB | CUDA reserved约2.66GB，与整卡峰不同 |
| 确认生成主批 | 逐个扩至2worker，整卡峰8911MiB，最少空闲7069MiB | 各约595.38秒并发墙钟 |
| D0主批 | 2worker，整卡峰约7439MiB，最少空闲8541MiB | 主批约318/322.8秒并发墙钟，另首个正式高N输出 |
| RAW评分 | 2个CPU worker、各单线程 | 817.65/807.49秒并发墙钟 |
| 最后143份CPU后端 | 41份完成后安全交接为4worker，互斥输出 | 活跃窗口约32分28秒，累计CPU约5171秒，4.40输出/分；尾部约559秒为单worker |

静态分区的高N尾部有负载不均；没有为了追求调度数再重复任务。最终渲染等待器约2468秒包含等候几何产物，不是纯绘图速度；全阶段缓存索引最终约10.5秒。并发墙钟不作为孤立性能基准。逐5秒资源、每阶段日志/收据保存在workflow，最终只读资源快照见archive/final_resources.json。

### 已记录的工程修复

1. 初始journal形参path与元数据字段重名，在第一次模型调用前失败；保留原现场，仅改日志形参。0前向、0VJP、0科学重放。
2. 77aa闭合C的导出函数原来拒绝空开放边。该输出50前向/45VJP/51状态已保存；仅CPU恢复导出，开放边数组设合法空集合，不重新跑模型。
3. CPU反复验证大屏障文件拖慢调度：保留原评分/GEO源码，先完整核验后在worker内校验未变化的元数据，最后再完整核验。仅在安全任务边界交接旧worker；超时等待日志保留，没有中断在途几何或重算评分。
4. 汇总等待器首次缺PYTHONPATH，在导入阶段退出；修正环境后读取已完成结果。0科学调用。
5. 最终人工记录中4条校准G3的cell_id缺少锁定λ后缀，按冻结任务表修正元数据。实际图片、hash、五轴标签和笔记均未改；原记录、修正收据和v2合并版全部保留。

因此可以说预算无超跑、无静默科学重试，不能说整个实现过程没有历史错误。

## 7. 复核、交付与停止

关键实际源码与原评分依赖已按字节归档，47条来源记录、45个唯一小文件；原评分器未改写。`code.diff`记录本轮新独立pilot文件，`SOURCE_HASHES.json`给出实际来源/hash。官方模型开始/load时完成身份认证，末次仅核对大小/mtime未变，避免无意义重复读取多GB权重。

| 资产 | SHA256 |
| --- | --- |
| 官方Bench生成器文件 | `00fc5ddb5bc5690cd53fe4f092ee3320e23437a287464fa5639120c2bb82a505` |
| 实际EMA张量 | `2cc63f56d79b88418840fc44ad4c954d1961ad898a7b81d198a35bf49f0deb25` |
| 官方D0文件 | `67765c728c503b61c21624c05997ea026c9de486d1529052fafa134be3336b14` |
| 最终generation v3 | `63fbd74ca8f91b910f2040a021c19ee6a61c05a7f5b1b6baffb4bcf467d31b3c` |
| 实际分析源码 | `f6a88d7c07d4303f7e4fa0c462285bf654821cf336634ff1204e98f3190d6a6e` |
| 调用审计 | `fa649a1299f8a839fe8b6b436cc1d360dcd2824305493bfd06a5398c27ee1a56` |

主要入口：

- [完整原177字段逐输出](analysis/final/full177_per_output.csv)、[逐父](analysis/final/full177_per_parent.csv)、[逐请求配对](analysis/final/paired_per_output.csv)、[阶段面积与删面](analysis/final/stage_retention_and_costs.csv)。
- [图审逐输出](analysis_numeric/final/visual/actual_review_per_output.csv)、[严格转移及门槛](analysis_numeric/final/report_helper/protocol_verdicts.json)、[R标签漂移](analysis_numeric/final/report_helper/R_label_drift_per_axis.csv)。
- [全部10000步引导记录](analysis_numeric/final/guidance/per_step.csv)、[逐输出引导](analysis_numeric/final/guidance/per_output.csv)、[调用账本](analysis_numeric/final/calls/call_audit.json)。
- [RAW/BASE/FINAL完整本地图集与固定位置截面](gallery/final/index_all.html)。RAW的51状态、未覆盖shadow、D0提议、PRE_GEO、索引网格与原评分收据仍在本地原目录。
- [恢复入口](RUNBOOK.md)、[实际源码hash](SOURCE_HASHES.json)、[代码差异](code.diff)、[工程修复目录](engineering_incidents/)。冻结PROTOCOL.md保留设计时的DESIGN_ONLY文字作为历史快照，实际执行状态以本报告/CLOSEOUT.json为准。

审阅ZIP包含完整数值/图审表、实际评分及pilot小源码、hash、账本、协议，以及4个明确选定请求的RAW/FINAL对照图。选图只是便于打开，不代替476条全数图审；全图和大网格/轨迹保留本地。ZIP大小、校验和与文件清单由PACKAGE_RECEIPT.json记录。

**决策：保留现有EXPOSURE及Chair路线的原证据地位；本版motif guidance作为已完成负结果归档，不切换默认。** 这次未证明一个能统一Chair/Bench的更优生成框架，不能以“先验被冻结”代替实际泛化证据。未来若重启此家族，需要实质改变对应或有效引导机制并另立预算；本轮不自动扩展。没有清理旧资产、commit或push。
'''
    for token, value in [('J_TABLE',jtable),('AXES_TABLE',atable),('PAIRED_TABLE',paired),('GEOMETRY_TABLE',gtable),
                         ('GATE_TABLE',gatetable),('RETENTION_TABLE',rtable),('SHADOW_TABLE',stable)]:
        report=report.replace('<!-- '+token+' -->',value)
    assert '<!--' not in report
    put('REPORT.md',report)
    runbook=OUT/'RUNBOOK.md'
    before=sha(runbook)
    with runbook.open('a',encoding='utf-8',newline='\n') as f:
        f.write('''
## 执行收口与只读恢复（2026-10-29）

本轮已COMPLETE，方法结论为NO_UPGRADE。200 RAW、184 D0/CPU、706原177字段行、476实际图审均封存；不存在等待继续的科学任务。请先读REPORT.md和CLOSEOUT.json。

仅检查完成状态：

```bash
cd /mnt/e/MeshFlow
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_full_prior_motif_guidance_v1.run status
```

阅读结果不需要加载权重：analysis/final与analysis_numeric/final含全部汇总；gallery/final/index_all.html含同位置截面。archive/sources保留实际小源码，SOURCE_HASHES.json及PACKAGE_RECEIPT.json可独立校验。

不要重新执行prepare、sample、D0或CPU任务来“恢复浏览”；不要以删除完成收据的方式强迫重跑。历史科学输入与状态保持只读。若发现某个文件损坏，先按已登记hash定位并报告，不自动生成替代噪声/输出。恢复脚本的存在不授权额外科学计算。

工程错误和安全CPU交接见engineering_incidents与workflow/post_four_incremental；4条人工记录cell_id修正见visual_review/root_FINAL_metadata_repair_receipt.json。不得覆盖原图审或校准标签。
''')
    put('archive/runbook_closeout_change.json',dict(before_sha256=before,after_sha256=sha(runbook),change='Append completed-state read-only recovery instructions'))
    keyfiles=['REPORT.md','RUNBOOK.md','PROTOCOL.md','registration.json','lock.json','SOURCE_HASHES.json','code.diff',
              'analysis/final/full177_per_output.csv','analysis_numeric/final/visual/actual_review_per_output.csv',
              'analysis_numeric/final/calls/call_audit.json','analysis_numeric/final/report_helper/protocol_verdicts.json',
              'workflow/post_four_incremental/final_global_full_RAW_verification.json','archive/final_small_source_verification.json',
              'archive/final_resources.json','workflow/close_report.py']
    close=dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),scientific_result='NO_UPGRADE',
               actual=calls['actual'],new_RAW=200,reused_R_outputs=46,total_original177_stage_rows=706,
               actual_review_unique_cells=476,pending_review=0,source_checks=len(source_checks),
               final_RAW_integrity='PASS',call_audit='PASS',scientific_replay=0,historical_engineering_errors_disclosed=True,
               scientific_workers_active=0,default_changed=False,commit=False,push=False,cleanup=False,
               files={p:sha(OUT/p) for p in keyfiles},package_status='PACKAGING_NEXT_NO_SCIENTIFIC_WORK_REMAINS')
    put('CLOSEOUT.json',close)
    print(json.dumps({k:v for k,v in close.items() if k!='files'},ensure_ascii=False))

if __name__=='__main__':
    main()
