"""CPU-only delivery from completed, frozen quick-pilot results; no model imports."""
from __future__ import annotations
import csv
import difflib
import json
import math
from pathlib import Path
import shutil
import statistics
import time
from .quick_common import ROOT, OUT, CODE, read, write, sha, event


ARMS = ('OT_BASE', 'OT_HYBRID')


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def fmt(value, digits=6):
    if value is None or value == 'NA':
        return 'NA'
    if isinstance(value, str):
        return value
    if not math.isfinite(value):
        return str(value)
    return f'{value:.{digits}g}'


def table(headers, values):
    return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|'] +
                     ['|'+'|'.join(str(x) for x in row)+'|' for row in values])


def source_delivery(out):
    generation = read(out/'registration/generation_sources_before.json')
    original_scoring = read(out/'scoring/definitions.json')['source_identity']
    identities = {'new': {}, 'original': {}}
    for source in sorted(CODE.glob('*.py')):
        target = out/'source_code/new'/source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        identities['new'][str(source.relative_to(ROOT))] = sha(source)
        assert sha(target) == sha(source)
    for name, expected in {**generation, **original_scoring}.items():
        source = ROOT/name
        assert sha(source) == expected, name
        target = out/'source_code/original'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        identities['original'][name] = expected
        assert sha(target) == expected
    identities['scope'] = 'New isolated source plus only actual generation/scoring dependencies; no checkpoint/raw copies.'
    identities['scientific_code_frozen_at_dispatch'] = read(out/'registration/new_code_before_generation.json')
    write(out/'source_code/source_hashes.json', identities)
    patches = [out/'registration/attempt1/registration_fix.diff',
               out/'engineering_revisions/resume_guard/code.diff']
    (out/'code.diff').write_text('\n'.join(p.read_text(encoding='utf-8') for p in patches), encoding='utf-8')
    return identities


def run(out=OUT):
    out = Path(out)
    started = time.perf_counter()
    accounting = read(out/'sampling_accounting.json')
    summary = read(out/'scoring/summary.json')
    numbers = read(out/'report_ready_numbers.json')
    source_audit = read(out/'registration/final_scoped_source_audit.json')
    score_audit = read(out/'scoring/final_cpu_audit.json')
    gallery = read(out/'gallery/summary.json')
    qa = read(out/'gallery/qa.json')
    tests = read(out/'scoring/tests/scoring_cpu_tests.json')
    resume_guard = read(out/'engineering_revisions/resume_guard/validation.json')
    assert accounting['total_forward_attempts'] == 2500
    assert accounting['cached'] == 32 and accounting['newly_generated'] == 48
    assert not accounting['failed_calls'] and not accounting['pending']
    assert all(x['status'] == 'PASS' for x in (source_audit, score_audit, qa, tests, resume_guard))
    assert summary['status'] == gallery['status'] == numbers['status'] == 'COMPLETE'
    assert not any(accounting[k] for k in ('training_updates','optimizer_updates','backward_attempts','failed_outputs','unavailable'))
    for name, expected in numbers['source_csv_sha256'].items():
        assert sha(out/name) == expected, name
    primary = {x['metric']: x for x in summary['primary']}
    outputs = rows(out/'per_output_metrics.csv')
    conditions = rows(out/'per_condition_summary.csv')
    main = [r for r in conditions if r['panel'] == 'main']
    paired = rows(out/'paired_parent_summary.csv')
    registered_conditions = read(out/'registration/conditions.json')
    assert len(outputs) == 80 and len(main) == 16
    parents = sorted({r['uid'] for r in main})
    report = []
    add = report.append
    add('**CHAIR_CONTROL_USABILITY_QUICK_PILOT_V1 · 实际完成报告**\n\n2026-10-05；冻结模型、小面板、探索性能力快测。')
    add('实际账本：训练更新 **0**，反向传播 **0**，optimizer 更新 **0**。主面板 64 项；嵌套面板 24 项与主面板重合 8 项，共 **80 个唯一输出**：旧 raw 核验复用 **32**，新生成 **48**，失败 **0**，输出缺失/NA **0**。另完成两条身份回放，两臂均与旧 raw 逐字节一致、最大绝对差为 0。完整 Euler50 尝试/返回 **50/50**；采样 NFE、顶层模型 forward、batch 调用均为 **2500**，额外单步预检 **0**；硬上限分别为 82 次完整尝试、4100 采样 NFE、4112 顶层 forward，剩余预算未使用。')
    add('新 CPU 几何评分 **80**；原评分 **32 份身份核验复用 + 48 份按原定义计算**，512 点指标逐输出核对通过。CF/FF 的原数值扫描 80 份完成；新 Gamma 76 份有效、4 份 NOT_APPLICABLE，原因见嵌套条件表。16 个主条件模型组、96 个无序候选对全部保留。人工评分 **未执行**：`human_usable=NA`、`U_single=NA`、`U_at4=NA`、`user_study=NOT_RUN`。104 张五视图图像已生成；主面板 64/64、嵌套 24/24 均可查看。')
    checkpoints = read(out/'registration/checkpoints.json')
    add(table(['模型','来源文件（chair_hybrid_coupling_v1/checkpoints/）','训练身份','文件 SHA256'],
        [[r['metadata']['arm'],Path(r['path']).name,
          f"pilot={r['metadata']['pilot_step']} / total={r['metadata']['conditional_total_step']} / λ={r['metadata']['lambda_variance']}",
          f"`{r['actual_sha256']}`"] for r in checkpoints]))
    add('本轮 HYBRID 是后续 MOMENT 记录中的 START/total5000；原 HYBRID 实验自己的共同起点 START/total4000 未用于比较。两端点严格加载生成器，未构造或恢复 AdamW；参数在采样前后哈希一致。沿用实际 Geo `P=I`、原 Euler50、BF16 forward 和 FP32 状态/步进、math SDPA、严格确定性、TF32 关闭。新输出保存全部 51 状态，C 逐位不变。模型输入仅包含冻结 Gaussian、C、mask 和 N，没有 GT/目标尺度/GT OT。')
    add('**读表方式。** 主统计先在每个条件内汇总固定四候选，再对 8 父等权；下表差值均为 HYBRID−BASE。改善父数遵循箭头方向，不把 32 候选当 32 个独立父。D=[mean d(Q_Y,F)²+mean d(Q_F,Y)²]/(2L0²)，两侧各2048个面积采样点到对方有限闭三角面的距离；Q_Y按条件固定，两个方向分别保存在逐输出表。接口是逐输出 RMS/L0 或 p95/L0 的均值；比例均以 0–1 表示。没有从平均 MSE 开方伪造平均 RMS，也没有设置可用阈值或显著性通过线。')
    selected = [
        ('D_sym2048_mean','四候选平均 D ↓'),
        ('candidate0_D_sym2048','固定 candidate0 的 D ↓'),
        ('GT_oracle_Best_at2_D_sym2048','GT-oracle Best@2 D ↓'),
        ('GT_oracle_Best_at4_D_sym2048','GT-oracle Best@4 D ↓'),
        *[(f'mean_Fscore2048_{t}',f'F-score @{float(t)*100:g}% L0 ↑') for t in ('.005','.01','.02','.05')],
        ('mean_Gamma32_edge_RMS_L0','Gamma→free 边 RMS/L0 ↓'),
        ('mean_Gamma32_edge_p95_L0','Gamma→free 边 p95/L0 ↓'),
        ('mean_Gamma32_surface_RMS_L0','Gamma→free 面 RMS/L0 ↓'),
        ('mean_Gamma32_surface_p95_L0','Gamma→free 面 p95/L0 ↓'),
        ('mean_Gamma32_edge_AttachCov_0.01','AttachCov 边 @1% ↑'),
        ('normalized_free_pair_distance','normalized_free_pair_distance（描述）'),
        ('mean_CF_proper_affected_free_fraction','CF proper 受影响 free 比例 ↓'),
        ('mean_FF_illegal_affected_free_fraction','FF illegal 受影响 free 比例 ↓'),
        ('mean_CF_proper_pair_count','CF proper 对数 ↓'),
        ('mean_FF_illegal_pair_count','FF illegal 对数 ↓'),
        ('mean_free_minimum_angle_lt5_fraction','free 最小角 <5° 比例（结合GT）'),
        ('mean_GT_free_minimum_angle_lt5_fraction','同条件 GT free 最小角 <5° 比例'),
        ('mean_free_near_degenerate_fraction','free 近退化比例 ↓'),
        ('mean_free_exact_zero_area_fraction','free 零面积比例 ↓'),
        ('mean_generation_seconds','单候选生成秒（历史/本轮混合）'),
        ('four_candidate_generation_seconds','四候选总生成秒（同上）'),
    ]
    main_rows = []
    for key,label in selected:
        # CSV keys use decimal strings with a leading zero.
        key = key.replace('_.' ,'_0.')
        r = primary[key]
        descriptive = key in ('normalized_free_pair_distance','mean_free_minimum_angle_lt5_fraction','mean_GT_free_minimum_angle_lt5_fraction') or 'seconds' in key
        main_rows.append([label, fmt(r['OT_BASE_parent_equal_mean']),fmt(r['OT_HYBRID_parent_equal_mean']),
                          fmt(r['paired_mean_delta']),fmt(r['paired_median_delta']),
                          '—' if descriptive else str(r['improved_parents'])+'/8',
                          '—' if descriptive else r['worst_parent'][:8]])
    main_rows += [['human U_single / U_at4','NA / NA','NA / NA','NA','NA','NA','NA']]
    add(table(['指标','OT_BASE','HYBRID','配对均差','配对中位差','改善父数','最差配对父前缀'],main_rows))
    add('全指标的配对方向、均差、中位差、最差父及其差值见 [primary_summary.csv](scoring/primary_summary.csv)；逐父完整数值见 [paired_parent_summary.csv](paired_parent_summary.csv)。两臂所有输出的 CF/FF coplanar 对数均为 0；表内 FF illegal 与 proper 在本批结果相同。这里是原评分器数值扫描的计数，不是严格拓扑认证；未把相交、小角或距离合并成“可用分”。')
    add('**完整容差曲线。** 面质量阈值为线性距离 τL0，以下 P/R/F 分别先逐输出计算，再按父等权。接口 AttachCov 对真实 known/free Gamma 弧长加权；边段版与表面版分别列出，均不包括 C 到自身的距离。')
    values = []
    for tau in (.005,.01,.02,.05):
        values.append([f'{100*tau:g}%']+[fmt(primary[f'mean_{m}2048_{tau}'][arm+'_parent_equal_mean'])
                            for arm in ARMS for m in ('Precision','Recall','Fscore')])
    add(table(['τ/L0','BASE P','BASE R','BASE F','HYBRID P','HYBRID R','HYBRID F'],values))
    values = []
    for tau in (.001,.003,.005,.01,.02,.05):
        values.append([f'{100*tau:g}%']+[fmt(primary[f'mean_Gamma32_{kind}_AttachCov_{tau}'][arm+'_parent_equal_mean'])
                            for kind in ('edge','surface') for arm in ARMS])
    add(table(['τ/L0','BASE 边','HYBRID 边','BASE 面','HYBRID 面'],values))
    add('HYBRID 平均 D 下降 **11.70%**，但仅 **4/8 父**改善；接口边 RMS 下降 **15.86%**、7/8 父改善。F-score 在 0.5%、1%、2% 三个细容差下较低，只有 5% 下略高；接口边 AttachCov 在 0.1%–1% 下也没有平均收益。因此更接近参考的整体几何和较小接口距离，没有同时变成更密的细尺度贴合。表面版 @1% 的提高也不能解释为边界已经连接，更不等于共享顶点或水密。')
    add('**八个父对象全部保留。** 下列每个 D 都是固定四候选均值，范围是该父同臂四候选最小–最大值；完整 UID 与每候选值保存在逐输出表，没有用表现选父或重抽坏样本。')
    values = []
    for uid in parents:
        grouped = {arm: next(r for r in main if r['uid']==uid and r['arm']==arm) for arm in ARMS}
        delta = float(grouped['OT_HYBRID']['D_sym2048_mean'])-float(grouped['OT_BASE']['D_sym2048_mean'])
        ranges = []
        for arm in ARMS:
            ds = [float(r['D_sym2048']) for r in outputs if r['uid']==uid and r['arm']==arm and r['context_level']=='1']
            ranges.append(fmt(min(ds))+'–'+fmt(max(ds)))
        values.append([uid[:8],fmt(float(grouped['OT_BASE']['D_sym2048_mean'])),fmt(float(grouped['OT_HYBRID']['D_sym2048_mean'])),fmt(delta),*ranges])
    add(table(['父 UID 前缀','BASE 均 D','HYBRID 均 D','H−B','BASE 候选范围','HYBRID 候选范围'],values))
    add('最差平均 D 配对父为 `f59627a077dcc2566c63b8b48495261a`，差值 +0.00101762；细容差 F-score@1% 最差配对父为 `86ec9c796a6c1789e792598ed58dd1cc`，差值 −0.149725。两者不是同一最坏父，报告没有用单个案例代表全部分布。')
    add('**同一 D-oracle 胜出候选的其他指标。** Best@2 仅从 0/1 选，Best@4 仅从 0/1/2/3 选；下表不为每一列另选候选。全部曲线和候选编号见 [per_condition_summary.csv](per_condition_summary.csv)。')
    values=[]
    for arm in ARMS:
        for choice in ('Best2','Best4'):
            r=numbers['same_winner_parent_equal_means'][arm][choice]
            values.append([arm, 'GT-oracle '+choice.replace('Best','Best@'),
                           *[fmt(r[k]) for k in ('D_sym2048','Fscore2048_0.01','Gamma32_edge_RMS_L0',
                                                'Gamma32_edge_AttachCov_0.01','CF_proper_affected_free_fraction',
                                                'FF_illegal_affected_free_fraction','free_minimum_angle_lt5_fraction')]])
    add(table(['模型','固定预算选择','D','F@1%','边 RMS/L0','边 Attach@1%','CF 受影响','FF 受影响','小角比例'],values))
    add('HYBRID Best@4 D 比自身四候选均值低 **44.93%**，比 Best@2 低 **13.65%**；相对 BASE Best@4 仅低 **4.94%**。候选潜力存在，但平均候选差距也大。HYBRID Best@4 的 F@1%/接口优于自身平均候选，CF/FF 受影响比例却略高；与 BASE Best@4 比较，它的 F@1% 仍较低。不能把 Best 解释成自动部署选择，也不能在没有人评阈值的情况下断言“多数候选不可用”或“多数已可用”。')
    add('normalized_free_pair_distance 从 0.00142841 降到 0.00124691，约下降 **12.71%**，并非多样性越大越好。它和 D、F-score 共同显示几何范围收窄且局部质量取舍仍在；本轮没有证据证明变化都是有意义的语义多样性。两臂小角比例分别为 0.09090/0.12364，同一 GT free 为 0.17829，单独小角计数不能判可用。所有四候选及全部六对均参与，没有剔除坏结果后计算“优质多样性”。')
    add('**增加一致已知片区。** 取字典序前两父，按冻结的源面身份/共享整边 BFS 构造 C1⊂C2⊂C3。选种子依照到 C1 质心集合的最小距离由远到近、源面 ID 打破平局，未看生成质量。顶点/尺度/N 不变；每层两臂共用实际 Gaussian，跨层因 free 数量变化不声称 Gaussian 逐位相同。C2/C3 的已知连通片确有多个，以下可以称多片条件。')
    values=[]
    for c in registered_conditions:
        if c['uid'] not in parents[:2]:
            continue
        span=math.sqrt(sum(float(x)**2 for x in c['known_span']))/c['L0']
        values.append([c['uid'][:8],f"C{c['context_level']}",f"{c['K']}/{c['N']}",
                       fmt(c['known_area_ratio']),c['known_components'],fmt(span),c['free_budget'],
                       fmt(c['common_area']),c['gamma_status']])
    add(table(['父','条件','known 面数','known 面积比','片数','C bbox 对角/L0','free 面数','公共区域面积（坐标²）','Gamma'],values))
    add('第二父 C3 的源 known/free 接口存在非流形/方向歧义，未使用局部可信子集冒充完整 Gamma；因此该条件 2 噪声×2 模型共 4 项接口为 NA，生成与其余几何评分正常。面数约 20% 不代表同样的信息量，例如第二父 C1 的面积占比只有约 1.18%。')
    add('公平公共区域为 GT 中始终不属于 C3 的面，固定同一 2048 点，仅计算它们到当前预测 free 的单向距离，复制的 C 不进入距离目标。下表先两噪声均值，再两父等权；每层自身完整 free 质量另存在逐条件表，因目标补集改变不拿它当纯控制增益。')
    values=[]
    for r in summary['multi_context']:
        values.append([r['arm'],f"C{r['context_level']}",fmt(r['common_RMS_L0']),fmt(r['common_p95_L0']),
                       *[fmt(r[f'common_recall_{t}']) for t in (.005,.01,.02,.05)]])
    add(table(['模型','条件','common RMS/L0','common p95/L0','R@0.5%','R@1%','R@2%','R@5%'],values))
    add('BASE 公共 RMS 从 0.05733→0.04804→0.03675；HYBRID 从 0.04084→0.04822→0.03638。HYBRID C2 的均值较 C1 退化，而 C3 较 C1 改善约 **10.91%**，两臂 C3 的细尺度公共覆盖均明显增加。这支持“增加片区在这个小面板上可能帮助未知区域”，不支持逐噪声单调或普遍任意多条件融合；跨层 free 减少也不能直接解释成后验不确定性降低。')
    add('**实际看图后的 AI_PROXY 初筛。** 以下只针对已查看图像的定性观察，不是全量人工盲评，不填写 A/B/C、usable 或任何可用比例；图片身份映射事先可见，故也不是 AI 盲评。')
    visual = [
        ('G01-R03','33ec… C1 / HYBRID / c0','轮廓可辨认为带扶手椅；橙色座面/支架与青色已知底座出现错层、穿出轮廓和叠层，原GT底座上方布局不同。形状可识别与局部相容不是同一结论。'),
        ('G01-R07','33ec… C1 / BASE / c0','椅形仍可识别；橙色座面架在青色底座之上、腿/支架错位，局部编辑可能牵动多面。两个模型都存在这种问题。'),
        ('G05-R03','86ec… C1 / BASE / c0','椅背顶部可见三角缺口/细面，整体仍接近椅形；相比同GT存在可见的局部残缺。'),
        ('G05-R04','86ec… C1 / HYBRID / c0','仍可见背部缺口、细长面和局部贴合问题；这一单候选不能代表该父四候选平均 F 的优劣。'),
        ('G08-R02','f596… C1 / HYBRID / c3','椅形可辨认，但腿长、座面上下布局和横撑与GT明显不同，并带毛刺状背顶/细面；几何偏离GT不全部等于构思用途失败。'),
        ('G12-R02','341b… C3 / HYBRID / c1','青色原条件本身就是方盒状扶手椅的部分粗三角面；橙色补全面在座面前沿和侧部出现锯齿/叠层。原GT也有粗大不规则三角面，不能把全部异常归因生成。'),
    ]
    add(table(['已查看图像','身份','可见依据与用途关系'],
              [[f'[{i}](gallery/images/{i}.png)',name,note] for i,name,note in visual]))
    add('已同时查看 G01、G05、G08、G12 的 GT 参照。可能直接妨碍“构思后继续局部编辑”的是 known/free 严重错位、多层交叠和缺失造成需要大幅重建，而非仅与唯一 GT 的款式不同。静态视图不能验证焊接关系、编辑工时或审阅者愿不愿保留；数值相交与图片只提供进一步评阅的依据。')
    write(out/'ai_proxy_visual_review.json',dict(status='QUALITATIVE_ONLY',reviewer='Codex AI_PROXY',
        actually_viewed=[dict(image=i,identity=name,observation=note,sha256=sha(out/f'gallery/images/{i}.png')) for i,name,note in visual],
        GT_references=[dict(image=i,sha256=sha(out/f'gallery/images/{i}.png')) for i in ('G01_GT','G05_GT','G08_GT','G12_GT')],
        human_usable=None,U_single=None,U_at4=None,user_study='NOT_RUN',
        limitation='Six selected examples, unblinded qualitative viewing only; no population labels or usability rate.'))
    add('**人评材料已就绪。** [技术图库](gallery/technical/index.html)含 C/GT，[盲评图库](gallery/blind/index.html)只含 C/候选和用途说明；每图四固定方位加线框，同父所有模型/层级统一画幅，104 张均解码通过，零裁剪/渲染错误。无需扩展原 GT 画幅的父为 8/8。盲评目录自包含；[映射表](blind_mapping.csv)单独留给分析者，不发给评阅者。[主面板空表](human_ratings_template.csv)64 行、[嵌套空表](human_ratings_nested_template.csv)24 行均未填分。')
    add('固定用途：“用于形状构思和后续选择/编辑的低多边形候选，不是直接用于制造或物理仿真的合格网格。”真实评阅者分别给几何相容、完整合理椅形、明显穿插/脱离/碎片/缺失对用途的妨碍程度打 0–3 分（高分表示更适合当前用途）；只有 C 保持且三项均≥2 才 usable。收到真实评分后才能计算全部候选标签均值 U_single 和各条件四候选至少一项 usable 的 U_at4，保留评阅者间分歧。GT-oracle、模型偏好和 U_at4 仍是三个不同问题。')
    add('**计时与资源。** 单 GPU worker 持有原 GPU 锁；两模型依次加载。RTX 5080，总显存 16303 MiB；派发前空闲约 13074 MiB、利用率 5%；WSL 可用 24 CPU、约 61 GiB RAM。遵守本轮单 worker 要求，同时让独立 CPU 评分/渲染以每进程 2 线程重叠工作。未终止其他用户进程。每臂首条已授权回放后记录资源，再继续正式输出。')
    time_rows=[]
    for arm in ARMS:
        for status,label in [('CACHED','历史正式调用'),('SAMPLED','本轮正式调用')]:
            group=[r for r in outputs if r['arm']==arm and r['generation_status']==status]
            durations=[float(r['generation_seconds']) for r in group]
            time_rows.append([arm,label,len(group),fmt(statistics.mean(durations)),fmt(sum(durations)),fmt(min(durations)),fmt(max(durations))])
    add(table(['模型','时间来源（主+嵌套唯一输出）','n','平均秒','合计秒','最小秒','最大秒'],time_rows))
    score_seconds=sum(float(r['scoring_seconds']) for r in outputs)
    add(f'本轮采样阶段从 08:02:18 到 08:05:24 UTC，阶段墙钟约 **186 秒**（含读取/加载/哈希/存盘）；50 个 rollout 包装计时合计 **{accounting["wall_seconds_by_kind"]["sampling"]:.2f} 秒**，其中 forward 内层计时 {accounting["wall_seconds_by_kind"]["sampling_forward"]:.2f} 秒，两者嵌套不能相加。逐输出保存的 80 份 CPU 评分耗时合计 **{score_seconds:.2f} 秒**（已含4项预登记加密复核，不含末尾向量存盘）；六对距离汇总、缓存读取和重叠等待另有开销，不能把这些计时相加当作端到端基准；图库阶段 **{gallery["seconds"]:.2f} 秒**，限定源审计 **{source_audit["seconds"]:.2f} 秒**。详细 CPU 账本见 `scoring/cpu_ledger.jsonl`，实际采样账本见 `ledger/calls.jsonl`。')
    add('两臂本轮正式输出的单 rollout 最大 PyTorch allocated 均为 588918272 bytes，reserved 均为 645922816 bytes；在每次 rollout 前重置峰值，范围包括驻留模型与 autocast 缓存。allocated 属于 reserved，未相加；它们不等于完整进程 VRAM，WDDM 下进程用量返回 N/A，另保留 nvidia-smi 全卡快照。新增24/臂的计时包含不同 N/上下文，主表四候选时间只汇总主面板。历史0/1和本轮2/3时间来源逐行注明；旧时段未重新隔离测量其资源竞争，软件路径由哈希和逐位回放确认，运行阶段/CPU竞争不同，不从微小时间差声称加速。缓存读取时间未用于生成吞吐。')
    add('**检查与可恢复性。** 本轮新评分 8 项轻量解析检查通过：同 GT free、固定法向平移、L0 尺度、排除复制 C、有限边段、Gamma 弧长权重/NA、固定 Best 及失败/缺失分母、固定随机查询。相同几何的双向距离为零，也覆盖相同候选多样性原语。前两父 candidate0×两模型的 4 项预登记敏感性复核完成，2048→8192 点 D 相对变化 −1.03% 至 +3.28%，两父内模型排序未变；Gamma 32→128 点边 RMS 最大绝对变化约 1.08e−5 L0。仍使用 2048/32 主结果，未改阈值或按加密结果选模型。')
    add(f'独立终核从已保存距离向量核对全部 80 份 D/P/R/F 和旧512指标、固定 winner/父均值、96 候选对，均 PASS；40 个模型配对 bank 的 raw 保存 Gaussian/C/mask 逐字节一致。仅核验本轮读取的原依赖 **{source_audit["old_dependency_files"]} 文件 / {source_audit["old_dependency_bytes"]} bytes**，另含必要新结果后去重核验 {source_audit["unique_hashed_files"]} 文件，缺失/差异均为 0；没有全仓审计。原模型/评分/旧 raw 不变，Git HEAD 保持 `{source_audit["head"]}`，未 commit/push。')
    add('生成前的 CPU 登记发现候选2/3被误标为嵌套、非嵌套父带空 common 数组，已在独立目录修正；首次修正脚本因完成标记文件名写错而中止，原件、diff 与说明保留在 `registration/attempt1/` / `revision_pre_sampling.json`，修正时模型调用为 0。生成结束后只强化恢复时的 requests/来源/既有结果哈希检查，未改采样数学；其执行验证通过，记录在 `engineering_revisions/resume_guard/`。GPU 执行期间对应冻结身份检查也通过，未发生静默重试。')
    add('恢复总入口为 `python -B -m experiments.mf_h1_local0.chair_control_usability_quick_v1.run_all --stage all`，分阶段可选 register/freeze/sample/score/gallery/report。已完成结果按身份复用；未决或失败模型调用会阻止自动重试，余量不能用于扩展实验。当前任务已完成并停止。只重建此报告可运行 `--stage report`，零模型调用。源码快照及哈希见 [source_code/source_hashes.json](source_code/source_hashes.json)，小范围工程修订见 [code.diff](code.diff)，原评分定义保留在 [scoring/definitions.json](scoring/definitions.json)。')
    add('**对七个决策问题的回答。**')
    add('1. HYBRID 同预算收益同时出现在平均 D、候选最低 D 和接口 RMS；收益大小和父分布不同，细容差 F/边贴合并未全面变好。\n2. Best@4 与平均候选差距明显，确实会遮蔽候选不稳定性；没有人评依据宣称多数候选低质量。四候选提高“存在较近参考结果”的机会，尚未测得真实用途成功率。\n3. 多样性更小且质量指标混合，不能把几何差异都当语义创意；高相交比例与可见错层提示质量代价仍在。\n4. C3 公共未观测区域总体改善，HYBRID C2 退化；这是2父×2噪声的描述性证据。\n5. 主要用途风险是 known/free 错层、多层叠加和局部缺面导致难以局部编辑；具体查看例与原GT对照列在 AI_PROXY 表。\n6. 人工可用性没有测得，本轮完成了全量待评材料，任何人类可用率均为 NA。\n7. 更支持先补真实用途评阅这一块证据，再决定训练方向；当前不能据此宣称方法全面优越或已经达到论文通过线，也没有理由因不满足精确连接就否定全部构思价值。')
    add('**唯一优先下一步：对现有64候选开展真实盲评，不新增训练或采样。** 检验假说：四候选中较好的几何潜力会转化为高于单候选的 U_at4，但 known/free 错位和叠层会限制该收益；同时检验 HYBRID 相对 BASE 的距离收益是否保留为用途收益。若评阅显示 U_at4 仍无实际收益且低分集中于几何相容/编辑障碍，再把局部接口相容与候选稳定性作为下一轮模型改进目标，而不是只优化平均 GT 距离。这个下一步未自动启动。')
    add('边界：这8父是已用过的开发对象，新噪声不是新形状，新指标是探索性适配，不是独立确认；没有无条件/其他方法对照，不证明优于全部3D控制方法或 Native 的全部创新。旧 MOMENT 结论和现行默认模型未改变。')
    add('必要数据：[protocol.json](protocol.json)、[requests.csv](requests.csv)、[逐输出表](per_output_metrics.csv)、[逐条件表](per_condition_summary.csv)、[逐父配对表](paired_parent_summary.csv)、[多片条件表](multi_context_summary.csv)、[采样账本](sampling_accounting.json)、[完整图库](gallery/index.html)。`requests.csv` 保留生成前 CACHED/READY 登记状态；最终状态在逐输出表与 `outputs/*/result.json`，没有把旧登记表悄悄改成结果表。保存的查询点/逐点距离在 `scoring/vectors/`，新 raw 及51状态在 `outputs/`，旧缓存继续引用原只读文件；无需复制大权重或制作大型交接包。')
    report_path=out/'QUICK_CONTROL_USABILITY_REPORT.md'
    report_path.write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    source_delivery(out)
    required=['QUICK_CONTROL_USABILITY_REPORT.md','protocol.json','requests.csv','per_output_metrics.csv',
              'per_condition_summary.csv','paired_parent_summary.csv','multi_context_summary.csv',
              'sampling_accounting.json','human_ratings_template.csv','human_ratings_nested_template.csv',
              'gallery/technical/index.html','gallery/blind/index.html','code.diff','source_code/source_hashes.json',
              'ai_proxy_visual_review.json','scoring/final_cpu_audit.json','registration/final_scoped_source_audit.json']
    hashes={p:sha(out/p) for p in required}
    result=dict(status='COMPLETE',model_calls=0,backward=0,optimizer_updates=0,
                report_path=str(report_path),delivery_sha256=hashes,seconds=time.perf_counter()-started,
                source_audit='PASS',scores='80/80',main_gallery='64/64',nested_gallery='24/24',
                user_study='NOT_RUN',human_usable=None,task_stopped=True)
    write(out/'delivery_manifest.json',result)
    event(out/'ledger/reporting.jsonl',dict(event='REPORT_COMPLETE',report_sha256=sha(report_path),model_calls=0))
    return result


if __name__ == '__main__':
    print(json.dumps(run(),ensure_ascii=False))
