"""Assemble the frozen, zero-training recorder report from completed artifacts."""
from pathlib import Path
import csv,json
from .recorder import OUT,OLD,ASSETS,ROOT
from experiments.mf_h1_local0.jedge5_diagnosis.common import read_json,write_json,file_sha256

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    geom=read_json(OUT/'geometry/endpoint_results.json');dyn=read_json(OUT/'dynamics/results.json');probes=read_json(OUT/'probes/combined_results.json')
    edge=read_json(OUT/'edge/trajectory_summary.json');teacher=read_json(OUT/'edge/teacher_summary.json')
    assert all(x['status']=='PASS' for x in (geom,dyn,probes,edge,teacher))
    end={(r['model'],r['patch'],r['seed'],r['nfe']):r for r in geom['rows']}
    rows=[]
    for label in ('B','J'):
        oldscore=read_json(OLD/'scores'/('A_'+label)/'results.json')['rows']
        for seed in (10401,10402):
            r50=end[label,4,seed,50];r100=end[label,4,seed,100]
            f32=next(r for r in oldscore if r['parent']==0 and r['patch']==4 and r['seed']==seed and r['precision']=='fp32' and r['nfe']==100)
            rows.append([label,seed,f"{r50['symmetric_RMS_bbox_pct']:.6f}",f"{r100['symmetric_RMS_bbox_pct']:.6f}",f"{f32['scalars']['remainder_symmetric_RMS_bbox_pct']:.6f}"])
    endpoint_table=table(['模型','P0 patch4 / seed','BF16 / 50','BF16 / 100','FP32 / 100（旧档）'],rows)
    dtable=table(['J / patch4 seed','D(.2)','D(.5)','D(.8)','D(1)'],[[r['seed']]+[f"{r['at_t'][k]:.3f}%" for k in ('0.2','0.5','0.8','1.0')] for r in dyn['summaries'] if r['model']=='J' and r['patch']==4])
    gt=table(['J / patch4 seed','NFE','free→R12 RMS','R12→free RMS','自由面积 / GT自由面积'],[[s,n,f"{end['J',4,s,n]['forward_RMS_bbox_pct']:.3f}%",f"{end['J',4,s,n]['backward_RMS_bbox_pct']:.3f}%",f"{100*end['J',4,s,n]['free_area_over_GT_free_area']:.2f}%"] for s in (10401,10402) for n in (50,100)])
    pt=table(['patch / seed','B / BF16','J / BF16','B / FP32','J / FP32'],[[f"{r['patch']} / {r['seed']}"]+[f"{r[p][m]:.3f}" for p,m in [('bf16','B'),('bf16','J'),('fp32','B'),('fp32','J')]] for r in probes['registered_t05_comparison']])
    context=table(['P0条件','开边数 / 回路数','开边长度/L','边界离拟合平面RMS/L','已知面积占比','C / 整体bbox体积比'],[[r['patch'],f"{r['boundary_edge_count']} / {r['boundary_component_count']}",f"{r['boundary_length_bbox']:.4f}",f"{100*r['boundary_weighted_best_plane_rms_bbox']:.3f}%",f"{100*r['C_area_fraction']:.3f}%",f"{100*r['C_bbox_volume_over_full_bbox']:.3f}%"] for r in geom['contexts'] if r['parent']==0 and r['patch'] in (0,4)])
    with (OUT/'geometry/endpoint_faces.csv').open() as f:face=list(csv.DictReader(f))
    top=sorted([r for r in face if r['key']=='J_P0_patch4_seed10402_bf16_Euler50'],key=lambda r:-float(r['forward_mse']))[:10]
    top_ids=[int(r['free_face']) for r in top]
    workers=[read_json(OUT/'trajectories'/m/'results.json') for m in ('B','J')]
    probe_workers=[read_json(OUT/'probes'/m/'results.json') for m in ('B','J')]
    failed=read_json(OUT/'trajectories/B/failed_attempt1.json')
    timing=dict(replay_success_worker_wall_seconds={w['model']:w['seconds'] for w in workers},failed_recorder_attempt_wall_seconds=failed['seconds'],
        replay_success_sum_sampler_wall_seconds=sum(r['audit']['seconds'] for w in workers for r in w['rows']),
        replay_success_sum_cuda_event_seconds=sum(r['audit']['model_forward_cuda_event_seconds'] for w in workers for r in w['rows']),
        probe_worker_wall_seconds={w['model']:w['wall_seconds'] for w in probe_workers},
        offline_endpoint_numeric_seconds=geom['seconds'],offline_shapes_numeric_seconds=read_json(OUT/'geometry/trajectory_results.json')['seconds'],
        offline_edge_fixture_seconds=read_json(OUT/'edge/fixture_record.json')['seconds'],offline_edge_teacher_seconds=teacher['seconds'],offline_edge_trajectory_seconds=edge['seconds'],
        offline_dynamics_numeric_seconds=dyn['seconds'],note='Worker walls overlap. CUDA events can include observer copy/synchronization and concurrent contention. Offline numeric seconds exclude plotting where function records precede plots; no hardware-normalized speed benchmark.')
    findings=[dict(id='F1',finding='步长敏感成立；10402在BF16/100仍3.547%，并未普遍救回。'),dict(id='F2',finding='J patch4早于t=.5已持续分离；未观测t=.5速度脉冲。'),dict(id='F3',finding='覆盖不足和自由表面积偏小是主要可观测异常；不支持仅3张巨面飞离或面积爆炸。'),dict(id='F4',finding='方向敏感性存在且精度相关，但不证明刚性、条件数或唯一失败原因。'),dict(id='F5',finding='真实Edge为固定GT对应训练项；生成最近Γ变化仅是离线几何描述。'),dict(id='F6',finding='37.5为特例loss值比；PCA与C统计均不能推出唯一原因。')]
    sources={p:dict(path=p,sha256=file_sha256(OUT/p)) for p in ['geometry/endpoint_results.json','geometry/trajectory_results.json','geometry/directional_plot_manifest.json','dynamics/results.json','probes/combined_results.json','edge/fixture_record.json','edge/teacher_summary.json','edge/trajectory_summary.json']}
    result=dict(status='PASS_EXECUTION',final_audit='final_audit.json',experiment='jedge5_blackbox_20261002_113600',training_updates=0,model_backward_calls=0,
        actual_model_forwards=1490,full_rollouts_completed=17,full_rollouts_with_raw_recording=16,failed_observer_attempt_completed_rollouts=1,
        probe_model_forwards=240,replay_model_forwards_including_failed_observer=1250,CPU_geometry_autograd_calls=3,OT_calls=0,
        condition_update='C copied/clamped in every Euler state; raw model known-slot velocities are not identically zero.',
        gpu_lock=dict(status='NOT_USED',reason='User supplemental shared-lock instruction arrived after all recorder/probe GPU work had finished. Prior resource snapshot/incremental dispatch used; no further GPU work after instruction. Repository and WSL /tmp search found no existing GPU lock path. Do not claim lock coordination or external-window de-duplication.'),
        counts_note='17 includes one completed B/patch0/10401/50 run whose recorder assertion failed before raw save, plus the necessary repeat. No new contexts, seeds, checkpoints or training.',
        shared_window_scope='Other windows matrix overlap was not audited; this run replayed 16 selected archived cases to capture true velocities, never the old full matrix.',
        findings=findings,timing=timing,sources=sources,generated_free_face_forward_top10_J_patch4_10402_Euler50=top_ids,
        identities='case_identity_manifest.json (final audit)',artifacts='artifact_manifest.json (final audit)')
    write_json(OUT/'results.json',result)
    report=f'''# JEdge5 定点飞行记录：步长敏感与覆盖误差

本轮已完成零训练定点记录。证据支持“特定上下文/噪声下步长敏感，伴随未知区域覆盖不足”；尚未证明 ODE 刚性、数值发散、100步收敛或唯一病因。模型、采样定义和历史输出没有改动。

## 固定范围与身份

只补录 P0 / patch0、patch4 × seed10401、10402 × B、J × BF16 Euler50、100，共16条已有案例。N=112、C=12、free=100。8份 FP32/100 全轨迹与 teacher 数据只读复用旧档；不增加 FP32 完整生成。扰动标准差为模型坐标1e-4，仅自由坐标，两个固定高斯方向61401/61402，24个状态，各双精度基线及正负扰动，共240次前向。

B=纯FM Native T1+Geo 累计3000，state `{ASSETS['B']['state']}`；J=JEdge5 累计3000，state `{ASSETS['J']['state']}`。每条模型、dtype、NFE、case、C/noise/raw/state/file hash 见 [身份清单](case_identity_manifest.json)，图片输入另见 [热图清单](geometry/directional_plot_manifest.json)。旧档来源为 `jedge5_diagnosis_20261002_102215`。

需先纠正数值前提：以下均为排除C后的对称表面RMS，单位% bbox。

{endpoint_table}

因此 BF16/100 显著恢复的是 J/10401；10402 仅由5.032%改善至3.547%。此前0.872%与4.530%那一对属于FP32/100，不能混入BF16结论。

## 1. 实际速度与轨迹分离

[逐步CSV](dynamics/per_step.csv)分别保存真实模型自由速度L2、Δv、Δv/Δt、相邻速度夹角、实际位移及raw已知槽速度；[对时CSV](dynamics/aligned_50_100.csv)按X50[k]与X100[2k]比较。模型参数FP32、前向BF16、积分FP32，严格确定性、TF32关闭、SDPA math。

D下表是固定自由槽位的三维点RMS/L，不是到GT表面的RMS。

{dtable}

两异常案例在t=.02已出现非零离散误差，在t≈.20/.22超过0.5% bbox，t=.5前已超过2%。后段增长更快，但不能描述成“最后10步才突然分叉”。正常B/patch0/10402终点D也有5.432%，表面RMS却仅.223%/.099%；槽位距离不能替代表面质量。沿表面位移或不同三角划分可能造成这种差异，本轮没有证明具体槽位交换。

没有看到t=.5速度脉冲：J/patch4/50在t=.48/.50/.52的速度点RMS分别为10401：29.520/29.664/29.770，10402：28.711/28.831/28.904（% bbox / 单位时间）。50步两例的最大Δv/Δt均在t=.02。100步末端t=.99差分增大也出现在正常B/patch0；50步并不查询t=.99，不能简单比较两者最大差分并叫作更强刚性。Δv沿轨迹同时含状态和显式时间变化，不是空间Jacobian。

![轨迹对时距离](dynamics/trajectory_divergence.png)

## 2. 方向敏感性探针

每案例按预登记规则选“D增量最大区间起点、最大Δv时刻、t=.5”，去重后补齐3点。基线BF16查询24/24与原轨迹记录v逐位相同。表为共同t=.5的两个方向平均中心响应增益 ||v(X+ε)-v(X-ε)|| / ||(X+ε)-(X-ε)||：

{pt}

同一模型内双精度使用完全同一状态与扰动；B/J在t=.5是各自轨迹状态，不能称跨模型同状态因果比较。全部48个同状态同方向精度配对的BF16增益均高于FP32，比值中位数B约2.83、J约2.41。微扰响应受精度影响，且FP32双边响应仍有非线性/非对称成分；只有两个方向、一个扰动尺度，不能估计条件数或Jacobian谱范数，也没有做步长稳定域检验。

J/patch4/10402在t=.5的FP32响应约为B的1.79倍，10401则略低，未形成所有坏例普遍更高的证据。各自“关键点”时间也不相同，不能当匹配条件横比。完整原始基线/±速度、实际FP32输入扰动、参数与RNG保护见 [探针合并结果](probes/combined_results.json)。

## 3. 面上误差与GT缺失区域：分开展示

独立评分仍使用原两方向各512个面积均匀采样点，16条三项RMS全部对账。逐面累加的是平方误差贡献；总体对称RMS=sqrt((前向MSE+后向MSE)/2)，绝不相加逐面RMS。

{gt}

J/patch4/10402/50的对称MSE按生成面归属后，top3占25.47%、top10占58.78%，累计90%需40张面。这里后向覆盖误差按最近生成面归属，仅用于加和，**不代表存在GT槽位对应**。只按预测面前向MSE排序的前十自由面ID为 `{top_ids}`（0-based，自由ID+12才是完整112面槽位）。逐面三角形、面积及坐标见 [逐面表](geometry/endpoint_faces.csv)。

10402最大边长约.690L，与正常B同例接近；其自由面积只有GT未知部分的62.33%，100步也只有68.05%。P0/patch4全部8条B/J轨迹均无固定面积阈值退化，J四条没有相邻步法向余弦变负。面积主要平滑缩小，最大单步总面积变化在第一步；末端窄面增多，10402的<5°面数为50步28/100、100步26/100。证据更符合覆盖不足/形状偏差和窄面问题，不支持“少数巨大面飞离”或面积爆炸。不能将面积比本身当水密体积或收缩机制证明。

下图上排只画预测面free→GT平方误差贡献（含1/512权重），下排在真实GT未知部分R12的原512点上画GT→free距离平方；可见根本未覆盖好的区域。每个方向跨图使用固定色阶、同视角同范围，两方向聚合口径不同，不共用一条色标。100步10401的覆盖RMS从5.021%降至.340%，而10402仍为3.955%。

![10402 双向误差与缺失区域](geometry/directional_error_P0_patch4_seed10402.png)

[10401对照热图](geometry/directional_error_P0_patch4_seed10401.png)；[正常patch0/10401](geometry/directional_error_P0_patch0_seed10401.png)；[正常patch0/10402](geometry/directional_error_P0_patch0_seed10402.png)。每时刻100面最小角、面积、边长比、最长边/最小高及法向变化均保存在 `geometry/*_face_shapes.npz` 和 [沿程表](geometry/trajectory_shapes.csv)。

## 4. Edge求积与投影：仅离线

**采样不执行Edge/L5损失、投影或t=.5训练门控。** 实际D_edge按可信GT源顶点ID预先确定对应的known/free边；仅在对应有限线段上进行内部/端点截断，不在全局候选边间动态选目标。自由生成没有GT面角身份，所以本轮没有伪造对应、没有对自由轨迹计算“训练Edge”。

两种离线数据严格分离：

- 真实teacher数据：复用已存P0/patch0、4的20个J BF16 teacher状态和有效源标签，重建Yhat并对账原Edge；最大标量差3.49e-10。保存每对应边、每方向5点的距离、权重、未截断投影参数、截断分支。端点两点对总值的份额，patch0均值38.98%、patch4均值32.01%。这些是固定teacher状态的离线复算，不是历史逐次optimizer训练的实时日志；t<.5的原始项也可作观测，但不表示训练时门控激活。未计算这些真实例的连续积分，不能把份额叫37.5倍。
- 自由轨迹proxy：100面×3边×5点，对C物理外边界Γ求最近段，保存距离、argmin、次近margin、平局和A-B-A。Γ与C边关联数1的集合一致。B的2269次ABA只有8次三个时刻均无平局；J的1966次只有9次。因此>99.5%的ABA窗口至少有一个平局，很多ID变化不能解释为实质目标跳跃。NFE不同机会数不同，必须使用各轨迹分母；这些proxy没有反馈到采样。

37.5特例已复现：共线等长段，ell=L=1、δ=.01，五点值δ²/8=1.25e-5，连续值δ³/(3ell)=3.333e-7，值比37.5；一阶导数比25。正交平移两个比均为1。它既不是普遍常数，也不是模型速度场刚度。

见 [teacher点级数据](edge/teacher_points.csv)、[teacher对账](edge/teacher_summary.json)、[Γ轨迹proxy](edge/trajectory_summary.json)、[固定夹具](edge/fixture_record.json)。

## 5. 噪声和输入C：仅描述

初始自由噪声300个点：10401均值约(0.0324,-0.0038,0.1716)，协方差最大/最小特征值比1.320；10402均值约(-0.0237,-0.0256,-0.0066)，比值1.145，反而更接近各向同性。高斯没有规范椅子朝向，主轴有正负号自由且近等特征值方向不稳，PCA不能证明10402需要“180°翻转”。两seed也不足以推断坏噪声因果。

{context}

patch4已知面积较少，但bbox体积比反而更大；bbox比不是可见表面积或条件信息量。闭环曲折度记录周长/(2×直径)，开放链才用弧长/端点弦长，避免闭环除以零。所有10个P0/P1 patch的C边关联、边界组件、平面偏离、源link异常都在 [条件与噪声结果](geometry/endpoint_results.json)，没有修补输入C，也没有由这些描述推断唯一病因。

## 执行、保护与限制

实际新增optimizer更新0、模型反向0、OT0。16条成功记录共1200次模型前向，24状态探针240次；另有首次记录器误设“raw已知速度必须为零”的断言，B/patch0/10401/50采样已完成但记录落盘前停止，50次前向单列保留，随后仅修正观察断言并重录。因此**实际1490次前向、17次完整采样执行、16份有效轨迹记录**。失效尝试的raw未保存，不能伪称17份都已记录。没有新增案例或训练。

C由采样器每步固定坐标实现逐位保持；raw已知槽模型输出并不恒零。有效16轨迹全部1216个状态C逐位保持，旧档保存的中间状态与终点全部逐位一致，16条速度记录可逐位重构实际Euler更新。已知槽raw速度从未施加到C。模型state哈希前后不变；公共模型/采样/损失文件、权重和历史输出保持不变。

CPU夹具仅3次几何autograd，另有29次公共Edge CPU求值（包含6次有限差分求值及20 teacher重建），都不是模型反向。几何/探针CPU测试及记录器hook测试有独立记录。图和数值来自同一冻结输入；没有重抽噪声、选最好检查点、修补网格或改变默认模型。

资源快照记录24 CPU核、约61GiB可用RAM；B首个运行设备显存增量约1.1GiB，以保守1.5GiB/worker估计，保留至少2GiB余量，再启动J，峰值并发2。WDDM进程显存为N/A，所以设备增量不是精确进程峰值。完整采样成功worker合计墙钟 {timing['replay_success_sum_sampler_wall_seconds']:.3f}s；包含加载/校验的worker各自时间、探针与离线几何时间见 [结果JSON](results.json)。并行时间有重叠，事件计时含观察同步，不能当独占算力benchmark。

用户关于“共用GPU锁”的补充在全部GPU调用结束后到达；本轮使用资源快照调度，**未使用共用锁**，不能声称已完成跨窗口互斥或另外两窗口的工作去重。随后只做CPU分析与绘图，没有再启动GPU，也没有重做此前全数值矩阵。查找本仓库和WSL /tmp未发现可复用锁路径；未创建另一把假共享锁。

最终能成立的是：步长敏感确实存在，10401的100步改善对应覆盖恢复；10402仍主要表现为覆盖不足。早期轨迹分离、方向敏感性、量化效应与条件面积差异已经量化，但本轮没有识别唯一因果链。所有记录和图已交付，到此停止，不自动训练、改loss、换求解器或扩大矩阵。
'''
    (OUT/'BLACKBOX_REPORT.md').write_text(report,encoding='utf-8')
    print('REPORT ASSEMBLED')

if __name__=='__main__':main()
