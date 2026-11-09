"""Write the Chinese report from completed frozen diagnostics only."""
from pathlib import Path
import json,statistics as st
from .registration import OUT,CASES
from experiments.mf_h1_local0.jedge5_diagnosis.common import read_json

def run():
 d=read_json(OUT/'results.json');a=d['analysis'];local=a['local']['trajectories'];amp=a['amplification']['rows'];suffix=read_json(OUT/'raw/suffix_rows.json')['rows'];scores={m:read_json(OUT/f'scores/{m}/results.json') for m in ('B','J')}
 lines=['# JEdge5 Local Error and Amplification — 窗口 B', '',
 '本轮已完成冻结 FP32 轨迹诊断。局部步长差会在原 Euler100 后缀中放大，但没有统一的病发时间，也没有证明 J 比 B 普遍更敏感。单步替换的几何变化很小且有取舍，现有结果不足以支持新增训练、更换时间门控或把它推广为新采样方法。', '',
 '新增训练、optimizer 更新、OT、模型反向、EMA、完整生成和历史轨迹重放均为 **0**。真实模型前向 **2797 次**，包含首个诊断实现断言错误前已完成的 **1 次**。完成 **36 条后缀**，没有额外 seed、模型、GT 配对或几何修补。', '',
 '## 1. 身份、执行路径与保护', '',
 'B/J 均为 raw 累计3000步、Native T1 + Geo。使用此前工作条件诊断 `jedge5_diagnosis_20261002_102215` 的实际 FP32 Euler100 数组、两个 Gaussian、C 与审计槽位。没有按 seed 重抽噪声，也没有读取 BF16 轨迹来替代。', '',
 '|模型|state SHA256|', '|---|---|',
 '|B / 纯FM|`f006c49c8c70efa4bf2df6ca96431c844a25a1bdd38e2db4756115a37a23545b`|',
 '|J / JEdge5|`b9c1df5326ad4815ec3c32ed20e08dc34ff6248ff05be2c412e051fcb728aae3`|', '',
 '上下文归档 SHA256：`2154881d77b946a2a18959503930fd1d405f88f0d1f56d64a8f0d86710ed01b4`。B/J 文件 SHA、实际公共源码及输入哈希详见 [protocol](registration/protocol.json)。70项主保护检查及独立评分源检查全部通过；每组模型载入前后 state SHA 一致。没有修改公共模型、采样器、metrics、窗口A文件、旧数据或权重；没有 commit/push。', '',
 'N112 / K12 / free100。所有模型参数、前向、时间、状态和积分均为 FP32；严格确定性、TF32关闭、SDPA math、eval/no_grad。在实际前向记录了坐标嵌入、QKV、attention、MLP和速度的 dtype。mask依据旧审计明确的 known/free/valid 槽位列表重建，而非从几何猜测；旧NPZ本身没有单独mask数组。', '',
 '## 2. 门禁、必要实现修正和预算', '',
 '- 144个主状态及12个 t=.8 补充状态：x_E 与归档下一状态 **156/156逐位相同**，传播因此获准。',
 '- CPU数学12项、worker解析场控制流、评分schema及分析一致性门禁通过，记录见 [registration](registration/) 和 [tests](tests/)。CPU解析场调用没有计入真实模型前向。',
 '- 全部36条后缀的所有保存状态逐位保持 C；无 NaN/Inf，面数保持112。',
 '- 主时间表缺少必需的 t=.8，预先单列每轨迹v0/v_mid两次调用，共24次；未替换主时间表任何节点。',
 '- 首次worker错误地断言“模型原始known速度必须为零”，在1次前向后中止。公共Native模型实际返回所有面的速度；C保真来自仅积分free并复制C。移除这一额外错误断言后显式记录 attempt2，保留首试FAIL日志与1F账目。没有改变模型或积分路径。详情见 [预检记录](registration/preflight_incidents.json)。', '',
 '|内容|实际模型前向|', '|---|---:|', '|144主状态×6|864|', '|t=.8 必需补充|24|', '|36方向状态×两eta×正负|144|', '|36固定后缀|1764|', '|首个错误断言前的已完成调用|1|', '|总计 / 硬上限|2797 / 4500|', '',
 '所有方向分母均高于预登记门槛：实际初始操作数每坐标 FP32 ULP 上界的 RMS×8。它是表示分辨率筛查，不是神经网络误差上界。方向增益分母使用实际FP32扰动后的坐标差；没有以理论2ηS替代。', '',
 '## 3. 局部差异与传播：逐案例结果', '',
 '这里“局部差”是两个半步与一个全步之差，不是精确局部截断误差。数值RMS按900个自由标量坐标平均；几何评分RMS按三维点距离平均，两者不混用。下表局部峰值用固定GT bbox百分比表示，放大比的分母始终是实际初始局部坐标差。', '',
 '|case|模型|12节点中局部峰t|峰值坐标RMS / bbox %|RMS(δh)/RMS(δh/2) 中位数|放大比 t=.2 / .5 / .8|', '|---|---|---:|---:|---:|---|']
 for p,j,s in CASES:
  for m in ('B','J'):
   r=next(x for x in local if (x['model'],x['parent'],x['patch'],x['seed'])==(m,p,j,s));aa=sorted([x for x in amp if (x['model'],x['parent'],x['patch'],x['seed'])==(m,p,j,s)],key=lambda x:x['t'])
   amps=' / '.join(format(x['amplification'],'.2f') for x in aa)
   lines.append(f"|P{p}/patch{j}/{s}|{m}|{r['peak_t']:.2f}|{100*r['peak_delta_h']['coordinate_rms_bbox']:.6f}|{r['delta_h_over_half']['median']:.3f}|{amps}|")
 lines += ['', '各轨迹峰值节点上的速度分解如下；数值为原模型坐标单位的逐坐标RMS，未乘步长。跨模型列在同一x/t/C上计算，与积分差是不同量。', '', '|case / 模型|t|space RMS|time RMS|cross RMS|space-time余弦|同状态跨模型速度差RMS|', '|---|---:|---:|---:|---:|---:|---:|']
 for r in local:
  dec=r['peak_decomposition'];terms=dec['terms'];cos=dec['pairwise_angles']['space__time']['cosine'];mv=r['peak_same_state_model_difference']['coordinate_rms']
  key='P{}/patch{}/{}/{}'.format(r['parent'],r['patch'],r['seed'],r['model'])
  lines.append('|{}|{:.2f}|{:.7f}|{:.7f}|{:.7f}|{:.4f}|{:.6f}|'.format(key,r['peak_t'],terms['space']['coordinate_rms'],terms['time']['coordinate_rms'],terms['cross']['coordinate_rms'],cos,mv))
 lines+=['', '144/144主状态在总区间减半后差异缩小；各轨迹δh/δh/2中位数为2.57–3.66，不能由此声称精确四倍关系。12条轨迹的所测峰值各有6条在t=0和t=.98；这只覆盖预登记节点，未扫描连续时间极值。', '',
 't=0峰值的显式时间项大于状态项，二者余弦约+0.289至+0.744；t=.98峰值由状态项主导，与时间项余弦约−0.953至−0.903，范数抵消比例约62.5%–72.7%。分解重构恒等式全部在记录的FP32容差内成立。三项范数不可相加当成总变化或因果贡献率；完整RMS、L2、max、夹角、交互项和同状态跨模型速度差保存在逐状态CSV。', '',
 '36/36后缀最终坐标差放大，范围4.23–36.79倍；B的18分支放大中位数12.74，J为14.72；逐case、逐切点配对，J仅9/18大于B。这是各自轨迹、各自局部差方向的比较，不是同状态同方向的Jacobian比较。B多数在t=.5替换的比例更大，J有明显case依赖。各切点初始局部差不同，较大的比值不必意味着较大的最终绝对影响。', '',
 '最终坐标RMS/bbox有33/36条处于0.002–0.006%；最大为J P1/patch0/10401在t=.2替换后的0.050409%，其次J P0/patch4/10402的0.014980%。这证明原离散Euler100后缀会放大这些方向上的差异；不能等同于连续ODE混沌、刚性或对完整50→100步变化的因果解释。', '',
 '两档η=1e−4、5e−5得到的方向增益相对差（|g₁−g₂| / max(g₁,g₂)）：中位数38.56%，最大63.54%。明显的尺度依赖使稳定线性化解释不可靠；本轮无法分离FP32舍入影响和非线性。没有做Jacobian谱、全局Lipschitz或FP64验证。', '',
 '## 4. 完整后缀终点几何：变化量及取舍', '',
 '下表均为“单步替换后终点−同模型原FP32 Euler100终点”。RMS/Gamma数值为bbox百分比的**百分点**；每case保留全部三个预登记切点，不挑最佳。详细双向RMS、双向coverage、Gamma表面/有限自由边RMS/p95、CF/FF及退化见 [amplification_per_case.csv](amplification_per_case.csv) 与 scores 的逐输出记录。', '',
 '|case|模型|3分支Δ剩余RMS范围(pp)|RMS下降数|ΔFF范围(对)|ΔGamma表面p95范围(pp)|', '|---|---|---:|---:|---:|---:|']
 for p,j,s in CASES:
  for m in ('B','J'):
   rs=[r for r in amp if (r['model'],r['parent'],r['patch'],r['seed'])==(m,p,j,s)]
   rr=[r['delta_remainder_symmetric_RMS_bbox_pct'] for r in rs];ff=[r['delta_FF_illegal_pair_count'] for r in rs];gg=[r['delta_boundary_p95_bbox_pct'] for r in rs]
   lines.append(f'|P{p}/patch{j}/{s}|{m}|{min(rr):+.6f} … {max(rr):+.6f}|{sum(x<0 for x in rr)}/3|{min(ff):+d} … {max(ff):+d}|{min(gg):+.6f} … {max(gg):+.6f}|')
 lines+=['', 'B和J均12/18分支RMS下降，但B均值变化−0.000685pp，J均值+0.000217pp。B的FF合计−1对（8降/5增/5平）；J合计+13对（5降/9增/4平）。这是针对固定诊断分支的描述，不是方法排名或泛化成绩。', '',
 '- **P0/patch0/10401**：数值放大明显，几何变化很小；B部分FF减少伴随CF增加，J也没有一致FF改善。',
 '- **P1/patch0/10401**：B三个切点剩余RMS均略降；J在t=.2的绝对坐标变化最大，但剩余RMS几乎不变，R12→free与Gamma尾部略变差。',
 '- **P0/patch4/10401**：B/J剩余RMS多略降，Gamma尾部与FF变化随切点不同，不能据此选择正式采样切点。',
 '- **P0/patch4/10402**：J三个切点剩余RMS均变差；t=.2时剩余RMS+0.009422pp、R12→free+0.014596pp、Gamma p95+0.014311pp，FF不变。B在t=.5减少5对FF，却使剩余RMS和Gamma尾部略变差。',
 '- **P1/patch2/10401**：J在t=.5少1对FF，却让剩余RMS+0.001140pp、Gamma p95+0.003780pp；t=.8也不能以FF下降忽略目标覆盖和Gamma尾部代价。',
 '- **P1/patch2/10402**：J在t=.5的RMS略降但FF增加5对；t=.8 FF增加4对且Gamma尾部变差，没有一致修复teacher/生成差异。', '',
 '所有36输出无NaN/Inf，C逐位保持；退化、窄角、面积、CF/FF共面及UNRESOLVED沿用原评分记录，没有把相交下降等同于拓扑合法。精确索引拓扑、水密及完整流形认证本轮 **NOT_RUN**；不能据空字段宣布为0。原C继承的审计直接复用，未修复或归咎于本轮free变动。', '',
 '原评分已有双向RMS，但只有固定GT coverage。本轮仅增补原seed9911的512个RMS采样点在两个方向上的≤1%bbox coverage；保留原固定GT coverage另列。12个旧终点没有重跑CF/FF，只补了两方向CPU距离；36个新终点完整调用原独立评分器。', '',
 '## 5. 既有teacher证据与本轮边界', '',
 '只读复用旧400行teacher中的240行（t=.5/.75/.9）及8条参数层探针，没有重做teacher、梯度或EMA。原联合几何/FM梯度比跨B/J及两精度平均约0.4260→0.0972→0.0145；分精度、训练C/其它C及参数层结果保存在 [teacher复用](analysis/teacher_reuse.json)。', '',
 '这些teacher输入含固定GT/噪声配对，与本轮自由轨迹不是同一状态集合，未计算逐点因果相关。训练时的几何loss门控从未在本轮冻结前向中执行。局部曲线和teacher时间分配可以并列观察，但不能据此认定t=.5门控是唯一病因。', '',
 '## 6. 资源和时间分账', '',
 'GPU为RTX5080（16303 MiB），WSL affinity与nproc --all显示24个逻辑核、约61 GiB可用RAM；worker受OMP_NUM_THREADS=1限制，内部nproc显示1。严格单GPU worker，每case组在CUDA初始化前取得共享flock，组结束退出进程并释放显存/锁；CPU B/J评分独立并行，每进程限制1线程。并行窗口没有被终止，锁文件没有删除。', '',
 '|阶段|GPU组/成功数|锁等待(s)|持锁总wall(s)|已记录模型CUDA-event(s)|', '|---|---:|---:|---:|---:|']
 for stage in ('local','suffix'):
  p=d['phases'][stage];lines.append(f"|{stage}|{p['attempted_workers']}/{p['successful_workers']}|{p['lock_wait_seconds']:.6f}|{p['lock_owned_work_seconds']:.3f}|{p['measured_model_cuda_event_seconds']:.3f}|")
 lines += ['', '两阶段已记录模型CUDA-event共112.207秒；首次失败1F没有完整event计时，其wall已计入local。持锁wall包含加载、CPU统计、传输和哈希，不能与event相加。GPU占用是共享机器测量，不称为隔离吞吐。PyTorch最大allocated约1.031 GiB、reserved约1.119 GiB；它们不是完整进程显存峰值。已保存nvidia-smi前/中/后快照，完整CUDA进程瞬时峰值未独立高频采集。', '',
 f"CPU几何评分B/J进程wall分别{scores['B']['seconds']:.3f}/{scores['J']['seconds']:.3f}秒，两个进程并行；不将其相加当用户等待时间，也不冒称GPU采样。实际pair审计和96次补充coverage距离调用分别记入scores账本。", '',
 '完整11图的进程wall记录为9.013秒，首次完整汇总为1.972秒（均含WSL/Python启动）。后续汇总修订、先行局部图及目视检查未独立计时，因此CPU分析总时间标记NOT_MEASURED，未通过重跑补造计时。见 [CPU分析计时](registration/analysis_figure_timing.json)。', '', '## 7. 交付和停止', '',
 '- [结果JSON](results.json)、[局部逐状态CSV](local_defect_per_state.csv)、[后缀逐分支CSV](amplification_per_case.csv)、[实际代码diff](code.diff)。',
 '- [局部步长差曲线](figures/local_difference_by_t.png)、[速度分解](figures/velocity_decomposition_by_t.png)、[误差传播曲线](figures/suffix_growth_by_t.png)、[方向尺度对比](figures/directional_sensitivity.png)。',
 '- 六个固定case各有B/J、原终点及三个单步替换终点的同视角full/free图，均在 [figures](figures/)，没有只画成功样本；原始局部和后缀数组在 [raw](raw/)。', '',
 '**下一步判断：暂不据此修改训练loss或时间门控。** 证据支持“离散后缀会放大特定局部差异”，但对误差出现时间、J相对B的敏感性和几何收益均有case依赖；方向探针又有明显尺度依赖，尚不足以将RMS问题限制到某一窄时间段。与窗口A独立，本报告不读取其新结果，不选全局求解器胜者。固定任务已完成并停止。']
 (OUT/'B_LOCAL_ERROR_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('report written')
if __name__=='__main__':run()

