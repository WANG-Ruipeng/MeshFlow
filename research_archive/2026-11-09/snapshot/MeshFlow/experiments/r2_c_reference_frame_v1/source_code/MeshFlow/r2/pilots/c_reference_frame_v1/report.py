"""Write a results report from frozen tables; no scientific computation or model calls."""
import csv,json
from pathlib import Path
from . import common as C
O=C.OUT
s=C.read(O/'summary.json');panels=s['per_panel'];conditions=s['per_condition']
labels=['EXP67_FROZEN','SMALL_C200','CREF_C0','CREF_C200']
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def p(panel,label,stage):return next(x for x in panels if (x['panel'],x['label'],x['stage'])==(panel,label,stage))
def pc(x):return f'{100*x:.3f}%'
parts=['''# R2_C_REFERENCE_FRAME_V1：条件参考坐标 200 步对照

**实际完成日期：2026-10-13。结论：出现跨噪声、跨条件的 RAW 采用正信号，但几何代价和接续没有通过；保留研究结果，不晋级默认模型。**

相对相同预算的 SMALL_C200，熟悉面板出现 4 份、DEV 出现 3 份明确 NO→YES，分布于 5 个条件。没有明确 YES→NO，但有两份 YES→UNCERTAIN，不能说没有损害。熟悉 POST 的 I 从 1.050% 增至 8.737%，FF 从 18.424% 增至 22.598%；合法接口输出由 15/32 降至 14/32。DEV 接口由 1/16 变为 0/16。当前不能称为更好的完整生成方案。

本轮已执行 200 次优化器更新、1600 次训练前向/反向、96 份新增 Euler50 RAW、96 份单次官方 D0 POST、192 份新阶段评分；另复用 96 RAW、96 POST 和对应 192 份旧评分。48 页四组匿名对照及 8 张全噪声图已实际审阅，保存 384 条 AI_PROXY 逐阶段判断和 32 条款式记录。没有只生成脚本或挑图。

## 1. 比较的是什么

| 名称 | 权重与操作 |
|---|---|
| EXP67_FROZEN | 原 EXP67 step6000；复用身份完全相同的既有输出 |
| SMALL_C200 | 原世界坐标下集中续训 200 步；复用既有输出 |
| CREF_C0 | EXP67 原权重直接置于新坐标/起点包装，零更新 |
| CREF_C200 | 同 EXP67 参数、AdamW 与 RNG 状态，在新包装下续训 200 步 |

CREF 只以输入 C 的面积加权中心 a(C) 作平移参照。宽度增强先作用于完整 C/Y，再计算 a；主干读取 C-a 和 free 的局部状态，现有 Geo13 仍读取原世界坐标 C，同步保留槽位、角点排列和 mask。没有旋转、缩放、语义标签、新参数或新 loss。

局部 free 继续从原独立标准 Gaussian 起步，HYBRID=0.25，free-only FM；局部目标匹配重新执行 OT。现有 OT 用欧氏距离，不是可以直接沿用旧排列的平方代价假设，因此实际重新计算了 1600 份匹配。没有把已缓存的世界 xt 减中心冒充新训练状态，也没有将噪声去均值。

最终 free 加回 a，外部 C 在全部 51 个保存状态中直接复制原输入字节。POST 一律为单次官方 D0、C 恢复、GEO_C、来源 C 角序恢复及固定 C 的 free 定向。没有 RBF、多轮去噪或自选阈值。

这检验的是“主干坐标表述＋条件参考起点”整体策略。保留 world Geo 意味着本轮不能宣称完整平移等变，也不能证明已经消除利用绝对位置记忆的可能。

## 2. 训练和面板身份

- 固定原 SMALL_C200 的 1600 条有序曝光记录，逐条核对原实际输入/标签身份；4 父各曝光 400 次。batch=8、microbatch=1，每批各父 2 条，4 clean＋4 width U(0.9,1.1)。
- 相同 lr=1e-5、AdamW betas=(0.9,0.95)、weight decay=0、clip=1；恢复 EXP67 的原优化器动量，局部 1—200、累计 6001—6200。不混合 EMA 权重与其他优化器。
- BF16 前向、FP32 参数/积分，TF32 关闭、SDPA math、严格 deterministic、CUDA 初始化前固定 CUBLAS 配置。无新增学习式前向探针。
- 固定终点 step200；step100 仅用于恢复，没有挑 best。
- 熟悉条件：73b369 HALF(K11/N231)、3f7417 HALF(K6/N168)、85b169 FULL(K12/N166)、8c9b20 FULL(K16/N183)，每条 8 份已冻结噪声，共 32。
- DEV：5b1d0d FULL(K32/N168)、608b46 FULL(K2/N140)、2d08a6 HALF(K1/N238)、610b16 HALF(K1/N228)，每条 4 份冻结噪声，共 16。
- C、N、来源、Gaussian 数组与每条件 2048 个查询点均复用冻结身份；本轮没有新增噪声。历史更早的旧噪声只留作单独参考，没有混入本轮分母。

DEV 父未进入这份 67 父条件训练，但已经被用于开发筛查，不能包装成未见独立测试，更不能声称官方预训练从未见过。K32 超过旧 FULL 条件的训练上限；K1 HALF 与集中训练的 K6/K11 不同。父、覆盖、K 的变化不能独立归因。

## 3. RAW 采用与 POST 接口

采用为真实图审的 YES/NO/UNCERTAIN，未把 UNCERTAIN 当作失败或成功。各行采用计数依次为 YES / NO / UNCERTAIN；RAW/POST 分别评阅。
''']
for panel,title in [('TRAIN_FAMILIAR','熟悉条件（4父×8噪声）'),('DEV_HELDOUT','DEV 条件（4父×4噪声）')]:
 rows=[]
 for label in labels:
  a=p(panel,label,'RAW');b=p(panel,label,'POST');rows.append([label,' / '.join(str(a['C_adoption__'+v]) for v in ['YES','NO','UNCERTAIN']),' / '.join(str(b['C_adoption__'+v]) for v in ['YES','NO','UNCERTAIN']),f"{b['valid_edge_outputs']}/{b['n']}",f"{a['whole_body_complete__YES']}/{a['n']}"])
 parts += ['\n### '+title+'\n',table(['模型','RAW采用 Y/N/U','POST采用 Y/N/U','POST有合法共边','RAW主体明确完整'],rows)]
parts += ['''
RAW 的严格 A-VMR/PMR 与合法共边四组均为 0；这与图上看起来接近不矛盾。图审采用评价用途和布局，不能认证精确索引连接。所有拓扑计算状态均完成，没有把缺失评分当 0。

对 SMALL_C200 的明确 RAW 收益：熟悉 73b n3/n4、3f n2、85 n6；DEV 2d n2/n3、610 n0。即 7 个输出、5 条 C，不是 7 个父或所有输入都改善。熟悉 8c 没有明确新增采用。熟悉 73b n1、DEV 608 n1 从 YES 变 UNCERTAIN；熟悉共 8/32、DEV 4/16 配对含不确定项。

同预算 SMALL→CREF 的明确 RAW NO→YES / YES→NO 为熟悉 4/0、DEV 3/0。相对 EXP67 是熟悉 4/1、DEV 2/0。相对 CREF_C0 是熟悉 5/1、DEV 2/0。CREF_C0 的熟悉采用从 EXP67 的 15/32 退到 10/32，POST 合法接口从 12/32 退到 0/32；直接换包装会破坏先验，不能把后续恢复全部记成超过原方案的新能力。

### 每个条件的分布

以下为 POST 的 YES/NO/UNCERTAIN 与“有至少一条合法共边”的噪声数。除 CREF_C0 的 8c 一份输出由 RAW 不确定转 POST 明确采用，其余采用计数在 RAW/POST 一致。
''']
rows=[]
for panel in ['TRAIN_FAMILIAR','DEV_HELDOUT']:
 for cid in sorted({x['condition_id'] for x in conditions if x['panel']==panel}):
  row=['熟悉' if panel=='TRAIN_FAMILIAR' else 'DEV',cid[:6]]
  for label in labels:
   x=next(x for x in conditions if x['condition_id']==cid and x['label']==label and x['stage']=='POST')
   row.append('/'.join(str(x['C_adoption__'+v]) for v in ['YES','NO','UNCERTAIN'])+f"；接口{x['valid_edge_outputs']}/{x['n']}")
  rows.append(row)
parts.append(table(['面板','父','EXP67','SMALL200','CREF0','CREF200'],rows))
parts += ['''
## 4. 完整质量代价

均为每父先平均噪声，再对 4 父作宏平均。表中全部是百分数；CF/FF 是受相应非法关系影响的 free 面比例，I 是固定 C 查询的冲突覆盖联合量。三者分母和几何判据不同，不能互相替代。A-VMR/PMR 使用现有 strict <1e-6 模型单位、free 顶点和全部源 C 开边界定义；保留原 8 阈值列，没有声称此单位已与外部论文尺度完全一致。
''']
for panel,title in [('TRAIN_FAMILIAR','熟悉 POST'),('DEV_HELDOUT','DEV POST')]:
 rows=[]
 for label in labels:
  x=p(panel,label,'POST');rows.append([label]+[pc(x[k]) for k in ['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction','A_VMR__raw_tau_1e-6','PMR__raw_tau_1e-6','opposite_winding_reached_free_area_fraction']])
 parts+=['\n### '+title+'\n',table(['模型','CF↓','FF↓','I↓','小角<5°↓','A-VMR↑','PMR↑','方向可达面积↑'],rows)]
parts+=['''
RAW 阶段相对 SMALL，熟悉 FF 89.420%→82.499%，DEV 90.102%→83.985%；不能据此说完整 POST 也改善。熟悉 POST FF 上升约 4.174 pp，主要来自 73b 的 11.609%→31.979%；DEV POST FF 上升约 1.763 pp。POST 删除/整理改变了实际 free 面数和分母，逐输出实际面数已保存。

熟悉 I 的净增量约 93% 来自 3f 和 73b。I 包含穿越覆盖与近距离平行表面代理，不应把其每一处都称为已认证的非法穿透。已实际读取 64 份已有 score/逐面对/逐点掩码，按原查询权重复算，未重新调用评分器。

原评分器的 dup 是：C 内部查询点离 C 边界至少 0.01L，与 free 有限三角面距离小于 0.005L，法线夹角不超过 15°（绝对点积，所以两向法线均可）。L 沿用原评分归一化。分母仍为全部 2048 查询点的权重，不是只除内部点。它不是 exact duplicate 或严格共面正面积重叠认证。

| POST 父均值 | I：SMALL→CREF | cross：SMALL→CREF | dup：SMALL→CREF |
|---|---|---|---|
| 3f7417 | 0.800%→14.435% | 0.800%→3.546% | 0→11.731% |
| 73b369 | 1.776%→16.742% | 1.776%→1.447% | 0→15.436% |

cross 和 dup 有交集，不能相加。3f 的 POST I 中位数为 5.591%，n2 占本父 I 总和约 65.96%；73b 中位 3.296%，n4/n6 占约 88.88%。但它们仍分别在 7/8、6/8 噪声上较 SMALL 增加，不能说只有单点。64 份记录的 exact duplicate faces 均为 0；现有逐点掩码缺少最近 free 面身份和正负侧，尚无法区分合理薄底面与多余座层，这一原因判断为 NA。73b 的 FF proper-pair 总数同时从 102 增到 533，是独立于 dup 代理的几何代价。详见 results/I_COST_NOTES.md 和完整逐噪声小表。

采用收益与质量代价需要逐例并列：
''']
with (O/'results/paired_decisions.csv').open(encoding='utf-8-sig') as f: pairs=list(csv.DictReader(f))
with (O/'evaluation/per_output.csv').open(encoding='utf-8-sig') as f: numeric=list(csv.DictReader(f))
idx={(x['label'],x['request_id'],x['stage']):x for x in numeric}
rows=[]
for x in pairs:
 if x['comparison']=='CREF_C200-minus-SMALL_C200' and x['stage']=='RAW' and x['C_adoption_pair']=='GAIN':
  a=idx['SMALL_C200',x['request_id'],'POST'];b=idx['CREF_C200',x['request_id'],'POST']
  rows.append([x['parent_id'][:6]+'/n'+x['candidate'],pc(float(a['ConflictCov_C']))+'→'+pc(float(b['ConflictCov_C'])),pc(float(a['FF_illegal_affected_free_fraction']))+'→'+pc(float(b['FF_illegal_affected_free_fraction'])),a['C_F_valid_edge_count']+'→'+b['C_F_valid_edge_count']])
parts.append(table(['RAW明确采用收益例','POST I','POST FF','POST合法共边数'],rows))
parts+=['''
3f/n2 的 RAW I 约 76.27%，其中 dup 约 75.68%、cross 约 3.76%，两者相交不能相加；POST I 仍约 76.17%。这是“座面用途看起来更对，但近距离平行面覆盖很高”的例子；它需要区分正常厚度与多余层，不能直接认证为可用成品，也不能把 76% 全称为非法穿透。73b/n4 同样出现明显 I 增量。相比之下，85/n6 采用改善、I 仍 0、FF 略降，并多出一条合法接口，是较干净的局部正例；2d/n3 也有较小代价的采用改善，但没有精确接口。

## 5. 图审、款式与判断稳定性

全部 48 请求页显示四匿名模型的 RAW/POST、全形与两个穿过 C 中心的同尺度剖面；每条件固定 A/B/C/D 映射，所有噪声同尺度。淡蓝轮廓只是投影，不能当真实剖交。384 条判断和 32 条风格记录先保存并核图像 hash，之后才读取映射和数值。

评阅者均为 AI_PROXY，参与者对历史实验图有一定熟悉度，匿名不保证无法识别模型；这不是人类盲评。完整性、采用、悬空、替代层、错层、支撑、碎片/折层与可见接触分别记录，没有强行互斥。可见接触与评分合法共边也没有互相代填。

CREF200 的 3f 在 7/8 噪声中接近同一弧顶/条背四腿家族，另一份为桶状壳体；因此不应仅凭成功数宣称多样性提高，也不能据此断言复制训练网格。其它条件有横档、实背、框架等变化，逐条件款式记录完整保留。本轮没有新增最近邻训练集复制检验，不声称排除了记忆。

旧缓存图重审存在变化：SMALL 的 RAW 1/48（8c/n4 NO→UNCERTAIN），EXP67 的 RAW 5/48（均转不确定，含 8c/n7 YES→UNCERTAIN）。因此 EXP67 熟悉 YES 由历史 16/32 变为本轮 15/32；数值评分和网格完全没变，旧报告未覆盖。使用历史 SMALL 标签重配，仍是熟悉 4、DEV 3 次明确收益、0 次明确损失，主收益计数不依赖本次重审漂移。使用历史 EXP 标签则是 5+3 次收益而非本轮 4+2；两套敏感性都保存。historical_review_sensitivity.csv 中不存在的转移类别空格表示次数 0，不是缺失测量。

噪声是条件内重复，不能把 48 份输出当作 48 个独立父。不做大样本显著性或普遍泛化结论，也没有用来源完整款式作为唯一正确答案。

## 6. 真实工作量、资源和工程记录

- 新局部 OT：1600 份，2 个 CPU worker，约 13.14 秒实际计算。
- 训练：200 更新、1600 forward、1600 backward，单次真实训练尝试完成，训练状态墙钟 340.60 秒（约 5 分 41 秒）。末步 FM=0.1080，仅为训练记录，不作为采用成功依据。
- 新采样：CREF0 48 个 Euler50 轨迹约 172.45 秒阶段时间；CREF200 48 个约 163.97 秒；合计 4800 次生成器前向。
- 单次 D0：两组各 48 次，实际阶段各约 60.29/61.07 秒。全部新增 RAW 保存后才进入 D0。
- 新 CPU 评分：192 阶段，2 worker×2线程，约 220.82 秒；渲染 56 图约 37.21 秒，与评分部分重叠。旧 192 阶段评分验证身份后复用，没有伪计新增模型调用。
- GPU 波次逐次检查资源后选择 1 worker，兼顾用户图形项目及约 6 GiB 使用意向。训练 PyTorch peak reserved 约 3.05 GiB（allocated 约 2.70 GiB）；采样 reserved 约 0.60 GiB；D0 约 1.78 GiB。训练监测的整设备使用峰值为 5903 MiB，含背景进程。WDDM 无法提供可靠单 PID 峰值，因此不把 PyTorch 分配器数值冒称完整进程显存证明。未终止无关进程，未发生 OOM。
- 调用账本及 569 项源资产 hash 检查 PASS；训练成功/尝试计数一致，无待定或失败模型调用。

实际有一次 POST 启动失败：96 RAW 已完成后，导入顺序误选安装包，找不到 source_winding；发生在 D0 加载/前向之前，D0 调用为 0。仅提前建立本仓库包路径，并加入显式父 hash 链校验；旧源码、diff、注册 hash 与失败日志均保留在 engineering/001_import_path。没有重采样或重训。独立 AST 检查确认未改变科学定义。

另有一次纯离线 CSV 显式零值排版尝试被不可覆盖保护拒绝；已撤销排版修改，原结果字节保持，重新聚合验证一致。记录在 engineering/002_offline_table_format_attempt.json，新增科学调用为 0。

实际总体耗时包含代码、审计、图审与报告，不能把 5 分 41 秒训练时间说成整个项目完成时间。各阶段计时及并行情况见 call_accounting.json；独立终检记录见 independent_audit/final_audit.json。

## 7. 决定与可检验假说

**本版本不晋级，也不把它归为完全没有信号。** 它在多个 C 上产生了新的 RAW 用途改善，超过“只让距离变近/多几条 POST 接口”的证据；但部分改善伴随重复覆盖，严格接续没有同步提高，整体质量门槛未过。

唯一优先后续应是先核对这些明确收益例中的替代层是否真的消失，还是移动到 C 附近形成重层。可检验假说是：条件参考坐标使座面高度更靠近 C，却仍可能保留同一区域的额外 free 表面；因此采用改善会与 dup 覆盖上升在部分同请求中共现。本轮已有逐点/逐例评分支持这种局部共现，但合理薄底面也会触发 dup；须先核最近面身份与正负侧，尚不能把它认定为唯一机制，更不能据此直接设计新 loss。当前 200 步固定试验到此停止；没有追加训练、切换默认、commit 或 push。

## 8. 交付与恢复

- 完整数值：[逐输出](evaluation/per_output.csv)、[逐条件](results/per_condition.csv)、[逐面板](results/per_panel.csv)、[配对决策](results/paired_decisions.csv)。
- 全部图：[图集入口](evaluation/gallery/index.html)；匿名原记录与解盲记录均在 evaluation/gallery/。大图原件与真实图审 hash 保留。
- 旧标签敏感性：[变化清单](results/baseline_review_agreement.csv)、[历史标签配对](results/historical_review_sensitivity.csv)。
- 实际训练权重：checkpoints/CREF_C200_step200.pt；轻量推理权重：inference/CREF_C200_generator_step200.pt。step100 恢复权重另留；均不放小包。
- 注册、数据/输入身份、调用账本、资源、工程修正、源码快照与源 hash 独立保存。新代码位于 r2/pilots/c_reference_frame_v1；原评分器、原资产及正式默认代码保持只读。
- [恢复入口](README_RESUME.md)。完成的模型调用只校验复用，不自动重做。
- R2_C_REFERENCE_FRAME_V1_REVIEW_SMALL.zip 为小于 10 MB 的审阅包，包含全部逐例 CSV、真实评分源码、code.diff、源 hash 和全部 56 图的统一压缩副本；不含大权重或原始轨迹。不能拿这个小包独立重跑几何评分，完整本地产物保留。
''']
report='\n'.join(parts)+'\n';(O/'REPORT.md').write_text(report,encoding='utf-8-sig')
print(json.dumps({'report':str(O/'REPORT.md'),'characters':len(report),'model_calls':0},ensure_ascii=False))