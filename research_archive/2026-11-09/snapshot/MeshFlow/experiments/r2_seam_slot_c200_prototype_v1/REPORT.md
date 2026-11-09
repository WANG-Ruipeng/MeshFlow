# SEAM_SLOT_C200 原型：C-only设计与CPU验证

**状态：原型代码与CPU合同验证完成；200步生成器训练、rollout及D0尚未执行。** 用户选择完整C-only占用预测，不使用来源接口oracle。没有把本轮合同检查写成生成效果，也没有自动启动6000步。

## 已经落地的最小实现

| 部分 | 实际内容 | 当前验证状态 |
|---|---|---|
| 训练数据 | 73、3f、85、8c四条TRAIN完整座面；同N、同来源几何 | 已打包，源文件hash未变 |
| 接口占用 | 仅C特征的24→32→1独立MLP，833参数，输出occupied/Null | CPU前向、反向、输入隔离检查通过，尚未学习 |
| Null | 不占用的边候选槽返回普通bulk，不删除面 | 全Null、全占用、交替占用共24个真实C layout检查通过 |
| 受限OT | seam固定原边owner及端点排列；只在bulk内部做原Hungarian＋六角排列 | 两臂共32份真实样本、32次OT通过 |
| 角点合同 | 原C与seam两锚点固定，只有有效自由角点计FM并积分 | 38项NumPy检查通过；小张量autograd通过 |
| 模型适配 | role、MLP/final按角选择时间；attention按面；Geo只看原C | 独立adapter已写，生产大模型前向尚未验证 |
| 输出身份 | 锚点直接复用C原索引 | CPU解码检查通过；未冒称bulk已焊接 |

38项NumPy检查包含混合角时间[1,1,t]、普通面退化合同、Null完整9坐标、padding排除、Euler夹持、损失分母及数值导数。数值导数最大误差6.92e-12，解析直线终点误差4.44e-16（FP64玩具合同）。真实32份FP32数据的终点、面/角身份与角点夹持分别保存于逐样本表，不把玩具精度当成BF16训练认证。

四条C合计66条开放边，62个来源占用标签、4个Null。3f有3个Null，8c有1个；73有22条开放边，不能把槽数写死为16。四条恰好没有多边同面owner及第三点引用C的特例；原型对这些输入明确报不支持，尚不覆盖所有47父。来源FF风险保留，没有修原网格。

正式推理入口只允许 `C_vertices / C_faces_original / C_faces_model / total_N` 四项；传入 `Y / parent_id / occupied_labels` 会报错。四条真实C的顶点重编号检查也通过。当前占用头未训练，保存的初始预测只用于layout机械检查，不报告占用准确率或接合能力。

## 修正后的200步方案

1. 从同一保留的 **FULL_S6000及匹配AdamW** 分叉新普通 `FULL4_C200` 与 `SEAM_SLOT_C200`，各200步，batch8/micro1，四父均衡、原HYBRID与精度。旧SMALL的3f/73是HALF，旧FULL_C200是67父全池，均不是这次机制的公平新对照。
2. 新占用头用独立fresh AdamW和明确的BCE，只读C特征，不与生成器共享参数。生成器仍用角级free-only FM；两者分别裁剪梯度。训练FM读取监督占用，推理必须读取预测占用，单列错接/漏接，不能在rollout回填来源答案。
3. 两臂各四C×四固定噪声，共32份新RAW与32份单次D0/POST。608/610保留DEV身份，不能为了凑困难例放进微池。
4. 主判据同时看RAW采用、合法CF边长、合法seam→bulk共边、从C到达的**bulk面积**以及CF/FF/I和主体。排除seam自身面积，防止“长一圈翻片”就获得高分。30%是可展示的门槛，不是完整性或self-attention机制证明。

占用BCE是新增离散预测目标，不能继续说完全没有新loss。来源占用只是某个参考完成的示范，不应成为开放生成的唯一GT成功标准。普通臂没有这个头；新臂的增量是“占用＋角点表示＋受限匹配”整体方案，本小试不会单独归因其中一个因素。

## 为什么没有采用原文的时间与门禁承诺

历史两臂200步训练各约321秒，合计已经10分41秒；若各16输出再处理和评分，计算量级约16分钟，首次实现、OT准备、检查与图审另计。不能承诺5—20分钟排完所有风险。

嫁接现成RAW再D0是另一种终态几何干预，可能在输入时先扯断邻域；它不能作为共享表示学习的必要通过门槛。本轮没有增加这项GPU任务。完整训练入口仍须把mixed-corner模型、占用优化器、预测占用采样、D0后锚点恢复与GEO索引核对接起来并在实测资源下运行。当前模块级CPU通过不等于完整训练器已验证。

## 文件与恢复

- [设计与训练合同](../../r2/prototypes/seam_slot_c200_v1/DESIGN.md)
- [CPU总入口](../../r2/prototypes/seam_slot_c200_v1/run.py)
- [数据与受限匹配](../../r2/prototypes/seam_slot_c200_v1/slot_data.py)
- [角点模型适配](../../r2/prototypes/seam_slot_c200_v1/model_adapter.py)
- [占用预测头](../../r2/prototypes/seam_slot_c200_v1/occupancy.py)
- [C-only推理入口](../../r2/prototypes/seam_slot_c200_v1/inference.py)
- [逐条件](conditions.csv)、[32份逐样本检查](per_sample.csv)、[推理layout检查](inference_layout_checks.csv)、[真实C及小头检查回执](model_contract_checks.json)。训练标签与C-only输入分目录存放。

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/prototypes/seam_slot_c200_v1/run.py --stage all
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/prototypes/seam_slot_c200_v1/check_inference.py
```

已完成时上述入口核验保存的hash后退出，不重复OT或小头前向。CPU诊断使用两档alpha(1,1.05)×两份固定随机数组；这些只是合同测试数据，未冒充正式200步训练流。准备阶段资源记录为RTX5080总16303MiB、空闲7403MiB、GPU11%，24逻辑CPU、CPU20%；本轮GPU worker为0，未停止其他进程。

生成器与D0构建/调用均0，优化器更新0，无新权重。小占用头有明确计数的CPU测试前向/反向，不能笼统写“零神经网络前向”。旧数据、评分器、源代码与报告保持只读；没有cleanup、改默认、commit或push。
