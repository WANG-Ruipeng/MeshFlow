# Bench 推理、后处理与评分接线核查

状态：**本次核查范围内未发现能解释坏结果的接线错误**。这不认证模型质量，也不等于已经排除训练、数据分布或适配不足的问题。

本次只读源码和已保存资产，不加载神经模型、不做前向、不重新生成、不重新评分。新增文件仅在本目录。Windows/WSL 创建失败后改用 Windows 的现成 Python 3.14 / NumPy 读取静态 NPZ；未重启 WSL。没有修改源几何、原条件、权重、旧 RAW/POST 或原评分。

## 1. 实际缓存：96 份终点输出，24 项逐输出核对全部通过

证据：`evaluation_cache_audit.json`；复核入口：`evaluation_cache_audit.py`。96 份指 48 个冻结请求 × NATIVE_S21000 / SEAM_S21000；每份同时读取 RAW、51 帧轨迹、D0 提议、pre_geo、POST 与来源 C。**2304 项检查通过，失败 0；模型调用、几何评分、新噪声均为 0。** 源文件 SHA256 完整保存在该 JSON。

- 输入 N/K 与 RAW 形状一致；模型 C 和来源 C 只差已保存角点排列，没有另行归一化或替换。
- 第 0 帧严格等于该请求保存的噪声加 C/SEAM 锚点；第 50 帧严格等于保存的 RAW；51 帧每个锚点都保持原坐标字节。
- RAW 索引解码没有改坐标；来源 C 角序逆映射正确。
- `pre_geo` 严格等于原 D0 提议，仅恢复规定 C/SEAM 锚点。不存在错恢复第三点、恢复了错误面槽或把整个 F 换回旧值的情况。
- POST 索引网格与 `output` 一致；来源面 ID 唯一；C 面仍在前 K 行；SEAM/BULK 身份随保留面 ID 正确搬运。
- C 顶点到 POST 顶点为双射；保留下来的 SEAM 锚点不仅坐标相同，也确实引用对应 C 顶点索引。
- 每份记录的生成器调用为 50、D0 为 1；GEO 半径为原 `.015/(.95*.3762)`；D0 权重 SHA 为官方 Bench 身份。
- 推理使用的 `source_condition.npz` 不含 Y、完整目标或 owner 标签。来源信息仅用于保留 C 自身身份及导出，并非把源物体补入输出。

这些检查能排除本轮常见的 N/K、噪声身份、角序、槽位角色和锚点错接；不能代替实际生成质量评判。

## 2. 与 Chair 的同阶段代码比较

| 环节 | Bench 实际接线 | Chair 同阶段对照 | 判断 |
|---|---|---|---|
| 权重 | `sample.py` 严格加载 generator state，校验 checkpoint SHA、张量 hash、MODEL_CONFIG，结束时复核 state 未变 | Chair generation 同样加载冻结 live generator | 实际 NATIVE 与 SEAM 来源不同且正确；本核查不重复 root 的全权重 CPU 审计 |
| Euler | 复用 `bench_control_transfer_v1.sample.rollout`；t=0..49/50，dt=1/50，BF16 前向、FP32 状态积分 | Chair `seam_abc_v1.generate.rollout` 相同算式与 clamping | 未发现符号、步长、终点或精度路径错接 |
| 条件与角色 | NATIVE 调原 Native；SEAM 调 `seam_forward`；C 进入 Geo；SEAM 两角点 known、第三点 free；B=0 退回原 Native | Chair SEAM 采用相同 corner-time/role 规则 | 没有把 SEAM 整张面误设 known，或把 seam 第三点锁死 |
| N 与尺度 | 总面数 N 输入生成器；D0 也读取整张 RAW 的总 N；未按生成 bbox 重归一化 | 同一模型坐标定义、固定 L 和 GEO 半径 | 名称中的 Chair 不代表误用了 Chair 数据尺度；未发现本轮额外缩放 |
| D0 | 官方 Bench EMA `0021000.pt`，SHA `67765c728c503b61c21624c05997ea026c9de486d1529052fafa134be3336b14`，synset 02828884 | Chair 独立 SHA 为 `6a31f4...`、synset 03001627 | **未误加载 Chair D0** |
| D0 调用 | `predict_proposal`：输入乘 .35739，直接 NeRF 编码并 decode，一次 FP32 residual 加法，再除同尺度 | 同一 portable 函数 | 无随机 posterior、额外去噪轮次、重新排序或 padding |
| D0 std/time | Bench 配置 fixed_std=.05，Chair=.02；这里走直接 decoder，不走加噪 encoder | 相同直接 decoder 路径 | .05 不是本轮额外抽噪声；此 D0 无 diffusion timestep 可误接，条件为面数 |
| GEO/定向 | 复用 `seam_slot_c200_v1.postprocess.geometry_from_proposal`、`geo_c.process`、来源角序导出 | Chair 原始 SEAM/GEO 路径一致；ABC 仅额外支持 shared-tip alias | 本轮未启用 alias，原函数适用；没有少执行已注册的处理阶段 |
| 评分 | 私有 namespace 调未改的 `seam_abc_v1.evaluate.score_one` 与 `endpoint_scoring.metric_row` | Chair quality-wave 同样调用这两者 | 124 字段数学与相交分类未换定义；只扩展标签、路径及闭合 C 的 NA |

实际终点采样身份：

- NATIVE_S21000 → `training/PLAIN/generator_step21000.pt`，SHA `36a8d84a6ecc97b90c0deda625e77534a43e6dbc1fe45c47c7acada74660f0fa`。
- SEAM_S21000 → `training/SEAM/generator_step21000.pt`，SHA `98b5fb8e562c59e18a40ef40f0cbc7497a5e4088d3bb36cfb35a4b2bcc70acc9`。
- 两者使用同一冻结评测 manifest，SHA `68384ef1f726e53d43ae9c610584df5303ab4aeaffbdb5c5728c54debdf3e650`。

## 3. 必须保留的比较边界

**Chair 的 `PLAIN_S2000` 并不等于 Bench 的 `NATIVE/PLAIN`。** Chair quality-wave 的 PLAIN 名称指不增加新质量损失的 SEAM 续训；它仍复用 LONG 的 SEAM 布局并走 `seam_forward`。Bench NATIVE 才是没有 seam 锚点的普通 Native。把这两个 PLAIN 标签当成相同表示会构成解释错误，但当前两臂采样代码未混淆它们。

Bench 本轮冻结为单次官方 Bench D0 → 原 GEO_C → 来源定向；**没有 HINGE、1.5%L 簇绑定或后续 CPU 补接**。因此可与 Chair 的原 POST 阶段比，不能把 Chair 额外后处理后的最好成品，当成本轮已执行同样处理的结果。

闭合 C 没有开放边，专属边界与 seam-to-bulk 项应为 NA；开放 C 预测全 Null 仍保留零接续，不被改成 NA。C 字节保持只认证输入保留，不认证用途采用。图中悬空、错层、细腿消失等仍需如实报告，不能被 anchor PASS 抵消。

## 4. 对极高 RAW FF 和断片的解释限度

原评分扫描的是保存网格 `output` 的 free 三角面，FF 分母是涉及非法相交的自由面占比，并非非法面对数除所有面对数。SEAM/BULK 对分类用的是保留面对应的原 RAW 槽位角色；本轮已验证这个映射未串位。

RAW 的约 93%–95% FF 本身不能证明单位错了。该值在 D0/GEO 之前就存在，**不能由后处理代码倒因果地产生**。POST 大量删面和细支撑变薄也是真实保存产物；完整后处理链减少 FF 与视觉支撑损失可以同时发生。

目前没有证据把极高 FF 归咎于 Bench/Chair 权重串用、N 错送、角点排列或评分面身份错误。也不能据此宣称坏结果只能来自数据或模型能力：训练端、官方初始化分布、条件跨度与多部位适配仍是其他核查的范围。这里不追加训练、阈值扫描或恢复旧模型。

RAW→POST 观察只能归因于**完整 D0＋锚点恢复＋GEO＋定向链**。没有分阶段重新评分，不能把某项改善或断片损失全部单归因于 D0。来源定向只变角序，不改变几何或删面。

## 5. 结论

本次找到的是两个需避免的比较误读——Chair PLAIN_S2000 的名称含义，以及额外后处理范围不一致；**未找到需要重跑终点评测的已证实推理/后处理接线 bug**。保留现有结果的负面质量证据与 C 侧接续证据，等待其他训练/数据审计结论合并，不因接线 PASS 就把当前模型认定为可用。

执行限制：0 模型调用、0 GPU、0 新样本、0 新噪声、0 几何重评分；原结果及源码未修改。
