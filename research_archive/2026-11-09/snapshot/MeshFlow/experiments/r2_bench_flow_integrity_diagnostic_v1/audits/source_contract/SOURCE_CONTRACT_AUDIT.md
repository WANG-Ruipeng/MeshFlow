# Bench 生产 FM 耦合路径：源码与实际 CPU trace

本项没有训练、模型前向、反向、自由采样或 GPU 计算。旧文件只读。新增代码为 `r2/pilots/bench_flow_integrity_v1/contract_audit.py`，可恢复入口为 `python -m r2.pilots.bench_flow_integrity_v1.contract_audit run`。已完成项按哈希复用，未完成的已尝试 OT 不静默重跑。

**结果：16/16 生产打包 trace 通过。当前代码没有把 SEAM 第三点的噪声跨不同固定锚排序交换。** 对既定 C 与布局 H，HYBRID 也没有改变自由坐标的标准高斯边缘分布。此结论不等于模型已学会一致共享顶点、不等于来源 owner 与部署 C-only 布局一致，更不认证自由生成质量。

## 实际执行范围

从覆盖实验冻结面板读取六条 C：27eb79、6b894、91e169、a8e987、df6fd3、fd355；每条两份既有噪声。按 `(condition_id, 原 manifest 的 seed)` 排序，注册索引决定 epsilon seed `2026102300+i` 与排列 seed `2026102400+i`。噪声没有重新抽取，完整源目标无增强，alpha=1。

- Native：12 次生产 CPU OT。
- SEAM source-owner：仅 6b894、91e169 的四份诊断，共 4 次 CPU OT。其布局读取来源监督，明确不是可部署 C-only 推理。
- 每次只在 t=0.5 调用一次真实 `D.supervised_sample`。随后对同一 x0/Y/U 构造 t=0.1、0.3、0.5、0.7、0.9；没有随时间重新 OT。
- 16 次 OT 墙钟合计 0.0833 秒；该数不含 CPU 源面对资格筛查、读写或 GPU 诊断。

预登记与输入/源码哈希见 `manifest.json`；GPU 可直接读取的 16 行包见 `probe_manifest.json`。每行包含 `trace_path/trace_sha256`、`pair_path/pair_sha256` 和完整父/条件/噪声身份。实际调用账本为 `ledger.jsonl`，逐项证据为每份 `audit.json` 和 `per_trace.csv`。没有大权重哈希或模型加载。

## 生产入口和实际逻辑

以下路径均相对仓库根目录，行号与 `manifest.json.source_inventory` 中的文件哈希对应。

| 环节 | 源码位置 | 实际行为 |
| --- | --- | --- |
| 新 Native 覆盖训练 | `r2/pilots/bench_seat_coverage_v1/train.py:208` | `D.sample_for_training(..., 'PLAIN')` |
| Native→SEAM 课程训练 | `r2/pilots/bench_full_seat_curriculum_v1/train.py:360` | Native 传 PLAIN；SEAM 传 SEAM；随后同一自由坐标 FM |
| 公共 CPU 路由 | `r2/pilots/bench_parts_convergence_v1/data.py:212` | 开放 C 的 Native 转 maintained FULL4 打包；SEAM 转 owner-only 扩展；闭合 C 单列普通 free 分支 |
| Native 的匹配 | `r2/src/meshflow_control/seam/slot_data.py:120`、`:149` | source occupancy 对 Native 置零；所有非 C 面作为 bulk 匹配到原始噪声槽 |
| SEAM 的 owner 锚 | `r2/pilots/seam_abc_v1/slot_data_extension.py:34` | 每个来源 owner 固定在 `K+edge_id`，角点 0/1 绑定对应 C 边，tip 角点 2 保持自由 |
| SEAM 的 OT 范围 | 同文件 `:44`、`:51` | 从匹配目标中排除 owner 面，只在 bulk 中执行 nested OT；Null/alias 返回 bulk，不复制来源面 |
| nested OT | `r2/src/meshflow_control/training/coupling.py:23` | 同一 mesh pair 内，六种角点排列的 9D 距离取最小，再 Hungarian 匹配面。实际 `cdist` 默认是欧氏 L2，**不是平方 L2**；不是跨 minibatch 对象 OT |
| inverse map | 同文件 `:45` | 反排列来源 Y、face ID、corner permutation 进入固定噪声槽；没有重排 z |
| HYBRID | owner 扩展 `:55`；maintained slot_data `:153` | 原 z 与独立 epsilon 逐槽混合 `sqrt(.75) z + .5 epsilon`，再精确写入固定角点 |
| 最后整面排列 | owner 扩展 `:64`；maintained slot_data `:163` | x0/Y/Xt/U、锚、mask、source face/corner 一起排列；不是单独交换 tip |
| 推理起点 | `r2/pilots/bench_control_transfer_v1/sample.py:75` | 现成标准高斯直接进入 free 槽、固定角点写 C；每步 Euler 后仍写回固定角点。无推理 OT |
| FM 分母 | `r2/src/meshflow_control/seam/model_adapter.py:119`；课程 trainer `:363` | 只对 valid 且非固定角点的三坐标求 SSE；微批使用整有效批自由标量总数作分母 |

原 maintained `slot_data.py` 中旧 SEAM 分支的 third-in-C 限制，不应误读成当前生产限制。当前 owner-only 扩展允许此类来源面，但其第三点仍是普通自由 tip，而非偷偷加入新的 GT 硬锚。

## 数组层验证

所有 16 份均保存原 z、epsilon、HYBRID、pre-OT 半尺度目标、OT 输入目标/角序/face map、逆映射后 Y、源 face/corner 身份、整面排列、anchor 与各 mask、x0/Xt/U、五时刻状态。

实际验证包括：

1. `2*pre_ot_full == full_target`；所有来源 face 恰好出现一次。
2. `full_target[source_face_ids, source_corner_permutations] == x1` 精确往返。
3. bulk 原始随机循环角序、face 随机排列和生产 OT 逆排列共同复原到完全相同的 target/身份。
4. 原 z/epsilon、来源 case 均未被就地修改；原始 noise slot 加 HYBRID 并恢复锚后与 x0 逐值相同。
5. 每个来源 owner 的 edge 槽/来源面/角序一致；所有 tip 的 x0 与同一原 edge 槽的混噪第三点完全相同，最大差为 0。
6. 固定坐标精确保留，固定角点 U=0，C face mask 与 seam 角点 mask 区分正确。
7. 自由标量分母为 `9(N-K)-6*active_owner_count`，与实际 free mask 的 `3*sum` 相等。
8. 五时刻 `Y = Xt + (1-t)U` 的差仅为 FP32 算术舍入（阈值 2e-6），锚差为 0；t=0.5 与生产返回 Xt 精确相同。

## 能排除什么，不能排除什么

对于固定 C/H，z 与 epsilon 独立标准高斯，混合方差 `.75+.25=1`。所以自由 x0 仍为标准高斯；这个结论来自独立仿射高斯代数，不是用 16 个 trace 的样本均值/方差“证明”分布。OT 选择依赖 z 的 Y 配对，改变联合耦合，不能因此说 z 的边缘也被重排或被条件化筛选。已保存噪声样本矩仅作描述。

生产 source-owner H 由训练来源提供，部署 H 由 C-only 头产生，两者可能改变 owner 数、锚边与自由度。这是已知的布局输入差异，本项既不将其命名为新增实现 bug，也不证明它无害。SEAM 四份仅描述其真实训练耦合；不能拿它与 Native 三模型作为相同输入的因果对照。

最近的覆盖候选是 Native，没有 SEAM owner 输入。因此即使后续发现 SEAM 部署布局存在误差，也不能用它解释 Native 在新 a8/27 上的全部问题。本项排除的是特定的“跨锚噪声交换/错误元数据搬运”解释；共享顶点预测误差、插值拟合是否兑现到自由生成、以及 D0/GEO 对细结构的损失仍需其他已注册证据。

最近历史路线是 R20（HYBRID 配方）与 R31（共享原边槽的有界正结果），共享源角点一致性另有 R07 试验。本次只读取并核验已有生产路径，不新增 loss、表示、匹配规则或训练候选。
