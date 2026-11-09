# 三主臂最终训练只读审计

结论：PASS。T1、Geo、Full 各自已从 MVP48 step1000 完成到 step5000；三臂共真实新增12000次 optimizer 更新、96000个微批前向及96000个微批反向。本文件只验收主训练，不包含独立拟合探针，也不将尚未完成的评价或交付算作完成。

审计过程只读训练资产并在本轮 audit 目录写证据；CPU 数值线程为2，未初始化 CUDA，未进行任何模型前向、反向、OT 或 optimizer 更新，未修改冻结代码。审计时刻：2026-10-02T22:34:57.209469+00:00。

## 更新、共享输入与身份

- 每臂 CSV、UPDATE_BEGIN、UPDATE_VALIDATED 均恰好4000条，step1001–5000连续且唯一；逐步计数、有效更新标志、有限 loss 和输入哈希一致。每臂终点 attempted/returned forward/backward 均32000，optimizer attempted/returned 均4000。
- 三臂4000批的实际 batch SHA 逐批完全相同，并与共享输入登记的4000个唯一 step 匹配。没有失败、重复训练或同目标安全恢复事件；每臂恰有3个 SEGMENT_COMPLETE 事件。
- 2000、3500、5000共9份固定 checkpoint 的实际文件 SHA 已重算，与 sidecar 相符。variant、logical_step、source1000身份、loss/context、冻结 protocol 哈希及累计 counts 均匹配完成 summary。三臂均 RESTORE_ALL，optimizer_reset_at_1000=false。
- 三份5000端点均实际以 CPU mmap 读取，验证 model/optimizer/RNG/stream 内容哈希；full_recovery_state=true，AdamW 各状态 step=5000，lr=1e-5、betas=(.9,.95)、wd=0，stream batch_index=5000、扩展 sample_index=32000，计划哈希一致。保留完整 Python/NumPy/Torch CPU/单卡 CUDA RNG。此检查没有执行模型，也没有重放 optimizer 更新。

| 模型 | 固定步数 | 完整恢复状态 | 文件SHA256前16位 |
|---|---:|---|---|
| T1 | 2000 | False | f29ab2bfd213fe14 |
| T1 | 3500 | False | b19114d24b396e38 |
| T1 | 5000 | True | 3a024846907fc3ec |
| Geo | 2000 | False | 1a2a2ed6c579f64a |
| Geo | 3500 | False | 12bc4672ab077c0f |
| Geo | 5000 | True | b845aa73c1d3ecf7 |
| Full | 2000 | False | cf3ed46cc3612ff9 |
| Full | 3500 | False | 535ddfd0a1408c1b |
| Full | 5000 | True | 37a76253a032bb8f |

2000/3500固定状态为模型检查点；每阶段另有完整 latest 支持继续训练。最终5000为完整恢复端点。完整文件、模型和源身份哈希见 MAIN_TRAINING_FINAL_AUDIT.json。

## 模块实际变化与 Full ramp

以下数值从实际5000模型张量与各自受保护的1000源张量重新计算，并与 checkpoint 记录交叉一致；数值说明参数确实更新，不等同于生成质量改善。T1没有Geo模块，0表示不适用。

| 模型 | 模块 | 参数数 | L2变化 | 最大绝对变化 |
|---|---|---:|---:|---:|
| T1 | role | 1536 | 0.051857128 | 0.006904831 |
| T1 | Geo | 0 | 0.000000000 | 0.000000000 |
| T1 | backbone | 130249731 | 6.747027564 | 0.007220238 |
| Geo | role | 1536 | 0.036132182 | 0.005474995 |
| Geo | Geo | 149888 | 0.513813892 | 0.005767667 |
| Geo | backbone | 130249731 | 6.683766436 | 0.006918907 |
| Full | role | 1536 | 0.036117472 | 0.005391951 |
| Full | Geo | 149888 | 0.508803476 | 0.005634646 |
| Full | backbone | 130249731 | 6.698968858 | 0.007000089 |

Full 冻结调用始终传 additional_step=logical_step-500，因此新1001–5000步对应501–4500，原 min(additional_step/50,1) 始终为1。另用全部4000行日志独立重建 active_samples/8 × (lambda_edge×mean_active_D_edge + lambda5×mean_active_L5)，与实际 geometry 的最大绝对残差为6.6420e-11；非零项实际/重建比例为0.9999998788–1.0000000860，符合 FP32 舍入。证据不支持任何1001/2001/3501重新预热。

## 九段实际计时

每段 summary 与对应 append-only SEGMENT_END 完全一致；CSV 行级 train_seconds、OT_seconds 的求和分别与该段 worker 字段一致。每段实际 OT attempted/returned 为更新数×8，九段合计96000，缓存命中0。

| 模型 | 步数范围 | 新增更新 | 训练段秒 | 输入准备秒 | 保存校验秒 | 加载秒 | worker墙钟秒 | 控制器墙钟秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | 1000→2000 | 1000 | 1084.645 | 24.033 | 85.428 | 16.874 | 1238.258 | 1243.221 |
| T1 | 2000→3500 | 1500 | 1630.318 | 36.087 | 133.813 | 17.564 | 1857.816 | 1863.069 |
| T1 | 3500→5000 | 1500 | 1645.010 | 36.059 | 129.751 | 17.422 | 1868.076 | 1873.288 |
| Geo | 1000→2000 | 1000 | 1109.398 | 24.099 | 88.899 | 18.092 | 1264.724 | 1269.840 |
| Geo | 2000→3500 | 1500 | 1667.180 | 36.207 | 134.853 | 17.564 | 1892.450 | 1897.581 |
| Geo | 3500→5000 | 1500 | 1671.776 | 36.104 | 127.819 | 18.524 | 1890.997 | 1896.324 |
| Full | 1000→2000 | 1000 | 1182.652 | 24.058 | 88.698 | 17.666 | 1337.633 | 1342.821 |
| Full | 2000→3500 | 1500 | 1789.324 | 36.205 | 132.695 | 17.729 | 2012.598 | 2017.856 |
| Full | 3500→5000 | 1500 | 1797.468 | 36.430 | 128.727 | 17.104 | 2016.445 | 2021.678 |

合计：训练段13577.770秒，输入准备289.282秒，保存校验1050.683秒，加载158.539秒，锁等待0.000301秒，worker墙钟15378.996秒，控制器子进程墙钟15425.678秒。worker墙钟减上述已分项时间余302.721秒；这是未单独计时及范围差异的合计，不把它伪造分摊给某个算法。

## 字段范围与命名限制

- 重要：finalize.py 将训练行的 forward_seconds 直接赋值为 worker_train_seconds。该字段实际包含 zero_grad、Full目标准备、8次模型前向/loss/反向、clip、AdamW、CUDA同步及参数有限性检查；不能把它当作纯前向耗时，更不能据此与推理forward latency比较。应在最终报告中称为“训练更新段耗时”。未修改冻结实现。
- OT_seconds 同样不是纯OT内核时间：它计入整个 next_effective_batch 的噪声/时间/排列、CPU OT、标签核对及样本哈希。
- worker_save_seconds 包括CPU拷贝、状态哈希、序列化、文件哈希和恢复内容校验，固定点还包括参数变化统计；函数末尾gc及外层latest指针/事件写入等尾部成本未全部计入。
- load_seconds 包括模型/AdamW/RNG/stream恢复与源文件身份哈希。worker_wall_seconds 从预备stream和初始资源检查之后开始，包含锁等待，但不含最初导入/准备及 finally 最终清理。控制器 elapsed_seconds 覆盖整个子进程，范围更宽；这两种墙钟不能重复相加。
- 这些是本次运行实际观测时间。资源监视、CPU审计和用户其他进程可能影响吞吐；不能称为无竞争的独立GPU性能基准。allocated包含于reserved，不相加；global显存峰值含其他进程，process峰值为null表示WSL未能取得。

## 恢复与最终整体验收边界

本次九段没有检测到同target安全恢复，故现存summary的成本合计与所有SEGMENT_END事件一致。若其他阶段恢复，同名summary可能被最后worker覆盖，而累计训练counts从latest继续；应以独立pid/时间/target的append-only SEGMENT_END与FAILURE补算成本，避免累加累计counts。

本文件证明三主臂12000新增更新已完成；整轮13000更新和104000微批仍需另核probe1000更新/8000微批。最终还须验收固定生成、teacher、替换条件、留出回归、报告与产物，不能仅凭主训练PASS宣称整轮完成。

证据文件：MAIN_TRAINING_FINAL_AUDIT.json、main_training_segments.csv、main_training_component_changes.csv。
