# 资源展示修订与执行账本复核

修订状态：PASS。原 resource_summary.csv 已完整保留为 audit/resource_summary_original.csv。本次只修改派生资源CSV及本说明；未修改冻结代码、模型、原始输出、评分、训练/评价ledger或 execution_accounting.json。没有模型调用、CUDA初始化、OT或optimizer更新。

现表共39行：11个训练段、28个评价worker，其中curve15、swap7、fit3、holdout3。wiring预检不在此39行资源表中，但其27个forward已计入总执行账本。所有worker均由已完成的真实summary/ledger生成，不从计划推算耗时。

## 字段修订

- 删除有歧义的 forward_seconds 名称，原值保留在 legacy_forward_seconds，并给出逐行scope。训练使用 training_update_seconds，实际包含zero_grad、目标准备、模型forward、loss、backward、clip、AdamW、同步和参数有限性检查；不能理解为网络纯前向耗时。
- 评价 sampling_seconds 为各实际生成 row.audit.seconds 之和，含原Euler50循环、C/有限性检查、同步及逐状态CPU传输，不是纯网络forward。teacher_diagnostic_seconds 为各teacher row.seconds之和，覆盖其forward及已计时的离线诊断；原计时点位于后续D_edge/L5辅助量计算之前，不能称为完整teacher全部CPU成本。
- model_load_seconds 直接读取真实字段。训练包含模型/AdamW/RNG/stream恢复与源checkpoint哈希；评价包含load_model和身份校验。scoring_seconds取各生成row实际scoring_seconds，包含raw/input/reference读取及CPU几何评分，后续评分JSON落盘不在该计时内。
- training_input_preparation_seconds沿用真实worker_OT_seconds，包含整个batch输入准备、Gaussian/t/排列、CPU OT、标签检查与样本哈希，不是纯OT时间。save_and_verify_seconds仅对训练填入已测保存/校验时间；评价raw保存没有独立计时，填NA。原OT/save/load字段另保留legacy列。
- network_forward_only_seconds、loss_only_seconds、backward_only_seconds、optimizer_only_seconds、OT_algorithm_only_seconds均未独立测量，填NA，不做残差推算。某项不适用与未测量均不伪造为0；实际没有teacher请求的worker，teacher_diagnostic_seconds为0。

## 真实计时合计

| 范围 | 秒 |
|---|---:|
| 39个worker elapsed之和 | 22350.550 |
| 锁等待 | 5.313 |
| 模型/状态加载 | 453.948 |
| 训练更新段 | 14665.459 |
| 生成采样 | 2543.483 |
| teacher已计时诊断 | 45.478 |
| CPU生成结果评分 | 2019.643 |
| 训练输入准备（含OT） | 312.275 |
| 训练保存和校验 | 1153.607 |

worker_elapsed_seconds已经包含自身的大部分子项，不能与表内子项再相加。训练worker计时起于最初stream/资源准备之后，且不含finally最终清理；评价计时包括锁等待至CPU评分结束。控制器子进程墙钟范围更宽，不能重复叠加。未单独测量的开销不从差值分摊。这些是实际运行时间，不构成独立无竞争的GPU基准。

## 显存范围

训练allocated/reserved峰值在恢复后重置，包含驻留状态、backward/optimizer和loss workspace；评价值为各rollout/teacher重置范围的最大值，含驻留模型。allocated属于reserved，不相加。

训练nvidia-smi监视记录中的global最大值为6455 MiB，包含其他进程占用。评价从真实阶段/请求前快照得到的最大global used为3913 MiB，另列global_gpu_used_snapshot_max_mib；它是离散快照最大值，不声称为连续监视峰值。评价未有连续global峰值数据的列仍保留NA。全部39行process专属显存峰值均为NA，原因是WSL未提供可用值，不以global值或0替代。

## 最终计数只读复核

逐worker ledger及四臂最终训练counts，与 execution_accounting.json 和 execution_budget_audit.json 相符：

- optimizer attempted/returned：13000。
- 训练微批forward attempted/returned、backward attempted/returned：各104000。
- 完整生成：1192次；对应1192×50=59600模型forward。
- teacher：738次forward。
- wiring：27次forward，0次optimizer更新。
- 全部模型forward：104000+59600+738+27=164365。

execution_budget_audit为PASS且errors为空；run_state为COMPLETE、worker_pid为空。未增加重试、生成、teacher、OT或更新。资源展示表修订本身不改变任何科学结论。

原CSV SHA256：9cd5d0dd2b8c404b05d8f98705691ffc6c4f3ae1a618aa2cc5cf011117e4561c。
修订CSV SHA256：beb652fb4352460d4efde02e26557ce351ac97a1b12a5bef854f84ee2714e7aa。
