# Native 6000 + SEAM 3000 最终训练核账

状态：**PASS**。两阶段训练、各自 workflow 和 attempt final_status 全部完成，workflow exit_code 均为 0。独立审计重新读取全部调用、曝光、训练日志与 JSON 旁车；未读取张量、重 hash 大权重、调用模型、执行评分或干预训练。详情见 [审计 JSON](final_training_9000.json)。

总计 9000 次更新、72000 次前向、72000 次反向、72000 实例曝光。两阶段各一个 attempt，失败、挂起、科学重试、训练头前向/更新、探针调用均为 0。逐例与冻结日程一致，包括父/C、identity/x-width 增广和全部随机种子。原优化器没有重置；COMMON→Native0 及 Native6000→SEAM0 的生成器、AdamW、RNG、冻结头四项状态身份均吻合。全局游标为 12000→18000→21000。

22 父 / 22 C 的实际曝光为：Native 20 父各 2182、2 父各 2180；SEAM 8 父各 1092、13 父各 1090、1 父 1094；累计 8 父各 3274、14 父各 3272。总 identity 与 x-width 各 36000 份。逐父完整记录见 [training_parent_exposure.csv](../results/training_parent_exposure.csv)。

Native free=bulk=9(N−K)，tip 无自由标量并记 NA；SEAM 采用来源 owner 的 S 个槽，tip=3S、bulk=9(N−K−S)、总分母9(N−K)−6S。逐例 SSE、整批分母、FM 贡献和角色分解均闭合。不同阶段自由坐标与表示不同，下表仅为各阶段内的来源监督训练诊断，不能替代生成质量或直接按跨阶段绝对 loss 排名。

| 阶段/角色 | 首100 velocity MSE | 末100 velocity MSE | 首100 endpoint MSE | 末100 endpoint MSE |
|---|---:|---:|---:|---:|
| Native all free / bulk | 0.224411 | 0.096127 | 0.087533 | 0.048944 |
| Native tip | NA | NA | NA | NA |
| SEAM all free | 0.106536 | 0.096694 | 0.055046 | 0.052163 |
| SEAM tip | 0.256504 | 0.028922 | 0.116963 | 0.014585 |
| SEAM bulk | 0.102951 | 0.098312 | 0.053566 | 0.053060 |

上述 MSE 都是各窗口累计 SSE / 累计自由标量数，不是每例或每步比率的均值。endpoint SSE 来自保存的 `(1−t)^2` 训练误差统计，无额外 endpoint 前向。完整逐步曲线、非重叠100步窗口、部位/时间分层及分母分别见 [loss_curves.csv](../results/loss_curves.csv)、[training_windows100.csv](../results/training_windows100.csv)、[training_strata.csv](../results/training_strata.csv)。[loss_curves.png](../results/loss_curves.png) 已实际查看：六面板完整、Native tip 未画成零、阶段独立配色，图示为各阶段内部 trailing100 pooled。

| 阶段 | trainer 墙钟秒 | input 秒 | update 秒 | allocator allocated / reserved 峰 MiB | 整设备已用峰 MiB |
|---|---:|---:|---:|---:|---:|
| Native | 8437.61 | 249.38 | 7786.01 | 2795.23 / 3156 | 6226 |
| SEAM | 4241.58 | 116.23 | 3912.91 | 2804.86 / 3148 | 6413 |

input 包含资源检查与样本/OT准备；update 包含前后向、统计、裁剪、AdamW 和逐样本写账，不含周期 checkpoint 写盘。两阶段 trainer 墙钟相加约 12679.20 秒，不能再加 input/update 分项。两段训练分别与 COMMON RAW 采样重叠119.65秒、NATIVE RAW采样重叠123.13秒；这是实际并发墙钟，不是独占计算速度。allocator 峰不是进程显存峰；WDDM 未提供独立进程峰，保持 NA。整设备峰包含背景及并发进程。

SEAM3000 最终生成器 state SHA256 为 `0a46da0705c22915f835d7515c3f0407f7d9f1af29673aad8a7d0b0df6dfb4d9`，冻结头为 `21229acce25e3e070ae3adfa580798e1534e704cc6512e006de1654d2fefb9c4`。终点 checkpoint 与推理导出的旁车身份一致；这是对生产者记录的状态 hash 的核对，本审计未独立重新加载权重核值。

已实际执行且仅执行一次 `summarize_training --final`，15.28秒、exit0；五个图表/CSV产物 hash 核验通过，正式输出为 [training_summary.json](../results/training_summary.json)。[单次执行收据](summarize_training_final_execution.json) 保留命令、开始/结束时间及产物身份。未做源归档。

训练结案不等于整个实验或模型质量结案。最终收口仍须独立核对 controller READY_REVIEW、全部 RAW/POST/评分科学账本及真实逐图审阅；外层 pipeline COMPLETE 不单独作为成功依据。
