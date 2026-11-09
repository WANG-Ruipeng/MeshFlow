# 回到 START：反馈 / CAPPED 实验权重与缓存清理

按用户要求，继续以最初的 **Chair HYBRID START / total5000** 为 baseline。正式默认入口原本就是 `chair-hybrid`；本轮核验并保留，没有另选 checkpoint。

| 已删除项目 | 文件数 | 逻辑字节 |
|---|---:|---:|
| 反馈探索权重（四臂500/1000/latest） | 12 | 见逐权重清单 |
| CAPPED复现T/L/G权重（500/1000/latest） | 9 | 见逐权重清单 |
| HALF alpha0.5推理权重 | 1 | 521,670,611 |
| 上述22份权重合计 | 22 | 33,388,111,578 |
| 两处正式训练派生OT缓存NPZ | 16,000 | 38,644,198 |
| 总计 | 16,022 | **33,426,755,776** |

约33.43GB为被删除文件的逻辑大小，不宣称精确物理磁盘释放量。删除采用固定清单、逐文件SHA/长度复核及PowerShell LiteralPath；没有递归删除目录。

涉及的运行目录仅为 `chair_control_feedback_auto_v1`、`chair_capped500_gradbudget_v1`、`chair_capped500_half_v1`。删除的缓存仅限前两者 `streams/explore/ot_cache/*.npz` 与 `streams/repeat/ot_cache/*.npz`，各8000份。对应的16,000份attempt JSON与训练计划保留，防止旧入口遇到缺缓存后静默重跑。首批门禁的8个小型测试NPZ保留；这六个运行/源码目录未发现其它tmp/temp或Python字节码缓存。

原始RAW及网格、条件/Gaussian/查询点、反馈池、全部评分和逐例CSV、报告、图集、实际评分源码、训练/调用日志、checkpoint sidecar以及已交付的审阅ZIP都保留原样。实验源码仍保留，既供历史审计，也避免破坏原评分依赖；不将它当作仍可自动恢复的训练任务。

保留的START两条路径：

- `native_t1/checkpoints/chair_hybrid_start_total5000.pt`
- `native_t1/runs/chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt`

两路径是同一个文件的2个硬链接，并非两份物理占用。文件SHA256均为 `d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80`，模型state SHA为 `166c71fef888f26494fc4a0af1776b77a484efa72a6f3ce06552c604fbecc972`。完整AdamW、RNG和输入流仍保存在原checkpoint内，CPU推理加载不恢复优化器。

官方数据和权重、Geo4000、既定FM/JEdge5/G0/surface/edge-only和原A_FM等保留checkpoint均不在本轮删除范围。

验证与账本：

- 删除前一次实际CPU strict load通过：194参数张量、130401155参数，missing/unexpected为空，完整194项已保存AdamW状态均step5000。前后CUDA未初始化，模型forward被拦截且实际调用为0。
- 正式CLI实际解析到 `chair-hybrid` / `load_chair_start`；维护代码未引用此次三个退役实验包。删除后的 `python -B -m native_t1 sample --help` 通过。
- `verification.json` 为最终逐文件核验收据；登记保护43,405个原文件，实际结果为全部SHA前后相同、清单内删除无残留，最终核验PASS。最终完成状态同时见 `completion.json`。
- 固定清单在 `deletion_manifest.json`，每次删除前后事件在 `operations.jsonl`，保留文件前后hash在 `protected_before.json` / `protected_after.json`。`cleanup.ps1` 可按Prepare/Delete/Verify阶段核对；若恰在删除后、DELETE_OK落盘前中断，恢复会停止并要求核对账本，不静默跳过。本次正常执行没有此类中断。
- 三处checkpoint目录新增RETIREMENT.md，WORKSPACE.md更新当前baseline和清理状态。旧报告/审阅包中的“权重保留”和旧审计PASS仍是当时历史记录，当前文件存在状态以本轮清单为准；旧恢复、再采样和半幅构建命令不再承诺可直接执行。

本轮实际模型前向、训练、反向和新生成全部0；只有一次CPU权重加载。没有修改正式模型/评分源码，没有commit或push，没有启动后续实验。
