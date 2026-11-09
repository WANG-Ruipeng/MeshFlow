# SEAM A/B/C 固定小试

用户授权的三项独立800步测试，旧四方向实验与旧评分源保持只读。
合同与产物：`experiments/r2_seam_abc_v1/PROTOCOL.md`。

在已有WSL虚拟环境、仓库根运行：

```bash
python -B -m r2.pilots.seam_abc_v1.resume
python -B -m r2.pilots.seam_abc_v1.resource_snapshot before_dispatch_NAME
python -B -m r2.pilots.seam_abc_v1.resume --stage train --arm A_COUPLE
python -B -m r2.pilots.seam_abc_v1.resume --stage train --arm B_POOL47
python -B -m r2.pilots.seam_abc_v1.resume --stage train --arm C_SHARED_TIP
python -B -m r2.pilots.seam_abc_v1.resume --stage generate --arm A_COUPLE
python -B -m r2.pilots.seam_abc_v1.resume --stage postprocess --arm A_COUPLE
python -B -m r2.pilots.seam_abc_v1.resume --stage score
python -B -m r2.pilots.seam_abc_v1.resume --stage summarize
python -B -m r2.pilots.seam_abc_v1.resume --stage render
```

`generate/postprocess`也用于B/C和冻结LONG（LONG仅12个DEV新请求）。默认status不运行模型。每次GPU派发仍须先按AGENTS实测资源；这些命令不是授权同时启动全部worker。训练匹配账本恢复、安全中断与失败禁止隐式重放由每臂runner执行。所有112份逻辑RAW（16旧缓存）完成后才允许D0。

本轮不自动扩步、不清理、不改默认、不commit/push。完成后状态入口只读，旧失败日志与工程修正另存。

最终结果与逐例证据见 `experiments/r2_seam_abc_v1/REPORT.md`。三臂各800新增更新与96份新RAW/POST已执行；C的28个推理请求全部Null，不能把其几何变化归功于实际共享引用。完成后默认入口只显示状态，封存合同拒绝再次训练/生成派发。

数值、图审与核账均是独立的 CPU 入口：

```bash
python -B -m r2.pilots.seam_abc_v1.analysis_tables
python -B -m r2.pilots.seam_abc_v1.training.summarize --output experiments/r2_seam_abc_v1/training_analysis/final
python -B -m r2.pilots.seam_abc_v1.training.plot_curves --summary experiments/r2_seam_abc_v1/training_analysis/final --output experiments/r2_seam_abc_v1/training_analysis/final_figures
python -B -m r2.pilots.seam_abc_v1.review_aggregate --validate-config
python -B -m r2.pilots.seam_abc_v1.final_audit --partial
```

全套产物到齐后实际执行 `final_audit`（不加partial）、`save_evidence` 和 `closeout`，封存报告、实际评分源码/新实现、代码差异与hash。`closeout`再次运行只核验封存文件，不启动任何模型。

图审必须实际打开56页并填写224行，聚合程序不会自动补标签。完整审计只有产物齐全后执行；缺产物的partial返回NOT_READY。各入口的完整参数以`--help`为准。训练与生成缓存只按成功回执复用；失败或中断不能通过删除账本来隐式重放。
