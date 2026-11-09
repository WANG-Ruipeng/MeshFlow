# 恢复与结果入口

主命令（在 WSL Ubuntu /mnt/e/MeshFlow，使用原 venv）：

```text
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_early_fm_v1.resume
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_early_fm_v1.resume --execute
```

不带参数只看状态。`--execute` 只推进被冻结的一个1500步训练臂与16请求评价；检测到现有训练进程时不重复启动。
完整安全游标允许原状态恢复；部分更新、OOM、数值失败、未完成rollout不自动重算。
已完成阶段按身份与hash复用；全部RAW完成后才调用单次官方Bench D0。评分3个CPU进程、渲染2个CPU进程，数值线程各1。
采样先启动8请求shard0，首个真实请求后检查资源，满足余量和计算利用率条件才启动另外8请求shard1；不加任何探针。

训练：`training/EARLY_FM/status.json`、`training_log.jsonl`、`exposure.jsonl`、`calls.jsonl`。
输入：`registration.json`、`frozen_time_weights.csv`；旧资产只引用、校验，不覆盖。
结果：`analysis/per_output_all_metrics.csv`、`analysis/per_request_paired.csv`、`analysis/per_parent_metrics.csv`。
图像：`gallery/index.html`，同位置截面；`visual_review_notes.json`由实际查看后填写，不能自动从指标生成采用标签。

完成训练与数值阶段不等于图审完成。真实AI_PROXY图审、报告和归档完成后写`COMPLETE.json`。
终态恢复只读验收，不再次训练、生成、评分，不改默认模型或commit/push。
