# 恢复入口

WSL Python：`/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`；工作目录：`/mnt/e/MeshFlow`。

总合同见`registration.json`，监督子臂见`registrations/*.json`，RL见独立注册。只有`dispatch`中与当前注册hash匹配的资源GO才能派发；恢复前先读当前status、calls和checkpoint收据，不能仅根据权重文件存在判断完成。

已完成臂再次调用只验证并返回，不重新训练。普通安全中断必须处于完整更新、与账本完全一致的checkpoint：

```text
python -m r2.pilots.bench_three_mechanisms_v1.training train --arm LOOP --resume
python -m r2.pilots.bench_three_mechanisms_v1.training train --arm PATCH --resume
python -m r2.pilots.bench_three_mechanisms_v1.training train --arm PLAIN --resume
```

历史失败状态不适用上述普通恢复。工程01的`--engineering-resume`仅针对注册中明确冻结的首个PLAIN失败状态hash，执行一次后不可再次使用。完整原失败收据不覆盖；原attempt保留。

监督训练保留50步滚动恢复点、固定200终点；普通对照另保留267点。发生崩溃而checkpoint落后于调用账本时禁止回退重算。完成后必须核对每臂实际更新、前向/反向、资源峰值、原资产hash，以及完整评分和真实图审。

评测代码保留单独入口；所有新增RAW完成后才允许D0。局部RL用同一预登记随机过程的ZERO作对照，PATCH共享身份恢复另记账。最终报告只依据实际完成产物，未执行项保持BLOCKED/NOT_RUN。

## 已完成实验的只读核验

本轮完成后不要重新运行训练或采样。以下总核验入口只读取检查点收据、文件hash、调用账本、252评分行、36图页和252个实际图审单元，不调用模型：

```text
python -B experiments/r2_bench_three_mechanisms_v1/control/final_contract.py
```

详细阶段依赖、CPU评分和图审入口见 [EVALUATION_AND_REVIEW_RUNBOOK.md](control/EVALUATION_AND_REVIEW_RUNBOOK.md)。正式结果见 [REPORT.md](REPORT.md)，推理导出与完整恢复状态见 [WEIGHTS.json](WEIGHTS.json)。原失败尝试和修正版本均保留，不能使用旧v1注册绕过v2身份检查。

收口流程依次为实际图审汇总、只读总核验、新时间戳结果快照、报告、源代码归档：

```text
python -B experiments/r2_bench_three_mechanisms_v1/control/summarize_visual.py
python -B experiments/r2_bench_three_mechanisms_v1/control/final_contract.py
python -B experiments/r2_bench_three_mechanisms_v1/control/finalize.py final
python -B experiments/r2_bench_three_mechanisms_v1/control/apply_visual_amendment.py
python -B experiments/r2_bench_three_mechanisms_v1/control/write_report.py
python -B experiments/r2_bench_three_mechanisms_v1/control/archive_source.py
```

图审只能使用实际view_image观察填写；汇总脚本不会补造缺失标签。完整报告、图审和源代码归档采用内容校验，已完成文件不静默覆盖。`finalize.py final`只新增时间戳数值快照，不产生训练、评分或图审。

原图审v1和正式评分全部保持；一次实际复看将LOOP f495/n1的BASE/FINAL支撑记为UNCERTAIN，独立修订保存于`visual_review/amendments/`，有效保守汇总为`summary_v2/`。复看未调用模型或新渲染，不能覆盖原v1来隐藏评阅分歧。
