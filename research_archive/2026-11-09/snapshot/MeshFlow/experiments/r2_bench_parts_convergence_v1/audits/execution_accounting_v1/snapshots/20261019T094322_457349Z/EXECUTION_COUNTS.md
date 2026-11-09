# Bench 执行计算账本（PRELIMINARY）

该表由已有账本与完成收据读取；没有运行模型、重评分或改变训练。
快照窗口：`2026-10-19T09:43:09.486444+00:00` 至 `2026-10-19T09:43:22.457318+00:00`。
各文件顺序取快照，训练继续时不是同一瞬间的原子快照。

## 训练与固定探针

| 臂 | 已完成唯一更新 | 最后 UPDATE_COMPLETE | 生成器训练前向返回 | 固定探针前向返回 | 完整探针次数 | 小头训练前向返回 | 小头探针前向 |
|---|---:|---:|---:|---:|---:|---:|---:|
| COMMON | 12000 | 12000 | 96000 | 1248 | 13 | 0 | 0 |
| PLAIN | 7152 | 19152 | 57224 | 768 | 8 | 0 | 0 |
| SEAM | 7101 | 19101 | 56809 | 768 | 8 | 50730 | 224 |

## 全过程采样与后处理（不只计算最终模型）

| 标签 | RAW 完成 | POST 完成 | 采样前向返回 | D0 返回 | 布局头返回 |
|---|---:|---:|---:|---:|---:|
| NATIVE_S10000 | 8 | 8 | 400 | 8 | 0 |
| NATIVE_S12000 | 48 | 48 | 2400 | 48 | 0 |
| NATIVE_S13000 | 8 | 8 | 400 | 8 | 0 |
| NATIVE_S15000 | 48 | 48 | 2400 | 48 | 0 |
| NATIVE_S17000 | 8 | 8 | 400 | 8 | 0 |
| NATIVE_S19000 | 8 | 8 | 400 | 8 | 0 |
| NATIVE_S2000 | 8 | 8 | 400 | 8 | 0 |
| NATIVE_S6000 | 8 | 8 | 400 | 8 | 0 |
| SEAM_S13000 | 8 | 8 | 400 | 8 | 5 |
| SEAM_S15000 | 48 | 48 | 2400 | 48 | 20 |
| SEAM_S17000 | 8 | 8 | 400 | 8 | 5 |
| SEAM_S19000 | 8 | 8 | 400 | 8 | 5 |

## 总数与恢复开销

```json
{
  "unique_generator_optimizer_updates": 26253,
  "canonical_generator_optimizer_returns": 26253,
  "canonical_generator_training_forward_returns": 210033,
  "canonical_generator_training_backward_returns": 210031,
  "canonical_generator_training_backward_starts": 210033,
  "canonical_head_optimizer_returns": 7088,
  "canonical_head_training_forward_returns": 50730,
  "canonical_head_training_backward_returns": 50730,
  "fixed_probe_generator_forward_returns": 2784,
  "fixed_probe_generator_forward_starts": 2784,
  "fixed_probe_completed_invocations": 29,
  "fixed_probe_completed_head_forwards": 224,
  "RAW_complete": 216,
  "POST_complete": 216,
  "sampling_generator_forward_starts": 10800,
  "sampling_generator_forward_returns": 10800,
  "D0_forward_starts": 216,
  "D0_forward_returns": 216,
  "inference_layout_head_forward_returns": 35,
  "inference_layout_unresolved": 0,
  "GEO_complete_calls": 216,
  "physical_generator_optimizer_returns_including_discarded": 26578,
  "physical_generator_training_forward_returns_including_discarded": 212637,
  "physical_generator_training_backward_returns_including_discarded": 212634,
  "physical_generator_training_backward_starts_including_discarded": 212637,
  "recovery_backward_unknown": 1,
  "all_generator_forward_returns_training_probe_sampling_including_discarded": 226221,
  "all_known_head_forward_returns": 50989
}
```

三次中断的额外已完成更新为 77 + 112 + 136 = 325；额外训练前向返回 2604、反向返回 2603，另有 1 次反向结果未知。这些是物理计算开销，不增加科学训练步数。原始恢复核验完整嵌入 JSON。
固定探针小头仅计入已完成探针记录的实际 head_forwards；若某探针未结束，其内部未单独记账的小头调用仍标 UNKNOWN。闭合 C 的布局头调用为 0，不按请求数量反推调用。

停止判定：`{"status": "CONTINUE_BOTH", "current_global_step": 19000, "stage_updates": 7000, "stop_training": false, "next_common_target": 20000}`。
一致性问题：`[]`。
尚未完成项与逐收据来源哈希见 JSON。本文件不能替代最终报告或代表模型已晋级。
