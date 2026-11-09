# A 瓶类冻结模型验证已完成（2026-09-25）

**14对象×三路×单次，共42次全部完成。** 原225筛选14合格，使用全部14；不追加训练。

| 指标 | Fixed | 正确3k | 错配3k |
|---|---:|---:|---:|
| 4秒主指标 | 12/14 | 11/14 | 13/14 |
| 16步终点 | 12/14 | 14/14 | 14/14 |
| 完整耗时跨对象中位数 | 5.654秒 | 5.839秒 | 5.828秒 |

终点覆盖有提升，但4秒主指标和正确条件的额外价值均未获支持。单次计时不证明稳定加速。42次运行异常0、资源前后超限0；独立核验PASS。官方长训保持暂停。

[完整14行账本与结论](../.worktrees/a-layout-value-20260924/output/a_bottle_condition14_once_20260925_01/RESULT_zh.md) · [质量余量](../.worktrees/a-layout-value-20260924/output/a_bottle_condition14_once_20260925_01/QUALITY_MARGINS_zh.md) · [独立核验](../.worktrees/a-layout-value-20260924/output/a_bottle_condition14_once_20260925_01/FINAL_VERIFICATION.json) · [225行筛选账本](../.worktrees/a-layout-value-20260924/output/a_bottle_study_20260925_01/SCREEN_LEDGER225.md)
