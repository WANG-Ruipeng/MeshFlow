# PoNQ Gate 1 实验适配

这个独立模块用于交付包复现、真实对象补测与参数隔离；不改官方 MeshFlow 和 PoNQ vendor，不包含生成训练。

2026-09-26 已完成的结果在工作树 `output/ponq_gate1_followup_20260926_01/REPORT_zh.md`，完整账本 92 行。源数据、冻结协议、逐次参数、输出网格、失败、环境、审计和哈希清单保存在各研究输出。

数值分类注意：原 `run_gate1.py` 只按所有候选 score 是否有限分类；`run_isolation.py` 还检查非负。候选 score 不等于汇总后的 graph capacity，所以不能把两种原 status 当统一成功率。保留原始输出，查看综合表和独立图容量诊断。已发现 speaker2 两份 clean 存在 arccos 浮点超界，实际图含 NaN；本轮没有修复 vendor。

## 文件

- `code/execute_study.py`：新建输出、复制代码、运行、精确审核、汇总与哈希。
- `code/prepare_real_data.py`：先冻结 UID 次序，筛选原始真实几何，保留拒绝见证。
- `code/run_selected.py`：使用已审核且哈希绑定的四个参考，不重复归一化。
- `code/parameter_isolation.py`、`run_isolation.py`：保留中心 g,c 的重编码与固定位置/匹配位移对照。
- `code/finalize_audit.py`：从协议和账本计算统计，动态生成非流形见证，不把历史特例当必需结果。
- `code/summarize_followup.py`：只读取研究结果，生成综合账本和科学图。
- `vendor/`：固定版本的官方提取器、必要 helper 与许可证；身份见 `SOURCE_PROVENANCE.json`。

## 本机复现

在 WSL 中使用以下 Python（共享原 meshflow 环境的 Torch，另有少量本地依赖）：

```bash
PY=/mnt/e/MeshFlow/.worktrees/ponq-gate1-20260926/output/ponq_gate1_env_20260926/bin/python
CODE=/mnt/e/MeshFlow/.worktrees/ponq-gate1-20260926/experiments/qem_set/code
OUT=/mnt/e/MeshFlow/.worktrees/ponq-gate1-20260926/output

$PY -B -m unittest discover -s "$CODE" -p 'test_*.py' -v
$PY -B "$CODE/execute_study.py" synthetic --out "$OUT/reproduction_synthetic_new"
$PY -B "$CODE/execute_study.py" real --out "$OUT/reproduction_real_new" --selection "$OUT/ponq_gate1_real_data_20260926_01/SELECTION.json"
$PY -B "$CODE/execute_study.py" isolation --out "$OUT/reproduction_isolation_new" --baseline "$OUT/reproduction_synthetic_new"
```

输出目录必须不存在；selection/baseline 使用绝对路径。复现不会自动晋级训练。具体版本以每次 `ENVIRONMENT.json` 为准，当前环境与原包不同，不能预期跨环境字节一致。

`summarize_followup.py` 是这轮日期命名输出的归档汇总器，`--parent` 指向上述工作树 output，`--out` 指向新目录。每个原始研究自带冻结的代码副本与哈希清单；后续修改模块不会更改已完成研究的源码快照。
