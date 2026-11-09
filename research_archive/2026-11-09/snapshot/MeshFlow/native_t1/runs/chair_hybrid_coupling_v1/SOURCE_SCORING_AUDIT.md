# 本轮评分源码与定义核验

新实现位于 `experiments/mf_h1_local0/chair_hybrid_coupling_v1/scoring.py` 和 `reporting.py`。公共模型、公共 metrics、旧评分脚本和旧结果均未修改。

- `control_mvp48/eval_scoring.py:score_free` 仍负责原双向 RMS、固定 coverage、Gamma 表面/有限边距离、CF/FF 全对扫描、退化与小角面。旧 Chair protocol 中 `source_scorer_identity` 的 8 个文件与当前实际文件 SHA256 全部一致。
- `eval_scoring.py:prepare_reference` 的 coverage_points 使用历史 seed9973，pairwise_target_points 使用 seed9911 的第二次表面采样。当前不调用 prepare_reference，不重抽 GT 点，直接读取冻结 NPZ 与 JSON 身份。
- 主终点 `fixed_query_MSE = mean((d(Qcoverage,F)/L)^2)`，无量纲、L 归一距离平方。它与上轮独立归因的同名字段相同，不是原 R12 RMS 的平方；原 R12 点集不同。A alpha=1 时 L=L0；新增 L0 版单列。
- `chair_domain_control_v1/width_analytics.py` 的旧 `matching = ... alpha==alpha` 按行取 wrong_mean_minus_diagonal。该历史字段不改。新 E[alpha,beta] 明确行=input、列=target，条件优势固定 beta 列。
- 宽度严格沿用等权角点 q95−q05。b_self、b_GT 的绝对误差先逐曲线算，再父内两噪声平均，再父等权。零/过小分母保留原分母并标 UNDEFINED。
- No-response、Post-scale 都只读本检查点自身 F1。Post-scale 使用 C1 与 C_alpha 的对应 X 拟合比例，不读 free GT 估计编辑。解析参照拥有旧条件与旧输出，不能冒称与单次 current-C-only 生成器同输入合同。
- 主终点父内任一噪声缺失则父格 BLOCKED；分母固定8。保留全部请求以及原 NOT_APPLICABLE 格，不补位、不静默删失败。C 面板仅作已观察留出回归。
- START 原分数只有在输入/GT/query/原评分代码身份一致、实际 raw 数组完全相同时复用。诊断回放 NPZ 可以附加状态，压缩文件hash不同不会自动判几何相同，必须校验实际 raw 数组hash。

实际 CPU 验证记录在 `tests/scoring_cpu_tests.json` 与 `tests/scoring_smoke/validation.json`：完美编辑对角误差近零、三个固定目标列严格胜；No-response 三列优势为0且无严格胜；相同行诱饵矩阵按列为0、按行非0。真实旧 raw smoke 使用 A 1例、C 1例、B 1条三alpha曲线，共5输出。完整原 A CPU 重评分1次，37个原指标与旧JSON差值均为0；fixed_query_MSE与独立归因原CSV差值为0。主指标层级配对fixture通过。

验证未加载真实模型，模型前向/反向/优化更新/完整生成均为0，没有GT查询重抽。原有限边 helper 会传递导入 PyTorch，CPU进程不加载模型也不初始化CUDA；不能把“导入了库”写成模型调用或把CPU评分算作生成。

本地未找到任务所列 `Native_Chair_Handoff_20261004` / `latest_independent_audit` 命名文件；已读取实际可用的 `native_t1/analysis/chair_control_followup_v1/ERRATA_MATRIX_DIRECTION.md`、矩阵独立复核源码与 `native_t1/runs/chair_control_attribution_v1/ATTRIBUTION_REPORT.md`、逐例CSV和实际原评分源码。本说明仅说明实际已核验范围。
