# CAPPED500 离线深挖复核

先读 `CAPPED500_DEEP_DIVE.md`。所有表保留完整8父、每父两噪声；本包不含权重、raw或完整旧run。原始数据仍在本地 `native_t1/runs/chair_control_feedback_auto_v1`，只读。`source_scoring/` 是原实际源码副本，仅供审核，并非新的评分实现。

本地CPU执行环境为WSL的 `/home/warpwang/miniforge3/envs/meshflow/bin/python`，使用 `-B`，`PYTHONPATH=/mnt/e/MeshFlow`，OMP/OpenBLAS/MKL各2线程。没有GPU dispatch、模型导入/前向、训练、重抽噪声或原评分器重跑。唯一新增几何量是冻结GT查询到原条件C的距离，用于描述性空间分桶。

分析代码按依赖顺序为：

1. `distribution/analyze_distribution.py`，其后 `augment_distribution.py` 和 `mse_tail_followup.py`：读取既有评分与注册信息，分析逐父、两噪声、LOPO、尾部及500→1000变化。
2. `geometry/analyze_saved_geometry.py`，其后 `summarize_geometry.py`：读取逐点距离、见证mask与原raw；核对原数字，输出完整逐点表及空间贡献。
3. `dynamics/analyze_dynamics.py`：读取4000步训练日志、123次实际hooks与检查点侧车信息；完整源hash验证通过时支持CACHE_PASS恢复。

三阶段已有实际运行回执，见各目录的 `completion.json` / `receipt.json`。`resource_dispatch.json` 记录分配依据。重复执行之前应阅读脚本的固定输入/输出路径；外部审阅者无需运行这些代码即可查看所有表。

总收尾入口 `finalize_review.py` 实际执行：重新验证所有已登记来源hash，复制本次已查看的4张原图原字节和6个小源码，核对CSV结构/关键行数，生成来源清单并打包，然后读取ZIP逐成员比对hash。它不运行上述分析、模型或评分器，可安全恢复收尾。源文件不匹配或ZIP达到10,000,000字节时会报错停止。

`code.diff` 为空，因为本轮没有改动维护中的科学代码。新增脚本仅位于本次独立结果目录。`PROVENANCE.json` 为新文件hash索引；`VERIFICATION.json` 是核验与零调用账本；ZIP外侧 `delivery/PACKAGE_RECEIPT.json` 提供包大小/hash，避免包hash自包含。

未知项目保留NA：900步没有既有DEV生成，日志未记录实际参数step范数，没有独立训练复现/新噪声检验/人类视觉评审。旧gate、START默认和旧报告均不改变。

为保证小于10MB，ZIP只省略一份与完整CSV/summary JSON重复的 distribution/distribution_results.json；该文件仍保留在本地并列入hash索引。所有CSV行、4张原图及实际评分源码均完整收入包内。
