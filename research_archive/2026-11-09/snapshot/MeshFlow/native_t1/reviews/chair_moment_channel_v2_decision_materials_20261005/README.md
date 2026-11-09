# MOMENT V2 下一步决策材料包

本包按用户指定的三组材料整理已有证据，用于判断下一项训练应优先解决什么。没有新增训练、模型前向、采样生成、几何评分或查询点抽样，不含大权重。原实验文件只读，复制件逐文件校验；新增内容仅为身份映射、格式导出、索引和校验记录。

## 1. 完整父对象与噪声配对

`tables/` 内是原CSV的逐字节复制件，未挑选有利父对象、未缩小分母：

|入口|内容|
|---|---|
|[primary_paired_parents.csv](tables/primary_paired_parents.csv)|完整8父，三组比较，共24行；先平均同父两噪声|
|[primary_noise_pairs.csv](tables/primary_noise_pairs.csv)|8父×2噪声×3组比较，共48行|
|[primary_summary.csv](tables/primary_summary.csv)、[primary_noise_summary.csv](tables/primary_noise_summary.csv)|三组总体及逐噪声汇总|
|[per_output.csv](tables/per_output.csv)、[per_parent.csv](tables/per_parent.csv)|全部A/B/C登记输出和父级结果，包含原NA|
|[paired_per_output.csv](tables/paired_per_output.csv)、[paired_per_parent.csv](tables/paired_per_parent.csv)|全组逐输出、逐父配对；可分table/split/ratio查看|
|[quality_paired_summary.csv](tables/quality_paired_summary.csv)、[coverage_partition.csv](tables/coverage_partition.csv)|双向距离、尾部、接口、相交、角度及覆盖四区贡献|

三组比较为 M−H、H−START、M−START。主面板为A20 validation、alpha=1，8父等权。主指标是固定512个GT free查询到生成自由表面的距离平方均值/L0²，不能替换成双向RMS，也不能把两噪声当16个独立父。

## 2. 宽度、目标排序与面板身份

- [width_per_curve.csv](tables/width_per_curve.csv)、[width_per_output.csv](tables/width_per_output.csv)：所有三检查点、模型/No-response/Post-scale曲线；分开 `r_base`、`b_self`、`b_GT`，绝对斜率误差先逐曲线计算。
- [cross_matrix.csv](tables/cross_matrix.csv)：完整1296个3×3单元，行=input alpha、列=target beta。
- [target_column_audit.csv](tables/target_column_audit.csv)、[target_column_noise_pairs.csv](tables/target_column_noise_pairs.csv)：固定目标列比较，严格胜出不含平局。
- [noise_pairs_per_parent.csv](tables/noise_pairs_per_parent.csv)、[control_curve_counts.csv](tables/control_curve_counts.csv) 和其余 `width_*` / `target_column_*` 表：完整父/噪声对应及配对汇总。
- [panels/README.md](panels/README.md)、[panel_parent_membership.csv](panels/panel_parent_membership.csv)、[A20_B_context_comparison.csv](panels/A20_B_context_comparison.csv)：A20与B的父对象重合及实际known-context身份核对。

**c1b8c2f0c4f3cc43aff7c08b060f5ed6 属于A20 validation，但不在B控制面板。** 不能把此父的A20退化直接解释为B控制问题。B验证的4父是A20验证8父的子集；同一UID也不能代替C身份核验，known-mask、source-face slots及坐标hash分别保留。

`C` 的两种含义已明确区分：known context（已知几何上下文）与 table C（已观察的holdout面板）。`panels/` 同时提供两者的身份与原始登记来源，不将table C视为全新独立测试。

## 3. c1b… 两噪声的空间证据与训练通道日志

目录：`geometry/c1b8c2f0c4f3cc43aff7c08b060f5ed6/`。

|文件|用途|
|---|---|
|`noise0/`、`noise1/` 下的 `START/`、`H_CONT_END/`、`MOMENT_XY_END/`|六个实际输出，每个包含原 `original_raw.npz`、原评分JSON、原逐点向量NPZ和生成身份|
|每个输出的 `free_mesh.obj`、`free_triangles.npy`|直接提取原free槽位，保持面与角顺序；未焊接、修复或变换坐标；NPY保留原FP32三角坐标|
|`shared_C_GT.npz`|两噪声共用的C、完整GT、GT free、known/valid mask、源面ID、L/L0/N/K；已验证两噪声相同|
|`C_known.obj`、`GT_free.obj`、`GT_full.obj`|上述已知/GT几何的三角汤格式导出|
|`original/input_noise0.npz`、`input_noise1.npz`|原输入银行，含原高斯噪声、C、mask和GT；GT用于离线评价，不是模型输入|
|`original/reference.npz`、`reference.json`|原查询点与身份：coverage、pairwise-target、Gamma各自保留，不混用|
|`frozen_queries.csv`|主指标所用512个coverage查询的原位置与query_index|
|`per_query_distances_START_H_M.csv`|两噪声×512点，共1024行；同一行并列三模型的原距离/L、换算世界单位距离及平方距离/L0²|
|`geometry_validation.json`|查询hash、原距离向量与评分JSON一致、MSE闭合、C保持及六份导出身份|

距离CSV读取已保存的 `distances_L`；世界单位列仅乘回原L。本例alpha=1，L=L0。`query_index` 是原coverage_points数组行号，没有最近点重匹配、重采样或重新求距离。本包没有新增最近面编号或部件定位；原评分JSON内已有细节原样保留，不推断未保存的信息。

`training/` 保留两臂各1000更新的原 `*_training_log.csv`、原 `*_events.jsonl`、原 `*_sample_exposure.csv`、summary及共享批次hash。`MOMENT_XY_relation_observations.csv` 是原UPDATE_DONE事件中32000条观察的无损字段展开（1000更新×8微批×4层），可用 `(moment_step, sample_index)` 与MOMENT曝光表关联，32000行关联均已核验。

|日志字段|读取边界|
|---|---|
|`b_RMS_mean`、`b_max`|真实有效关系对的logit修正量级；每更新RMS为32个微批/层RMS的算术平均，不是把所有pair合并后的RMS|
|`u_RMS_mean`、`u_max`|实际成对value增量量级，同样保留原聚合方式；逐观察量在展开CSV与原事件内|
|`content_delta_RMS_mean`、`weighted_pair_value_RMS_mean`|内容差量与聚合成对value的原遥测|
|`new_parameter_delta_l2`|当次optimizer更新前后新增参数差的实际L2范数|
|`new_*grad_norm*`、首/末投影非零梯度数|新增参数梯度可达性、梯度量级及裁剪前后值|
|`old_grad_norm_before_clip/after_clip`|旧参数梯度量级；**不是**旧参数实际更新范数，后者没有逐步保存，应标NA，不能用梯度冒充|

这些是训练输入上的遥测；c1b是验证父，不应把训练b/u直接当成其Euler轨迹上的通道因果贡献。本包提供判断所需现有材料，不预先判定通道“太弱”“作用错误”或主干漂移原因。

## 来源、格式与校验

- [source_copy_manifest.json](source_copy_manifest.json)：主体103份原文件的工作区来源、文件hash、字节数和包内位置；面板来源另见 `panels/sources.csv`。
- [MANIFEST.sha256.json](MANIFEST.sha256.json)：包内全部文件的SHA256（清单自身不递归列入）。ZIP创建后另验证CRC和每个成员hash。
- [preparation_validation.json](preparation_validation.json)：六输出/查询/日志与零模型调用核验。
- `source_code/`：实际评分和b/u遥测定义源码及原diff，仅供阅读，无需导入或运行模型。
- `provenance/`：冻结协议、轻量检查点身份、完成账本、训练选择与原独立审阅；没有 `.pt/.pth/.ckpt` 权重。
- `build_bundle.py` 与 `panels/build_panel_mapping.py`：本次复制、导出、映射与压缩过程的可复核代码。

原CSV中出现的工作区绝对路径是来源记录，其他父的raw没有随本包扩大复制。指定c1b的六输出、同一C/GT、查询和距离已全部自包含。CSV为UTF-8（原表原字节保留，新增表含BOM）；NPZ/NPY可以 `numpy.load(..., allow_pickle=False)` 读取。没有引入任何默认模型、loss或训练设置变更。
