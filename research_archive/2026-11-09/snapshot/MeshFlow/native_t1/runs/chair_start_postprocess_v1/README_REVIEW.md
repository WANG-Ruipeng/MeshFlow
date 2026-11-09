# START 后处理复核包

先读 START_POSTPROCESS_REPORT.md；浏览全部 32 例请打开 OPEN_GALLERY.html。

本包包含 64 行原始/处理后评分、32 行逐输出配对、逐父与逐噪声表、全部 64 份 PLY、32 份处理后 NPZ/来源映射、32 份 RAW 终点数组、8 组 C/GT 条件和冻结查询、32 对照图与32辅助图、实际评分依赖源码、code.diff、哈希、执行账本和独立审计。RAW 是既有生成的终点副本，没有重新采样或替代源数据。包内不含模型权重或完整生成轨迹。

分析是已经观察过的 8 父 × 4 噪声上的描述性比较。C 原样保留；学习式去噪 NOT_RUN_ASSET_UNAVAILABLE。数值收益不能等同于人工可用性或水密证明。

为满足 <10 MB，完整逐点评分向量和无损 PNG/渲染缓存留在原本地 run；省略清单及其哈希见 OMITTED_LOCAL_ARTIFACTS.json。它们没有被删除。逐例 CSV 和实际评分代码完整包含。

运行路径记录保留实际绝对路径，source_snapshot 是原样源码快照而非便携安装包。原工作区的可恢复总入口为：
`python -B -m experiments.mf_h1_local0.chair_start_postprocess_v1.run all`
完整性复核：`...run verify`；只刷新交付：`...deliver`。
上述入口只在既有本地条件/RAW/冻结评分资产存在时使用，不下载权重、不训练、不重生成；已完成的科学阶段复用其缓存。

PACKAGE_CONTENTS.json 列出其自身以外所有 ZIP 项的字节数和 SHA256；可先验证各项再查看。外部 DELIVERY_RECEIPT.json 记录压缩包哈希和大小。原模型、原数据、原 RAW、原评分器/报告和正式默认 profile 均未修改，没有提交或推送 Git。
