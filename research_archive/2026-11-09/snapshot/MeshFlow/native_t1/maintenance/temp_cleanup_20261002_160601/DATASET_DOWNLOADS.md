# 下一阶段数据盘点与下载清单（2026-10-02）

现在可以直接用本地椅子库准备多父对象实验，不需要先下载更多椅子。当前瓶颈是 Native 数据入口仍固定两个父对象、112 面，以及正式划分和条件任务尚未冻结。

## 已有数据

|本地资产|实际对象规模|使用判断|
|---|---:|---|
|`experiments/mf_h1_local0/assets/shapenet-03001627`|3,135 UID；3,125 组精确无序三角几何|多对象椅子主实验可用；官方 train 2,822 / test 313|
|`output/t1_publish_20260930/chair_n112.npz`|2 个父对象的已准备条件/目标|保留为开发回归，不当作新对象测试|
|`downloaded_data/ss_overfit`|128 文件实际为同一头牛、1 组几何|仅调试；test 与 train 重复|
|旧 worktree 的 closed16 数据|16 个原始父对象及其细分版本|细分不增加父对象数；其 train=test，不能直接作为独立测试|

椅子库全量 CPU 扫描：28–800 面，中位数 470；恰好 112 面的只有 154 件。按现有官方 loader 的严格 `20 < N < 800` 条件，保留 3,119 件（train 2,809 / test 310）。这些是 loader 计数，不是最终质量筛选结果。

发现 9 组精确重复几何、10 份多余副本；它们均未跨官方 train/test。尚未检查近重复和同源家族，不能将这个结果写成“无数据泄漏”。扫描记录了每个对象及重复组，见 [dataset_scan.json](dataset_scan.json)。

无非法顶点或面索引；1,686 件含精确零面积面，824 件含原始索引边界边，2,318 件含非流形边。这些检查基于原始索引拓扑，没有焊接、修复、自交检测或水密筛选；应根据各项监督的适用条件标注，不宜用全库水密作为默认入选条件。若需要清理/简化，先生成并固定父网格，再从该网格抽取 C 和自由补集，保持同一坐标变换及面片身份。

官方 train 里的对象即使没有参加 Native 条件微调，也不能宣称“官方预训练未见”。正式报告需区分条件微调未见对象与官方预训练未见对象；UID/原始来源映射、近重复分组和旧开发对象排除仍需建立。

## 建议下载顺序

以下建议依据下一阶段的三角面片条件补全、Geo 消融及语义几何干预任务作出；不是已冻结的正式实验协议。本轮只查询文件元数据，没有下载数据集。

|优先级|数据与官方入口|建议取得的文件 / 下载量|用途与限制|
|---|---|---|---|
|现在直接使用|本地 MeshFlow chair 数据|无需下载|先建立多父对象划分、可变面数输入和 C 采样|
|需要跨类别时|[MeshFlow 官方预处理数据](https://huggingface.co/datasets/qsun2001/meshflow/tree/main/obj_data)|`shapenet-cls.tar.gz`，25.48 MB|历史完整归档清单确认含 bench 560、chair 3,135、lamp 627、table 5,155 份网格 NPZ；每类有 train/test。当前只提取 chair，可只解压缺失类别；这些是文件数，仍需身份与几何去重|
|语义控制任务优先申请|[PartNet 官方下载](https://huggingface.co/datasets/ShapeNet/PartNet-archive)|原始网格/标注包 `data_v0_chunk.z01`–`.z10` 和 `data_v0_chunk.zip`，全部分卷合计约 107.69 GB|座面、靠背、支撑等部件语义，适合设计明确的 C 干预；要原始网格与层次标注，仅下载 `sem_seg_h5.zip` / `ins_seg_h5.zip` 点云任务包不能满足本任务|
|需要原始来源与自行预处理时|[ShapeNetCore 官方下载](https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive)|PartNet 对齐路线优先 `ShapeNetCore.v1.zip`，32.19 GB；一般原始库可选 `ShapeNetCore.v2.zip`，26.79 GB；不必同时下载两版|补原始对象与来源追踪；和当前椅子库可能同源，不能自动视作独立外部测试|
|后续扩展|[MeshFlow Sketchfab 包](https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/sketchfab.tar.gz)|669.21 MB|预处理后的更多资产，先核对来源、类别、重复与预训练覆盖|
|后续扩展|[MeshFlow Objaverse 几何包](https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/objaverse_occ_v5_ids.tar.gz) + [split 包](https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/split.tar.gz)|170.38 MB + 2.19 MB|轻量扩展候选；这是 MeshFlow 发布的预处理包，不是完整 Objaverse 原始库|

PartNet 和 ShapeNetCore 当前官方入口需要登录并申请访问，填写真实姓名、导师/PI 和所属机构。[ShapeNet 官方组织页](https://huggingface.co/ShapeNet) 已说明改由 Hugging Face 分发。

[PartNet 项目页](https://partnet.cs.stanford.edu/) 提供部件标注，并公布 chair/table 等类别到 ShapeNet-v1 的变换关系。这使其适合语义控制任务，但不自带“改变座面宽度后应如何补全”的成对真值；这类变体仍需构造、检验并按原始父对象整组划分。

如果现在只准备一个下载动作，先申请 PartNet；如果先做低成本跨类别数据准备，下载 25.48 MB 的 MeshFlow `shapenet-cls.tar.gz`。正式椅子主实验的数据准备可以同步从现有库开始，无需等下载完成。

## 下载后的放置与接入

建议将原始下载放到 `E:\MeshFlow\downloaded_data\incoming\`，校验后解压到带版本号的独立数据目录；各工作窗口共享只读原始数据，派生网格、划分、条件清单另存，不复制四套数据。

接入前需要固定：来源与父对象 ID、去重组、官方预训练覆盖、train/validation/test、统一父对象归一化、面数范围、C 的抽取方式、自由补集及质量标签。不要单独归一化 C，也不要让同一父对象的细分/简化/编辑变体跨划分。

本地扫描证据：[dataset_scan.json](dataset_scan.json)。MeshFlow 文件字节数与 LFS 校验元数据：[official_data_files.json](official_data_files.json)。官方代码对预处理数据的下载说明：[MeshFlow README](https://github.com/qiisun/MeshFlow#data)。

历史包成员证据：`experiments/mf_h1_local0/outputs/asset_extraction_20260928/extraction_manifest.json`。该包当前官方 LFS SHA256 为 `06707caff4ce465f402e6b69f81056204c579921d371eb2c9e4e9a3929f7aef2`，与历史下载记录一致。`shapenet.tar.gz`（50.28 MB）另有发布，但本轮没有核实其完整类别/对象清单，不能与已核实的 class split 包混称。
