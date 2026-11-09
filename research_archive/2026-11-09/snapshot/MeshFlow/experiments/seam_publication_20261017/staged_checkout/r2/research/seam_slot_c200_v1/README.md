# SEAM_SLOT_C200 历史源码与评分审阅归档

本目录随 SEAM 代码整理入库，保存实际执行过的原始源码身份、评分定义和后续诊断源码。它没有启动训练、模型前向、生成、重新评分或后处理，也没有改变模型默认配置。

当前方法说明与完整结果边界见 [SEAM 文档](../../docs/seam_slot.md)。本目录的 `source_manifest.json` 给出每个来源文件的路径、SHA-256、字节数，以及审阅副本的位置。

## 三种源码身份

- `original_repository_sources`：22 份原文件按原仓库路径发布，包括 `r2/prototypes/seam_slot_c200_v1/` 的核心与检查代码、冻结的训练/生成/后处理 pilot，以及它使用的两个历史辅助文件。保留原字节，不把历史脚本改写成新的方法。
- `scoring_source/`：19 份实际评分入口及其运行路径所需数学代码的原字节副本。`repository/` 保留原仓库相对层级；`legacy/` 保留原 `E:/MeshFlow-Control` 相对层级。评分使用的面对分类、I、CF/FF、实际索引共边和可达图定义都可以审阅。原评分程序与历史结果未改动。
- `related_source/`：占用布局对照和几何保护诊断的历史源码副本。来源占用 `SOURCE_ORACLE` 是诊断对照；容量保护是单独实验。它们没有自动接入可安装的 SEAM 核心，也没有变成默认采样或默认后处理。目录内图审转录、打包脚本仅是历史来源记录，不应重新运行来冒充新图审。

## 原 pilot 的运行边界

原 pilot 是冻结实验的本地恢复代码，不是任意云端路径上的通用训练 CLI。它依赖保留的完整权重、数据/噪声/查询清单、实验目录、原绝对路径和冻结 hash。训练固定四条 TRAIN 条件、200 更新、batch 8 / microbatch 1；生成固定四条件各四噪声。不得仅改标签或根路径就将其解释为新的实验。

历史脚本使用 Linux/WSL 的 `fcntl`、`resource`、`/proc`、`nvidia-smi`；采样的资源记录还需要 `psutil`。`generate.root_path` 只允许原实验目录。旧 `resume.ps1` 写死本机 workspace 与外部 venv，因此没有作为公共入口发布。旧打包、压缩和图审转录脚本也不是本次主要 pilot 发布清单的一部分。

评分审阅副本不是 standalone scorer：原 `user_condition_finetune_v1/evaluation.py` 从 `/mnt/e/MeshFlow-Control/src/meshflow_control/evaluation` 动态导入冻结数学代码，且原入口仍读取冻结本地查询与元数据。复制代码不等于重新验证了可迁移运行入口。副本中的路径、注释和历史输出目录按原样保留，不应直接执行副本里的主函数。

## 历史 DESIGN 与现在的结果

`r2/prototypes/seam_slot_c200_v1/DESIGN.md` 是开训前写成并冻结的设计文本，其中“尚未执行 200 步”等句子反映其当时状态。后续训练和评价已经实际完成；请读新的 SEAM 文档及其指向的证据，不把开训前设计当作最终状态。这里保留该文件原字节，避免改写实验历史。

SEAM 的接续正信号与 FF/I 代价必须同时保留。C/F 共边增加不单独证明生成主体连通；共享底边不能认证 bulk 无交，也不能认证对未见父泛化。来源占用对照不等于仅给 C 的可部署方法。

## 不调用模型的最小检查

在仓库根目录、已有 NumPy 的 Python 环境中：

```bash
python -B -m r2.prototypes.seam_slot_c200_v1.check_corner_contract
```

该检查只做 NumPy 的角点 mask、自由坐标分母、Euler 锚点保持、共享索引与输入拒绝测试，不读取权重、不导入 Torch、不构造模型。`check_inference.py` 会运行小占用头前向/反向，它不属于“零模型调用”的检查命令。

可安装的整理版本及其纯 CPU 检查说明见上述 SEAM 文档。其目录调整不表示重新跑过历史 GPU 实验。
