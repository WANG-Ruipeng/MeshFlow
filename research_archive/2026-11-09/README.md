# MeshFlow 研究文字归档

归档日期：2026-11-09。用途：本地资产清理后，仍能理解已经尝试的研究路线、实验条件、实现和结论，继续设计与实现下一轮研究。

用户选择保留研究文字，并明确排除逐面几何诊断、逐步机器账本和其他过细记录。本目录不是训练恢复包，也不是所有磁盘文本的逐文件备份。

## 从这里开始

1. [研究交接：下一次从哪里继续](RESUME_RESEARCH.md)
2. [2026-10-28 研究综述、当前配方及证据边界](snapshot/MeshFlow/maintenance/research_synthesis_20261028/RESEARCH_SYNTHESIS.md)
3. [完整工作记录，含 10 月 29—30 日后续结果](snapshot/MeshFlow/WORKSPACE.md)
4. [历史研究路线记忆](snapshot/MeshFlow/r2/docs/research_route_memory.md)与[可搜索路线登记](snapshot/MeshFlow/r2/docs/research_route_registry.json)
5. [全部收录 Markdown 文档索引](DOCUMENT_INDEX.csv)。其中有历史草稿、代码快照和封存副本；行数不等于独立实验数量。

## 保存了什么

- 研究报告、方案、运行说明、负结果收口、门禁阻塞及有限正结果的原始文字。
- 当前实现及能找到的历史源码快照、配置、协议、条件/种子清单、关键逐输出/逐条件/逐父结果表。
- 三个研究目录的材料：`MeshFlow`、`MeshFlow-Control`、`MeshFlow-Control-artifacts`，以及旧 ZIP/TAR 中符合上述范围的文字成员。
- 本地尚未提交的研究代码及文档也包含在快照中。它们不冒称属于原代码提交；原提交为 `7af33f892b8fb6194fbc54748b6010cf54507169`。

原文件按字节保留，不重算指标、不合并评分版本、不改写历史结论。`snapshot/` 保留来源目录结构；`archived_text/` 保存从旧封存包取出的文字，[封存包映射](ARCHIVE_CONTAINERS.csv)记录来源。

## 明确没有保存

权重、优化器/RNG二进制状态、数据数组、网格、图片、视频、缓存和环境目录；逐面/逐顶点诊断、逐步调用账本、中间状态、细碎日志，以及不符合研究交接筛选规则的结构化文件。结构化记录通常限定单文件不超过 4 MiB；详细规则与统计见 [SCOPE.json](SCOPE.json) 和 [归档脚本](archive_tools/build_archive.py)。

[未备份权重列表](WEIGHTS_NOT_BACKED_UP.csv)仅记录扫描时存在的文件身份与大小，**不是云端权重备份**。历史报告中的退役权重和本地路径原样保留，不表示这些资产仍可取得。旧文档指向未收录资产的链接可能无法打开；可用 [文件来源与 SHA256 清单](MANIFEST.jsonl)查找已收录文字。

## 核验与以后使用

从 GitHub 下载/克隆本目录后，使用 Python 标准库执行：

```sh
python verify_archive.py
```

它只验证文件数量、长度和 SHA256，不运行研究代码、不调用模型。先读研究交接与原报告，再选择需要参考的实现。不要直接运行旧 `resume`/`controller` 脚本；它们可能依赖本次没有备份、或此前已退役的资产。

本目录只增加研究文字档案，不改变主仓库模型默认、训练行为或历史评分。
