# HINGE：新增共边绕序守卫的独立复测

**保留 `PLAIN_S2000 → 单次 D0 → GEO_C/来源定向 → HINGE_OPP015` 作为研究组合，2.5%L 留作对照。** 这是有限但明确的局部后处理正结果；未修改维护 CLI 默认。权重、优化器和原始生成资产不进入此归档。

28 个固定请求（16 TRAIN、12 DEV），两个位移门限独立从原 POST 开始，共 56 次 CPU 处理与原评分；零训练、零模型前向、零重新 D0、零新噪声。主档相对原 POST：FF 14.3104%→13.4902%，CF 2.9116%→2.7878%，实际索引图单面孤岛 150→121；原 4,624 条合法共边全部保留，新增 105 条均反向，零新增非法 CF/FF 面对、删面或非流形边。

**限制同样保留：** DEV 的 bulk 可达≥30% 仍为 0/12；`610/n1` 的 I 增加 0.048828 个百分点，少数请求的平均 q、角度 p05、可达面积比例有微小代价。零新增非法面对不等于零新增冲突覆盖，更不等于视觉全无损伤。28 个已反复检查的开发请求不是独立新父泛化集。完整结论见 [报告](evidence/REPORT.md)。

## 唯一科学改动

旧 HINGE 只预先要求被提名的 hinge 边反向，整簇合并可能附带产生其他双面共边。本版在提交前检查**此次全部新产生的 incidence=2 边**，若任一同向就回滚，覆盖附带 CF/FF 边及几何非法的边。身份以来源面和来源角点追踪，避免顶点代表编号改变导致假新增。

此守卫不翻面、不删面、不重新聚类，保留原有 209 条同向边。其余候选排序、形状/法线守卫、非法面对与旧合法边保护、PRE_GEO 累计位移门限保持原定义。旧 1.5%L 的 111 条新增合法边减少为 105 条，是公开记录的守卫成本。

## 完整证据

- [逐输出全表](evidence/combined_per_output.csv)：原 POST、旧015/025、新015/025，共140行，全部124个原数值字段＋11个版本化拓扑字段及状态/身份。
- [逐输出配对](evidence/paired_per_output.csv)、[逐父配对](evidence/paired_per_parent.csv)、[全部不利变化](evidence/all_regressions.csv)、[完整汇总](evidence/metric_summary.csv)。没有只选改善字段。
- [连接账本](evidence/connection_ledger.csv)、[拒绝漏斗](evidence/rejection_funnel.csv)、[逐格身份账本ZIP](evidence/per_cell_ledgers.zip)。ZIP含56格所有候选/拒绝、旧边去向、新边、非法面对进出、逐面形状和图身份的原字节文件；其[内容清单](evidence/per_cell_ledger_manifest.json)逐文件列 hash。
- [28张同位置截面图集](evidence/gallery/index.html)。[实际 AI_PROXY 记录](evidence/gallery/ai_proxy_review.csv)仅7个预定病例；其余21张只生成，未冒称已看。非盲、非人类评分，截面不能认证三维拓扑。
- [独立审计](evidence/independent_audit.json)、[独立统计验证](evidence/independent_statistics_validation.json)、[协议](evidence/protocol.json)、执行/资源回执与 [code.diff](evidence/code.diff)。原1,172个保护资产未变。
- [135字段原定义索引](evidence/metric_catalog_original84.csv)保留旧审计原字节，其中84行coverage指**上一轮**，本轮140行覆盖见 [analysis完成回执](evidence/analysis_complete.json)。这不是35-family指标库全部已实现的声明。仓库任务政策见 `native_t1/docs/control_metrics.md`，未把未跟踪的政策文件混入此次发布。

FF/CF是受非法面对影响的free面比例，不是非法面对数除所有面对组合。I是固定C表面查询的冲突覆盖。实际索引图与假设精确坐标合并图分开；合法图不自动等于方向一致图。无唯一生成GT、无指定必接接口、无外露任务时，相应指标保持NA。所有HINGE方法继承同一RAW，不作为新增独立采样。

## 实际源码与可迁移 CPU 数组入口

`source/repository/` 和 `source/legacy/` 保存原实现字节与目录层级，[source_manifest.json](source_manifest.json)记录62份源文件及hash。包括真正的新增引擎、评分数学、拓扑、图像、独立审计/分析代码；没有只提供调用本地缺失包的外壳。历史 `score_recipe_factorial_conflict.py` 仅用于记录评分来源hash，其未执行的传递依赖不在闭包内。

环境使用现有 Python、NumPy、SciPy、Trimesh；仅画图还需 Matplotlib、Pillow。此处没有自动安装、下载或模型调用。只核验源码/证据不需要实验网格：

```bash
python -B verify_archive.py --check-engine-import
```

对一份**显式提供且身份匹配**的既有数组运行CPU解码：

```bash
python -B replay_array.py \
  --post /path/to/post.npz \
  --pre-geo /path/to/pre_geo.npz \
  --raw /path/to/raw.npz \
  --pairs /path/to/original_POST/pairs.json \
  --delta 0.015 \
  --output /path/to/new_empty_result \
  --reference /optional/path/to/formal_post.npz
```

`--reference`可省略；输出目录必须不存在。入口只设置归档模块搜索路径与原适配器的数学源码目录，不修改或替换科学函数。缺输入返回`BLOCKED`，不猜测映射、不重抽噪声或生成替代资产。完整面板的原`run.py`仍保留本地冻结路径合同，异地不能无资产直接跑完56格；图集渲染/完整评分同样需要未分发的大型冻结资产和查询缓存。

已实际从`/tmp`隔离运行上述归档入口，在`608b46b0/n0@1.5%L`与正式结果核验：**63个数组的键、形状、dtype、原始字节全部相同，NPZ SHA-256也完全一致**。加载的31个科学/运行模块均来自归档，torch未加载，模型/GPU调用为0。见 [portable decode回执](validation/portable_decode_receipt.json)。这是一次固定资产的可迁移代码闭包检验，不是跨硬件性能/所有输入的普遍证明。

## 归档边界

所有原数据、raw、报告和评分器只读，源码快照未改写；[evidence_manifest.json](evidence_manifest.json)给出证据hash。大型NPZ、噪声、训练数据、权重及稠密逐查询pair缓存不进入Git，原本地资产仍保留。`retained_profile.json`只登记既有权重sidecar身份，本轮未重新加载或重hash大权重。

`build_archive.py`用于从原本地文件重建白名单证据，遇到缺失源文件或内容不同的已有目标会拒绝。`replay_array.py`和`verify_archive.py`是独立归档入口；它们不自动改默认、不启动实验扩展、不执行Git。
