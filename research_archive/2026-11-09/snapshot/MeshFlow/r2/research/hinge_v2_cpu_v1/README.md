# HINGE V2：有限位置合并与三边环补面

**保留为可选的纯 CPU 后处理研究模块。** 它接在已完成的 `HINGE_OPP015` 之后，复用同一份生成结果，不训练、不调用生成器或 D0。完整输入链为：

```text
PLAIN_S2000 RAW
  → 单次官方 D0 → GEO_C → 来源 C 角序恢复与 free 定向
  → HINGE_OPP015（全部新增共边反向绕序守卫）
  → ALPHA（有限位置合并）→ HOLE（精确三边环补面）
```

本归档是在原实验完成后，经用户明确授权整理并发布。原报告中“未 commit/push”描述的是当时的实验边界；报告保留原字节。本次未改变正式 CLI 默认、模型或旧评分定义。前置模块见 [HINGE_OPP015 归档](../hinge_opposite_v1/README.md)。

## 做了什么

- **ALPHA**：沿用 HINGE 已共享一个实际索引顶点的开放边候选和固定排序，先尝试原中点规则，再尝试有限位置 `α∈{0,.25,.5,.75,1}`，FP32 目标去重。整个原 GEO 簇一起移动；含 C 的簇固定到 C。每个原角点相对 PRE_GEO 的累计位移上限仍为 `1.5%L`。
- **HOLE**：只给实际索引已围成的三开放边环增加一张三角形。三个邻面必须都是 free，且至少一张是 SEAM；要求统一反向绕序、最小角≥5°、q≤10，并通过非法 CF/FF、未决接触、非流形和旧合法边保护。它不是根据近邻猜洞，也不填 C 的开放边界。
- **ALPHA_HOLE**：先合并，再对保存的结果补面；一次固定队列扫描，没有扩大半径、额外迭代、法向偏移、删原面、拆边或重网格化。

新增面明确标记 `BRIDGE`，来源父面/RAW 槽 ID 为 `-1`，不冒充原生成面。合法图可以经桥到达原 BULK，但桥面积不计入原 BULK 分母。代码中保留全部新增双面共边的反向绕序守卫；旧的同向边和非流形缺陷不会因此自动消失。

## 固定面板上的结果

28 个请求：4 个 TRAIN 父×4噪声、3 个 DEV 父×4噪声。所有方法使用同一 RAW、D0、C、评分查询和截面位置。下表是原 `HINGE_OPP015` 与最终组合的比较；比例为逐请求宏均值，边/面数量为全组总和。

| 指标 | HINGE_OPP015 | ALPHA_HOLE |
|---|---:|---:|
| 固定原 free 面分母 FF | 13.49022% | 13.04628% |
| 实际输出面分母 FF | 13.49022% | 12.99550% |
| 固定原 free 面分母 CF | 2.78779% | 2.78779% |
| I：固定 C 查询冲突覆盖 | 1.68457% | 1.69678% |
| 非法 FF 面对数 | 555 | 540 |
| 合法共边数 | 4,729 | 4,840 |
| 其中合法 C/F 共边 | 315 | 315 |
| 合法图单面孤岛 | 221 | 198 |
| 未过滤实际索引图单面孤岛 | 121 | 102 |
| 原 BULK 可达面积宏均值 | 35.15260% | 37.75007% |
| 新桥面 | 0 | 16 |
| 原面新增删除 | 0 | 0 |

**真实 FF 改善来自 ALPHA：** 消除15对旧非法 FF，新增非法 CF/FF 为0。HOLE 只增加合法面；单独补面的实际 FF 下降来自分母变化，固定原面分母 FF 不变。因此同时报告两种分母，不把桥面算成消除自交。

组合保留全部4,729条原合法共边，新增111条，其中48条属于桥面与邻面。不是新增111条 C/F 接口。新非流形边、同向双面边、未决接触、原面删除均为0；这不表示旧网格已经无交、水密或全局流形。

| 分组 | 原 BULK 可达面积：基线→组合 | ≥30% 请求：基线→组合 | 固定原面 FF：基线→组合 |
|---|---:|---:|---:|
| TRAIN，16份 | 61.51706%→64.26007% | 14/16→14/16 | 12.47663%→12.47663% |
| DEV，12份 | 0%→2.40341% | 0/12→0/12 | 14.84167%→13.80582% |

`608/n2` 通过新桥接入38张原 BULK 面，达到28.820738%可达面积；`2d/n2` 新接入1张、0.020161%。这是局部进展，未达到预定30%门槛，不能写成 DEV 普遍成功。最终组合的 TRAIN 新可达原面为 `73/n0 +3、73/n1 +1、73/n2 +19、85/n3 +1`；原报告的 `73/n2 +18` 是 ALPHA 阶段叙述，最终组合逐例账本在此明确。

I 全组增加0.012207个百分点，实际上涨在 `608/n1` 和 `608/n3`；三个请求的平均 q 也略有上升。零新增精确非法面对与零 I 代价不是同一回事。28份来自长期观察的7父；结论是开发面板上的局部后处理收益，不是独立新父/新噪声泛化结论，也不认证新增桥符合所有用户的设计意图。

## 完整证据与截面图

- [原报告](evidence/REPORT.md)、[全部179字段完整比较](evidence/comparison/FULL_COMPARISON.md)、[179字段分组CSV](evidence/comparison/all_179_metrics.csv)。保留全部字段，没有只挑有利指标。
- [逐输出完整表](evidence/scores/per_output.csv)：140行；基线及四处理臂。原135字段保留，新增44个固定原面分母与BRIDGE字段。四臂包括中点重扫对照，避免把二次扫描收益全归给多位置选择。
- [逐输出配对](evidence/paired_per_output.csv)、[逐父配对](evidence/paired_per_parent.csv)、[所有代价记录](evidence/all_regressions.csv)、[分组汇总](evidence/metric_summary.csv)。
- [28份同位置外观与前/侧截面图](evidence/gallery/index.html)、[TRAIN实际图审](evidence/gallery/review_train.csv)、[DEV实际图审](evidence/review_dev.csv)。全28张已实际查看，记录为非盲 `AI_PROXY`，不冒称人类评分。
- [112格独立几何审计](evidence/independent_audit.json)、[完整面对扫描对账](evidence/independent_score_pair_audit.json)、[140份合法图及3,080项关键指标核验](evidence/independent_key_metrics_audit.json)。

局部解释图来自完整面板，不能替代全组：

![3f/n0三边环缺口补面](evidence/local_geometry/HOLE_3f741759_n0_rank000.png)

![608/n2新增桥面](evidence/local_geometry/ALPHA_HOLE_608b46b0_n2_rank000.png)

`单面孤岛`有两种口径：旧 HINGE 的未过滤实际索引图为121→102；排除非法面对后的合法图为221→198。两者均保存，不混用分母或名称。A-VMR/PMR各既定阈值不变；I使用固定的2,048个C查询。无唯一生成GT，本归档没有来源款式重建排名。

## 可迁移的 CPU 数组入口

`source/repository/` 与 `source/legacy/` 是实际原实现和评分源码快照；[源码清单](source_manifest.json)与[证据清单](evidence_manifest.json)保存路径、大小、SHA-256。`replay_array.py`只连接这些冻结函数与显式资产，不改科学算法。

需要 Python、NumPy、SciPy、Trimesh；仅渲染需要 Matplotlib、Pillow。没有自动安装、下载或模型调用。先验证归档：

```bash
python -B verify_archive.py --check-engine-import
```

从一份已完成 `HINGE_OPP015` 的数组继续：

```bash
python -B -I replay_array.py \
  --mode alpha-hole \
  --post /data/hinge_opp015/post.npz \
  --pairs /data/hinge_opp015/pairs.json \
  --baseline-receipt /data/hinge_opp015/complete.json \
  --pre-geo /data/original_post/pre_geo.npz \
  --raw /data/raw/raw.npz \
  --output /data/new_empty_result
```

其他模式为 `midpoint`、`alpha`、`hole`。可加 `--reference` 和 `--reference-pairs` 对照既有正式结果。输出目录必须不存在；缺失、身份不符或已修改的资产返回明确失败，不自动恢复模型、补造映射或重抽噪声。

本版锁定原模型坐标与 `L=9.208138264587333`、位移档 `0.015L`。输入需含真实GEO角点簇、来源面/角点、SEAM锚点及原面对缓存，且由原完成回执绑定；不是任意PLY的通用修复器。没有这些元数据时不能仅凭坐标相近猜测身份。完整面板评分与历史总入口保留原本地资产路径合同，需额外提供未分发的网格与查询缓存。

本次已从 `/tmp`、Python `-I` 隔离运行 `608/n2` 和 `73/n0` 两份固定资产：输出的全部数组键、形状、dtype和字节均与正式结果一致，完整面对缓存也一致；加载的科研模块均来自归档，未导入torch。见[迁移验证与环境](validation/publication_checks.json)、[608/n2回执](validation/608b46b0_n2_replay_receipt.json)、[73/n0回执](validation/73b369b9_n0_replay_receipt.json)。这只检查代码迁移，不增加实验面板。

原几何矩阵64.23秒、完整评分30.50秒是原4-worker阶段墙钟，不是包含开发/图审的总任务时间，也不是新的速度基准。发布迁移检查另记CPU峰值与资源账本。

## 归档与恢复边界

保留原报告、评分器、源码和输入；大型权重、优化器、RAW/POST NPZ、噪声和稠密缓存不进入Git。本地完整资产仍在 `experiments/r2_hinge_v2_cpu_v1/`。逐格候选、拒绝、面形状和合法边账本以压缩证据保存，具体条目见证据清单。

`build_archive.py`按白名单从本地完成结果重建快照，`verify_archive.py`独立核验文件与压缩包条目；完成后的历史 `resume.py`仅校验原账本并退出，不追加科学尝试。发布不代表自动采用为默认方法，不启动训练或扩大几何搜索。
