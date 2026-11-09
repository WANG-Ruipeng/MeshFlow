# 当前 Bench 研究参考：完整座面 Native → SEAM

版本：`BENCH_CURRENT_RESEARCH_REFERENCE_V1`；登记日期：2026-10-23。

用户明确要求保留本轮完整座面路线，作为目前 Bench 的优选研究参考，并继续检验新增结构数据。这里的“当前 SOTA”指用户指定的本地研究主线，不是已经证明在所有 Bench 输入、独立父或全部指标上最优，也不改变维护中的 `chair-hybrid` CLI 默认。

原实验 [R2_BENCH_FULL_SEAT_CURRICULUM_V1](../../experiments/r2_bench_full_seat_curriculum_v1/REPORT.md) 及其原始报告、权重、数据、RAW/POST、评分和图审继续只读保存。本页记录用户后续保留决定，不回写原报告的结论或完成时间。

## 两个必须保留的终点

| 身份 | 用途 | 完整恢复权重 | 推理导出 |
|---|---|---|---|
| `NATIVE_S6000`，累计 global18000 | 完整座面主体组织参照；新增独立结构覆盖实验的保留起点 | [checkpoint_step6000.pt](../../experiments/r2_bench_full_seat_curriculum_v1/training/NATIVE/checkpoint_step6000.pt) | [generator_step6000.pt](../../experiments/r2_bench_full_seat_curriculum_v1/training/NATIVE/generator_step6000.pt) |
| `SEAM_S3000`，累计 global21000 | 当前优选 Bench SEAM 研究参考；冻结比较与回归 | [checkpoint_step3000.pt](../../experiments/r2_bench_full_seat_curriculum_v1/training/SEAM/checkpoint_step3000.pt) | [generator_step3000.pt](../../experiments/r2_bench_full_seat_curriculum_v1/training/SEAM/generator_step3000.pt) |

两段均从保留的 Bench COMMON12000 直系继续，Native新增6000更新，再同池SEAM新增3000更新；不是从官方权重刚开始训练的6000/3000步。完整恢复权重包含匹配优化器与RNG，推理导出不能替代它用于连续恢复。四个 `.pt`、对应 `.json` 身份旁车和固定 C-only 头均登记为保护资产。

机器清单：[retained_assets.json](../maintenance/bench_reference_20261023/retained_assets.json)。本次只核验路径、大小、已有小文件收据及完成状态；大权重哈希复用封口时已保存的身份收据，没有重新加载或重新哈希多GB权重。清单是维护保留合同，不设置文件系统ACL，也不授权清理任何其他历史文件。

## 已验证的配置与证据范围

数据为22个独立训练父、22条经图审的完整座面，K=2–51，N=128–256。原Native/T1+role+Geo、free-only FM、OT后HYBRID0.25、batch8/micro1、lr=1e-5与原AdamW保持；每步4父，各identity与x-width U[0.9,1.1]。SEAM训练用来源owner，部署推理用冻结旧833参数C-only头，没有oracle推理或头更新。

固定评价为当前训练8父×2噪声及既有DEV4父×2噪声，每模型24请求。COMMON、Native、SEAM共72输出；各保留RAW、BASE_POST和FINAL，共216评分与216条非盲AI_PROXY实际图审。处理链固定为Euler50→单次官方Bench D0→C/SEAM保护、GEO_C及来源定向→HINGE_OPP015→ALPHA→HOLE。DEV是反复使用的开发集，不能改名为独立测试；官方预训练是否见过来源父，本轮未证明。

下表按COMMON / Native / SEAM排序，数值沿用原报告，不重新评分。

| 观察 | COMMON12000 | Native6000 | SEAM3000 |
|---|---:|---:|---:|
| TRAIN RAW：C合理采用，AI_PROXY | 2/16 | 14/16 | 15/16 |
| TRAIN FINAL：FF非法影响free面比例↓ | 34.58% | 14.57% | 10.25% |
| TRAIN FINAL：CF非法影响free面比例↓ | 13.23% | 9.33% | 2.09% |
| TRAIN FINAL：I↓ | 3.28% | 1.87% | 3.37% |
| TRAIN FINAL：整体合理，AI_PROXY | 14/16 | 8/16 | 9/16 |
| TRAIN FINAL：主要支撑存在，AI_PROXY | 16/16 | 11/16 | 11/16 |
| DEV FINAL：C合理采用，AI_PROXY | 4/8 | 1/8 | 2/8 |
| DEV FINAL：FF非法影响free面比例↓ | 26.87% | 24.05% | 27.21% |
| DEV FINAL：I↓ | 15.65% | 12.85% | 13.15% |
| DEV FINAL：小角面比例↓ | 30.13% | 58.01% | 55.88% |

SEAM最终24/24输出有合法C/F共边，但bulk可达面积≥30%仅TRAIN2/16、DEV0/8，训练两份来自同一91e169父。硬边槽接住C、进入主体、合理采用C和主体完整是不同证据；Native没有SEAM专属分区，对应字段为NA。I为固定C支持域冲突覆盖，不是整个网格自交率。FF/CF当前分母为实际保留free面；必须连同删面、留存面积和图审解释。

L型座面6b894c、35948e/n1等熟悉条件有真实主体组织收益。7b58a3和9e429b等RAW已有采用，BASE阶段出现支撑损伤，不能把全部问题再归于生成器；同时DEV仍有替代灰座面，不能把全部问题归于后处理。此轮未分开D0与GEO_C的责任，不能仅凭RAW→BASE变化指定单一责任模块。

## 后续使用边界

新增数据实验应冻结独立版本，从保留Native6000开始并回放全部合格旧父；新增父与同父变体分别计数，保持旧DEV隔离。用户认可本路线，不能据此只回放原来的漂亮输出或只报告更好的指标。是否进入SEAM、具体预算、评测和停止条件由下一轮独立协议定义；本登记本身不触发训练。

来源owner可用于合法监督与匹配，推理不得用Y修正布局。旧C-only头当前预测是否正确，与完整C/Y监督数据是否成立，应分开登记；Native扩数据不受旧SEAM头预测的额外门槛限制。

本保留决定未新增loss、修复器、去噪轮数或默认模型选择；没有删权重、commit或push。下一轮即使有增量，也保留这两个终点及其完整证据用于回归。

## 原始依据

- [正式完成与26项产物哈希](../../experiments/r2_bench_full_seat_curriculum_v1/COMPLETE.json)
- [权重身份及封口表收据](../../experiments/r2_bench_full_seat_curriculum_v1/results/report_tables_receipt.json)
- [全部逐输出CSV](../../experiments/r2_bench_full_seat_curriculum_v1/results/per_output.csv)与[完整主表](../../experiments/r2_bench_full_seat_curriculum_v1/results/PRIMARY_TABLES.md)
- [实际图审记录](../../experiments/r2_bench_full_seat_curriculum_v1/visual_review/summary_v1/actual_observations.csv)与[三阶段图库及固定截面](../../experiments/r2_bench_full_seat_curriculum_v1/gallery/index.html)
- [方法与比较边界](../../experiments/r2_bench_full_seat_curriculum_v1/METHODS_AND_LIMITATIONS.md)及[恢复入口](../../experiments/r2_bench_full_seat_curriculum_v1/RUNBOOK.md)
