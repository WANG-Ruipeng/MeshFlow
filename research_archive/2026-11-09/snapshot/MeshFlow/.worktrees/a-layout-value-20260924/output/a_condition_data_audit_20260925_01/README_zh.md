# 新16父对象条件验证：本地来源审计

当前同类、原train、父对象独立且沿用原几何规则的可用新对象数为 **0/16，缺口16**。本轮没有创建新的网格/主场，没有运行模型、优化、MC或新几何资格筛选，也未更改已有数据。

## 已找到的原始来源

Windows原始归档均位于 `C:\Users\warpwang\Downloads\`；WSL对应 `/mnt/c/Users/warpwang/Downloads/`：

| 归档 | 实际NPZ父对象文件 | 与本任务关系 |
|---|---:|---|
| shapenet-rebuttal2.tar.gz | 1252（含三个类别） | loudspeaker / 03691459 有417 UID：原train376、官方test41 |
| shapenet-rebuttal.tar.gz | 1451（其他三个类别） | 不能默认换类别凑新16 |
| shapenet-cls.tar.gz | 9477（其他四个类别） | 不能默认换类别凑新16 |
| shapenet.tar.gz | 17372（聚合包） | 包含相同417 speaker UID；其speaker train/test集合与rebuttal2完全相同 |

每包实际SHA256、UID列表、split元数据及成员路径均在 `AUDIT.json`。split的对象数组只解析pickle字符串指令，未执行pickle。

原 `shapenet-rebuttal2.tar.gz` 的SHA256与 `assets/parents.json` 完全匹配；现有16 raw成员文件SHA也逐项匹配。旧盘点仅覆盖此前明确指定的5个目录，这次扩展到Downloads中的原始tar，未改写旧盘点。

聚合包并不是rebuttal2的字节副本：417个speaker NPZ文件均不同，faces数组均不相同，vertices数组有157份相同。它们是相同父UID的不同网格版本/表示，不能算新父对象；旧筛查结论也不能无条件移植到聚合包的网格版本。初步身份比较确实读取了全部417的数值数组，包括41个test成员，仅用于逐数组相同比较；没有对test执行资格筛选、模型/优化/MC，也不会将其纳入新池。正式审计脚本的数值几何哈希只记录376 train。

## 历史接触必须分层记账

- **376个原train UID均已做过几何筛查**：旧 `REAL16_RECOVERY.json` 记录316个拓扑拒绝、43个径向单交拒绝、17个通过径向与exact检查。
- 17个几何通过者中，1个只有20面，被既定 `20<F<800` 规则排除；其余16正是当前开发池。不能放宽面数规则把该对象补入，也仍不够新16。
- **模型/teacher/后续开发池是这16个**。当前工作区顶层RUN_PROTOCOL的真实speaker UID并集没有超出它们；3k模型的参数拟合只用固定train12，另外4个虽非此次微调样本，但早已用于开发诊断。
- 360个“现有16之外”的train UID并非此前未触达对象：已经进入旧几何筛查，且在匹配原归档版本的旧规则下不合格。
- 41个官方test只列身份与排除集合，不默认为可调用确认集。

因此，严格的新父对象池为0。聚合包中未带类别凭据的其他UID也不能自行称为新增音箱。

## 原A预训练来源核对

当前 `experiments/a_radial/assets/original_A_10.pt` 的SHA256为：

`7af5a04d38f6dc85023e1f9c06d385eb1e6f559fadc776676e94b5120eab2a3d`

它与本地 `A_RESEARCH_STATE_20260923_RESTART.zip` 内两份 `original_A_10.pt` 逐文件哈希一致，且历史 `A_SOURCE_VERIFIED.json` / `A_SOURCE_EXTRACT.json` 记录相同哈希。原domain-transport交接文档声明：29,907参数DomainVelocity基于192个程序化平滑/特征径向对象（ID0..191），128面离线教师、3,000次更新训练；文档明确其当时不涉及真实资产。

**证据强度限于训练文档声明＋精确权重文件来源链。** 当前可读历史ZIP没有完整192样本/teacher清单与训练数据可供逐项重验，不能把这次审计写成完整预训练样本审计，也不能仅排除16就宣称任意未来对象从未参加任何训练。

## 保持一致的主场构建

原代码入口：交接包 `tools/recover16.py`（源集合、拓扑、10244方向单交、exact）；工作树 `experiments/a_radial/prepare_data.py`（精确512细分、2050支持方向、logr）；`mesh_ops.py`、`exporter.py` 与 `assets/support.npz`。实际文件SHA在审计中绑定。

保持bbox中心和bbox对角线归一化；10244方向必须恰好一个正交点；2050节点主场为 `log(r / sqrt(mean(r*r)))` float32；参考为同主场4096面曲面。保持封闭、定向、单连通亏格零与exact检查，不补洞、不取多交最近点、不悄悄放宽条件。512构建仅共边中点细分、不移动旧顶点，所以源F大于512并不能用当前细分器降到512；需要预先明确适用性，不允许为凑数加入未授权简化。

## 后续建议顺序（尚未执行）

本节选择规则服从 `output/a_condition16_plan_20260925_01/STUDY_PLAN.json`，不另立随机排序或改变已声明的研究范围。

1. 在同意的来源中保留类别、父UID、原split和源SHA，先冻结来源清单与模型hash；按计划排除既往真实训练、teacher和开发父对象及其源/几何重复。对376个曾经几何筛查的UID全部排除，仅是当前保守待定的范围，不把纯几何筛查等同于参数训练或teacher/模型开发使用。其他类别或官方test需要另行改变研究范围，不能默认替换。
2. 在保持原几何资格规则的前提下，父对象排序唯一采用 `SHA256('a-condition16-parent-v1:' + uid)` 升序，取前16个合格者，完整保留资格失败；不读取A/Fixed或优化器结果来选对象。
3. 不足16就停止数据准备并明确缺口，等待来源/范围决定，不补洞、换类别或看效果后追加挑选。错误条件配对也服从同一计划：按 `SHA256('a-condition16-donor-v1:' + uid)` 排序并循环映射到下一UID；在任何新对象模型推理前冻结数据manifest及配对映射。

本次为已知相关工作区、Downloads原始归档和WSL项目/数据目录的审计，不是全磁盘法证搜索；没有把未核验路径的资产宣称为可用。只读身份审计不能代替新对象的几何资格检查。
