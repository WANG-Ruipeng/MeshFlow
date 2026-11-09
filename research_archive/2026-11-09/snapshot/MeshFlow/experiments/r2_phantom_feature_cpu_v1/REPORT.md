# R2_PHANTOM_FEATURE_CPU_V1

2026-10-20；状态：COMPLETE。范围：读取冻结缓存的 CPU 描述量检查，不改变模型、网格或原评分器。

**结论：支持研究“生成部分如何使用 C”，但本次结果不支持把附件的两个几何描述量直接当作好坏判据或训练目标。** 它们没有精准分开现成好坏例；同一 C、同一模型的反例中，采用较合理的结果反而有更大的平行邻层面积、较短的 seam 距离。附件所称“共同根因”“100% 根治”尚未得到验证。

## 1. 实际执行范围

| 主面板 | 请求 | 冻结模型 | 分组 |
|---|---:|---|---|
| Chair SEAT | 28 | PLAIN_S2000 | TRAIN16 / DEV12 |
| Chair BACK | 16 | 同一 PLAIN_S2000，live global15000 | FAMILIAR8 / DEV8 |
| Bench 多部位 | 48 | CLEAN1500 的 NEW_HEAD | TRAIN16 / DEV32 |

92 请求各读取 RAW、BASE_POST、FINAL_POST，共276个缓存网格。另读取用户图1的 SEAM_S21000、图3的 SEAM_S15000 对应请求各3阶段，共 **282个阶段结果**。这些阶段不是282份独立生成。图2的 NEW_HEAD 已包含在主面板内。

全部282项完成，无缺失输入；零训练、零模型前向/反向、零新生成、零新噪声、零新增表面随机查询。没有导入 torch，没有安装包，没有重新运行 D0，也没有重算旧 CF/FF/I。旧 NPZ、图片、评分和模型均只读。所有新文件位于本目录。

## 2. 两项描述量的精确定义

版本：`finite_parallel_prism_union_and_finite_seam_distance_v1`。采用原请求的固定尺度 `L=9.208138264587333`，不根据本轮输出重新归一化。

**Phantom_Layer_Ratio，报告中称“平行邻层面积比”：**

1. 每张非退化 C 三角面建立有限投影范围的双向法向柱体，`|h|≤0.06L`。
2. bulk 面与对应 C 面满足 `|n_bulk·n_C|>cos(30°)` 才进入计算。
3. 将实际 bulk 三角形与柱体裁剪，在该 bulk 面平面内对所有命中 C 柱体的区域求并集，保留实际三维面积；同一 bulk 面不因命中多张 C 而重复累计。
4. 分子是各 bulk 面命中面积之和，分母是保留 C 的三角面积之和。不同 bulk 层仍分别累计，因此可以超过100%；分母不是去重后的 C 表面并集。
5. 主列包含全部 bulk，另保留排除“与 C 共享索引顶点”的 `no_shared_C_vertex_layer_ratio`，对应附件的共点豁免条件。还记录质心落入时计整面的近似版本，便于发现质心近似的偏差。下文关键反例在共享顶点豁免版本中数值不变。
6. SEAM 和后加 BRIDGE 不计入 bulk 分子。这里没有检测“同功能”“非法相交”或“真实薄壁”。

**Seam_Extension：**

追踪原始 seam 槽唯一自由角点，通过 raw-slot 身份及 `source_corner_ids` / `base_source_corner_ids` 恢复最终输出中的第三点，计算其到 **有限 C 三角面集合** 的最短距离，再除以 L。保存每点数值及均值、中位数、最小/最大值、保留/删除数量。

这是无符号距离，故没有命名成“正确外展程度”：距离大可能是合理侧壁，也可能是长刺或错位；距离小可能是合理薄边，也可能是贴面。闭合 C、无激活 seam、全部删除应分别记 NA，不能记作零距离。本轮27个闭合 C 阶段结果均正确为 `NA_CLOSED_C`。

## 3. 最有信息量的是同一 C 的反例

Bench NEW_HEAD，`1dd740…__SEAT_SURFACE_1`，相同模型、C、N，仅噪声不同：

| 阶段/请求 | 既有采用判断 | 平行邻层面积 / C面积 | seam 第三点均距 |
|---|---|---:|---:|
| RAW n0 | PLAUSIBLE | 253.85% | 1.561%L |
| RAW n1 | REPLACED_OR_IGNORED | 37.74% | 4.419%L |
| FINAL n0 | PLAUSIBLE | **199.67%** | **1.832%L** |
| FINAL n1 | REPLACED_OR_IGNORED | **4.86%** | **3.905%L** |

已实际查看这对原图：n0 中 C 的座面用途较合理，生成的是有平行表面的箱体式结构；n1 中另有灰色座面，C 的采用较差。这里的“较合理”只指采用，不认证箱体所有面都合法、接续完整或整体质量完美。

**若按“邻层面积越低、seam 距离越大越好”优化，这一对会被排反。** RAW 已有同方向反例，不是最终后处理才把分数反转。两例的共享 C 顶点豁免面积比与主列相同；质心近似也未修复排序。

第二个同 C 正负对照是 `122a480…__LEG_SUPPORT_1`：FINAL 的采用较合理 n0 面积比52.90%，被替代 n1为0；两者均闭 C，seam 距离为 NA。

全表只有 **2个独立条件** 具有明确正负采用标签对照；各自3阶段构成6条记录，面积比方向全部与附件预期相反。这足以否定“现成面板100%精准区分”的承诺，但不能反向推出“面积越高越好”。

主面板 Bench FINAL 中，29个 PLAUSIBLE 的面积比均值79.20%，6个 REPLACED_OR_IGNORED 为30.40%；13个 UNCERTAIN 单列。跨条件几何复杂度、面积和部位不同，这些均值仅作描述，不是因果比较或新指标排名。

## 4. 用户三张图逐一对照

| 用户图及精确请求 | 阶段 | 平行邻层面积比 | seam 均距 | 解释 |
|---|---|---:|---:|---|
| 图1：SEAM_S21000，`1dd740…SEAT_SURFACE_1/n1` | BASE → FINAL | 10.89% → 11.27% | 4.384%L → 4.384%L | 该旧模型确有替代座面现象；不是 NEW_HEAD 的数值 |
| 图2：NEW_HEAD，`2d1e5…LEG_SUPPORT_1/n0`，side=2.62175 | RAW → FINAL | **0 → 0** | **13.396%L → 13.038%L** | 直接漏过用户指出的坏例；均距是15个 seam 的均值，不是截图中一条橙线 |
| 图3：SEAM_S15000，`459f90…LEG_SUPPORT_1/n0`，front≈0.18782 | RAW → BASE | 58.43% → 43.68% | **NA_CLOSED_C** | C 已含座板及弯曲支撑，K102；真实开放边为0、seam 为0 |

图2在数据中的条件类型是 LEG_SUPPORT，不能把截图中蓝色水平截线直接当成一块孤立平座面。该例旧 AI_PROXY 为 UNCERTAIN，用户认为是坏例；保留两者区别，不将用户判断悄悄改写成旧标签。

图3能检出部分平行邻层，但它没有开放边。因此附件“开放边集中在缺腿侧，接口重心自然偏向右边”的具体解释在该例不成立。不能把 seam 指标 NA 解释为“seam 缩在边界不展开”。输入标签名称也不能代替实际几何内容。

以上身份通过原 gallery manifest、固定截面位置和原图核对。实际查看的是重点例及其噪声配对，不声称重新人工审阅全部92请求。用户可继续直接看图，无需先接受这两项新指标。

## 5. 对三个方向的判断

### 5.1 平移增强：三个方向中优先保留的独立候选

它能检验“减少固定坐标捷径是否有助于采用”，不需要先证明每个坏例都是同一个病因。但随机平移增强不保证严格平移等变，也不能保证改善 Chair 和 Bench。

附件正文幅度±0.15L、结尾表格±0.08L不一致，正式实验应先锁定一个幅度。还需明确只平移终点 `(C,Y)`、保持标准噪声，还是同时平移起点；两者改变的条件路径不同。前者在时刻 t 的 free 平移为 `tΔ`，不能冒称整条轨迹只是同一刚体平移。

最近旧路线是 CREF：曾有分布于5条件的7份 NO→YES，但有 I/FF 与 DEV 接口代价，并且参考坐标和起点同时变化。这里真正的新检验应是 **现有 SEAM 上单独改变平移增强，保持初始化、预算、其余目标和评测不变**；不要再同时叠加 couple、柱体损失、噪声重定位和占用头变化。

成功观察应是同一面板上视觉采用/完整性改善且接续、FF/I等代价可控；本次 CPU 检查没有验证该训练效果，也没有启动训练。

### 5.2 法向柱体排斥：当前定义不宜直接进入 loss 或 guidance

本次反例说明，“有限柱体内近乎平行”还不能识别功能重复。合法底面、厚度或箱体与多余层会混在一起；单靠共点豁免也未解决关键反例。垂直面本身也不因此自动合法。

此外，附件势能的高度项 `(H-|h|)^2` 在柱体内部的负梯度主要增加法向间隔，不会自动把面沿切向送往缺腿处。面积、法向也可通过缩面、旋转来降低势能；落出有限投影范围也可能降低得分而未补上缺失结构。若对 `Yhat=X+(1-t)vθ` 求完整 `∇X`，还涉及生成器 Jacobian，不能称零反向的廉价推理。H、角度、权重和时间窗均是需锁定的超参数。

这不否定所有有限表面约束，只是否定当前“检测已精准，因此立即用于统一训练”的前提。

### 5.3 条件起点：有可检验增量，但需要修正定义

`LOCAL_NOISE` 已试过 seam-tip 局部起点；相对等预算 LONG 有 FF −1.281pp、I −0.040pp，同时有逐例接续和视觉代价。本次新增的是外向偏移、σ=0.35及 bulk 中心偏移，不能记成从未研究的整条路线。

若 `Z, ε` 都取均值 μ 的高斯，直接用原 HYBRID：

`sqrt(0.75) Z + 0.5 ε`

其均值是 `1.3660254 μ`，不是 μ。要保持训练与推理同一均值，可写成：

`Z_H = μ + σ (sqrt(0.75) ξ + 0.5 η)`，`ξ,η~N(0,I)`。

开放边为空、占用全关以及多分离部件均需定义 fallback。尤其本次图3就是闭 C，无法用开放边重心推导缺腿侧。边界中心不等于缺失结构中心；外向初始位置也不等于物理初始动量。

全负条件补回占用头训练是单独的数据覆盖/校准问题，可独立核查，但不能自动把宏观双层问题归因于占用头。已有 COUPLE 也应保留原证据边界，不能作为未经试验的新“侧边引力”叠入。

### 5.4 附件的因果叙述仍需纠正

目前没有早期轨迹证据证明 `t<0.4` 被世界坐标先验锁死。Hungarian OT 是匹配，不会把 t=0 的噪声搬到来源腿的位置；t=0.2也不自动等于已定位。标准 Gaussian 不是全空间均匀分布。仅由图3不能确定唯一正确完成只剩一条腿、其余 free 槽均属多余。C 条件生成允许不同合理完成。

## 6. 验证、账本与资源

- 12项解析几何检查通过，包括多层、重三角化、有限边界、部分相交、法向反转和刚体/尺度不变性。
- 独立核查13项通过：25组多边形并集对独立容斥算法最大差 `1.63e-11`；抽查15份数组、109个 seam tip 的角点身份及有限距离，距离最大差 `2.08e-17`；全282份面积汇总/上界、闭 C NA 和删面计数通过。
- 复用225条逐阶段旧图审；57条缺失保留 NA：Chair SEAT RAW28、FINAL28和旧 SEAM_S21000 RAW1。没有跨阶段复制采用标签。Chair SEAT BASE 是27YES/1UNCERTAIN，BACK每阶段14YES/2UNCERTAIN，均缺乏明确 NO，不能据此认证坏例检测率。
- 初始 RTX5080 总16303MiB、已用2696MiB、空闲13284MiB、compute 0%。进程名/逐进程显存在 WDDM 查询中不可用，未伪造；未停止用户进程。
- 主机24逻辑CPU；WSL可用RAM约65.86GB，初始负载约0.04。先执行一个真实缓存单元，观测 RSS约40.56MB，再用4个单线程CPU worker。主扫描墙钟 **5.44秒**；这是计算时间，不含编写、资产定位、图审和独立核查。RSS是采样读数，不是精确阶段峰值。
- 工程记录：首次 WSL/主机资源查询受沙盒限制，改为获准的只读执行；一次shell引号错误在Python启动前失败；发现shapely未安装后采用本地NumPy裁剪，无安装。无GPU尝试。
- 独立审计发现辅助 `source_C_span_over_L` 最初误存模型单位。现保留 `source_C_span_model` 并修正归一化字段，282个新cell的前后hash保存在 `span_metadata_correction.json`；逆迁移重建全部旧hash，两项主要描述量未改。
- 独立参考检查初次使用FP32得到约 `4.74e-9` 的距离差，改为FP64独立参考后通过，生产描述量未改。该尝试写入审计JSON。
- 冻结输入及最终源码/产物hash见 `completion.json`、`artifact_hashes.json`；原输入在结束时逐一核对。

## 7. 文件与恢复入口

- `per_output.csv`：282行全部阶段结果。
- `per_bulk_face.csv` / `per_seam_tip.csv`：逐bulk面/逐seam第三点，保留输出身份。
- `per_output_with_review.csv`：原图审、完整原记录、来源行号及NA。
- `same_C_noise_pairs.csv`：同C噪声配对，不将三阶段作为独立重复。
- `key_examples.csv`：上述重点例及配对。
- `manifest.json`、`cells/`：冻结输入hash与逐cell结果。
- `run.py`：CPU几何实现；`compare_labels.py`：旧标签关联；`finalize.py`：只核对输入及产物完整性。

只检查已有完成状态：

```powershell
wsl -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/experiments/r2_phantom_feature_cpu_v1/run.py --stage status
```

核验冻结输入和产物，刷新本目录核验账本：

```powershell
wsl -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/experiments/r2_phantom_feature_cpu_v1/finalize.py
```

未完成时才使用 `run.py --stage run --workers 4`，先检查资源；它复用已完成cell、仅计算未完成项。不得重新 prepare 覆盖冻结manifest。已完成的本轮不需要再运行。

本轮在此收束：保留实际几何描述量与反例，不晋级成评价主指标，不启动附件末尾1500步训练、不修改默认模型、不清权重、不commit/push。

## 8. 相关证据

- [原请求附件](C:/Users/warpwang/.codex/attachments/c3d0efdd-e863-42e2-93a6-d1d638fafa15/Pasted%20text.txt)
- [研究路线记忆](../../r2/docs/research_route_memory.md)
- [LOCAL_NOISE等四方向报告](../r2_seam_quality_four_v1/REPORT.md)
- [Bench CLEAN报告](../r2_bench_clean_multipart_seam_v1/REPORT.md)
- [同C较合理采用 n0 图](../r2_bench_clean_multipart_seam_v1/gallery/BENCH_CLEAN_FIXED48_MAIN_AUX/MAIN__1dd7409c95d59d0da192483aa282f8e5__SEAT_SURFACE_1__n0.png)
- [同C被替代 n1 图](../r2_bench_clean_multipart_seam_v1/gallery/BENCH_CLEAN_FIXED48_MAIN_AUX/MAIN__1dd7409c95d59d0da192483aa282f8e5__SEAT_SURFACE_1__n1.png)
- [用户图2的完整阶段页](../r2_bench_clean_multipart_seam_v1/gallery/BENCH_CLEAN_FIXED48_MAIN_AUX/MAIN__2d1e5eeb255e79dcab53bf7fa22f3636__LEG_SUPPORT_1__n0.png)
- [用户图3的历史模型阶段页](../r2_bench_clean_multipart_seam_v1/gallery/BENCH_CLEAN_FIXED48_MAIN_AUX/AUX__459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n0.png)
