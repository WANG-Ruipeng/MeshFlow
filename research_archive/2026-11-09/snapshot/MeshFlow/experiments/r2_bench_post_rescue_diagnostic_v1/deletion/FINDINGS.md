# Bench GEO_C 删面归因（纯 CPU 缓存）

核对 SEAM_S15000 与 SEAM_S21000 各 48 请求，共 96 输出、18,688 个生成槽位。来源映射齐全，RAW、D0 提议、恢复锚点后的 PRE_GEO、聚类后未删面的全部坐标均已按原映射逐位核验；没有复跑 GEO 或模型。

固定半径 R=0.04197095609838 模型单位。GEO 不按“小面积”直接删面；它先按固定代表点聚类，再删完全重复顶点、严格共线或完全重复三角面。

| 检查点 | free 槽位 | 删除 | 聚类后退化 | 重复面 | 被删 PRE 面积占比 |
|---|---:|---:|---:|---:|---:|
| SEAM_S15000 | 8652 | 4349 (50.27%) | 3747 | 602 | 13.58% |
| SEAM_S21000 | 8652 | 3652 (42.21%) | 3016 | 636 | 12.90% |

全部被删面在 PRE_GEO 都是非零面积；未发现 PRE 已退化或与最终重复伙伴已经精确重合的例子。删除发生在这次合并之后，但不意味着 RAW/D0 没有预先产生接近退化或接近重复的几何。

## 最短边与真实半径

| 检查点 | 被删面 min edge≤R | ≤2R | RAW已≤R | D0首次≤R | 恢复首次≤R | 三阶段始终>R |
|---|---:|---:|---:|---:|---:|---:|
| SEAM_S15000 | 3795/4349 (87.26%) | 3932/4349 (90.41%) | 1079 | 2716 | 1 | 553 |
| SEAM_S21000 | 3073/3652 (84.15%) | 3255/3652 (89.13%) | 755 | 2318 | 0 | 579 |

后四列按首次达到≤R划为互斥历史类别，允许之后回升；15000有1面在RAW≤R、PRE又>R，所以实际PRE>R为554面，历史“始终>R”为553面。各退化删除面均满足 PRE 最短边≤2R；两个端点可以分别移动到同一代表点，因此最短边>R仍会合并。重复面删除没有这个面内短边必要条件。

## RAW → D0 / PRE 的收缩

| 检查点 | D0/RAW 最短边中位数 | PRE/RAW 最短边中位数 | D0/RAW 面积中位数 | 被删群体 RAW→D0→PRE 面积和 |
|---|---:|---:|---:|---|
| SEAM_S15000 | 0.1139 | 0.1141 | 0.0680 | 487.280 → 258.205 → 258.376 |
| SEAM_S21000 | 0.1259 | 0.1269 | 0.0831 | 473.067 → 250.406 → 250.548 |

这支持“D0 改变局部尺度，GEO 再触发精确坍缩”的阶段性机制，不能据此认证 D0 删除了正确木条；D0也可能在压缩 RAW 中的冗余/噪声面。不同阶段面积和不是可见表面覆盖率。

## 最强支持与反例

- SEAM_S15000：1594 个被删面满足 RAW 最短边≥2R、D0后≤R；但也有 651 个 PRE 最短边≤R的面和 1108 个最小高<R的面被保留。小尺度不是充分删除条件。
- SEAM_S21000：1436 个被删面满足 RAW 最短边≥2R、D0后≤R；但也有 632 个 PRE 最短边≤R的面和 1119 个最小高<R的面被保留。小尺度不是充分删除条件。
- RAW_wide_D0_short_GEO_collapse：SEAM_S15000 / f495bd67af8d9f0cadd1cf65e8dc4de__COMBO_PART_ASSEMBLY__n1 / slot 72，min edge RAW/D0/PRE=0.247233/0.024027/0.024027，最终 exact_repeated_vertex。完整局部图数据见 `examples_min_edge/RAW_wide_D0_short_GEO_collapse__SEAM_S15000__slot72/geometry.npz`。
- RAW_short_GEO_collapse：SEAM_S15000 / 1dd7409c95d59d0da192483aa282f8e5__SEAT_SURFACE_1__n1 / slot 68，min edge RAW/D0/PRE=0.041503/0.002027/0.002027，最终 exact_repeated_vertex。完整局部图数据见 `examples_min_edge/RAW_short_GEO_collapse__SEAM_S15000__slot68/geometry.npz`。
- RAW_wide_D0_short_GEO_collapse：SEAM_S21000 / 3db1d3904f058b574a9ca2c84bced0dd__COMBO_PART_ASSEMBLY__n1 / slot 97，min edge RAW/D0/PRE=0.207164/0.038188/0.038188，最终 exact_repeated_vertex。完整局部图数据见 `examples_min_edge/RAW_wide_D0_short_GEO_collapse__SEAM_S21000__slot97/geometry.npz`。
- RAW_short_GEO_collapse：SEAM_S21000 / f5d370e9fe5b7452dbbc9440457e303e__BACK_SURFACE_1__n1 / slot 118，min edge RAW/D0/PRE=0.040116/0.001774/0.001774，最终 exact_repeated_vertex。完整局部图数据见 `examples_min_edge/RAW_short_GEO_collapse__SEAM_S21000__slot118/geometry.npz`。

其余 12 份例图数据覆盖原已细片、最大 PRE 面积退化、重复面及保留细面的反例；均未凭几何猜“正常木条”。最小高=2×面积/最长边，是三角形几何量，不是物体部件的物理宽度。

## 交付与执行边界

`per_face.csv` 保留每个槽位四阶段边长/最小高/面积、删除理由、代表点与顶点映射；`per_output.csv`、`per_output_stage_area.csv`、`attribution_summary.csv` 和 `minimum_edge_area_summary.json` 提供数目/面积/比例完整分解。

两 CPU worker，每个 1 BLAS 线程；首个真实案例 0.070 秒、进程峰值 RSS 37,504 KiB。原审计在最后 JSON 序列化 numpy.int64 时退出；完整 CSV/例图及所有身份核对已完成。不完整 JSON 已封存，仅从 CSV 收口并复核 hash，没有新增几何处理。见 `completion.json`。

没有模型前向、训练、评分、半径扫描或旧文件改写；此结果不判断最终视觉缺口是否可接受，也不单独认证误删。
