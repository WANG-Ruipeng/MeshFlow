# Control MVP48 独立评价 — Session C

1. **T1是否跨出了双对象沙盒？** T1 在条件微调未参与的父对象上有效输出 32/32；排除 C 后双向表面 RMS 6.95316（8/8 父对象） %L，GT coverage@1%L 0.104187（8/8 父对象）。这些值及固定视图用于判断未知区域补全质量；仅输出有限值或复制 C 不能证明合理补全，本轮未登记绝对成功门槛，也未全面验证几何编辑控制。

2. **Geo是否有初步的独立父对象增量？** Geo−T1：双向表面 RMS（%L）均值差 0.0290159，父对象配对中位差 0.0447798，改善 2/8 父对象；GT coverage均值差 0.00335693，父对象配对中位差 -0.00170898，改善 3/8 父对象。改善数以父对象为单位，个别条件和噪声波动见逐输出表。

3. **Full的接口/质量收益是否值得保留？** Full−Geo：有限边接口 RMS（%L）均值差 -0.00118235，父对象配对中位差 -0.0231805，改善 5/8 父对象；表面接口 RMS（%L）均值差 0.00293366，父对象配对中位差 -0.0193861，改善 5/8 父对象；GT coverage均值差 -0.00610352，父对象配对中位差 -0.00415039，改善 2/8 父对象；CF 受影响 free 比例均值差 0.00295895，父对象配对中位差 0，改善 3/8 父对象；FF 非法受影响 free 比例均值差 0.00204797，父对象配对中位差 0.00136107，改善 4/8 父对象。平均有限边接口改善伴随覆盖下降，应保留这项代价，不能宣布 Full 全面占优。

4. **下一步应扩大数据、增加训练预算，还是先修数据/实现？** 当前只支持 8 个条件微调未见父对象、1000 步预算下的描述性判断。若视图和覆盖仍不足，下一轮应在新登记协议中结合训练走势决定增加预算；若改善稳定跨父对象，再扩大独立数据检验。本轮的单个终点 teacher 检查不能说明学习曲线或充分收敛。该预算下无效不等于整个方法被否定。本次到冻结预算即停止；不自动扩训、换 checkpoint、扫描求解器或更改默认模型。

正式请求 **96**，实际尝试 **96/96**，评分有效输出 **96/96**；已尝试失败 **0**，待运行或待评分 **0**。所有请求均保留，登记占位行不计作失败尝试。
均值先在父对象内平均两块 C × 两噪声，再对 8 个父对象等权汇总。任何父对象少一个有效输出，该父对象完整均值为 NA；完整八对象总均值也为 NA。可用分母诊断另列于 JSON，不能替代主均值。

## 完整性、接口与质量

数值沿用原评分器的单位与阈值。C 本身排除于剩余几何分数，每块 C 使用自己的 GT 自由补集；不同 N 的相交对数需结合受影响 free 比例。

| 指标 | 单位 | T1 | Geo | Full | 有效分母 T1 / Geo / Full |
| --- | --- | --- | --- | --- | --- |
| remainder_symmetric_RMS_bbox_pct | % L | 6.95316 | 6.98217 | 6.97827 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| free_to_R12_RMS_bbox_pct | % L | 6.50029 | 6.4996 | 6.46592 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| R12_to_free_RMS_bbox_pct | % L | 7.19211 | 7.23445 | 7.27094 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| fixed_R12_coverage_RMS_bbox_pct | % L | 7.20721 | 7.24908 | 7.28381 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| boundary_RMS_bbox_pct | % L | 6.33968 | 6.44452 | 6.44745 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| boundary_p95_bbox_pct | % L | 10.5512 | 10.6617 | 10.6322 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| Gamma_to_free_edges_RMS_bbox_pct | % L | 6.42696 | 6.52911 | 6.52793 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| Gamma_to_free_edges_p95_bbox_pct | % L | 10.5879 | 10.7218 | 10.677 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| fixed_R12_coverage_within_1pct_fraction | fraction | 0.104187 | 0.107544 | 0.10144 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| boundary_coverage_fraction | fraction | 0.00677485 | 0.0104765 | 0.00943307 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| Gamma_to_free_edges_coverage_0.001_fraction | fraction | 0.00029179 | 0.00119767 | 0.000936675 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| CF_proper_pair_count | count | 29.0312 | 28.3125 | 28.1875 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| CF_coplanar_pair_count | count | 0 | 0 | 0 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| CF_unresolved_or_degenerate_pair_count | count | 0 | 0 | 0 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| FF_proper_pair_count | count | 250.125 | 255.188 | 254.688 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| FF_coplanar_pair_count | count | 0 | 0 | 0 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| FF_illegal_pair_count | count | 250.125 | 255.188 | 254.688 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| FF_unresolved_or_degenerate_pair_count | count | 0 | 0 | 0 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| free_near_degenerate_count | count | 0 | 0 | 0 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| free_exact_zero_area_count | count | 0 | 0 | 0 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| free_nan_coordinate_count | count | 0 | 0 | 0 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| free_inf_coordinate_count | count | 0 | 0 | 0 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| CF_proper_affected_free_fraction | fraction | 0.127901 | 0.125448 | 0.128407 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| FF_proper_affected_free_fraction | fraction | 0.92025 | 0.919618 | 0.921666 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| FF_illegal_affected_free_fraction | fraction | 0.92025 | 0.919618 | 0.921666 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| free_minimum_angle_lt5_fraction | fraction | 0.152382 | 0.151657 | 0.151196 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| free_q_gt10_fraction | fraction | 0.074347 | 0.0726669 | 0.0725543 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| free_near_degenerate_fraction | fraction | 0 | 0 | 0 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| free_exact_zero_area_fraction | fraction | 0 | 0 | 0 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| free_angle_p05_degrees | degrees | 5.84785 | 5.75507 | 5.79529 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| free_q_mean | dimensionless | 4.18794 | 4.31057 | 4.40283 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| C_preserved | bool | 1 | 1 | 1 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| actual_N | faces | 173.125 | 173.125 | 173.125 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| actual_K | faces | 34.375 | 34.375 | 34.375 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| actual_free_faces | faces | 138.75 | 138.75 | 138.75 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| free_total_surface_area | model coordinate squared | 76.8587 | 76.4886 | 74.8794 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| GT_free_total_surface_area | model coordinate squared | 90.9796 | 90.9796 | 90.9796 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |
| free_area_over_GT_free_area | ratio | 0.961317 | 0.955304 | 0.928104 | 8/8父,32/32输出 / 8/8父,32/32输出 / 8/8父,32/32输出 |

## 父对象配对增量

差值方向固定为后者−前者：Geo−T1、Full−Geo、Full−T1。误差/非法相交越低越好，coverage 越高越好；描述性字段不判断改善。中位差和改善数的单位是父对象，不把 32 个 case 当作 32 个独立形状。

### Geo-T1

| 指标 | 均值差 | 父对象配对中位差 | 改善父对象/有效父对象 | 最差父对象(Δ) | 最差对象/C/噪声(Δ) |
| --- | --- | --- | --- | --- | --- |
| remainder_symmetric_RMS_bbox_pct | 0.0290159 | 0.0447798 | 2/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.19012160265257627}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.7086967139503475}] |
| free_to_R12_RMS_bbox_pct | -0.00068589 | 0.019291 | 3/8 | [{"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.1377326257855387}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.4446682337090131}] |
| R12_to_free_RMS_bbox_pct | 0.0423354 | 0.0117552 | 2/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.34965424649348265}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.9844154301973136}] |
| fixed_R12_coverage_RMS_bbox_pct | 0.0418655 | 0.0294971 | 2/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.39856435005780355}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 1.0094873629129308}] |
| boundary_RMS_bbox_pct | 0.104837 | 0.0947619 | 3/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.42457757681712494}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.7175694696695842}] |
| boundary_p95_bbox_pct | 0.110522 | 0.0791587 | 3/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.5149182986755694}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 1.452612216936462}] |
| Gamma_to_free_edges_RMS_bbox_pct | 0.102151 | 0.0978675 | 3/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.4100333135602714}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.705360351524523}] |
| Gamma_to_free_edges_p95_bbox_pct | 0.133907 | 0.0848652 | 2/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.5149182986755694}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 1.452612216936462}] |
| fixed_R12_coverage_within_1pct_fraction | 0.00335693 | -0.00170898 | 3/8 | [{"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": -0.017578125}] | [{"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": -0.109375}] |
| boundary_coverage_fraction | 0.00370164 | 0.000254701 | 4/8 | [{"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": -0.009405515683485216}] | [{"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": -0.029994350220413652}] |
| Gamma_to_free_edges_coverage_0.001_fraction | 0.000905876 | 0 | 3/8 | [{"parent_id": "183974726abab0454aa6191ddaf4b676", "value": -3.830043959336411e-06}] | [{"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": -1.5320175837345643e-05}] |
| CF_proper_pair_count | -0.71875 | 0.125 | 3/8 | [{"parent_id": "d1caa22978e516655c31590a2627247", "value": 1.5}] | [{"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 8.0}] |
| CF_coplanar_pair_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| CF_unresolved_or_degenerate_pair_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| FF_proper_pair_count | 5.0625 | 3.875 | 2/8 | [{"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 20.25}] | [{"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 81.0}] |
| FF_coplanar_pair_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| FF_illegal_pair_count | 5.0625 | 3.875 | 2/8 | [{"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 20.25}] | [{"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 81.0}] |
| FF_unresolved_or_degenerate_pair_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_near_degenerate_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_exact_zero_area_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_nan_coordinate_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_inf_coordinate_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| CF_proper_affected_free_fraction | -0.00245275 | 0.00234848 | 3/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.006355932203389834}] | [{"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.027522935779816515}] |
| FF_proper_affected_free_fraction | -0.00063199 | -0.00227273 | 4/8 | [{"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.021523178807947074}] | [{"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.04635761589403975}] |
| FF_illegal_affected_free_fraction | -0.00063199 | -0.00227273 | 4/8 | [{"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.021523178807947074}] | [{"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.04635761589403975}] |
| free_minimum_angle_lt5_fraction | -0.00072435 | 0.000313456 | 4/8 | [{"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.008561643835616445}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.04242424242424242}] |
| free_q_gt10_fraction | -0.00168016 | -7.57576e-05 | 4/8 | [{"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.004716981132075478}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.02424242424242426}] |
| free_near_degenerate_fraction | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_exact_zero_area_fraction | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_angle_p05_degrees | -0.0927744 | -0.10128 | 3/8 | [{"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": -0.42713210128010104}] | [{"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": -1.0603899379504123}] |
| free_q_mean | 0.122629 | 0.0768837 | 1/8 | [{"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.3990637433742714}] | [{"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 2.335206439019424}] |
| C_preserved | 0 | 0 | 描述性 | NA | NA |
| actual_N | 0 | 0 | 描述性 | NA | NA |
| actual_K | 0 | 0 | 描述性 | NA | NA |
| actual_free_faces | 0 | 0 | 描述性 | NA | NA |
| free_total_surface_area | -0.370126 | -0.567443 | 描述性 | NA | NA |
| GT_free_total_surface_area | 0 | 0 | 描述性 | NA | NA |
| free_area_over_GT_free_area | -0.00601263 | -0.00771633 | 描述性 | NA | NA |

### Full-Geo

| 指标 | 均值差 | 父对象配对中位差 | 改善父对象/有效父对象 | 最差父对象(Δ) | 最差对象/C/噪声(Δ) |
| --- | --- | --- | --- | --- | --- |
| remainder_symmetric_RMS_bbox_pct | -0.00390346 | 0.00248291 | 4/8 | [{"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.10958460789988589}] | [{"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.4293960103248331}] |
| free_to_R12_RMS_bbox_pct | -0.0336797 | -0.00325453 | 4/8 | [{"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.15364826947790888}] | [{"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.49664545659719783}] |
| R12_to_free_RMS_bbox_pct | 0.0364976 | 0.0240291 | 3/8 | [{"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.16302339175236558}] | [{"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.726134652043652}] |
| fixed_R12_coverage_RMS_bbox_pct | 0.034732 | 0.0297487 | 3/8 | [{"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.15283567049811797}] | [{"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.6808287296979518}] |
| boundary_RMS_bbox_pct | 0.00293366 | -0.0193861 | 5/8 | [{"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.13947229248810622}] | [{"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.5165372270170883}] |
| boundary_p95_bbox_pct | -0.0295634 | 0.00453228 | 4/8 | [{"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.13082466367983425}] | [{"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.5975246847042257}] |
| Gamma_to_free_edges_RMS_bbox_pct | -0.00118235 | -0.0231805 | 5/8 | [{"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.13754668168555817}] | [{"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.48848081773512764}] |
| Gamma_to_free_edges_p95_bbox_pct | -0.0448639 | -0.00162194 | 4/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.10620611119146339}] | [{"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.23648228728253784}] |
| fixed_R12_coverage_within_1pct_fraction | -0.00610352 | -0.00415039 | 2/8 | [{"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": -0.02880859375}] | [{"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": -0.123046875}] |
| boundary_coverage_fraction | -0.00104342 | -0.000471473 | 2/8 | [{"parent_id": "d1caa22978e516655c31590a2627247", "value": -0.006512698305193222}] | [{"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": -0.05198431150388404}] |
| Gamma_to_free_edges_coverage_0.001_fraction | -0.000260991 | 0 | 2/8 | [{"parent_id": "d1caa22978e516655c31590a2627247", "value": -0.0034611911631558475}] | [{"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": -0.019431874758649107}] |
| CF_proper_pair_count | -0.125 | -0.5 | 5/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 3.0}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 12.0}] |
| CF_coplanar_pair_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| CF_unresolved_or_degenerate_pair_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| FF_proper_pair_count | -0.5 | -2 | 6/8 | [{"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 8.75}] | [{"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 37.0}] |
| FF_coplanar_pair_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| FF_illegal_pair_count | -0.5 | -2 | 6/8 | [{"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 8.75}] | [{"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 37.0}] |
| FF_unresolved_or_degenerate_pair_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_near_degenerate_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_exact_zero_area_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_nan_coordinate_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_inf_coordinate_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| CF_proper_affected_free_fraction | 0.00295895 | 0 | 3/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.015151515151515138}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.03030303030303033}] |
| FF_proper_affected_free_fraction | 0.00204797 | 0.00136107 | 4/8 | [{"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.015410958904109706}] | [{"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.06849315068493156}] |
| FF_illegal_affected_free_fraction | 0.00204797 | 0.00136107 | 4/8 | [{"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.015410958904109706}] | [{"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.06849315068493156}] |
| free_minimum_angle_lt5_fraction | -0.000461922 | -0.000856164 | 4/8 | [{"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.008333333333333331}] | [{"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.03999999999999998}] |
| free_q_gt10_fraction | -0.00011262 | -6.93889e-18 | 4/8 | [{"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0034246575342465613}] | [{"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.018181818181818174}] |
| free_near_degenerate_fraction | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_exact_zero_area_fraction | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_angle_p05_degrees | 0.0402187 | 0.08612 | 6/8 | [{"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": -0.37685330765290637}] | [{"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": -1.2689719821572876}] |
| free_q_mean | 0.092259 | 0.0145095 | 4/8 | [{"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.7450625694820321}] | [{"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 3.2579015756686793}] |
| C_preserved | 0 | 0 | 描述性 | NA | NA |
| actual_N | 0 | 0 | 描述性 | NA | NA |
| actual_K | 0 | 0 | 描述性 | NA | NA |
| actual_free_faces | 0 | 0 | 描述性 | NA | NA |
| free_total_surface_area | -1.60924 | -0.940608 | 描述性 | NA | NA |
| GT_free_total_surface_area | 0 | 0 | 描述性 | NA | NA |
| free_area_over_GT_free_area | -0.0272006 | -0.0106018 | 描述性 | NA | NA |

### Full-T1

| 指标 | 均值差 | 父对象配对中位差 | 改善父对象/有效父对象 | 最差父对象(Δ) | 最差对象/C/噪声(Δ) |
| --- | --- | --- | --- | --- | --- |
| remainder_symmetric_RMS_bbox_pct | 0.0251124 | 0.0431878 | 3/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.17209806328211918}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.6821253504371958}] |
| free_to_R12_RMS_bbox_pct | -0.0343656 | -0.0333361 | 5/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.08766713538220916}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.5300811909860501}] |
| R12_to_free_RMS_bbox_pct | 0.078833 | 0.0877596 | 2/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.3663237996949764}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.991948630836565}] |
| fixed_R12_coverage_RMS_bbox_pct | 0.0765974 | 0.062441 | 2/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.4159647088928047}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 1.0148908794527065}] |
| boundary_RMS_bbox_pct | 0.107771 | 0.0628884 | 2/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.4624417597474828}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.7407573599663984}] |
| boundary_p95_bbox_pct | 0.080959 | 0.0188154 | 4/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.5941929041927914}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.9231398756731082}] |
| Gamma_to_free_edges_RMS_bbox_pct | 0.100969 | 0.0617258 | 2/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.46087695622750324}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.7272639395871767}] |
| Gamma_to_free_edges_p95_bbox_pct | 0.0890432 | 0.0379024 | 3/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.5907772445517434}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.9231398756731082}] |
| fixed_R12_coverage_within_1pct_fraction | -0.00274658 | -0.00537109 | 2/8 | [{"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": -0.01513671875}] | [{"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": -0.08984375}] |
| boundary_coverage_fraction | 0.00265822 | -0.000683399 | 4/8 | [{"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": -0.0050476450231199}] | [{"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": -0.039681839678016334}] |
| Gamma_to_free_edges_coverage_0.001_fraction | 0.000644885 | 0.000243802 | 4/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}] | [{"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": -0.003864042020883794}] |
| CF_proper_pair_count | -0.84375 | -0.875 | 5/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 3.25}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 11.0}] |
| CF_coplanar_pair_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| CF_unresolved_or_degenerate_pair_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| FF_proper_pair_count | 4.5625 | 2 | 3/8 | [{"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 23.75}] | [{"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 79.0}] |
| FF_coplanar_pair_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| FF_illegal_pair_count | 4.5625 | 2 | 3/8 | [{"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 23.75}] | [{"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 79.0}] |
| FF_unresolved_or_degenerate_pair_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_near_degenerate_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_exact_zero_area_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_nan_coordinate_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_inf_coordinate_count | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| CF_proper_affected_free_fraction | 0.000506198 | -0.000757576 | 4/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.018181818181818188}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.054545454545454564}] |
| FF_proper_affected_free_fraction | 0.00141598 | 0.00126248 | 4/8 | [{"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0281456953642385}] | [{"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.07272727272727275}] |
| FF_illegal_affected_free_fraction | 0.00141598 | 0.00126248 | 4/8 | [{"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0281456953642385}] | [{"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.07272727272727275}] |
| free_minimum_angle_lt5_fraction | -0.00118627 | -0.0018788 | 4/8 | [{"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.015151515151515194}] | [{"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.048484848484848464}] |
| free_q_gt10_fraction | -0.00179278 | -0.000757576 | 4/8 | [{"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.006849315068493143}] | [{"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.02649006622516556}] |
| free_near_degenerate_fraction | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_exact_zero_area_fraction | 0 | 0 | 0/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": 0.0}, {"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.0}, {"parent_id": "22b48b577aab6201bb99d7af794753a2", "value": 0.0}, {"parent_id": "4f7e9b96404fe19af37f22e93a631b3a", "value": 0.0}, {"parent_id": "76fe7cf10c5dbf1edcb466b6f48b5810", "value": 0.0}, {"parent_id": "79d5977609f99087473f10e6caaeca56", "value": 0.0}, {"parent_id": "d1caa22978e516655c31590a2627247", "value": 0.0}, {"parent_id": "f268849e43f64861cb3a30f2c37c38a6", "value": 0.0}] | [{"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["1459c329e2123d4fe5b03ab845ae95c", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["22b48b577aab6201bb99d7af794753a2", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["76fe7cf10c5dbf1edcb466b6f48b5810", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["79d5977609f99087473f10e6caaeca56", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["d1caa22978e516655c31590a2627247", "p1", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p0", 1], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 0], "value": 0.0}, {"parent_condition_noise": ["f268849e43f64861cb3a30f2c37c38a6", "p1", 1], "value": 0.0}] |
| free_angle_p05_degrees | -0.0525556 | -0.128144 | 3/8 | [{"parent_id": "1459c329e2123d4fe5b03ab845ae95c", "value": -0.3428469445310398}] | [{"parent_condition_noise": ["4f7e9b96404fe19af37f22e93a631b3a", "p0", 0], "value": -0.8324017396590815}] |
| free_q_mean | 0.214888 | 0.162898 | 2/8 | [{"parent_id": "183974726abab0454aa6191ddaf4b676", "value": 0.7640387191259004}] | [{"parent_condition_noise": ["183974726abab0454aa6191ddaf4b676", "p0", 0], "value": 3.4278593296745132}] |
| C_preserved | 0 | 0 | 描述性 | NA | NA |
| actual_N | 0 | 0 | 描述性 | NA | NA |
| actual_K | 0 | 0 | 描述性 | NA | NA |
| actual_free_faces | 0 | 0 | 描述性 | NA | NA |
| free_total_surface_area | -1.97937 | -0.990567 | 描述性 | NA | NA |
| GT_free_total_surface_area | 0 | 0 | 描述性 | NA | NA |
| free_area_over_GT_free_area | -0.0332132 | -0.0164206 | 描述性 | NA | NA |

## 失败、验证与控制边界

96 条正式请求均有有效评分；无输出因几何表现差被替换或删除。

验证 teacher 请求 **24**；已登记 **24**、表内记录 **24**、实际尝试 **24/24**、有效检查 **24/24**；登记或待运行记录不是已执行的前向。检查固定 t=.5、第一块 C、冻结 teacher 噪声及配对。teacher 的 GT/OT 状态与自由生成严格分开，仅解释学习状态，不替代完整生成、不挑 checkpoint、不推断完整控制能力。

同 N/K 正确/交换条件比较：**NOT_RUN**。No distinct parents share all of actual N,K and registered free Gaussian bytes

同一父对象的两块 C 来自同一完整目标，输出不同不等于几何编辑成功，也不要求完整形状不同。若未完成冻结的跨父对象四格比较，本轮尚未全面验证几何编辑控制。

精确拓扑与水密认证：**NOT_RUN**。C 保持、实际面数、退化、NaN/Inf、UNRESOLVED、角质量以逐输出原记录为准，未运行的认证不填 0。

## 协议、身份与实际账本

协议目标终点：T1_step1000、Geo_step1000、Full_step1000；实际统一严格加载及身份核对状态详见 results.json 的 identity，共同官方初始化、数据 hash、任务步数、分支来源按 protocol 核对。无 step500 选择，无 raw/EMA 混选。

原 Euler50；BF16 前向、FP32 参数/状态/积分；严格确定性、TF32 关闭、SDPA math、eval/no_grad；每条 51 状态保持 C。raw known 速度不要求恒零。没有训练 loss、自由生成 GT/OT 初始化、Guidance、Pair 或几何修补。

正式 NFE 请求 4800；显式记录实际 NFE 的输出 96/96，已记录实际前向合计 4800；完整实际合计 4800。teacher、必要重复和失败尝试另见实际账本，不混入 96 条。

```json
{
  "sampling": {
    "model_forward_attempts": 4800,
    "model_forward_returns": 4800,
    "backbone_forward_attempts": 4800,
    "backbone_forward_returns": 4800,
    "euler_updates": 4800,
    "complete_rollouts": 96,
    "rollout_attempts": 96,
    "teacher_attempts": 0,
    "teacher_returns": 0,
    "model_backward_calls": 0,
    "optimizer_updates": 0,
    "geometry_gradient_calls": 0,
    "postprocessing_repairs": 0
  },
  "teacher": {
    "model_forward_attempts": 24,
    "model_forward_returns": 24,
    "backbone_forward_attempts": 24,
    "backbone_forward_returns": 24,
    "euler_updates": 0,
    "complete_rollouts": 0,
    "rollout_attempts": 0,
    "teacher_attempts": 24,
    "teacher_returns": 24,
    "model_backward_calls": 0,
    "optimizer_updates": 0,
    "geometry_gradient_calls": 0,
    "postprocessing_repairs": 0
  },
  "scoring": {
    "endpoint_scores_attempted": 96,
    "prediction_surface_sample_calls": 96,
    "remainder_surface_distance_calls": 192,
    "remainder_point_triangle_pairs": 13639680,
    "fixed_GT_coverage_distance_calls": 96,
    "fixed_GT_coverage_point_triangle_pairs": 6819840,
    "Gamma_surface_calls_attempted": 96,
    "Gamma_surface_point_triangle_pairs": 6936474,
    "Gamma_surface_calls_returned": 96,
    "gamma_to_free_edges_calls_attempted": 96,
    "gamma_to_free_edges_point_segment_pairs": 20809422,
    "gamma_to_free_edges_zero_length_occurrences": 0,
    "gamma_to_free_edges_calls_returned": 96,
    "free_shape_quality_calls": 96,
    "GT_free_area_calculations": 96,
    "verified_pair_tests": 1412328,
    "primary_pair_tests": 1412328,
    "independent_pair_tests": 1412328,
    "independent_AABB_disjoint": 1315931,
    "endpoint_scores_returned": 96
  },
  "scoring_wall_seconds": 257.34985328199764,
  "GPU_status": "PASS",
  "scoring_status": "PASS",
  "abrupt_exit_counts_may_be_incomplete": false,
  "interrupted_request": null,
  "CPU_preparation_OT_attempts": 8,
  "CPU_preparation_OT_returns": 8,
  "CPU_scoring_workers": 1,
  "GPU_workers": 1,
  "repeats_requested": 0,
  "repeats_actual": 0,
  "generation_GT_OT_calls": 0,
  "postprocessing_repairs": 0,
  "retries": 0,
  "phase_timing": {
    "total_worker_seconds": 239.700552581,
    "sampling_seconds": 149.5105364610008,
    "teacher_seconds": 1.2775823790002505
  },
  "resources": {
    "phase_snapshots": [
      {
        "phase": "lock_admission",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:27:27Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 2778, 13202, 1",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:27:27 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   30C    P8             13W /  360W |    2778MiB /  16303MiB |      2%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|  No running processes found                                                             |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.013671875,
            1.13525390625,
            1.09912109375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "64395704 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 14.9  0.0    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "   2759  0.0  0.0    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 2778.0,
          "free_MiB": 13202.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 62886.4296875,
          "initial": true
        }
      },
      {
        "phase": "T1_load",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:00Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 2770, 13210, 1",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:00 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   30C    P8             12W /  360W |    2770MiB /  16303MiB |      1%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|  No running processes found                                                             |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.0068359375,
            1.12060546875,
            1.09326171875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "64013008 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 21.6  0.9    2 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "   2759  0.0  0.0    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 2770.0,
          "free_MiB": 13210.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 62512.703125,
          "initial": true
        }
      },
      {
        "phase": "case00_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:10Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3639, 12341, 4",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:10 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   30C    P1             21W /  360W |    3639MiB /  16303MiB |     10%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.00537109375,
            1.11572265625,
            1.09130859375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "63758112 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 21.8  1.1    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "   2759  0.0  0.0    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3639.0,
          "free_MiB": 12341.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 62263.78125,
          "initial": false
        }
      },
      {
        "phase": "case01_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:13Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3737, 12243, 22",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:13 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   32C    P1             63W /  360W |    3737MiB /  16303MiB |     12%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.08544921875,
            1.130859375,
            1.09619140625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62832724 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 24.8  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "   2759  0.0  0.0    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3737.0,
          "free_MiB": 12243.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61360.08203125,
          "initial": false
        }
      },
      {
        "phase": "case02_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:15Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3729, 12251, 21",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:15 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   32C    P1             63W /  360W |    3729MiB /  16303MiB |      7%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.08544921875,
            1.130859375,
            1.09619140625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62797952 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 27.1  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "   2759  0.0  0.1    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3729.0,
          "free_MiB": 12251.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61326.125,
          "initial": false
        }
      },
      {
        "phase": "case03_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:17Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3737, 12243, 24",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:17 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   32C    P1             66W /  360W |    3737MiB /  16303MiB |     24%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.15869140625,
            1.1455078125,
            1.10107421875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62544316 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 28.7  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.0  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3737.0,
          "free_MiB": 12243.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61078.43359375,
          "initial": false
        }
      },
      {
        "phase": "case04_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:19Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3737, 12243, 33",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:19 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P1             70W /  360W |    3733MiB /  16303MiB |     33%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.15869140625,
            1.1455078125,
            1.10107421875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62541036 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 30.0  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.1  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3737.0,
          "free_MiB": 12243.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61075.23046875,
          "initial": false
        }
      },
      {
        "phase": "case05_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:20Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3732, 12248, 30",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:20 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             72W /  360W |    3732MiB /  16303MiB |     30%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.15869140625,
            1.1455078125,
            1.10107421875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62545216 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 31.2  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3732.0,
          "free_MiB": 12248.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61079.3125,
          "initial": false
        }
      },
      {
        "phase": "case06_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:22Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 35",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:22 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             75W /  360W |    3739MiB /  16303MiB |     35%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.22607421875,
            1.16015625,
            1.10595703125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62562208 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 32.5  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61095.90625,
          "initial": false
        }
      },
      {
        "phase": "case07_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:24Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3731, 12249, 37",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:24 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             74W /  360W |    3731MiB /  16303MiB |     15%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.22607421875,
            1.16015625,
            1.10595703125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62564928 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 33.7  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.3  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3731.0,
          "free_MiB": 12249.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61098.5625,
          "initial": false
        }
      },
      {
        "phase": "case08_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:26Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 18",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:26 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   32C    P1             66W /  360W |    3739MiB /  16303MiB |     18%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.20751953125,
            1.1572265625,
            1.10498046875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62568144 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 35.4  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.3  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61101.703125,
          "initial": false
        }
      },
      {
        "phase": "case09_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:28Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3731, 12249, 24",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:28 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P1             64W /  360W |    3731MiB /  16303MiB |     24%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.20751953125,
            1.1572265625,
            1.10498046875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62582088 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 36.9  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.3  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3731.0,
          "free_MiB": 12249.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61115.3203125,
          "initial": false
        }
      },
      {
        "phase": "case10_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:30Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 36",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:30 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             68W /  360W |    3739MiB /  16303MiB |     13%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.20751953125,
            1.1572265625,
            1.10498046875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62519800 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 38.3  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.4  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61054.4921875,
          "initial": false
        }
      },
      {
        "phase": "case11_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:32Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3732, 12248, 28",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:32 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             69W /  360W |    3732MiB /  16303MiB |     28%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.27099609375,
            1.17138671875,
            1.10986328125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62509820 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 39.3  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.4  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3732.0,
          "free_MiB": 12248.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61044.74609375,
          "initial": false
        }
      },
      {
        "phase": "case12_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:33Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 34",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:33 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             73W /  360W |    3739MiB /  16303MiB |     19%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.27099609375,
            1.17138671875,
            1.10986328125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62520636 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 40.2  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61055.30859375,
          "initial": false
        }
      },
      {
        "phase": "case13_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:35Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3732, 12248, 30",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:35 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P1             67W /  360W |    3732MiB /  16303MiB |     30%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.27099609375,
            1.17138671875,
            1.10986328125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62503416 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 41.1  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3732.0,
          "free_MiB": 12248.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61038.4921875,
          "initial": false
        }
      },
      {
        "phase": "case14_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:37Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3732, 12248, 31",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:37 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             71W /  360W |    3732MiB /  16303MiB |     31%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.32958984375,
            1.185546875,
            1.11474609375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62519864 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 42.0  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.6  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3732.0,
          "free_MiB": 12248.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61054.5546875,
          "initial": false
        }
      },
      {
        "phase": "case15_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:38Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 36",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:38 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             73W /  360W |    3739MiB /  16303MiB |     22%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.32958984375,
            1.185546875,
            1.11474609375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62509488 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 42.8  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.7  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61044.421875,
          "initial": false
        }
      },
      {
        "phase": "case16_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:40Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3732, 12248, 35",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:40 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             72W /  360W |    3732MiB /  16303MiB |     35%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.32958984375,
            1.185546875,
            1.11474609375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62518544 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 43.5  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.7  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3732.0,
          "free_MiB": 12248.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61053.265625,
          "initial": false
        }
      },
      {
        "phase": "case17_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:41Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 27",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:41 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             74W /  360W |    3739MiB /  16303MiB |     27%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.38330078125,
            1.19921875,
            1.11962890625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62508836 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 44.3  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61043.78515625,
          "initial": false
        }
      },
      {
        "phase": "case18_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:43Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3732, 12248, 25",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:43 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             71W /  360W |    3732MiB /  16303MiB |     25%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.38330078125,
            1.19921875,
            1.11962890625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62515116 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 45.0  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.9  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3732.0,
          "free_MiB": 12248.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61049.91796875,
          "initial": false
        }
      },
      {
        "phase": "case19_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:44Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3732, 12248, 24",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:44 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             73W /  360W |    3732MiB /  16303MiB |     24%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.38330078125,
            1.19921875,
            1.11962890625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62535128 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 45.7  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  0.9  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3732.0,
          "free_MiB": 12248.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61069.4609375,
          "initial": false
        }
      },
      {
        "phase": "case20_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:46Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 35",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:46 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             74W /  360W |    3739MiB /  16303MiB |     35%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.43310546875,
            1.212890625,
            1.12451171875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62543524 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 46.5  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.0  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61077.66015625,
          "initial": false
        }
      },
      {
        "phase": "case21_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:48Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3732, 12248, 37",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:48 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             74W /  360W |    3732MiB /  16303MiB |     21%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.43310546875,
            1.212890625,
            1.12451171875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62531404 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 47.2  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.1  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3732.0,
          "free_MiB": 12248.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61065.82421875,
          "initial": false
        }
      },
      {
        "phase": "case22_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:49Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3740, 12240, 35",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:49 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             75W /  360W |    3740MiB /  16303MiB |     21%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.43310546875,
            1.212890625,
            1.12451171875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62557452 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 47.9  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.1  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3740.0,
          "free_MiB": 12240.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61091.26171875,
          "initial": false
        }
      },
      {
        "phase": "case23_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:51Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3744, 12236, 36",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:51 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             76W /  360W |    3744MiB /  16303MiB |     36%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.478515625,
            1.22607421875,
            1.12939453125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62553444 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 48.5  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3744.0,
          "free_MiB": 12236.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61087.34765625,
          "initial": false
        }
      },
      {
        "phase": "case24_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:52Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3744, 12236, 37",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:53 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             73W /  360W |    3744MiB /  16303MiB |     37%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.478515625,
            1.22607421875,
            1.12939453125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62564172 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 49.1  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3744.0,
          "free_MiB": 12236.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61097.82421875,
          "initial": false
        }
      },
      {
        "phase": "case25_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:54Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3747, 12233, 26",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:54 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             70W /  360W |    3746MiB /  16303MiB |     26%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.478515625,
            1.22607421875,
            1.12939453125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62552316 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 49.7  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.3  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3747.0,
          "free_MiB": 12233.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61086.24609375,
          "initial": false
        }
      },
      {
        "phase": "case26_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:56Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3747, 12233, 25",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:56 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             72W /  360W |    3747MiB /  16303MiB |     25%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.5205078125,
            1.2392578125,
            1.13427734375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62534224 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 50.3  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.4  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3747.0,
          "free_MiB": 12233.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61068.578125,
          "initial": false
        }
      },
      {
        "phase": "case27_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:57Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3748, 12232, 33",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:57 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             69W /  360W |    3748MiB /  16303MiB |     33%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.5205078125,
            1.2392578125,
            1.13427734375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62524844 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 50.8  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.4  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3748.0,
          "free_MiB": 12232.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61059.41796875,
          "initial": false
        }
      },
      {
        "phase": "case28_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:28:59Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3748, 12232, 35",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:28:59 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             71W /  360W |    3748MiB /  16303MiB |     14%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.5205078125,
            1.2392578125,
            1.13427734375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62511492 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 51.4  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3748.0,
          "free_MiB": 12232.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61046.37890625,
          "initial": false
        }
      },
      {
        "phase": "case29_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:00Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3737, 12243, 28",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:01 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             70W /  360W |    3737MiB /  16303MiB |     28%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.5205078125,
            1.2392578125,
            1.13427734375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62510760 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 51.9  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.6  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3737.0,
          "free_MiB": 12243.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61045.6640625,
          "initial": false
        }
      },
      {
        "phase": "case30_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:02Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3742, 12238, 33",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:02 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             73W /  360W |    3742MiB /  16303MiB |     33%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.55908203125,
            1.251953125,
            1.13916015625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62504556 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 52.4  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.6  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3742.0,
          "free_MiB": 12238.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61039.60546875,
          "initial": false
        }
      },
      {
        "phase": "case31_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:03Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3742, 12238, 31",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:04 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             70W /  360W |    3742MiB /  16303MiB |     31%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.55908203125,
            1.251953125,
            1.13916015625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62464912 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 52.9  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.7  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3742.0,
          "free_MiB": 12238.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61000.890625,
          "initial": false
        }
      },
      {
        "phase": "teacher00_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:05Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 35",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:05 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             71W /  360W |    3739MiB /  16303MiB |     22%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.55908203125,
            1.251953125,
            1.13916015625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62466872 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 53.4  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.7  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61002.8046875,
          "initial": false
        }
      },
      {
        "phase": "teacher01_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:05Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 5",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:05 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             67W /  360W |    3739MiB /  16303MiB |      5%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.55908203125,
            1.251953125,
            1.13916015625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62472364 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 53.4  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.7  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61008.16796875,
          "initial": false
        }
      },
      {
        "phase": "teacher02_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:05Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 5",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:05 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             61W /  360W |    3739MiB /  16303MiB |      5%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.55908203125,
            1.251953125,
            1.13916015625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62456836 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 53.3  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60993.00390625,
          "initial": false
        }
      },
      {
        "phase": "teacher03_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:06Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 5",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:06 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P1             54W /  360W |    3739MiB /  16303MiB |      5%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.5947265625,
            1.2646484375,
            1.14404296875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62458384 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 53.3  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60994.515625,
          "initial": false
        }
      },
      {
        "phase": "teacher04_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:06Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 2",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:06 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P1             46W /  360W |    3739MiB /  16303MiB |      2%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.5947265625,
            1.2646484375,
            1.14404296875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62456528 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 53.3  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60992.703125,
          "initial": false
        }
      },
      {
        "phase": "teacher05_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:06Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 4",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:06 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P1             45W /  360W |    3739MiB /  16303MiB |      4%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.5947265625,
            1.2646484375,
            1.14404296875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62448712 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 53.3  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60985.0703125,
          "initial": false
        }
      },
      {
        "phase": "teacher06_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:06Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 4",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:06 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P1             43W /  360W |    3739MiB /  16303MiB |      4%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.5947265625,
            1.2646484375,
            1.14404296875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62468248 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 53.2  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 61004.1484375,
          "initial": false
        }
      },
      {
        "phase": "teacher07_T1",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:06Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 6",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:07 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P3             43W /  360W |    3739MiB /  16303MiB |      6%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.5947265625,
            1.2646484375,
            1.14404296875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62442408 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 53.2  2.8    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60978.9140625,
          "initial": false
        }
      },
      {
        "phase": "T1_finished",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:07Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3157, 12823, 6",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:07 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P3             36W /  360W |    3157MiB /  16303MiB |      6%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.5947265625,
            1.2646484375,
            1.14404296875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62400400 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 53.3  2.9    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3157.0,
          "free_MiB": 12823.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60937.890625,
          "initial": false
        }
      },
      {
        "phase": "Geo_load",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:07Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3157, 12823, 6",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:07 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P3             36W /  360W |    3157MiB /  16303MiB |      0%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.5947265625,
            1.2646484375,
            1.14404296875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62401792 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 53.3  2.9    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  1.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3157.0,
          "free_MiB": 12823.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60939.25,
          "initial": true
        }
      },
      {
        "phase": "case00_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:17Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3709, 12271, 53",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:17 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   31C    P5             17W /  360W |    3709MiB /  16303MiB |     32%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.65771484375,
            1.28955078125,
            1.15380859375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62456720 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 51.1  2.9    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  2.0  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3709.0,
          "free_MiB": 12271.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60992.890625,
          "initial": false
        }
      },
      {
        "phase": "case01_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:19Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3730, 12250, 33",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:19 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P1             68W /  360W |    3730MiB /  16303MiB |     11%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.65771484375,
            1.28955078125,
            1.15380859375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62408088 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 51.7  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  2.1  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3730.0,
          "free_MiB": 12250.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60945.3984375,
          "initial": false
        }
      },
      {
        "phase": "case02_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:21Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3738, 12242, 32",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:21 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             68W /  360W |    3738MiB /  16303MiB |     17%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.685546875,
            1.3017578125,
            1.15869140625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62400024 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 52.2  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  2.1  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3738.0,
          "free_MiB": 12242.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60937.5234375,
          "initial": false
        }
      },
      {
        "phase": "case03_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:23Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3740, 12240, 28",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:23 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             70W /  360W |    3740MiB /  16303MiB |     28%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.685546875,
            1.3017578125,
            1.15869140625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62405548 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 52.7  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  2.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3740.0,
          "free_MiB": 12240.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60942.91796875,
          "initial": false
        }
      },
      {
        "phase": "case04_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:24Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3732, 12248, 27",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:25 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             67W /  360W |    3732MiB /  16303MiB |     27%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.685546875,
            1.3017578125,
            1.15869140625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62400076 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 53.2  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  2.3  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3732.0,
          "free_MiB": 12248.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60937.57421875,
          "initial": false
        }
      },
      {
        "phase": "case05_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:27Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3749, 12231, 15",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:27 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             63W /  360W |    3749MiB /  16303MiB |      5%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.7109375,
            1.3134765625,
            1.16357421875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62399624 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 53.8  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  2.4  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3749.0,
          "free_MiB": 12231.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60937.1328125,
          "initial": false
        }
      },
      {
        "phase": "case06_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:28Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3749, 12231, 35",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:28 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             72W /  360W |    3748MiB /  16303MiB |     15%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.7109375,
            1.3134765625,
            1.16357421875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62405948 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 54.2  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  2.4  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3749.0,
          "free_MiB": 12231.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60943.30859375,
          "initial": false
        }
      },
      {
        "phase": "case07_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:30Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3744, 12236, 28",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:30 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             72W /  360W |    3744MiB /  16303MiB |     28%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.7109375,
            1.3134765625,
            1.16357421875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62410124 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 54.6  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  2.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3744.0,
          "free_MiB": 12236.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60947.38671875,
          "initial": false
        }
      },
      {
        "phase": "case08_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:32Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3735, 12245, 34",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:32 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             74W /  360W |    3735MiB /  16303MiB |     34%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.734375,
            1.3251953125,
            1.16845703125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62421388 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 55.0  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  2.6  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3735.0,
          "free_MiB": 12245.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60958.38671875,
          "initial": false
        }
      },
      {
        "phase": "case09_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:33Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3743, 12237, 34",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:33 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             72W /  360W |    3743MiB /  16303MiB |     20%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.734375,
            1.3251953125,
            1.16845703125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62416556 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 55.4  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  2.6  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3743.0,
          "free_MiB": 12237.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60953.66796875,
          "initial": false
        }
      },
      {
        "phase": "case10_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:35Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3777, 12203, 23",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:35 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             72W /  360W |    3777MiB /  16303MiB |     23%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.734375,
            1.3251953125,
            1.16845703125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62417784 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 55.9  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  2.7  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3777.0,
          "free_MiB": 12203.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60954.8671875,
          "initial": false
        }
      },
      {
        "phase": "case11_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:37Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3766, 12214, 34",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:37 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             72W /  360W |    3766MiB /  16303MiB |     34%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.755859375,
            1.33642578125,
            1.17333984375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62420208 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 56.2  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  2.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3766.0,
          "free_MiB": 12214.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60957.234375,
          "initial": false
        }
      },
      {
        "phase": "case12_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:39Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 29",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:39 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             69W /  360W |    3771MiB /  16303MiB |     29%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.755859375,
            1.33642578125,
            1.17333984375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62443856 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 56.7  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  2.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60980.328125,
          "initial": false
        }
      },
      {
        "phase": "case13_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:41Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 35",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:41 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             70W /  360W |    3771MiB /  16303MiB |     12%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.755859375,
            1.33642578125,
            1.17333984375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62454988 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 57.1  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  2.9  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60991.19921875,
          "initial": false
        }
      },
      {
        "phase": "case14_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:42Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 35",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:42 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             71W /  360W |    3771MiB /  16303MiB |     20%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.77587890625,
            1.34765625,
            1.17822265625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62462924 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 57.5  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  3.0  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60998.94921875,
          "initial": false
        }
      },
      {
        "phase": "case15_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:44Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 31",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:44 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             71W /  360W |    3771MiB /  16303MiB |     31%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.77587890625,
            1.34765625,
            1.17822265625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62454972 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 57.8  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  3.0  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60991.18359375,
          "initial": false
        }
      },
      {
        "phase": "case16_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:46Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 35",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:46 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             70W /  360W |    3771MiB /  16303MiB |     35%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.7939453125,
            1.35888671875,
            1.18310546875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62447760 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 58.2  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  3.1  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60984.140625,
          "initial": false
        }
      },
      {
        "phase": "case17_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:48Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 33",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:48 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             72W /  360W |    3771MiB /  16303MiB |     33%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.7939453125,
            1.35888671875,
            1.18310546875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62434456 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 58.5  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  3.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60971.1484375,
          "initial": false
        }
      },
      {
        "phase": "case18_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:49Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 24",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:50 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             72W /  360W |    3771MiB /  16303MiB |     24%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.7939453125,
            1.35888671875,
            1.18310546875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62413048 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 58.9  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  3.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60950.2421875,
          "initial": false
        }
      },
      {
        "phase": "case19_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:51Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 34",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:51 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             72W /  360W |    3771MiB /  16303MiB |     34%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.810546875,
            1.36962890625,
            1.1875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62406708 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 59.2  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  3.3  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60944.05078125,
          "initial": false
        }
      },
      {
        "phase": "case20_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:53Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 33",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:53 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             70W /  360W |    3771MiB /  16303MiB |     13%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.810546875,
            1.36962890625,
            1.1875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62418312 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 59.5  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  3.4  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60955.3828125,
          "initial": false
        }
      },
      {
        "phase": "case21_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:54Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 33",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:55 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             72W /  360W |    3771MiB /  16303MiB |     33%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.810546875,
            1.36962890625,
            1.1875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62374828 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 59.9  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  3.4  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60912.91796875,
          "initial": false
        }
      },
      {
        "phase": "case22_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:56Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 36",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:56 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             75W /  360W |    3771MiB /  16303MiB |     36%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.826171875,
            1.38037109375,
            1.19189453125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62366284 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 60.2  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  3.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60904.57421875,
          "initial": false
        }
      },
      {
        "phase": "case23_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:29:58Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 25",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:29:58 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             74W /  360W |    3771MiB /  16303MiB |     25%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.826171875,
            1.38037109375,
            1.19189453125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62358112 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 60.5  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  3.6  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60896.59375,
          "initial": false
        }
      },
      {
        "phase": "case24_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:00Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 35",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:00 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             73W /  360W |    3771MiB /  16303MiB |     17%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.826171875,
            1.38037109375,
            1.19189453125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62382276 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 60.8  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  3.6  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60920.19140625,
          "initial": false
        }
      },
      {
        "phase": "case25_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:02Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 27",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:02 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             67W /  360W |    3771MiB /  16303MiB |     27%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.84033203125,
            1.39111328125,
            1.1962890625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62374160 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 61.1  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  3.7  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60912.265625,
          "initial": false
        }
      },
      {
        "phase": "case26_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:03Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 25",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:03 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             68W /  360W |    3771MiB /  16303MiB |     18%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.84033203125,
            1.39111328125,
            1.1962890625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62412424 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 61.4  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  3.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60949.6328125,
          "initial": false
        }
      },
      {
        "phase": "case27_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:05Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 32",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:05 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             70W /  360W |    3771MiB /  16303MiB |     32%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.84033203125,
            1.39111328125,
            1.1962890625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62415176 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 61.7  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  3.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60952.3203125,
          "initial": false
        }
      },
      {
        "phase": "case28_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:07Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 29",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:07 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             69W /  360W |    3771MiB /  16303MiB |      9%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.853515625,
            1.4013671875,
            1.20068359375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62413444 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 61.9  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  3.9  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60950.62890625,
          "initial": false
        }
      },
      {
        "phase": "case29_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:08Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 30",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:09 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             68W /  360W |    3771MiB /  16303MiB |     30%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.853515625,
            1.4013671875,
            1.20068359375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62433244 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.2  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.0  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60969.96484375,
          "initial": false
        }
      },
      {
        "phase": "case30_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:10Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 34",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:10 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             70W /  360W |    3771MiB /  16303MiB |     18%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.853515625,
            1.4013671875,
            1.20068359375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62439280 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.5  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.0  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60975.859375,
          "initial": false
        }
      },
      {
        "phase": "case31_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:12Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 22",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:12 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             71W /  360W |    3771MiB /  16303MiB |     22%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.86572265625,
            1.41162109375,
            1.205078125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62428808 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.7  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.1  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60965.6328125,
          "initial": false
        }
      },
      {
        "phase": "teacher00_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:13Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 35",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:14 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             69W /  360W |    3771MiB /  16303MiB |     35%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.86572265625,
            1.41162109375,
            1.205078125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62447996 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.9  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.1  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60984.37109375,
          "initial": false
        }
      },
      {
        "phase": "teacher01_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:14Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 1",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:14 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             65W /  360W |    3771MiB /  16303MiB |      1%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.86572265625,
            1.41162109375,
            1.205078125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62413696 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.9  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.1  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60950.875,
          "initial": false
        }
      },
      {
        "phase": "teacher02_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:14Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 5",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:14 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             55W /  360W |    3771MiB /  16303MiB |      5%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.86572265625,
            1.41162109375,
            1.205078125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62377208 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.9  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60915.2421875,
          "initial": false
        }
      },
      {
        "phase": "teacher03_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:14Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 4",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:14 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             50W /  360W |    3771MiB /  16303MiB |      4%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.86572265625,
            1.41162109375,
            1.205078125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62382356 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.8  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60920.26953125,
          "initial": false
        }
      },
      {
        "phase": "teacher04_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:14Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 5",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:15 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             43W /  360W |    3771MiB /  16303MiB |      5%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.86572265625,
            1.41162109375,
            1.205078125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62387204 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.8  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60925.00390625,
          "initial": false
        }
      },
      {
        "phase": "teacher05_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:15Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 4",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:15 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             41W /  360W |    3771MiB /  16303MiB |      4%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.86572265625,
            1.41162109375,
            1.205078125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62367492 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.8  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60905.75390625,
          "initial": false
        }
      },
      {
        "phase": "teacher06_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:15Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 2",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:15 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P3             39W /  360W |    3771MiB /  16303MiB |      2%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.86572265625,
            1.41162109375,
            1.205078125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62356272 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.8  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60894.796875,
          "initial": false
        }
      },
      {
        "phase": "teacher07_Geo",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:15Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 3",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:15 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P3             40W /  360W |    3771MiB /  16303MiB |      5%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.86572265625,
            1.41162109375,
            1.205078125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62358852 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.7  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60897.31640625,
          "initial": false
        }
      },
      {
        "phase": "Geo_finished",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:16Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3187, 12793, 6",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:16 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P3             35W /  360W |    3187MiB /  16303MiB |      3%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.876953125,
            1.421875,
            1.20947265625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62328916 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.8  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3187.0,
          "free_MiB": 12793.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60868.08203125,
          "initial": false
        }
      },
      {
        "phase": "Full_load",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:16Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3187, 12793, 3",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:16 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P3             34W /  360W |    3187MiB /  16303MiB |      3%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.876953125,
            1.421875,
            1.20947265625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62326176 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.7  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3187.0,
          "free_MiB": 12793.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60865.40625,
          "initial": true
        }
      },
      {
        "phase": "case00_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:26Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3739, 12241, 71",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:26 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   32C    P5             16W /  360W |    3739MiB /  16303MiB |     20%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.896484375,
            1.44140625,
            1.21826171875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62404580 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 60.8  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.6  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3739.0,
          "free_MiB": 12241.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60941.97265625,
          "initial": false
        }
      },
      {
        "phase": "case01_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:28Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3767, 12213, 24",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:28 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             69W /  360W |    3767MiB /  16303MiB |     24%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.896484375,
            1.44140625,
            1.21826171875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62413064 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 61.1  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.7  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3767.0,
          "free_MiB": 12213.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60950.2578125,
          "initial": false
        }
      },
      {
        "phase": "case02_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:31Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3767, 12213, 21",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:31 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             60W /  360W |    3767MiB /  16303MiB |      8%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.896484375,
            1.44140625,
            1.21826171875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62407004 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 61.5  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.7  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3767.0,
          "free_MiB": 12213.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60944.33984375,
          "initial": false
        }
      },
      {
        "phase": "case03_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:33Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3767, 12213, 14",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:33 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P1             60W /  360W |    3767MiB /  16303MiB |     11%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.82421875,
            1.43359375,
            1.216796875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62402308 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 61.9  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.7  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3767.0,
          "free_MiB": 12213.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60939.75390625,
          "initial": false
        }
      },
      {
        "phase": "case04_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:35Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3767, 12213, 26",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:35 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             68W /  360W |    3767MiB /  16303MiB |     26%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.82421875,
            1.43359375,
            1.216796875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62395816 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.2  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.7  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3767.0,
          "free_MiB": 12213.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60933.4140625,
          "initial": false
        }
      },
      {
        "phase": "case05_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:37Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 31",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:37 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             73W /  360W |    3771MiB /  16303MiB |     31%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.83837890625,
            1.443359375,
            1.22119140625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62397964 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.4  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60935.51171875,
          "initial": false
        }
      },
      {
        "phase": "case06_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:39Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 32",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:39 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             74W /  360W |    3771MiB /  16303MiB |     32%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.83837890625,
            1.443359375,
            1.22119140625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62394932 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.6  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60932.55078125,
          "initial": false
        }
      },
      {
        "phase": "case07_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:40Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 16",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:40 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             70W /  360W |    3771MiB /  16303MiB |     13%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.83837890625,
            1.443359375,
            1.22119140625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62381600 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 62.9  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  4.9  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60919.53125,
          "initial": false
        }
      },
      {
        "phase": "case08_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:42Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 19",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:42 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             73W /  360W |    3771MiB /  16303MiB |     19%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.8515625,
            1.45263671875,
            1.2255859375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62365384 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 63.1  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.0  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60903.6953125,
          "initial": false
        }
      },
      {
        "phase": "case09_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:44Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 30",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:44 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             71W /  360W |    3771MiB /  16303MiB |     30%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.8515625,
            1.45263671875,
            1.2255859375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62329968 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 63.3  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.0  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60869.109375,
          "initial": false
        }
      },
      {
        "phase": "case10_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:46Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 30",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:46 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             72W /  360W |    3771MiB /  16303MiB |     30%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.8515625,
            1.45263671875,
            1.2255859375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62358672 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 63.5  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.1  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60897.140625,
          "initial": false
        }
      },
      {
        "phase": "case11_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:47Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 36",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:47 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             72W /  360W |    3771MiB /  16303MiB |     17%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.86376953125,
            1.4619140625,
            1.22998046875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62391504 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 63.7  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.1  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60929.203125,
          "initial": false
        }
      },
      {
        "phase": "case12_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:49Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 28",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:49 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             70W /  360W |    3771MiB /  16303MiB |     28%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.86376953125,
            1.4619140625,
            1.22998046875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62376220 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 63.9  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60914.27734375,
          "initial": false
        }
      },
      {
        "phase": "case13_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:51Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 27",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:51 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             70W /  360W |    3771MiB /  16303MiB |     27%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.86376953125,
            1.4619140625,
            1.22998046875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62389904 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 64.1  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.3  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60927.640625,
          "initial": false
        }
      },
      {
        "phase": "case14_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:52Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 33",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:52 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             70W /  360W |    3771MiB /  16303MiB |     11%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.875,
            1.47119140625,
            1.234375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62386676 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 64.3  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.3  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60924.48828125,
          "initial": false
        }
      },
      {
        "phase": "case15_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:54Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 31",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:54 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             68W /  360W |    3771MiB /  16303MiB |     13%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.875,
            1.47119140625,
            1.234375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62383764 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 64.5  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.4  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60921.64453125,
          "initial": false
        }
      },
      {
        "phase": "case16_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:56Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 25",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:56 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             73W /  360W |    3771MiB /  16303MiB |     25%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.88525390625,
            1.47998046875,
            1.23876953125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62384792 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 64.7  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60922.6484375,
          "initial": false
        }
      },
      {
        "phase": "case17_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:58Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 34",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:58 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             72W /  360W |    3771MiB /  16303MiB |     34%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.88525390625,
            1.47998046875,
            1.23876953125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62380376 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 64.9  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60918.3359375,
          "initial": false
        }
      },
      {
        "phase": "case18_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:30:59Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 18",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:30:59 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             71W /  360W |    3771MiB /  16303MiB |     18%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.88525390625,
            1.47998046875,
            1.23876953125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62377164 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 65.1  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.6  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60915.19921875,
          "initial": false
        }
      },
      {
        "phase": "case19_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:01Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 32",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:01 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             71W /  360W |    3771MiB /  16303MiB |     32%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.89453125,
            1.48876953125,
            1.2431640625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62381632 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 65.2  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.6  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60919.5625,
          "initial": false
        }
      },
      {
        "phase": "case20_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:03Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 34",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:03 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             71W /  360W |    3771MiB /  16303MiB |     17%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.89453125,
            1.48876953125,
            1.2431640625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62371180 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 65.4  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.7  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60909.35546875,
          "initial": false
        }
      },
      {
        "phase": "case21_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:04Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 27",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:04 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   36C    P1             71W /  360W |    3771MiB /  16303MiB |      8%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.89453125,
            1.48876953125,
            1.2431640625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62367064 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 65.6  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60905.3359375,
          "initial": false
        }
      },
      {
        "phase": "case22_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:06Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 29",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:06 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             72W /  360W |    3771MiB /  16303MiB |     29%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.9033203125,
            1.49755859375,
            1.24755859375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62372772 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 65.8  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.8  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60910.91015625,
          "initial": false
        }
      },
      {
        "phase": "case23_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:08Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 35",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:08 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   36C    P1             74W /  360W |    3771MiB /  16303MiB |     35%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.9033203125,
            1.49755859375,
            1.24755859375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62362064 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 65.9  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.9  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60900.453125,
          "initial": false
        }
      },
      {
        "phase": "case24_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:09Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 35",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:09 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   36C    P1             73W /  360W |    3771MiB /  16303MiB |     18%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.9033203125,
            1.49755859375,
            1.24755859375
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62345784 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 66.1  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  5.9  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60884.5546875,
          "initial": false
        }
      },
      {
        "phase": "case25_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:11Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 33",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:11 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   36C    P1             70W /  360W |    3771MiB /  16303MiB |     33%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.9111328125,
            1.50634765625,
            1.251953125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62338476 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 66.3  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.0  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60877.41796875,
          "initial": false
        }
      },
      {
        "phase": "case26_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:13Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 19",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:13 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             70W /  360W |    3771MiB /  16303MiB |     19%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.9111328125,
            1.50634765625,
            1.251953125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62361104 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 66.5  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.1  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60899.515625,
          "initial": false
        }
      },
      {
        "phase": "case27_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:15Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 32",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:15 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             69W /  360W |    3771MiB /  16303MiB |     15%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.9111328125,
            1.50634765625,
            1.251953125
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62386088 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 66.6  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.1  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60923.9140625,
          "initial": false
        }
      },
      {
        "phase": "case28_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:17Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 33",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:17 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   36C    P1             70W /  360W |    3771MiB /  16303MiB |     33%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.91845703125,
            1.5146484375,
            1.25634765625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62371180 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 66.8  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.2  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60909.35546875,
          "initial": false
        }
      },
      {
        "phase": "case29_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:18Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 34",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:18 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             69W /  360W |    3771MiB /  16303MiB |     14%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.91845703125,
            1.5146484375,
            1.25634765625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62393584 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 66.9  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.3  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60931.234375,
          "initial": false
        }
      },
      {
        "phase": "case30_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:20Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 32",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:20 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   36C    P1             70W /  360W |    3771MiB /  16303MiB |     32%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.91845703125,
            1.5146484375,
            1.25634765625
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62393388 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 67.1  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.3  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60931.04296875,
          "initial": false
        }
      },
      {
        "phase": "case31_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:22Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 27",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:22 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   35C    P1             69W /  360W |    3771MiB /  16303MiB |     27%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.92529296875,
            1.52294921875,
            1.2607421875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62420000 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 67.3  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.4  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60957.03125,
          "initial": false
        }
      },
      {
        "phase": "teacher00_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:24Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 19",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:24 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   36C    P1             70W /  360W |    3771MiB /  16303MiB |     19%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.92529296875,
            1.52294921875,
            1.2607421875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62428108 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 67.4  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.4  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60964.94921875,
          "initial": false
        }
      },
      {
        "phase": "teacher01_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:24Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 3",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:24 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   36C    P1             63W /  360W |    3771MiB /  16303MiB |      3%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.92529296875,
            1.52294921875,
            1.2607421875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62415892 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 67.4  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60953.01953125,
          "initial": false
        }
      },
      {
        "phase": "teacher02_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:24Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 1",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:24 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             57W /  360W |    3771MiB /  16303MiB |      1%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.92529296875,
            1.52294921875,
            1.2607421875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62386812 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 67.4  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60924.62109375,
          "initial": false
        }
      },
      {
        "phase": "teacher03_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:24Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 5",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:25 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             49W /  360W |    3771MiB /  16303MiB |      5%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.92529296875,
            1.52294921875,
            1.2607421875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62400900 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 67.4  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60938.37890625,
          "initial": false
        }
      },
      {
        "phase": "teacher04_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:25Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 1",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:25 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P1             42W /  360W |    3771MiB /  16303MiB |      3%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.92529296875,
            1.52294921875,
            1.2607421875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62427732 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 67.4  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60964.58203125,
          "initial": false
        }
      },
      {
        "phase": "teacher05_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:25Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 3",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:25 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P3             40W /  360W |    3771MiB /  16303MiB |      5%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.92529296875,
            1.52294921875,
            1.2607421875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62408164 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 67.3  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60945.47265625,
          "initial": false
        }
      },
      {
        "phase": "teacher06_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:25Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 4",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:25 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P3             39W /  360W |    3771MiB /  16303MiB |      4%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.92529296875,
            1.52294921875,
            1.2607421875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62420096 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 67.3  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60957.125,
          "initial": false
        }
      },
      {
        "phase": "teacher07_Full",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:26Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3771, 12209, 1",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:26 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   34C    P3             38W /  360W |    3771MiB /  16303MiB |      5%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.92529296875,
            1.52294921875,
            1.2607421875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62395652 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 67.3  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3771.0,
          "free_MiB": 12209.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60933.25390625,
          "initial": false
        }
      },
      {
        "phase": "Full_finished",
        "snapshot": {
          "timestamp_utc": "2026-10-02T10:31:26Z",
          "gpu": {
            "returncode": 0,
            "stdout": "NVIDIA GeForce RTX 5080, 16303, 3187, 12793, 4",
            "stderr": ""
          },
          "gpu_processes": {
            "returncode": 0,
            "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
            "stderr": ""
          },
          "gpu_process_table": {
            "returncode": 0,
            "stdout": "Fri Oct  2 18:31:26 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P3             33W /  360W |    3187MiB /  16303MiB |      4%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
            "stderr": ""
          },
          "cpu_cores": 24,
          "cpu_affinity_cores": 24,
          "cpu_load": [
            1.931640625,
            1.53125,
            1.26513671875
          ],
          "memory": {
            "MemTotal": "65552008 kB",
            "MemAvailable": "62390372 kB",
            "SwapFree": "16777216 kB"
          },
          "cpu_processes": [
            "PID %CPU %MEM NLWP COMMAND",
            "   7345 67.3  3.0    5 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
            "   2759  6.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
            "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
            "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
            "      1  0.0  0.0    1 /sbin/init",
            "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
            "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
            "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
            "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
            "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
            "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
            "    411  0.0  0.0    1 -bash",
            "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
            "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
            "      2  0.0  0.0    2 /init",
            "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
            "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
            "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
            "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
            "    332  0.0  0.0    1 login -- warpwang"
          ]
        },
        "admission": {
          "total_MiB": 16303.0,
          "used_MiB": 3187.0,
          "free_MiB": 12793.0,
          "headroom_MiB": 2048.0,
          "initial_admission_reserve_MiB": 4096,
          "available_RAM_MiB": 60928.09765625,
          "initial": false
        }
      }
    ],
    "monitor": "evaluation/gpu_resources.jsonl",
    "finished": {
      "timestamp_utc": "2026-10-02T10:31:27Z",
      "gpu": {
        "returncode": 0,
        "stdout": "NVIDIA GeForce RTX 5080, 16303, 3187, 12793, 0",
        "stderr": ""
      },
      "gpu_processes": {
        "returncode": 0,
        "stdout": "pid, process_name, used_gpu_memory [MiB]\n7345, [Not Found], [N/A]",
        "stderr": ""
      },
      "gpu_process_table": {
        "returncode": 0,
        "stdout": "Fri Oct  2 18:31:27 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA GeForce RTX 5080        On  |   00000000:02:00.0  On |                  N/A |\n| 33%   33C    P3             31W /  360W |    3187MiB /  16303MiB |      0%      Default |\n|                                         |                        |                  N/A |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A            7345      C   /python3.10                           N/A      |\n+-----------------------------------------------------------------------------------------+",
        "stderr": ""
      },
      "cpu_cores": 24,
      "cpu_affinity_cores": 24,
      "cpu_load": [
        1.931640625,
        1.53125,
        1.26513671875
      ],
      "memory": {
        "MemTotal": "65552008 kB",
        "MemAvailable": "62371124 kB",
        "SwapFree": "16777216 kB"
      },
      "cpu_processes": [
        "PID %CPU %MEM NLWP COMMAND",
        "   7345 67.2  3.0    4 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_gpu_worker",
        "   2759  6.5  0.8    1 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.control_mvp48.eval_watch",
        "     45  0.0  0.0    1 /usr/lib/systemd/systemd-journald",
        "     84  0.0  0.0    1 /usr/lib/systemd/systemd-udevd",
        "      1  0.0  0.0    1 /sbin/init",
        "    214  0.0  0.0    4 /usr/sbin/rsyslogd -n -iNONE",
        "    191  0.0  0.0    1 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers",
        "    236  0.0  0.0    2 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal",
        "    383  0.0  0.0    1 /usr/lib/systemd/systemd --user",
        "    237  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
        "    195  0.0  0.0    1 /usr/lib/systemd/systemd-logind",
        "     77  0.0  0.0    1 /usr/lib/systemd/systemd-resolved",
        "    411  0.0  0.0    1 -bash",
        "    183  0.0  0.0    1 /bin/sh /usr/lib/systemd/scripts/chronyd-starter.sh -n -F 1",
        "    186  0.0  0.0    1 @dbus-daemon --system --address=systemd: --nofork --nopidfile --systemd-activation --syslog-only",
        "      2  0.0  0.0    2 /init",
        "      8  0.0  0.0    2 plan9 --control-socket 7 --log-level 4 --server-fd 8 --pipe-fd 10 --log-truncate",
        "    185  0.0  0.0    1 /usr/sbin/cron -f -P",
        "    227  0.0  0.0    1 /usr/sbin/agetty --noreset --noclear --issue-file=/etc/issue:/etc/issue.d:/run/issue.d:/usr/lib/issue.d - linux",
        "    260  0.0  0.0    1 /usr/sbin/chronyd -n -F 1 -x",
        "    332  0.0  0.0    1 login -- warpwang"
      ]
    }
  }
}
```

GPU 阶段要求使用共同锁且只有一个 worker；CPU 评分可与后续推理重叠。实际资源快照、峰值及分阶段 wall 以账本为准，并行争用下的 wall 不作为隔离性能比较。

## 交付与限制

[results.json](results.json)、[逐请求 96 行](per_output.csv)、[逐父对象 8 行](per_parent.csv)、[24 请求 teacher 表](teacher_validation.csv)、[固定视图](../figures/)。正式视图目标是固定第一份噪声，8 个对象的两块 C 均展示 GT、C、T1、Geo、Full，同视角同范围，不删除失败面板。准备阶段的 GT/C 预览与 NOT_RUN 模型占位面板不代表已生成模型输出；实际图件状态见 results.json 的 figures。

本轮证据范围是条件微调未参与的 8 个父对象，不等于官方预训练从未见过；没有论文级泛化或统计定论。1000 步无效果仅能称该预算下未成立，不能否定整个方法；持续学习的终点不能称充分收敛。没有临时 10% 成功门槛或偏向 Full 的总分。

正式 96 条完成后停止；不自动扩训、扫 loss、改求解器或切换默认模型。
