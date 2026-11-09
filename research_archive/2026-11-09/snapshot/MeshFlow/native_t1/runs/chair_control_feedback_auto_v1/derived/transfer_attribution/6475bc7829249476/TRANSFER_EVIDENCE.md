# TRAIN_FEEDBACK 与 DEV 迁移归因快照

只读既有完整评分，零模型调用、零几何重评分，未访问 CONFIRM。每池保留 8 父 × 2 固定噪声；先父内噪声均值，再父等权。
两池父对象不同，以下增益差是描述性分布差，不是同父因果比较。TRAIN_FEEDBACK 父对象、条件和 START 轨迹参与了辅助监督；其重采样使用原固定噪声。DEV 曾反复用于研究反馈，不能称独立盲测。
正向增益统一表示改善：S/E 增加为正，I/形状误差减少为正。原门槛与适用性均未更改。

| checkpoint | reference | metric | TRAIN gain | DEV gain | TRAIN改善父/8 | DEV改善父/8 | TRAIN父中位 | DEV父中位 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| explore_FM_CONTROL_1000 | START | SeamCov | 0.072166 | -0.020074 | 5.000000 | 2.000000 | 0.006779 | -0.023203 |
| explore_FM_CONTROL_1000 | START | ConflictCov_C | -0.019806 | 0.002533 | 3.000000 | 3.000000 | -0.000977 | -0.001099 |
| explore_FM_CONTROL_1000 | START | ExposedRetain | 0.008179 | 0.013825 | 3.000000 | 5.000000 | -0.000244 | 0.005787 |
| explore_REPLAY_FM_CONTROL_1000 | START | SeamCov | 0.131762 | -0.005783 | 5.000000 | 2.000000 | 0.105806 | -0.017322 |
| explore_REPLAY_FM_CONTROL_1000 | START | ConflictCov_C | -0.024170 | -0.000153 | 3.000000 | 3.000000 | -0.000366 | -0.009888 |
| explore_REPLAY_FM_CONTROL_1000 | START | ExposedRetain | 0.006102 | 0.008862 | 3.000000 | 5.000000 | -0.000366 | 0.010946 |
| explore_TRAJECTORY_REPLAY_1000 | START | SeamCov | 0.211476 | -0.052553 | 5.000000 | 2.000000 | 0.240793 | -0.040316 |
| explore_TRAJECTORY_REPLAY_1000 | START | ConflictCov_C | 0.011902 | 0.009460 | 5.000000 | 5.000000 | 0.015137 | 0.009766 |
| explore_TRAJECTORY_REPLAY_1000 | START | ExposedRetain | 0.034917 | 0.096941 | 6.000000 | 4.000000 | 0.018807 | 0.010864 |
| explore_TRAJECTORY_REPLAY_1000 | explore_REPLAY_FM_CONTROL_1000 | SeamCov | 0.079714 | -0.046770 | 3.000000 | 2.000000 | 0.000000 | -0.031311 |
| explore_TRAJECTORY_REPLAY_1000 | explore_REPLAY_FM_CONTROL_1000 | ConflictCov_C | 0.036072 | 0.009613 | 5.000000 | 6.000000 | 0.005249 | 0.019897 |
| explore_TRAJECTORY_REPLAY_1000 | explore_REPLAY_FM_CONTROL_1000 | ExposedRetain | 0.028815 | 0.088078 | 6.000000 | 5.000000 | 0.007693 | 0.005127 |

## 形状代价（相对参考均值百分比，正数为误差增加）

| checkpoint | reference | metric | TRAIN变化% | DEV变化% |
|---|---|---|---:|---:|
| explore_FM_CONTROL_1000 | START | D_sym2048 | -20.850406 | 15.159058 |
| explore_FM_CONTROL_1000 | START | fixed_query_MSE | 7.884286 | 41.394927 |
| explore_FM_CONTROL_1000 | START | fixed_query_p95_L | 5.401399 | 13.326837 |
| explore_REPLAY_FM_CONTROL_1000 | START | D_sym2048 | -22.939235 | 14.769195 |
| explore_REPLAY_FM_CONTROL_1000 | START | fixed_query_MSE | 6.504515 | 40.707613 |
| explore_REPLAY_FM_CONTROL_1000 | START | fixed_query_p95_L | 0.793036 | 11.042471 |
| explore_TRAJECTORY_REPLAY_1000 | START | D_sym2048 | -57.398606 | 13.401796 |
| explore_TRAJECTORY_REPLAY_1000 | START | fixed_query_MSE | -50.950195 | 17.061275 |
| explore_TRAJECTORY_REPLAY_1000 | START | fixed_query_p95_L | -44.602697 | 8.524510 |
| explore_TRAJECTORY_REPLAY_1000 | explore_REPLAY_FM_CONTROL_1000 | D_sym2048 | -44.717140 | -1.191434 |
| explore_TRAJECTORY_REPLAY_1000 | explore_REPLAY_FM_CONTROL_1000 | fixed_query_MSE | -53.945797 | -16.805301 |
| explore_TRAJECTORY_REPLAY_1000 | explore_REPLAY_FM_CONTROL_1000 | fixed_query_p95_L | -45.038561 | -2.267566 |

## 集中度与解释边界

- explore_FM_CONTROL_1000 对 START：DEV E 的最大贡献父为 f59627a077dcc2566c63b8b48495261a，占净均值增益 64.18%；去掉它的七父描述性增益 0.005660。
- explore_REPLAY_FM_CONTROL_1000 对 START：DEV E 的最大贡献父为 c1b8c2f0c4f3cc43aff7c08b060f5ed6，占净均值增益 39.60%；去掉它的七父描述性增益 0.006118。
- explore_TRAJECTORY_REPLAY_1000 对 START：DEV E 的最大贡献父为 f59627a077dcc2566c63b8b48495261a，占净均值增益 88.23%；去掉它的七父描述性增益 0.013035。
- explore_TRAJECTORY_REPLAY_1000 对 explore_REPLAY_FM_CONTROL_1000：DEV E 的最大贡献父为 f59627a077dcc2566c63b8b48495261a，占净均值增益 94.61%；去掉它的七父描述性增益 0.005423。

- 训练池有收益但 DEV 未同步改善时，可报告“已观察到的局部收益没有普遍迁移”；这些表不能单独证明过拟合、梯度主因或失效机制。
- 如果相对 teacher 有有限改善但相对 START 质量退化，两点必须并列。不能把“优于继续训练的对照”写成“优于 START”。
- 均值、父中位、改善父数量、两个噪声符号及单父贡献必须一起阅读。单父大改善不等于普遍改善；没有给 LOPO 引入新的晋级门槛。
- Capped H3 仅在真实完整评分出现后进入相同表；本快照不预测其结果、不改变冻结 recipe 或最终门槛。
