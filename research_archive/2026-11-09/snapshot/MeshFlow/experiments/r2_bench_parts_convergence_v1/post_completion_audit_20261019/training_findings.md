# Bench 多部位训练／数据接线独立审计

审计日期：2026-10-19。只读历史代码、case、训练日志、已保存布局及固定损失探针；新增文件仅在本目录。未构建模型、未调用任何神经网络、未训练、未使用 GPU、未重评分或新生成。

## 结论

**本次没有找到足以判定整轮 Bench 训练无效的掩码、OT 身份、损失分母、官方参数继承或训练流接线错误。** 这不是“代码一定没问题”的证明，也不是只复核 hash：实际读取全部316条case，执行40个代表性打包检查／76次CPU OT，并逐条比对240,000次生产曝光及30,000次更新的损失分母与归约值。

找到了需要分开解释的三点：

1. **终点预测布局与训练来源布局确有失配。** DEV 75条来源owner中漏26条，召回65.33%；两条腿条件全部关边。这是已保存输出可复算的现象，但不能由此断言全部bulk断连都由漏边造成。
2. **Bench不是把Chair成功设置只换类别复跑。** Chair `PLAIN_S2000` 的“PLAIN”仍为SEAM表示的普通续训，且使用冻结LONG布局；Bench `PLAIN` 是普通Native。Bench SEAM的独立头从COMMON12000分叉时新建，并在多部位TRAIN池上持续训练。这个差别已在注册中披露，属于实验合同差异，不是此次发现的隐藏切换。
3. **收敛监控不覆盖所有部位。** 注册major_parts包含SEAT_BACK，但32C固定loss probe没有SEAT_BACK；相应检查为NA并被跳过。不可据该探针认证所有部位平台。本轮实际因SEAM tip仍变化而到达预算上限，未因此提前停止或声称整体收敛。

## 1. 实际执行与独立核验结果

运行入口：[audit_training_wiring.py](audit_training_wiring.py)。实测18.04秒CPU脚本时间；WSL启动等待不计为脚本计算时间。数组核验未调用旧verify脚本，也未使用评分结果决定PASS。

| 核验 | 实际内容 | 结果 |
| --- | --- | --- |
| 316 case来源身份 | C/Y完整互补；source face/vertex ID与模型角序对应；`pre_ot_full*2 == full_target`；C坐标与source目标精确一致 | PASS |
| owner标签 | 每个正边对应真实free源面；该面只占一个owner槽；锚点顺序等于C开放边的反向两个端点 | PASS |
| 40代表性打包 | 覆盖闭合C、开放全Null、有owner、alias、third-in-C及N/K/B极值，原样与0.93宽度增强，PLAIN/SEAM | PASS |
| CPU OT及角点搬运 | 映射后目标由`source_face_ids`和`source_corner_permutations`精确还原；来源面不丢失、不复制 | PASS，76次CPU OT |
| 实际训练流 | COMMON96,000、PLAIN72,000、SEAM72,000次曝光逐条对应冻结schedule的条件、父、part、宽度、noise/epsilon/time/permutation seed | PASS |
| 实际batch损失 | 30,000条更新全部满足日志分母=该batch有效free标量总数；FM等于记录SSE/分母；seam与bulk SSE相加等于all-free | PASS |
| 已保存推理布局 | 24C的edge顺序与训练来源一致；从预测布尔向量重新构建的anchor/known/free/seam/bulk数组与保存布局完全一致 | PASS |

明细：[case_identity_audit.json](case_identity_audit.json)、[packing_audit.json](packing_audit.json)、[stream_and_denominator_audit.json](stream_and_denominator_audit.json)、[training_audit_receipt.json](training_audit_receipt.json)。

## 2. 掩码、OT、归一化和梯度路径

实际生产代码为 `r2/pilots/bench_parts_convergence_v1/train.py`、`data.py`。开放C的普通臂直接调用维护的Chair `slot_data.supervised_sample(...,'FULL4_C200')`；SEAM调用Chair TRAIN47的 `seam_abc_v1/slot_data_extension.py`。本次开放C代表样本逐字段与这些Chair函数的输出相等。

- 普通臂固定全部C角点，监督全部free角点；SEAM额外固定owner槽的前两个角点，只监督第三点。分母分别为`9(N-K)`和`9(N-K)-6*owner_count`。没有把固定端点作为零目标塞进有效分母。
- 先用原半尺度目标与原始Gaussian完成bulk OT，再以`pre*2`的模型尺度目标构造速度；随后使用`ZH=sqrt(.75)*Z+.5*epsilon`。OT后的source ID和角点排列同时搬运。没有把原source数组误当已排列目标。
- 整个物体先做官方FP32 bbox规范化，pre-OT除0.3762、模型坐标乘2；C取该同一物体中的面，没有再独立缩放C。宽度增强同步作用于full、pre和C_vertices；当次edge features重新由增强后的C生成。
- 每8个microbatch共享同一个有效free标量分母，再累计梯度；没有先逐样本平均再除8。不同K/owner数确实产生不同样本权重，这是Chair既有定义。
- Head的features detach，独立参数和AdamW；BCE不向生成器反传。生成器没有通过预测占用头选择训练布局，始终使用来源owner布局监督；推理才使用C-only头预测。这是既有teacher-forced布局合同，也构成失配风险。
- alias不分配第二张重复面；已选owner的第三点即使来源顶点在C中，也继续是自由点，不新增隐藏GT锚点。本轮TRAIN有11条alias、80个owner的第三点在C中；DEV总池有2条alias、5个third-in-C。这是既有Chair owner-only扩展，不是保证此类顶点会在推理自动合为共享索引。
- 49条闭合C（TRAIN44、DEV5）没有seam槽，走零seam的Native退化路径。SEAM适配器检测mask与普通C-face mask相同后直接调用原Native forward；相应head前向跳过。该路径没有把闭合腿任意切开，也没有无效全零head loss。

对应源码：`seam/model_adapter.py::corner_masked_fm_loss`、`seam_forward`；`training/coupling.py::nested_ot_map/to_noise_slots`；`bench_parts_convergence_v1/data.py::_closed_sample/sample_for_training`。本审计没有再执行模型前向来证明算术等价，只对静态直接委托和数组/实际日志做了核验。

## 3. 初始化、优化器与训练随机流

`bench_parts_convergence_v1/train.py`实际导入 `bench_control_transfer_v1/train.py::initialize`。它从注册的`output/120m-ot-v-bench/checkpoints/01000000.pt`读取**EMA**，验证官方Bench synset与模型配置，剥离统一`module.`前缀，严格载入Native backbone；role和Geo出口检查为零；生成器AdamW fresh。

COMMON实际初始化记录：官方backbone状态SHA `2cc63f56d79b88418840fc44ad4c954d1961ad898a7b81d198a35bf49f0deb25`；条件生成器初始状态SHA `28e1dc6903c28820f9bd39a99eb9bdadc1f4c6289af4a686dfb664e4ce996132`；130,401,155参数。PLAIN/SEAM实际初始化审计都指向同一COMMON12000完整检查点和状态SHA `7f96d80f8ed4cd6f803a4d5b94fb91560497cb1ea134de69eebf5916de57e5ba`。未发现载入Chair、旧Bench或EMA权重混用非EMA优化器的源码路径。

两臂载入相同完整生成器和AdamW，再恢复对应RNG；从schedule绝对索引12001续行。新头初始化使用局部CPU RNG上下文，且fork恢复COMMON RNG后才进入训练。每次模型更新后执行参数顺序、FP32与AdamW逐参数cursor检查；本次另从实际曝光复核两臂72,000样本相同随机身份。此前工程恢复的重算不在此再审，已有独立物理计算账本不能被此PASS覆盖或删去。

实际曝光显示“9000步”不等于四条件LONG的集中训练：本轮SEAM每父1124–1126次，每条件94–1126次，中位280次；COMMON每条件中位374次但仍是Native表示阶段。Chair LONG四C每条件约2000次的集中暴露与这些数值不能直接互换，误差值也不能跨不同条件池解释为相同收敛程度。

## 4. 终点占用/owner失配：从保存布局重算，零头前向

以下按**独立C**统计，不把每条C的两份噪声重复计数。来源owner是监督布局/诊断参照，**不是开放生成唯一正确答案**；来源Null上生成新合法连接可能是合理完成，因此FP不能直接当成几何错误。

| 分组 | C总数/开放C | 来源owner | TP | FN | FP | TN | owner召回 | 完全一致开放布局 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 8/7 | 58 | 52 | 6 | 6 | 8 | 89.66% | 3/7 |
| DEV | 16/13 | 75 | 49 | 26 | 0 | 11 | 65.33% | 4/13 |

终点评测24C的来源alias数量为**0**，故没有alias误开可归因。全316池中的alias案例未进入这个终点面板，不能把池中alias机制当成该面板的直接解释。

两条完全关边的DEV：

- `2d1e5e...__LEG_SUPPORT_1`：15条开放边，来源owner10，预测0。
- `5dda4ef...__LEG_SUPPORT_1`：4条开放边，来源owner4，预测0。

其余12条漏边散布于另外7条DEV条件；9条DEV开放C的布局与来源不一致。另一方面，`2d BACK`、`9d LEG`、`9fd SEAT`、`f495 COMBO_PART_ASSEMBLY`的布局完整对应来源，仍不能仅凭这些布局推出bulk已接入。**漏边确实存在，但“全部DEV bulk零可达是头的问题”尚无因果证据。**

训练head使用冻结平衡BCE权重 `[1.668421052631579,0.713963963963964]`，来自1554 owner/665 inactive，应用顺序是negative/positive，未颠倒。`logit>=0`也按注册执行。该加权输出不是未经加权的owner后验概率；理想加权BCE下零logit对应未加权posterior约0.7003的决策线。这是需要理解的模型/校准选择，不是权重符号写反。此审计不调阈值、不重采样。

逐C和逐边角色、预测、logit：[final_layout_source_diagnostic.json](final_layout_source_diagnostic.json)。

## 5. 与Chair成功链路的真实差别

| 项目 | Chair LONG / PLAIN_S2000 | 本轮Bench |
| --- | --- | --- |
| “PLAIN”命名 | PLAIN_S2000是SEAM生成器无couple普通续训 | PLAIN为原Native，无seam角点锚定 |
| 起点 | 已有Chair条件模型/FULL，再SEAM200、LONG、TRAIN47续训链 | 官方Bench EMA从0适配COMMON12000，再分叉 |
| 条件 | 主要完整座面，LONG集中4C；后续TRAIN47 | 同时包含277条多部位TRAIN C，64父 |
| 头与布局 | LONG头已学四C；PLAIN_S2000使用冻结LONG C-only布局 | 新833参数头在多部位数据上训练，推理用本头 |
| 生成器/FM/OT | 原Native/T1+role+Geo与同一seam适配/受限OT | 同一实现；未新增couple或新loss |
| 图中可用后处理 | S2000后续另有HINGE等独立维护正结果 | 注册为单次Bench D0+原GEO_C+来源定向，没有HINGE |

因此同代码内的Bench Native/SEAM配对是清楚的；跨Chair的视觉差别不能被解释为“唯一变量是类别，所以SEAM类别迁移失效”，也不能用Chair HINGE后的图直接归罪于Bench训练实现。后处理细节不属于本训练审计的重评分范围。

## 6. 可观测的泛化差距与监控边界

终点固定监督探针（读取已有96次前向结果，未新算）SEAM TRAIN/DEV分别为：ALL-free MSE **0.10904/1.37408**，bulk **0.10984/1.36093**，seam-tip **0.02956/2.75868**。这是监督插值状态上的明显差距，提示不能用TRAIN较低tip误差替代未见条件能力；它不是生成质量分数，也不能因为不匹配唯一来源完成就判开放生成不正确。

32C probe在TRAIN和DEV都没有SEAT_BACK，而完整池有TRAIN12/DEV4条SEAT_BACK。注册major_parts仍包含这一类；`convergence.py`对缺失项用`NA_INTERVAL_SKIPPED`，不会阻断平台判定。因此**“所有主要部位均已监测到平台”不成立**。本轮其他SEAM稳定门槛未过，最终是预定上限停止，并非这个缺口造成提前收口；现有训练产物不需因此作废。以后若用这一controller认证全part平台，应在冻结前补齐探针覆盖或要求缺失类别阻断该项认证；不能事后改本轮probe伪造覆盖。

## 审计边界

本报告没有证明现有条件任务足够可辨识、源目标全部干净、模型容量足够、head可校准、或SEAM方法一定泛化。源码路径与数组核验未发现错误，不能消除优化/表示限制；反过来，显著泛化差距也不能单独证明实现错误。若继续，应先针对可测失配选择受控验证，而不是把当前结果整体改标“工程bug导致”。此次不启动任何后续实验。
