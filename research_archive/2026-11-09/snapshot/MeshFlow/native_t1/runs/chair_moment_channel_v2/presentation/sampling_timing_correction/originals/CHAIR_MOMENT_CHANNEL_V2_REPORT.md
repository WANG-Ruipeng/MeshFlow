# Chair Moment Channel Learning V2

**QUALITY_RESULT：NO_CLEAR_LEARNING_SIGNAL**。IMPLEMENTED：DONE；TRAINED：DONE；EVALUATED：DONE。

共同起点为 Hybrid OT_HYBRID / pilot1000 / conditional_total5000 / λ=.25。H_CONT 与 MOMENT_XY 各自最多继续 1000 更新；正式终点必须是 moment_step1000 / conditional_total6000。旧起点的 total5000 不是新终点。两臂继承同一旧 AdamW 状态，MOMENT 的新增参数和优化器状态单列；保持 Geo、FM、λ=.25 及默认模型不变。

本轮检验可学习质心矩关系通道能否超过等预算续训，并保留共同起点质量。V1/r1 的有限表面积分失败不充当 V2 门禁，也不能预设本轮模型优劣。

## 实际执行与预算

下列数字来自本轮 accounting.json 的实际账本摘要。初始化两次完整 Euler50 回放单列；138 个 START 历史缓存不计新生成。额外 I/P 前向为 0。未消耗余量不授权追加科学任务。

```json
{
  "actual_full_sampling_attempts": 278,
  "actual_full_sampling_returns": 278,
  "applicable_unique_generation_requests": 414,
  "failed_actual_calls": 0,
  "generation_not_applicable": 18,
  "historical_raw_cache_hits": 138,
  "isolated_module_test_updates": 1,
  "optimizer_updates": 2000,
  "precheck_forwards": 120,
  "precheck_zero_update_backwards": 44,
  "processed_generation_not_applicable": 18,
  "processed_generation_records": 432,
  "registered_generation_request_limit": 432,
  "registered_unique_generation_requests": 432,
  "sampling_forwards": 13900,
  "scientific_optimizer_update_limit": 2000,
  "total_model_forwards": 30020,
  "training_backwards": 16000,
  "training_microbatch_forwards": 16000,
  "unresolved_calls": 0
}
```

|标签|新采样|历史缓存|原 NA|采样计时样本|采样秒均值|采样秒中位数|
|---|---|---|---|---|---|---|
|START|0|138|6|0|NA|NA|
|H_CONT_END|138|0|6|0|NA|NA|
|MOMENT_XY_END|138|0|6|0|NA|NA|

训练总耗时、每更新耗时与显存作用域保留在 paired_results/cost_summary.json 的实际 training_summary 和 stage 字段；初始化回放不混入正式 endpoint 行。采样缓存耗时不作为模型吞吐。

|初始化回放|状态|逐字节相等|最大差|模型调用|
|---|---|---|---|---|
|A_103b75dfd146976563ed57e35c972b4b_A20_a100_n0|SAMPLED|True|0|50|
|A_103b75dfd146976563ed57e35c972b4b_A40_a100_n0|SAMPLED|True|0|50|

初始化回放结果：PASS；完成数：2；采样 NFE：100。

## 数值合同与成本门禁

本轮总门禁：**PASS**；真实预检：PASS。门禁未通过或固定终点未齐全时，正式方法质量结论为 NA。

FP32 新接受合同为 e_inf=max|Δ|/max(1,max|v_ref|)≤5e-5，e_rms=RMS(Δ)/max(1,RMS(v_ref))≤1e-5。旧逐元素 atol=1e-6、rtol=1e-5 的 flags 原样并列保留；新归一化接受不改写旧 FAIL。BF16 非零置换及长度变化只作数值披露，不套 FP32 门槛。两精度零初始化输出及同形状 padding 内容改变必须逐字节相等，无效 padding 输入梯度必须为零。

|实测检查|fixture|精度|要求|bitwise|e_inf|e_rms|旧 allclose|本轮通过|
|---|---|---|---|---|---|---|---|---|
|zero_identity|ordinary_20|FP32|bitwise|True|0|0|True|True|
|zero_identity|ordinary_20|BF16|bitwise|True|0|0|True|True|
|zero_identity|maximum_20|FP32|bitwise|True|0|0|True|True|
|zero_identity|maximum_20|BF16|bitwise|True|0|0|True|True|
|zero_identity|ordinary_40|FP32|bitwise|True|0|0|True|True|
|zero_identity|ordinary_40|BF16|bitwise|True|0|0|True|True|
|zero_identity|maximum_40|FP32|bitwise|True|0|0|True|True|
|zero_identity|maximum_40|BF16|bitwise|True|0|0|True|True|
|nonzero_permutation|ordinary_20|FP32|normalized|False|1.96379425e-06|6.05420004e-07|False|True|
|variable_padding|ordinary_20|FP32|normalized|True|0|0|True|True|
|same_shape_padding|ordinary_20|FP32|bitwise|True|0|0|True|True|
|same_shape_padding|ordinary_20|FP32|bitwise|True|0|0|True|True|
|H_permuted|ordinary_20|FP32|normalized|False|1.38922637e-06|5.75155663e-07|False|True|
|H_pad_large|ordinary_20|FP32|normalized|True|0|0|True|True|
|nonzero_permutation|ordinary_20|BF16|descriptive|False|0.0132450331|0.00544075534|False|True|
|variable_padding|ordinary_20|BF16|descriptive|True|0|0|True|True|
|same_shape_padding|ordinary_20|BF16|bitwise|True|0|0|True|True|
|same_shape_padding|ordinary_20|BF16|bitwise|True|0|0|True|True|
|H_permuted|ordinary_20|BF16|descriptive|False|0.0165562914|0.00536745252|False|True|
|H_pad_large|ordinary_20|BF16|descriptive|True|0|0|True|True|
|nonzero_permutation|maximum_20|FP32|normalized|False|2.61414892e-06|6.12621849e-07|False|True|
|variable_padding|maximum_20|FP32|normalized|True|0|0|True|True|
|same_shape_padding|maximum_20|FP32|bitwise|True|0|0|True|True|
|same_shape_padding|maximum_20|FP32|bitwise|True|0|0|True|True|
|H_permuted|maximum_20|FP32|normalized|False|2.94640218e-06|6.38499624e-07|False|True|
|H_pad_large|maximum_20|FP32|normalized|True|0|0|True|True|
|nonzero_permutation|maximum_20|BF16|descriptive|False|0.00988142292|0.00454583906|False|True|
|variable_padding|maximum_20|BF16|descriptive|True|0|0|True|True|
|same_shape_padding|maximum_20|BF16|bitwise|True|0|0|True|True|
|same_shape_padding|maximum_20|BF16|bitwise|True|0|0|True|True|
|H_permuted|maximum_20|BF16|descriptive|False|0.0158102767|0.00514996119|False|True|
|H_pad_large|maximum_20|BF16|descriptive|True|0|0|True|True|
|nonzero_permutation|ordinary_40|FP32|normalized|False|1.09528539e-06|6.44582634e-07|False|True|
|variable_padding|ordinary_40|FP32|normalized|True|0|0|True|True|
|same_shape_padding|ordinary_40|FP32|bitwise|True|0|0|True|True|
|same_shape_padding|ordinary_40|FP32|bitwise|True|0|0|True|True|
|H_permuted|ordinary_40|FP32|normalized|False|1.22380015e-06|6.29857032e-07|False|True|
|H_pad_large|ordinary_40|FP32|normalized|True|0|0|True|True|
|nonzero_permutation|ordinary_40|BF16|descriptive|False|0.00932835821|0.00616575233|False|True|
|variable_padding|ordinary_40|BF16|descriptive|True|0|0|True|True|
|same_shape_padding|ordinary_40|BF16|bitwise|True|0|0|True|True|
|same_shape_padding|ordinary_40|BF16|bitwise|True|0|0|True|True|
|H_permuted|ordinary_40|BF16|descriptive|False|0.0149253731|0.0061354082|False|True|
|H_pad_large|ordinary_40|BF16|descriptive|True|0|0|True|True|
|nonzero_permutation|maximum_40|FP32|normalized|False|1.84552218e-06|6.69782412e-07|False|True|
|variable_padding|maximum_40|FP32|normalized|True|0|0|True|True|
|same_shape_padding|maximum_40|FP32|bitwise|True|0|0|True|True|
|same_shape_padding|maximum_40|FP32|bitwise|True|0|0|True|True|
|H_permuted|maximum_40|FP32|normalized|False|1.58199405e-06|6.54798261e-07|False|True|
|H_pad_large|maximum_40|FP32|normalized|True|0|0|True|True|
|nonzero_permutation|maximum_40|BF16|descriptive|False|0.011627907|0.00617492578|False|True|
|variable_padding|maximum_40|BF16|descriptive|True|0|0|True|True|
|same_shape_padding|maximum_40|BF16|bitwise|True|0|0|True|True|
|same_shape_padding|maximum_40|BF16|bitwise|True|0|0|True|True|
|H_permuted|maximum_40|BF16|descriptive|False|0.00968992248|0.0058433102|False|True|
|H_pad_large|maximum_40|BF16|descriptive|True|0|0|True|True|

|padding 梯度|精度|无效坐标数|maxabs|通过|
|---|---|---|---|---|
|ordinary_20|FP32|99|0|True|
|ordinary_20|BF16|99|0|True|
|maximum_20|FP32|99|0|True|
|maximum_20|BF16|99|0|True|
|ordinary_40|FP32|99|0|True|
|ordinary_40|BF16|99|0|True|
|maximum_40|FP32|99|0|True|
|maximum_40|BF16|99|0|True|

|臂|model/optimizer/RNG 已恢复|
|---|---|
|H_CONT|True|
|MOMENT_XY|True|

完整逐层定位、首次差异与梯度接线以 precheck/result.json、保存数组及 CPU 测试原表为准。不能只因旧 H 有置换误差就把 MOMENT 的全部误差归给旧 backbone；同形状 padding 不变性和零输入梯度是直接执行合同。

|fixture|臂|成本项|重复数|中位秒|最小秒|最大秒|M/H|该任务门禁|
|---|---|---|---|---|---|---|---|---|
|ordinary_20|H_CONT|forward_seconds|3|0.068115065|0.043864814|0.091231174|1.44771112|True|
|ordinary_20|H_CONT|whole_forward_backward_seconds|3|0.145796014|0.105019961|0.187155752|1.87024173|True|
|ordinary_20|MOMENT_V1_REFERENCE|forward_seconds|1|0.167239478|0.167239478|0.167239478|1.44771112|True|
|ordinary_20|MOMENT_V1_REFERENCE|whole_forward_backward_seconds|1|0.316031291|0.316031291|0.316031291|1.87024173|True|
|ordinary_20|MOMENT_XY|forward_seconds|3|0.098610937|0.093828658|0.13835789|1.44771112|True|
|ordinary_20|MOMENT_XY|whole_forward_backward_seconds|3|0.272673789|0.226000127|0.297963678|1.87024173|True|
|maximum_20|H_CONT|forward_seconds|3|0.055468526|0.049207115|0.066146251|1.98652612|True|
|maximum_20|H_CONT|whole_forward_backward_seconds|3|0.135626289|0.113316363|0.162451739|2.07411923|True|
|maximum_20|MOMENT_V1_REFERENCE|forward_seconds|1|0.122383889|0.122383889|0.122383889|1.98652612|True|
|maximum_20|MOMENT_V1_REFERENCE|whole_forward_backward_seconds|1|0.290134163|0.290134163|0.290134163|2.07411923|True|
|maximum_20|MOMENT_XY|forward_seconds|3|0.110189676|0.105069219|0.133094157|1.98652612|True|
|maximum_20|MOMENT_XY|whole_forward_backward_seconds|3|0.281305094|0.275632931|0.2889072|2.07411923|True|
|ordinary_40|H_CONT|forward_seconds|3|0.042752637|0.041628241|0.056604745|2.1180541|True|
|ordinary_40|H_CONT|whole_forward_backward_seconds|3|0.122097747|0.115271169|0.159105824|1.90028392|True|
|ordinary_40|MOMENT_V1_REFERENCE|forward_seconds|1|0.127597102|0.127597102|0.127597102|2.1180541|True|
|ordinary_40|MOMENT_V1_REFERENCE|whole_forward_backward_seconds|1|0.260920978|0.260920978|0.260920978|1.90028392|True|
|ordinary_40|MOMENT_XY|forward_seconds|3|0.090552398|0.089296216|0.110185526|2.1180541|True|
|ordinary_40|MOMENT_XY|whole_forward_backward_seconds|3|0.232020385|0.23093327|0.270038755|1.90028392|True|
|maximum_40|H_CONT|forward_seconds|3|0.065038942|0.047380354|0.066917599|2.35458763|True|
|maximum_40|H_CONT|whole_forward_backward_seconds|3|0.123304451|0.112118084|0.126062322|2.09278539|True|
|maximum_40|MOMENT_V1_REFERENCE|forward_seconds|1|0.104230204|0.104230204|0.104230204|2.35458763|True|
|maximum_40|MOMENT_V1_REFERENCE|whole_forward_backward_seconds|1|0.276986182|0.276986182|0.276986182|2.09278539|True|
|maximum_40|MOMENT_XY|forward_seconds|3|0.153139888|0.109782379|0.169330298|2.35458763|True|
|maximum_40|MOMENT_XY|whole_forward_backward_seconds|3|0.258049754|0.230630543|0.317960402|2.09278539|True|

成本要求每个原任务完整 forward 和 forward+backward 均不超过 H_CONT 的 3 倍，并保留至少 15% 显存。正式 H/M 使用预注册重复；MOMENT_V1_REFERENCE 只是一轮等价优化的旧实现成本参照，单次测量不足以给出稳健排名。计时保留重复值、median/min/max；不能把并发墙钟当成孤立算子性能。门禁成本状态：PASS。

## 固定主终点与起点保真

fixed_query_MSE = mean([d(q,F)/L0]^2)，q 是原冻结 512 个 GT free coverage 查询点。主面板仅 A20 / validation / alpha=1，此时 L=L0。先在每父内平均两份原噪声，再对 8 父等权。配对差中位数从逐父差计算；缺失噪声不缩小分母。两份采样噪声不代表两个独立训练随机流。

|比较（后−前）|前值|后值|父均差|配对差中位数|改善/8|最差父|最差差|有效/8|开发性几何候选|
|---|---|---|---|---|---|---|---|---|---|
|H_CONT_END-START|0.0015310739|0.00159614391|6.50700038e-05|-6.25293333e-06|4/8|c1b8c2f0c4f3cc43aff7c08b060f5ed6|0.00164053694|8/8|False|
|MOMENT_XY_END-H_CONT_END|0.00159614391|0.00164336613|4.72222232e-05|3.46263117e-05|3/8|c1b8c2f0c4f3cc43aff7c08b060f5ed6|0.000190913277|8/8|False|
|MOMENT_XY_END-START|0.0015310739|0.00164336613|0.000112292227|2.62320657e-05|3/8|c1b8c2f0c4f3cc43aff7c08b060f5ed6|0.00183145022|8/8|False|

几何候选必须同时满足父均差<0、逐父差中位数<0、改善≥6/8，且 8 父两噪声齐全。这是开发性筛选，不是显著性检验。MOMENT−H_CONT 测新增通道效应；MOMENT−START 直接检查保真；只击败退化对照不等于改善起点。

|比较|噪声|前值|后值|父均差|配对差中位数|改善父|有效/8|
|---|---|---|---|---|---|---|---|
|H_CONT_END-START|0|0.00162863006|0.00127849476|-0.000350135299|-7.52178193e-05|5|8/8|
|H_CONT_END-START|1|0.00143351775|0.00191379305|0.000480275307|-1.06112237e-05|4|8/8|
|MOMENT_XY_END-H_CONT_END|0|0.00127849476|0.00132319174|4.46969783e-05|-1.32869748e-06|4|8/8|
|MOMENT_XY_END-H_CONT_END|1|0.00191379305|0.00196354052|4.97474682e-05|-2.66724732e-06|4|8/8|
|MOMENT_XY_END-START|0|0.00162863006|0.00132319174|-0.000305438321|-2.33921471e-05|4|8/8|
|MOMENT_XY_END-START|1|0.00143351775|0.00196354052|0.000530022775|2.82632965e-05|4|8/8|

## 几何方向与完整质量代价

以下为真实原方向值：GT→F = R12_to_free_RMS_bbox_pct/100；F→GT = free_to_R12_RMS_bbox_pct/100。原双向 RMS 使用另一组冻结 GT 点和预测表面积采样，不能与主终点 MSE、coverage 查询点或 RMS 的平方混用。各方向先逐输出计算，再噪声均值、父等权。

|A20 validation 标签|主 MSE|GT→F RMS/L|F→GT RMS/L|原双向 RMS/L|预测小角率|同 GT 小角率|小角率绝对差|
|---|---|---|---|---|---|---|---|
|START|0.0015310739|0.0354906758|0.0379022235|0.0371301102|0.146756635|0.1782926|0.13389986|
|H_CONT_END|0.00159614391|0.0361460484|0.0368976819|0.0370724351|0.103444725|0.1782926|0.100110922|
|MOMENT_XY_END|0.00164336613|0.0365229656|0.0372867138|0.0374433391|0.106654148|0.1782926|0.101037477|

小角参照来自每个完全相同的 GT free 补集及原评分公式。小角率升降、或更接近 GT 分布，均不能独立证明完整网格质量；角度分位数、退化计数和 GT 角度数组另行保留。

|比较|面板/划分/比例|质量项|前值|后值|配对差|改善父|退化父|有效/登记|
|---|---|---|---|---|---|---|---|---|
|H_CONT_END-START|A/train/0.2|Gamma_to_free_edges_RMS_bbox_pct|2.75071435|1.98749773|-0.763216618|6|2|8/8|
|H_CONT_END-START|A/train/0.2|Gamma_to_free_edges_coverage_0.001_fraction|0.0293113784|0.0478014169|0.0184900385|3|4|8/8|
|H_CONT_END-START|A/train/0.2|CF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|H_CONT_END-START|A/train/0.2|FF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|H_CONT_END-START|A/train/0.2|CF_proper_affected_free_fraction|0.0736577208|0.0644660418|-0.00919167901|5|2|8/8|
|H_CONT_END-START|A/train/0.2|FF_illegal_affected_free_fraction|0.884851803|0.872797785|-0.012054018|6|1|8/8|
|H_CONT_END-START|A/train/0.2|free_minimum_angle_lt5_fraction|0.344644305|0.346403042|0.00175873771|5|3|8/8|
|H_CONT_END-START|A/train/0.2|free_near_degenerate_fraction|0|0|0|0|0|8/8|
|H_CONT_END-START|A/train/0.2|fixed_query_p95_L|0.0784553699|0.0688686643|-0.00958670569|5|3|8/8|
|H_CONT_END-START|A/train/0.2|coverage_fraction|0.586303711|0.623168945|0.0368652344|4|4|8/8|
|H_CONT_END-START|A/train/0.2|GT_to_F_original_RMS_L|0.0423794185|0.0340387698|-0.0083406487|6|2|8/8|
|H_CONT_END-START|A/train/0.2|F_to_GT_original_RMS_L|0.042163594|0.0351922226|-0.00697137145|3|5|8/8|
|H_CONT_END-START|A/train/0.2|bidirectional_original_RMS_L|0.0442654759|0.0363877118|-0.00787776406|4|4|8/8|
|H_CONT_END-START|A/train/0.4|Gamma_to_free_edges_RMS_bbox_pct|1.45867841|0.692886044|-0.765792369|6|1|7/8|
|H_CONT_END-START|A/train/0.4|Gamma_to_free_edges_coverage_0.001_fraction|0.0623733626|0.0944921359|0.0321187733|5|2|7/8|
|H_CONT_END-START|A/train/0.4|CF_unresolved_or_degenerate_pair_count|0|0|0|0|0|7/8|
|H_CONT_END-START|A/train/0.4|FF_unresolved_or_degenerate_pair_count|0|0|0|0|0|7/8|
|H_CONT_END-START|A/train/0.4|CF_proper_affected_free_fraction|0.127044934|0.12563041|-0.00141452394|3|1|7/8|
|H_CONT_END-START|A/train/0.4|FF_illegal_affected_free_fraction|0.813767863|0.796142171|-0.0176256919|4|3|7/8|
|H_CONT_END-START|A/train/0.4|free_minimum_angle_lt5_fraction|0.411390141|0.427835967|0.016445826|1|4|7/8|
|H_CONT_END-START|A/train/0.4|free_near_degenerate_fraction|0|0|0|0|0|7/8|
|H_CONT_END-START|A/train/0.4|fixed_query_p95_L|0.0564662299|0.0340150371|-0.0224511928|6|1|7/8|
|H_CONT_END-START|A/train/0.4|coverage_fraction|0.735770089|0.846819196|0.111049107|5|2|7/8|
|H_CONT_END-START|A/train/0.4|GT_to_F_original_RMS_L|0.02549237|0.0156916795|-0.00980069054|5|2|7/8|
|H_CONT_END-START|A/train/0.4|F_to_GT_original_RMS_L|0.0334423563|0.0219682123|-0.011474144|6|1|7/8|
|H_CONT_END-START|A/train/0.4|bidirectional_original_RMS_L|0.0336910034|0.0211411599|-0.0125498435|6|1|7/8|
|H_CONT_END-START|A/validation/0.2|Gamma_to_free_edges_RMS_bbox_pct|3.25389446|3.54236429|0.288469836|5|3|8/8|
|H_CONT_END-START|A/validation/0.2|Gamma_to_free_edges_coverage_0.001_fraction|0.00356081392|0.00374090998|0.000180096065|4|3|8/8|
|H_CONT_END-START|A/validation/0.2|CF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|H_CONT_END-START|A/validation/0.2|FF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|H_CONT_END-START|A/validation/0.2|CF_proper_affected_free_fraction|0.0601349117|0.0642753414|0.00414042965|2|6|8/8|
|H_CONT_END-START|A/validation/0.2|FF_illegal_affected_free_fraction|0.926270307|0.912083382|-0.0141869249|5|3|8/8|
|H_CONT_END-START|A/validation/0.2|free_minimum_angle_lt5_fraction|0.146756635|0.103444725|-0.0433119105|6|2|8/8|
|H_CONT_END-START|A/validation/0.2|free_near_degenerate_fraction|0|0|0|0|0|8/8|
|H_CONT_END-START|A/validation/0.2|fixed_query_p95_L|0.0688447336|0.0705683421|0.00172360855|2|6|8/8|
|H_CONT_END-START|A/validation/0.2|coverage_fraction|0.322998047|0.349609375|0.0266113281|5|3|8/8|
|H_CONT_END-START|A/validation/0.2|GT_to_F_original_RMS_L|0.0354906758|0.0361460484|0.000655372563|4|4|8/8|
|H_CONT_END-START|A/validation/0.2|F_to_GT_original_RMS_L|0.0379022235|0.0368976819|-0.00100454165|4|4|8/8|
|H_CONT_END-START|A/validation/0.2|bidirectional_original_RMS_L|0.0371301102|0.0370724351|-5.76751391e-05|4|4|8/8|
|H_CONT_END-START|A/validation/0.4|Gamma_to_free_edges_RMS_bbox_pct|3.58848382|3.78184245|0.193358633|2|4|6/8|
|H_CONT_END-START|A/validation/0.4|Gamma_to_free_edges_coverage_0.001_fraction|0.00772880716|0.0060890601|-0.00163974706|1|5|6/8|
|H_CONT_END-START|A/validation/0.4|CF_unresolved_or_degenerate_pair_count|0|0|0|0|0|6/8|
|H_CONT_END-START|A/validation/0.4|FF_unresolved_or_degenerate_pair_count|0|0|0|0|0|6/8|
|H_CONT_END-START|A/validation/0.4|CF_proper_affected_free_fraction|0.124005408|0.0972346394|-0.0267707689|4|2|6/8|
|H_CONT_END-START|A/validation/0.4|FF_illegal_affected_free_fraction|0.795531369|0.827846377|0.0323150089|1|4|6/8|
|H_CONT_END-START|A/validation/0.4|free_minimum_angle_lt5_fraction|0.0713944917|0.0634258424|-0.00796864933|4|2|6/8|
|H_CONT_END-START|A/validation/0.4|free_near_degenerate_fraction|0|0|0|0|0|6/8|
|H_CONT_END-START|A/validation/0.4|fixed_query_p95_L|0.0477609319|0.0530776412|0.00531670926|3|3|6/8|
|H_CONT_END-START|A/validation/0.4|coverage_fraction|0.489095052|0.478515625|-0.0105794271|5|1|6/8|
|H_CONT_END-START|A/validation/0.4|GT_to_F_original_RMS_L|0.0235015075|0.0252522992|0.00175079167|2|4|6/8|
|H_CONT_END-START|A/validation/0.4|F_to_GT_original_RMS_L|0.0248270135|0.0238422287|-0.000984784817|4|2|6/8|
|H_CONT_END-START|A/validation/0.4|bidirectional_original_RMS_L|0.0249682466|0.0253061713|0.000337924693|2|4|6/8|
|H_CONT_END-START|C/holdout/0.2|Gamma_to_free_edges_RMS_bbox_pct|5.53945628|5.39512072|-0.144335558|4|4|8/8|
|H_CONT_END-START|C/holdout/0.2|Gamma_to_free_edges_coverage_0.001_fraction|0.00144493129|0.00109027836|-0.00035465293|2|3|8/8|
|H_CONT_END-START|C/holdout/0.2|CF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|H_CONT_END-START|C/holdout/0.2|FF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|H_CONT_END-START|C/holdout/0.2|CF_proper_affected_free_fraction|0.10839684|0.106093888|-0.0023029513|5|3|8/8|
|H_CONT_END-START|C/holdout/0.2|FF_illegal_affected_free_fraction|0.907462619|0.904800319|-0.00266230003|4|4|8/8|
|H_CONT_END-START|C/holdout/0.2|free_minimum_angle_lt5_fraction|0.155340129|0.142169554|-0.0131705751|4|4|8/8|
|H_CONT_END-START|C/holdout/0.2|free_near_degenerate_fraction|0|0|0|0|0|8/8|
|H_CONT_END-START|C/holdout/0.2|fixed_query_p95_L|0.116751343|0.11787365|0.00112230699|3|5|8/8|
|H_CONT_END-START|C/holdout/0.2|coverage_fraction|0.134094238|0.140686035|0.00659179688|4|4|8/8|
|H_CONT_END-START|C/holdout/0.2|GT_to_F_original_RMS_L|0.0689005351|0.0680007811|-0.00089975405|4|4|8/8|
|H_CONT_END-START|C/holdout/0.2|F_to_GT_original_RMS_L|0.0617709302|0.0629091718|0.00113824164|3|5|8/8|
|H_CONT_END-START|C/holdout/0.2|bidirectional_original_RMS_L|0.0663016449|0.0665064051|0.000204760168|3|5|8/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|Gamma_to_free_edges_RMS_bbox_pct|1.98749773|1.91544634|-0.0720513962|5|3|8/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|Gamma_to_free_edges_coverage_0.001_fraction|0.0478014169|0.0618776634|0.0140762465|6|1|8/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|CF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|FF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|CF_proper_affected_free_fraction|0.0644660418|0.0635812242|-0.000884817638|4|2|8/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|FF_illegal_affected_free_fraction|0.872797785|0.880136873|0.00733908746|3|5|8/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|free_minimum_angle_lt5_fraction|0.346403042|0.342239403|-0.0041636393|4|4|8/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|free_near_degenerate_fraction|0|0|0|0|0|8/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|fixed_query_p95_L|0.0688686643|0.0639804532|-0.00488821109|5|3|8/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|coverage_fraction|0.623168945|0.627441406|0.00427246094|5|2|8/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|GT_to_F_original_RMS_L|0.0340387698|0.0316510341|-0.00238773567|6|2|8/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|F_to_GT_original_RMS_L|0.0351922226|0.035689769|0.000497546402|4|4|8/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.2|bidirectional_original_RMS_L|0.0363877118|0.0356146523|-0.000773059507|5|3|8/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|Gamma_to_free_edges_RMS_bbox_pct|0.692886044|0.681110225|-0.0117758184|2|5|7/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|Gamma_to_free_edges_coverage_0.001_fraction|0.0944921359|0.0808882644|-0.0136038715|2|5|7/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|CF_unresolved_or_degenerate_pair_count|0|0|0|0|0|7/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|FF_unresolved_or_degenerate_pair_count|0|0|0|0|0|7/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|CF_proper_affected_free_fraction|0.12563041|0.128612789|0.00298237847|3|4|7/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|FF_illegal_affected_free_fraction|0.796142171|0.790057368|-0.00608480334|5|2|7/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|free_minimum_angle_lt5_fraction|0.427835967|0.42875863|0.000922662933|4|3|7/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|free_near_degenerate_fraction|0|0|0|0|0|7/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|fixed_query_p95_L|0.0340150371|0.0301670196|-0.00384801754|6|1|7/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|coverage_fraction|0.846819196|0.853934152|0.00711495536|5|1|7/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|GT_to_F_original_RMS_L|0.0156916795|0.0143726702|-0.0013190093|4|3|7/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|F_to_GT_original_RMS_L|0.0219682123|0.0219629035|-5.30882341e-06|2|5|7/8|
|MOMENT_XY_END-H_CONT_END|A/train/0.4|bidirectional_original_RMS_L|0.0211411599|0.0202108161|-0.00093034387|3|4|7/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|Gamma_to_free_edges_RMS_bbox_pct|3.54236429|3.6998947|0.157530411|3|5|8/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|Gamma_to_free_edges_coverage_0.001_fraction|0.00374090998|0.00289292998|-0.000847980003|3|4|8/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|CF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|FF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|CF_proper_affected_free_fraction|0.0642753414|0.0635169158|-0.000758425638|3|1|8/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|FF_illegal_affected_free_fraction|0.912083382|0.911316981|-0.000766401107|4|3|8/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|free_minimum_angle_lt5_fraction|0.103444725|0.106654148|0.00320942377|1|4|8/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|free_near_degenerate_fraction|0|0|0|0|0|8/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|fixed_query_p95_L|0.0705683421|0.0710179676|0.000449625478|4|4|8/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|coverage_fraction|0.349609375|0.344116211|-0.00549316406|4|4|8/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|GT_to_F_original_RMS_L|0.0361460484|0.0365229656|0.000376917244|3|5|8/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|F_to_GT_original_RMS_L|0.0368976819|0.0372867138|0.000389031897|4|4|8/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.2|bidirectional_original_RMS_L|0.0370724351|0.0374433391|0.00037090403|3|5|8/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|Gamma_to_free_edges_RMS_bbox_pct|3.78184245|3.84771334|0.0658708908|1|5|6/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|Gamma_to_free_edges_coverage_0.001_fraction|0.0060890601|0.00791354906|0.00182448897|3|2|6/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|CF_unresolved_or_degenerate_pair_count|0|0|0|0|0|6/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|FF_unresolved_or_degenerate_pair_count|0|0|0|0|0|6/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|CF_proper_affected_free_fraction|0.0972346394|0.0936497014|-0.00358493801|3|2|6/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|FF_illegal_affected_free_fraction|0.827846377|0.822788069|-0.00505830881|4|2|6/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|free_minimum_angle_lt5_fraction|0.0634258424|0.0606270282|-0.00279881423|2|2|6/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|free_near_degenerate_fraction|0|0|0|0|0|6/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|fixed_query_p95_L|0.0530776412|0.053563402|0.000485760795|1|5|6/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|coverage_fraction|0.478515625|0.469726562|-0.0087890625|2|4|6/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|GT_to_F_original_RMS_L|0.0252522992|0.0255742349|0.000321935709|3|3|6/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|F_to_GT_original_RMS_L|0.0238422287|0.0248268573|0.000984628663|2|4|6/8|
|MOMENT_XY_END-H_CONT_END|A/validation/0.4|bidirectional_original_RMS_L|0.0253061713|0.0259420292|0.000635857904|2|4|6/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|Gamma_to_free_edges_RMS_bbox_pct|5.39512072|5.38660488|-0.00851584348|3|5|8/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|Gamma_to_free_edges_coverage_0.001_fraction|0.00109027836|0.00299737002|0.00190709166|6|2|8/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|CF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|FF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|CF_proper_affected_free_fraction|0.106093888|0.102396281|-0.00369760708|4|3|8/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|FF_illegal_affected_free_fraction|0.904800319|0.907777396|0.00297707631|4|3|8/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|free_minimum_angle_lt5_fraction|0.142169554|0.142412782|0.000243228583|5|3|8/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|free_near_degenerate_fraction|0|0|0|0|0|8/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|fixed_query_p95_L|0.11787365|0.116551061|-0.00132258831|6|2|8/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|coverage_fraction|0.140686035|0.145324707|0.00463867188|5|3|8/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|GT_to_F_original_RMS_L|0.0680007811|0.06778788|-0.000212901069|5|3|8/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|F_to_GT_original_RMS_L|0.0629091718|0.063155765|0.000246593158|4|4|8/8|
|MOMENT_XY_END-H_CONT_END|C/holdout/0.2|bidirectional_original_RMS_L|0.0665064051|0.0665858982|7.94931127e-05|3|5|8/8|
|MOMENT_XY_END-START|A/train/0.2|Gamma_to_free_edges_RMS_bbox_pct|2.75071435|1.91544634|-0.835268014|8|0|8/8|
|MOMENT_XY_END-START|A/train/0.2|Gamma_to_free_edges_coverage_0.001_fraction|0.0293113784|0.0618776634|0.032566285|6|2|8/8|
|MOMENT_XY_END-START|A/train/0.2|CF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|MOMENT_XY_END-START|A/train/0.2|FF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|MOMENT_XY_END-START|A/train/0.2|CF_proper_affected_free_fraction|0.0736577208|0.0635812242|-0.0100764966|6|1|8/8|
|MOMENT_XY_END-START|A/train/0.2|FF_illegal_affected_free_fraction|0.884851803|0.880136873|-0.00471493054|5|3|8/8|
|MOMENT_XY_END-START|A/train/0.2|free_minimum_angle_lt5_fraction|0.344644305|0.342239403|-0.00240490159|4|4|8/8|
|MOMENT_XY_END-START|A/train/0.2|free_near_degenerate_fraction|0|0|0|0|0|8/8|
|MOMENT_XY_END-START|A/train/0.2|fixed_query_p95_L|0.0784553699|0.0639804532|-0.0144749168|6|2|8/8|
|MOMENT_XY_END-START|A/train/0.2|coverage_fraction|0.586303711|0.627441406|0.0411376953|5|2|8/8|
|MOMENT_XY_END-START|A/train/0.2|GT_to_F_original_RMS_L|0.0423794185|0.0316510341|-0.0107283844|6|2|8/8|
|MOMENT_XY_END-START|A/train/0.2|F_to_GT_original_RMS_L|0.042163594|0.035689769|-0.00647382504|3|5|8/8|
|MOMENT_XY_END-START|A/train/0.2|bidirectional_original_RMS_L|0.0442654759|0.0356146523|-0.00865082357|5|3|8/8|
|MOMENT_XY_END-START|A/train/0.4|Gamma_to_free_edges_RMS_bbox_pct|1.45867841|0.681110225|-0.777568187|6|1|7/8|
|MOMENT_XY_END-START|A/train/0.4|Gamma_to_free_edges_coverage_0.001_fraction|0.0623733626|0.0808882644|0.0185149018|5|2|7/8|
|MOMENT_XY_END-START|A/train/0.4|CF_unresolved_or_degenerate_pair_count|0|0|0|0|0|7/8|
|MOMENT_XY_END-START|A/train/0.4|FF_unresolved_or_degenerate_pair_count|0|0|0|0|0|7/8|
|MOMENT_XY_END-START|A/train/0.4|CF_proper_affected_free_fraction|0.127044934|0.128612789|0.00156785452|3|4|7/8|
|MOMENT_XY_END-START|A/train/0.4|FF_illegal_affected_free_fraction|0.813767863|0.790057368|-0.0237104953|3|3|7/8|
|MOMENT_XY_END-START|A/train/0.4|free_minimum_angle_lt5_fraction|0.411390141|0.42875863|0.0173684889|2|5|7/8|
|MOMENT_XY_END-START|A/train/0.4|free_near_degenerate_fraction|0|0|0|0|0|7/8|
|MOMENT_XY_END-START|A/train/0.4|fixed_query_p95_L|0.0564662299|0.0301670196|-0.0262992103|6|1|7/8|
|MOMENT_XY_END-START|A/train/0.4|coverage_fraction|0.735770089|0.853934152|0.118164062|5|2|7/8|
|MOMENT_XY_END-START|A/train/0.4|GT_to_F_original_RMS_L|0.02549237|0.0143726702|-0.0111196998|5|2|7/8|
|MOMENT_XY_END-START|A/train/0.4|F_to_GT_original_RMS_L|0.0334423563|0.0219629035|-0.0114794528|5|2|7/8|
|MOMENT_XY_END-START|A/train/0.4|bidirectional_original_RMS_L|0.0336910034|0.0202108161|-0.0134801874|4|3|7/8|
|MOMENT_XY_END-START|A/validation/0.2|Gamma_to_free_edges_RMS_bbox_pct|3.25389446|3.6998947|0.446000247|2|6|8/8|
|MOMENT_XY_END-START|A/validation/0.2|Gamma_to_free_edges_coverage_0.001_fraction|0.00356081392|0.00289292998|-0.000667883938|3|3|8/8|
|MOMENT_XY_END-START|A/validation/0.2|CF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|MOMENT_XY_END-START|A/validation/0.2|FF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|MOMENT_XY_END-START|A/validation/0.2|CF_proper_affected_free_fraction|0.0601349117|0.0635169158|0.00338200401|2|4|8/8|
|MOMENT_XY_END-START|A/validation/0.2|FF_illegal_affected_free_fraction|0.926270307|0.911316981|-0.014953326|5|3|8/8|
|MOMENT_XY_END-START|A/validation/0.2|free_minimum_angle_lt5_fraction|0.146756635|0.106654148|-0.0401024868|6|2|8/8|
|MOMENT_XY_END-START|A/validation/0.2|free_near_degenerate_fraction|0|0|0|0|0|8/8|
|MOMENT_XY_END-START|A/validation/0.2|fixed_query_p95_L|0.0688447336|0.0710179676|0.00217323403|2|6|8/8|
|MOMENT_XY_END-START|A/validation/0.2|coverage_fraction|0.322998047|0.344116211|0.0211181641|5|3|8/8|
|MOMENT_XY_END-START|A/validation/0.2|GT_to_F_original_RMS_L|0.0354906758|0.0365229656|0.00103228981|3|5|8/8|
|MOMENT_XY_END-START|A/validation/0.2|F_to_GT_original_RMS_L|0.0379022235|0.0372867138|-0.000615509752|3|5|8/8|
|MOMENT_XY_END-START|A/validation/0.2|bidirectional_original_RMS_L|0.0371301102|0.0374433391|0.000313228891|4|4|8/8|
|MOMENT_XY_END-START|A/validation/0.4|Gamma_to_free_edges_RMS_bbox_pct|3.58848382|3.84771334|0.259229524|2|4|6/8|
|MOMENT_XY_END-START|A/validation/0.4|Gamma_to_free_edges_coverage_0.001_fraction|0.00772880716|0.00791354906|0.000184741904|2|4|6/8|
|MOMENT_XY_END-START|A/validation/0.4|CF_unresolved_or_degenerate_pair_count|0|0|0|0|0|6/8|
|MOMENT_XY_END-START|A/validation/0.4|FF_unresolved_or_degenerate_pair_count|0|0|0|0|0|6/8|
|MOMENT_XY_END-START|A/validation/0.4|CF_proper_affected_free_fraction|0.124005408|0.0936497014|-0.0303557069|4|1|6/8|
|MOMENT_XY_END-START|A/validation/0.4|FF_illegal_affected_free_fraction|0.795531369|0.822788069|0.0272567001|3|3|6/8|
|MOMENT_XY_END-START|A/validation/0.4|free_minimum_angle_lt5_fraction|0.0713944917|0.0606270282|-0.0107674636|3|3|6/8|
|MOMENT_XY_END-START|A/validation/0.4|free_near_degenerate_fraction|0|0|0|0|0|6/8|
|MOMENT_XY_END-START|A/validation/0.4|fixed_query_p95_L|0.0477609319|0.053563402|0.00580247005|3|3|6/8|
|MOMENT_XY_END-START|A/validation/0.4|coverage_fraction|0.489095052|0.469726562|-0.0193684896|5|1|6/8|
|MOMENT_XY_END-START|A/validation/0.4|GT_to_F_original_RMS_L|0.0235015075|0.0255742349|0.00207272738|2|4|6/8|
|MOMENT_XY_END-START|A/validation/0.4|F_to_GT_original_RMS_L|0.0248270135|0.0248268573|-1.56154838e-07|4|2|6/8|
|MOMENT_XY_END-START|A/validation/0.4|bidirectional_original_RMS_L|0.0249682466|0.0259420292|0.000973782597|3|3|6/8|
|MOMENT_XY_END-START|C/holdout/0.2|Gamma_to_free_edges_RMS_bbox_pct|5.53945628|5.38660488|-0.152851402|4|4|8/8|
|MOMENT_XY_END-START|C/holdout/0.2|Gamma_to_free_edges_coverage_0.001_fraction|0.00144493129|0.00299737002|0.00155243873|6|2|8/8|
|MOMENT_XY_END-START|C/holdout/0.2|CF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|MOMENT_XY_END-START|C/holdout/0.2|FF_unresolved_or_degenerate_pair_count|0|0|0|0|0|8/8|
|MOMENT_XY_END-START|C/holdout/0.2|CF_proper_affected_free_fraction|0.10839684|0.102396281|-0.00600055839|4|4|8/8|
|MOMENT_XY_END-START|C/holdout/0.2|FF_illegal_affected_free_fraction|0.907462619|0.907777396|0.000314776278|3|5|8/8|
|MOMENT_XY_END-START|C/holdout/0.2|free_minimum_angle_lt5_fraction|0.155340129|0.142412782|-0.0129273465|5|3|8/8|
|MOMENT_XY_END-START|C/holdout/0.2|free_near_degenerate_fraction|0|0|0|0|0|8/8|
|MOMENT_XY_END-START|C/holdout/0.2|fixed_query_p95_L|0.116751343|0.116551061|-0.000200281322|3|5|8/8|
|MOMENT_XY_END-START|C/holdout/0.2|coverage_fraction|0.134094238|0.145324707|0.0112304688|5|3|8/8|
|MOMENT_XY_END-START|C/holdout/0.2|GT_to_F_original_RMS_L|0.0689005351|0.06778788|-0.00111265512|5|3|8/8|
|MOMENT_XY_END-START|C/holdout/0.2|F_to_GT_original_RMS_L|0.0617709302|0.063155765|0.0013848348|4|4|8/8|
|MOMENT_XY_END-START|C/holdout/0.2|bidirectional_original_RMS_L|0.0663016449|0.0665858982|0.00028425328|4|4|8/8|

A train/validation 的 20%/40% 独立列出；原不可构造 40% 项保留 NA。C 是此前已观察的留出回归面板，不是全新独立测试，也不参与选择。C/N/finite、X/Y/Z span 与 median、Gamma、FF/CF unresolved 和退化角度的所有字段见逐输出与配对 CSV；严格全局 watertight 审计为 NOT_RUN。

coverage_partition.csv 使用相同冻结查询逐点拆成 both_in / enter / leave / both_out，分区贡献为平方距离差总和除原完整 Q，四区精确闭合总 MSE 差。新增覆盖必须同距离尾部、接口和相交代价一起评估。

## 宽度响应与固定目标列

W 为 free 三角角点 X 的 q95−q05。r_base=W(F1)/W(Y1)；b_self=[W(F1.1)−W(F.9)]/[.2W(F1)]；b_GT 的分母换为 .2W(Y1)。每条父/噪声曲线先算 |b−1|，再父内噪声平均、父等权，不能先平均有正负偏差的斜率。

|比较|划分|候选|指标|前值|后值|父均差|配对差中位数|有效/登记|
|---|---|---|---|---|---|---|---|---|
|H_CONT_END-START|train|MODEL|b_self_abs_error|0.357601786|0.317278245|-0.0403235411|-0.0429790173|4/4|
|H_CONT_END-START|train|MODEL|b_GT_abs_error|0.37659858|0.343446158|-0.0331524221|-0.0229492605|4/4|
|H_CONT_END-START|train|MODEL|baseline_abs_ratio_error|0.0479415182|0.0375440923|-0.0103974259|-0.00835985|4/4|
|H_CONT_END-START|train|MODEL|mean_A_column|0.00589272289|0.00652112177|0.000628398887|0.00063197954|4/4|
|H_CONT_END-START|train|MODEL|all_three_strict_win|0.125|0.5|0.375|0.25|4/4|
|H_CONT_END-START|validation|MODEL|b_self_abs_error|0.63863272|0.493957672|-0.144675048|-0.139134329|4/4|
|H_CONT_END-START|validation|MODEL|b_GT_abs_error|0.73187766|0.569586862|-0.162290798|-0.167849465|4/4|
|H_CONT_END-START|validation|MODEL|baseline_abs_ratio_error|0.304713682|0.280225705|-0.024487977|-0.0124317803|4/4|
|H_CONT_END-START|validation|MODEL|mean_A_column|0.000971952147|0.00174960854|0.000777656395|0.000451031203|4/4|
|H_CONT_END-START|validation|MODEL|all_three_strict_win|0|0|0|0|4/4|
|H_CONT_END-START|train|No-response|b_self_abs_error|1|1|0|0|4/4|
|H_CONT_END-START|train|No-response|b_GT_abs_error|1|1|0|0|4/4|
|H_CONT_END-START|train|No-response|baseline_abs_ratio_error|0.0479415182|0.0375440923|-0.0103974259|-0.00835985|4/4|
|H_CONT_END-START|train|No-response|mean_A_column|0|0|0|0|4/4|
|H_CONT_END-START|train|No-response|all_three_strict_win|0|0|0|0|4/4|
|H_CONT_END-START|validation|No-response|b_self_abs_error|1|1|0|0|4/4|
|H_CONT_END-START|validation|No-response|b_GT_abs_error|1|1|0|0|4/4|
|H_CONT_END-START|validation|No-response|baseline_abs_ratio_error|0.304713682|0.280225705|-0.024487977|-0.0124317803|4/4|
|H_CONT_END-START|validation|No-response|mean_A_column|0|0|0|0|4/4|
|H_CONT_END-START|validation|No-response|all_three_strict_win|0|0|0|0|4/4|
|H_CONT_END-START|train|Post-scale|b_self_abs_error|1.52134995e-07|1.52134995e-07|4.16333634e-17|5.55111512e-17|4/4|
|H_CONT_END-START|train|Post-scale|b_GT_abs_error|0.0479414141|0.0375439866|-0.0103974276|-0.00835985032|4/4|
|H_CONT_END-START|train|Post-scale|baseline_abs_ratio_error|0.0479415182|0.0375440923|-0.0103974259|-0.00835985|4/4|
|H_CONT_END-START|train|Post-scale|mean_A_column|0.00862338784|0.00920452676|0.000581138921|0.000575983982|4/4|
|H_CONT_END-START|train|Post-scale|all_three_strict_win|0.75|1|0.25|0.25|4/4|
|H_CONT_END-START|validation|Post-scale|b_self_abs_error|2.61356318e-07|2.61356319e-07|5.55111512e-17|5.55111512e-17|4/4|
|H_CONT_END-START|validation|Post-scale|b_GT_abs_error|0.304713592|0.28022551|-0.0244880817|-0.0124319828|4/4|
|H_CONT_END-START|validation|Post-scale|baseline_abs_ratio_error|0.304713682|0.280225705|-0.024487977|-0.0124317803|4/4|
|H_CONT_END-START|validation|Post-scale|mean_A_column|0.00161645358|0.0018989109|0.000282457326|0.000212336956|4/4|
|H_CONT_END-START|validation|Post-scale|all_three_strict_win|0|0.125|0.125|0|4/4|
|MOMENT_XY_END-H_CONT_END|train|MODEL|b_self_abs_error|0.317278245|0.318691869|0.00141362379|0.00201334328|4/4|
|MOMENT_XY_END-H_CONT_END|train|MODEL|b_GT_abs_error|0.343446158|0.344337127|0.000890968863|0.00103642388|4/4|
|MOMENT_XY_END-H_CONT_END|train|MODEL|baseline_abs_ratio_error|0.0375440923|0.0367322129|-0.000811879406|-0.000762185435|4/4|
|MOMENT_XY_END-H_CONT_END|train|MODEL|mean_A_column|0.00652112177|0.00661576836|9.46465879e-05|9.16543512e-05|4/4|
|MOMENT_XY_END-H_CONT_END|train|MODEL|all_three_strict_win|0.5|0.5|0|0|4/4|
|MOMENT_XY_END-H_CONT_END|validation|MODEL|b_self_abs_error|0.493957672|0.489388831|-0.00456884074|-0.0081945921|4/4|
|MOMENT_XY_END-H_CONT_END|validation|MODEL|b_GT_abs_error|0.569586862|0.5639053|-0.00568156194|-0.0076449162|4/4|
|MOMENT_XY_END-H_CONT_END|validation|MODEL|baseline_abs_ratio_error|0.280225705|0.279150914|-0.00107479132|0.000604853283|4/4|
|MOMENT_XY_END-H_CONT_END|validation|MODEL|mean_A_column|0.00174960854|0.00172305424|-2.65543062e-05|-1.99427769e-05|4/4|
|MOMENT_XY_END-H_CONT_END|validation|MODEL|all_three_strict_win|0|0|0|0|4/4|
|MOMENT_XY_END-H_CONT_END|train|No-response|b_self_abs_error|1|1|0|0|4/4|
|MOMENT_XY_END-H_CONT_END|train|No-response|b_GT_abs_error|1|1|0|0|4/4|
|MOMENT_XY_END-H_CONT_END|train|No-response|baseline_abs_ratio_error|0.0375440923|0.0367322129|-0.000811879406|-0.000762185435|4/4|
|MOMENT_XY_END-H_CONT_END|train|No-response|mean_A_column|0|0|0|0|4/4|
|MOMENT_XY_END-H_CONT_END|train|No-response|all_three_strict_win|0|0|0|0|4/4|
|MOMENT_XY_END-H_CONT_END|validation|No-response|b_self_abs_error|1|1|0|0|4/4|
|MOMENT_XY_END-H_CONT_END|validation|No-response|b_GT_abs_error|1|1|0|0|4/4|
|MOMENT_XY_END-H_CONT_END|validation|No-response|baseline_abs_ratio_error|0.280225705|0.279150914|-0.00107479132|0.000604853283|4/4|
|MOMENT_XY_END-H_CONT_END|validation|No-response|mean_A_column|0|0|0|0|4/4|
|MOMENT_XY_END-H_CONT_END|validation|No-response|all_three_strict_win|0|0|0|0|4/4|
|MOMENT_XY_END-H_CONT_END|train|Post-scale|b_self_abs_error|1.52134995e-07|1.52134995e-07|-1.38777878e-17|0|4/4|
|MOMENT_XY_END-H_CONT_END|train|Post-scale|b_GT_abs_error|0.0375439866|0.036732107|-0.000811879564|-0.000762185661|4/4|
|MOMENT_XY_END-H_CONT_END|train|Post-scale|baseline_abs_ratio_error|0.0375440923|0.0367322129|-0.000811879406|-0.000762185435|4/4|
|MOMENT_XY_END-H_CONT_END|train|Post-scale|mean_A_column|0.00920452676|0.00932202907|0.00011750231|8.16158562e-05|4/4|
|MOMENT_XY_END-H_CONT_END|train|Post-scale|all_three_strict_win|1|1|0|0|4/4|
|MOMENT_XY_END-H_CONT_END|validation|Post-scale|b_self_abs_error|2.61356319e-07|2.61356319e-07|4.16333634e-16|6.10622664e-16|4/4|
|MOMENT_XY_END-H_CONT_END|validation|Post-scale|b_GT_abs_error|0.28022551|0.279150818|-0.00107469259|0.000604853384|4/4|
|MOMENT_XY_END-H_CONT_END|validation|Post-scale|baseline_abs_ratio_error|0.280225705|0.279150914|-0.00107479132|0.000604853283|4/4|
|MOMENT_XY_END-H_CONT_END|validation|Post-scale|mean_A_column|0.0018989109|0.00183932964|-5.95812662e-05|-1.67290566e-05|4/4|
|MOMENT_XY_END-H_CONT_END|validation|Post-scale|all_three_strict_win|0.125|0.125|0|0|4/4|
|MOMENT_XY_END-START|train|MODEL|b_self_abs_error|0.357601786|0.318691869|-0.0389099173|-0.040965674|4/4|
|MOMENT_XY_END-START|train|MODEL|b_GT_abs_error|0.37659858|0.344337127|-0.0322614533|-0.0219128367|4/4|
|MOMENT_XY_END-START|train|MODEL|baseline_abs_ratio_error|0.0479415182|0.0367322129|-0.0112093053|-0.009757333|4/4|
|MOMENT_XY_END-START|train|MODEL|mean_A_column|0.00589272289|0.00661576836|0.000723045475|0.000723633891|4/4|
|MOMENT_XY_END-START|train|MODEL|all_three_strict_win|0.125|0.5|0.375|0.25|4/4|
|MOMENT_XY_END-START|validation|MODEL|b_self_abs_error|0.63863272|0.489388831|-0.149243889|-0.101943897|4/4|
|MOMENT_XY_END-START|validation|MODEL|b_GT_abs_error|0.73187766|0.5639053|-0.16797236|-0.175494381|4/4|
|MOMENT_XY_END-START|validation|MODEL|baseline_abs_ratio_error|0.304713682|0.279150914|-0.0255627684|-0.0172054009|4/4|
|MOMENT_XY_END-START|validation|MODEL|mean_A_column|0.000971952147|0.00172305424|0.000751102089|0.000431088426|4/4|
|MOMENT_XY_END-START|validation|MODEL|all_three_strict_win|0|0|0|0|4/4|
|MOMENT_XY_END-START|train|No-response|b_self_abs_error|1|1|0|0|4/4|
|MOMENT_XY_END-START|train|No-response|b_GT_abs_error|1|1|0|0|4/4|
|MOMENT_XY_END-START|train|No-response|baseline_abs_ratio_error|0.0479415182|0.0367322129|-0.0112093053|-0.009757333|4/4|
|MOMENT_XY_END-START|train|No-response|mean_A_column|0|0|0|0|4/4|
|MOMENT_XY_END-START|train|No-response|all_three_strict_win|0|0|0|0|4/4|
|MOMENT_XY_END-START|validation|No-response|b_self_abs_error|1|1|0|0|4/4|
|MOMENT_XY_END-START|validation|No-response|b_GT_abs_error|1|1|0|0|4/4|
|MOMENT_XY_END-START|validation|No-response|baseline_abs_ratio_error|0.304713682|0.279150914|-0.0255627684|-0.0172054009|4/4|
|MOMENT_XY_END-START|validation|No-response|mean_A_column|0|0|0|0|4/4|
|MOMENT_XY_END-START|validation|No-response|all_three_strict_win|0|0|0|0|4/4|
|MOMENT_XY_END-START|train|Post-scale|b_self_abs_error|1.52134995e-07|1.52134995e-07|2.77555756e-17|2.22044605e-16|4/4|
|MOMENT_XY_END-START|train|Post-scale|b_GT_abs_error|0.0479414141|0.036732107|-0.0112093071|-0.00975733335|4/4|
|MOMENT_XY_END-START|train|Post-scale|baseline_abs_ratio_error|0.0479415182|0.0367322129|-0.0112093053|-0.009757333|4/4|
|MOMENT_XY_END-START|train|Post-scale|mean_A_column|0.00862338784|0.00932202907|0.000698641232|0.000755589439|4/4|
|MOMENT_XY_END-START|train|Post-scale|all_three_strict_win|0.75|1|0.25|0.25|4/4|
|MOMENT_XY_END-START|validation|Post-scale|b_self_abs_error|2.61356318e-07|2.61356319e-07|4.71844785e-16|2.77555756e-16|4/4|
|MOMENT_XY_END-START|validation|Post-scale|b_GT_abs_error|0.304713592|0.279150818|-0.0255627743|-0.0172054065|4/4|
|MOMENT_XY_END-START|validation|Post-scale|baseline_abs_ratio_error|0.304713682|0.279150914|-0.0255627684|-0.0172054009|4/4|
|MOMENT_XY_END-START|validation|Post-scale|mean_A_column|0.00161645358|0.00183932964|0.00022287606|0.00020211351|4/4|
|MOMENT_XY_END-START|validation|Post-scale|all_three_strict_win|0|0.125|0.125|0|4/4|

E[alpha,beta] 行是生成输入，列是评分目标。固定 beta 比较正确输入与另外两输入：A_beta=mean(E[alpha≠beta,beta])−E[beta,beta]。严格胜要求正确输入同时小于两错误输入；平局不算胜。验证共 4 父×2 噪声×3 目标=24 列，每噪声 12 列；全部三列正确/同一输入通吃以 8 条曲线为分母，每噪声 4 条。列不能冒充 24 个独立父。

|标签|划分|候选|噪声|严格列胜/登记|三列全胜/登记|同输入通吃/登记|有效列|平均 A|
|---|---|---|---|---|---|---|---|---|
|START|train|MODEL|all|16/24|1/8|1/8|24|0.00589272289|
|START|train|MODEL|0|9/12|1/4|0/4|12|0.00611272736|
|START|train|MODEL|1|7/12|0/4|1/4|12|0.00567271841|
|START|train|No-response|all|0/24|0/8|0/8|24|0|
|START|train|No-response|0|0/12|0/4|0/4|12|0|
|START|train|No-response|1|0/12|0/4|0/4|12|0|
|START|train|Post-scale|all|21/24|6/8|0/8|24|0.00862338784|
|START|train|Post-scale|0|11/12|3/4|0/4|12|0.00973063328|
|START|train|Post-scale|1|10/12|3/4|0/4|12|0.0075161424|
|START|validation|MODEL|all|8/24|0/8|8/8|24|0.000971952147|
|START|validation|MODEL|0|4/12|0/4|4/4|12|0.000362536621|
|START|validation|MODEL|1|4/12|0/4|4/4|12|0.00158136767|
|START|validation|No-response|all|0/24|0/8|0/8|24|0|
|START|validation|No-response|0|0/12|0/4|0/4|12|0|
|START|validation|No-response|1|0/12|0/4|0/4|12|0|
|START|validation|Post-scale|all|8/24|0/8|7/8|24|0.00161645358|
|START|validation|Post-scale|0|4/12|0/4|4/4|12|0.0016405772|
|START|validation|Post-scale|1|4/12|0/4|3/4|12|0.00159232996|
|H_CONT_END|train|MODEL|all|19/24|4/8|1/8|24|0.00652112177|
|H_CONT_END|train|MODEL|0|10/12|3/4|1/4|12|0.00597588841|
|H_CONT_END|train|MODEL|1|9/12|1/4|0/4|12|0.00706635514|
|H_CONT_END|train|No-response|all|0/24|0/8|0/8|24|0|
|H_CONT_END|train|No-response|0|0/12|0/4|0/4|12|0|
|H_CONT_END|train|No-response|1|0/12|0/4|0/4|12|0|
|H_CONT_END|train|Post-scale|all|24/24|8/8|0/8|24|0.00920452676|
|H_CONT_END|train|Post-scale|0|12/12|4/4|0/4|12|0.00827971331|
|H_CONT_END|train|Post-scale|1|12/12|4/4|0/4|12|0.0101293402|
|H_CONT_END|validation|MODEL|all|9/24|0/8|6/8|24|0.00174960854|
|H_CONT_END|validation|MODEL|0|4/12|0/4|4/4|12|0.00147058951|
|H_CONT_END|validation|MODEL|1|5/12|0/4|2/4|12|0.00202862757|
|H_CONT_END|validation|No-response|all|0/24|0/8|0/8|24|0|
|H_CONT_END|validation|No-response|0|0/12|0/4|0/4|12|0|
|H_CONT_END|validation|No-response|1|0/12|0/4|0/4|12|0|
|H_CONT_END|validation|Post-scale|all|10/24|1/8|7/8|24|0.0018989109|
|H_CONT_END|validation|Post-scale|0|4/12|0/4|4/4|12|0.00169878291|
|H_CONT_END|validation|Post-scale|1|6/12|1/4|3/4|12|0.0020990389|
|MOMENT_XY_END|train|MODEL|all|19/24|4/8|1/8|24|0.00661576836|
|MOMENT_XY_END|train|MODEL|0|10/12|3/4|1/4|12|0.00610518783|
|MOMENT_XY_END|train|MODEL|1|9/12|1/4|0/4|12|0.0071263489|
|MOMENT_XY_END|train|No-response|all|0/24|0/8|0/8|24|0|
|MOMENT_XY_END|train|No-response|0|0/12|0/4|0/4|12|0|
|MOMENT_XY_END|train|No-response|1|0/12|0/4|0/4|12|0|
|MOMENT_XY_END|train|Post-scale|all|24/24|8/8|0/8|24|0.00932202907|
|MOMENT_XY_END|train|Post-scale|0|12/12|4/4|0/4|12|0.00836198756|
|MOMENT_XY_END|train|Post-scale|1|12/12|4/4|0/4|12|0.0102820706|
|MOMENT_XY_END|validation|MODEL|all|9/24|0/8|5/8|24|0.00172305424|
|MOMENT_XY_END|validation|MODEL|0|4/12|0/4|3/4|12|0.00146033526|
|MOMENT_XY_END|validation|MODEL|1|5/12|0/4|2/4|12|0.00198577321|
|MOMENT_XY_END|validation|No-response|all|0/24|0/8|0/8|24|0|
|MOMENT_XY_END|validation|No-response|0|0/12|0/4|0/4|12|0|
|MOMENT_XY_END|validation|No-response|1|0/12|0/4|0/4|12|0|
|MOMENT_XY_END|validation|Post-scale|all|10/24|1/8|7/8|24|0.00183932964|
|MOMENT_XY_END|validation|Post-scale|0|4/12|0/4|4/4|12|0.00168702925|
|MOMENT_XY_END|validation|Post-scale|1|6/12|1/4|3/4|12|0.00199163003|

完整 3×3 单元和每 beta/每噪声列配对见 cross_matrix.csv、target_column_audit.csv、target_column_noise_summary.csv。No-response 与 Post-scale 分别使用各自检查点的 F1；Post-scale 比例只用 C1 与 C_alpha 拟合。解析参照使用编辑前 C/输出，不等同当前 C-only 模型输入合同，不计新模型生成。No-response 固定列应为 A=0、严格胜=0；同列方向正控结果见 tests/v2_scoring_cpu_tests.json。

本报告控制信号筛查要求正确列/三列全胜出现增益、同输入通吃不增加，且逐曲线绝对斜率误差不增加并在两噪声保持方向一致；不是单独看平均 A 或平均斜率。质量退化项逐项披露，不将任何数值容差当成允许质量损失的阈值。

## 七个问题与下一步

1. MOMENT 是否超过等预算 H_CONT：开发性几何候选=False；主终点差=4.72222232e-05，配对差中位数=3.46263117e-05，改善=3/8；两噪声主终点同向改善=False。
2. 是否保留 START：MOMENT−START 主终点差=0.000112292227；完整 GT→F、F→GT 及质量变化单列，不以相对续训胜出代替起点保真。
3. 是否改善条件正确性：控制一致性筛查=False；两噪声控制一致=False。严格列、三列全胜、同输入通吃和绝对响应误差共同判断。
4. 质量代价：MOMENT−H_CONT 各独立 A/C 分组内可见守门项退化 15 项。此数不是独立实验次数，也不声称统计显著；小角分布与 GT 参照需结合相交/退化解释。
5. padding 修复是否有证据：实际 same_shape_padding、variable_padding、零梯度及逐层定位以上表/原表为准；新归一化接受与旧逐元素 flags 并列，缺失为 NA。
6. 等价优化成本：每任务 forward 与 F+B 的实测重复、范围、M/H 比及旧实现参照均已列出；门禁=PASS，不能用尚未测到的值作性能结论。
7. 优先下一步：接受当前有限预算内未见清晰学习收益；下一步优先离线定位逐父失败模式，检验是否存在一致、可复现的关系通道受益子集。 检验假说：若通道有稳定收益，同一父的两噪声应出现一致的主终点与正确目标列改善；否则当前证据支持暂不扩大训练。

本轮仅一对训练随机流的有限预算 pilot。训练 loss、成功生成、宽度变化或通道梯度非零不能单独证明几何收益。固定任务结束即停止；不会自动扩大预算、改 loss、改 λ、切换默认模型或启动下一实验。

## 可复核产物

- paired_results/per_output.csv、per_parent.csv、paired_per_output.csv、paired_per_parent.csv：全 432 登记行及逐父配对。
- primary_paired_parents.csv、primary_noise_pairs.csv：8 父×两噪声的原始前值、后值与差值。
- width_per_output.csv、width_per_curve.csv、noise_pairs_per_parent.csv：模型与两个解析参照的完整曲线。
- cross_matrix.csv、target_column_audit.csv、target_column_noise_pairs.csv：实际行列方向及所有目标列。
- quality_paired_summary.csv、coverage_partition.csv：双方向实值、覆盖与尾部分区、接口及相交代价。
- numerical_comparisons.csv、precheck_cost_summary.csv：实际数值合同和成本重复；不存在额外 I/P 矩阵。
- evaluation/<label>/scoring/scores/：原实际评分明细和身份；source_code/、registration/：评分源码、协议及来源 hash。
