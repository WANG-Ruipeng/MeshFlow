# Native T1：纯GPU几何引导与保护已知面的焊接验证

状态：**PASS_EXECUTION**。Raw引导与固定W后处理分层比较；所有统计来自已保存输出，不根据图片挑噪声。
9201–9204是已反复使用的开发噪声；确认9401–9408只检验两个训练父对象上的新噪声稳健性。L为已有完整对象bbox元数据，尚未验证只有C且未知完整尺度的部署。

新增训练更新、模型参数更新和主干反向的实际计数见账本；本轮没有上下文Guidance、边界embedding、神经去噪器或Phase3。

## 1. 形状能量只降数值，还是改善实际面质量且保持几何？

实测EA下降：P0 0.0030→0.0027；P1 0.0410→0.0133。<5°比例P0只下降0.25个百分点、P1下降2个百分点，两个父对象FF均增加；面形状有改善，几何联合目标未达到。P0(4条)：R12 1.6109→1.6124%L；窄角面10.50→10.25%；CF面19.25→19.00%；FF非法pairs 122.2500→127.5000；P1(4条)：R12 0.7557→0.7637%L；窄角面15.75→13.75%；CF面13.75→14.25%；FF非法pairs 97.0000→103.7500。q、<5°面比例、angle p05、真实退化和R12分别列出；EA降低不替代实际质量或几何保持。面积下限触发不等于实际退化。

## 2. 顶点核和7点核的盲区；GR是否减少真实相交？

GR的ER下降：P0 0.0850→0.0743；P1 0.1000→0.0891，但两父CF穿刺面率、FF非法pair和整段边界RMS均上升。核能量下降没有转化为真实穿刺减少。P0(4条)：R12 1.6109→1.6171%L；窄角面10.50→10.75%；CF面19.25→21.25%；FF非法pairs 122.2500→129.5000；P1(4条)：R12 0.7557→0.7552%L；窄角面15.75→15.25%；CF面13.75→16.25%；FF非法pairs 97.0000→108.0000。三个顶点远离有限面时顶点核会漏掉面内部穿刺；7个固定样点仍只是一种离散表面偏置，不能证明所有交线都被捕获。恰在表面时核最大而梯度可为零，合法接触也可能被排斥；实际探针及GT相容性记录另列。GR没有CPU整面分离或验收门禁。

## 3. GAR组合是否更好，是否有相反作用或代价？

GAR也未过门槛。80个引导状态的∥gA∥/∥gR∥均值1.2294/0.9979；53个定义余弦均值0.0104、范围[-0.0420,0.1412]，负值17个。局部反向不足以把失败唯一归因于梯度冲突；两父CF比GR更高，FF也未优于G0。P0(4条)：R12 1.6109→1.6187%L；窄角面10.50→10.50%；CF面19.25→22.50%；FF非法pairs 122.2500→126.7500；P1(4条)：R12 0.7557→0.7557%L；窄角面15.75→14.50%；CF面13.75→16.75%；FF非法pairs 97.0000→108.7500。下表同时保留GA/GR/GAR，合梯度仅共用一次rho；逐步两项梯度范数和余弦由GPU诊断记录，不能预设正交或协同。

## 4. W在原T1上解决了多少连接问题？

W合并了部分近邻，但8个G0+W均未闭合且顶点link均失败；P0/P1非流形顶点均值从0/2升到47.25/59.5。W全部32输出出现14个跨面重复（分布12输出），单面重复index和实际退化为0。不能把局部门禁通过说成全局修复。P0原T1经W：顶点数均值312.0000→234.7500，单关联边312.0000→296.2500，共边连通分量101.0000→93.5000；边界RMS 1.0179→1.0158%L，FF非法pairs 122.2500→103.0000；P1原T1经W：顶点数均值312.0000→214.0000，单关联边314.0000→290.2500，共边连通分量101.0000→90.0000；边界RMS 0.2298→0.2274%L，FF非法pairs 97.0000→75.0000。减少顶点数、边界边或组件可说明部分坐标错位被合并，不能自动说明一般T-junction、缺失面连接或非法相交已解决；顶点link、边关联和完整静态相交独立检查。没有删面、补洞或重三角化。

## 5. Guidance+W是否优于同样W处理后的G0？

三种Guidance+W在两个父对象上的CF穿刺面率与FF非法pair均高于G0+W，没有建立联合优势。GA+W：P0(4条)：R12 1.6113→1.6112%L；窄角面10.50→10.00%；CF面17.50→18.25%；FF非法pairs 103.0000→104.7500；P1(4条)：R12 0.7566→0.7581%L；窄角面16.00→14.25%；CF面12.75→13.75%；FF非法pairs 75.0000→82.2500；GR+W：P0(4条)：R12 1.6113→1.6160%L；窄角面10.50→10.50%；CF面17.50→21.00%；FF非法pairs 103.0000→106.5000；P1(4条)：R12 0.7566→0.7639%L；窄角面16.00→15.50%；CF面12.75→16.25%；FF非法pairs 75.0000→85.2500；GAR+W：P0(4条)：R12 1.6113→1.6180%L；窄角面10.50→10.25%；CF面17.50→22.75%；FF非法pairs 103.0000→107.0000；P1(4条)：R12 0.7566→0.7563%L；窄角面16.00→14.25%；CF面12.75→16.50%；FF非法pairs 75.0000→85.2500。这里基准始终是G0+W；没有把W自身收益归给Guidance。

## 6. 最终C、面数、闭合、顶点流形和无相交分别如何？

64/64个Raw/W端点及OBJ保持C，112面/free100；64/64均检出非法相交，组合闭合通过0。Raw顶点link通过16/32（允许边界），W通过0/32。T-junction穷举0仅表示未检出该容差下顶点落在非端点边上的狭义情况，不证明闭合。逐输出三标签与拓扑状态见后表。C坐标评分、生成51态保持及OBJ读回保持分开；面数固定112/100。所有边关联数为2不替代顶点link和相交审计。NO_ILLEGAL_TRIANGLE_INTERSECTIONS_DETECTED只指声明容差下的静态完整pair检查，CCD仍NOT_RUN。

## 7. 新噪声复现了吗？真实额外成本是多少？

三组均未过Raw门槛，9401–9408没有生成，不能宣称新噪声复现。GA完整生成2.0529s，GPU几何前向+反向0.0277s；GR完整生成2.0234s，GPU几何前向+反向0.0398s；GAR完整生成2.0876s，GPU几何前向+反向0.0680s（均值）。G0仅两次重放，不能作稳定加速比。确认状态：NOT_RUN；唯一选择：None。未选中候选时不执行新噪声，不强选不合格的最好组。。模型、GPU几何前向/坐标反向、D2H、W/导出/离线审计分列；未预设任何毫秒或显存指标。

## 8. 下一步证据支持什么？

建议停止在当前参数下扩大GA/GR/GAR修复验证：实现与确定性通过，几何联合收益未建立。先定位合法接触与整体拓扑约束的表达；其他训练支持/模块的收益仍未测试，本轮不自动转向。依预登记Raw门槛判断本轮GPU能量是否值得独立验证，不能靠+W图或单个噪声改选。若未通过，说明当前固定参数下未建立实用收益；边界标记、训练支持的上下文Guidance和神经细化均未在本轮测试，不能宣称已经胜出。到本轮预算或门禁停止，不新增训练、改阈值或修补到过线。

## 实际入口、版本与冻结门禁

| 实测项 | 记录 |
| --- | --- |
| protocol SHA256 | 8afba144e65d582298c6c91f5dbe2c37ba975d3eafb1e75c53aa7653f33b000c |
| checkpoint SHA256 | 7928d77507bf1fe610876f61b5b11e614263323b5fc0f595cc6740ffd24af634 |
| strict load | PASS |
| missing / unexpected | [] / [] |
| 参数总数 / 可训练数 | 130251267 / 0 |
| state before | 37e31fe5925b8f870b9a4d3bbbbc84edc7bf30177946a6f67f4529a7f8e97172 |
| state after | 37e31fe5925b8f870b9a4d3bbbbc84edc7bf30177946a6f67f4529a7f8e97172 |
| state hash equality | PASS |
| 原入口与预校验入口逐位一致 | {'status': 'PASS', 'bitwise': True, 'extra_model_forwards': 2} |
| 几何图仅当前free叶子 | {'status': 'PASS', 'leaf_count': 1, 'only_leaf': 'Detached current free Y', 'C_requires_grad': False, 'model_graph': False} |
| 实际运行时配置 | {'cublas_workspace_config': ':4096:8', 'deterministic_algorithms': True, 'deterministic_warn_only': False, 'tf32_matmul': False, 'tf32_cudnn': False, 'cudnn_benchmark': False, 'external_flash_available': False, 'sdpa_math_enabled': True, 'sdpa_flash_enabled': False, 'sdpa_mem_efficient_enabled': False, 'sdpa_cudnn_enabled': False, 'cuda_autocast_enabled': True, 'cuda_autocast_dtype': 'torch.bfloat16', 'gradient_enabled': False, 'parameter_dtype': 'torch.float32', 'parameter_elements_checked': 130251267, 'coordinate_tensors_checked': 1, 'active_context_depth': 0, 'native_active_context_depth': 1, 'native_precision': 'bf16', 'all_modules_eval': True, 'gradient_checkpointing': False, 'trainable_parameters': 0, 'all_backbone_parameters_trainable': False, 'role_parameters': 1536, 'frozen_backbone_guard_used': False} |
| Git起始状态 | ?? experiments/ |
| 资源观测 | {'GPU': 'NVIDIA GeForce RTX5080', 'driver': '591.86', 'total_MiB': 16303, 'used_MiB': 4096, 'utilization_pct': 1, 'observed_local_time': '2026-09-29T16:03:51+08:00', 'source': 'nvidia-smi query before init, transcribed into init record'} |
| 实际环境版本 | {'python': '3.10.21', 'torch': '2.7.1+cu128', 'torch_CUDA': '12.8', 'numpy': '1.26.4', 'scipy': '1.15.3', 'platform': 'Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43', 'source': 'Read-only version query after completed experiment; no model invocation'} |

| G0重放父 | seed | 51态逐位一致 | 状态 | 完整生成秒 |
| --- | --- | --- | --- | --- |
| 0 | 9201 | PASS | PASS | 1.8999 |
| 1 | 9201 | PASS | PASS | 2.1412 |

引导重复：51态逐位一致=True；GPU诊断逐位一致=True。

| 保护范围 | 状态 | 文件数 | 变化/失败 |
| --- | --- | --- | --- |
| critical_files_after | PASS | 25 | [] |
| historical_metadata_after | PASS | 850 | [] |
| new_core_source_check | PASS | 5 | [] |

模型、积分和C沿用原FP32坐标；仅几何能量临时使用Y=X/L、C/L，位移再乘L返回原坐标。几何反向只针对当前free坐标。C逐位复写属于构造约束，不能被解释为模型学习到了精确已知面。历史文件保护、冻结模型状态和原入口复现分别核对。

## 数学、数值与GT相容性探针

固定能量探针状态：PASS；保护W探针状态：PASS；离线评分helper：PASS。每个固定探针的CPU FP64/实际GPU FP32值、有限差分、刚体/尺度/排列核验、零梯度和合法接触结果都保留在JSON与fixed_probes.npz，未选择性删去失败样例。

| 训练对象GT | EA | ER | ∥gA∥ | ∥gR∥ | q>10面 | floor面 | 共享顶点槽 | 共享槽gR RMS/max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 0.0000 | 0.1171 | 0.0000 | 6.3179e-07 | 0 | 0 | 70 | 7.5370e-08/3.1409e-07 |
| P1 | 0.0025 | 0.0928 | 1.5745 | 0.5586 | 8 | 0 | 48 | 0.0546/0.2384 |

| 固定反例 | 探针状态 | 顶点核E | 七点ER | 七点梯度范数 | 独立判交 |
| --- | --- | --- | --- | --- | --- |
| 顶点漏检、七点激活 | PASS_COUNTEREXAMPLE | 1.1254e-07 | 0.3099 | 7.4347 | proper_transverse_penetration |
| 两种固定点库均近零响应 | PASS_COUNTEREXAMPLE | 0.0000 | 1.1558e-299 | 0.0000 | proper_transverse_penetration |
| 恰在表面：能量最大、梯度零 | PASS_COUNTEREXAMPLE | NOT_RUN / N/A | 1.0000 | 0.0000 | NOT_RUN / N/A |
| 合法共享边也受力 | PASS_COUNTEREXAMPLE | NOT_RUN / N/A | 0.8574 | 53.8472 | shared_edge_contact |
| 合法90度折角也受力 | PASS_COUNTEREXAMPLE | NOT_RUN / N/A | 0.8804 | 52.4680 | shared_edge_contact |

反例PASS表示成功复现已声明局限，并非几何修复成功。两种点库漏检例在FP64中的极小值也可能在FP32下下溢为零。等边穿刺例EA=0；重合退化例在数值下限政策下也不能依赖EA复活。降低核能量后仍穿刺的独立反例已保留。

GT探针只检查势能是否会偏离合法原始几何，GT及其梯度绝不传入采样器或W。离面核并不区分非法穿刺与合法接触；共享位置处单个距离项导数为零，也不保证与该点相连整面的总梯度为零。

## 实际端点能量、相交面质量与GPU梯度

| 组/层/父 | EA GPU均值 | ER GPU均值 | 角度p05均值 | 非法free面均值 | 其中q≤10且非近退化 | 非法面q中位/p95均值 | 非法面面积/L²中位/p05均值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G0/Raw/P0 | 0.0030 | 0.0850 | 5.2631 | 93.0000 | 92.0000 | 2.0268/7.3459 | 0.0083/0.0011 |
| G0/Raw/P1 | 0.0410 | 0.1000 | 4.7634 | 91.2500 | 84.5000 | 2.2329/10.6482 | 0.0047/0.0001 |
| G0/W/P0 | 0.0030 | 0.0851 | 5.2679 | 88.0000 | 87.0000 | 2.1101/7.4965 | 0.0083/0.0011 |
| G0/W/P1 | 0.0409 | 0.1000 | 4.7227 | 81.2500 | 74.2500 | 2.2192/11.9420 | 0.0043/0.0001 |
| GA/Raw/P0 | 0.0027 | 0.0850 | 5.3108 | 95.7500 | 95.2500 | 2.0792/7.4712 | 0.0082/0.0011 |
| GA/Raw/P1 | 0.0133 | 0.0996 | 4.9013 | 89.5000 | 83.2500 | 2.2450/10.7978 | 0.0044/0.0002 |
| GA/W/P0 | 0.0027 | 0.0851 | 5.3049 | 89.7500 | 89.2500 | 2.0907/7.3963 | 0.0083/0.0011 |
| GA/W/P1 | 0.0129 | 0.0997 | 4.8541 | 83.2500 | 77.2500 | 2.2628/10.6912 | 0.0043/0.0001 |
| GR/Raw/P0 | 0.0027 | 0.0743 | 5.2677 | 94.7500 | 93.7500 | 2.0572/7.4462 | 0.0083/0.0011 |
| GR/Raw/P1 | 0.0417 | 0.0891 | 4.8156 | 91.7500 | 84.5000 | 2.2504/11.5286 | 0.0044/0.0001 |
| GR/W/P0 | 0.0027 | 0.0742 | 5.2495 | 90.0000 | 89.0000 | 2.1129/7.4479 | 0.0084/0.0011 |
| GR/W/P1 | 0.0415 | 0.0892 | 4.7460 | 86.2500 | 79.7500 | 2.2432/11.5149 | 0.0052/0.0001 |
| GAR/Raw/P0 | 0.0026 | 0.0742 | 5.3274 | 94.5000 | 94.0000 | 2.0771/7.4441 | 0.0083/0.0011 |
| GAR/Raw/P1 | 0.0164 | 0.0894 | 4.8364 | 92.2500 | 86.2500 | 2.2261/10.6324 | 0.0051/0.0002 |
| GAR/W/P0 | 0.0026 | 0.0743 | 5.3285 | 91.7500 | 91.2500 | 2.1149/7.4431 | 0.0084/0.0011 |
| GAR/W/P1 | 0.0164 | 0.0897 | 4.8507 | 84.7500 | 79.0000 | 2.2302/10.8360 | 0.0043/0.0001 |

这是每条输出分布统计的噪声均值，不是把所有面混池计算。G0 Raw中P0/P1平均93/91.25张非法面，其中92/84.5张q≤10且非近退化；相交并非全由极差小面构成，也没有安全删面的证据。

| 阶段/组 | 状态数 | ∥gA∥/∥gR∥均值 | 合梯度均值 | 定义余弦n/均值/范围 | 负余弦数 | 截断顶点均值 |
| --- | --- | --- | --- | --- | --- | --- |
| development/GA | 80 | 1.2634/NOT_EVALUATED | 1.2634 | 0/NOT_RUN / N/A/[NOT_RUN / N/A,NOT_RUN / N/A] | 0 | 2.7125 |
| development/GR | 80 | NOT_EVALUATED/0.9975 | 0.9975 | 0/NOT_RUN / N/A/[NOT_RUN / N/A,NOT_RUN / N/A] | 0 | 35.2875 |
| development/GAR | 80 | 1.2294/0.9979 | 1.9136 | 53/0.0104/[-0.0420,0.1412] | 17 | 37.2875 |

GA未评估R项、GR未评估A项；原始日志的占位0不被当作能量为零的观测。余弦只汇总定义的状态，夹角不足以直接预测整体网格或积分终点效果。

三标签分层计数：`{"development/Raw": {"outputs": 32, "counts": {"C_PRESERVED": {"PASS": 32}, "C_indexing": {"PASS_BITWISE_WITH_CORNER_CORRESPONDENCE": 32}, "C_OBJ": {"PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE": 32}, "COMBINATORIAL_CLOSURE_STATUS": {"OPEN_BOUNDARY": 16, "NONMANIFOLD_OPEN": 16}, "EMBEDDED_GEOMETRY_VALIDITY_STATUS": {"FAIL_DETECTED_ILLEGAL_INTERSECTIONS": 32}, "vertex_link_status": {"PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY": 16, "FAIL_NONMANIFOLD_OR_UNUSED_VERTEX": 16}}}, "development/W": {"outputs": 32, "counts": {"C_PRESERVED": {"PASS": 32}, "C_indexing": {"PASS_BITWISE_WITH_CORNER_CORRESPONDENCE": 32}, "C_OBJ": {"PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE": 32}, "COMBINATORIAL_CLOSURE_STATUS": {"NONMANIFOLD_OPEN": 20, "INVALID_REPEATED_INDEX_OR_DUPLICATE_FACE": 12}, "EMBEDDED_GEOMETRY_VALIDITY_STATUS": {"FAIL_DETECTED_ILLEGAL_INTERSECTIONS": 32}, "vertex_link_status": {"FAIL_NONMANIFOLD_OR_UNUSED_VERTEX": 32}}}}`。

## 参数、资产与实现范围

GA：q=三边平方和/(4√3面积)，q_cut=10；规范面积分母下限1e-10，log的q下限1e-12。另报精确零面积、规范面积≤1e-12近退化、面积<1e-10触发下限，三者不混为一谈。
GR：有限C三角形距离，3顶点+3边中点+质心等权7点，sigma=.005；min为分段可微，不是光滑碰撞屏障。k40–49才开启，dt=.02，lambda=1，单顶点截断rho=.001L；GAR合梯度共用上限。W容差=.001L，保护C坐标，不经GT选择锚点/代表点，不删面、补洞或改变面剖分。

[冻结参数](protocol.json)、[确认前候选选择记录](selection.json)、[实际代码diff](code.diff)及[唯一结果JSON](results.json)。没有选择阶段或执行中断时，相应文件/字段以实际记录为准，不宣称已通过。

## Raw与同一W分层结果

| 阶段/组/层/父 | n | R12均值/中位 %L | CF面% | CF对/共面 | FF非法/未定 | Gamma RMS/p95 %L | 覆盖% | <5°面% | q>10面% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development/G0/Raw/P0 | 4 | 1.6109/1.4229 | 19.2500 | 25.0000/0.0000 | 122.2500/0.0000 | 1.0179/1.5156 | 48.2021 | 10.5000 | 1.2500 |
| development/G0/Raw/P1 | 4 | 0.7557/0.9633 | 13.7500 | 20.2500/0.0000 | 97.0000/0.0000 | 0.2298/0.5334 | 53.4162 | 15.7500 | 7.2500 |
| development/G0/W/P0 | 4 | 1.6113/1.4224 | 17.5000 | 23.5000/0.0000 | 103.0000/0.0000 | 1.0158/1.5164 | 46.6366 | 10.5000 | 1.2500 |
| development/G0/W/P1 | 4 | 0.7566/0.9640 | 12.7500 | 18.0000/0.0000 | 75.0000/0.0000 | 0.2274/0.5331 | 54.1510 | 16.0000 | 7.7500 |
| development/GA/Raw/P0 | 4 | 1.6124/1.4232 | 19.0000 | 24.5000/0.0000 | 127.5000/0.0000 | 1.0181/1.5112 | 48.1560 | 10.2500 | 0.7500 |
| development/GA/Raw/P1 | 4 | 0.7637/0.9789 | 14.2500 | 21.7500/0.0000 | 103.7500/0.0000 | 0.2274/0.5142 | 48.2394 | 13.7500 | 6.5000 |
| development/GA/W/P0 | 4 | 1.6112/1.4229 | 18.2500 | 24.2500/0.0000 | 104.7500/0.0000 | 1.0147/1.5190 | 49.0716 | 10.0000 | 0.7500 |
| development/GA/W/P1 | 4 | 0.7581/0.9652 | 13.7500 | 20.2500/0.0000 | 82.2500/0.0000 | 0.2257/0.5140 | 48.7064 | 14.2500 | 6.2500 |
| development/GR/Raw/P0 | 4 | 1.6171/1.4236 | 21.2500 | 30.0000/0.0000 | 129.5000/0.0000 | 1.0515/1.5933 | 35.1399 | 10.7500 | 1.2500 |
| development/GR/Raw/P1 | 4 | 0.7552/0.9618 | 16.2500 | 22.7500/0.0000 | 108.0000/0.0000 | 0.2825/0.6444 | 41.5514 | 15.2500 | 7.5000 |
| development/GR/W/P0 | 4 | 1.6160/1.4228 | 21.0000 | 29.5000/0.0000 | 106.5000/0.0000 | 1.0516/1.5970 | 35.0957 | 10.5000 | 1.2500 |
| development/GR/W/P1 | 4 | 0.7639/0.9778 | 16.2500 | 23.5000/0.0000 | 85.2500/0.0000 | 0.2830/0.6518 | 42.4261 | 15.5000 | 6.7500 |
| development/GAR/Raw/P0 | 4 | 1.6187/1.4244 | 22.5000 | 31.2500/0.0000 | 126.7500/0.0000 | 1.0526/1.5944 | 35.3802 | 10.5000 | 0.7500 |
| development/GAR/Raw/P1 | 4 | 0.7557/0.9624 | 16.7500 | 25.2500/0.0000 | 108.7500/0.0000 | 0.2857/0.6514 | 40.1614 | 14.5000 | 6.7500 |
| development/GAR/W/P0 | 4 | 1.6180/1.4236 | 22.7500 | 30.5000/0.0000 | 107.0000/0.0000 | 1.0494/1.5888 | 35.4791 | 10.2500 | 0.7500 |
| development/GAR/W/P1 | 4 | 0.7563/0.9615 | 16.5000 | 25.2500/0.0000 | 85.2500/0.0000 | 0.2835/0.6504 | 40.7416 | 14.2500 | 6.5000 |

CF/FF主计数保持原primary分类口径，同时完整保存independent与verified分类；有分歧或近退化时UNRESOLVED不被归为无交。CF百分比分母为100张free面，pair数另列；整段Gamma coverage阈值1e-3L，p95按长度加权。

## 逐输出质量、几何与三标签

| 阶段/组/层/父/seed | R12 %L | free→GT / GT→free | 全表面 %L | q中位/p95 | 面积floor/零/近退化 | C | 组合闭合 | 嵌入几何 | vertex link |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development/G0/Raw/P0/9201 | 3.5064 | 3.3010/3.7004 | 2.7403 | 2.1627/8.8073 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/G0/Raw/P0/9202 | 0.0913 | 0.0886/0.0940 | 0.0829 | 2.0989/7.2010 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/G0/Raw/P0/9203 | 0.1642 | 0.1352/0.1889 | 0.1664 | 2.0445/7.0772 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/G0/Raw/P0/9204 | 2.6815 | 3.7892/0.1505 | 0.0915 | 2.0325/7.2741 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/G0/Raw/P1/9201 | 0.0739 | 0.0681/0.0794 | 0.0598 | 2.2557/10.0374 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/G0/Raw/P1/9202 | 1.0222 | 0.9239/1.1119 | 0.6156 | 2.2670/10.3241 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/G0/Raw/P1/9203 | 0.9097 | 1.1206/0.6321 | 0.3184 | 2.2455/10.4234 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/G0/Raw/P1/9204 | 1.0168 | 1.4193/0.2313 | 0.1536 | 2.2421/11.2909 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/G0/W/P0/9201 | 3.5105 | 3.3098/3.7004 | 2.7379 | 2.1627/8.8073 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/G0/W/P0/9202 | 0.0899 | 0.0882/0.0917 | 0.0810 | 2.0910/7.1606 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/G0/W/P0/9203 | 0.1638 | 0.1352/0.1880 | 0.1656 | 2.0433/7.0659 | 0/0/0 | PASS | INVALID_REPEATED_INDEX_OR_DUPLICATE_FACE | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/G0/W/P0/9204 | 2.6810 | 3.7885/0.1504 | 0.0895 | 2.0299/7.2794 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/G0/W/P1/9201 | 0.0771 | 0.0712/0.0826 | 0.0638 | 2.2641/10.0374 | 0/0/0 | PASS | INVALID_REPEATED_INDEX_OR_DUPLICATE_FACE | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/G0/W/P1/9202 | 1.0211 | 0.9212/1.1121 | 0.6145 | 2.2634/10.2925 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/G0/W/P1/9203 | 0.9094 | 1.1205/0.6315 | 0.3181 | 2.2519/10.3494 | 0/0/0 | PASS | INVALID_REPEATED_INDEX_OR_DUPLICATE_FACE | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/G0/W/P1/9204 | 1.0186 | 1.4217/0.2318 | 0.1552 | 2.2306/11.6400 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GA/Raw/P0/9201 | 3.5116 | 3.3124/3.7000 | 2.7349 | 2.1639/8.7482 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/GA/Raw/P0/9202 | 0.0919 | 0.0893/0.0944 | 0.0836 | 2.1012/7.2702 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/GA/Raw/P0/9203 | 0.1642 | 0.1352/0.1889 | 0.1664 | 2.0445/7.0772 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/GA/Raw/P0/9204 | 2.6821 | 3.7901/0.1499 | 0.0896 | 2.0301/7.2786 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/GA/Raw/P1/9201 | 0.0736 | 0.0680/0.0788 | 0.0600 | 2.2555/9.6118 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GA/Raw/P1/9202 | 1.0232 | 0.9252/1.1126 | 0.6157 | 2.2649/9.4699 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GA/Raw/P1/9203 | 0.9419 | 1.1727/0.6318 | 0.3188 | 2.2470/10.1658 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GA/Raw/P1/9204 | 1.0160 | 1.4181/0.2311 | 0.1524 | 2.2396/12.7875 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GA/W/P0/9201 | 3.5114 | 3.3121/3.7000 | 2.7355 | 2.1639/8.7482 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GA/W/P0/9202 | 0.0878 | 0.0850/0.0905 | 0.0797 | 2.1003/7.1930 | 0/0/0 | PASS | INVALID_REPEATED_INDEX_OR_DUPLICATE_FACE | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GA/W/P0/9203 | 0.1638 | 0.1352/0.1880 | 0.1656 | 2.0433/7.0659 | 0/0/0 | PASS | INVALID_REPEATED_INDEX_OR_DUPLICATE_FACE | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GA/W/P0/9204 | 2.6820 | 3.7899/0.1508 | 0.0885 | 2.0314/7.2858 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GA/W/P1/9201 | 0.0793 | 0.0736/0.0847 | 0.0660 | 2.2497/9.0269 | 0/0/0 | PASS | INVALID_REPEATED_INDEX_OR_DUPLICATE_FACE | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GA/W/P1/9202 | 1.0227 | 0.9235/1.1131 | 0.6142 | 2.2662/8.8508 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GA/W/P1/9203 | 0.9412 | 1.1722/0.6307 | 0.3167 | 2.2429/10.1825 | 0/0/0 | PASS | INVALID_REPEATED_INDEX_OR_DUPLICATE_FACE | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GA/W/P1/9204 | 0.9891 | 1.3796/0.2307 | 0.1511 | 2.2359/12.7957 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GAR/Raw/P0/9201 | 3.5117 | 3.3136/3.6992 | 2.7449 | 2.1720/8.6587 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/GAR/Raw/P0/9202 | 0.1142 | 0.1113/0.1171 | 0.1008 | 2.1003/7.2239 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/GAR/Raw/P0/9203 | 0.1673 | 0.1394/0.1911 | 0.1690 | 2.0416/7.0923 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/GAR/Raw/P0/9204 | 2.6816 | 3.7892/0.1536 | 0.1094 | 2.0521/7.2863 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/GAR/Raw/P1/9201 | 0.0814 | 0.0758/0.0867 | 0.0663 | 2.2572/9.0149 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GAR/Raw/P1/9202 | 1.0161 | 0.9150/1.1079 | 0.6147 | 2.2688/10.1565 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GAR/Raw/P1/9203 | 0.9088 | 1.1202/0.6301 | 0.3216 | 2.2467/10.2240 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GAR/Raw/P1/9204 | 1.0165 | 1.4182/0.2349 | 0.1658 | 2.2448/12.6225 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GAR/W/P0/9201 | 3.5115 | 3.3133/3.6992 | 2.7448 | 2.1727/8.6587 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GAR/W/P0/9202 | 0.1133 | 0.1109/0.1156 | 0.0992 | 2.0923/7.1721 | 0/0/0 | PASS | INVALID_REPEATED_INDEX_OR_DUPLICATE_FACE | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GAR/W/P0/9203 | 0.1657 | 0.1384/0.1890 | 0.1677 | 2.0463/7.0461 | 0/0/0 | PASS | INVALID_REPEATED_INDEX_OR_DUPLICATE_FACE | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GAR/W/P0/9204 | 2.6816 | 3.7892/0.1540 | 0.1114 | 2.0515/7.2863 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GAR/W/P1/9201 | 0.0867 | 0.0806/0.0925 | 0.0720 | 2.2574/8.8471 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GAR/W/P1/9202 | 1.0147 | 0.9117/1.1082 | 0.6149 | 2.2708/9.4481 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GAR/W/P1/9203 | 0.9083 | 1.1199/0.6291 | 0.3205 | 2.2493/10.2240 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GAR/W/P1/9204 | 1.0156 | 1.4169/0.2351 | 0.1647 | 2.2392/12.6225 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GR/Raw/P0/9201 | 3.5094 | 3.3070/3.7008 | 2.7325 | 2.1716/8.8661 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/GR/Raw/P0/9202 | 0.1115 | 0.1083/0.1146 | 0.0993 | 2.1016/7.2082 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/GR/Raw/P0/9203 | 0.1673 | 0.1394/0.1911 | 0.1690 | 2.0416/7.0923 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/GR/Raw/P0/9204 | 2.6800 | 3.7869/0.1544 | 0.1091 | 2.0509/7.3019 | 0/0/0 | PASS | OPEN_BOUNDARY | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | PASS_MANIFOLD_WITH_POSSIBLE_BOUNDARY |
| development/GR/Raw/P1/9201 | 0.0800 | 0.0736/0.0860 | 0.0653 | 2.2580/11.1912 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GR/Raw/P1/9202 | 1.0152 | 0.9130/1.1080 | 0.6146 | 2.2659/10.1808 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GR/Raw/P1/9203 | 0.9085 | 1.1197/0.6300 | 0.3210 | 2.2487/10.0994 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GR/Raw/P1/9204 | 1.0170 | 1.4192/0.2334 | 0.1645 | 2.2507/11.6330 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GR/W/P0/9201 | 3.5081 | 3.3036/3.7013 | 2.7327 | 2.1716/8.8661 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GR/W/P0/9202 | 0.1103 | 0.1075/0.1130 | 0.0981 | 2.0966/7.1498 | 0/0/0 | PASS | INVALID_REPEATED_INDEX_OR_DUPLICATE_FACE | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GR/W/P0/9203 | 0.1657 | 0.1384/0.1890 | 0.1677 | 2.0463/7.0461 | 0/0/0 | PASS | INVALID_REPEATED_INDEX_OR_DUPLICATE_FACE | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GR/W/P0/9204 | 2.6799 | 3.7868/0.1548 | 0.1077 | 2.0448/7.3054 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GR/W/P1/9201 | 0.0834 | 0.0771/0.0894 | 0.0688 | 2.2671/11.1912 | 0/0/0 | PASS | INVALID_REPEATED_INDEX_OR_DUPLICATE_FACE | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GR/W/P1/9202 | 1.0154 | 0.9135/1.1079 | 0.6145 | 2.2646/10.1544 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GR/W/P1/9203 | 0.9403 | 1.1710/0.6301 | 0.3204 | 2.2481/9.8495 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |
| development/GR/W/P1/9204 | 1.0167 | 1.4188/0.2332 | 0.1666 | 2.2423/11.6330 | 0/0/0 | PASS | NONMANIFOLD_OPEN | FAIL_DETECTED_ILLEGAL_INTERSECTIONS | FAIL_NONMANIFOLD_OR_UNUSED_VERTEX |

| 阶段/组/层/父/seed | 顶点数 | 边1/2/>2 | 共边组件 | T-junction数/范围 | 非流形顶点 | 一致定向 | 相交面数 | 其中q≤10且非近退化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development/G0/Raw/P0/9201 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 96 | 92 |
| development/G0/Raw/P0/9202 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 93 | 93 |
| development/G0/Raw/P0/9203 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 97 | 97 |
| development/G0/Raw/P0/9204 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 86 | 86 |
| development/G0/Raw/P1/9201 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 88 | 83 |
| development/G0/Raw/P1/9202 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 94 | 86 |
| development/G0/Raw/P1/9203 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 95 | 89 |
| development/G0/Raw/P1/9204 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 88 | 80 |
| development/G0/W/P0/9201 | 309 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 3 | INCONSISTENT_OR_NONMANIFOLD | 96 | 92 |
| development/G0/W/P0/9202 | 214 | 290/23/0 | 90 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 62 | INCONSISTENT_OR_NONMANIFOLD | 87 | 87 |
| development/G0/W/P0/9203 | 215 | 287/23/1 | 90 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 55 | INCONSISTENT_OR_NONMANIFOLD | 90 | 90 |
| development/G0/W/P0/9204 | 201 | 296/20/0 | 93 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 69 | INCONSISTENT_OR_NONMANIFOLD | 79 | 79 |
| development/G0/W/P1/9201 | 203 | 271/31/1 | 81 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 56 | INCONSISTENT_OR_NONMANIFOLD | 77 | 72 |
| development/G0/W/P1/9202 | 219 | 300/18/0 | 94 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 60 | INCONSISTENT_OR_NONMANIFOLD | 82 | 76 |
| development/G0/W/P1/9203 | 203 | 294/21/0 | 93 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 69 | INCONSISTENT_OR_NONMANIFOLD | 83 | 75 |
| development/G0/W/P1/9204 | 231 | 296/20/0 | 92 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 53 | INCONSISTENT_OR_NONMANIFOLD | 83 | 74 |
| development/GA/Raw/P0/9201 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 96 | 94 |
| development/GA/Raw/P0/9202 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 96 | 96 |
| development/GA/Raw/P0/9203 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 97 | 97 |
| development/GA/Raw/P0/9204 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 94 | 94 |
| development/GA/Raw/P1/9201 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 89 | 85 |
| development/GA/Raw/P1/9202 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 89 | 84 |
| development/GA/Raw/P1/9203 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 93 | 87 |
| development/GA/Raw/P1/9204 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 87 | 77 |
| development/GA/W/P0/9201 | 310 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 96 | 94 |
| development/GA/W/P0/9202 | 214 | 290/23/0 | 92 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 61 | INCONSISTENT_OR_NONMANIFOLD | 91 | 91 |
| development/GA/W/P0/9203 | 215 | 287/23/1 | 90 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 55 | INCONSISTENT_OR_NONMANIFOLD | 90 | 90 |
| development/GA/W/P0/9204 | 204 | 288/24/0 | 89 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 64 | INCONSISTENT_OR_NONMANIFOLD | 82 | 82 |
| development/GA/W/P1/9201 | 203 | 275/29/1 | 83 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 58 | INCONSISTENT_OR_NONMANIFOLD | 76 | 72 |
| development/GA/W/P1/9202 | 223 | 304/16/0 | 96 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 55 | INCONSISTENT_OR_NONMANIFOLD | 80 | 76 |
| development/GA/W/P1/9203 | 194 | 288/24/0 | 92 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 70 | INCONSISTENT_OR_NONMANIFOLD | 90 | 84 |
| development/GA/W/P1/9204 | 236 | 304/16/0 | 96 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 52 | INCONSISTENT_OR_NONMANIFOLD | 87 | 77 |
| development/GAR/Raw/P0/9201 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 96 | 94 |
| development/GAR/Raw/P0/9202 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 95 | 95 |
| development/GAR/Raw/P0/9203 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 98 | 98 |
| development/GAR/Raw/P0/9204 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 89 | 89 |
| development/GAR/Raw/P1/9201 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 93 | 89 |
| development/GAR/Raw/P1/9202 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 90 | 85 |
| development/GAR/Raw/P1/9203 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 96 | 90 |
| development/GAR/Raw/P1/9204 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 90 | 81 |
| development/GAR/W/P0/9201 | 310 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 96 | 94 |
| development/GAR/W/P0/9202 | 214 | 286/25/0 | 90 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 59 | INCONSISTENT_OR_NONMANIFOLD | 90 | 90 |
| development/GAR/W/P0/9203 | 228 | 282/27/0 | 88 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 40 | INCONSISTENT_OR_NONMANIFOLD | 96 | 96 |
| development/GAR/W/P0/9204 | 217 | 294/21/0 | 92 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 58 | INCONSISTENT_OR_NONMANIFOLD | 85 | 85 |
| development/GAR/W/P1/9201 | 212 | 282/27/0 | 85 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 50 | INCONSISTENT_OR_NONMANIFOLD | 79 | 75 |
| development/GAR/W/P1/9202 | 226 | 296/20/0 | 92 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 55 | INCONSISTENT_OR_NONMANIFOLD | 82 | 78 |
| development/GAR/W/P1/9203 | 210 | 294/21/0 | 91 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 55 | INCONSISTENT_OR_NONMANIFOLD | 93 | 87 |
| development/GAR/W/P1/9204 | 250 | 306/15/0 | 97 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 45 | INCONSISTENT_OR_NONMANIFOLD | 85 | 76 |
| development/GR/Raw/P0/9201 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 96 | 92 |
| development/GR/Raw/P0/9202 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 96 | 96 |
| development/GR/Raw/P0/9203 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 98 | 98 |
| development/GR/Raw/P0/9204 | 312 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 0 | INCONSISTENT_OR_NONMANIFOLD | 89 | 89 |
| development/GR/Raw/P1/9201 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 96 | 89 |
| development/GR/Raw/P1/9202 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 85 | 77 |
| development/GR/Raw/P1/9203 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 97 | 91 |
| development/GR/Raw/P1/9204 | 312 | 314/11/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 2 | INCONSISTENT_OR_NONMANIFOLD | 89 | 81 |
| development/GR/W/P0/9201 | 309 | 312/12/0 | 101 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 3 | INCONSISTENT_OR_NONMANIFOLD | 96 | 92 |
| development/GR/W/P0/9202 | 217 | 284/26/0 | 89 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 54 | INCONSISTENT_OR_NONMANIFOLD | 89 | 89 |
| development/GR/W/P0/9203 | 228 | 282/27/0 | 88 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 40 | INCONSISTENT_OR_NONMANIFOLD | 96 | 96 |
| development/GR/W/P0/9204 | 203 | 280/28/0 | 85 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 57 | INCONSISTENT_OR_NONMANIFOLD | 79 | 79 |
| development/GR/W/P1/9201 | 204 | 268/34/0 | 82 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 54 | INCONSISTENT_OR_NONMANIFOLD | 83 | 76 |
| development/GR/W/P1/9202 | 230 | 300/18/0 | 94 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 56 | INCONSISTENT_OR_NONMANIFOLD | 80 | 74 |
| development/GR/W/P1/9203 | 204 | 294/21/0 | 91 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 64 | INCONSISTENT_OR_NONMANIFOLD | 91 | 86 |
| development/GR/W/P1/9204 | 239 | 302/17/0 | 95 | 0 / EXHAUSTIVE_VERTEX_EDGE_CHECK | 52 | INCONSISTENT_OR_NONMANIFOLD | 91 | 83 |

相交面面积与q的完整分布在JSON中。出现q≤10且非近退化的相交面会反驳‘所有相交都来自极差小面’的简化假设；即使某面很差，也没有证明可安全删除。本轮没有prune。

## 配对比较与候选冻结

| 比较/阶段/组/层/父/seed | R12 Δ百分点 | 窄角比例Δ | Gamma RMS Δ | 新CF proper | 新FF proper | 新FF共面 |
| --- | --- | --- | --- | --- | --- | --- |
| W_same_generation/development/G0/W/P0/9201 | 0.0041 | 0.0000 | -0.0011 | 0 | 0 | 0 |
| W_same_generation/development/G0/W/P0/9202 | -0.0014 | 0.0000 | -0.0051 | 0 | 11 | 0 |
| W_same_generation/development/G0/W/P0/9203 | -0.0005 | 0.0000 | -0.0014 | 3 | 11 | 1 |
| W_same_generation/development/G0/W/P0/9204 | -0.0005 | 0.0000 | -0.0010 | 3 | 15 | 0 |
| W_same_generation/development/G0/W/P1/9201 | 0.0032 | 0.0000 | -0.0032 | 2 | 7 | 1 |
| W_same_generation/development/G0/W/P1/9202 | -0.0011 | 0.0000 | -0.0001 | 1 | 20 | 0 |
| W_same_generation/development/G0/W/P1/9203 | -0.0003 | 0.0100 | -0.0029 | 2 | 15 | 1 |
| W_same_generation/development/G0/W/P1/9204 | 0.0018 | 0.0000 | -0.0030 | 0 | 13 | 0 |
| guidance_same_layer/development/GAR/Raw/P0/9201 | 0.0053 | -0.0100 | 0.0092 | 0 | 5 | 0 |
| guidance_same_layer/development/GAR/W/P0/9201 | 0.0010 | -0.0100 | 0.0079 | 0 | 5 | 0 |
| W_same_generation/development/GAR/W/P0/9201 | -0.0001 | 0.0000 | -0.0023 | 0 | 0 | 0 |
| guidance_same_layer/development/GAR/Raw/P0/9202 | 0.0229 | 0.0000 | 0.0918 | 15 | 24 | 0 |
| guidance_same_layer/development/GAR/W/P0/9202 | 0.0233 | -0.0100 | 0.0891 | 15 | 34 | 1 |
| W_same_generation/development/GAR/W/P0/9202 | -0.0009 | -0.0100 | -0.0078 | 1 | 14 | 1 |
| guidance_same_layer/development/GAR/Raw/P0/9203 | 0.0030 | 0.0100 | 0.0080 | 22 | 32 | 0 |
| guidance_same_layer/development/GAR/W/P0/9203 | 0.0019 | 0.0100 | 0.0052 | 23 | 28 | 0 |
| W_same_generation/development/GAR/W/P0/9203 | -0.0016 | 0.0000 | -0.0042 | 1 | 11 | 1 |
| guidance_same_layer/development/GAR/Raw/P0/9204 | 0.0001 | 0.0000 | 0.0297 | 11 | 21 | 0 |
| guidance_same_layer/development/GAR/W/P0/9204 | 0.0006 | 0.0000 | 0.0322 | 12 | 31 | 0 |
| W_same_generation/development/GAR/W/P0/9204 | -2.8271e-06 | 0.0000 | 0.0015 | 2 | 8 | 0 |
| guidance_same_layer/development/GAR/Raw/P1/9201 | 0.0075 | -0.0200 | 0.1058 | 15 | 33 | 0 |
| guidance_same_layer/development/GAR/W/P1/9201 | 0.0096 | -0.0200 | 0.1067 | 17 | 29 | 0 |
| W_same_generation/development/GAR/W/P1/9201 | 0.0053 | 0.0000 | -0.0024 | 1 | 10 | 0 |
| guidance_same_layer/development/GAR/Raw/P1/9202 | -0.0061 | -0.0100 | 0.0384 | 2 | 24 | 0 |
| guidance_same_layer/development/GAR/W/P1/9202 | -0.0064 | -0.0200 | 0.0387 | 2 | 21 | 0 |
| W_same_generation/development/GAR/W/P1/9202 | -0.0014 | -0.0100 | 9.8816e-05 | 1 | 11 | 0 |
| guidance_same_layer/development/GAR/Raw/P1/9203 | -0.0009 | -0.0100 | 0.0627 | 11 | 34 | 0 |
| guidance_same_layer/development/GAR/W/P1/9203 | -0.0011 | -0.0200 | 0.0654 | 10 | 37 | 0 |
| W_same_generation/development/GAR/W/P1/9203 | -0.0005 | 0.0000 | -0.0002 | 0 | 9 | 0 |
| guidance_same_layer/development/GAR/Raw/P1/9204 | -0.0003 | -0.0100 | 0.0170 | 13 | 35 | 0 |
| guidance_same_layer/development/GAR/W/P1/9204 | -0.0030 | -0.0100 | 0.0136 | 15 | 41 | 0 |
| W_same_generation/development/GAR/W/P1/9204 | -0.0009 | 0.0000 | -0.0064 | 1 | 10 | 0 |
| guidance_same_layer/development/GA/Raw/P0/9201 | 0.0052 | -0.0100 | -0.0004 | 0 | 7 | 0 |
| guidance_same_layer/development/GA/W/P0/9201 | 0.0009 | -0.0100 | -1.5635e-05 | 0 | 7 | 0 |
| W_same_generation/development/GA/W/P0/9201 | -0.0001 | 0.0000 | -0.0007 | 0 | 0 | 0 |
| guidance_same_layer/development/GA/Raw/P0/9202 | 0.0005 | 0.0000 | -0.0002 | 1 | 21 | 0 |
| guidance_same_layer/development/GA/W/P0/9202 | -0.0021 | -0.0100 | -0.0043 | 2 | 21 | 1 |
| W_same_generation/development/GA/W/P0/9202 | -0.0041 | -0.0100 | -0.0092 | 1 | 8 | 1 |
| guidance_same_layer/development/GA/Raw/P0/9203 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 |
| guidance_same_layer/development/GA/W/P0/9203 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 |
| W_same_generation/development/GA/W/P0/9203 | -0.0005 | 0.0000 | -0.0014 | 3 | 11 | 1 |
| guidance_same_layer/development/GA/Raw/P0/9204 | 0.0006 | 0.0000 | 0.0012 | 4 | 26 | 0 |
| guidance_same_layer/development/GA/W/P0/9204 | 0.0010 | 0.0000 | 4.3783e-05 | 4 | 29 | 0 |
| W_same_generation/development/GA/W/P0/9204 | -0.0002 | 0.0000 | -0.0022 | 3 | 14 | 0 |
| guidance_same_layer/development/GA/Raw/P1/9201 | -0.0004 | -0.0100 | 0.0107 | 9 | 23 | 0 |
| guidance_same_layer/development/GA/W/P1/9201 | 0.0022 | -0.0100 | 0.0138 | 8 | 28 | 1 |
| W_same_generation/development/GA/W/P1/9201 | 0.0058 | 0.0000 | -0.0002 | 1 | 11 | 1 |
| guidance_same_layer/development/GA/Raw/P1/9202 | 0.0010 | -0.0400 | -0.0190 | 2 | 25 | 0 |
| guidance_same_layer/development/GA/W/P1/9202 | 0.0016 | -0.0300 | -0.0184 | 1 | 18 | 0 |
| W_same_generation/development/GA/W/P1/9202 | -0.0005 | 0.0100 | 0.0004 | 0 | 8 | 0 |
| guidance_same_layer/development/GA/Raw/P1/9203 | 0.0322 | -0.0200 | 0.0087 | 2 | 29 | 0 |
| guidance_same_layer/development/GA/W/P1/9203 | 0.0318 | -0.0200 | 0.0090 | 2 | 38 | 1 |
| W_same_generation/development/GA/W/P1/9203 | -0.0007 | 0.0100 | -0.0027 | 0 | 17 | 2 |
| guidance_same_layer/development/GA/Raw/P1/9204 | -0.0008 | -0.0100 | -0.0099 | 0 | 20 | 0 |
| guidance_same_layer/development/GA/W/P1/9204 | -0.0295 | -0.0100 | -0.0111 | 5 | 31 | 0 |
| W_same_generation/development/GA/W/P1/9204 | -0.0269 | 0.0000 | -0.0042 | 5 | 15 | 0 |
| guidance_same_layer/development/GR/Raw/P0/9201 | 0.0030 | 0.0000 | 0.0091 | 0 | 4 | 0 |
| guidance_same_layer/development/GR/W/P0/9201 | -0.0024 | 0.0000 | 0.0102 | 0 | 4 | 0 |
| W_same_generation/development/GR/W/P0/9201 | -0.0013 | 0.0000 | 0.0000 | 0 | 0 | 0 |
| guidance_same_layer/development/GR/Raw/P0/9202 | 0.0202 | 0.0000 | 0.0948 | 12 | 34 | 0 |
| guidance_same_layer/development/GR/W/P0/9202 | 0.0203 | -0.0100 | 0.1002 | 14 | 29 | 1 |
| W_same_generation/development/GR/W/P0/9202 | -0.0012 | -0.0100 | 0.0002 | 2 | 8 | 1 |
| guidance_same_layer/development/GR/Raw/P0/9203 | 0.0030 | 0.0100 | 0.0080 | 22 | 32 | 0 |
| guidance_same_layer/development/GR/W/P0/9203 | 0.0019 | 0.0100 | 0.0052 | 23 | 28 | 0 |
| W_same_generation/development/GR/W/P0/9203 | -0.0016 | 0.0000 | -0.0042 | 1 | 11 | 1 |
| guidance_same_layer/development/GR/Raw/P0/9204 | -0.0015 | 0.0000 | 0.0225 | 9 | 20 | 0 |
| guidance_same_layer/development/GR/W/P0/9204 | -0.0011 | 0.0000 | 0.0275 | 7 | 32 | 0 |
| W_same_generation/development/GR/W/P0/9204 | -8.1342e-05 | 0.0000 | 0.0040 | 0 | 14 | 0 |
| guidance_same_layer/development/GR/Raw/P1/9201 | 0.0061 | -0.0100 | 0.1075 | 13 | 37 | 0 |
| guidance_same_layer/development/GR/W/P1/9201 | 0.0063 | -0.0100 | 0.1088 | 15 | 29 | 2 |
| W_same_generation/development/GR/W/P1/9201 | 0.0034 | 0.0000 | -0.0019 | 1 | 7 | 2 |
| guidance_same_layer/development/GR/Raw/P1/9202 | -0.0070 | 0.0100 | 0.0275 | 2 | 18 | 0 |
| guidance_same_layer/development/GR/W/P1/9202 | -0.0058 | 0.0000 | 0.0272 | 2 | 13 | 0 |
| W_same_generation/development/GR/W/P1/9202 | 0.0002 | -0.0100 | -0.0005 | 0 | 9 | 0 |
| guidance_same_layer/development/GR/Raw/P1/9203 | -0.0012 | -0.0200 | 0.0580 | 11 | 34 | 0 |
| guidance_same_layer/development/GR/W/P1/9203 | 0.0309 | -0.0100 | 0.0659 | 10 | 38 | 0 |
| W_same_generation/development/GR/W/P1/9203 | 0.0318 | 0.0200 | 0.0050 | 0 | 23 | 0 |
| guidance_same_layer/development/GR/Raw/P1/9204 | 0.0002 | 0.0000 | 0.0180 | 10 | 25 | 0 |
| guidance_same_layer/development/GR/W/P1/9204 | -0.0019 | 0.0000 | 0.0205 | 15 | 33 | 0 |
| W_same_generation/development/GR/W/P1/9204 | -0.0003 | 0.0000 | -0.0005 | 4 | 13 | 0 |

guidance_same_layer在Raw上比较G0 Raw、在W上比较G0+W；W_same_generation比较同一Raw前后。新增与移除pair集合同时保留，不能用总数下降掩盖替换出的新相交。W候选没有全局碰撞接受门禁。

选择记录：`{"status": "NO_QUALIFIED_CANDIDATE", "candidate": null, "eligible": [], "baseline_parents": {"0": {"R12": 1.6108610810670028, "CF": 0.19250000000000003, "FF": 122.25, "narrow": 0.10500000000000001, "all_C": true, "degenerate_by_seed": {"9201": 0, "9202": 0, "9203": 0, "9204": 0}}, "1": {"R12": 0.7556630216336697, "CF": 0.1375, "FF": 97.0, "narrow": 0.1575, "all_C": true, "degenerate_by_seed": {"9201": 0, "9202": 0, "9203": 0, "9204": 0}}}, "decisions": {"GA": {"pass_gate": false, "hard_gates_pass": true, "option_A_allparents": false, "option_B_allparents": false, "parents": {"0": {"R12": 1.6124483956055462, "CF": 0.19000000000000003, "FF": 127.5, "narrow": 0.10250000000000001, "all_C": true, "degenerate_by_seed": {"9201": 0, "9202": 0, "9203": 0, "9204": 0}}, "1": {"R12": 0.7636633554777621, "CF": 0.14250000000000002, "FF": 103.75, "narrow": 0.1375, "all_C": true, "degenerate_by_seed": {"9201": 0, "9202": 0, "9203": 0, "9204": 0}}}, "checks": {"0": {"R12_within5pct": true, "C_exact": true, "no_added_degeneracy": true, "option_A": false, "option_B": false}, "1": {"R12_within5pct": true, "C_exact": true, "no_added_degeneracy": true, "option_A": false, "option_B": false}}, "ranking_key": [0.16625, 115.625, 1.1880558755416541, 2.052917766749996]}, "GR": {"pass_gate": false, "hard_gates_pass": true, "option_A_allparents": false, "option_B_allparents": false, "parents": {"0": {"R12": 1.6170552257439907, "CF": 0.2125, "FF": 129.5, "narrow": 0.1075, "all_C": true, "degenerate_by_seed": {"9201": 0, "9202": 0, "9203": 0, "9204": 0}}, "1": {"R12": 0.7551745263782015, "CF": 0.16249999999999998, "FF": 108.0, "narrow": 0.1525, "all_C": true, "degenerate_by_seed": {"9201": 0, "9202": 0, "9203": 0, "9204": 0}}}, "checks": {"0": {"R12_within5pct": true, "C_exact": true, "no_added_degeneracy": true, "option_A": false, "option_B": false}, "1": {"R12_within5pct": true, "C_exact": true, "no_added_degeneracy": true, "option_A": false, "option_B": false}}, "ranking_key": [0.1875, 118.75, 1.186114876061096, 2.0233740147500043]}, "GAR": {"pass_gate": false, "hard_gates_pass": true, "option_A_allparents": false, "option_B_allparents": false, "parents": {"0": {"R12": 1.6186863281009332, "CF": 0.225, "FF": 126.75, "narrow": 0.105, "all_C": true, "degenerate_by_seed": {"9201": 0, "9202": 0, "9203": 0, "9204": 0}}, "1": {"R12": 0.7557002556672985, "CF": 0.16749999999999998, "FF": 108.75, "narrow": 0.14500000000000002, "all_C": true, "degenerate_by_seed": {"9201": 0, "9202": 0, "9203": 0, "9204": 0}}}, "checks": {"0": {"R12_within5pct": true, "C_exact": true, "no_added_degeneracy": true, "option_A": false, "option_B": false}, "1": {"R12_within5pct": true, "C_exact": true, "no_added_degeneracy": true, "option_A": false, "option_B": false}}, "ranking_key": [0.19624999999999998, 117.75, 1.1871932918841157, 2.0875820855000047]}}, "raw_only": true, "selection_saved_before_any_new_noise": true, "confirmation_seeds": [9401, 9402, 9403, 9404, 9405, 9406, 9407, 9408], "no_nonfinite": "All51-state finite runtime assertions passed", "selected_time_utc": "2026-09-29T08:28:40Z", "selection_file": {"path": "/mnt/e/MeshFlow/experiments/mf_h1_local0/outputs/gpu_guidance_and_protected_weld_20260929_160351/selection.json", "bytes": 4889, "sha256": "8c82b2697feb4ddaab984ba3a0e6424bf946242c45b1f5e19f3721bbda79cbde"}}`。只用Raw门槛决定候选；先冻结selection，再读取确认结果，不能由后处理或确认图反向选参数。

## 同噪声自由部分绑定

| 阶段/组/层/seed | R0/C0 | R0/C1 | R1/C0 | R1/C1 | 方向0/1 | 双正 |
| --- | --- | --- | --- | --- | --- | --- |
| development/G0/Raw/9201 | 3.5064 | 8.1333 | 4.7665 | 0.0739 | 4.6269/4.6926 | PASS |
| development/G0/Raw/9202 | 0.0913 | 8.1360 | 9.4326 | 1.0222 | 8.0446/8.4104 | PASS |
| development/G0/Raw/9203 | 0.1642 | 8.3616 | 9.3513 | 0.9097 | 8.1974/8.4415 | PASS |
| development/G0/Raw/9204 | 2.6815 | 8.1996 | 8.9860 | 1.0168 | 5.5181/7.9692 | PASS |
| development/G0/W/9201 | 3.5105 | 8.1351 | 4.7635 | 0.0771 | 4.6246/4.6864 | PASS |
| development/G0/W/9202 | 0.0899 | 8.1517 | 9.3997 | 1.0211 | 8.0617/8.3786 | PASS |
| development/G0/W/9203 | 0.1638 | 8.3311 | 9.3356 | 0.9094 | 8.1673/8.4261 | PASS |
| development/G0/W/9204 | 2.6810 | 8.2014 | 8.9972 | 1.0186 | 5.5205/7.9786 | PASS |
| development/GA/Raw/9201 | 3.5116 | 8.1331 | 4.7670 | 0.0736 | 4.6215/4.6934 | PASS |
| development/GA/Raw/9202 | 0.0919 | 8.1462 | 9.4160 | 1.0232 | 8.0543/8.3928 | PASS |
| development/GA/Raw/9203 | 0.1642 | 8.3589 | 9.3513 | 0.9419 | 8.1947/8.4094 | PASS |
| development/GA/Raw/9204 | 2.6821 | 8.2177 | 8.9787 | 1.0160 | 5.5355/7.9627 | PASS |
| development/GA/W/9201 | 3.5114 | 8.1220 | 4.7673 | 0.0793 | 4.6105/4.6879 | PASS |
| development/GA/W/9202 | 0.0878 | 8.1499 | 9.4014 | 1.0227 | 8.0620/8.3787 | PASS |
| development/GA/W/9203 | 0.1638 | 8.3451 | 9.3356 | 0.9412 | 8.1814/8.3944 | PASS |
| development/GA/W/9204 | 2.6820 | 8.2031 | 8.9580 | 0.9891 | 5.5211/7.9689 | PASS |
| development/GAR/Raw/9201 | 3.5117 | 8.1226 | 4.7625 | 0.0814 | 4.6109/4.6811 | PASS |
| development/GAR/Raw/9202 | 0.1142 | 8.1392 | 9.4117 | 1.0161 | 8.0250/8.3956 | PASS |
| development/GAR/Raw/9203 | 0.1673 | 8.3736 | 9.3358 | 0.9088 | 8.2063/8.4270 | PASS |
| development/GAR/Raw/9204 | 2.6816 | 8.2205 | 8.9292 | 1.0165 | 5.5389/7.9127 | PASS |
| development/GAR/W/9201 | 3.5115 | 8.1359 | 4.7628 | 0.0867 | 4.6244/4.6761 | PASS |
| development/GAR/W/9202 | 0.1133 | 8.1126 | 9.4098 | 1.0147 | 7.9994/8.3951 | PASS |
| development/GAR/W/9203 | 0.1657 | 8.3819 | 9.3350 | 0.9083 | 8.2162/8.4267 | PASS |
| development/GAR/W/9204 | 2.6816 | 8.2282 | 8.9521 | 1.0156 | 5.5466/7.9364 | PASS |
| development/GR/Raw/9201 | 3.5094 | 8.1234 | 4.7698 | 0.0800 | 4.6140/4.6898 | PASS |
| development/GR/Raw/9202 | 0.1115 | 8.1470 | 9.4023 | 1.0152 | 8.0355/8.3871 | PASS |
| development/GR/Raw/9203 | 0.1673 | 8.3586 | 9.3358 | 0.9085 | 8.1913/8.4273 | PASS |
| development/GR/Raw/9204 | 2.6800 | 8.2171 | 8.9521 | 1.0170 | 5.5371/7.9351 | PASS |
| development/GR/W/9201 | 3.5081 | 8.1364 | 4.7705 | 0.0834 | 4.6283/4.6871 | PASS |
| development/GR/W/9202 | 0.1103 | 8.1606 | 9.4186 | 1.0154 | 8.0503/8.4032 | PASS |
| development/GR/W/9203 | 0.1657 | 8.3644 | 9.3350 | 0.9403 | 8.1988/8.3947 | PASS |
| development/GR/W/9204 | 2.6799 | 8.2164 | 8.9411 | 1.0167 | 5.5365/7.9244 | PASS |

同一输出固定free集合在两个目标间评分，无重新配面，GT只用于评价。

## 实测成本与调用账本

| 阶段/组/父/seed | 模型CUDA秒 | 能量前向秒 | 坐标反向秒 | 总CUDA秒 | setup秒 | 循环墙钟秒 | 完整生成秒 | 最终D2H秒 | Torch峰值GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development/GAR/P0/9201 | 2.3024 | 0.0277 | 0.0387 | 2.4302 | 0.0073 | 2.4302 | 2.4407 | 0.0030 | 0.5353 |
| development/GAR/P0/9202 | 1.8358 | 0.0177 | 0.0481 | 1.9257 | 0.0058 | 1.9258 | 1.9358 | 0.0041 | 0.5353 |
| development/GAR/P0/9203 | 1.8129 | 0.0196 | 0.0365 | 1.8942 | 0.0056 | 1.8943 | 1.9033 | 0.0033 | 0.5353 |
| development/GAR/P0/9204 | 1.8842 | 0.0246 | 0.0616 | 1.9993 | 0.0059 | 1.9994 | 2.0121 | 0.0067 | 0.5353 |
| development/GAR/P1/9201 | 2.0088 | 0.0223 | 0.0404 | 2.0988 | 0.0064 | 2.0988 | 2.1135 | 0.0081 | 0.5353 |
| development/GAR/P1/9202 | 1.8805 | 0.0193 | 0.0398 | 1.9669 | 0.0063 | 1.9669 | 1.9766 | 0.0032 | 0.5353 |
| development/GAR/P1/9203 | 1.8934 | 0.0210 | 0.0452 | 1.9852 | 0.0069 | 1.9853 | 1.9978 | 0.0054 | 0.5353 |
| development/GAR/P1/9204 | 2.1977 | 0.0320 | 0.0494 | 2.3068 | 0.0075 | 2.3068 | 2.3209 | 0.0064 | 0.5353 |
| development/GA/P0/9201 | 2.2474 | 0.0079 | 0.0193 | 2.3026 | 0.0074 | 2.3026 | 2.3152 | 0.0050 | 0.5353 |
| development/GA/P0/9202 | 2.0080 | 0.0080 | 0.0206 | 2.0624 | 0.0072 | 2.0624 | 2.0735 | 0.0037 | 0.5353 |
| development/GA/P0/9203 | 2.0280 | 0.0075 | 0.0194 | 2.0872 | 0.0051 | 2.0872 | 2.0987 | 0.0062 | 0.5353 |
| development/GA/P0/9204 | 1.8582 | 0.0083 | 0.0195 | 1.9115 | 0.0076 | 1.9115 | 1.9256 | 0.0063 | 0.5353 |
| development/GA/P1/9201 | 1.9540 | 0.0088 | 0.0243 | 2.0171 | 0.0077 | 2.0171 | 2.0304 | 0.0055 | 0.5353 |
| development/GA/P1/9202 | 1.9672 | 0.0119 | 0.0170 | 2.0257 | 0.0085 | 2.0257 | 2.0378 | 0.0035 | 0.5353 |
| development/GA/P1/9203 | 1.8625 | 0.0079 | 0.0183 | 1.9176 | 0.0098 | 1.9176 | 1.9312 | 0.0037 | 0.5353 |
| development/GA/P1/9204 | 1.9438 | 0.0060 | 0.0172 | 1.9954 | 0.0087 | 1.9954 | 2.0110 | 0.0067 | 0.5353 |
| development/GR/P0/9201 | 2.0569 | 0.0156 | 0.0270 | 2.1253 | 0.0063 | 2.1253 | 2.1338 | 0.0020 | 0.5353 |
| development/GR/P0/9202 | 2.0205 | 0.0104 | 0.0283 | 2.0885 | 0.0068 | 2.0885 | 2.1000 | 0.0045 | 0.5353 |
| development/GR/P0/9203 | 2.0982 | 0.0151 | 0.0360 | 2.1754 | 0.0062 | 2.1754 | 2.1872 | 0.0055 | 0.5353 |
| development/GR/P0/9204 | 1.9041 | 0.0110 | 0.0238 | 1.9630 | 0.0066 | 1.9630 | 1.9723 | 0.0026 | 0.5353 |
| development/GR/P1/9201 | 1.7614 | 0.0114 | 0.0258 | 1.8229 | 0.0057 | 1.8229 | 1.8333 | 0.0045 | 0.5353 |
| development/GR/P1/9202 | 1.8837 | 0.0133 | 0.0289 | 1.9531 | 0.0088 | 1.9531 | 1.9660 | 0.0039 | 0.5353 |
| development/GR/P1/9203 | 1.9667 | 0.0151 | 0.0243 | 2.0387 | 0.0078 | 2.0388 | 2.0555 | 0.0087 | 0.5353 |
| development/GR/P1/9204 | 1.8667 | 0.0105 | 0.0221 | 1.9250 | 0.0082 | 1.9251 | 1.9388 | 0.0055 | 0.5353 |

完整生成墙钟包括setup、50步循环和最终D2H，不包含随后NPZ写盘；loop wall另列。CUDA event为设备时间跨度，可能包含主机发射间隙，不是进程CPU时间。GPU几何不等于零CPU调度。采样器没有逐步显式张量标量读回或CPU几何；原主干bool索引可能隐式同步，未单独profile。模型CPU dispatch未单独profile；CUDA event、完整墙钟、D2H和CPU后处理不可混算。峰值为Torch CUDA已分配内存，不是整个进程或驱动内存。预热、测试、重放、重复调用与正式样本分账。

| 实际计数 | 值 |
| --- | --- |
| model_forward_attempts | 1354 |
| model_forward_returns | 1354 |
| model_backward_calls | 0 |
| optimizer_updates | 0 |
| model_parameter_updates | 0 |
| generation_attempts | 27 |
| complete_generations | 27 |
| euler_updates | 1350 |
| active_forward_category | FINISHED |

几何能量账本（采样、探针、预热、离线端点分域）：`{"energy_forward_attempts": 361, "energy_forward_calls": 361, "energy_A_evaluations": 267, "energy_R_evaluations": 260, "vertex_gaussian_evaluations": 4, "geometry_backward_attempts": 402, "geometry_backward_calls": 402, "CPU_energy_forward_calls": 24, "CUDA_energy_forward_calls": 337, "CPU_geometry_backward_calls": 24, "CUDA_geometry_backward_calls": 378, "finite_difference_energy_evaluations": 57, "offline_primary_pair_tests": 7, "offline_independent_pair_tests": 7, "fixture_runs": 1, "model_forwards": 0, "model_backwards": 0, "optimizer_updates": 0}`。

CPU独立判交/评分账本：`{"verified_pair_tests": 393732, "primary_pair_tests": 470532, "independent_pair_tests": 393732, "independent_AABB_disjoint": 356944, "static_context_audits": 2, "raw_index_calls": 32, "indexed_topology_audits": 64, "OBJ_FP32_roundtrip_checks": 64, "historical_seam_evaluations": 64, "endpoint_metric_evaluations": 64, "boundary_distance_evaluations": 64, "static_context_audit_reuses": 64, "gpu_guidance_variant_scores": 64, "W_operator_calls": 32}`。

预热分账：`{"extra_model_forwards": 2, "geometry_ledger_delta": {"energy_forward_attempts": 6, "energy_forward_calls": 6, "energy_A_evaluations": 4, "energy_R_evaluations": 4, "vertex_gaussian_evaluations": 0, "geometry_backward_attempts": 8, "geometry_backward_calls": 8, "CPU_energy_forward_calls": 0, "CUDA_energy_forward_calls": 6, "CPU_geometry_backward_calls": 0, "CUDA_geometry_backward_calls": 8, "finite_difference_energy_evaluations": 0, "offline_primary_pair_tests": 0, "offline_independent_pair_tests": 0, "fixture_runs": 0, "model_forwards": 0, "model_backwards": 0, "optimizer_updates": 0}}`。开发实现检查分账：`{"energy_fixture_runs": 1, "energy_forwards": 38, "geometry_backwards": 48, "CPU_energy_forwards": 24, "CUDA_energy_forwards": 14, "CPU_geometry_backwards": 24, "CUDA_geometry_backwards": 24, "independent_numpy_energy_reference_evaluations": 57, "primary_pair_tests": 7, "independent_pair_tests": 7, "direct_finite_triangle_probe": 1, "evaluation_helper_runs": 1, "evaluation_fixture_families": 9, "W_fixture_runs": 2, "W_calls": 24, "topology_audits": 8, "direct_protected_guards": 2, "Raw_exact_index": 2, "temporary_OBJ_roundtrips": 2, "W_candidates": 90, "W_accepted": 84, "W_rejected": 6, "failed_executions": 0, "model_forwards": 0, "model_backwards": 0, "optimizer_updates": 0, "scope": "Development checks before frozen formal fixtures, separately disclosed; not formal samples"}`。

主干前向类别分账：`{"original_forward_equivalence": 1, "prevalidated_forward_equivalence": 1, "model_warmup": 2, "baseline_replay": 100, "development": 1200, "guided_repeat": 50}`。完整生成类别分账：`{"baseline_replay": 2, "development": 24, "guided_repeat": 1}`。

W处理、索引/OBJ导出与独立拓扑/几何评分耗时在weld逐输出记录分别保留；没有把它们混进GPU完整生成时间。

## 固定图与停止

## W、OBJ与独立CPU审计成本

| 阶段/组/父/seed | 接受/拒绝 | 最大位移 | W算子秒 | 精确索引秒 | 调用墙钟秒 | OBJ往返秒 | 拓扑秒 | C构造/OBJ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development/G0/P0/9201 | 3/0 | 0.0069 | 0.0857 | 0.0019 | 0.0882 | 0.0139 | 0.0109 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/G0/P0/9202 | 98/0 | 0.0071 | 0.0701 | 0.0016 | 0.0723 | 0.0153 | 0.0078 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/G0/P0/9203 | 97/0 | 0.0072 | 0.0678 | 0.0019 | 0.0703 | 0.0152 | 0.0077 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/G0/P0/9204 | 111/0 | 0.0072 | 0.0631 | 0.0017 | 0.0655 | 0.0147 | 0.0076 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/G0/P1/9201 | 109/0 | 0.0060 | 0.0679 | 0.0017 | 0.0702 | 0.0196 | 0.0072 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/G0/P1/9202 | 93/0 | 0.0061 | 0.0725 | 0.0016 | 0.0748 | 0.0185 | 0.0114 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/G0/P1/9203 | 109/0 | 0.0060 | 0.0702 | 0.0015 | 0.0728 | 0.0197 | 0.0083 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/G0/P1/9204 | 81/1 | 0.0061 | 0.0734 | 0.0015 | 0.0757 | 0.0148 | 0.0079 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GAR/P0/9201 | 2/0 | 0.0042 | 0.0857 | 0.0021 | 0.0885 | 0.0213 | 0.0109 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GAR/P0/9202 | 98/0 | 0.0072 | 0.0671 | 0.0015 | 0.0693 | 0.0160 | 0.0098 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GAR/P0/9203 | 84/0 | 0.0072 | 0.0675 | 0.0019 | 0.0701 | 0.0158 | 0.0076 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GAR/P0/9204 | 95/0 | 0.0072 | 0.0697 | 0.0016 | 0.0719 | 0.0169 | 0.0084 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GAR/P1/9201 | 100/0 | 0.0061 | 0.0674 | 0.0018 | 0.0698 | 0.0156 | 0.0075 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GAR/P1/9202 | 86/0 | 0.0060 | 0.0711 | 0.0024 | 0.0740 | 0.0164 | 0.1381 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GAR/P1/9203 | 102/0 | 0.0061 | 0.0644 | 0.0016 | 0.0666 | 0.0163 | 0.0071 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GAR/P1/9204 | 62/0 | 0.0059 | 0.0734 | 0.0021 | 0.0761 | 0.0149 | 0.0088 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GA/P0/9201 | 2/0 | 0.0060 | 0.0875 | 0.0022 | 0.0903 | 0.0166 | 0.0121 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GA/P0/9202 | 98/0 | 0.0073 | 0.0653 | 0.0019 | 0.0679 | 0.0145 | 0.0074 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GA/P0/9203 | 97/0 | 0.0072 | 0.0679 | 0.0017 | 0.0702 | 0.0164 | 0.0100 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GA/P0/9204 | 108/0 | 0.0072 | 0.0632 | 0.0016 | 0.0654 | 0.0180 | 0.0071 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GA/P1/9201 | 109/0 | 0.0061 | 0.0645 | 0.0015 | 0.0667 | 0.0150 | 0.0071 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GA/P1/9202 | 89/0 | 0.0061 | 0.0729 | 0.0018 | 0.0753 | 0.0156 | 0.0075 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GA/P1/9203 | 118/0 | 0.0060 | 0.0672 | 0.0018 | 0.0697 | 0.0172 | 0.0074 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GA/P1/9204 | 76/0 | 0.0061 | 0.0721 | 0.0016 | 0.0744 | 0.0162 | 0.0086 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GR/P0/9201 | 3/0 | 0.0069 | 0.0863 | 0.0019 | 0.0889 | 0.0172 | 0.0108 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GR/P0/9202 | 95/0 | 0.0071 | 0.0678 | 0.0017 | 0.0701 | 0.0139 | 0.0073 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GR/P0/9203 | 84/0 | 0.0072 | 0.0721 | 0.0016 | 0.0744 | 0.0139 | 0.0079 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GR/P0/9204 | 109/0 | 0.0073 | 0.0648 | 0.0016 | 0.0670 | 0.0098 | 0.0086 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GR/P1/9201 | 108/0 | 0.0061 | 0.0617 | 0.0021 | 0.0644 | 0.0166 | 0.0073 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GR/P1/9202 | 82/0 | 0.0061 | 0.0760 | 0.0017 | 0.0783 | 0.0163 | 0.0091 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GR/P1/9203 | 108/0 | 0.0061 | 0.0656 | 0.0016 | 0.0679 | 0.0144 | 0.0071 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |
| development/GR/P1/9204 | 73/1 | 0.0061 | 0.0767 | 0.0017 | 0.0790 | 0.0177 | 0.0091 | PASS_BITWISE_WITH_CORNER_CORRESPONDENCE / PASS_EXPORTED_OBJ_FP32_BITWISE_WITH_CORNER_CORRESPONDENCE |

W算子内部计时与外层调用墙钟、精确索引/OBJ导出/CPU拓扑审计各有口径，不能直接相加重复计费。每条输出的独立几何/表面审计秒数保留在offline_summary；表格最大位移使用原模型坐标单位，协议上限为.001L。

![固定两对象seed9201 Raw/+W及free-only](fixed_P0_P1_seed9201_Raw_W_full_free.png)

SHA256：`a0575443e7057ba8b25d2b5b5fb160921966c52d8a986550f0d55c31d942c0c5`。

图只读取存档NPZ和已存相交记录；蓝色C、红色CF proper、紫色CF共面、橙色线框为FF非法涉及面。free-only行不叠C，所有组/层共享相机和完整范围，不遮盖差例。
本轮新增训练更新=0、模型参数更新=0、主干反向=0；没有删面、补洞或改写已知面，不启动Phase3。按实际门禁和固定预算停止，不自动加训练、改阈值或追加修补。
