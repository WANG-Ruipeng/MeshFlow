# METHODS — R2_ANCHOR_AWARE_DENOISER_PILOT_V1

本轮仅适配发布的Chair去噪器。生成器、48份既有RAW、生成Gaussian、来源条件、GEO_C与原质量评分器保持固定。来源M只用于同形状合成扰动恢复，不用于真实生成RAW的唯一GT排名。

## 数据与接口

活动117条训练条件经既有CC/CY诊断筛选为112条、25父：排除2条确认C/Y共面重叠与3条C内部UNKNOWN；不因Y内部缺陷自动删父。112条全部有可核验的两面来源接口，共1173边。59条仍有Y内部非法事件，10条仍有Y内部UNKNOWN，不能称作无交干净训练集。6条DEV保留63条来源接口。

C/Y严格互补，原面ID、角点排列、顶点ID与FP32坐标逐一回读。SOURCE_SUPPORTED_INTERFACE仅是来源参考分层，不输入模型，不等同用户要求封闭的全部接口。源网格自然开放边、非流形/歧义边另存。来源原始绕序与规范化角点排列后的绕序分开记录。

每组1000更新，每批8不同父，先均匀抽父再父内均匀抽合法条件。每批4条alpha=1和4条FP32 uniform(0.9,1.1)；后者按既有实现共同缩放C/Y的x坐标，绕原点、不重新居中。固定8000条共享流，父曝光294—344，均值320；这是均匀采样实现，不声称每父实际次数完全相同。

先将完整目标乘a=0.95×0.3762=0.35739。C保持干净；每个Y三角角点副本独立加sigma=0.02 Gaussian，单位为denoiser坐标。sigma可追溯到发布检查点config.model.fixed_std。没有encode/sample内部二次加噪，没有OT。噪声、样本、alpha、面/角排列和完整逆映射全部预先物化；两组逐批读取同一文件并验证SHA。18条合成DEV输入在训练前冻结：6 clean+12 noisy。

## 模型与两条训练

同一个发布EMA，455,592,963参数、296个原state keys，strict=True。匹配官方源码commit db86bc32423c316c00bcc2f645a26243e8010f20；原权重SHA256为6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36。复用本地7,289,979,546字节文件，不复制或重新下载。

两组都采用PE→post_quant_linear→decoder→坐标残差。after-projection处添加激活dtype的m×e_known，e_known为1024维零初始化向量：
- D_CONT_NO_ROLE：固定零buffer，内部marker恒0；真实known身份仅供外部构造、损失与恢复使用。
- D_ANCHOR：真实每面三个角点known标记，e_known可训练；全部原参数也训练。
- D0_OFFICIAL：冻结官方EMA，真实RAW结果复用原合法缓存。

仅添加一次坐标残差，在FP32中相加；角色不加到XYZ。N仍是完整三角汤面数，保留官方face-count条件与AdaLN。known隐藏特征不detach，可以接收free损失经attention传来的梯度。无新增几何token、真实接口对应或seat/back文字输入。

唯一损失：
L = sum_over_all_free_coordinates (prediction_d - target_d)^2 / (9 × sum_batch_free_faces)。

effective batch8，microbatch1，fresh AdamW step0，lr=1e-5，betas=(0.9,0.95)，weight_decay=0，global gradient clip=1，foreach=False；无scheduler，无新EMA。训练train、face-count dropout0.1，两组相同torch CPU/CUDA RNG起点；逐更新记录RNG散列以核对配对。推理eval、不消耗随机流。

官方实际构造路径默认启用24层非重入activation checkpointing，preserve_rng_state=True；配置文本中的gradient_checkpointing=false未被该AutoencoderKL构造接口传入。此次明确沿用实际True，两组一致。重算的内部层调用另述，不冒充新增逻辑样本。BF16 forward、FP32参数/损失/残差，TF32关闭，严格deterministic，SDPA math，microbatch1无padding、mask=None。

预检32真实前向、16反向、2个完整AdamW隔离更新，随后丢弃。覆盖K=1/N=245座面与K=20/N=218靠背；minK、maxN和seat三个选择标准落在同一输入，因此是两种独特输入、八项组别/标准检查，不是八个不同对象。验证零角色与官方同精度输出一致、free梯度到known特征、role梯度、C恢复与导出回读。正式训练重新strict加载官方EMA、fresh optimizer。

## 评价与几何整理

真实评价每个新去噪器从原48份RAW各前向一次；没有对旧POST继续去噪、没有多轮或候选选择。保存完整proposal、恢复C后的preGEO几何与最终POST；原C直接从原模型坐标数组写回，避免缩放往返误差。

三去噪器×48请求=144份POST。四个生成器来源分别汇总，USER_MIX为主要场景。原48份RAW单独参考，不当作144个独立对象。开发面板已多次观察，不称独立泛化测试。

固定GEO_C只执行一次，半径0.015/a=0.04197095609837992；唯一锚点吸附、原free代表点规则、拒绝多锚冲突均沿用旧代码。仅删除规则允许的free精确退化/重复面，无扩大半径、传递聚类、均值漂移、量化、补洞或平滑。C坐标/顺序/绕序/面多重性保持，最终N允许减少并报告。

全开放C源拓扑边界的A-VMR/PMR保留原8档、固定L=9.208138264587333，只查询F角点，严格小于阈值。SOURCE_SUPPORTED_INTERFACE另列，不改原分母。论文尺度兼容性仍UNKNOWN；PMR不是接合成功率。

实际索引网格新增诊断：共享C/F边总数/长度、非流形、绕序；有效边要求恰好两面、非退化、非重复，且固定几何分类是已决合法接触，无正面积重合。绕序另外报告。以全部C面为起点，在有效C/F、F/F共边图上BFS，报告可达free面积/全部free面积；它反映连接延展，不能自动认定语义主体。共点与T接触不计共边。

最终质量完整保留原CF、FF、I及分项、小角、q>10、退化、删面/面积/位移。FF面对通过source_face_ids区分保留后改善、删面涉及、仍非法和新非法，避免分母变化冒充全部几何改善。preGEO分别计算边界距离、匹配、角度/面积，区分神经位移与最终焊接。

合成54前向只比较同形状扰动恢复：free坐标误差、来源接口对应free角点误差、clean不必要移动、C保持；与真实RAW结果分表。全部144份POST保留统一相机完整视图、线框和必要局部图，由AI_PROXY审阅，非盲法人评。

## 资源、写入与停止

仅1个GPU worker，两臂顺序；首次完整AdamW测得约6.90GiB allocated、7.06GiB reserved，另保留驱动/桌面显存与余量。GPU阶段进程墙钟采用保守累计计时，上限10800秒，含加载/保存等阶段开销；CPU准备与离线评分另计。CPU独立审计可并行。

所有运行时写入在本轮OUT内，解析前拒绝符号链接，再检查resolve包含关系；旧数据/权重/报告路径负例在打开写文件前拒绝。文件先写同目录临时文件、回读/哈希后原子替换；旧原件前后核验。

保存500恢复点但不评价，1000是唯一终点。正常完成的检查点/输出核验后跳过。可在完整更新边界保存模型、AdamW、RNG、流位置后恢复；未决或失败模型调用不静默重放。预算/资源/数值阻塞如实记部分完成，不改LoRA、精度、batch或噪声补救。

不训练生成器，不新增loss，不切换默认模型/去噪器/GEO_C，不commit/push。固定任务完成后停止。
