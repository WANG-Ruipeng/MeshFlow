请基于当前已经完成的零训练调试结果，执行下一阶段：稳定运行配置固化 + 小样本条件可学习性诊断。本次不做新的架构搜索，不扩大为Phase 3，不继续比较G是否优于Q。
工作区E:\MeshFlow，实验experiments/mf_h1_local0，最新数值调试outputs/debug_repeatability_20260928_150208。原官方EMA、Q/G step200和历史输出保持不变。BF16分组scatter求和已定位；严格确定性BF16同进程/新进程重复通过；G缓存仅推理验证。
目标：少量真实训练对象判断普通Q能否学到正确提示到相应目标，而非非零变化。
稳定公共配置：新进程CUDA前CUBLAS_WORKSPACE_CONFIG=:4096:8；torch.use_deterministic_algorithms(True,warn_only=False)；BF16autocast，FP32参数/积分坐标；TF32off，SDPAmath；原时间表/面数条件/CFG/50步。训练、teacher、完整生成共用并在入口断言。官方源码/历史不改。不支持确定性反向时记录停止，禁止静默关闭。Q训练不缓存几何；不跨更新缓存encoded或截梯度。
先零更新补齐有效残差诊断：旧step200和固定teacher/采样状态，3/6/9/12层记录h、r、h_after、下一模块输入dtype；RMS(r)/RMS(h)；r_eff=h_after.float()-h.float()；RMS(r_eff)和RMS(r_eff-r.float())；仅r非零元素中h_after==h比例；correct/mismatch有效残差差；有hook/无hook输出一致。不凭残差小/多元素不变判根因。明确数值路径错误则报告停止后续训练；若仅量化风险则记录跟踪，禁止擅自FP32/scale/init/lr改动。
独立learning_diagnosis入口和新输出，不能偷偷改旧Phase2上限。原16train按manifest顺序前4不同父对象，旧4diagnostic不参加训练。每对象固定edge/face/mixed两边一面，共12组；只几何选提示，同原坐标变换；真实面数mask，不截断复制凑512。
两臂A Q_correct提示匹配目标，B Q_independent提示同类型在4父对象独立均匀抽样，每次重抽，允许1/4正确；不得固定循环或排除正确。
同一新初始化seed10、零出口，不续旧step200不覆盖旧。冻结官方，target/noise/OT/t/N/mask逐步对账；B donor独立RNG不影响目标噪声。AdamW lr1e-4 betas(.9,.95) wd0 clip1 effective8 micro2，原maskedFMvelocityMSE，无新loss/scheduler/EMA。每臂1000更新，总2000上限；保存0/200/500/1000，不选最好终点、不延长。获准执行新增训练须明确不再零训练；冻结、数值失败、中断更新如实计数。
每预定checkpoint固定独立评价噪声，不能用GT耦合训练noise。4train对象×3prompt×2noise完整生成；两模型均同批正确提示并same-noise交叉四格评分。原语内部独立checks到三角表面RMS、Base/Qcorrect/Qindependent同口径、父对象绑定改善、完整表面诊断/窄长面/退化/NaN/Inf；自交水密流形未做NOT_RUN。这是训练对象可学习性非泛化。旧4diagnostic可最终旁路一次但不据此选模型或超参（可省略）。
Teacher固定target/noise配对，t=.05/.2/.5/.8/.95比较off/correct/mismatch；训练时间分布保持，不做时间重权。e=u-v_off,d=v_correct-v_off，同mask归一化统计mean||d||2、2mean<e,d>、Lcorrect-Loff，核对代数；区分残差大小方向。少数预定步记录条件encoder/dynamicMLP/output的梯度和实际参数更新量；第一步上游零梯度可正常，后续长期不更新需报告。
解释：correct明显胜Base和independent且自由生成改善才支持条件可学习性，非泛化/H1优势；两臂相似不证明正确条件收益；teacher改善生成不改善则记差异，不换loss/加步；1000仍不明标当前预算未建立控制；不训G/S/P、不加模块。
只交付一个LEARNING_DIAGNOSIS_REPORT.md、必要结果JSON、code diff、本轮adapter ckpt；不复制官方2GB和旧数据。最终回答：稳定配置是否进入实际路径；加法实际保留多少；correct是否胜independent；新noise完整生成有无收益；下步扩数据还是定位学习。完成停止不自动下一阶段。
