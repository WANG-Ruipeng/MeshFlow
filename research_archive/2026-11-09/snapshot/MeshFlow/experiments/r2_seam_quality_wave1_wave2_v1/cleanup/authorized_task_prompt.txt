# R2_SEAM_QUALITY_WAVE1_WAVE2_V1 实验执行任务书

**实验代号**：`R2_SEAM_QUALITY_WAVE1_WAVE2_V1`

**核心目标**：在保留 `SEAM` 共享原边表示已取得的真实接续收益前提下，通过 **Wave 1（零训练轨迹中间点与权重空间内插检查）** 与 **Wave 2（从 `B_POOL47_step1000` 分叉的单变量训练改动）**，检验能否压低 DEV 上的 `bulk` 自交（FF）、表面冲突（I）与散裂程度（连通分量、开放边及可见裂口），同时保住或提升 `seam → bulk` 的大面积主体接续。

**总则**：所有阶段严格遵守单变量控制、确定性运行、完整 124 数值字段保留与非盲 `AI_PROXY` 图审规范。不修改维护默认模型，不自动执行 Git commit/push。

---

## 阶段 0：前置权重白名单清理（Pre-Execution Cleanup）

在启动任何计算前，严格按白名单清理已结项分支的中间 rolling 恢复点与冗余训练终点（保留所有推理导出与正式参照），释放约 **23.32 GiB** 磁盘空间。

### 0.1 允许删除的白名单（共 16 份文件）

1. **近期 rolling 与中间恢复点（12 份，约 17.49 GiB）**：
* `r2_seam_pool47_couple_v1`：`COUPLE3000_step1000.pt`、`COUPLE3000_step2000.pt` 以及 `PLAIN3000` / `COUPLE3000` 的 rolling 恢复文件（共 4 份）。
* `seam_slot_c200_v1`、`r2_seam_quality_four_v1`、`r2_seam_abc_v1`：各实验目录下的旧 rolling 恢复文件（共 8 份）。


2. **已结项分支的完整训练终点（4 份，约 5.83 GiB，前提是核验同目录 `inference/*_generator.pt` 存在且 SHA256 完好）**：
* `r2_seam_quality_four_v1/checkpoints/LOCAL_NOISE_step1000.pt`
* `r2_seam_quality_four_v1/checkpoints/ANTI_COVER_step1000.pt`
* `r2_seam_abc_v1/checkpoints/A_COUPLE_step1000.pt`
* `r2_seam_abc_v1/checkpoints/C_SHARED_TIP_step1000.pt`



### 0.2 绝对只读保护清单（严禁修改或删除）

* **本次实验核心起点与输入权重**：
* `E:/MeshFlow/experiments/r2_seam_abc_v1/checkpoints/B_POOL47_step1000.pt`（Wave 2 唯一训练起点，Wave 1 融合起点）
* `E:/MeshFlow/experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step1000.pt`（Wave 1 候选）
* `E:/MeshFlow/experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step2000.pt`（Wave 1 候选）
* `E:/MeshFlow/experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step3000.pt`（Wave 1 融合终点与主参照）
* `E:/MeshFlow/experiments/r2_seam_pool47_couple_v1/checkpoints/COUPLE3000_step3000.pt`
* `E:/MeshFlow/experiments/r2_seam_quality_four_v1/checkpoints/LONG_step1000.pt`（接续主参照）
* `SEAM_SLOT_C200_step200.pt`、`FULL_S6000`、`EXP67`、`START`、官方 `Chair EMA`、官方 `D0 (0030000.pt)` 及所有 `inference/` 下的导出权重、占用头、RAW/POST 缓存与评分源码。


* **清理审计**：生成 `cleanup/deleted_weights.csv`（记录路径、字节数、删除前 SHA256）与 `cleanup/retained_verified.csv`（重新校验保留大权重 SHA256）。

---

## 公共评估与后处理冻结合同（Wave 1 & Wave 2 通用）

所有新增候选模型（Wave 1 的 3 个模型、Wave 2 的 2 个模型）均采用完全一致的评估面板与后处理流水线：

1. **评估面板（28 请求）**：
* **熟悉 TRAIN（16 请求）**：`73b369b9`、`3f741759`、`85b16941`、`8c9b2023` × 原 4 份固定高斯噪声。
* **未见 DEV（12 请求）**：`2d08a64e`、`610b1621`、`608b46b0` × 原 4 份固定高斯噪声。


2. **推理占用布局**：
* 全部固定复用 **`LONG` 已缓存的完整 C-only 预测布局**（TRAIN 激活 `22/6/9/11`，DEV 激活 `16/4/4`）。不调用新占用头前向，不偷看来源标签。


3. **采样与精度配置**：
* 50 步 Euler 积分（$\Delta t = 0.02$），BF16 网络前向，FP32 参数与状态积分；严格启用确定性算法（`torch.use_deterministic_algorithms(True)`，TF32 关闭，`SDPA math`，`CUBLAS_WORKSPACE_CONFIG=:4096:8`）。
* $C$ 原面及激活 `seam` 槽的前两个固定角点在全部 51 个积分状态逐位保持原 FP32 坐标。


4. **标准后处理与局部簇绑定（每个候选均产出 3 个阶段结果）**：
* **`RAW`**：Euler50 原始输出。
* **`POST`**：全部 RAW 落盘后，执行单次官方 `D0` $\to$ 恢复 $C$ 与 `seam` 硬锚点（`PRE_GEO`） $\to$ 原 `GEO_C` 合并与清理 $\to$ 来源 $C$ 角序恢复与固定 $C$ 的 free 定向导出。
* **`BOUND015`**：在 `POST` 代表点、真实 `GEO_C` 角点簇与 `PRE_GEO` 原角点基础上，运行已冻结的 **`1.5%L` 局部 `seam–bulk` 簇绑定**（两自由簇向中点赋值，含 $C$ 锚点簇不动，受影响簇全部 `PRE_GEO` 原角点总位移 $\le 1.5\%L$，零新增非法 CF 熔断，形状/法向恶化熔断，直接记录 `TIP` 与 `LATERAL_EDGE` 接受数及新旧非法 FF 面对进出）。


5. **只读基线参照（直接复用已核验缓存，不重跑）**：
* `LONG`（+ `LONG_BOUND015`）
* `B_LONG_LAYOUT`（+ `B_BOUND015`，代表从 B 起跑的 +0 步状态）
* `PLAIN3000`（+ `PLAIN3000_BOUND015`，代表从 B 起跑的 +3000 步状态）



---

## Wave 1：零训练轨迹中间点与权重空间内插实验

**科学问题**：在 `B_LONG_LAYOUT`（+0 步，DEV FF $15.16\%$、连通分量 $14.58$）到 `PLAIN3000`（+3000 步，TRAIN FF $13.71\%$、DEV CF/I 改善且 `608/n3` 破冰，但 DEV FF 升至 $19.12\%$、连通分量升至 $19.67$）之间，真实优化轨迹的早期检查点或起点—终点权重弦中点，能否在不重训的前提下取得更优的“接续—表面完整性—低 FF/I”平衡？

### 1.1 候选模型定义（共 3 个零训练模型）

1. **`PLAIN_S1000`**：
* 直接从 `PLAIN3000_step1000.pt` 提取 `generator` 状态字典（对应从 `B_POOL47_step1000` 续训 1000 步，global 14000），导出为独立推理权重 `inference/PLAIN_S1000_generator.pt`。


2. **`PLAIN_S2000`**：
* 直接从 `PLAIN3000_step2000.pt` 提取 `generator` 状态字典（对应从 `B_POOL47_step1000` 续训 2000 步，global 15000），导出为独立推理权重 `inference/PLAIN_S2000_generator.pt`。


3. **`WISE_B_PLAIN_050`（权重空间等权线性内插）**：
* 读取 `B_POOL47_step1000.pt` 的生成器参数 $\theta_{\text{B}}$ 与 `PLAIN3000_step3000.pt` 的生成器参数 $\theta_{\text{P3000}}$（严格校验 194 个张量键名、形状与 dtype 完全一致，不混合优化器状态）。
* 在 FP32 下逐张量计算固定中点（$\alpha = 0.5$，不扫描其他 $\alpha$）：

$$\theta_{\text{WISE}} = 0.5 \cdot \theta_{\text{B}} + 0.5 \cdot \theta_{\text{P3000}}$$


* 导出为 `inference/WISE_B_PLAIN_050_generator.pt` 并登记 SHA256。



### 1.2 Wave 1 预算与执行

* **模型训练/反向更新**：0。
* **新增生成与去噪**：3 模型 × 28 请求 = **84 RAW**（4200 次生成器前向）+ **84 次单步 D0**。
* **新增 CPU 绑定与评分**：84 份 `RAW` 评分 + 84 份 `POST` 评分 + 84 份 `BOUND015` 绑定与评分 = **252 个评分单元**。
* **Wave 1 观察与门禁**：
完成 Wave 1 评分后生成阶段对比表（汇总 `B(0) → S1000 → S2000 → S3000` 曲线及 `WISE_050`）。无论 Wave 1 是否找到更优折中点，均继续执行 Wave 2 以独立检验训练机制改动；但 Wave 1 的最优检查点将作为 Wave 2 相同步数下的直接同预算基准（特别是 `PLAIN_S1000` 与 `PLAIN_S2000` 直接作为 Wave 2 在 1000 步/2000 步的零改动等预算对照！）。

---

## Wave 2：从 `B_POOL47_step1000` 分叉的单变量训练实验

**核心设计原则**：既然 Wave 1 已经完成了 `PLAIN_S1000`（+1000 步）和 `PLAIN_S2000`（+2000 步）的完整评测，Wave 2 的两个新训练臂无需再跑满昂贵的 3000 步，而是**固定训练预算为 +1000 步（主评价点，直接与 `PLAIN_S1000` 逐样本 100% 配对比较）**，这样每臂仅需 8000 个训练实例（约 25 分钟 GPU），即可在严格同预算下完成因果检验！

### 2.1 共同训练合同（与 `PLAIN3000` 前 1000 步严格配对）

* **唯一起点**：`E:/MeshFlow/experiments/r2_seam_abc_v1/checkpoints/B_POOL47_step1000.pt`（含活生成器、匹配 AdamW 状态与 RNG，从 global 13000 训练至 global 14000）。
* **数据流与曝光配对**：直接只读复用 `seam_pool47_couple_v1` 训练计划的前 1000 步（前 8000 个实例，1000 × batch 8）及对应的只读受限 OT 缓存、宽度增强参数、时间 $t$ 与高斯/混噪数组。47 父均衡曝光（每父约 170 次）。
* **保持不变项**：`SEAM` 单边 owner 表示、学习率 $1\times 10^{-5}$、全局梯度裁剪 $1.0$、不加 `COUPLE` 项、不更新也不调用占用头。

---

### 2.2 训练臂 A：`CLEAN_BULK_FM`（源内部非法碰撞面降权训练）

* **检验假说**：`TRAIN47` 中有 31 个父对象的来源目标 $Y$ 内部自带 1138 对非法 `Y/Y` 自交面对。在全参数续训中，对这些本来就互相穿插的目标 `bulk` 面施加全额速度回归，是否加剧了生成的 `bulk–bulk` 自交（FF）与散裂？在保持 47 父全覆盖（不丢弃 `85`、`8c` 等父对象）的前提下，仅对源 $Y$ 中已确认参与非法自交的 `bulk` 目标面降低回归权重，能否改善 `POST FF` 与表面连续性？
* **唯一改动（面级静态质量加权 FM）**：
1. **静态源面标签提取（训练前 CPU 预计算并核验）**：
读取 `TRAIN47` 已核验的来源面对分类缓存（31 父、1138 对确认非法 `Y/Y` 面对，以及 3 对 `UNKNOWN` 面对）。对于每条训练条件 $c$ 的目标补集 $Y_c$ 中的每一张源自由面 $m \in Y_c$：
* 若源面 $m$ 在 $Y_c$ 内部参与了任何一条确认非法 `Y/Y` 面对或 `UNKNOWN` 面对，标记 `is_corrupted_source_face[m] = True`；否则为 `False`。
* **保护接口规则**：所有作为 `seam` owner 的源接口面（360 张），其自由第三点权重固定保持为 $w = 1.0$（不降权，确保 `seam` 接口学习强度与 `PLAIN` 完全一致）。


2. **随受限 OT 搬运权重**：
对于每个训练样本中的常规 `bulk` 槽位 $f$，根据受限 OT 配对到的目标源面身份 $\pi(f)$，赋予静态标量权重：

$$w_f = \begin{cases} 0.2, & \text{if } \text{is\_corrupted\_source\_face}[\pi(f)] = \text{True} \\ 1.0, & \text{otherwise} \end{cases}$$



*(固定常数 $\omega_{\text{dirty}} = 0.2$：保留微弱位置牵引防止槽位飞向无穷远，但大幅削弱其穿插几何对速度场的梯度主导；不扫描该超参数。)*
3. **加权有效自由标量归一化**：
设槽位 $i$ 的有效自由坐标维数为 $d_i$（激活 `seam` 槽 $d_i = 3$，权重 $w_i = 1.0$；常规 `bulk` 槽 $d_i = 9$，权重 $w_i \in \{0.2, 1.0\}$；已知 $C$ 槽 $d_i = 0$）。整个有效 batch（8 个样本）的流匹配损失定义为：

$$\mathcal{L}_{\text{CLEAN\_FM}} = \frac{\sum_{b=1}^{8} \sum_{i \in \text{free}_b} w_{b,i} \sum_{c=1}^{d_{b,i}} \big(v_\theta(X_t)_{b,i,c} - u_{b,i,c}\big)^2}{\sum_{b=1}^{8} \sum_{i \in \text{free}_b} d_{b,i} w_{b,i}}$$


4. **诊断记录**：每步额外记录干净 `bulk` 面、降权脏 `bulk` 面与 `seam` 第三点各自的未加权速度 MSE，以便与 `PLAIN_S1000` 同窗口直接对比。



---

### 2.3 训练臂 B：`SURGICAL_LATE4`（冻结前 8 层主干、仅微调后 4 层与角色/几何接口）

* **检验假说**：`B_POOL47_step1000` 在 DEV 上已具备较好的整体凝聚力（连通分量 $14.58$、FF $15.16\%$、主体完整 $12/12$），而继续全参数微调时，浅中层自注意力为迎合 `[1, 1, t]` 刚性边界发生的全局表征漂移，导致未见 DEV 条件下的 `bulk` 拆散成 $19.67$ 个分量。锁死前 8 个 DiT Block 的全局骨架表征，仅训练最后 4 个 Block 及条件/输出接口，能否在降低 `seam` 第三点误差和 `CF/I` 的同时，保住 `B` 的低散裂与低 DEV `FF`？
* **唯一改动（参数可训练掩码与优化器子集恢复）**：
1. **冻结范围**：
将生成器的 `blocks.0` 至 `blocks.7`（前 8 个完整 DiT Block）的所有参数设为 `requires_grad = False`。
2. **可训练范围**：
保留以下模块为 `requires_grad = True`：
* 输入坐标/时间/角色/Geo 嵌入层（`x_embedder`, `t_embedder`, `role_embedding`, `geo_encoder` 等前处理层）；
* 深层主干 `blocks.8`、`blocks.9`、`blocks.10`、`blocks.11`（最后 4 个 DiT Block）；
* 最终输出层 `final_layer`。


3. **优化器精确状态继承（严禁重置动量）**：
从 `B_POOL47_step1000.pt` 加载完整模型权重后，遍历原优化器 state_dict，**对所有 `requires_grad = True` 的参数，按精确参数身份 100% 继承其在 step 1000 时的 AdamW 一阶/二阶动量（`exp_avg`, `exp_avg_sq`）与 `step=13000` 时钟**；冻结参数移出优化器更新列表（或确保 `grad=None` 零更新，核验训练前后冻结层 100% 逐字节不变）。
4. **损失函数**：与 `PLAIN` 完全相同的标准未加权 `SEAM` 流匹配损失。



### 2.4 Wave 2 预算与执行

* **训练预算**：两臂各 1000 次更新（8000 前向、8000 反向），保存 `CLEAN_BULK_FM_step1000.pt` 与 `SURGICAL_LATE4_step1000.pt` 及独立推理导出。
* **采样与后处理**：两臂 × 28 请求 = **56 RAW**（2800 次生成器前向）+ **56 次单步 D0** + **56 份 `POST**` + **56 份 `BOUND015**` = **168 个评分单元**。

---

## 阶段 3：统一汇总、图审与晋级验收标准

### 3.1 完整对比主表（8 个模型 × 3 个阶段 `RAW / POST / BOUND015`）

在 `TRAIN (16)` 与 `DEV (12)` 上分开汇总全部 **124 个数值字段**，重点并列以下 8 组模型：

1. `LONG`（历史 4 父接续参照）
2. `B_LONG_LAYOUT`（+0 步起点参照）
3. `PLAIN_S1000`（Wave 1：+1000 步普通续训，**作为 Wave 2 的直接等预算基准**）
4. `PLAIN_S2000`（Wave 1：+2000 步普通续训）
5. `PLAIN3000`（历史 +3000 步普通续训终点）
6. `WISE_B_PLAIN_050`（Wave 1：B 与 PLAIN3000 权重 0.5 内插）
7. `CLEAN_BULK_FM_S1000`（Wave 2：+1000 步源碰撞面降权）
8. `SURGICAL_LATE4_S1000`（Wave 2：+1000 步锁前 8 层微调）

### 3.2 必看核心指标矩阵（拒绝单一均值掩盖代价）

每组必须同时报告：

* **接续与可达性**：`bulk 可达面积 ≥30% 请求数`、`free_only_reached_bulk_area_fraction`（均值/中位数）、`legal_seam_bulk_edge_count`、`C_F_valid_edge_count`、`BOUND015` 下的 `TIP` 与 `LATERAL_EDGE` 接受数。
* **冲突与自交**：`FF_illegal_affected_free_fraction`、`FF_illegal_pair_count`（拆分 `bulk-bulk` / `seam-bulk` / `seam-seam`）、`CF_illegal_affected_free_fraction`、`ConflictCov_C (I)`（拆分 `cross` 与 `dup`），以及 `BOUND015` 的新增/消除非法 FF 面对账本。
* **散、裂、碎与形状诊断**：`EXACT_COORD_face_components`（连通分量数）、`EXACT_COORD_boundary_edges`（开放边数）、`EXACT_COORD_nonmanifold_edges`（非流形边数）、`free_minimum_angle_lt5_fraction`、`free_q_gt10_fraction`、`retained_free_faces` 与 `deleted_free_faces`。

### 3.3 实际图审合同（`AI_PROXY`）

* 对新增的 5 个模型（Wave 1 三模型 + Wave 2 两模型）在 `POST`（及关键 `BOUND015` 变化例）上生成与此前同尺度、同相机（双透视视角 + 正侧有限剖面）的配对图集。
* 逐例记录 `主体完整`、`C 座面采用`、`明显碎片/折片`、`可见裂口` 的 `YES / NO / UNCERTAIN` 计数，重点核查 `2d/n3`、`608/n3`、`610/n1,n2`、`73/n1,n2`、`3f/n2`、`85/n1` 等关键请求。

### 3.4 科学判读与停止规则

1. **Wave 1 判读**：若 `PLAIN_S1000`、`PLAIN_S2000` 或 `WISE_B_PLAIN_050` 在 DEV 上能将 `POST FF` 控制在 $\le 16.0\%$、裂口/碎片不劣化于 `B_LONG_LAYOUT`，同时降低 `CF/I` 或保留 `PLAIN` 的接续增量，则确立为比 `PLAIN3000` 更优的训练停点/融合方案。
2. **Wave 2 判读（严格对比等预算 `PLAIN_S1000`）**：
* 若 `CLEAN_BULK_FM_S1000` 在 `TRAIN` 和 `DEV` 上的 `bulk–bulk` 非法面对数及 `FF` 低于同步数的 `PLAIN_S1000`，且未引起额外删面或裂口，则证实“源目标内部自交污染是续训 FF 恶化的真实成因之一”。
* 若 `SURGICAL_LATE4_S1000` 在 DEV 上的连通分量数、开放边数和 `FF` 显著低于 `PLAIN_S1000`，同时 `seam` 第三点误差和 `CF/I` 正常下降，则证实“冻结前 8 层主干能有效抑制多父 SEAM 微调中的主体散裂”。


3. **停止边界**：本任务书全部 5 个候选模型的采样、D0、绑定、评分、图审与独立执行审计完成后立即写入 `completion.json` 并停止，不自动开启组合训练或追加步数。