本轮是用户批准的 A-LAYOUT-VALUE 布局诊断。原 AE/FM、原 A 权重、官方核心代码均保持不变；数据是 16 个已见音箱的固定主场，不是生成样本或留出集。

工作树：E:/MeshFlow/.worktrees/a-layout-value-20260924
WSL：/mnt/e/MeshFlow/.worktrees/a-layout-value-20260924
Python：/home/warpwang/miniforge3/envs/meshflow/bin/python
数据：downloaded_data/layout_closed16_512_20260924
本轮输出：output/layout_value_20260924_01

实验定义以 RUN_PROTOCOL.json 为准，写于真实对象结果产生前。前四个 UID 用于接口与耗时检查，后续沿用同一配置。每个对象六条对照；有限预算优化两种初值各 500 次或 300 秒，不代表全局最优。A 初值的生成和可行化成本计入总时间。

方法解释：

- Fixed、原 A 与两种方向优化共享固定连接和相同 F4096 分片主场。接受方向后重新查询半径。
- 原 A 是旧教师路径训练的冻结网络，未为本轮三维角度约束训练。
- 优化使用 2048 参考面积样本及每个输出三角形的三点面积加权积分。活动最近面和球面片区离散选择，片内 float64 自动微分。
- 最小角及 q05 相对 Fixed 不降低（1e-6 度舍入容差）。这是本轮严格诊断合同，不是通用应用质量标准。
- A 初值不满足约束时显式恢复可行性。失败保持失败，不回退 Fixed。非光滑角度约束可能使局部优化停滞；不能以此证明好布局不存在。
- VTK QEM 来自原交接包。受约束简化使用实际安装的 CGAL6.1.1 classic-plane policy，附新三角形法线、最低角及精确相交拒绝。此版本无公开 plane-and-line policy，未冒称该方法。全局 q05 在终点审核，不宣称每次collapse均满足。
- QEM输入均为相同 F4096 主场。未达512面保留实际面数，不能把较多面数带来的误差优势用于等预算排名。
- 独立评价每方向8192面积样本，距离到实际三角面，参考包围盒统一归一。采样最大值不叫精确Hausdorff。
- 公共端点审计以存储float64坐标的精确谓词检查相交；数值方向guard不叫任意输入/连续轨迹形式认证。
- 原始单次运行时间全部保留。HOT_TIMING.json另报Fixed/A/VTK的3次热测及重复网格哈希；优化只有原始有预算轨迹。
- 表格保留所有失败、质量不满足、面数不满足。16对象都是开发集，无MeshFlow胜负、SIGGRAPH录用或泛化结论。

重放已完成运行只会跳过完成结果。要重新实验须使用新输出路径。以下命令从工作树根的WSL运行：

    PY=/home/warpwang/miniforge3/envs/meshflow/bin/python
    $PY -m experiments.a_radial.layout_value.run --run output/layout_value_20260924_01 --selection smoke --arms fixed a vtk_qem
    $PY -m experiments.a_radial.layout_value.run --run output/layout_value_20260924_01 --selection smoke --arms opt_fixed opt_a
    $PY -m experiments.a_radial.layout_value.run --run output/layout_value_20260924_01 --selection smoke --arms cgal_constrained
    $PY -m experiments.a_radial.layout_value.run --run output/layout_value_20260924_01 --selection remaining
    $PY -m experiments.a_radial.layout_value.benchmark_hot --run output/layout_value_20260924_01
    $PY -m experiments.a_radial.layout_value.summarize --run output/layout_value_20260924_01 --recheck --plots

最后两条生成新timing/figures文件，已有目录时拒绝覆盖。源码在 experiments/a_radial/layout_value，原包代码没有被替换。invocations记录每次调用的Python源码哈希；CGAL每个backend另存原生源码/二进制哈希。

安装新增：numba、vtk、pytest、rtree、matplotlib；WSL编译工具、CGAL/Eigen及GMP/MPFR。没有重装显卡驱动或PyTorch。

已运行正确性检查（详见 CHECKS.json）：
原包20通过，初始2跳过随后真实loader/OT和AE CUDA测试2通过；新增几何9、优化8、CGAL3通过。AE CUDA测试不是完整EquiDiT前反向；本轮未训练官方生成模型。

兼容性记录：交接包verify_bundle.py调用hashlib.file_digest，使用WSL系统python3完成197项校验；研究代码继续使用Python3.10的meshflow环境。原生CGAL的topology_noncollapsable计数来自OnNonCollapsable回调，包含过滤拒绝，不能作为纯拓扑拒绝数解释。
