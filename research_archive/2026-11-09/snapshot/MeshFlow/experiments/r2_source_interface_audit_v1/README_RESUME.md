R2_SOURCE_INTERFACE_AUDIT_V1

结果入口：REPORT.md。此目录为独立审计，原MINI、原实验和原数据只读。最新用户反馈是本轮测量授权，突破上一轮MINI“只投影旧表”的范围仅限来源接口诊断；不包含训练、模型推理或新生成。

主要表：

- source_interfaces.csv：1032行=258条件×2索引域×2方向状态。
- AVMR_PMR_all8.csv：2064行=258×8。
- boundary_vertices.csv：3055行，每个C源边界顶点到Y最近顶点。
- C_Y_edge_details.csv：5970行，双索引域，含原面ID、拒绝原因及方向。
- DEV_source_vs_existing_POST.csv：768行=384既有POST×2来源拓扑参照；不是768份新模型输出。
- DEV_task_comparison.csv：12行，4任务×3模型。
- source_strict_incomplete_conditions.csv：7条来源严格PMR未满条件，无样本删除。
- statistics/：源问题事件/重复目标曝光/小角的纯算术复核。
- exact_count_audit/：整数分数精确符号复核。
- conditions/<id>/result.json：258个完整新接口诊断，含逐边图、原方向和整理方向、来源身份。
- protocol/registration.json：冻结预算、输入/hash及函数定义；protected_after.json为实际只读验证。
- code/frozen/：原实验实际函数文件快照；audit.py只执行选定纯NumPy函数，不导入原实验总入口。

新几何测量主键(condition_id,topology_domain,direction)；阈值表(condition_id,setting)。SOURCE_INDEX是真实来源索引；EXACT_COORD_UINT32只是坐标逐位相等索引对照，无容差合并。EXPORT_ORIENTED只是同一固定C规则的内存方向诊断，不改磁盘源网格。source中的free指Y。

在本机恢复或验证：
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/experiments/r2_source_interface_audit_v1/code/resume.py

已完成项只验证并复用；已开始却没有结果或BLOCKED项拒绝静默重试。各原资产绝对路径是本机测量的输入依赖；表格和审阅包可独立读取。原始主入口code/audit.py是本次实际冻结执行版本，保留其hash，不改写旧登记。统计两个子目录各自含可复算入口。

可达面积是合法共边图到Y的面积比例；不证明整个可达区域无相交、不保证外法线/水密、不宣称语义采用。C开放边界不是必接边界标签。计数分母和父等权宏均值分开，fraction乘100才是百分比。原报告、旧评分、原MINI及全部未知状态均保留。
