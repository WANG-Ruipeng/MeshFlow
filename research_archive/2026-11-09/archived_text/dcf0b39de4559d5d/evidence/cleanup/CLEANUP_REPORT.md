# 2026-10-14 训练前旧检查点清理

按用户明确要求，在本轮任何训练之前清理不再使用的旧权重。已逐文件核对绝对路径、大小及SHA256，删除66个文件，文件名下逻辑大小92.065 GiB；磁盘实测可用空间增加87.692 GiB（硬链接别名不重复释放，期间其他磁盘活动也可能影响差值）。

保留本轮EXP67完整step6000及匹配AdamW/RNG、SMALL/CREF终点推理权重、官方Chair生成器和D0、公共教师及维护中的START/R2等基线。23个大权重重新核对哈希。RAW、评分、报告、日志、原数据和源代码未删除。删除旧优化器检查点的历史分支今后不可继续恢复，终点及历史结果按保留清单存档。

删除范围主要为旧anchor/student/ABC、重复latest、已经结束的中间点，以及不再维护的续训对照；没有处理其他worktree资产。没有递归删除文件夹。磁盘清理不等于释放RAM/VRAM。

明细：deleted_weights.csv；保留：retained_verified.csv；实际执行：deletion_ledger.jsonl；完成收据：completion.json。本次清理后继续已授权的R2_FULL_SEAT_LEARNING_CURVE_V1，未自动commit或push。