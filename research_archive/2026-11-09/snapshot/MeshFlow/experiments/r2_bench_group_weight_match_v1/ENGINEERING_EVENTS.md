# 工程事件（与科学调用分开）

- 准备阶段资源记录首次按脚本文件路径启动，Python未把仓库根加入模块路径，`import r2`失败。未加载模型、未做前向/反向/更新、未抽噪声，资源记录文件尚未创建。随后使用仓库既有模块入口 `-m experiments.r2_bench_group_weight_match_v1.record_resources`，保留此记录。这不是训练恢复或科学重放。
