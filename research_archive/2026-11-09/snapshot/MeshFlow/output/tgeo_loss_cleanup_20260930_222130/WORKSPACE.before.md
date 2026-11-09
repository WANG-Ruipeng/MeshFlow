# 当前工作区：Native T1 + 可选 T_geo

日常维护入口在 [native_t1/README.md](native_t1/README.md)。纯 T1 仍为默认基线；T_geo 是从同一纯 T1 起点添加单面几何编码的候选，不包含 T_graph、FC/FF 辅助损失、Guidance 或后处理。

- `python -m native_t1 prepare`：由外部官方数据生成固定两对象 N112 / K4/8/12 输入。
- `python -m native_t1 train`：显式预算的纯 FM 全量训练；默认纯 T1，指定 `--context-encoder geo` 启用 T_geo。
- `sample --profile trained`：用户训练的纯 T1 checkpoint。
- `sample --profile trained-geo`：用户训练的 T_geo checkpoint。
- 原 `a-continue` 默认 profile 保留本地累计1500步 A_continue 的固定身份，不代表自动采用最新训练终点。

稳定路径：严格确定性、关闭 TF32、SDPA math、BF16 主干前向、FP32 参数和积分、C 固定及原50步 Euler。T_geo 几何分支为 FP32，每次前向重算，不缓存训练条件，不输入自由 GT。

当前本地候选证据在 `native_t1/runs/geometry_context_revised_20260930_153000/`。正向信号限于两个训练对象的新片；原片表现及穿刺/边界质量存在取舍。公开摘要见 `native_t1/docs/tgeo_pilot.md`。

`experiments/` 和 `native_t1/runs/` 中的历史权重、数据、原始输出按原路径保留且忽略上传。本次整理前的完整小型源码快照及退出维护的实验脚本在 `output/tgeo_publish_20260930_165000/`，供历史复现；公共包不依赖该目录。

本轮整理不新增训练或完整生成。远端只提交维护代码、测试、必要配方和文档；官方权重、模型终点、数据、缓存、运行日志不提交。
