# R2_SEAM_POOL47_COUPLE_V1

本轮范围已冻结于 `protocol.json`，实际执行记录写入各阶段账本。旧资产只读；不改默认模型、不提交或推送。

1. 保存的 B_LONG_LAYOUT：28 请求 × 1.5% / 2.0%L，共 56 次 CPU 绑定；另作 DEV 有限边距离与 source alias 去向诊断。
2. TRAIN47 + DEV3：冻结的纯 C 角部几何提议，随后才与来源标签比较。该规则独立验证，不加入正式推理。
3. B_POOL47 同一终点和匹配 AdamW 分叉：PLAIN3000 / COUPLE3000，各新增 3000 更新。后者仅增加原 A 的 λ=0.1 couple，使用同一 24000 样本缓存；无去重、无新占用头。
4. 每臂固定 28 请求：同噪声、同 LONG C-only 布局、Euler50、单次官方 D0、原 GEO_C 与来源定向导出；另保留每份 POST 的 1.5%L 绑定结果。全 56 RAW 保存后才开始 D0。

`3000` 是终点预算，不以某个训练 MSE 达标作为停止或延长条件。来源重建只作诊断。FF 受影响面比例、非法面对数量与 I 的固定 C 表面冲突覆盖分别保留，不能互换。

## 恢复入口

在 WSL Ubuntu、项目根目录运行：

```bash
export PYTHONPATH=/mnt/e/MeshFlow/r2/src:/mnt/e/MeshFlow
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.seam_pool47_couple_v1.resume status
```

入口默认只读。训练和采样必须具有已冻结的 dispatch go 文件；失败/未完成调用不自动重放。已完成阶段验证 hash 后复用。实际恢复命令、错误与资源决策随执行补记，最终结论见 `REPORT.md`。
