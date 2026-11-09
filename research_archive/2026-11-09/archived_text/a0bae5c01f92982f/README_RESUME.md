# R2_SEAT_QUICK_LEARNABILITY_V1 恢复入口

固定两臂各200更新，EXP67完整step6000与同AdamW恢复，末点6200。SMALL_C200四父各400次曝光；FULL_C200全67父各23或24次曝光。batch8/micro1、4clean/4width、FM与HYBRID0.25和精度不变。集中组每批四父各2次，明确允许重复父。全池对照含座面和靠背；本轮对比集中座面条件曝光整体策略，不单独归因某一因素。

固定4条TRAIN C，每条2现有评价噪声，原C/N/噪声和查询复用。只评价两个200末点与冻结EXP67；不选best、不增加seed。新增16 Euler50和16单次官方D0，RAW全部完成后才D0/GEO_C/来源方向恢复。48逻辑评分stage和逐图审阅，不用唯一来源GT排序。

Windows PowerShell入口（WSL Ubuntu）：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.seat_quick_learnability_v1.run --stage all
```

也可用 `--stage train` / `--stage evaluate`。不要在已有训练进程运行时再次调用。所有模型失败/悬空调用均禁止自动重放。资源不足时等待；INTERRUPTED_SAFE可在重新核验资源后恢复，OOM/FAILED需先核账。只复用已完成同身份输出，不重抽噪声。

本轮用户允许实验约6GiB显存，单GPU worker；训练使用5GiB PyTorch allocator上限，另为CUDA上下文留余量。该上限不是进程显存证明。启动前按4GiB进程保守估计×1.2，再留2GiB全卡余量；更新间监测低余量则边界保存并释放进程。不停止其他用户进程。WDDM若不报告单进程显存，则记NA，并分别报告allocated/reserved、全卡占用及背景差分近似。

100更新仅恢复点，200为固定评价端点。全量权重在checkpoints，推理权重在inference。已有资产只读；不改默认、不commit/push。输入风险见inputs/PREPARATION.md。

本轮是已见条件的小池可学性检查；短预算无信号不证明架构无法学习，集中组变好也不证明未见条件泛化。