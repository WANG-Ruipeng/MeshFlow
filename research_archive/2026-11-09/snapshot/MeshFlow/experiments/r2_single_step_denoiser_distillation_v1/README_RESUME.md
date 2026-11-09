# README_RESUME

本轮目录：`E:\MeshFlow\experiments\r2_single_step_denoiser_distillation_v1`。
代码：`E:\MeshFlow\r2\pilots\single_step_denoiser_distillation_v1`。
现有环境：WSL Ubuntu `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`，不安装或升级依赖。

## 总入口

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.single_step_denoiser_distillation_v1.run --stage all
```

分阶段入口依次为 `prepare`、`register`、`generate`、`teacher-preflight`、`cache-teacher`、`student-preflight`、`train`、`infer`、`benchmark`、`score`、`gallery`、`finalize`。已有完成收据时，核对身份后跳过，不增加模型调用。`all` 在最终 `completion.json` 已存在时只执行最终完整性核验。

`score` 内只有CPU GEO/评分和缓存误差计算；`gallery` 只渲染实际数组，不运行模型。报告表入口为：

```sh
python -B -m r2.pilots.single_step_denoiser_distillation_v1.report_tables
python -B -m r2.pilots.single_step_denoiser_distillation_v1.audit --stage final
python -B -m r2.pilots.single_step_denoiser_distillation_v1.finalize --verify
```

最后的144条AI_PROXY审阅与REPORT需要真实图像及数据审阅；自动入口不会制造审阅结论，缺少这些材料时最终打包会明确拒绝完成。

## 中断处理边界

`gpu_phases/*.json`、`gpu_calls.jsonl`、逐输入receipt与原子文件共同判断完成。已完成的整阶段可直接重入核验跳过。未闭合模型调用或已经尝试但没有完整产物的阶段，入口会停止；禁止删账本、换噪声或将未闭合调用悄悄重放。

正式训练每100更新保存完整latest，500/1000还保留学生推理权重，最终4000同时保存完整checkpoint和纯推理权重。安全SIGTERM只设置停止标记，等当前完整更新及监测/检查点闭合后记录 `PAUSED_AT_CLOSED_CHECKPOINT`。仅这种边界允许 `train` 校验同源码、同数据流、同完整AdamW/RNG和累计调用数后继续，累计仍然4000更新。任意外部崩溃或半batch不能按这个路径隐式回滚。

达到GPU预算时保存最近完成更新，记录 `STOPPED_GPU_BUDGET` 并停止后续模型调用；不把预算耗尽视为继续运行的授权。数值或科学设置错误保留现场，不扫描学习率/精度/模型/数据比例进行自动补救。

## 本轮工程修复记录

- 生成首次启动在日志函数参数绑定处报错，尚未进入第一次模型前向。`audit/engineering_fix_001/` 保存原始失败、差分和证据：1次预留、0实际F/0输出，约6.94秒占用仍计入总时间。修复日志参数名后使用同一冻结输入继续；未删除失败历史。
- 训练冻结流为0基索引，优化器更新为1基。此接口映射在任何正式训练前修正，未改变样本顺序或步数。
- 并行CPU评分产生1条START日志丢失，48份独立结果和COMPLETE记录完整。`evaluation/compatibility/engineering_fix_001/` 保留原账本与该身份、源码差分和产物哈希；不补造旧START、不重评分。后续使用每job独立事件文件。
- 合成DEV恢复诊断从原始冻结source输入读取完整target_d，教师KD目标不变。该字段读取修正与评分数学源码不变证明一同保留。

## 权重与独立推理

本地最终权重在 `checkpoints/student_inference.pt`；完整训练状态在 `checkpoints/student_final4000.pt`。review ZIP不包含大权重或RAW/NPZ数组，保留路径、哈希和逐例CSV。教师原始大checkpoint从不重新打包。

独立推理接口见 `STUDENT_INFERENCE_API.md`。默认帮助/输入验证不做前向；未来显式 `--execute` 才会产生新调用，当前固定实验不会自动执行。可接任意C掩码，原C逐字节恢复，支持可选原GEO_C和源面/角映射。只需学生权重和小型原架构代码，不需要教师权重。

本轮不替换R2默认生成器或官方D0、不启动OPD/多轮修正/新锚点训练、不自动commit/push。完整任务完成后停止。