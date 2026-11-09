# 运行与恢复入口

本轮独立实验 SMALL32 / EXPANDED67，共同6000步已由真实预检锁定。旧12h轮仍为DATA_SCALE_BLOCKED。当前状态以controller/status.json、training/*/status.json和实际账本为准。

## 只读状态
```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/pilots/user_data_scaling_32_67_v1/resume_all.py --stage status
```
## 同一协议入口
```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/pilots/user_data_scaling_32_67_v1/resume_all.py --stage run
```
已完成dispatch仅在身份匹配时跳过。失败/中断目录不会自动重放，必须核对失败位置、持久账本、最后完整检查点和输入游标；不能跳坏batch或从旧点重复确认更新。当前已有运行时不可再启动第二个控制器，所有权锁会阻止并发。

6000步每2000物理段轮换SMALL/EXPANDED，每500保存完整latest，固定4000与最终完整权重及推理导出保留。截止条件在registration.json；20:15:34北京时间停止新训练更新，23:15:34停止新科学任务，不改截止时间续跑。

所有RAW之后统一单D0常驻处理，再原评分和完整图集。入口在实际图审缺失时停止于WAITING_ACTUAL_VISUAL_REVIEW；由当前会话实际审阅图片、记录AI_PROXY覆盖/未审，之后 --stage aggregate 完成父级汇总。图片已渲染不等于已审阅。报告与小包在这一阶段后生成。

首次控制器日志参数错误发生在Popen之前，保存在engineering/controller_dispatch_fix；新的evaluation_prepare_v2只校验已冻结的评测输入，未重抽噪声。训练读取器的C-only NPZ接线修复另存engineering/reader_schema_fix。源码/模型科学定义未因此改变。

禁止用历史USER_MIX替代新训练臂，禁止改数据池/噪声/默认/后处理参数，无commit/push。

## 2026-10-12 15:37 资源拦截后的显式恢复

SMALL6000完整完成；EXPANDED67仍为4000。原下一段在创建空派发目录后、启动子进程前被资源余量门槛拦截，未发生模型调用或优化器更新。旧空目录和阻塞记录保留。根据完整训练段的既有实测，新增进程保守估计改用5120 MiB；原15%/2GiB安全余量不变，WDDM进程峰仍NA。证据与旧状态副本见engineering/resource_dispatch_recovery。

显式恢复命令（保留同一segment_6000训练attempt、完整AdamW/RNG/输入游标）：

```bash
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/pilots/user_data_scaling_32_67_v1/recover_resource_dispatch.py --execute
```

该入口只允许闭合的EXPANDED4000状态，重新检查资源后执行尚未启动的4000→6000段；完成后调用原总入口继续同一评测流程。两臂都完成后，原resume_all.py --stage run也会跳过已完成训练。任何新的失败仍需检查记录，不自动重放。
