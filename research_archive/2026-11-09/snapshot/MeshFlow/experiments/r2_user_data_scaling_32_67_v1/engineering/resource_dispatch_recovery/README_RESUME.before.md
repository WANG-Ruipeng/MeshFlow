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
