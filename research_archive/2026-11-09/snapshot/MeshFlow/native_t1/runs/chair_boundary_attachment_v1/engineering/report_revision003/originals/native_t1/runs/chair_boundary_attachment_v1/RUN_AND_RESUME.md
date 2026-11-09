# 运行与恢复

完整入口验证冻结协议和输入，跳过DONE阶段后继续既定剩余工作；不能借此增加预算。

    wsl -d Ubuntu --cd /mnt/e/MeshFlow -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -u -m experiments.mf_h1_local0.chair_boundary_attachment_v1.run_all --through all

阶段：prepare / precheck / train / sample / score / all。每个GPU worker先取共享锁；一次一个。不要删他人锁或杀无关进程。

发生错误先读controller_status、workers、logs、ledger。存在CALL_BEGIN无RETURN/FAILED、重复调用或未完成UPDATE_BEGIN时不能盲重启；先保留原件/traceback/diff并核对已返回调用。

训练只能恢复完整checkpoint、AdamW、RNG、stream cursor；不重抽batch、不重置优化器、不拼接数学版本。固定1000终点不挑best；每200步恢复点只是恢复用途。

已有raw的纯CPU重评分入口：

    wsl -d Ubuntu --cd /mnt/e/MeshFlow -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.chair_boundary_attachment_v1.evaluation --label START_HYBRID

label只可替换为冻结的四个*_END。评分版本/input/hash不同不得覆盖旧数值结果；工程修订先保存原件，然后独立版本目录、CPU账本。

最终纯CPU入口：aggregation.finalize(out)、gallery.run(out)、reporting.run(out)。图库默认要求160个P1齐备；allow_partial仅阶段预览。报告只读现有文件并校验checkpoint文件hash，不加载模型。

当前报告状态：INCOMPLETE。固定任务完成后停止，不借用节省预算增加候选、重试或训练。
