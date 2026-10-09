# MeshFlow 仓库的三条路径

## 目录与职责

```text
MeshFlow/
├── train.py                    原版无条件训练
├── flow_matching.py            原版 FM 路径/运输
├── datasets/mesh_dataset.py     原版数据加载
├── models/equidit.py            原版 EquiDiT 骨干
├── configs/snet/                原版类别配置
├── inference.py                原版无条件生成
├── native_t1/                  已有条件扩展，START 默认采样
│   ├── __main__.py              旧公开 CLI
│   ├── train_cli.py, training.py 旧 N112 训练
│   ├── model.py, context_geometry.py
│   ├── chair_model.py, chair_checkpoint.py, chair_sampling.py
│   ├── recipes/chair_hybrid_start.json
│   └── postprocessing.py, postprocess_cli.py   可选保持 C 的 CPU 后处理
└── r2/                         当前 R2 配方的独立维护入口
    ├── __main__.py              仓库根目录 python -m r2
    ├── pyproject.toml           可编辑安装 / meshflow-r2
    ├── src/meshflow_control/
    │   ├── r2_cli.py            只公开 R2 命令
    │   ├── r2_spec.py           配方说明与保留终点 hash
    │   ├── r2_runs.py           资产检查、单配方登记、训练入口
    │   ├── r2_checkpoints.py    主实验/确认实验 checkpoint 适配
    │   ├── r2_sampling.py       单次采样、预算账本
    │   ├── models/native.py    Native/T1、role、Geo 的模型包装
    │   ├── models/geometry.py  C 几何条件分支
    │   ├── models/backbone/    随包保留的官方骨干及来源说明
    │   ├── data/recipe_factorial.py    C20/C40 混合冻结输入
    │   ├── data/schedule40.py, stream.py, dataset.py
    │   ├── training/recipe_factorial.py 6000 步训练/状态保存
    │   ├── training/schedule40.py       复用 COMMON 的 FM 更新算术
    │   ├── training/coupling.py, losses.py
    │   ├── training/recipe_confirmation.py 确认实验 checkpoint 兼容
    │   ├── runtime.py, precision.py
    │   └── sampling.py, mesh_io.py      Euler50 与保真 RAW 导出
    ├── tools/                  历史注册需要的源码文件
    ├── tests/                  CPU 协议和入口测试
    └── docs/                   配方、来源、复现和验证
```

`python -m native_t1 train` 是旧 N112 训练器，**不是 START 续训，也不是 R2**。根目录 `train.py` 是无条件训练。R2 的正式入口是 `python -m r2 train`。

## R2 一次更新如何进行

1. `r2_runs.train` 检查登记的源码、资产、R2 配方和续训状态，取得当前运行目录的 GPU 所有权锁。
2. `training.recipe_factorial.train_recipe` 创建官方权重初始化的模型和新的 AdamW，或加载同一运行的完整中断状态。
3. `data.recipe_factorial.RecipeFactorialStream` 从冻结计划取一批 8 个不同父对象。每批四个 C20、四个 C40，分别一半原样、一半宽度增广。
4. 读取已冻结的 free-only OT 配对，再形成 HYBRID 噪声与 FM 路径。训练器逐样本核对实际输入/标签哈希。
5. 调用 `training.schedule40.update` 的 COMMON 路径：不构建辅助头或教师，只对有效 free 变量做 FM。8 个 microbatch 共享有效批次的分母。
6. 记录曝光、loss、梯度、耗时和 CUDA 分配/保留峰值；按协议保存模型、AdamW、RNG、流位置。完成后导出不含优化器的生成器。

确认实验拥有独立输入命名空间和 checkpoint schema，不能与主实验的完整恢复状态混用。新公开训练入口复用主实验协议；确认实验目前只公开权重检查和采样兼容。

## 后处理位置

既有保持 C 的后处理仍在 `native_t1/postprocessing.py`，运行说明见 [后处理文档](../../native_t1/docs/postprocessing.md)。R2 的 `sample` 只导出 RAW，不自动执行后处理。本轮未做 R2 输出到该后处理的端到端运行验证。

C 的三角坐标硬保持，不等于接合合理、条件用途正确，也不等于保留源模型的顶点 ID 图。R2 推理只接受 C、总 N、噪声与时间；源 GT 不作为推理输入。

## 为什么暂时保留部分旧名字

冻结输入计划和 checkpoint 把若干源码 SHA256 纳入身份。改写 `recipe_factorial.py`、`schedule40.py` 等文件，即使仅重命名，也会使旧缓存或恢复检查失败。因此此轮把计算核心作为有来源清单的兼容层迁入，在外部新增 R2 入口；没有对旧文件做重命名式重构。

内部导入闭包包含历史 readout/teacher/context 模块，R2 执行路径不启用它们。它们的存在不表示退役实验重新成为维护目标。
