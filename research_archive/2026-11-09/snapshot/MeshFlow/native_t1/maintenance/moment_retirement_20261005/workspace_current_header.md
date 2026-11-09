# 当前工作区：Chair HYBRID START

维护入口为 `native_t1/`。2026-10-05 按用户明确要求退回并正式接入 START：原 OT_HYBRID pilot1000 / conditional_total5000，Native T1 + C-only Geo + FM，训练 coupling lambda=0.25。默认采样 profile 现为 `chair-hybrid`；MOMENT V2 及其 finite-surface 前序活动包已封存并移除。

## 当前使用方式与边界

```bash
python -B -m native_t1 sample --condition /path/to/C.npy --num-faces 163 --seed 2452526625 --out native_t1/runs/chair_start_sample
```

总 N 必须显式指定 128..256；C 为原坐标尺度的有限 FP32 数据，0<K<N。使用原 MT19937 free-face 噪声、50 步 Euler、BF16 forward / FP32 参数与积分，已知 C 在 51 个状态均精确保留。详见 [START 说明](native_t1/docs/chair_start.md)。

正式本地权重为 `native_t1/checkpoints/chair_hybrid_start_total5000.pt`，是原 `chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt` 的硬链接。文件 SHA256：`d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80`。原完整 AdamW、RNG 和输入流状态保留；推理不恢复优化器。Git 只纳入正式源码、测试、文档及身份 manifest，权重继续本地保存。

旧 `--profile fm-geo`、`edge5` 等保持显式可用的 N112 路径。`train` / `train-edge5` 仍是显式预算、fresh AdamW 的旧 N112 训练入口，不是 START 续训入口。任何新训练需要独立的数据、coupling 和预算合同，不因改变默认模型自动开始。

## 保留的正式和历史权重

|角色|路径|累计步数|
|---|---|---:|
|START 默认|native_t1/checkpoints/chair_hybrid_start_total5000.pt|5000|
|旧 FM 显式 profile|native_t1/checkpoints/ablations/fm_cumulative3000.pt|3000|
|JEdge5 显式可选|native_t1/checkpoints/jedge5_cumulative3000.pt|3000|
|旧共同起点 G0|native_t1/checkpoints/tgeo_g0_cumulative2500.pt|2500|
|surface 旧参照|native_t1/checkpoints/ablations/surface_cumulative3000.pt|3000|
|Edge-only 旧参照|native_t1/checkpoints/ablations/edge_only_cumulative3000.pt|3000|

## 2026-10-05 MOMENT 封存完成

删除 MOMENT/H_CONT 正式 V2 权重各 5 份、8000 个派生 OT NPZ 和三个退役活动源码包，共 8061 文件、15,677,075,950 bytes（约 15.68 GB 逻辑字节）。源码先行归档；原始结果、报告、逐例 CSV、日志、评分源码快照、决策 ZIP 和小型合成 CPU 测试 fixture 保留。

11,488 个保留文件前后 SHA256 一致。START CPU strict load 通过，公共 CPU 测试 109/109 通过，删除后 CLI 导入通过；真实模型前向、训练更新和生成均 0。根据本轮用户明确授权完成本地 Git 提交，没有 push。详见 [清理报告](native_t1/maintenance/moment_retirement_20261005/CLEANUP_REPORT.md) 和其中的账本。

后续仍遵守 [AGENTS.md](AGENTS.md) 的资源调度与保护边界。以下是按日期保留的历史记录；其中旧的默认模型、HEAD、实验状态及已删除路径以其记录日期为准。

