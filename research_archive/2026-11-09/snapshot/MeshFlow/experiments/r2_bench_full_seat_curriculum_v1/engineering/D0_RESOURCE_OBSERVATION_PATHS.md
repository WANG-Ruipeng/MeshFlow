# 本轮 D0 实测资源观察位置

本文件只说明已实现的记录位置，不启动 D0、不增加采样或探针。三个已注册标签
各 24 次 D0；必须先通过全部 72 份 RAW 的 hash/receipt 屏障。

令 `L` 是 `COMMON_S12000`、`NATIVE_S6000`、`SEAM_S3000` 之一，
`R` 是冻结 `eval_manifest.json` 的 request_id。以下路径均相对本实验根目录。

| 记录 | 路径 | 可读时点与含义 |
|---|---|---|
| D0 冻结注册 | `evaluation/inference_registrations/L__D0.json` | 实际 worker 通过 72 RAW 屏障后；绑定官方权重、源码、全部 RAW receipt |
| 本次 attempt | `evaluation/L/postprocess_attempt_<time_ns>/` | 一个 label 的独立 writer；原始 attempt 不覆盖 |
| 启动前资源 | `.../resources_before.json` | 含 GPU used/free/util、进程、CPU/RAM，由已有 guard 检查 |
| 加载后资源 | `.../resources_loaded.json` | 权重实际加载后；只是快照，不是峰值 |
| 第 1/2 个产物后资源 | `.../resources_output1.json`、`resources_output2.json` | 已完成该次真实 D0 与 GEO；同样是快照 |
| 单请求 D0 allocator 峰 | `evaluation/L/R/post/proposal_receipt.json` | 该次真实前向完成立即写出；`peak_allocated_bytes` / `peak_reserved_bytes`，每次 forward 前 reset；含驻留模型，但不含完整进程上下文 |
| 单请求调用账本 | `evaluation/L/R/post/calls.jsonl` | `D0_FORWARD_STARTED` / `D0_FORWARD_RETURNED`；每请求严格 1 次 |
| D0+GEO 产物 | `evaluation/L/R/post/receipt.json` | `D0_forwards=1`、`GEO_calls=1` 与所有文件 hash；应先确认前两份真实产物完成且无 failure |
| 1 秒设备观察 | `evaluation/L/D0_resource_observer_<time_ns>/resource_observation.json` | **仅 worker 的 finally 阶段落盘**；运行中不能假定文件已包含最新样本。结束后提供整阶段设备峰、进程归属可用性与全部采样 |
| 标签完成 | `evaluation/L/post_complete_full.json` | 24 份实际 POST receipt 全部完成；`new_D0_forwards` / `new_GEO_calls` 应各为 24 |

首个 worker 运行中，如需决定是否加第二个已注册 label，root 应结合前两份
`proposal_receipt.json` 的 allocator 峰、实际 `resources_output1/2.json`，以及
当时实际 Windows/WSL `nvidia-smi` 的独立连续观察。不要等待的同时假造
observer 已落盘，也不要把 allocator 或加载后快照叫作完整进程 VRAM 峰。
运行期额外资源观察只是设备查询，不允许额外模型前向。

可比旧记录在 `engineering/d0_comparable_memory.json`：相同官方 Bench D0、
FP32 参数/BF16 前向、N128..256，旧实际 D0 allocator 约 1795.70 MiB allocated、
1824 MiB reserved；旧整设备峰 3546 MiB。旧单进程显存为 NVIDIA_SMI 不可用，
3546 MiB 不能当作排除其他进程后的 worker 精确增量。本轮仍以真实新 worker
实测为准；controller 的 6000 MiB 保守 D0 增量不能仅凭 allocator 值下调。

每次新增 worker 前重新读取设备、进程、CPU/RAM与评分进程负载。至少保留
max(2 GiB, 设备总显存 10%) 的额外余量；有可归属可比进程峰时用
`floor(max(0, free - headroom) / (1.2 * observed_worker_peak))` 估计新增槽数。
若进程归属不可用，使用保守预算并说明限制，不把整设备峰自动转换成增量。
低显存不代表算力空闲；新增必须有实际吞吐收益，并遵守 CPU 线程限制。

只允许不同 label 的独立模型实例与输出目录并发。手动 dispatch 与 controller
必须共享进度记录，防止同一 label 出现第二个 writer；正在运行的同 label
不得因尚无 completion 就再次启动。每份已有 proposal/POST 保留原调用账本，
异常或 OOM 不静默重试。本文没有授权修改预算、阈值、精度或采样数。
