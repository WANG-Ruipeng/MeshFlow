# 夜间运行状态只读核验 / Read-only operational review

**检查时状态正常，后续完成尚未得到验证。** 观测时间：2026-09-25 04:26:03 北京时间。本页描述该时刻的快照，不是当前瞬时进度。

- 官方源码：实际训练所在独立 worktree 的 11 个受保护文件，均与固定提交 bd7d8a138465c583694f9f86083ddd2979d10f87 的 git blob 及既有 SOURCE_MANIFEST 一致。只使用只读 ls-tree / rev-parse，未改 git。该结论不泛指原 E:\MeshFlow checkout 的全部文件。
- 冻结执行链：PLAN、队列源码快照、formal source manifest、绑定协议和门禁文件的 SHA 均一致。顺序为 balanced240k 完成核验 → seed10 50k → seed11 50k → 固定 final EMA 采样和评测。训练循环没有 loss 或质量早停；运行异常可以停止并保留失败，不会自动重试。
- 后台进程：Windows WSL launcher 存在；WSL controller PID3683 → run_all PID5679 → seed10 trainer PID5760，只有一条官方训练链。独立实时页进程 PID7925 存在。读取时队列最后心跳距当时 14.65 秒。
- 实际进度：seed10 最新完成记录为 2,700 / 50,000；seed11 尚未开始，为 0 / 50,000。固定训练预算尚余 97,300 次有效更新。进度每 100 步写入，不能当作瞬时计数。审查期间先看到 2,600，随后看到 2,700，与进程活动一致。
- 状态字段：formal 根 STATUS.json 的 step=0 是本次 invocation 起点；当前进度应读 training/seed_10/PROGRESS.json。实时页和队列心跳使用后者，未误把起点当作实时步数。
- 检查点：当时尚未达到首次固定 step5000 保存点，现有完整恢复检查点为 step0，实际大小 1,042,158,237 字节。已核对保存完成记录和文件大小，没有重复读取巨大检查点内容或加载模型。固定保存周期为每 5,000 步，最终 50,000 也保存。
- 数值与日志：保存的 2,700 步 loss 和 gradient norm 均为有限数值。扫描当前 queue、train、controller 日志，未发现 Traceback、NaN/Inf、OOM 或 non-finite；当前运行没有 FAILED / QUEUE_FAILED 标记。历史 SCHEMA_PREFLIGHT_INITIAL_FAILURE 等失败记录仍保留，不误认为当前训练失败。
- GPU 只读查询：NVIDIA GeForce RTX 5080，驱动 591.86，总显存 16,303 MiB、已用 6,973 MiB，瞬时利用率 47%，温度 47°C。只读进程查询记录 trainer PID5760。这不是性能测量，也没有执行新 GPU probe。
- 磁盘：E 盘可用 2,191,008,120,832 字节；WSL 根可用 982,670,651,392 字节。两种子含 step0 共规划 22 份检查点。按四组 float32 模型等价数组加每份 1 GiB 的状态和 trace 余量估算，约需 69.5 GB，当前空间足够。这是规划估算，不是未来文件大小的硬上限证明。

两个种子必须都完成固定 50,000 步，后处理才接受固定 final EMA，每 seed 64 个固定样本，Raw 与 Public Post 分别记账。原 AE/FM8000 为主五行表，64000 只作为单列附加表；Public Post 不代表论文完整后处理。观测时采样和评测尚未启动。后续仍可能遇到算法失败或运行异常，不保证完成时间、科学收益或收敛。

根目录实时页：E:\MeshFlow\output\LIVE_TRAINING_20260925.md。观测时距其最后更新 23.70 秒。

English readback: The fixed official queue was alive and advancing at 04:26:03 Beijing time. All 11 protected upstream files and registered source bindings matched. Seed10 had 2,700/50,000 recorded effective updates, seed11 had not started, leaving 97,300 planned updates. Inspected logs and saved numeric traces contained no runtime error or non-finite values. Disk space exceeded the checkpoint planning reserve. The frozen driver runs both budgets before fixed-final EMA sampling and evaluation, with no performance-based early stop or automatic retry. This is an operational snapshot, not a promise of future completion, convergence, or scientific success.

## 编码修复记录

此前通过 PowerShell 向 WSL stdin 传入中文，导致本说明中的中文成为字面问号；英文正常不足以证明中文读回成功。乱码稿完整保存在同目录 REVIEW_zh_en.encoding_failed.md，原稿 SHA256 为 55035fe6736a63bb302848fad12f68bb93cfd2efd93537e1fe2db4f64f6bae7e。

本次仅修正这份未封存说明文档：直接向 Windows 路径写入无 BOM 的 UTF-8，并使用 Get-Content -Encoding UTF8 实际读回中文。原运行数据、OPERATIONAL_READBACK.json、WINDOWS_READBACK.json 和 readback.py 均未修改；未运行训练、推理、GPU 计算或新的采样。修正后的文件哈希在修复工具输出及交接消息中记录。

详细证据仍为同目录的 OPERATIONAL_READBACK.json、WINDOWS_READBACK.json 和 readback.py。OPERATIONAL_READBACK.json SHA256 为 029b0cbe87e4c40eb289b031034b6b8ef98538d2694b449841bd882111137c68。