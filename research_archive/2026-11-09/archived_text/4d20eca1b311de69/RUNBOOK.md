# R2_BENCH_FULL_PRIOR_MOTIF_GUIDANCE_V1

用户于2026-10-29明确授权执行已交付设计。预算：200份新RAW、184次单次官方D0、4860次输入VJP；模型参数更新0，新增free噪声0。主设计来源为 `maintenance/motif_guidance_design_20261029/PROTOCOL.md`，冻结副本与输入身份由prepare登记。

## 环境与入口

实际环境：WSL Ubuntu；工作目录 `/mnt/e/MeshFlow`；Python `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`。环境变量 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；每科学进程CPU线程1。由Windows调用WSL时需要允许访问实际WSL运行环境，不启动可见窗口。

总入口：`python -B -m r2.pilots.bench_full_prior_motif_guidance_v1.run`。

- `resources`：保存只读资源快照。
- `status`：列出现有产物和锁定状态。
- `prepare`：冻结现有输入、模型身份、计划任务、评分/可视化定义，并产生每请求唯一影子噪声；完整free噪声数组复用。
- `sample --jobs-json <显式任务清单> --worker <唯一worker标识>`：只运行清单内既定任务；首任务测量真实VJP资源。已有完整产物由生成器核验后跳过；未完成任务必须有可恢复的安全步状态。
- `evaluate -- <evaluate子命令参数>`：记录子进程资源、日志、返回码；阶段依赖与评分预算由evaluate检查。

全部阶段均有独立输出目录。`workflow/attempts/`保存进程、日志、5秒资源观测和完成收据；`workflow/ledger.jsonl`只追加。不停止其他用户进程，不从OOM/非有限值静默重试。

## 阶段依赖

1. 准备与CPU计算图/身份检查。
2. 40份校准RAW：G0/G1各8，G3三强度各8。首条正式测量是bcb/n0、G3 λ=0.1，包含在40份内。
3. 校准RAW评分及8请求的真实匿名五轴图审，固定一个全局λ，封存 `lock.json`。
4. 补齐G2校准8份以及四臂确认38×4，共160份RAW。选中λ校准8份不再生成。
5. 全200份RAW保存后，主表184份执行单次D0与原后端。其余16份仅RAW。
6. 完整评分、同位置截面和实际图审，父等权比较，分别报告校准/确认及训练/DEV/留出。
7. 报告、源码与hash归档后停止。不得自动增加训练、强度、噪声、Chair实验、后端，也不切默认、删除旧资产、commit或push。

若输入身份、资源或数值合同不成立，记录明确失败及已消耗计算。工程修复可以版本化；不能未经记录重放已经发生的科学计算。恢复应读取完整收据和安全步账本，禁止仅凭目录存在就当成完成。

## 执行收口与只读恢复（2026-10-29）

本轮已COMPLETE，方法结论为NO_UPGRADE。200 RAW、184 D0/CPU、706原177字段行、476实际图审均封存；不存在等待继续的科学任务。请先读REPORT.md和CLOSEOUT.json。

仅检查完成状态：

```bash
cd /mnt/e/MeshFlow
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_full_prior_motif_guidance_v1.run status
```

阅读结果不需要加载权重：analysis/final与analysis_numeric/final含全部汇总；gallery/final/index_all.html含同位置截面。archive/sources保留实际小源码，SOURCE_HASHES.json及PACKAGE_RECEIPT.json可独立校验。

不要重新执行prepare、sample、D0或CPU任务来“恢复浏览”；不要以删除完成收据的方式强迫重跑。历史科学输入与状态保持只读。若发现某个文件损坏，先按已登记hash定位并报告，不自动生成替代噪声/输出。恢复脚本的存在不授权额外科学计算。

工程错误和安全CPU交接见engineering_incidents与workflow/post_four_incremental；4条人工记录cell_id修正见visual_review/root_FINAL_metadata_repair_receipt.json。不得覆盖原图审或校准标签。
