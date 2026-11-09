# 恢复与复核入口

本轮是固定预算的零训练实验。代码在 `r2/pilots/bench_loop_motif_mini_v1/`，数据在本目录。不要以恢复为由增加请求、噪声、循环层段或引导强度。

在仓库根目录使用 WSL Python：

```powershell
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/r2/pilots/bench_loop_motif_mini_v1/run.py status
```

生成、D0、CPU 各阶段都有逐请求账本及完成标记。`run.py launch --phase <generate|d0|raw|post> --indices <固定任务索引> --worker <新执行标识> --peak <实测规划MiB>` 只能补尚未开始的授权任务或验证已完成项；已开始但未封存的模型调用会拒绝静默重算。恢复前检查 `tasks.json`、逐任务 `calls.jsonl`、`receipt.json` 和 worker 日志。当前完整产物存在时不应再启动模型。资源检查和 all16 RAW barrier 继续生效。旧原始文件只读。

`analyze.py render --indices ... --stage RAW|CLEAN_FINAL_POST` 是CPU绘图，可验证已完成图像。`run.py collect`/`analyze.py numeric`、`diagnostics.py` 是一次性派生归档命令，采用独立文件与拒绝覆盖的完成标记；重复复算请指定独立新目录并保留原归档，不能直接抹掉旧账本。CSV 可直接独立重算均值及配对差；源评分字段和查询未改。

实际预算：800 个采样向量场、360 个采样输入VJP；独立 r1 检查另为2个向量场与2个VJP；16次单次D0；48行新增原177字段评分。重复层每向量场16次层调用，CFG双分支以同一批次执行，不把800次逻辑调用说成原模型的等算力。无训练、无新增噪声。

所有51状态保留在 `raw/*/states`，完整连续proposal和索引网格保留在 `post`，各CPU处理链和原始评分保留在 `cpu`。小包不含官方大权重，也不含完整中间轨迹；固定请求、完整CSV、实际代码、hash及最终网格均在包内/本地保留。

独立分析实现修正：首次CPU绘图在处理历史标签 `G3_L0p3` 时停止，尚未输出图像；随后仅将显示/汇总标签规范为 G3，并保留原 cell_id、路径和lambda，不修改模型、评分或请求。标题同样在独立分析适配器里更正为 R/G0/G1/G3/L0/L3。没有因此重跑任何模型调用。

图审是实际查看32个新 RAW/FINAL 单元的 AI_PROXY，不是匿名人类研究。原64个参照评级原样复用；6b/n0采用不确定单独保留，未为了制造成功计入联合正例。
