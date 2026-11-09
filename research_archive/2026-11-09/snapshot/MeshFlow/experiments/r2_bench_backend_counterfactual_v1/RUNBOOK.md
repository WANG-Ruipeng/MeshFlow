# 可恢复纯CPU入口

在现有 WSL 环境、仓库根目录运行：

```powershell
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_backend_counterfactual_v1.run
```

完成封存后，入口仅核验保护输入、代码和结果hash，不重复GEO、评分或渲染。未完成时按固定8+2任务及独立收据恢复；遇不完整失败目录停止，不静默重做。实际图审和报告不能通过自动入口冒充完成。

本轮已经执行的命令包括core run（10次GEO）、support准备及analyse、evaluate --manifest ... --workers 2、render --manifest ...（固定1worker）、summarize。首次core注册的旧包导入错误发生于0GEO/0模型；注册前修正源码搜索路径。正式执行无失败/重算。score/render并行执行的耗时不能当作独占硬件基准。

完成后`run --seal`用于单次封存；之后仍仅核验。原生成器、D0、原RAW、报告及评分器只读，零CUDA/模型调用。没有自动切换默认、commit或push。
