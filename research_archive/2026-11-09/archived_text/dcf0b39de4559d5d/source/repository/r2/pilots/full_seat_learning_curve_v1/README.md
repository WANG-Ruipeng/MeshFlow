# R2 FULL seat learning curve V1

本地独立实验。科学定义冻结在 `experiments/r2_full_seat_learning_curve_v1/registration.json`，实际执行账本及恢复点保存在同一实验目录。维护的默认模型和原评分器不变。

## 固定合同

- 从 EXP67 global_step6000 的 live 生成器、匹配 AdamW 与 RNG 继续；源文件没有条件 EMA，不拼接状态。
- TRAIN67 原50条SEAT_FULL中，47条满足K≥2、完整且连贯输入；三条多连通分量按输入合同排除，不按生成结果筛选。保留来源free中的既有风险。
- 新增6000步，batch8/micro1，原free-only FM、free OT后HYBRID0.25；四clean加四width增强、原优化器设置和确定性BF16前向/FP32参数保持。
- 只在正常训练前向中记录真实来源接口角点/其余角点的误差，不增加loss或模型调用。变化输入上的日志不是固定probe，(1-t)^2换算不等于rollout。
- 固定local0/1000/3000/6000（global6000/7000/9000/12000）。每点同六父×四噪声，共24请求；预定主结果是6000步，不挑best。
- Euler50、单次官方D0、GEO_C、来源C角序恢复和free定向。SMALL_C200与CREF_C200只复用冻结参照，不参与训练。
- 六父面板已在开发中观察过，TRAIN/DEV分开，不作独立大测试集或唯一GT重建排名。

## 恢复入口

PowerShell在仓库根目录运行：

```powershell
.\r2\pilots\full_seat_learning_curve_v1\resume.ps1
.\r2\pilots\full_seat_learning_curve_v1\resume.ps1 -Stage all
```

默认仅查状态；all核验已完成阶段并执行剩余固定预算。若已有合法训练进程，等待它安全释放GPU。一个GPU阶段顺序执行，CPU评分最多两条独立管线，每条最多两个worker，按资源实际派发。失败保留现场并停止新增派发，不能直接重放缺少完整回执的模型调用。

完成数值流程不等于完成研究报告：还需实际看全组图、保存192行主模型AI_PROXY、编写REPORT.md，才能运行 `artifact_archive --build`。总入口不会伪造图审或自动写成功结论。

## 主要源码

`data.py`：只读条件资格、来源合法接口mask与继承证据。
`prepare.py`：冻结身份、数据、预算和源码哈希。
`train.py`：原FM续训、真实调用账本、固定点保存、精确恢复。
`evaluate.py`：严格缓存和统一RAW/POST/原指标/有限边界诊断。
`analysis.py`、`comparison.py`：有分母的训练误差与全请求配对表。
`render.py`、`sections.py`：深度正确的图片与固定有限剖面，不自动给采用标签。
`orchestrate.py`、`run.py`：可恢复的阶段依赖和资源记录。
`artifact_archive.py`：报告与实际图审完成后的独立小审阅包；权重/RAW留本地。

本轮训练前的66个旧权重删除及23个保留权重核对，单独记录在 `cleanup/`。清理释放磁盘空间，不是RAM或VRAM。禁止回写旧RAW/报告，禁止自动切默认、commit或push。