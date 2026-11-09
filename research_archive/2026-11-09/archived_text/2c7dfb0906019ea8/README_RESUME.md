# R2 EXP67 固定噪声宽度响应诊断

本轮只执行附件建议的候选一。冻结EXP67 step6000，不训练、不反向、不修改模型、loss、采样定义或默认模型。完成固定面板后停止，不自动commit/push。

- 8个DEV父，每父1条C：4 SEAT_HALF、2 SEAT_FULL、1 BACK_HALF、1 BACK_FULL。
- 原两份free Gaussian；只对C的X坐标围绕模型原点乘FP32 0.95、1、1.05。共48份新RAW。
- 原Euler50，2400次生成器前向；原C采样轨迹索引5/15/30/45，各用C±固定free再评估，共128次前向。
- 合计2528生成器前向，48单次官方D0。没有额外探针或FP32对照、没有新噪声。
- 全部RAW完成后才进入D0→恢复canonical C→GEO_C→恢复来源C角序→固定C整理free方向。
- GPU microbatch1、一个worker为附件明确合同；CPU评分可依实测使用两个worker并与独立渲染重叠。
- 原数据、权重、旧报告和评分源码只读；全部新产物在本实验目录。
- 各新C使用独立源映射、查询身份。原查询点固定仿射，不重抽点；面积变化使用原权重乘新旧面积比再归一。
- 固定四时刻的BF16实际输入嵌入、速度和50步轨迹均保存。只预注册精确零与连续响应，不事后挑“弱响应”阈值，不把整体仿射斜率1作为开放生成标准。
- 宽度响应不能认证高度采用。数值变化、合理采用、邻近匹配、真实共边分别报告。
- 图审须实际查看32页、96个stage/variant输出，标非盲AI_PROXY，保留UNCERTAIN；渲染不是已审。

恢复总入口（Windows PowerShell）：

wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.condition_width_response_v1.run --stage all

支持prepare / generate / postprocess / score / render / analyze / aggregate分段。已完成单元只读验证；已reserve而未return的模型调用不自动重放，先检查失败日志与残留产物。冻结科学代码与输入hash变化会阻止继续。工程修复须独立保留变更与失败身份，不能更换面板、噪声或预算。

主要证据：
- registration.json、selection_candidates.csv、selected_conditions.csv、requests.csv、input_transform_receipts.json
- evaluation/raw、evaluation/post；每个单元有receipt和调用账本
- probes：64个固定状态配对，128份真实模型输出与实际入口known hidden位模式
- analysis：逐输出/逐曲线/逐轨迹/逐固定状态/逐free面速度响应
- evaluation/scores：96份原质量、8档AVMR/PMR、实际合法图与距离
- evaluation/gallery：32页固定视角配对图、逐输出真实图审
- audit：基准复现、原资产hash、独立执行核账
- REPORT.md：完成后的结论、边界、下一步假说

本地日期跨2026-10-12/13；实际执行时间以UTC账本为准。此README不代替真实完成状态和报告。
