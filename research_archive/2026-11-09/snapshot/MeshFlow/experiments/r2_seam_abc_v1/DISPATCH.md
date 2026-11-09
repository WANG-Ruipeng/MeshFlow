# 实际调度记录

- 初始RTX5080总16303MiB，空闲约12774MiB；24逻辑CPU，available RAM约65.8GB。Windows CIM在沙箱下不可读，WSL实测补齐。WDDM不提供可靠单进程显存，保持NA。
- 首个A正式F/B/AdamW到32步，PyTorch reserved约3124MiB（不是进程显存）；整卡从约3198MiB增至约6702MiB，观测增量约3504MiB。按3600MiB保守增量和1.2余量估算后增开LONG的12份登记DEV采样；A与LONG并行结束后LONG释放。
- B缓存/合同冻结后实测free9680MiB、available RAM约60.5GB，增开B第二训练worker。
- A175步/B8步时整卡used9817MiB、free6163MiB、GPU51%。以headroom2048MiB、1.2×3600MiB计算，第三个同类worker槽数floor((6163−2048)/4320)=0；C等待。没有因低显存占用推断可无限并发，也没有改变batch/精度来塞入第三臂。
- 每次后续派发先另存resources/*.json；具体训练/推理峰值和阶段墙钟由各runner账本记录。CUDA并发墙钟不作孤立模型性能对比。未停止无关用户进程。

源文件保护、预算和科学定义以PROTOCOL与各分臂冻结JSON为准，调度不增加训练/采样数量。

- A完成800步并退出后，`before_C_training.json`记录free9689MiB、RAM约55.4GB；C作为第二训练worker启动。其前4个真实更新包含pointer和生成器反向/优化器分配，reserved约3090MiB、整卡used9762MiB；无额外探针。
- `before_A_generation.json`记录free6184MiB、GPU51%、RAM约53.4GB。采样独立估计2600MiB，floor((6184−2048)/(1.2×2600))=1，增加A固定28请求的采样worker；此时为2训练+1采样，不能将它误记为3训练并发。
- PowerShell受限环境的Get-PSDrive未给出可用容量（空值算成0），改用WSL df核实E盘仍有1.8TiB空闲；不是磁盘耗尽。只读AST命令一次被跨shell引号解析拒绝，改为标准输入后3个新增汇总/状态脚本语法通过；两次均零模型调用。
- B完成后free9743MiB/RAM62.2GB，B的28次采样与C训练重叠；A采样此前已结束。C最终采样前free13233MiB/RAM65.8GB，训练全部退出。三臂各800步均成功，无OOM、失败或额外模型重放。
- 全部112份逻辑RAW（96新、16缓存）齐全后，LONG12份新D0先执行以实测。驻留整卡约4925MiB，相比基线2756MiB增量约2169MiB，继续使用3000MiB保守派发估计。A/B/C依次在前臂达到代表峰值后复测启动；装载/核验与前臂短处理存在进程重叠，不据此声称多D0并发加速。每臂实际墙钟与调用均见post_complete.json；总96次新D0，不包含16份旧POST缓存。
- 所有模型进程结束后，before_CPU_score_render记录free13220MiB、RAM约66.0GB、24CPU。评分与渲染独立运行，各先执行一份登记任务并实测RSS，再最多2个CPU worker、每worker1线程；训练汇总/图审不增加模型调用。

- 最终只读CSV查询曾因UTF-8 BOM令首列`baseline`被误读而中止；改用utf-8-sig后完成查询。原CSV和评分未改，零模型调用。实际56页、224条图审齐全后以独立聚合器核对原单元与图片hash，不从数值补写视觉标签。

- 独立最终审计在WSL挂载盘反复调用Path.resolve逐级stat，出现大量I/O等待。曾讨论只读审计的路径缓存优化，但原进程已自行正常完成，因此没有实际中断、重启或修改审计源码。一次审计442.524秒，核验19158文件hash并PASS；零训练/模型/评分重放。
