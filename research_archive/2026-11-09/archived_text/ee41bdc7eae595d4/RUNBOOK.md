# 固定交叉候选恢复入口

当前仅独立准备。先行匿名复核门通过、CPU输入与代码门通过后，才允许新GPU调用。

新候选`SHIFT_RELATION_T050`，完整48请求、两个冻结24请求shard，每份25个RELATION后缀前向。`python -B -m r2.pilots.bench_shift_relation_crossover_v1.sample --check-inputs`仅CPU核验；正式采样指定`--label SHIFT_RELATION_T050 --shard 0`或`1`。不得重复启动未完成shard或覆盖失败记录。

两个shard完成并核验后，`post_d0 --label SHIFT_RELATION_T050`运行48次单次官方D0。`evaluation raw/post`执行原CPU评分；`collect`与`gallery run`仅聚合和渲染。固定执行链由独立workflow记账，最终仍须实际图审、报告、完成收据。新权重数为0，模型保持冻结。

所有模块前缀`r2.pilots.bench_shift_relation_crossover_v1`。原JOINT_SHIFT/RELATION目录不得回写。完成后任何resume仅只读核验，不自动重跑。
