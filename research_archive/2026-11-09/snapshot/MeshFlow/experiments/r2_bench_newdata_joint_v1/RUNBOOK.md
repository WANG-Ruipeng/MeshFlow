# 有限恢复与产物入口

本轮只授权一个1500更新候选和冻结的28请求双模型评价。运行环境为WSL Ubuntu，工作目录`/mnt/e/MeshFlow`，Python为`/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`。

## 只读查看

```bash
python -B -m r2.pilots.bench_newdata_joint_v1.run
python -B -m r2.pilots.bench_newdata_joint_v1.summarize_training --snapshot
```

`workflow/*/status.json`记录各阶段；`training/NEW_DATA_JOINT/status.json`记录已完成更新游标。不要因监督会话退出而启动第二个训练进程，也不要删除`launch.json`或`writer.lock`来绕开检查。

## 已冻结的阶段次序

1. `train`：唯一1500更新训练。`raw-baseline-v2`与训练在实测资源允许后并发，基准20份新RAW、8份旧DEV复用。
2. `raw-new`：读取完整1500终点，28份新RAW。两个模型全部56份RAW身份齐备后，才允许D0。
3. `d0-baseline`与`d0-new`：分别20/28次单次官方D0；另外8份旧基准POST复用。新增worker前测资源、逐个启动。
4. `score-raw`、`score-post`：固定原评分与相同后处理。已存在的完整评分收据只读复用。
5. `collect`、`gallery`：168评分行和84页双列图；图审是独立的真实查看记录，生成PNG不等于已图审。

每个尚未启动的阶段使用：

```bash
python -B -m r2.pilots.bench_newdata_joint_v1.run --launch STAGE
```

完成阶段再次调用仅返回状态。失败阶段不会自动重试。原`raw-baseline`曾在首个模型前向之前因CPU工具导入清空CUDA可见性而退出；保留其日志及零实际调用证据，实际采样使用独立`raw-baseline-v2`。不重新使用原入口，不把它的STARTED记录计作成功前向；详见`engineering/pre_forward_import_recovery.json`。

## 训练恢复限制

```bash
python -B -m r2.pilots.bench_newdata_joint_v1.resume
```

默认只读。只有已核对的完整更新安全游标允许`--execute`；失败于更新中途时保留记录，不能静默回滚重算。完成终点不因再次运行入口而延长。

## 完成后的CPU整理

```bash
python -B -m r2.pilots.bench_newdata_joint_v1.summarize_training --final
python -B -m r2.pilots.bench_newdata_joint_v1.head_analysis run
python -B -m r2.pilots.bench_newdata_joint_v1.summarize_results
python -B -m r2.pilots.bench_newdata_joint_v1.visual_review validate
python -B -m r2.pilots.bench_newdata_joint_v1.visual_review aggregate
```

`head_analysis`只读实际推理布局，来源标签仅作事后诊断，不纠正推理结果。`summarize_results`只读原CSV，保留I的PASS/LOW_SUPPORT及分母。图审必须先逐页看完，再填写对应CSV；不得自动从分数生成标签。

训练完整检查点、推理导出和独立小头位于`training/NEW_DATA_JOINT/`；产物的实际文件名、hash和完成状态以该目录终点sidecar及最终完成收据为准。原数据、公共代码、旧RAW/POST、原评分器只读；没有默认切换、额外训练、commit或push。
