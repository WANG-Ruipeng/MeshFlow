# R2 Bench 粗布局可用性配对实验

已授权范围：同EXPOSURE global25500起点，PLAIN/LAYOUT各1000更新，固定终点；同30父、原N、CLEAN/X_WIDTH配对与组FM系数。候选111424参数粗布局支路，来源P属于额外信息诊断，不是C-only部署或理论上界。原6父12请求，每个新模式PLAIN1000/LAYOUT1000/LAYOUT_NULL各12份，新增36 RAW、36次单次官方D0；完整后端与位置截面不改。

工作目录 `/mnt/e/MeshFlow`；Python `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`。

## 阶段入口

```text
python -B -m r2.pilots.bench_layout_oracle_v1.control status
python -B -m r2.pilots.bench_layout_oracle_v1.control dispatch --arm PLAIN --estimate-mib 5000
python -B -m r2.pilots.bench_layout_oracle_v1.training train --arm PLAIN
python -B -m r2.pilots.bench_layout_oracle_v1.control dispatch --arm LAYOUT --estimate-mib 5000
python -B -m r2.pilots.bench_layout_oracle_v1.training train --arm LAYOUT
```

每次新增worker前实际重测资源；正式首臂有代表性更新后才决定第二臂。旧common资源snapshot不可替代新测量。训练checkpoint每100完整更新滚动保存，1000步固定终点同时导出。仅INTERRUPTED_SAFE且calls/checkpoint游标完全一致可显式`--resume`，禁止丢失更新静默回退。已完成臂再次进入只核验，不重训。

两臂全完成后按evaluation.py/post.py入口派发；所有36新RAW完成后才能开始D0。最终统计完整177字段，图审必须实际看图，不能用数值生成标签。来源P、NULL和普通续训单独报告；源款式不是唯一正确答案。

本轮不新增布局预测器，不改完整坐标系、不追加训练、不挑best、不切默认、不清理、commit或push。

## 已完成阶段与复核

PLAIN / LAYOUT 均已完成固定1000步，终点在 `training/<ARM>/checkpoint_step1000.pt`，推理导出为 `generator_step1000.pt`。不得再调用 dispatch 扩大预算。日常复核只需 `control status` 和下面的离线 verifier。

本轮实际后续入口如下（生产入口会核对已完成标记及源身份；不要删除标记强制重跑）：

```text
python -B -m r2.pilots.bench_layout_oracle_v1.evaluation sample --label PLAIN1000 --estimated-peak-mib 3500
python -B -m r2.pilots.bench_layout_oracle_v1.evaluation sample --label LAYOUT1000 --estimated-peak-mib 3500
python -B -m r2.pilots.bench_layout_oracle_v1.evaluation sample --label LAYOUT_NULL --estimated-peak-mib 3500
# 三组全部 RAW 完成后，才可进入 D0：
python -B -m r2.pilots.bench_layout_oracle_v1.post d0 --label PLAIN1000 --estimated-peak-mib 3500
python -B -m r2.pilots.bench_layout_oracle_v1.post d0 --label LAYOUT1000 --estimated-peak-mib 3500
python -B -m r2.pilots.bench_layout_oracle_v1.post d0 --label LAYOUT_NULL --estimated-peak-mib 3500
python -B -m r2.pilots.bench_layout_oracle_v1.post raw --workers 2
python -B -m r2.pilots.bench_layout_oracle_v1.post post --labels PLAIN1000 --workers 2
python -B -m r2.pilots.bench_layout_oracle_v1.post post --labels LAYOUT1000 LAYOUT_NULL --workers 2
python -B -m r2.pilots.bench_layout_oracle_v1.evaluation collect
python -B -m r2.pilots.bench_layout_oracle_v1.gallery --workers 2
python -B experiments/r2_bench_layout_oracle_v1/control/summarize.py
python -B experiments/r2_bench_layout_oracle_v1/control/final_verify.py
```

`final_verify.py` 只读取训练账本、权重哈希、固定输入、保存网格/评分、图页和实际图审记录，不调用模型、不重评分。图审记录不是脚本生成，恢复时不得自动补标签。

采样每组12份/600次向量场调用；D0每组12次。所有36份RAW均早于首个D0。EXPOSURE为冻结缓存参照，未重新采样或评分。训练/生成失败均没有静默重试；独立CPU训练汇总的一次切片笔误与保留partial见 `engineering_events`、`analysis/training_complete_v2`。
