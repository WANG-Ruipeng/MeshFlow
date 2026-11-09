# 恢复与查看

执行环境：WSL Ubuntu，仓库 `/mnt/e/MeshFlow`，Python `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`；`PYTHONPATH=r2/src:native_t1/src`。

只读状态：

```sh
python -B -m r2.pilots.bench_full_seat_curriculum_v1.run
```

登记冻结后，首次启动对应阶段：

```sh
python -B -m r2.pilots.bench_full_seat_curriculum_v1.run --launch native
python -B -m r2.pilots.bench_full_seat_curriculum_v1.run --launch seam
```

第二条仅在 Native 正式 COMPLETE 后执行；SEAM 会自行核验完整 Native 终点、优化器、RNG和数据身份。重复 launch 只返回已有状态，不重复训练。

若训练状态为 `INTERRUPTED_SAFE`，先实测资源并重新核对账本，才允许使用 `train --stage NATIVE --resume` 或 `train --stage SEAM --resume` 精确续跑。`FAILED`、`OOM`、未完成调用、日志超过检查点或缺失更新不得自动回滚重算。恢复记录与原始失败记录分开保留。

固定预算 Native6000、SEAM3000，终点分别累计18000、21000；不得按曲线追加或挑best。最终完整完成标记以根 `COMPLETE.json` 为准；仅训练 COMPLETE 不代表评分与实际图审完成。

评测只消费 registration 中三个固定导出，不重抽manifest噪声。全部72份RAW齐全才运行单次D0；保存RAW/BASE/FINAL，所有图包含原位截面。旧产物只读，不自动清理、切默认、commit/push。
