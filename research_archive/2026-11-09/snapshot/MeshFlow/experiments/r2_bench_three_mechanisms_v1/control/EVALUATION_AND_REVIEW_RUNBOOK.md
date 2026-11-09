# 固定评测与实际图审入口

全部命令在 `/mnt/e/MeshFlow`，使用已有 `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B`。
派发前继续读取实际GPU资源；此文档不额外授权样本或重试。

1. 全部训练终点完成；先执行 `experiments/r2_bench_three_mechanisms_v1/control/verify_rl.py --write`。
2. 各臂调用 `-m r2.pilots.bench_three_mechanisms_v1.evaluation sample --label LABEL --estimated-peak-mib MEASURED`。
   LABEL固定：PLAIN200、PLAIN267、PATCH200、LOOP200、RL_ZERO、RL16。
   同一label只有一个writer；不同label可独立进程，逐个加载并实测峰值后再确定是否并行。
   训练barrier与完整72 RAW barrier不得绕过。
3. `-m r2.pilots.bench_three_mechanisms_v1.post verify`。
4. 各臂 `-m r2.pilots.bench_three_mechanisms_v1.post d0 --label LABEL --estimated-peak-mib MEASURED`。
   PATCH必须通过此入口，不能直接调用旧engine().d0，否则缺少共享恢复。
   各label独立；同一label不能两个D0 writer。均为单次官方D0，禁止重复采样或再去噪。
5. 72 D0全部完成后，干净CPU进程：
   `-m r2.pilots.bench_three_mechanisms_v1.post raw --workers 2`
   然后 `-m r2.pilots.bench_three_mechanisms_v1.post post --workers 2`。
   每进程线程1；CPU dispatcher先实际一请求再根据RAM估计最多2worker。
   RAW评分和POST评分彼此目录独立，但不要同时调用覆盖相同label/stage的两个controller。
   本轮稳妥默认每次一个CPU controller，避免两个controller各自扩worker造成超额负载。
6. `-m r2.pilots.bench_three_mechanisms_v1.evaluation collect`。
   所有评测采用独立 `score_registration_v2.json`；原v1保持原地只读。
7. `-m r2.pilots.bench_three_mechanisms_v1.gallery --workers 2`。
   36页、每页7模型行×4panel，原双视角及原front/side截面，不生成标签。
8. `experiments/r2_bench_three_mechanisms_v1/control/finalize.py crops` 可生成无缩放原图裁切：
   每页行1–3、4–5、6–7，共108图，便于实际原分辨率查看。原图不变，裁切有source/hash/box。

实际图审三份：

| assignment | 固定请求索引 | 父 | 页数 / cells |
|---|---|---|---:|
| review_part_01 | 0:4 | 459、f495（DEV） | 12 / 84 |
| review_part_02 | 4:8 | a8、27（FOCUS） | 12 / 84 |
| review_part_03 | 8:12 | df6（HOLDOUT）、bcb（NEW_WIDE_TRAIN） | 12 / 84 |

同一请求全部模型、三阶段由同一实际评阅者看，允许不确定。不读评分或旧标签造图审。
每个模型行需查看双视角+两截面；七行整图不足辨认时必须打开无缩放裁切，记录实际view事件。
`visual_review/templates/`只是空模板，写到`completed/`时才能按真实观察完成。

`control/finalize.py snapshot` 输出新时间戳训练/资源/指标快照与PNG，不覆盖旧快照。
缺少评分或图审时只标PARTIAL/PENDING；`final`必须所有完成才认可收口。
REPORT.md已按真实数值与图审收口；原草稿保留在report_snapshots/initial_pending_REPORT.md。图审v1保留，f495/n1 LOOP两阶段支撑的实际复看不确定性另列amendment与summary_v2，不覆盖原计数。完整收口入口见上级RUNBOOK.md。
