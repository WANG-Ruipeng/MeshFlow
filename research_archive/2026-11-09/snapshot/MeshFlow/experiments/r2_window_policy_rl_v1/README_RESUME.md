# R2_WINDOW_POLICY_RL_V1：执行与恢复

本轮固定合同：先完成冻结 FULL_S6000 的 256 份标定及真实图审；门通过后执行 64 个 B=1/G=8 batch 机会，最多64次更新，不补抽全零/全一组；终点为 batch_063。终评固定 TRAIN32＋DEV12 请求，FULL_DET/FULL_STOCH/RL_STOCH 共132输出，统一单次官方D0、GEO_C及来源定向。

`resume.ps1` 是本地分阶段入口。工作目录为 `E:\MeshFlow`，使用已存在 WSL Ubuntu 与 `/mnt/e/MeshFlow-Control-artifacts/venv/bin/python`。不需要重新下载权重。源码在 `r2/pilots/window_policy_rl_v1/`。

按依赖顺序：

```powershell
.\experiments\r2_window_policy_rl_v1\resume.ps1 -Phase prepare
.\experiments\r2_window_policy_rl_v1\resume.ps1 -Phase calibration-sample
.\experiments\r2_window_policy_rl_v1\resume.ps1 -Phase calibration-d0
.\experiments\r2_window_policy_rl_v1\resume.ps1 -Phase calibration-cpu
.\experiments\r2_window_policy_rl_v1\resume.ps1 -Phase calibration-gallery
# 真实查看原图并提交 reviews_actual_v2.json；不得自动填写语义标签
.\experiments\r2_window_policy_rl_v1\resume.ps1 -Phase calibration-analyze
.\experiments\r2_window_policy_rl_v1\resume.ps1 -Phase train
.\experiments\r2_window_policy_rl_v1\resume.ps1 -Phase final-sample
.\experiments\r2_window_policy_rl_v1\resume.ps1 -Phase final-d0
.\experiments\r2_window_policy_rl_v1\resume.ps1 -Phase final-cpu
.\experiments\r2_window_policy_rl_v1\resume.ps1 -Phase final-gallery
# 真实查看44页匿名原图并提交132行 reviews_actual.json
.\experiments\r2_window_policy_rl_v1\resume.ps1 -Phase final-analyze
```

D0阶段产生连续提议；必须等CPU阶段完成GEO_C、POST receipt与原评分后再出图。已有完整阶段按身份核对后复用；不得把一次失败阶段的半成品当作完成，也不得删除账本后静默重跑。模型调用阶段遇失败应保留现场按合同停止，不自动重抽、改变精度或补训练。

已冻结的参考终评88份RAW在训练后半段独立完成；正式final-sample只新增44份RL。它们使用同一预登记job，完成全部132份RAW后才运行终评D0；没有用终评反馈选权重。

训练权重为 `training/batch_063/update/checkpoint.pt`，包含本轮LoRA和AdamW状态，必须配套原FULL_S6000基座及窗口开关使用，不是独立完整生成器。基座：`../r2_full_seat_learning_curve_v1/inference/FULL_S6000_generator.pt`。D0：`../r2_postprocess_avmr_pmr_v1/assets/chair_denoiser_0030000.pt`。最终checkpoint选择与hash见 `training/final.json`；模型从未自动切为默认。

保留全部RAW/POST、状态buffer、原score、各批适配器、真实图审和资源日志。小审阅包只省去大数组与权重，不省逐例CSV和实际评分源码；原资产继续留本地。包不等同于独立训练环境分发。

核验与打包工具均为CPU：`training_summary.py`、`final_analysis.py`、`scientific_plots.py`；`package_review.py`默认只检查，显式`--package`才写ZIP。只读取现存证据，不追加模型调用或评分。完成本轮后停止，无默认切换、无commit/push。
