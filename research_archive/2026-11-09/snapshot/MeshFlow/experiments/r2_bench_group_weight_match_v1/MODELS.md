# 模型身份

M / W_MATCHED 固定1500步终点，global22500 / native10500，live权重（非EMA）。

- 推理导出：`/mnt/e/MeshFlow/experiments/r2_bench_group_weight_match_v1/training/W_MATCHED/generator_step1500.pt`
- SHA256：`fbbbc6b8996ca7d7eafd725a4547eb42fdeb061f4ce422defbb1b7ab7c155d32`
- 完整恢复状态：`/mnt/e/MeshFlow/experiments/r2_bench_group_weight_match_v1/training/W_MATCHED/checkpoint_step1500.pt`
- SHA256：`0fc20a5414d7e247de0e877844528dfaa9fdf777062d92c7e215bd3bb3309575`

从共同P0恢复生成器、匹配AdamW和RNG；不从B或W的22500终点续训。结构与原W相同，唯一变化是组MSE系数。保留研究权重不等于晋级默认；结论见REPORT.md。
