# 本轮实际终点权重

两条训练均完成固定预算、匹配优化器和随机状态核验。以下是研究终点，不表示已晋级或切换默认模型。原 B、M、官方模型及旧资产未修改；小审阅包不含大权重。

| 终点 | 完整恢复状态 | 推理导出 | 累计更新 |
|---|---|---|---:|
| EXPOSURE（M +3000 Native） | [checkpoint_step3000.pt](exposure/training/M_PLUS3000/checkpoint_step3000.pt) | [generator_step3000.pt](exposure/training/M_PLUS3000/generator_step3000.pt) | 25500 |
| SEAM（B +1500） | [checkpoint_step1500.pt](seam/training/SEAM1500/checkpoint_step1500.pt) | [generator_step1500.pt](seam/training/SEAM1500/generator_step1500.pt) | 24000 |

完整恢复文件含 live 生成器、匹配 AdamW、原小头和恢复元数据；约1.565GB。推理导出约0.522GB。SEAM 继续使用冻结原 C-only 小头；EXPOSURE 推理不调用小头。

文件 SHA256（不是 generator state hash）：

```text
EXPOSURE checkpoint_step3000.pt
3c1e9e1db0f9a462632bd70a91b5acaa4bef1dd40ad20ad90a194ab17bbd49fd
EXPOSURE generator_step3000.pt
70e51d7913f99ad017d023e47965387b28692075ed9964b61d9d55dd1f0ff6b3
SEAM checkpoint_step1500.pt
dfb6a6e831cad0694f6f17188c4a7eff5f6b6ae407cafbbb135c0ca3be788097
SEAM generator_step1500.pt
3161771ace3ba5a1ae28aecca3cee5accb5a47e5a62127da91f9134d69af99e1
```

逐调用、真实曝光、参数与优化器身份见 [EXPOSURE 收口](exposure/closeout/training.json) 和 [SEAM 收口](seam/closeout/training.json)。两者均无失败、重放或额外探针。CPU HYBRID 与 BESTOF4 不产生新权重。
