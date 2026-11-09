# 2026-10-25 资产退役补记

原实验/资产目录：`E:/MeshFlow/experiments/mf_h1_local0`。

本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。

实际移除 23 份权重及 0 份已有校验 ZIP 的展开审阅副本，逻辑大小 0.234 GiB。

以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。

若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。

原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。

| 移除路径 | 类型 | Bytes | SHA256 | 原因 |
| --- | --- | ---: | --- | --- |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/inference_injection_pilot_20260928_183412/stage_b/checkpoints/adapter_step0000.pt` | RETIRED_WEIGHT | 3858827 | `02672738f2f60a22b399cc37965c9f0158837a1cb4ee43de08978b52184cadea` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/inference_injection_pilot_20260928_183412/stage_b/checkpoints/adapter_step0100.pt` | RETIRED_WEIGHT | 11497654 | `8e6ab476e32b270100cf5e81b99bea69837c5ac7223bf5b3baa8bbabed93ee8e` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/inference_injection_pilot_20260928_183412/stage_b/checkpoints/adapter_step0250.pt` | RETIRED_WEIGHT | 11497654 | `194c8df171cf132d84ee3a697f355f73b9d0d81f8085838902aaa765634dee3f` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/inference_injection_pilot_20260928_183412/stage_b/checkpoints/adapter_step0500.pt` | RETIRED_WEIGHT | 11497654 | `76d6a9d4ee9a38c18a709599272ab9392a1ecfe3575e6e8a3c7345f867bff2fc` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/learning_diagnosis_20260928_155505/Q_correct/checkpoints/adapter_step0000.pt` | RETIRED_WEIGHT | 4780273 | `e7af2c1d94c44ea3d0377ef223fb918d01c4b2b8a7e1187dace35fcd8552c1be` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/learning_diagnosis_20260928_155505/Q_correct/checkpoints/adapter_step0200.pt` | RETIRED_WEIGHT | 14432637 | `1a660bf278381c76f4531a07ae0fe139fdba979b76e349d9d5677304ca35dae5` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/learning_diagnosis_20260928_155505/Q_correct/checkpoints/adapter_step0500.pt` | RETIRED_WEIGHT | 14691517 | `c361835b5d3b0cecd6e94e0cabe730c85db30413659444a0041ae1597900429c` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/learning_diagnosis_20260928_155505/Q_correct/checkpoints/adapter_step1000.pt` | RETIRED_WEIGHT | 15123069 | `034325073569edf82149cd11b91895e8c95135d9bb2603b2c8244a513ec04a8a` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/learning_diagnosis_20260928_155505/Q_independent/checkpoints/adapter_step0000.pt` | RETIRED_WEIGHT | 4780273 | `bd72b913dd90e2ec1a5da1e1e17c1febfdf565f4cdb5d14492a72a3cacc4fb33` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/learning_diagnosis_20260928_155505/Q_independent/checkpoints/adapter_step0200.pt` | RETIRED_WEIGHT | 14432637 | `69597d554861fbf85dca5d446bd58790b0c87f4308a1a69cc36ce103fc77466d` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/learning_diagnosis_20260928_155505/Q_independent/checkpoints/adapter_step0500.pt` | RETIRED_WEIGHT | 14691517 | `5890d422082c94729b7ba1ddf6fd9f9992728bdcd12a7ced4f903c90ae396d72` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/learning_diagnosis_20260928_155505/Q_independent/checkpoints/adapter_step1000.pt` | RETIRED_WEIGHT | 15123069 | `87b2bc41e69341aeaa58b29d64794634c6983af71c56304ffc97ea9aa7a295eb` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/nxC_112_sandbox_20260928_173427/stage_b/checkpoints/adapter_step0000.pt` | RETIRED_WEIGHT | 3855947 | `257f85b7bf9f98c555196e495b8b90c49fd2030cf69ba382a6a91d4033708329` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/nxC_112_sandbox_20260928_173427/stage_b/checkpoints/adapter_step0100.pt` | RETIRED_WEIGHT | 11494774 | `117f6f8500de119d36937b403f1292800c677fd75c5705c1ec00fd7151d67527` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/nxC_112_sandbox_20260928_173427/stage_b/checkpoints/adapter_step0250.pt` | RETIRED_WEIGHT | 11494774 | `dcff67c777c7c61634cd5f6b83a58970b2ec8048c21e96725c396f14fc9fb65a` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/nxC_112_sandbox_20260928_173427/stage_b/checkpoints/adapter_step0500.pt` | RETIRED_WEIGHT | 11494774 | `63b571dc17304c7232f34ae8ec34679f50b6be7e1f88344793348a6a70d3cf9e` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/run_20260928_01/G_closest/checkpoints/adapter_step0100.pt` | RETIRED_WEIGHT | 14286703 | `1adc0e1ce4f4fa500db95c59000411ac5e6e0691c216200a33e580d87073d11b` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/run_20260928_01/G_closest/checkpoints/adapter_step0200.pt` | RETIRED_WEIGHT | 14324527 | `3baf57473d451b45dffd65ae98015b73ffbc056495159f3cc3cae93b67a25c4f` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/run_20260928_01/initial_adapter_seed10.pt` | RETIRED_WEIGHT | 4735821 | `676e815cd9dc2ea5f7c08dcb7170bc607e917493f87a4a6baa961f8485e82a2b` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/run_20260928_01/official_tests/official_probe_inputs.pt` | RETIRED_WEIGHT | 8198 | `8fb562402d699895685cc01121f6eef34be040a5a072afb4c3cfa3ac09857a26` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/run_20260928_01/official_tests/official_probe_resume_step5.pt` | RETIRED_WEIGHT | 14231757 | `181abc930119e4a6a3acfdb53679fd38ad13a426612f5366cf6214f223ceceed` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/run_20260928_01/Q_center/checkpoints/adapter_step0100.pt` | RETIRED_WEIGHT | 14286703 | `f17e04f19e2167f4000a7c6b97a3bb7f7e704242b50ada285871cec051f85c1d` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |
| `E:/MeshFlow/experiments/mf_h1_local0/outputs/run_20260928_01/Q_center/checkpoints/adapter_step0200.pt` | RETIRED_WEIGHT | 14324527 | `95c58717124a8095c83e9d6768b57a03c7918ad86b227b90009bfad1d5e5d98c` | Superseded/closed historical Chair candidate or continuation; not needed to load selected S2000, START or FULL_S6000 |

本目录继续保留的权重：

- [training_stream_state.pt](<outputs/a_then_fc_20260930_001700/training_stream_state.pt>)：0.000 GiB，SHA256 `cb85730651de3f4dd2c515a2c19b1f6bb4174680af2cf80a6a83966cf105d30e`。

完整清理事实见 [清理报告](<../../maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md>) 与 [实际完成回执](<../../maintenance/storage_cleanup_20261025/cleanup_receipt.json>)。

本次不切换默认模型，不训练、不生成，不 commit/push。
