# 当前保留模型与流程身份

依据2026-10-25实际清理完成回执；以下是资产可用性与证据范围，不是统一排行榜，也不改变现有CLI默认。大权重的 hash 由最终清理核验验证，本文不重新加载模型。

| 角色 | 保留内容与用途 | 证据边界 |
| --- | --- | --- |
| Chair 主研究流程 | PLAIN_S2000完整恢复与推理权重 + LONG C-only头 + 官方Chair D0；GEO_C/来源角序 → HINGE_OPP015，ALPHA/HOLE按归档合同使用 | 保存已验证的有限几何/视觉正结果；CPU后处理无额外神经权重 |
| Chair BACK 分支 | MULTIPART_S1500完整恢复、推理与配套占用头 | 独立的靠背正结果，不自动替代S2000 |
| Chair 比较与维护 | FULL_S6000完整状态/推理；START维护默认与原硬链接名称 | FULL为充分训练比较参考，START仍是维护CLI默认 |
| Bench 起点 | COMMON12000完整状态/推理，官方Bench生成器及D0 | 保护历史来源与公共模型 |
| Bench 当前研究参考 | Native6000、SEAM3000完整状态与推理，固定C-only小头 | 用户已保留的完整座面路线；不宣称DEV整体泛化已解决 |
| Bench Coverage1500 | 最终完整状态与推理，现成RAW/评测数据 | 新输入上的RAW组织有有限正证据，不能取代SEAM3000整体参考 |
| Bench SHAPE500后端 | candidate_denoiser与step0500完整恢复 | 保形/视觉局部收益保留，但未通过整体几何门槛，不替换官方D0 |

RELDEF500权重退役：相对SHAPE没有新增严格视觉正例，DEV冲突代价仍在；原预测提议、网格、指标、图审和报告保留。最近RAW局部修复试点未通过数据准入，未产生新权重。

补充清理已移除第一阶段暂留的237份旧AE/FM/PONQ小权重。以下表格为补充后实际仍在的资产；最终清单与hash见[effective_retained_assets.json](effective_retained_assets.json)。

## 精确保留权重

| 角色 | 文件 | GiB | SHA256 |
| --- | --- | ---: | --- |
| 公开教师或保留的小型测试资产（按路径区分） | [E:/MeshFlow-Control-artifacts/experiments/chair_contextual_alignment_sie_v1/teacher/assets/model.safetensors](<../../../MeshFlow-Control-artifacts/experiments/chair_contextual_alignment_sie_v1/teacher/assets/model.safetensors>) | 0.757 | `f0cee7c815135c44a515eff72ab3040499744920442bc25567cd04efc93f8f65` |
| 官方资产（保留） | [E:/MeshFlow/checkpoints/official_bench_denoiser/0021000.pt](<../../checkpoints/official_bench_denoiser/0021000.pt>) | 6.789 | `67765c728c503b61c21624c05997ea026c9de486d1529052fafa134be3336b14` |
| 公开教师或保留的小型测试资产（按路径区分） | [E:/MeshFlow/experiments/mf_h1_local0/outputs/a_then_fc_20260930_001700/training_stream_state.pt](<../../experiments/mf_h1_local0/outputs/a_then_fc_20260930_001700/training_stream_state.pt>) | 0.000 | `cb85730651de3f4dd2c515a2c19b1f6bb4174680af2cf80a6a83966cf105d30e` |
| Bench SHAPE500：保形候选，未晋级 | [E:/MeshFlow/experiments/r2_bench_backend_shape_preservation_v1/training/candidate_denoiser.pt](<../../experiments/r2_bench_backend_shape_preservation_v1/training/candidate_denoiser.pt>) | 1.697 | `60f0aaf0003907913ed06d7d40f516fa88cf912517d37e8b155056acd298a5cc` |
| Bench SHAPE500：保形候选，未晋级 | [E:/MeshFlow/experiments/r2_bench_backend_shape_preservation_v1/training/checkpoints/step0500.pt](<../../experiments/r2_bench_backend_shape_preservation_v1/training/checkpoints/step0500.pt>) | 5.092 | `ee34b1605f1a37437556407940c537154c611bc310c20f1b0dd48ce2c9ee94a1` |
| Bench Native6000 | [E:/MeshFlow/experiments/r2_bench_full_seat_curriculum_v1/training/NATIVE/checkpoint_step6000.pt](<../../experiments/r2_bench_full_seat_curriculum_v1/training/NATIVE/checkpoint_step6000.pt>) | 1.458 | `31aef34d1126c4ecb7a22201ec3b874288417d7b57dda8e56f7dfe7eb49a8bfe` |
| Bench Native6000 | [E:/MeshFlow/experiments/r2_bench_full_seat_curriculum_v1/training/NATIVE/generator_step6000.pt](<../../experiments/r2_bench_full_seat_curriculum_v1/training/NATIVE/generator_step6000.pt>) | 0.486 | `3d296ab1067c6edd1ddbafad759e75090970739aaac1a9ccc23ac9ecc9467ede` |
| Bench SEAM3000 当前研究参考 | [E:/MeshFlow/experiments/r2_bench_full_seat_curriculum_v1/training/SEAM/checkpoint_step3000.pt](<../../experiments/r2_bench_full_seat_curriculum_v1/training/SEAM/checkpoint_step3000.pt>) | 1.458 | `db85c70b5d1014387f741003b19347be29a077246a1cc9c1c497c1e5012e39f9` |
| Bench SEAM3000 当前研究参考 | [E:/MeshFlow/experiments/r2_bench_full_seat_curriculum_v1/training/SEAM/generator_step3000.pt](<../../experiments/r2_bench_full_seat_curriculum_v1/training/SEAM/generator_step3000.pt>) | 0.486 | `3b4da0254584d3140b430eee50183eb50d04fff6dde83a237dd163efbf2d712d` |
| Bench COMMON12000 历史起点 | [E:/MeshFlow/experiments/r2_bench_parts_convergence_v1/training/COMMON/generator_step12000.pt](<../../experiments/r2_bench_parts_convergence_v1/training/COMMON/generator_step12000.pt>) | 0.486 | `a8eb21c00a05ef5965ce1b7783d773593efc7013665d784ee7cb1dd3096abcc1` |
| Bench COMMON12000 历史起点 | [E:/MeshFlow/experiments/r2_bench_parts_convergence_v1/training/COMMON/step12000.pt](<../../experiments/r2_bench_parts_convergence_v1/training/COMMON/step12000.pt>) | 1.458 | `40aecccd1838a5b04dadeac13e72b9cf7315f3721c16e180b97844a04c2f8032` |
| Bench Coverage1500：RAW 有限正结果 | [E:/MeshFlow/experiments/r2_bench_seat_coverage_v1/training/NATIVE/checkpoint_step1500.pt](<../../experiments/r2_bench_seat_coverage_v1/training/NATIVE/checkpoint_step1500.pt>) | 1.458 | `68703028dc20d84f210daa5b8c6eb4468299bf5aa2a5bace8881e8bd500fcf10` |
| Bench Coverage1500：RAW 有限正结果 | [E:/MeshFlow/experiments/r2_bench_seat_coverage_v1/training/NATIVE/generator_step1500.pt](<../../experiments/r2_bench_seat_coverage_v1/training/NATIVE/generator_step1500.pt>) | 0.486 | `30a50ab69ef52c8b4549b5c33671252ab6c1816a32578e6611f9c7b258e4acc3` |
| Bench 冻结 C-only 布局头 | [E:/MeshFlow/experiments/r2_bench_three_family_v1/production_contract/old_s15000_head_only.pt](<../../experiments/r2_bench_three_family_v1/production_contract/old_s15000_head_only.pt>) | 0.000 | `9802355787ff2abf9c02f983f9ca8547bec2182c32162a309cb5048927d51ce1` |
| Chair MULTIPART_S1500：BACK 有限正结果 | [E:/MeshFlow/experiments/r2_chair_multipart_seam_v1/inference/MULTIPART_S1500_generator.pt](<../../experiments/r2_chair_multipart_seam_v1/inference/MULTIPART_S1500_generator.pt>) | 0.486 | `2234a5b20a5a1cace60ce8bbcea33b711dad0a34f53f75369da3a49dae4f6855` |
| Chair MULTIPART_S1500：BACK 有限正结果 | [E:/MeshFlow/experiments/r2_chair_multipart_seam_v1/inference/MULTIPART_S1500_occupancy.pt](<../../experiments/r2_chair_multipart_seam_v1/inference/MULTIPART_S1500_occupancy.pt>) | 0.000 | `58e1fd5077bd797ae59dbcb71d1528766b47c2bf298e090daeae8674ddac8f86` |
| Chair MULTIPART_S1500：BACK 有限正结果 | [E:/MeshFlow/experiments/r2_chair_multipart_seam_v1/training/checkpoint_step1500.pt](<../../experiments/r2_chair_multipart_seam_v1/training/checkpoint_step1500.pt>) | 1.458 | `9e3181eed193d12c187dec5c2608f129d1c3c9ea02f097605455685710f35f7b` |
| Chair FULL_S6000 方法比较参考 | [E:/MeshFlow/experiments/r2_full_seat_learning_curve_v1/checkpoints/FULL_SEAT_step6000.pt](<../../experiments/r2_full_seat_learning_curve_v1/checkpoints/FULL_SEAT_step6000.pt>) | 1.458 | `5f41df3d3c11a88fdeb9ad70682a4b18020a1d5838ee08a1956e5ba2c927af33` |
| Chair FULL_S6000 方法比较参考 | [E:/MeshFlow/experiments/r2_full_seat_learning_curve_v1/inference/FULL_S6000_generator.pt](<../../experiments/r2_full_seat_learning_curve_v1/inference/FULL_S6000_generator.pt>) | 0.486 | `86b38d35c44d3a6fd83a5b8144b57813f5238efab557e0cf46a7872bbde5c5cc` |
| 官方 Chair D0 | [E:/MeshFlow/experiments/r2_postprocess_avmr_pmr_v1/assets/chair_denoiser_0030000.pt](<../../experiments/r2_postprocess_avmr_pmr_v1/assets/chair_denoiser_0030000.pt>) | 6.789 | `6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36` |
| Chair PLAIN_S2000 完整恢复 | [E:/MeshFlow/experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step2000.pt](<../../experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step2000.pt>) | 1.458 | `ea6675a8d77364936fb13443ea0378011423404ee28a8f3074b7f8bd4ef3f56a` |
| Chair S2000 配套 LONG C-only 头 | [E:/MeshFlow/experiments/r2_seam_quality_four_v1/inference/LONG_occupancy.pt](<../../experiments/r2_seam_quality_four_v1/inference/LONG_occupancy.pt>) | 0.000 | `6e8f9ab606403c9c2b96d7f6bf6a2c67dacd11d4c45179369414e420a75982df` |
| Chair PLAIN_S2000 推理 | [E:/MeshFlow/experiments/r2_seam_quality_wave1_wave2_v1/inference/PLAIN_S2000_generator.pt](<../../experiments/r2_seam_quality_wave1_wave2_v1/inference/PLAIN_S2000_generator.pt>) | 0.486 | `46be9a29d6d9e150205cc46cf8c952ae1cd32b2aa58bff24cb879eb451932f43` |
| Chair START 维护默认 | [E:/MeshFlow/native_t1/checkpoints/chair_hybrid_start_total5000.pt](<../../native_t1/checkpoints/chair_hybrid_start_total5000.pt>) | 1.458 | `d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80` |
| Chair START 维护默认 | [E:/MeshFlow/native_t1/runs/chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt](<../../native_t1/runs/chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt>) | 1.458 | `d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80` |
| 公开教师或保留的小型测试资产（按路径区分） | [E:/MeshFlow/native_t1/runs/chair_moment_channel_v2/revisions/pretraining_recovery_fix/cpu_fixture_synthetic_only/checkpoints/MOMENT_XY_moment200_total5200.pt](<../../native_t1/runs/chair_moment_channel_v2/revisions/pretraining_recovery_fix/cpu_fixture_synthetic_only/checkpoints/MOMENT_XY_moment200_total5200.pt>) | 0.001 | `dbd4ac30d6b70546aee68eb58f3e0f8c008e7f2af8c44b3c1055183c92bcbed5` |
| 官方资产（保留） | [E:/MeshFlow/native_t1/runs/control_mvp48_v1/checkpoints/official_chair_last.pt](<../../native_t1/runs/control_mvp48_v1/checkpoints/official_chair_last.pt>) | 1.941 | `bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d` |
| 官方资产（保留） | [E:/MeshFlow/output/120m-ot-v-bench/checkpoints/01000000.pt](<../../output/120m-ot-v-bench/checkpoints/01000000.pt>) | 1.941 | `00fc5ddb5bc5690cd53fe4f092ee3320e23437a287464fa5639120c2bb82a505` |

同名历史中间/滚动文件不被视为终点副本；只保留本表列出的实际资产。START的两个保留路径为同一硬链接分配，不能按两个独立物理大文件计算空间。

全部保护文件（含数据清单、证据和归档）见 [prepared_manifest.json](<prepared_manifest.json>)；实际移除路径见 [清理报告](<CLEANUP_REPORT.md>)。保留终点仍需其既有配置、元数据和数据合同；不从缺失历史检查点静默恢复，不自动启动任何新任务。
