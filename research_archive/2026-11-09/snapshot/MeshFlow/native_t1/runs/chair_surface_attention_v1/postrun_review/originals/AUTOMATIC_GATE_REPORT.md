# Chair Finite Surface Attention V1

实际调用摘要：`{"actual_full_sampling_attempts": 0, "actual_full_sampling_returns": 0, "applicable_unique_generation_requests": 552, "failed_actual_calls": 0, "generation_not_applicable": 24, "historical_raw_cache_hits": 138, "isolated_module_test_updates": 1, "optimizer_updates": 0, "precheck_forwards": 10, "precheck_zero_update_backwards": 1, "processed_generation_not_applicable": 6, "processed_generation_records": 144, "registered_generation_request_limit": 576, "registered_unique_generation_requests": 576, "sampling_forwards": 0, "scientific_optimizer_update_limit": 3000, "total_model_forwards": 10, "training_backwards": 0, "training_microbatch_forwards": 0, "unresolved_calls": 0}`。

IMPLEMENTED：IMPLEMENTATION_BLOCKED；TRAINED：NOT_RUN；EVALUATED：INCOMPLETE；QUALITY_RESULT：**NA_NOT_EVALUATED**。

共同起点为 chair_hybrid_coupling_v1 / OT_HYBRID / pilot1000 / conditional_total5000 / λ=.25。新终点为 surface_step1000、conditional_total6000。H_CONT控制等预算续训；MOMENT_XY与SURFACE_XY同容量、同接入方式，区别仅为几何关系定义。

## 数值与成本门禁

门禁状态：**IMPLEMENTATION_BLOCKED**。未通过门禁时正式训练和质量比较保持NOT_RUN/NA；不能把未运行当成方法质量失败。

```json
{
  "attention": {
    "path": "tests/attention/results.json",
    "status": "PASS"
  },
  "changes_to_nodes_bandwidth_data_thresholds": 0,
  "continuation_after_numeric_failure": "Only the already-budgeted independent real-model wiring/cost audit; no formal training/sampling",
  "cost": {
    "arms": {
      "H_CONT": {
        "measured_pairs": 0,
        "median_backward_cuda_milliseconds": null,
        "median_forward_backward_seconds": null,
        "median_forward_cuda_milliseconds": null,
        "peak_allocated_bytes": null,
        "peak_reserved_bytes": null
      },
      "MOMENT_XY": {
        "measured_pairs": 0,
        "median_backward_cuda_milliseconds": null,
        "median_forward_backward_seconds": null,
        "median_forward_cuda_milliseconds": null,
        "peak_allocated_bytes": null,
        "peak_reserved_bytes": null
      },
      "SURFACE_XY": {
        "measured_pairs": 0,
        "median_backward_cuda_milliseconds": null,
        "median_forward_backward_seconds": null,
        "median_forward_cuda_milliseconds": null,
        "peak_allocated_bytes": null,
        "peak_reserved_bytes": null
      }
    },
    "path": "precheck/result.json",
    "per_fixture": [
      {
        "complete": false,
        "fixture_id": "ordinary_20",
        "medians": {
          "H_CONT": {
            "forward_backward_seconds": null,
            "forward_seconds": null,
            "pairs": 0
          },
          "MOMENT_XY": {
            "forward_backward_seconds": null,
            "forward_seconds": null,
            "pairs": 0
          },
          "SURFACE_XY": {
            "forward_backward_seconds": null,
            "forward_seconds": null,
            "pairs": 0
          }
        },
        "passed": false,
        "surface_to_H_CONT_forward_backward_median_ratio": null,
        "surface_to_H_CONT_forward_median_ratio": null,
        "task_id": "e682c43e2f4e7463d0ad32b8d8cec005_train20_2"
      },
      {
        "complete": false,
        "fixture_id": "maximum_20",
        "medians": {
          "H_CONT": {
            "forward_backward_seconds": null,
            "forward_seconds": null,
            "pairs": 0
          },
          "MOMENT_XY": {
            "forward_backward_seconds": null,
            "forward_seconds": null,
            "pairs": 0
          },
          "SURFACE_XY": {
            "forward_backward_seconds": null,
            "forward_seconds": null,
            "pairs": 0
          }
        },
        "passed": false,
        "surface_to_H_CONT_forward_backward_median_ratio": null,
        "surface_to_H_CONT_forward_median_ratio": null,
        "task_id": "d6dd3de6d6ea6d0b1931ace01cf1b948_train20_7"
      },
      {
        "complete": false,
        "fixture_id": "ordinary_40",
        "medians": {
          "H_CONT": {
            "forward_backward_seconds": null,
            "forward_seconds": null,
            "pairs": 0
          },
          "MOMENT_XY": {
            "forward_backward_seconds": null,
            "forward_seconds": null,
            "pairs": 0
          },
          "SURFACE_XY": {
            "forward_backward_seconds": null,
            "forward_seconds": null,
            "pairs": 0
          }
        },
        "passed": false,
        "surface_to_H_CONT_forward_backward_median_ratio": null,
        "surface_to_H_CONT_forward_median_ratio": null,
        "task_id": "47da08d9c7cd7e104b3c42e318f3affc_train40_5"
      },
      {
        "complete": false,
        "fixture_id": "maximum_40",
        "medians": {
          "H_CONT": {
            "forward_backward_seconds": null,
            "forward_seconds": null,
            "pairs": 0
          },
          "MOMENT_XY": {
            "forward_backward_seconds": null,
            "forward_seconds": null,
            "pairs": 0
          },
          "SURFACE_XY": {
            "forward_backward_seconds": null,
            "forward_seconds": null,
            "pairs": 0
          }
        },
        "passed": false,
        "surface_to_H_CONT_forward_backward_median_ratio": null,
        "surface_to_H_CONT_forward_median_ratio": null,
        "task_id": "d6dd3de6d6ea6d0b1931ace01cf1b948_train40_7"
      }
    ],
    "process_budget_mib": 10199.55,
    "process_memory_note": "NVIDIA process memory may be NA under WSL; never substituted with Torch allocated. Sampled global used may bound process memory conservatively at sample instants; brief unsampled peaks remain possible. Global reserve and Torch peaks are independent.",
    "reserve_mib": 2445.45,
    "rows_path": "/mnt/e/MeshFlow/native_t1/runs/chair_surface_attention_v1/precheck/cost_rows.json",
    "status": "INCOMPLETE",
    "threshold": 3.0,
    "timing_scope": "Single serial worker; same four tasks and two pairs each. Wall whole F/B includes synchronized runtime checks and loss construction, ending before gradient inspection; CUDA forward/backward events separately retained."
  },
  "extra_I_P_model_calls": 0,
  "formal_training_allowed": false,
  "numeric": {
    "accuracy_pass_fixtures": 0,
    "maxima": {
      "extreme_max_errors": {
        "compressed_mass_abs": 0.0063674240569291385,
        "covariance_Frobenius": 0.006247837299453371,
        "mu_L2": 0.19206912945515223
      },
      "ordinary_max_errors": {
        "compressed_mass_abs": 0.09744185411720929,
        "covariance_Frobenius": 0.01376171015075803,
        "mu_L2": 0.10853565363665542
      },
      "reference_max_errors": {
        "compressed_mass_abs": 0.00015928856304281425,
        "covariance_Frobenius": 2.8087208388407445e-05,
        "mu_L2": 0.0011148664959007117
      }
    },
    "ordinary_threshold": 0.01,
    "reference_pass_fixtures": 3,
    "reference_scope": "Fixed subsets only; one subset not converged, no certified true-integral bound",
    "reference_threshold": 0.001,
    "status": "IMPLEMENTATION_BLOCKED",
    "summary_path": "tests/relations/relations_gate_summary.json"
  },
  "real_model": {
    "actual_backwards": 1,
    "actual_forwards": 10,
    "comparisons": [
      {
        "arm": "H_CONT",
        "atol": 1e-06,
        "exact": false,
        "finite": true,
        "fixture_id": "ordinary_20",
        "kind": "original_repeat",
        "max_abs": 0.0,
        "outside_tolerance": 0,
        "passed": true,
        "precision": "FP32",
        "repeat": 1,
        "rms": 0.0,
        "rtol": 1e-05,
        "stage": "zero_init"
      },
      {
        "arm": "H_CONT",
        "atol": 0.0,
        "exact": true,
        "finite": true,
        "fixture_id": "ordinary_20",
        "kind": "original_repeat",
        "max_abs": 0.0,
        "outside_tolerance": 0,
        "passed": true,
        "precision": "BF16",
        "repeat": 1,
        "rms": 0.0,
        "rtol": 0.0,
        "stage": "zero_init"
      },
      {
        "arm": "H_CONT",
        "atol": 1e-06,
        "exact": false,
        "finite": true,
        "fixture_id": "ordinary_40",
        "kind": "original_repeat",
        "max_abs": 0.0,
        "outside_tolerance": 0,
        "passed": true,
        "precision": "FP32",
        "repeat": 1,
        "rms": 0.0,
        "rtol": 1e-05,
        "stage": "zero_init"
      },
      {
        "arm": "H_CONT",
        "atol": 0.0,
        "exact": true,
        "finite": true,
        "fixture_id": "ordinary_40",
        "kind": "original_repeat",
        "max_abs": 0.0,
        "outside_tolerance": 0,
        "passed": true,
        "precision": "BF16",
        "repeat": 1,
        "rms": 0.0,
        "rtol": 0.0,
        "stage": "zero_init"
      }
    ],
    "errors": [
      {
        "arm": "H_CONT",
        "error": "TypeError(\"dict() got multiple values for keyword argument 'time_unix'\")",
        "kind": "ACTUAL_EXCEPTION_NO_RETRY",
        "traceback": "Traceback (most recent call last):\n  File \"/mnt/e/MeshFlow/experiments/mf_h1_local0/chair_surface_attention_v1/precheck.py\", line 288, in run\n    monitor.poll();memory=monitor.summary(started)\n  File \"/mnt/e/MeshFlow/experiments/mf_h1_local0/chair_surface_attention_v1/precheck.py\", line 137, in poll\n    with self.lock:self.rows.append(row);append_event(self.path,row)\n  File \"/mnt/e/MeshFlow/experiments/mf_h1_local0/chair_hybrid_coupling_v1/common.py\", line 64, in append_event\n    row = dict(time_unix=time.time(), pid=os.getpid(), **value)\nTypeError: dict() got multiple values for keyword argument 'time_unix'\n"
      },
      {
        "backward_attempts": 1,
        "forward_attempts": 10,
        "kind": "FROZEN_SCHEDULE_INCOMPLETE"
      }
    ],
    "path": "stages/precheck.json",
    "result_status": "IMPLEMENTATION_BLOCKED",
    "stage_status": "DONE"
  },
  "status": "IMPLEMENTATION_BLOCKED",
  "synthetic": {
    "path": "tests/relations/relations_synthetic.json",
    "status": "PASS"
  }
}
```

192点候选、12288点参考与49152点收敛复核分别登记。关系准确度不是完整模型几何评分；CPU求积MSE不能与fixed_query_MSE或双向RMS混用。成本门槛是同任务完整forward及forward+backward中位耗时不超过H_CONT的3倍，并保留至少15%显存余量。

## 固定主终点

`fixed_query_MSE = mean([d(q,F)/L0]^2)`，q为原512个冻结GT free coverage查询点，点到三角表面距离，A20 alpha=1时L=L0。先父内平均两份原噪声，再8父等权。逐父配对差中位数不是两个组中位数之差；缺失任一父/噪声不能缩分母宣布通过。

|比较|父均值差|逐父差中位数|改善父/8|最差退化父|最差差值|几何候选|
|---|---|---|---|---|---|---|
|H_CONT-START|NA|NA|NA/8|NA|NA|NA|
|MOMENT_XY-H_CONT|NA|NA|NA/8|NA|NA|NA|
|MOMENT_XY-START|NA|NA|NA/8|NA|NA|NA|
|SURFACE_XY-H_CONT|NA|NA|NA/8|NA|NA|NA|
|SURFACE_XY-MOMENT_XY|NA|NA|NA/8|NA|NA|NA|
|SURFACE_XY-START|NA|NA|NA/8|NA|NA|NA|

|比较|噪声|均值差|改善父|有效父|
|---|---|---|---|---|
|H_CONT-START|0|NA|NA|0|
|H_CONT-START|1|NA|NA|0|
|MOMENT_XY-H_CONT|0|NA|NA|0|
|MOMENT_XY-H_CONT|1|NA|NA|0|
|MOMENT_XY-START|0|NA|NA|0|
|MOMENT_XY-START|1|NA|NA|0|
|SURFACE_XY-H_CONT|0|NA|NA|0|
|SURFACE_XY-H_CONT|1|NA|NA|0|
|SURFACE_XY-MOMENT_XY|0|NA|NA|0|
|SURFACE_XY-MOMENT_XY|1|NA|NA|0|
|SURFACE_XY-START|0|NA|NA|0|
|SURFACE_XY-START|1|NA|NA|0|

几何候选要求均值差<0、逐父差中位数<0且至少6/8父改善。这是单配对训练流开发性筛选，不是显著性检验。只有SURFACE同时胜过H_CONT与MOMENT，才给有限表面候选初步几何支持。

## 两个实际表面方向、覆盖与质量

GT→F与F→GT均给实际数值。它们使用原seed9911双向面积采样合同；fixed_query使用另一组seed9973点。下表方向指标单位为L归一RMS，保留原%L字段。

|检查点|表/划分/C|父数|GT→F RMS/L|F→GT RMS/L|双向 RMS/L|fixed MSE|覆盖|p95/L|预测小角率|同GT小角率|
|---|---|---|---|---|---|---|---|---|---|---|
|H_CONT|A/train/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|H_CONT|A/train/0.4|0|NA|NA|NA|NA|NA|NA|NA|NA|
|H_CONT|A/validation/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|H_CONT|A/validation/0.4|0|NA|NA|NA|NA|NA|NA|NA|NA|
|H_CONT|C/holdout/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|MOMENT_XY|A/train/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|MOMENT_XY|A/train/0.4|0|NA|NA|NA|NA|NA|NA|NA|NA|
|MOMENT_XY|A/validation/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|MOMENT_XY|A/validation/0.4|0|NA|NA|NA|NA|NA|NA|NA|NA|
|MOMENT_XY|C/holdout/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|START|A/train/0.2|8|0.04237942|0.04216359|0.04426548|0.00381398|0.5863037|0.07845537|0.3446443|0.3513428|
|START|A/train/0.4|7|0.02549237|0.03344236|0.033691|0.001721154|0.7357701|0.05646623|0.4113901|0.4213715|
|START|A/validation/0.2|8|0.03549068|0.03790222|0.03713011|0.001531074|0.322998|0.06884473|0.1467566|0.1782926|
|START|A/validation/0.4|6|0.02350151|0.02482701|0.02496825|0.0006156471|0.4890951|0.04776093|0.07139449|0.1064386|
|START|C/holdout/0.2|8|0.06890054|0.06177093|0.06630164|0.005545295|0.1340942|0.1167513|0.1553401|0.3480847|
|SURFACE_XY|A/train/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|SURFACE_XY|A/train/0.4|0|NA|NA|NA|NA|NA|NA|NA|NA|
|SURFACE_XY|A/validation/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|
|SURFACE_XY|A/validation/0.4|0|NA|NA|NA|NA|NA|NA|NA|NA|
|SURFACE_XY|C/holdout/0.2|0|NA|NA|NA|NA|NA|NA|NA|NA|

同GT free角度使用原shape_quality公式，保存全部角点角度、每面最小角和分位数；零边沿用原cosine=1的数值约定，不把零面积面称作合法面积表面。该参照只用于评分，未进入模型或loss。

|比较|表/划分/C|指标|配对均值变化|改善父|退化父|有效/登记|
|---|---|---|---|---|---|---|
|SURFACE_XY-H_CONT|A/train/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.2|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.2|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.2|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.4|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.4|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.4|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.4|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.4|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/train/0.4|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.2|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.2|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.2|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.4|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.4|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.4|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.4|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.4|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|A/validation/0.4|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|C/holdout/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|C/holdout/0.2|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|C/holdout/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|C/holdout/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|C/holdout/0.2|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-H_CONT|C/holdout/0.2|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.2|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.2|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.2|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.4|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.4|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.4|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.4|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.4|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/train/0.4|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.2|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.2|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.2|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.4|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.4|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.4|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.4|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.4|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|A/validation/0.4|F_to_GT_original_RMS_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|C/holdout/0.2|Gamma_to_free_edges_RMS_bbox_pct|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|C/holdout/0.2|CF_proper_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|C/holdout/0.2|FF_illegal_affected_free_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|C/holdout/0.2|free_minimum_angle_lt5_fraction|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|C/holdout/0.2|fixed_query_p95_L|NA|NA|NA|0/8|
|SURFACE_XY-MOMENT_XY|C/holdout/0.2|F_to_GT_original_RMS_L|NA|NA|NA|0/8|

完整原方向、Gamma覆盖、未决相交、退化与有限值/C/N保持均在per_output、paired_per_parent和quality_paired_summary中。A训练/验证与20%/40%分开；原不可构造40%保持NA。C为已观察留出回归，不参与选择终点。

## 条件控制与解析参照

W为free角点X的q95−q05。r_base、b_self、b_GT分列；先逐曲线求|b−1|再平均，不能抵消欠响应与过响应。分母退化标UNDEFINED。矩阵E[alpha,beta]的行是输入、列是目标，A_column固定beta列；严格胜出需同时小于两个错误输入。

|比较|划分|指标|父均值变化|配对差中位数|
|---|---|---|---|---|
|H_CONT-START|train|b_self_abs_error|NA|NA|
|H_CONT-START|train|b_GT_abs_error|NA|NA|
|H_CONT-START|train|mean_A_column|NA|NA|
|H_CONT-START|train|all_three_strict_win|NA|NA|
|H_CONT-START|validation|b_self_abs_error|NA|NA|
|H_CONT-START|validation|b_GT_abs_error|NA|NA|
|H_CONT-START|validation|mean_A_column|NA|NA|
|H_CONT-START|validation|all_three_strict_win|NA|NA|
|MOMENT_XY-H_CONT|train|b_self_abs_error|NA|NA|
|MOMENT_XY-H_CONT|train|b_GT_abs_error|NA|NA|
|MOMENT_XY-H_CONT|train|mean_A_column|NA|NA|
|MOMENT_XY-H_CONT|train|all_three_strict_win|NA|NA|
|MOMENT_XY-H_CONT|validation|b_self_abs_error|NA|NA|
|MOMENT_XY-H_CONT|validation|b_GT_abs_error|NA|NA|
|MOMENT_XY-H_CONT|validation|mean_A_column|NA|NA|
|MOMENT_XY-H_CONT|validation|all_three_strict_win|NA|NA|
|MOMENT_XY-START|train|b_self_abs_error|NA|NA|
|MOMENT_XY-START|train|b_GT_abs_error|NA|NA|
|MOMENT_XY-START|train|mean_A_column|NA|NA|
|MOMENT_XY-START|train|all_three_strict_win|NA|NA|
|MOMENT_XY-START|validation|b_self_abs_error|NA|NA|
|MOMENT_XY-START|validation|b_GT_abs_error|NA|NA|
|MOMENT_XY-START|validation|mean_A_column|NA|NA|
|MOMENT_XY-START|validation|all_three_strict_win|NA|NA|
|SURFACE_XY-H_CONT|train|b_self_abs_error|NA|NA|
|SURFACE_XY-H_CONT|train|b_GT_abs_error|NA|NA|
|SURFACE_XY-H_CONT|train|mean_A_column|NA|NA|
|SURFACE_XY-H_CONT|train|all_three_strict_win|NA|NA|
|SURFACE_XY-H_CONT|validation|b_self_abs_error|NA|NA|
|SURFACE_XY-H_CONT|validation|b_GT_abs_error|NA|NA|
|SURFACE_XY-H_CONT|validation|mean_A_column|NA|NA|
|SURFACE_XY-H_CONT|validation|all_three_strict_win|NA|NA|
|SURFACE_XY-MOMENT_XY|train|b_self_abs_error|NA|NA|
|SURFACE_XY-MOMENT_XY|train|b_GT_abs_error|NA|NA|
|SURFACE_XY-MOMENT_XY|train|mean_A_column|NA|NA|
|SURFACE_XY-MOMENT_XY|train|all_three_strict_win|NA|NA|
|SURFACE_XY-MOMENT_XY|validation|b_self_abs_error|NA|NA|
|SURFACE_XY-MOMENT_XY|validation|b_GT_abs_error|NA|NA|
|SURFACE_XY-MOMENT_XY|validation|mean_A_column|NA|NA|
|SURFACE_XY-MOMENT_XY|validation|all_three_strict_win|NA|NA|
|SURFACE_XY-START|train|b_self_abs_error|NA|NA|
|SURFACE_XY-START|train|b_GT_abs_error|NA|NA|
|SURFACE_XY-START|train|mean_A_column|NA|NA|
|SURFACE_XY-START|train|all_three_strict_win|NA|NA|
|SURFACE_XY-START|validation|b_self_abs_error|NA|NA|
|SURFACE_XY-START|validation|b_GT_abs_error|NA|NA|
|SURFACE_XY-START|validation|mean_A_column|NA|NA|
|SURFACE_XY-START|validation|all_three_strict_win|NA|NA|

|比较|划分|beta|指标|变化|
|---|---|---|---|---|
|SURFACE_XY-H_CONT|train|0.9|A_column|NA|
|SURFACE_XY-H_CONT|train|0.9|column_strict_win|NA|
|SURFACE_XY-H_CONT|train|1|A_column|NA|
|SURFACE_XY-H_CONT|train|1|column_strict_win|NA|
|SURFACE_XY-H_CONT|train|1.1|A_column|NA|
|SURFACE_XY-H_CONT|train|1.1|column_strict_win|NA|
|SURFACE_XY-H_CONT|validation|0.9|A_column|NA|
|SURFACE_XY-H_CONT|validation|0.9|column_strict_win|NA|
|SURFACE_XY-H_CONT|validation|1|A_column|NA|
|SURFACE_XY-H_CONT|validation|1|column_strict_win|NA|
|SURFACE_XY-H_CONT|validation|1.1|A_column|NA|
|SURFACE_XY-H_CONT|validation|1.1|column_strict_win|NA|
|SURFACE_XY-MOMENT_XY|train|0.9|A_column|NA|
|SURFACE_XY-MOMENT_XY|train|0.9|column_strict_win|NA|
|SURFACE_XY-MOMENT_XY|train|1|A_column|NA|
|SURFACE_XY-MOMENT_XY|train|1|column_strict_win|NA|
|SURFACE_XY-MOMENT_XY|train|1.1|A_column|NA|
|SURFACE_XY-MOMENT_XY|train|1.1|column_strict_win|NA|
|SURFACE_XY-MOMENT_XY|validation|0.9|A_column|NA|
|SURFACE_XY-MOMENT_XY|validation|0.9|column_strict_win|NA|
|SURFACE_XY-MOMENT_XY|validation|1|A_column|NA|
|SURFACE_XY-MOMENT_XY|validation|1|column_strict_win|NA|
|SURFACE_XY-MOMENT_XY|validation|1.1|A_column|NA|
|SURFACE_XY-MOMENT_XY|validation|1.1|column_strict_win|NA|

完整矩阵、三目标全胜与同一输入三目标通吃、两噪声分别的曲线和列结果都有CSV。No-response和Post-scale只从各检查点自己的F1构造，Post-scale只由C1→C_alpha拟合；它们拥有旧条件与旧输出，不与单次生成冒称同输入合同。无额外I/P模型前向。

## 实际轨迹求积与成本

轨迹关系诊断状态：NOT_RUN。结果引用 `trajectory_quadrature_results.json`，仅使用预登记正式SURFACE轨迹k={0,25,49}，不重生成或改变主评分。

逐输出实际采样耗时及峰值见 sampling_cost_per_output.csv；完整账本见 accounting.json。每更新耗时与关系/注意力/模型耗时以原成本门禁、训练日志和阶段资源文件为准；缺失数值标NA。Euler50不等于推理成本不变；gradient checkpointing内部重算需与顶层forward分别解释。

## 四个决策问题

1. 总体有效性，SURFACE_XY vs H_CONT：NA。主比较原值与配对差见上表。
2. 有限表面机制，SURFACE_XY vs MOMENT_XY：NA；MOMENT_XY vs H_CONT：NA。同容量对照未被续训对照代替。
3. 条件能力：以逐曲线绝对斜率误差、逐beta列选择及两噪声分别变化判断。仅宽度响应改善不能宣布控制解决；门禁阻塞时均为NA。
4. 质量与成本：同时读取反向表面误差、尾部、接口、相交、小角面、留出及真实延迟。生成成功、MSE下降、条件选择改善和干净网格是不同结论。

**下一步判断：**数值或成本门禁未通过，正式质量比较均为NA。先处理本次实测阻塞；不以未运行结果判断有限表面或质心矩方法优劣。

本轮固定预算完成或门禁失败即停止，不调λ/带宽/层数/求积点，不改loss/学习率，不扩数据，不改默认模型。水密NOT_RUN；置换数值检查不是SE(3)等变证明；C硬保持是执行合同。


## Measured cost summary

|Checkpoint|Actual samples|Cache hits|Timed samples|Mean sample seconds|Median sample seconds|Train seconds/update|Train peak allocated bytes|Train peak reserved bytes|
|---|---|---|---|---|---|---|---|---|
|START|0|138|0|NA|NA|NA|NA|NA|
|H_CONT|0|0|0|NA|NA|NA|NA|NA|
|MOMENT_XY|0|0|0|NA|NA|NA|NA|NA|
|SURFACE_XY|0|0|0|NA|NA|NA|NA|NA|

Historical cache hits have no new sampling latency. Training phase wall time includes OT/checkpoint I/O; it is not isolated network time.

## Exact control curve counts

|Checkpoint|Split|Candidate|Noise|Valid/registered|All three targets strictly correct|One input strictly wins all targets|
|---|---|---|---|---|---|---|
|START|train|MODEL|all|8/8|1|1|
|START|train|MODEL|0|4/4|1|0|
|START|train|MODEL|1|4/4|0|1|
|START|validation|MODEL|all|8/8|0|8|
|START|validation|MODEL|0|4/4|0|4|
|START|validation|MODEL|1|4/4|0|4|
|H_CONT|train|MODEL|all|0/8|NA|NA|
|H_CONT|train|MODEL|0|0/4|NA|NA|
|H_CONT|train|MODEL|1|0/4|NA|NA|
|H_CONT|validation|MODEL|all|0/8|NA|NA|
|H_CONT|validation|MODEL|0|0/4|NA|NA|
|H_CONT|validation|MODEL|1|0/4|NA|NA|
|MOMENT_XY|train|MODEL|all|0/8|NA|NA|
|MOMENT_XY|train|MODEL|0|0/4|NA|NA|
|MOMENT_XY|train|MODEL|1|0/4|NA|NA|
|MOMENT_XY|validation|MODEL|all|0/8|NA|NA|
|MOMENT_XY|validation|MODEL|0|0/4|NA|NA|
|MOMENT_XY|validation|MODEL|1|0/4|NA|NA|
|SURFACE_XY|train|MODEL|all|0/8|NA|NA|
|SURFACE_XY|train|MODEL|0|0/4|NA|NA|
|SURFACE_XY|train|MODEL|1|0/4|NA|NA|
|SURFACE_XY|validation|MODEL|all|0/8|NA|NA|
|SURFACE_XY|validation|MODEL|0|0/4|NA|NA|
|SURFACE_XY|validation|MODEL|1|0/4|NA|NA|

## 实际门禁失败证据

本次正式 optimizer 更新为0，三臂均未训练。192点与12288点的差异在四个真实训练任务均未达到冻结阈值；ordinary_40的参考子集也未达到0.001收敛要求。因此这些数值不是已认证的真实积分误差界。其余三个参考子集通过，不等于覆盖了全空间。

|Fixture|N/K|mu L2 max|Covariance max|Mass max|Reference converged|
|---|---:|---:|---:|---:|---|
|ordinary_20|195/39|0.10827779|0.00778421|0.06803465|True|
|maximum_20|245/49|0.07164192|0.00223280|0.03266055|True|
|ordinary_40|206/82|0.10853565|0.01376171|0.09744185|False|
|maximum_40|245/98|0.07509304|0.00261257|0.04840983|True|

![按时间和任务的求积差异](figures/quadrature_gate.png)

逐query/known的8576行差异、96行参考收敛差异、原始冻结C和查询均已保留。任何求积点数、阈值、带宽、自然N/K均未调整。真实模型预检只用于接线与工程成本核验，不能覆盖或推翻这次准确度失败。

CPU隔离小模块测试有1次SGD测试更新和6次小模块反向，单独计账；它们不是Chair训练更新。关系单测另有2次autograd检查。正式训练、完整新采样、I/P交叉前向和真实SURFACE轨迹诊断均未执行。

下一步优先检验：固定窄带宽下，较长三角边上的离散节点是否不足以解析核权重在边/角附近的集中。先对本次冻结失败输入建立收敛的积分参考，再检验保持同一关系定义的数值实现能否同时满足准确度与成本门槛；本轮不执行该后续，不追加点数扫描或训练。有限表面与质心矩的质量比较当前均为NA，尚不能据此决定采用其中一个。

完整失败定位见 [NUMERICAL_FAILURE_ANALYSIS.md](NUMERICAL_FAILURE_ANALYSIS.md)；源码及原评分代码见 [source_code/source_hashes.json](source_code/source_hashes.json)。
