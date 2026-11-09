"""Post-training/pre-evaluation ledger audit; no collector or model imports."""
from __future__ import annotations

import json
from .run import DEFAULT, training_data, write_csv, write_json, sha, resolve


def main():
    summary, windows, parents, conditions, inputs = training_data(DEFAULT)
    dest = DEFAULT / "analysis_notes"
    table = dest / "training_windows_pre_eval.csv"
    write_csv(table, windows)
    micro_table = dest / "first_identity_micro_pre_eval.csv"
    write_csv(micro_table, summary["first_identity_micro_audit"]["rows"])
    for path, digest in inputs.items():
        if sha(resolve(path)) != digest:
            raise ValueError("Source changed during ledger audit: " + path)
    result = dict(status=("PASS_TRAINING_COMPLETE_EVALUATION_PENDING" if summary["science_audit_status"] == "PASS"
                          else "FAIL_IDENTITY_MICRO_CHECK_EVALUATION_PENDING"),
                  schema="BENCH_JOINT_SHIFT_PRE_EVAL_TRAINING_AUDIT_V1",
                  training=summary, parent_exposures=parents, condition_exposures=conditions,
                  input_hashes=inputs, training_windows_path=str(table),
                  training_windows_sha256=sha(table),
                  analysis_script_sha256=sha(resolve(__file__).with_name("run.py")),
                  stream_audit_script_sha256=sha(resolve(__file__).with_name("stream_audit.py")),
                  first_identity_micro_path=str(micro_table), first_identity_micro_sha256=sha(micro_table),
                  audit_script_sha256=sha(__file__),
                  model_calls=0, scorer_calls=0, collector_calls=0,
                  frozen_training_modified=False,
                  scope="Only completed training logs/call ledger/schedule/exposure; no evaluation result claimed")
    write_json(dest / "training_audit_pre_eval.json", result)
    print(json.dumps(dict(status=result["status"], training=summary,
                         output=str(dest / "training_audit_pre_eval.json")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
