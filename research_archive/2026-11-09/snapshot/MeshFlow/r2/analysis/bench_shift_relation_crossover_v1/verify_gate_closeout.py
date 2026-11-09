"""Read-only protected-asset verification after the pre-GPU gate stop."""
import json
from r2.pilots.bench_shift_relation_crossover_v1.eval_plan import plan, verify
from r2.pilots.bench_shift_relation_crossover_v1.common import OUT, read, sha, write_new


def run():
    p = plan()
    verify(p["protected"])
    verify(p["source_hashes"])
    done = read(OUT / "COMPLETE.json")
    assert done["candidate_sampled"] is False
    assert sha(OUT / "REPORT.md") == done["report_sha256"]
    result = dict(status="PASS", protected_files=len(p["protected"]),
        source_files=len(p["source_hashes"]), new_model_calls=0,
        source_assets_unchanged=True, completion_sha256=sha(OUT / "COMPLETE.json"),
        engineering_note="A preceding read-only python -c audit command failed on Windows/WSL quotation (NameError before hash verification). This file-based CPU audit replaces that command; no model/scorer/experiment attempt or scientific replay occurred.")
    write_new(OUT / "gate_verification/final_source_audit.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
