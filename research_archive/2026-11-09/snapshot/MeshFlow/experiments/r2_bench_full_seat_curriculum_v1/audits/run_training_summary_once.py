"""Explicit once-only CPU invocation receipt for the authorized final summary."""
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    audit = read(OUT / "audits/final_training_9000.json")
    if audit["status"] != "PASS":
        raise ValueError("Independent final ledger audit must pass first")
    for stage in ("NATIVE", "SEAM"):
        status = read(OUT / "training" / stage / "status.json")
        workflow = read(OUT / "workflow" / stage.lower() / "status.json")
        final = read(OUT / "training" / stage / "attempts" / status["attempt"] / "final_status.json")
        if status["status"] != "COMPLETE" or workflow["status"] != "COMPLETE" or final != status:
            raise ValueError("Final state/workflow not sealed")
    receipt = OUT / "audits/summarize_training_final_execution.json"
    if receipt.exists() or (OUT / "results/training_summary.json").exists():
        raise ValueError("Final summary invocation/output already exists; no automatic second execution")
    command = [sys.executable, "-B", "-m", "r2.pilots.bench_full_seat_curriculum_v1.summarize_training", "--final"]
    start = time.perf_counter()
    result = dict(schema="BENCH_FULL_SEAT_TRAINING_SUMMARY_ONCE_V1", status="RUNNING", command=command,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getpid(),
                  final_invocation_count=1, model_calls=0, scoring_calls=0, source_archive=False)
    with receipt.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    stdout = OUT / "audits/summarize_training_final_stdout.json"
    stderr = OUT / "audits/summarize_training_final_stderr.log"
    with stdout.open("xb") as out, stderr.open("xb") as err:
        completed = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err)
    result.update(returncode=completed.returncode, seconds=time.perf_counter() - start,
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  stdout=str(stdout.relative_to(ROOT)), stderr=str(stderr.relative_to(ROOT)),
                  status="COMPLETE" if completed.returncode == 0 else "FAILED_NO_AUTOMATIC_RETRY")
    validation_error = None
    if completed.returncode == 0:
        try:
            summary_path = OUT / "results/training_summary.json"
            summary = read(summary_path)
            if summary["status"] != "COMPLETE" or any(summary["own_activity"].values()):
                raise ValueError("Final summary status or zero-call contract differs")
            verified = {}
            for name, expected in summary["artifacts"].items():
                p = OUT / "results" / name
                actual = hashlib.sha256(p.read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError("Summary artifact identity mismatch: " + name)
                verified[name] = actual
            result.update(summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                          verified_artifacts=verified, scientific_budget_changed=False)
        except (ValueError, KeyError, OSError) as error:
            validation_error = repr(error)
            result.update(status="FAILED_VALIDATION_NO_AUTOMATIC_RETRY", error=validation_error)
    temporary = receipt.with_suffix(".partial")
    with temporary.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(temporary, receipt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if completed.returncode:
        sys.exit(completed.returncode)
    if validation_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
