"""Read-only Windows/WSL Git line-ending reconciliation, no experiment calls."""
from pathlib import Path
import json, subprocess
ROOT=Path("/mnt/e/MeshFlow")
RUN=ROOT/"native_t1/runs/geometry_context_20260930_142000"
record_path=RUN/"delivery_integrity.json"
record=json.loads(record_path.read_text())
first=RUN/"delivery_integrity_attempt1.json"
if first.exists(): raise FileExistsError(first)
first.write_bytes(record_path.read_bytes())
names=record["protected_tracked_changes"]
checks=[]
for name in names:
    raw=(ROOT/name).read_bytes()
    blob=subprocess.check_output(["git","show",record["baseline_commit"]+":"+name],cwd=ROOT)
    checks.append(dict(path=name,working_copy_CRLF=raw.count(b"\r\n"),
                       canonical_content_identical=raw.replace(b"\r\n",b"\n")==blob.replace(b"\r\n",b"\n")))
protected=subprocess.check_output(["git","-c","core.autocrlf=true","diff","--name-only",record["baseline_commit"],
    "--","models","configs","native_t1/data.py","native_t1/training.py","native_t1/sampling.py",
    "native_t1/runtime.py","native_t1/baseline.py","native_t1/portable_checkpoint.py","native_t1/train_cli.py"],
    cwd=ROOT,text=True).strip().splitlines()
record["git_line_ending_check"]=dict(first_probe="WSL Git did not inherit Windows global core.autocrlf=true and reported CRLF-only differences",
    corrected_probe="git -c core.autocrlf=true diff; no repository/global config was changed",
    raw_files_not_modified=True,per_file_checks=checks,first_probe_record=str(first))
record["protected_tracked_changes"]=protected
record["status"]="PASS" if not protected and all(c["canonical_content_identical"] for c in checks) and all(
    s["unchanged"] or s["declared_post_stop_guard_amendment"] for s in record["execution_sources"].values()) and all(
    s["unchanged"] for s in record["assets"].values()) and record["original_gate_file_unchanged"] and record["HEAD"]==record["baseline_commit"] else "FAIL"
record_path.write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(json.dumps(dict(status=record["status"],CRLF_only_files=len(checks),remaining_protected_changes=protected,
                     additional_model_calls=0)))
if record["status"]!="PASS": raise RuntimeError("Protected source differs beyond line endings")
