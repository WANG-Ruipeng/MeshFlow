"""CPU-only delivery verification after the failed, immutable GPU gate."""
from pathlib import Path
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT = Path("/mnt/e/MeshFlow")
RUN = ROOT / "native_t1/runs/geometry_context_20260930_142000"
BASE = "2502d5f89c54706cc4700d3cdea4a12fd6792529"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write(name, value):
    (RUN/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")

before_gate = sha(RUN/"gates.json")
registration = json.loads((RUN/"registration.json").read_text(encoding="utf-8-sig"))
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
start = time.perf_counter()
command = [sys.executable,"-B","-m","unittest","discover","-s","native_t1/tests","-v"]
test = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
elapsed = time.perf_counter()-start
transcript = test.stdout+test.stderr
(RUN/"final_cpu_tests.log").write_text(transcript, encoding="utf-8")
match = re.search(r"Ran (\d+) tests in ([0-9.]+)s", transcript)
record = dict(status="PASS" if test.returncode==0 else "FAIL", exit_code=test.returncode,
              tests=int(match.group(1)) if match else None, seconds=elapsed,
              command=command, CUDA_VISIBLE_DEVICES="", complete_backbone_forwards=0,
              real_model_backwards=0, experiment_optimizer_updates=0, complete_rollouts=0,
              note="Includes CPU fixture/encoder autograd and synthetic OT checks; these are not real model training or generation.",
              log_sha256=sha(RUN/"final_cpu_tests.log"))
write("final_cpu_tests.json",record)
if test.returncode:
    raise RuntimeError("CPU suite failed; preserve transcript")

amendment_path = RUN/"post_stop_guard_amendment.json"
amendment = json.loads(amendment_path.read_text(encoding="utf-8-sig")) if amendment_path.exists() else {}
sources = {}
for name, expected in registration["execution_sources"].items():
    actual = sha(ROOT/name)
    guard_amendment = (name==amendment.get("path") and expected==amendment.get("before_sha256") and actual==amendment.get("after_sha256"))
    sources[name] = dict(expected=expected, actual=actual, unchanged=actual==expected, declared_post_stop_guard_amendment=guard_amendment)
assets={}
for label,path,key in (
    ("common_initialization",registration["initialization_path"],"initialization_file_sha256"),
    ("training_inputs",registration["inputs_path"],"inputs_sha256"),
    ("evaluation_inputs",str(RUN/"registration/evaluation_inputs.npz"),"evaluation_inputs_sha256")):
    actual=sha(path);expected=registration[key]
    assets[label]=dict(path=path,bytes=Path(path).stat().st_size,sha256=actual,expected_sha256=expected,unchanged=actual==expected)
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
protected=subprocess.run(["git","diff","--name-only",BASE,"--","models","configs","native_t1/data.py","native_t1/training.py","native_t1/sampling.py","native_t1/runtime.py","native_t1/baseline.py","native_t1/portable_checkpoint.py","native_t1/train_cli.py"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
gate_unchanged=sha(RUN/"gates.json")==before_gate
run_artifacts=dict(arm_directories=[name for name in ("T_base","T_geo","T_graph") if (RUN/name).exists()],
                   checkpoint_files=[str(p.relative_to(RUN)) for p in RUN.rglob("*.pt")],
                   generation_json_exists=(RUN/"generation.json").exists(),
                   raw_directory_exists=(RUN/"raw").exists())
integrity=dict(status="PASS" if all(v["unchanged"] or v["declared_post_stop_guard_amendment"] for v in sources.values()) and all(v["unchanged"] for v in assets.values()) and not protected and head==BASE and gate_unchanged else "FAIL",
               baseline_commit=BASE,HEAD=head,execution_sources=sources,assets=assets,post_stop_guard_amendment=amendment,
               protected_tracked_changes=protected.splitlines(),original_gate_file_unchanged=gate_unchanged,
               gate_sha256=before_gate,artifact_inventory=run_artifacts,
               no_commit_or_push=True,zero_additional_model_calls=True)
write("delivery_integrity.json",integrity)

names=[
"native_t1/model.py","native_t1/context_geometry.py","native_t1/geometry_checkpoint.py",
"native_t1/geometry_evaluation.py","native_t1/geometry_gates.py","native_t1/geometry_generation.py",
"native_t1/geometry_pilot.py","native_t1/geometry_results.py",
"native_t1/tests/test_context_geometry.py","native_t1/tests/test_geometry_checkpoint.py",
"native_t1/tests/test_geometry_evaluation.py","native_t1/tests/test_geometry_entry_guards.py",
"output/geometry_context_pilot_audit_20260930/audit_outputs.py"]
parts=[]; entries=[]
for name in names:
    path=ROOT/name
    old=subprocess.run(["git","show",BASE+":"+name],cwd=ROOT,capture_output=True)
    oldtext=old.stdout.decode("utf-8-sig") if old.returncode==0 else ""
    newtext=path.read_text(encoding="utf-8-sig")
    header="diff --git a/"+name+" b/"+name+"\n"
    if old.returncode: header+="new file mode 100644\n"
    delta="".join(difflib.unified_diff(oldtext.splitlines(keepends=True),newtext.splitlines(keepends=True),
                                      fromfile="a/"+name if old.returncode==0 else "/dev/null",tofile="b/"+name))
    if delta:
        parts.append(header+delta)
    entries.append(dict(path=name,bytes=path.stat().st_size,sha256=sha(path),
                        status="modified" if old.returncode==0 else "new"))
(RUN/"code.diff").write_text("".join(parts),encoding="utf-8")
write("code_manifest.json",dict(baseline_commit=BASE,files=entries,diff_sha256=sha(RUN/"code.diff"),
     scope="Only this task's implementation/tests plus separate local strict-audit entry; no weights, raw datasets, historical code or executable cache.",
     no_commit_or_push=True))
print(json.dumps(dict(tests=record,integrity_status=integrity["status"],diff_bytes=(RUN/"code.diff").stat().st_size),ensure_ascii=False))
if integrity["status"]!="PASS": raise RuntimeError("Delivery integrity failed")
