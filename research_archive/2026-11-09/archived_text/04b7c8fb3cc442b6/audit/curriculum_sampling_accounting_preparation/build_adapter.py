from pathlib import Path
root=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
original=root/"audit/sampling_accounting/audit_sampling.py"
text=original.read_text()
body=text[text.index("def main():"):]
old='''    if any(p.name!="audit_sampling.py" for p in DEST.iterdir()):
        raise FileExistsError("Preserve prior sampling accounting output")
    phase=RUN/"phases/main"'''
new='''    phase=RUN/"phases/course"
    missing=readiness(phase)
    if missing:
        print(json.dumps(dict(status="PENDING",missing=missing,model_operations=0)),flush=True)
        return 2
    if DEST.exists():
        raise FileExistsError("Preserve prior curriculum sampling accounting output")
    DEST.mkdir(parents=True)'''
assert old in body
body=body.replace(old,new)
body=body.replace('    registration=base.read_json(registration_path)\n','''    registration=base.read_json(registration_path)
    if (registration.get("kind") != "course" or registration.get("phase") != "course"
        or tuple(registration.get("labels",())) != COURSES
        or registration.get("expected_outputs") != 128
        or registration.get("registered_per_model") != 64
        or registration.get("nfe_per_output") != 50
        or registration.get("max_cell_attempts") != 1):
        raise ValueError("Frozen course phase registration differs")
''')
body=body.replace('    if sha(request_path)!=registration["requests_sha256"]:', '''    if len(requests)!=64 or len({q["request_id"] for q in requests})!=64:
        failures.append(dict(kind="64 unique requests required"))
    if sha(OLD_AUDITOR)!=OLD_AUDITOR_SHA:
        failures.append(dict(kind="original pure-ledger source hash"))
    record(OLD_AUDITOR,"reused_main_ledger_source")
    for label in COURSES:
        record(phase/"sampling"/label/"completed.json","sampling_worker_completion")
    if sha(request_path)!=registration["requests_sha256"]:''')
body=body.replace('phase="main",checkpoint_sha256=', 'phase="course",checkpoint_sha256=')
body=body.replace('len(rows)!=384 or len(expected)!=384','len(rows)!=128 or len(expected)!=128')
body=body.replace('actual=len(rows),expected=384','actual=len(rows),expected=128')
body=body.replace('new!=312 or reused!=72 or totals["sampling_forward_returned"]!=15600','new!=128 or reused!=0 or totals["sampling_forward_returned"]!=6400')
body=body.replace('expected main execution budget differs','expected course execution budget differs')
body=body.replace('schema="recipe_factorial_sampling_accounting_v1"','schema="recipe_curriculum_sampling_accounting_v1"')
body=body.replace('Completed main execution ledger','Completed course execution ledger')
body=body.replace('expected_forward_budget=15600','expected_forward_budget=6400')
body=body.replace('totals["sampling_forward_returned"]==15600','totals["sampling_forward_returned"]==6400')
body=body.replace('# Main sampling accounting','# Curriculum sampling accounting')
body=body.replace('All 384 registered outputs','All 128 registered outputs')
body=body.replace('The expected15600','The expected6400')
body=body.replace('glob("*/*/receipt.json")','rglob("receipt.json")')
body=body.replace('glob("*/*/calls.jsonl")','rglob("calls.jsonl")')
body=body.replace('    failure_markers=list((phase/"sampling").glob("*/*failure.json"))','    failure_markers=list((phase/"sampling").rglob("*failure.json"))')
body=body.replace('    retry_markers=list((phase/"sampling").glob("*/serial_retry_dispatch.json"))','    retry_markers=list((phase/"sampling").rglob("serial_retry_dispatch.json"))')
header='''"""Read-only course sampling accounting; reuses the audited main ledger parser."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time

REPO=Path("/mnt/e/MeshFlow-Control")
RUN=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
DEST=RUN/"audit/curriculum_sampling_accounting"
COURSES=("C20_40_MIX","C40_20_MIX")
OLD_AUDITOR=RUN/"audit/sampling_accounting/audit_sampling.py"
OLD_AUDITOR_SHA="c3e973ed1c42fa7820ad76d34e8b58dabfc66a87ff73983d3ff1f128736ce7e7"
_spec=importlib.util.spec_from_file_location("_completed_main_sampling_ledger",OLD_AUDITOR)
_prior=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_prior)
sha,csv_write,ledger,base=_prior.sha,_prior.csv_write,_prior.ledger,_prior.base

def readiness(phase):
    missing=[]
    for label in COURSES:
        done=phase/"sampling"/label/"completed.json"
        if not done.is_file():
            missing.append(str(done));continue
        if base.read_json(done).get("status")!="COMPLETE":
            missing.append(str(done)+": not COMPLETE")
    if len(list((phase/"outputs").glob("*/*/receipt.json")))!=128:
        missing.append("128 immutable output receipt files not yet present")
    return missing

'''
dest=root/"audit/curriculum_sampling_accounting_preparation"
dest.mkdir(parents=True,exist_ok=True)
(dest/"audit_sampling.py").write_text(header+body)
print(dest/"audit_sampling.py")
