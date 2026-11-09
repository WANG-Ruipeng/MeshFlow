"""Verify the published HINGE V2 archive without geometry or model execution.

The default check uses only the standard library. The optional isolated import
requires the documented CPU environment and exercises the archived dependency
closure, including the original scoring modules, without loading mesh assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checked(relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Expected a nonempty POSIX relative archive path")
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or ":" in relative:
        raise ValueError("Archive path escapes its directory: " + relative)
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Archive path escapes its directory: " + relative)
    return path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_files(items, label, failures):
    seen = set()
    for item in items:
        relative = item["path"] if label == "support" else item["archive_relative_path"]
        if relative in seen:
            failures.append(dict(path=relative, status="DUPLICATE_MANIFEST_PATH", manifest=label))
            continue
        seen.add(relative)
        path = checked(relative)
        if not path.is_file():
            failures.append(dict(path=relative, status="MISSING_FILE", manifest=label))
        elif digest(path) != item["sha256"]:
            failures.append(dict(path=relative, status="HASH_MISMATCH", manifest=label))
        elif "bytes" in item and path.stat().st_size != item["bytes"]:
            failures.append(dict(path=relative, status="SIZE_MISMATCH", manifest=label))
    return seen


def isolated_import():
    # -I excludes cwd and PYTHONPATH. This rebases a filesystem setting only;
    # no scientific function, predicate or numerical threshold is replaced.
    code = r'''
import hashlib, json, os, sys
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
archive = Path(sys.argv[1]).resolve()
root = archive / 'source'
sys.path.insert(0, str(root / 'repository'))
from r2.tools.hinge_v2_cpu_v1 import alpha, holes, scoring
Q = alpha.Q
adapter = Q._module('_partial_binding_original_math_adapter',
                    Q.ROOT / 'r2/pilots/user_condition_finetune_v1/evaluation.py')
adapter.LEGACY = root / 'legacy/src/meshflow_control/evaluation'
helpers = Q.helpers()
modules = {}
for name, mod in sys.modules.items():
    if not (name == 'r2' or name.startswith('r2.')
            or name.startswith('_partial_binding_')
            or name.startswith('_r2_user_pilot_original_evaluation')):
        continue
    filename = getattr(mod, '__file__', None)
    if filename:
        path = Path(filename).resolve()
        assert path.is_relative_to(root), (name, str(path))
        modules[name] = dict(path=path.relative_to(archive).as_posix(),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    else:
        for directory in getattr(mod, '__path__', ()):
            assert Path(directory).resolve().is_relative_to(root), (name, directory)
assert modules and 'torch' not in sys.modules
print(json.dumps(dict(status='PASS_ISOLATED_ARCHIVE_IMPORT',
    versions=dict(alpha=alpha.VERSION, holes=holes.VERSION, scoring=scoring.VERSION),
    modules=modules, original_scorer_modules=sorted(helpers),
    model_calls=0, geometry_calls=0, asset_reads=0)))
'''
    with tempfile.TemporaryDirectory(prefix="hinge_v2_archive_import_") as tmp:
        process = subprocess.run([sys.executable, "-B", "-I", "-c", code, str(ROOT)],
                                 cwd=tmp, text=True, capture_output=True)
    if process.returncode:
        return dict(status="BLOCKED_IMPORT_ENVIRONMENT", returncode=process.returncode,
                    stdout=process.stdout, stderr=process.stderr)
    return json.loads(process.stdout)


def run(check_import=False):
    failures = []
    source = read(ROOT / "source_manifest.json")["source_files"]
    evidence = read(ROOT / "evidence_manifest.json")["files"]
    if not source or not evidence:
        raise ValueError("Source and evidence manifests must both be nonempty")
    source_paths = verify_files(source, "source", failures)
    evidence_paths = verify_files(evidence, "evidence", failures)
    actual_source_paths = {path.relative_to(ROOT).as_posix()
                           for path in (ROOT / "source").rglob("*")
                           if path.is_file() and "__pycache__" not in path.parts}
    if actual_source_paths != source_paths:
        failures.append(dict(status="SOURCE_FILE_SET_MISMATCH",
                             unexpected=sorted(actual_source_paths - source_paths),
                             missing=sorted(source_paths - actual_source_paths)))
    required_sources = {
        "source/repository/r2/tools/hinge_v2_cpu_v1/alpha.py",
        "source/repository/r2/tools/hinge_v2_cpu_v1/holes.py",
        "source/repository/r2/tools/hinge_v2_cpu_v1/scoring.py",
        "source/legacy/src/meshflow_control/evaluation/geometry.py",
    }
    for path in sorted(required_sources - source_paths):
        failures.append(dict(path=path, status="REQUIRED_SOURCE_NOT_MANIFESTED"))
    support_count = 0
    support_manifest = ROOT / "support_manifest.json"
    if support_manifest.is_file():
        support = read(support_manifest)["files"]
        support_count = len(support)
        verify_files(support, "support", failures)
    ledger_relative = "evidence/per_cell_ledger_manifest.json"
    zip_relative = "evidence/per_cell_ledgers.zip"
    for path in (ledger_relative, zip_relative):
        if path not in evidence_paths:
            failures.append(dict(path=path, status="REQUIRED_LEDGER_NOT_MANIFESTED"))
    entries = read(checked(ledger_relative))["files"]
    count = 0
    indexed_names = [item["path"] for item in entries]
    if len(indexed_names) != len(set(indexed_names)):
        failures.append(dict(status="DUPLICATE_ZIP_MANIFEST_ENTRY"))
    with zipfile.ZipFile(checked(zip_relative)) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)):
            failures.append(dict(status="DUPLICATE_ZIP_MEMBER"))
        if set(names) != set(indexed_names):
            failures.append(dict(status="ZIP_ENTRY_SET_MISMATCH",
                                 missing=sorted(set(indexed_names) - set(names)),
                                 unexpected=sorted(set(names) - set(indexed_names))))
        for item in entries:
            # We never extract, but reject unsafe names for downstream readers.
            checked(item["path"])
            if item["path"] not in names:
                continue
            data = zipped.read(item["path"])
            count += 1
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                failures.append(dict(path=item["path"], status="ZIP_ENTRY_HASH_OR_SIZE_MISMATCH"))
    result = dict(status="FAILED" if failures else "PASS_ARCHIVE_HASHES",
                  source_files=len(source), evidence_files=len(evidence),
                  support_files=support_count, verified_zip_records=count,
                  frozen_mesh_assets="NOT_DISTRIBUTED_REPLAY_REQUIRES_EXPLICIT_LOCAL_INPUTS",
                  model_calls=0, geometry_calls=0, failures=failures)
    if check_import and not failures:
        imported = isolated_import()
        result["isolated_import"] = imported
        if not imported["status"].startswith("PASS"):
            result["status"] = imported["status"]
        else:
            declared = {item["archive_relative_path"]: item["sha256"] for item in source}
            for name, item in imported["modules"].items():
                if declared.get(item["path"]) != item["sha256"]:
                    failures.append(dict(module=name, path=item["path"], status="IMPORTED_SOURCE_NOT_MANIFESTED"))
            if failures:
                result["status"] = "FAILED"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-engine-import", "--import-engine", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.check_engine_import)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        result = dict(status="FAILED_ARCHIVE_FORMAT_OR_MISSING_FILE",
                      error_type=type(error).__name__, error=str(error),
                      model_calls=0, geometry_calls=0)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"].startswith("PASS") else 2)
