"""Verify the published evidence without assets, models, geometry scans or Git.

The optional isolated import checks the archived implementation, not a wrapper
that silently reaches back to an untracked local repository. It does not claim
that frozen experimental assets have been distributed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Archive path escapes owned directory: " + relative)
    return path


def run(check_import=False):
    manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8-sig"))
    failures = []
    for item in manifest["source_files"]:
        path = checked(item["archive_relative_path"])
        if not path.is_file():
            failures.append(dict(path=item["archive_relative_path"], status="MISSING_SOURCE"))
        elif digest(path) != item["sha256"]:
            failures.append(dict(path=item["archive_relative_path"], status="HASH_MISMATCH"))
    evidence = ROOT / "evidence_manifest.json"
    evidence_count = 0
    if evidence.exists():
        for item in json.loads(evidence.read_text(encoding="utf-8-sig"))["files"]:
            evidence_count += 1
            path = checked(item["archive_relative_path"])
            if not path.is_file() or digest(path) != item["sha256"]:
                failures.append(dict(path=item["archive_relative_path"], status="MISSING_OR_CHANGED_EVIDENCE"))
    zipped_records = 0
    ledger_index = ROOT / 'evidence/per_cell_ledger_manifest.json'
    if ledger_index.exists():
        entries = json.loads(ledger_index.read_text(encoding='utf-8'))['files']
        with zipfile.ZipFile(ROOT/'evidence/per_cell_ledgers.zip') as z:
            if set(z.namelist()) != {r['path'] for r in entries}:
                failures.append(dict(status='ZIP_ENTRY_SET_MISMATCH'))
            for r in entries:
                data=z.read(r['path']); zipped_records += 1
                if len(data)!=r['bytes'] or hashlib.sha256(data).hexdigest()!=r['sha256']:
                    failures.append(dict(path=r['path'],status='ZIP_ENTRY_HASH_MISMATCH'))
    if failures:
        return dict(status="FAILED", failures=failures)
    result = dict(status="PASS_SOURCE_AND_AVAILABLE_EVIDENCE_HASHES",
                  source_files=len(manifest["source_files"]), evidence_files=evidence_count,
                  verified_zip_records=zipped_records,
                  frozen_assets="NOT_DISTRIBUTED_REPLAY_BLOCKED_UNTIL_PROVIDED",
                  model_calls=0, geometry_scans=0)
    if check_import:
        # -I suppresses both cwd and user PYTHONPATH; only this archive is added.
        # No function or scientific definition is rebound.
        code = """
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from r2.tools.hinge_opposite_v1 import engine
loaded = {name: str(Path(mod.__file__).resolve())
          for name, mod in sys.modules.items()
          if name.startswith('r2') and getattr(mod, '__file__', None)}
assert loaded and all(Path(path).is_relative_to(root) for path in loaded.values()), loaded
assert 'torch' not in sys.modules
print(json.dumps(dict(status='PASS_ISOLATED_ARCHIVE_ENGINE_IMPORT',
                     version=engine.VERSION, modules=loaded, model_calls=0,
                     geometry_calls=0, asset_reads=0)))
"""
        with tempfile.TemporaryDirectory(prefix="hinge_archive_import_") as tmp:
            proc = subprocess.run([sys.executable, "-B", "-I", "-c", code,
                                   str(ROOT / "source/repository")],
                                  cwd=tmp, text=True, capture_output=True)
        if proc.returncode:
            result.update(status="BLOCKED_IMPORT_ENVIRONMENT", stderr=proc.stderr)
        else:
            result["isolated_import"] = json.loads(proc.stdout)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-engine-import", action="store_true")
    args = parser.parse_args()
    result = run(args.check_engine_import)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"].startswith("PASS") else 2)
