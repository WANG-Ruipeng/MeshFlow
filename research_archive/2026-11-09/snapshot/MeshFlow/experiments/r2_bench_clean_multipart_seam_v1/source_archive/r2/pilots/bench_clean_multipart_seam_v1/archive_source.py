"""Archive actual small implementation files and a diff; never copies weights."""
from __future__ import annotations
import argparse
import difflib
import json
import shutil
from pathlib import Path
from .common import ROOT, OUT, read, sha, resolve, atomic_write


def run():
    destination = OUT / "source_archive"
    destination.mkdir(parents=True, exist_ok=True)
    paths = set((ROOT / "r2/pilots/bench_clean_multipart_seam_v1").glob("*.py"))
    paths.add(ROOT / "r2/pilots/bench_clean_multipart_seam_v1/README.md")
    for registration, key in (
        ("training_registration.json", "training_code_hashes"),
        ("historical_registration.json", "code"),
        ("historical_execution_revision_01.json", "code"),
    ):
        p = OUT / registration
        if p.exists():
            paths.update(resolve(x) for x in read(p).get(key, {}))
    for module in (
        "r2/pilots/bench_parts_convergence_v1/postprocess.py",
        "r2/pilots/bench_control_transfer_v1/sample.py",
        "r2/pilots/bench_control_transfer_v1/postprocess.py",
        "r2/pilots/seam_slot_c200_v1/postprocess.py",
        "r2/pilots/chair_multipart_seam_v1/comparison.py",
        "native_t1/docs/control_metrics.md",
    ):
        paths.add(ROOT / module)
    records = []
    diff = []
    for path in sorted(paths):
        try:
            relative = path.resolve().relative_to(ROOT.resolve())
        except ValueError:
            # Some retained scorers live in the adjacent MeshFlow-Control tree.
            # Archive their actual sources under an explicitly separate prefix.
            relative = Path("external") / path.resolve().relative_to(ROOT.resolve().parent)
        if path.suffix.lower() not in (".py", ".md", ".json"):
            raise ValueError("Only small source/doc assets are archived")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha(target) == sha(path), "Do not overwrite an existing source archive"
        else:
            shutil.copyfile(path, target)
        records.append(dict(path=str(path), archive=str(target), sha256=sha(path), bytes=path.stat().st_size))
        if str(relative).replace("\\", "/").startswith("r2/pilots/bench_clean_multipart_seam_v1/"):
            content = path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
            diff.extend(difflib.unified_diff([], content, fromfile="/dev/null",
                                            tofile=str(relative).replace("\\", "/")))
    diffpath = OUT / "code.diff"
    text = "".join(diff)
    if diffpath.exists():
        assert diffpath.read_text(encoding="utf-8") == text
    else:
        diffpath.write_text(text, encoding="utf-8")
    result = dict(status="COMPLETE", source_files=records, diff_path=str(diffpath),
                  diff_sha256=sha(diffpath), weights_copied=0, model_calls=0,
                  note="New isolated pilot files vs /dev/null; reused historical source archived verbatim. No unrelated working-tree changes.")
    atomic_write(OUT / "source_hashes.json", result)
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = run()
    print(json.dumps(dict(status=result["status"], files=len(result["source_files"])), ensure_ascii=False))
