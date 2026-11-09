"""Package the completed CPU-only gate evidence, excluding model/data assets."""
import json
from pathlib import Path
import zipfile
from . import run as A

C = A.core


def run():
    exp = A.DEFAULT
    done = C.read(exp / "COMPLETE.json")
    assert done["status"] == "COMPLETE_GATE_STOP_BEFORE_GPU"
    assert C.sha(exp / "REPORT.md") == done["report_sha256"]
    sources = {}
    for path in exp.glob("*.json"):
        if path.name != "package_receipt.json":
            sources[path.relative_to(exp).as_posix()] = path
    for path in exp.glob("*.md"):
        sources[path.relative_to(exp).as_posix()] = path
    for folder in ("blind_review", "code_provenance", "gate_verification"):
        root = exp / folder
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    assert path.suffix.lower() in {".png", ".json", ".jsonl", ".csv", ".md", ".diff", ".txt"}
                    sources[path.relative_to(exp).as_posix()] = path
    for root in (A.ROOT / "r2/pilots/bench_shift_relation_crossover_v1", Path(__file__).parent):
        for path in root.glob("*.py"):
            sources["source/" + path.relative_to(A.ROOT).as_posix()] = path
    inventory = {name:dict(bytes=p.stat().st_size,sha256=C.sha(p)) for name,p in sorted(sources.items())}
    target = exp / "BENCH_SHIFT_RELATION_GATE_REVIEW.zip"
    if target.exists():
        old = C.read(exp / "package_receipt.json")
        assert old["zip_sha256"] == C.sha(target) and old["inventory"] == inventory
        return old
    with zipfile.ZipFile(target,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(sources.items()):
            z.write(path,name)
        z.writestr("PACKAGE_MANIFEST.json",json.dumps(inventory,ensure_ascii=False,indent=2))
    assert target.stat().st_size < 10_000_000
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(inventory)+1
        import hashlib
        for name,item in inventory.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == item["sha256"]
    receipt = dict(status="PASS",zip_path=str(target),bytes=target.stat().st_size,
        zip_sha256=C.sha(target),inventory=inventory,model_weights_included=False,
        new_model_calls=0,new_score_calls=0,actual_review_pairs=32,actual_review_cells=64)
    C.write_json(exp / "package_receipt.json",receipt)
    return {k:v for k,v in receipt.items() if k!="inventory"}


if __name__ == "__main__":
    print(json.dumps(run(),ensure_ascii=False,indent=2))
