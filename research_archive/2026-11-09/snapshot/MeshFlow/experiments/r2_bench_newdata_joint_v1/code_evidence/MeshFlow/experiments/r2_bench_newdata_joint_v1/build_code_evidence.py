"""Snapshot actual experiment code and its static local dependencies; no imports/models.

This is an evidence copier, not an execution environment or a new scientific run.
It refuses to overwrite its destination and verifies producer-frozen source hashes.
"""
from __future__ import annotations
import argparse
import ast
import collections
import datetime
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DEST = OUT / "code_evidence"
EXTERNAL = ROOT.parent / "MeshFlow-Control"
PACKAGE = ROOT / "r2/pilots/bench_newdata_joint_v1"
HISTORY = ROOT / "r2/pilots/bench_relation_training_v1"


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def resolve(text):
    text = str(text).replace("\\", "/")
    if text.startswith("E:/"):
        text = "/mnt/e/" + text[3:] if str(ROOT).startswith("/mnt/") else text
    elif text.startswith("/mnt/e/") and not str(ROOT).startswith("/mnt/"):
        text = "E:/" + text[7:]
    p = Path(text)
    return p if p.is_absolute() else ROOT / p


def relative(p):
    if p.is_relative_to(ROOT):
        return Path("MeshFlow") / p.relative_to(ROOT)
    if p.is_relative_to(EXTERNAL):
        return Path("external/MeshFlow-Control") / p.relative_to(EXTERNAL)
    raise ValueError("Source outside declared readonly roots: " + str(p))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def location(module, preferred=None):
    parts = module.split(".") if module else []
    roots = [ROOT, ROOT / "r2/src", EXTERNAL / "src"]
    if preferred is not None:
        roots = [preferred] + [r for r in roots if r != preferred]
    for base in roots:
        target = base.joinpath(*parts)
        for p in (target.with_suffix(".py"), target / "__init__.py"):
            if p.is_file():
                return p
    return None


def module_context(p):
    for base in (ROOT / "r2/src", EXTERNAL / "src", ROOT):
        if p.is_relative_to(base):
            names = list(p.relative_to(base).with_suffix("").parts)
            if names[-1] == "__init__":
                names.pop()
            else:
                names.pop()
            return names, base
    return [], None


def direct_imports(p, payload):
    """Conservative AST closure; no claim that every imported branch executed."""
    tree = ast.parse(payload.decode("utf-8-sig"), filename=str(p))
    package, base = module_context(p)
    found = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [(a.name, None) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            prefix = package[:len(package) - node.level + 1] if node.level else []
            mod = ".".join(prefix + ([node.module] if node.module else []))
            names = [(mod, base if node.level else None)]
            names += [(mod + ("." if mod else "") + a.name,
                       base if node.level else None) for a in node.names if a.name != "*"]
        for name, preferred in names:
            candidate = location(name, preferred)
            if candidate is not None and candidate != p:
                found.add(candidate)
    return found


def build():
    if DEST.exists():
        raise RuntimeError("Evidence directory exists; preserve it and use verify, never overwrite.")
    roles = collections.defaultdict(set)
    frozen = collections.defaultdict(list)
    origins = {}
    registry_paths = [OUT / "training_registration.json",
                      OUT / "evaluation/score_registration.json",
                      *sorted((OUT / "evaluation/inference_registrations").glob("*.json"))]
    for p in registry_paths:
        data = p.read_bytes()
        obj = json.loads(data)
        origins[str(p)] = sha_bytes(data)
        for key in ("training_code_hashes", "code", "source_hashes"):
            for source, digest in obj.get(key, {}).items():
                target = resolve(source)
                if target.suffix != ".py":
                    raise ValueError("Unexpected non-source in code map: " + str(target))
                roles[target].add("PRODUCER_FROZEN:" + p.name + ":" + key)
                frozen[target].append(dict(registration=str(p), field=key, sha256=digest))
    for p in PACKAGE.glob("*.py"):
        roles[p].add("CURRENT_EXPERIMENT_ENTRYPOINT_OR_HELPER")
    roles[Path(__file__).resolve()].add("EVIDENCE_BUILDER_NO_PROJECT_IMPORT")
    for p in (ROOT / "r2/src/meshflow_control").rglob("*.py"):
        roles[p].add("SUPPORTING_CORE_SOURCE_NOT_ALL_BRANCHES_EXECUTED")
    extras = ["r2/pilots/bench_adaptation_strength_v1/collect_v2.py",
              "r2/pilots/chair_multipart_seam_v1/gallery.py",
              "r2/pilots/regular_patch_control_v1/depth_gallery.py",
              "r2/tools/cref_layer_audit_v1/render.py"]
    for name in extras:
        roles[ROOT / name].add("DIRECT_COLLECTOR_OR_RENDERER_SOURCE")
    comparisons = {}
    for p in sorted(PACKAGE.glob("*.py")):
        historical = HISTORY / p.name
        if p.name == "sample_v2.py":
            historical = PACKAGE / "sample.py"
            title = "ENGINEERING FIX: failed pre-forward v1 -> successful v2; no scientific definition change"
        elif historical.is_file():
            title = "HISTORICAL COUNTERPART: RELATION_V1 -> NEWDATA_JOINT_V1"
        elif p.name == "eval_runtime.py":
            historical = HISTORY / "eval_plan.py"
            title = "API REFACTOR: historical runtime helpers -> separated new runtime; not a verbatim file rename"
        else:
            historical = None
            title = "NEW FILE: no asserted historical file equivalence"
        if historical is not None:
            roles[historical].add("DIFF_HISTORICAL_COMPARISON_NOT_NEW_EXECUTION")
        comparisons[p] = (historical, title)
    queue = collections.deque(sorted(roles))
    payloads = {}
    imports = {}
    while queue:
        p = queue.popleft()
        if p in payloads:
            continue
        if not p.is_file() or p.suffix != ".py":
            raise ValueError("Missing source " + str(p))
        relative(p)
        raw = p.read_bytes()
        for item in frozen[p]:
            if sha_bytes(raw) != item["sha256"]:
                raise ValueError("Producer frozen code changed: " + str(p))
        payloads[p] = raw
        dependencies = direct_imports(p, raw)
        imports[p] = dependencies
        for dep in sorted(dependencies):
            roles[dep].add("STATIC_LOCAL_IMPORT_DEPENDENCY")
            queue.append(dep)
    # Verify one immutable byte snapshot before writing any archive files.
    for p, raw in payloads.items():
        if p.read_bytes() != raw:
            raise RuntimeError("Source changed during archive preparation: " + str(p))
    DEST.mkdir()
    records = []
    for p, raw in sorted(payloads.items()):
        dest = DEST / relative(p)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        records.append(dict(original_path=str(p), archive_path=relative(p).as_posix(),
                            sha256=sha_bytes(raw), bytes=len(raw), roles=sorted(roles[p]),
                            producer_frozen_in=frozen[p],
                            static_local_imports=[relative(d).as_posix() for d in sorted(imports[p])]))
    chunks = ["# NEWDATA_JOINT_V1 review diff, not an in-place apply-patch instruction.\n",
              "# Originals are read-only; each comparison is labelled by its actual historical role.\n",
              "# sample.py is retained failed v1 evidence; sample_v2.py is the actual successful sampler.\n\n"]
    comparison_records = []
    for p, (historical, title) in comparisons.items():
        old_lines = [] if historical is None else payloads[historical].decode("utf-8-sig").splitlines(True)
        new_lines = payloads[p].decode("utf-8-sig").splitlines(True)
        old_name = "/dev/null" if historical is None else relative(historical).as_posix()
        new_name = relative(p).as_posix()
        chunks.append("\n# " + title + "\n")
        delta = list(difflib.unified_diff(old_lines, new_lines, fromfile=old_name,
                                        tofile=new_name, n=3))
        chunks.extend(delta or ["# Byte text unchanged for this comparison.\n"])
        comparison_records.append(dict(new_file=new_name, old_file=old_name, comparison=title,
                                       changed=bool(delta)))
    diff = "".join(chunks).encode("utf-8")
    (DEST / "code.diff").write_bytes(diff)
    note = """# Actual code evidence: Bench new-data joint 1500

This archive contains source code only: no weights, optimizers, raw meshes, generated
images, or copied third-party Python installations. It is a review snapshot, not a
standalone runnable installation. MeshFlow files preserve repository-relative paths
under `MeshFlow/`; the original scorer from the sibling checkout is under
`external/MeshFlow-Control/` with that checkout's relative paths preserved.

`source_hashes.json` distinguishes code whose bytes are bound by actual producer
registrations from current helper snapshots, historical comparison files, and the
conservative AST-resolved local import closure. An imported branch or a supporting
core module being present does not certify that its functions executed. No module
was imported by this copier; it parsed source as text only. No source was edited.

`code.diff` compares new wrappers to actual RELATION_V1 counterpart files when one
exists; new standalone helpers are full additions. `eval_runtime.py` is explicitly
marked as an API refactor from the historical combined eval_plan. The failed
pre-forward `sample.py` is preserved separately; successful `sample_v2.py` has its
own explicit engineering delta. This diff is review evidence, not a patch to apply
blindly to the historical experiment. All legacy scoring and geometry formulas
are retained verbatim in their archived source files.

Run the top-level experiment `build_code_evidence.py verify` to verify archived
bytes and current source equality. This does not load models or score geometry.
Rendered pages and actual visual-review labels have separate receipts and are not
certified by this source snapshot. Later helper changes must receive a new version,
not silently replace this archive.
"""
    (DEST / "README.md").write_text(note, encoding="utf-8")
    dump(DEST / "source_hashes.json", dict(
        schema="BENCH_NEWDATA_CODE_EVIDENCE_V1", status="COMPLETE",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model_calls=0, scoring_calls=0, original_file_mutations=0,
        registrations=origins, source_files=records, comparisons=comparison_records,
        source_file_count=len(records), source_bytes=sum(r["bytes"] for r in records),
        files={"code.diff": sha_bytes(diff), "README.md": sha_bytes(note.encode("utf-8"))}))
    return verify()


def verify():
    manifest = json.loads((DEST / "source_hashes.json").read_text(encoding="utf-8"))
    for row in manifest["source_files"]:
        p = DEST / row["archive_path"]
        if sha_bytes(p.read_bytes()) != row["sha256"]:
            raise ValueError("Archive changed: " + str(p))
        if sha_bytes(resolve(row["original_path"]).read_bytes()) != row["sha256"]:
            raise ValueError("Original source changed since archive: " + row["original_path"])
    for name, digest in manifest["files"].items():
        if sha_bytes((DEST / name).read_bytes()) != digest:
            raise ValueError("Evidence sidecar changed: " + name)
    for name, digest in manifest["registrations"].items():
        if sha_bytes(resolve(name).read_bytes()) != digest:
            raise ValueError("Frozen registry changed: " + name)
    return dict(status="PASS", source_files=manifest["source_file_count"],
                source_bytes=manifest["source_bytes"], model_calls=0, scoring_calls=0,
                source_hashes_sha256=sha_bytes((DEST / "source_hashes.json").read_bytes()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    args = parser.parse_args()
    print(json.dumps(build() if args.mode == "build" else verify(), indent=2))
