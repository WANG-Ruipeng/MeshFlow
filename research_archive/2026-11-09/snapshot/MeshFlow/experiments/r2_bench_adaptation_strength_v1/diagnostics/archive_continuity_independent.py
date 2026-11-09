"""Independent small-file continuity check; standard library, no models/scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
EXTERNAL = ROOT.parent / "MeshFlow-Control"
LIMIT = 8 << 20
LABELS = ("COMMON_S0", "OFFICIAL_S0", "A_S500", "B_S500", "C_S500",
          "A_S1500", "B_S1500", "C_S1500")
SOURCE_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml"}


def path_of(value):
    value = str(value).replace("\\", "/")
    for prefix in ("E:/MeshFlow/", "/mnt/e/MeshFlow/"):
        if value.startswith(prefix):
            return (ROOT / value[len(prefix):]).resolve()
    for prefix in ("E:/MeshFlow-Control/", "/mnt/e/MeshFlow-Control/"):
        if value.startswith(prefix):
            return (EXTERNAL / value[len(prefix):]).resolve()
    return Path(value).resolve()


def read(path):
    return json.loads(path_of(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-audit", required=True)
    parser.add_argument("--completion-audit", required=True)
    args = parser.parse_args()
    errors, files, checks = [], {}, []
    hash_cache = {}

    def check(ok, name, **facts):
        row = dict(check=name, passed=bool(ok), **facts)
        checks.append(row)
        if not ok:
            errors.append(row)

    def digest(path):
        path = path_of(path)
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".npz", ".npy", ".zip"}:
            raise ValueError("Refuse large/model/geometry artifact: " + str(path))
        before = path.stat()
        if before.st_size > LIMIT:
            raise ValueError("Outside small-file size limit: " + str(path))
        identity = (str(path), before.st_size, before.st_mtime_ns)
        if identity not in hash_cache:
            value = hashlib.sha256(path.read_bytes()).hexdigest()
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                raise ValueError("File changed while reading: " + str(path))
            hash_cache[identity] = value
        value = hash_cache[identity]
        files[str(path)] = value
        return value

    prior_path = path_of(args.prior_audit)
    completion_path = path_of(args.completion_audit)
    prior_path.relative_to(OUT / "closeout_audits_v2")
    completion_path.relative_to(OUT / "closeout_audits_v2")
    prior = read(prior_path)
    prior_hashes_path = prior_path.with_name("verified_hashes.json")
    previous = {str(path_of(p)): v for p, v in read(prior_hashes_path).items()}
    completed = read(completion_path)
    prior_sha, hashes_sha = digest(prior_path), digest(prior_hashes_path)
    completion_sha = digest(completion_path)
    check(
        prior["status"] == "FAIL" and not prior.get("missing")
        and len(prior.get("failures", [])) == 1
        and prior["failures"][0]["check"] == "AUDIT_EXECUTION"
        and "Refuse source archive outside workspace: /mnt/e/MeshFlow-Control/"
        in prior["failures"][0].get("error", "")
        and all(c["passed"] for c in prior["checks"]),
        "PRIOR_ONLY_KNOWN_ARCHIVE_FAILURE",
    )
    check(len(previous) == prior["verified_file_count"]
          and all(v["matches_registered"] for v in previous.values()),
          "PRIOR_VERIFIED_MAP_COMPLETE")
    check(completed["status"] == "PASS"
          and path_of(completed["prior_audit"]) == prior_path
          and completed["prior_audit_sha256"] == prior_sha
          and completed["prior_verified_hashes_sha256"] == hashes_sha,
          "COMPLETION_BINDS_PRIOR_AUDIT")
    check(completed["inherited_content_checks"] == len(prior["checks"])
          and completed["inherited_verified_file_count"] == len(previous),
          "COMPLETION_INHERITED_COUNTS")

    selected = set()
    for value in previous:
        path = Path(value)
        if ("registration" in path.name.lower()
                or path.name.lower() in {"completion.json", "complete.json"}
                or (path.is_relative_to(EXTERNAL) and path.suffix.lower() in SOURCE_SUFFIXES)):
            selected.add(path)

    registration_names = [
        "training_registration.json", "registration.json",
        "remaining_runtime_code_registration.json",
        "evaluation/score_registration.json", "gallery/manifest.json",
        "gallery/definition.json", "visual_review/protocol.json",
        "visual_review/assignments.json",
    ] + ["evaluation/inference_registrations/" + label + ".json" for label in LABELS]
    other_names = [
        "eval_manifest.json", "diagnostics/results_v2_independent_audit.json",
        "diagnostics/results_v2_independent_audit.initial.json",
        "diagnostics/collector_v2_revision.json", "results_v2/completion.json",
        "visual_review/completion.json", "result_analysis/completion.json",
        "training_summary/latest.json",
    ]
    selected.update(OUT / name for name in registration_names + other_names)
    selected.update(path_of(p) for p in completed["current_small_file_hashes"])
    script_snapshot_names = [
        ROOT / "r2/pilots/bench_adaptation_strength_v1/archive_completion_v2.py",
        ROOT / "r2/pilots/bench_adaptation_strength_v1/closeout_audit_v2.py",
        Path(__file__).resolve(),
    ]
    selected.update(script_snapshot_names)

    continuity, snapshots = [], []
    for path in sorted(selected):
        actual = digest(path)
        old = previous.get(str(path))
        if old is not None:
            passed = actual == old["sha256"] and path.stat().st_size == old["bytes"]
            check(passed, "PREVIOUS_SMALL_FILE_UNCHANGED", path=str(path))
            continuity.append(dict(path=str(path), sha256=actual, previous_sha256=old["sha256"],
                                   passed=passed))
        else:
            snapshots.append(dict(path=str(path), sha256=actual,
                                  verification="Current small-file snapshot, not prior-hash continuity"))
    for value, expected in completed["current_small_file_hashes"].items():
        check(digest(value) == expected, "COMPLETION_DEPENDENCY_HASH", path=str(path_of(value)))

    registered_source_paths = set()
    for name in registration_names:
        document = read(OUT / name)
        for key in ("training_code_hashes", "code", "source_hashes", "protected"):
            for value, expected in document.get(key, {}).items():
                source = path_of(value)
                if source.suffix.lower() not in SOURCE_SUFFIXES:
                    continue
                registered_source_paths.add(source)
                check(digest(source) == expected, "CURRENT_REGISTERED_SOURCE_HASH",
                      path=str(source), registration=name, field=key)
                if str(source) in previous:
                    check(digest(source) == previous[str(source)]["sha256"],
                          "REGISTERED_SOURCE_PRIOR_CONTINUITY", path=str(source))

    archive_root = path_of(completed["archive"]["source_archive"])
    archive_root.relative_to(OUT / "source_archive")
    manifest_path = archive_root / "manifest.json"
    manifest = read(manifest_path)
    digest(manifest_path)
    check(manifest["status"] == "PASS" and manifest["copied_weights"] == 0
          and len(manifest["files"]) == completed["archive"]["source_files"],
          "ARCHIVE_MANIFEST_COUNTS")
    manifest_source_paths = set()
    external_copies = []
    for row in manifest["files"]:
        source = path_of(row["source"])
        archived = path_of(row["archive_path"])
        archived.relative_to(archive_root)
        manifest_source_paths.add(source)
        # All archive entries are small source/registration artifacts.
        check(digest(source) == row["sha256"] == digest(archived),
              "ARCHIVE_COPY_CONTENT", source=str(source), copy=str(archived))
        if source.is_relative_to(EXTERNAL):
            old = previous.get(str(source))
            check(old is not None and old["matches_registered"]
                  and digest(source) == old["sha256"]
                  and source in registered_source_paths,
                  "EXTERNAL_SOURCE_EXPLICIT_PRIOR_AND_CURRENT_REGISTRATION", source=str(source))
            expected_destination = archive_root / "_external_sources/MeshFlow-Control" / source.relative_to(EXTERNAL)
            check(archived == expected_destination, "EXTERNAL_ARCHIVE_NAMESPACE", source=str(source))
            external_copies.append(str(source))
    expected_external = {str(Path(p)) for p in previous
                         if Path(p).is_relative_to(EXTERNAL) and Path(p).suffix.lower() in SOURCE_SUFFIXES}
    check(set(external_copies) == expected_external, "ALL_PREVIOUS_EXTERNAL_SOURCES_ARCHIVED",
          expected=len(expected_external), actual=len(external_copies))
    check(OUT / "diagnostics/results_v2_independent_audit.initial.json" in manifest_source_paths,
          "INITIAL_INDEPENDENT_RECEIPT_ARCHIVED")
    check(len(manifest_source_paths) == len(manifest["files"]), "ARCHIVE_UNIQUE_SOURCE_IDENTITIES")

    result = dict(
        schema="BENCH_ADAPTATION_ARCHIVE_CONTINUITY_INDEPENDENT_V1",
        status="FAIL" if errors else "PASS",
        prior_audit=str(prior_path), prior_audit_sha256=prior_sha,
        prior_verified_hashes_sha256=hashes_sha,
        archive_completion_audit=str(completion_path), archive_completion_sha256=completion_sha,
        prior_small_file_continuity=continuity,
        current_only_snapshots=snapshots,
        external_source_count=len(external_copies),
        checks=checks, errors=errors, files=files,
        model_loads=0, model_calls=0, scoring_calls=0, large_weight_content_reads=0,
        source_files_modified=0,
        limitation="Current/prior content continuity checked only for small files. "
                   "No model weight contents were read. This does not revalidate visual judgments.",
    )
    receipt = OUT / "diagnostics/archive_continuity_independent.json"
    with receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], checks=len(checks), errors=len(errors),
                          prior_small_files=len(continuity), external_sources=len(external_copies),
                          large_weight_content_reads=0, receipt=str(receipt))))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
