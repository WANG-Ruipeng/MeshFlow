"""Finish only the archive step of a completed content audit.

Preserves the failed audit. Reuses its timestamped content verification rather
than claiming to rehash large weights. Source/registration files and checkpoint
receipts are reread; no model, scoring, training, cleanup or Git operations.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import time
import json

from .closeout_audit_v2 import (
    Audit, ROOT, OUT, LABELS, read, write, resolve, archive_and_inventory,
)


class ArchiveAudit(Audit):
    def __init__(self, prior_hashes):
        super().__init__()
        self.prior_hashes = prior_hashes
        self.verified_hashes.update(prior_hashes)
        self.inherited_large = {}

    def sha(self, path):
        path = resolve(path).resolve()
        previous = self.prior_hashes.get(str(path))
        if path.suffix.lower() == ".pt" and previous is not None:
            if not previous["matches_registered"] or path.stat().st_size != previous["bytes"]:
                raise ValueError("Previously verified weight identity/size differs: " + str(path))
            receipt_path = path.with_suffix(".json")
            receipt_previous = self.prior_hashes.get(str(receipt_path))
            if receipt_previous is None:
                # Not all receipts are themselves listed in the old hash map.
                # Bind their current content and its SHA field explicitly.
                receipt_sha = super().sha(receipt_path)
            else:
                receipt_sha = super().sha(receipt_path)
                if receipt_sha != receipt_previous["sha256"]:
                    raise ValueError("Weight receipt changed: " + str(receipt_path))
            if read(receipt_path)["sha256"] != previous["sha256"]:
                raise ValueError("Weight receipt no longer names verified content: " + str(path))
            self.inherited_large[str(path)] = dict(
                sha256=previous["sha256"], bytes=previous["bytes"],
                current_receipt_sha256=receipt_sha,
                verification="Inherited from linked completed content audit; current size and receipt checked, not rehashed.",
            )
            return previous["sha256"]
        return super().sha(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-audit", type=Path, required=True)
    args = parser.parse_args()
    prior_path = args.prior_audit.resolve()
    prior_path.relative_to(OUT / "closeout_audits_v2")
    old = read(prior_path)
    failures = old.get("failures", [])
    if (old["status"] != "FAIL" or old.get("missing") or len(failures) != 1
            or failures[0].get("check") != "AUDIT_EXECUTION"
            or "Refuse source archive outside workspace: /mnt/e/MeshFlow-Control/src/meshflow_control/"
            not in failures[0].get("error", "")
            or not old.get("checks") or not all(x["passed"] for x in old["checks"])
            or old["readiness"]["actual_review_cells"] != 672):
        raise ValueError("Only the known archive-root failure may be completed")
    hashes_path = prior_path.with_name("verified_hashes.json")
    previous = read(hashes_path)
    if len(previous) != old["verified_file_count"] or not all(
        x["matches_registered"] for x in previous.values()
    ):
        raise ValueError("Previous verified content inventory is incomplete")
    started = time.perf_counter()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = OUT / "closeout_audits_v2" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    audit = ArchiveAudit(previous)
    dependencies = {}
    registrations = [
        ("training_registration.json", ("training_code_hashes",)),
        ("registration.json", ()),
        ("remaining_runtime_code_registration.json", ("code", "source_hashes")),
        ("evaluation/score_registration.json", ("source_hashes",)),
        ("gallery/manifest.json", ("protected",)),
        ("gallery/definition.json", ("protected",)),
        ("visual_review/protocol.json", ()),
        ("visual_review/assignments.json", ()),
    ]
    registrations += [("evaluation/inference_registrations/" + label + ".json", ("code",))
                      for label in LABELS]
    for name, mappings in registrations:
        path = OUT / name
        document = read(path)
        audit.registration_paths.add(path.resolve())
        dependencies[str(path)] = audit.sha(path)
        for mapping in mappings:
            for name, expected in document.get(mapping, {}).items():
                source = resolve(name).resolve()
                if source.suffix.lower() in (".py", ".md", ".toml", ".yaml", ".yml"):
                    audit.hash_equals(source, expected, context="archive registered source")
                    audit.archive_sources.add(source)
    # Include every previously verified small source, including scorer imports.
    for name, identity in previous.items():
        source = Path(name)
        if source.suffix.lower() in (".py", ".md", ".toml", ".yaml", ".yml"):
            audit.hash_equals(source, identity["sha256"], context="prior verified source continuity")
            audit.archive_sources.add(source)
    for name in (
        "eval_manifest.json", "diagnostics/results_v2_independent_audit.json",
        "diagnostics/results_v2_independent_audit.initial.json",
        "diagnostics/collector_v2_revision.json",
    ):
        path = OUT / name
        if path.is_file():
            audit.registration_paths.add(path.resolve())
            dependencies[str(path)] = audit.sha(path)
    for name in (
        "results_v2/completion.json", "visual_review/completion.json",
        "result_analysis/completion.json", "training_summary/latest.json",
    ):
        path = OUT / name
        dependencies[str(path)] = audit.sha(path)
    if audit.failures or audit.missing:
        raise ValueError("Registered source changed: " + repr(audit.failures + audit.missing))
    archive = archive_and_inventory(audit, folder)
    if audit.failures or audit.missing:
        raise ValueError("Archive inventory failed: " + repr(audit.failures + audit.missing))
    result = dict(
        schema="BENCH_ADAPTATION_ARCHIVE_COMPLETION_V2", status="PASS",
        final_requested=True, prior_audit=str(prior_path),
        prior_audit_sha256=audit.sha(prior_path),
        prior_verified_hashes_sha256=audit.sha(hashes_path),
        inherited_content_checks=len(old["checks"]),
        inherited_verified_file_count=len(previous),
        current_small_file_hashes=dependencies,
        inherited_large_weight_hashes=audit.inherited_large,
        limitation="Content/call checks inherit the linked earlier audit. No saved mtime signatures existed. "
                   "Large weights were not rehashed; current sizes and receipts were checked. "
                   "Registered source files were freshly hashed. Original failure is preserved.",
        resolved_failure=failures[0], archive=archive,
        failures=[], missing=[], model_calls=0, scoring_calls=0,
        copied_weights=0, deleted_files=0, git_actions=0,
        elapsed_seconds=time.perf_counter() - started,
        audit_path=str(folder / "closeout_audit.json"),
    )
    write(folder / "verified_hashes.json", audit.verified_hashes)
    write(folder / "closeout_audit.json", result)
    print(json.dumps(dict(status="PASS", audit_path=result["audit_path"],
                         source_files=archive["source_files"], disk=archive["disk"],
                         inherited_content_checks=result["inherited_content_checks"],
                         elapsed_seconds=result["elapsed_seconds"]), indent=2))


if __name__ == "__main__":
    main()
