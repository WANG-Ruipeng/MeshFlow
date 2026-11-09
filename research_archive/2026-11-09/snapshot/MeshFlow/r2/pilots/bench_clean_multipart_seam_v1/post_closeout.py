"""Read-only accounting of the completed CPU cascade, including partial attempts."""
from __future__ import annotations

import collections
import csv
import sys

from . import post_cpu as P
from .eval_prepare import LABELS

C = P.C
ROOT = P.ROOT


def run():
    dest = ROOT / "cpu_execution_accounting.json"
    if dest.exists():
        result = C.read(dest)
        P.E.verify(result["files"])
        return result
    extra = ROOT / "post_extras/all/completion.json"
    P.E.verify(C.read(extra)["files"])
    requests = C.read(ROOT / "eval_manifest.json")["requests"]
    summaries = []
    pair_origins = collections.Counter()
    files = {str(extra): C.sha(extra)}
    for label in [*P.LABELS, *LABELS]:
        totals = collections.Counter()
        historical = label in P.LABELS
        for request in requests:
            cell = ROOT / ("historical" if historical else "cpu") / label / request["request_id"]
            receipt = cell / "complete.json"
            P.E.verify(C.read(receipt)["files"])
            files[str(receipt)] = C.sha(receipt)
            totals["completed_cells"] += 1
            for stage in ("BASE_POST", "CLEAN_FINAL_POST"):
                score = C.read(cell / stage / "scores/score.json")
                pair_origins[stage + ":" + score["pair_origin"]] += 1
            for stage in ("HINGE_OPP015", "ALPHA", "CLEAN_FINAL_POST"):
                item = C.read(cell / stage / "summary.json")
                totals[stage + "_accepted"] += item["accepted"]
                totals[stage + "_candidate_full_pair_scans"] += item.get("candidate_full_geometry_scans", 0)
                totals[stage + "_new_bridge_against_current_pair_scans"] += item.get("new_face_pair_scans", 0)
            final = C.read(cell / "base_final_paired_ledger.json")
            for key, value in final["summary"].items():
                if isinstance(value, (int, float)):
                    totals[key] += value
            for key, value in final["full_indexed_guard"].items():
                if isinstance(value, bool):
                    if not value:
                        raise AssertionError("C identity guard failed")
                else:
                    totals[key] += value
        summaries.append(dict(label=label, **totals))
    partial_base = list((ROOT / "engineering_revision_01").glob("*/BASE_POST/scores/complete.json"))
    partial_hinge = list((ROOT / "engineering_revision_01").glob("*/HINGE_OPP015/summary.json"))
    partial_final = list((ROOT / "engineering_revision_01").glob("*/CLEAN_FINAL_POST/scores/complete.json"))
    assert len(partial_base) == 2 and len(partial_hinge) == 2 and len(partial_final) == 0
    for path in [*partial_base, *partial_hinge]:
        files[str(path)] = C.sha(path)
    pair_caches = list((ROOT / "base_pair_caches").glob("*/*/complete.json"))
    assert len(pair_caches) == 96
    with (ROOT / "post_extras/all/safety_ledger.csv").open(encoding="utf-8-sig", newline="") as stream:
        safety = list(csv.DictReader(stream))
    assert len(safety) == 240
    zero_fields = (
        "CF_new_illegal_pairs", "FF_new_illegal_pairs", "CF_lost_legal_edges",
        "FF_lost_legal_edges", "deleted_source_faces", "new_nonmanifold", "new_same_winding",
    )
    for field in zero_fields:
        assert all(float(row[field]) == 0 for row in safety), field
    result = dict(
        status="COMPLETE", historical_cells=144, new_cells=96, complete_cells=240,
        BASE_scores=242, FINAL_scores=240, formal_CPU_score_cells=480,
        extra_partial_BASE_scores=len(partial_base), extra_partial_HINGE_attempts=len(partial_hinge),
        extra_partial_FINAL_scores=0, BASE_original_full_pair_scans=96,
        final_pair_sources=dict(pair_origins), per_label=summaries,
        partial_HINGE_candidate_full_pair_scans=sum(
            C.read(path).get("candidate_full_geometry_scans", 0) for path in partial_hinge),
        model_calls=0, D0_calls=0, new_query_draws=0, all_C_coordinates_winding_indices_preserved=True,
        all_zero_safety_fields=list(zero_fields),
        pair_cache_contract=(
            "No-BRIDGE FINAL reuses the full pair cache of the final accepted HINGE/ALPHA geometry, "
            "or the unchanged BASE cache. Each accepted geometric proposal is checked with the "
            "original full detector; identical ALPHA geometries may use exact cache hits. "
            "BRIDGE FINAL is independently fully scanned and compared with the incremental cache. "
            "BASE score expansions reuse complete pair caches; new BASE pair caches are scanned once."
        ),
        additional_notes=(
            "Topology and extras are read-only derivations, not additional metric-score cells. "
            "The two failed engineering partials are preserved; concurrency drain caused no cell replay. "
            "I uses fixed C queries and is not certified nonincreasing by the finite-pair guard."
        ),
        code_sha256=C.sha(__file__), files=files,
    )
    if "torch" in sys.modules:
        raise AssertionError("CPU accounting imported neural framework")
    C.write(dest, result)
    return result


if __name__ == "__main__":
    result = run()
    print({key: value for key, value in result.items() if key not in ("files", "per_label")})
