"""Prepare fresh nonblind image-review assignments only after the gallery exists.

CPU/file validation only: no image ratings, numeric results, private anonymous
labels, models, mesh scoring or pilot imports are read or produced here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from . import run as A
from ..bench_joint_shift_v1.closeout import AXES

C = A.core
FIELDS = (
    "request_id", "parent_id", "condition_id", "label", "stage", "page_id",
    "image_path", "image_sha256", "actually_viewed", *AXES, "note",
)

PROTOCOL = """# Crossover: actual image review protocol

This is a fresh **nonblind AI_PROXY** review, not a human study or the earlier
anonymous two-model gate. Review every one of the 144 completed PNG pages:
48 fixed requests times RAW, BASE_POST and CLEAN_FINAL_POST. Each page has
three columns, giving 432 model-stage records. No labels are prefilled.

Left to right: **RELATION_V1_S1500 / JOINT_SHIFT_S1500 / SHIFT_RELATION_T050**.
The candidate reuses SHIFT through state[25], then RELATION for the final
25 Euler steps. Column order and model names are not quality rankings.
Both references must be inspected anew in the same round as the candidate.

On each full PNG, inspect all three columns, both original views (azimuth
-60 and 120, elevation 23), and both fixed-position front/side sections.
The frame, scale and section planes are inherited and common to all models
and stages. Blue is C, orange SEAM, gray BULK, and green an added bridge.
Read actual part_type from the assignment: legacy condition names can differ
from the validated part type. Do not grade similarity to a unique source GT.

Use exactly these five axes, assessed independently:

- overall_shape: REASONABLE / INCOMPLETE / UNCERTAIN
- C_use: PLAUSIBLE / REPLACED_OR_IGNORED / UNCERTAIN
- support_retention: PRESENT / MISSING_OR_DAMAGED / UNCERTAIN
- visible_fragment_or_gap: YES / NO_OBVIOUS / UNCERTAIN
- layering_or_fold: YES / NO_OBVIOUS / UNCERTAIN

These mean plausible whole Bench structure, participation of C in that
structure, visible support, visible fragments/gaps, and visible layers/folds.
A complete-looking Bench can still replace C; plausible use of C can coexist
with missing support or a damaged body. Use UNCERTAIN for occlusion, clipping,
ambiguous input, projection or insufficient finite-section evidence.
NO_OBVIOUS is not a certificate of defect absence. Section endpoints alone do
not establish cracks, floating faces, exact intersection or indexed topology.

Actually open the full page with view_image (or an equivalent full-image
viewer) before recording its three cells. Contact sheets do not replace this.
Do not copy earlier RELATION/SHIFT labels, anonymous-gate ratings, or historical
conclusions. Do not infer labels from CF/FF/I, numeric tables, reachability,
training loss, expected gains, or any method ranking. Do not prefill unviewed
pages. Keep a concise image-grounded note for every model-stage cell.

Each JSON record must contain exactly the required identity/evidence fields
listed in assignments.json: request_id, parent_id, condition_id, label, stage,
page_id, image_path, image_sha256, actually_viewed:true, all five axes, and note.
Use the registered PNG hash, not a placeholder. The wrapper declares
reviewer_kind:"AI_PROXY", blind:false, protocol_sha256 and records:[...].
An optional actual_view_source can name the viewer. Save small completed
batches without overwriting earlier viewed records; on interruption leave
remaining pages pending rather than filling them from memory or metrics.

There are three deterministic four-parent groups, each with 16 requests,
48 pages and 144 records: history TRAIN first, then two sorted DEV groups.
Historical TRAIN means exposure during COMMON12000, not membership in the
current 26-parent adaptation pool. It is regression evidence, not unseen-parent
generalization. The DEV panel is established development evidence, not a new
independent paper test set. Pretraining exposure is not resolved here.

Keep RAW, BASE_POST and CLEAN_FINAL_POST separate. BASE_POST combines official
D0, C/SEAM restoration, GEO_C and orientation; changes cannot be attributed
to D0 alone. Record both gains and losses, with uncertainty retained. Numerical
topology, conflicts, deleted faces and legal interfaces remain separate evidence.
Any later adjudication must be a separate documented record, not a silent edit.

This protocol grants no extra generation, noise, model calls, scoring, training,
checkpoint selection, image filtering or frozen-code changes. Only actual review
records are written by reviewers; these assignments do not claim any image has
already been viewed. No review or success outcome is assumed.
"""


def waiting(reason, missing=()):
    return dict(status="WAITING", reason=reason, missing=list(missing),
                model_calls=0, scoring_calls=0, labels_written=0,
                assignments_written=False, protocol_written=False)


def prepare(exp=A.DEFAULT):
    exp = C.resolve(exp).resolve()
    required = [exp / p for p in (
        "blind_review/gate.json", "eval_manifest.json", "gallery/completion.json",
        "gallery/manifest.json", "gallery/definition.json",
    )]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        return waiting("Gallery and passed anonymous gate must be complete", missing)
    gate, manifest, gallery, plan, definition = [C.read(p) for p in required]
    if gate.get("status") != "PASS":
        return waiting("Anonymous gate has not passed")
    if gallery.get("status") != "RENDER_COMPLETE_PENDING_ACTUAL_REVIEW":
        return waiting("Gallery renderer has not completed")
    if gallery.get("sheets") != 144 or len(gallery.get("images", [])) != 144:
        return waiting("All 144 gallery pages are required")
    queries = manifest["requests"]
    assert len(queries) == 48 and len({q["request_id"] for q in queries}) == 48
    qmap = {q["request_id"]: q for q in queries}
    counts = Counter(q["parent_id"] for q in queries)
    assert len(counts) == 12 and set(counts.values()) == {4}
    train = sorted({q["parent_id"] for q in queries if q["split"] == "TRAIN"})
    dev = sorted({q["parent_id"] for q in queries if q["split"] == "DEV"})
    assert len(train) == 4 and len(dev) == 8 and not set(train) & set(dev)
    assert all(q["split"] in ("TRAIN", "DEV") for q in queries)
    groups = {"group1": train, "group2": dev[:4], "group3": dev[4:]}
    assert definition["manifest_sha256"] == C.sha(required[3])
    definition_hash = C.sha(required[4])
    expected = {(s, q["request_id"]) for s in A.STAGES for q in queries}
    planned = {(p["sheet"], p["request_id"]): p for p in plan["pages"]}
    assert len(plan["pages"]) == len(planned) == 144 and set(planned) == expected
    images = {}
    pending_files = []
    for im in gallery["images"]:
        key = (im["sheet"], im["request_id"])
        assert key in expected and key not in images
        assert im["page_id"] == im["sheet"] + "__" + im["request_id"]
        assert im["definition_sha256"] == definition_hash
        q = qmap[im["request_id"]]
        p = planned[key]
        for field in ("condition_id", "part_type", "split"):
            assert im[field] == p[field] == q[field]
        assert p["parent_id"] == q["parent_id"]
        for value in (im, p):
            assert [cell["model"] for cell in value["cells"]] == list(A.LABELS)
            assert all(cell["stage"] == im["sheet"] for cell in value["cells"])
        png = C.resolve(im["path"]).resolve()
        assert png == exp / "gallery" / (im["page_id"] + ".png")
        if not png.is_file():
            pending_files.append(str(png))
        else:
            assert C.sha(png) == im["image_sha256"], "PNG hash mismatch: " + str(png)
            with png.open("rb") as f:
                assert f.read(8) == b"\x89PNG\r\n\x1a\n", "Not a PNG: " + str(png)
        images[key] = im
    assert set(images) == expected
    if pending_files:
        return waiting("Gallery receipt exists but PNGs are missing", pending_files)

    protocol_bytes = PROTOCOL.encode("utf-8")
    result = dict(
        schema="BENCH_CROSSOVER_ACTUAL_VISUAL_REVIEW_V1", status="PENDING_ACTUAL_REVIEW",
        reviewer_kind="AI_PROXY", blind=False, protocol="protocol.md",
        protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
        eval_manifest_sha256=C.sha(required[1]), gallery_completion_sha256=C.sha(required[2]),
        source_hashes={str(p): C.sha(p) for p in required},
        grouping_rule="Sorted 4 historical TRAIN parents; sorted DEV parents first4/last4",
        groups=groups, expected_pages_per_group=48, expected_model_stage_rows_per_group=144,
        expected_total_pages=144, expected_total_model_stage_rows=432,
        labels=list(A.LABELS), stages=list(A.STAGES), axes=AXES,
        required_record_fields=list(FIELDS), tasks={}, model_calls=0, scoring_calls=0,
    )
    for group, parents in groups.items():
        qs = sorted((q for q in queries if q["parent_id"] in parents),
                    key=lambda q: (q["parent_id"], q["condition_id"], q["noise_index"], q["request_id"]))
        pages = []
        for q in qs:
            for stage in A.STAGES:
                im = images[(stage, q["request_id"])]
                pages.append(dict(
                    **{k: q[k] for k in ("request_id", "parent_id", "condition_id", "part_type", "noise_index")},
                    historical_group=q["split"], stage=stage, page_id=im["page_id"],
                    image_path=im["path"], image_sha256=im["image_sha256"],
                    status="PENDING_ACTUAL_VIEW",
                ))
        assert len(qs) == 16 and len(pages) == 48
        result["tasks"][group] = dict(
            parents=parents, request_count=16, page_count=48, expected_model_stage_rows=144,
            record_file="review_" + group + ".json", required_record_fields=list(FIELDS),
            required_wrapper=dict(reviewer_kind="AI_PROXY", blind=False,
                                  protocol_sha256=result["protocol_sha256"]),
            instruction="Read protocol.md. Actually view each full PNG, then record all three columns. "
                        "Do not copy old or anonymous labels or consult numeric rankings. Save completed batches. "
                        "Use exact identities/hashes below, five allowed axes and image-grounded notes; "
                        "write only review_" + group + ".json. No unviewed or prefilled records.",
            pages=pages,
        )
    outputs = {
        exp / "visual_review/protocol.md": protocol_bytes,
        exp / "visual_review/assignments.json":
            (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    }
    # Preflight both files before any write; idempotent reruns never replace a protocol.
    for path, content in outputs.items():
        if path.exists() and path.read_bytes() != content:
            raise ValueError("Refusing to replace visual-review preparation: " + str(path))
    for path, content in outputs.items():
        C.write_bytes(path, content)
    return dict(status="PREPARED_PENDING_ACTUAL_REVIEW", pages=144, records_required=432,
                groups={g: t["record_file"] for g, t in result["tasks"].items()},
                model_calls=0, scoring_calls=0, labels_written=0,
                files={str(p): C.sha(p) for p in outputs})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default=str(A.DEFAULT))
    args = parser.parse_args()
    print(json.dumps(prepare(args.exp), ensure_ascii=False, indent=2))
