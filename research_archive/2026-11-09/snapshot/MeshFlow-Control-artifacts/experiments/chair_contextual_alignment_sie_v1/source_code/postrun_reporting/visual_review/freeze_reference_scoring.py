"""Freeze completed, manually observed source-reference image review.
Reads only anonymous assignments, prior frozen observations, manually authored
reference notes, and the assigned PNG files. No methods, numerical results,
model, geometry, scoring, or rendering imports.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import io
import json

RUN = Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_contextual_alignment_sie_v1")
ROOT = RUN / "visual_review"
EXPECTED_BLIND_SHA = "2fe85cb1098c721f1d77f9077f3623ac7869d8b2ca5e583e39583af92497c34d"
VIEWS = ["front_3q", "back_3q", "front", "side", "wire"]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def resolve(path):
    if len(path) > 2 and path[1] == ":":
        return Path("/mnt/" + path[0].lower() + "/" + path[3:].replace("\\", "/"))
    return Path(path)

def write_frozen(path, data):
    if path.exists():
        assert path.read_bytes() == data, "Refusing to replace a different frozen file: " + str(path)
    else:
        path.write_bytes(data)

def run():
    blind_path = ROOT / "blind_scoring.csv"
    assert sha(blind_path) == EXPECTED_BLIND_SHA
    global_freeze_path = ROOT / "source_hidden_freeze.json"
    global_freeze = read(global_freeze_path)
    assert global_freeze["status"] == "PASS"
    assert global_freeze["blind_output_rows"] == 192 and global_freeze["blind_pages"] == 48
    assert EXPECTED_BLIND_SHA in global_freeze["files"].values()
    freeze_time = datetime.fromisoformat(global_freeze["frozen_utc"])
    assignment_path = RUN / "audit" / "visual_assignments_anonymous.json"
    assignments = [x for x in read(assignment_path)["rows"] if x["reviewer"] == "scoring"]
    notes_path = ROOT / "reference_scoring_observations.json"
    notes = read(notes_path)
    assert len(assignments) == len(notes) == 16
    with blind_path.open(encoding="utf-8", newline="") as f:
        blind = list(csv.DictReader(f))
    assert len(blind) == 64
    by_key = {(r["request_id"], r["anonymous_method"]): r for r in blind}
    assert len(by_key) == 64
    out_rows, pages = [], []
    for i, (a, n) in enumerate(zip(assignments, notes)):
        assert n["index"] == i and len(n["methods"]) == 4
        observed = datetime.strptime(n["observed_utc"], "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        assert observed > freeze_time
        image_path = resolve(a["reference_page"])
        assert image_path.is_file()
        image_sha = sha(image_path)
        pages.append({"request_id":a["request_id"], "page":a["reference_page"],
                      "page_sha256":image_sha, "bytes":image_path.stat().st_size,
                      "observed_utc":n["observed_utc"],
                      "original_pixel_inspection_crops":[[0,0,1650,980],[0,960,1650,1913]],
                      "all_five_views_read":True, "source_reference_read":True})
        for j, method in enumerate(a["anonymous_methods"]):
            original = by_key[(a["request_id"], method)]
            assert original["stage"] == "SOURCE_HIDDEN"
            assert json.loads(original["view_scope"]) == VIEWS
            r = dict(original)
            r.update(stage="REFERENCE", page=a["reference_page"], page_sha256=image_sha,
                     observed_utc=n["observed_utc"], review_status="REVIEWED_REFERENCE",
                     source_specific_difference=n["methods"][j][0],
                     alternative_valid_completion=n["methods"][j][1],
                     source_reference_description=n["source_summary"],
                     notes=n["methods"][j][2] + " " + original["notes"],
                     blind_observation_row_id=a["request_id"] + "__" + method,
                     blind_observations_sha256=EXPECTED_BLIND_SHA,
                     source_hidden_observations_frozen_before_reference=True,
                     reference_viewed_before_this_observation=True,
                     prior_mapping_exposure=True, human_usability="NOT_RUN")
            assert all(str(r[k]).strip() for k in (
                "notes","source_specific_difference","alternative_valid_completion",
                "blind_observation_row_id","observed_utc","page_sha256"))
            out_rows.append(r)
    assert len(out_rows) == 64
    fields = list(blind[0]) + [k for k in out_rows[0] if k not in blind[0]]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(out_rows)
    csv_path = ROOT / "reference_scoring.csv"
    write_frozen(csv_path, stream.getvalue().encode("utf-8"))
    assert sha(blind_path) == EXPECTED_BLIND_SHA
    receipt = {
        "status":"COMPLETE_ASSIGNED_REFERENCE_REVIEW",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "review_kind":"AI_VISUAL_REVIEW",
        "reviewer":"scoring",
        "stage":"REFERENCE",
        "assigned_pages":16,
        "reviewed_pages":16,
        "reviewed_output_rows":64,
        "view_scope":VIEWS,
        "source_reference_is_one_valid_completion_not_unique_target":True,
        "method_blinding":"Anonymous labels and source-hidden-first AI review; NOT strict method blinding",
        "prior_mapping_exposure":True,
        "method_mapping_reopened_during_this_review":False,
        "numeric_results_read_during_this_review":False,
        "human_usability":"NOT_RUN",
        "visual_rate_or_promotion_computed":False,
        "model_calls":0,
        "scoring_calls":0,
        "render_calls":0,
        "inspection":"Each existing PNG actually viewed at original pixels in two vertical crops; no saved image modifications",
        "original_blind_csv_unchanged":True,
        "blind_csv_sha256":EXPECTED_BLIND_SHA,
        "global_source_hidden_freeze_sha256":sha(global_freeze_path),
        "global_source_hidden_freeze_utc":global_freeze["frozen_utc"],
        "all_reference_observation_times_after_global_freeze":True,
        "assignment_sha256":sha(assignment_path),
        "reference_csv":{"path":str(csv_path),"sha256":sha(csv_path),"rows":64},
        "reference_manual_notes":{"path":str(notes_path),"sha256":sha(notes_path),"pages":16},
        "pages":pages
    }
    receipt_path = ROOT / "reference_scoring_receipt.json"
    if receipt_path.exists():
        existing = read(receipt_path)
        receipt["created_utc"] = existing["created_utc"]
    write_frozen(receipt_path, (json.dumps(receipt,ensure_ascii=False,indent=2)+"\n").encode("utf-8"))
    print(json.dumps({"status":receipt["status"],"pages":16,"rows":64,
                      "reference_csv_sha256":sha(csv_path),
                      "receipt_sha256":sha(receipt_path),
                      "blind_sha256_unchanged":sha(blind_path)},indent=2))

if __name__ == "__main__":
    run()
