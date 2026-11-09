"""Validate actual AI_PROXY observations and summarize the frozen 414 cells.

This report utility reads no numerical scores or model code. Missing or pending
reviews produce only a PENDING message, never a completed report. Historical
EXPOSURE labels are compared separately after current reviews are complete.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import statistics
import tempfile

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent.parent
REVIEW = OUT / "visual_review"
DEST = REVIEW / "summary_v1"
HISTORICAL = ROOT / "experiments/r2_bench_four_routes_v1/exposure/visual_review/summary_v1/actual_observations.csv"
CONTRASTS = (("PLAIN1500", "MULTILOD1500"), ("EXPOSURE", "PLAIN1500"), ("EXPOSURE", "MULTILOD1500"))
POSITIVE = {"overall_shape": "REASONABLE", "C_use": "PLAUSIBLE", "support_retention": "PRESENT"}
JOINT = "three_axis_joint"
JOINT_NOTE = "REASONABLE AND PLAUSIBLE AND PRESENT; this is not a no-fragment, no-intersection or watertightness certificate."
SCHEMA = "BENCH_MULTILOD_ACTUAL_VISUAL_SUMMARY_V1"


def resolve(path):
    s = str(path).replace("\\", "/")
    for prefix in ("E:/MeshFlow/", "/mnt/e/MeshFlow/"):
        if s.startswith(prefix):
            return ROOT / s[len(prefix):]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def csv_bytes(rows):
    if not rows:
        raise ValueError("Refuse a headerless empty report")
    fields = list(rows[0])
    if any(set(r) != set(fields) for r in rows):
        raise ValueError("CSV row schema mismatch")
    f = io.StringIO(newline="")
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return f.getvalue().encode("utf-8")


def joint(row):
    values = [row[k] for k in POSITIVE]
    if all(row[k] == v for k, v in POSITIVE.items()):
        return "YES"
    if any(row[k] not in (v, "UNCERTAIN") for k, v in POSITIVE.items()):
        return "NO"
    if "UNCERTAIN" in values:
        return "UNCERTAIN"
    raise ValueError("Invalid three-axis values")


def expand(doc, axes):
    """Rows are preferred. Existing blank page templates can also be completed."""
    if "rows" in doc:
        if "pages" in doc:
            raise ValueError("Use rows OR pages, never both")
        return doc["rows"]
    result = []
    for page in doc.get("pages", []):
        base = {k: v for k, v in page.items() if k != "ratings"}
        for label, rating in page.get("ratings", {}).items():
            result.append({**base, "label": label, **rating})
    return result


def attest(row, doc, expected, axes, image_cache, protected):
    key = (row.get("request_id"), row.get("label"), row.get("stage"))
    if key not in expected:
        raise ValueError(f"Unregistered cell {key}")
    unit = expected[key]
    for field in ("page_id", "parent_id", "condition_id", "noise_index", "report_group"):
        if field in row and row[field] != unit[field]:
            raise ValueError(f"{key}: incorrect {field}")
    if row.get("page_id") != unit["page_id"]:
        raise ValueError(f"{key}: missing/incorrect page_id")
    if row.get("actually_viewed") is not True or row.get("views_and_sections_checked") is not True:
        raise ValueError(f"{key}: view/sections not attested")
    if row.get("tool") != "view_image":
        raise ValueError(f"{key}: actual view_image event is required")
    event = row.get("view_event_id")
    if not isinstance(event, str) or not event.strip() or event.lower() in ("pending", "todo", "null"):
        raise ValueError(f"{key}: missing actual view event")
    timestamp = datetime.fromisoformat(str(row.get("viewed_at_utc", "")).replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError(f"{key}: view time must be UTC")
    if timestamp > datetime.now(timezone.utc):
        raise ValueError(f"{key}: future view time")
    reviewer = row.get("reviewer") or doc.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError(f"{key}: missing reviewer identity")
    if doc.get("reviewer_kind") != "AI_PROXY" or doc.get("blind") is not False:
        raise ValueError("Current protocol is nonblind AI_PROXY")
    for axis, values in axes.items():
        if row.get(axis) not in values:
            raise ValueError(f"{key}: missing or invalid {axis}")
    if not isinstance(row.get("note"), str) or not row["note"].strip():
        raise ValueError(f"{key}: concrete visual evidence note is required")
    page_id = unit["page_id"]
    image = OUT / "gallery" / page_id / "comparison.png"
    receipt = image.with_name("comparison.json")
    if "image_path" in row and resolve(row["image_path"]) != image:
        raise ValueError(f"{key}: image path mismatch")
    if page_id not in image_cache:
        saved = read(receipt)
        h, rh = sha(image), sha(receipt)
        if saved.get("image_sha256") != h or saved.get("page_id") != page_id:
            raise ValueError(f"{key}: gallery receipt/image mismatch")
        identity = saved["identity"]
        if identity.get("request_id") != unit["request_id"] or identity.get("stage") != unit["stage"]:
            raise ValueError(f"{key}: gallery request/stage mismatch")
        labels = [c["label"] for c in identity["cells"]]
        if labels != ["EXPOSURE", "PLAIN1500", "MULTILOD1500"]:
            raise ValueError(f"{key}: gallery model rows differ")
        image_cache[page_id] = (h, rh, timestamp, reviewer, event)
        protected[str(image)] = h
        protected[str(receipt)] = rh
    h, rh, _, _, _ = image_cache[page_id]
    if row.get("image_sha256") != h or row.get("image_receipt_sha256") != rh:
        raise ValueError(f"{key}: observed image/receipt hash missing or changed")
    return {**unit, **{a: row[a] for a in axes}, JOINT: joint(row), "note": row["note"],
            "reviewer": reviewer, "reviewer_kind": "AI_PROXY", "blind": False,
            "actually_viewed": True, "views_and_sections_checked": True,
            "tool": row["tool"], "view_event_id": event, "viewed_at_utc": row["viewed_at_utc"],
            "image_path": str(image), "image_sha256": h, "image_receipt_sha256": rh}


def load_current():
    protocol_path = REVIEW / "protocol.json"
    assignment_path = REVIEW / "assignments.json"
    protocol, assignments = read(protocol_path), read(assignment_path)
    if assignments["protocol_sha256"] != sha(protocol_path):
        raise ValueError("Frozen protocol changed")
    for pathkey, hashkey in (("criteria_source_path", "criteria_source_sha256"), ("axes_source_path", "axes_source_sha256"), ("manifest_path", "manifest_sha256")):
        if sha(resolve(protocol[pathkey])) != protocol[hashkey]:
            raise ValueError(f"Frozen {pathkey} changed")
    expected = {(u["request_id"], u["label"], u["stage"]): u for u in protocol["expected_units"]}
    if len(expected) != 414 or len(protocol["expected_units"]) != 414:
        raise ValueError("Expected exactly 414 unique cells")
    docs, pending = [], []
    protected = {str(protocol_path): sha(protocol_path), str(assignment_path): sha(assignment_path)}
    for assignment in assignments["assignments"]:
        path = resolve(assignment["completed_record_path"])
        if not path.exists():
            pending.append({"assignment_id": assignment["assignment_id"], "reason": "MISSING_SHARD", "path": str(path)})
            continue
        doc = read(path)
        if doc.get("status") != "COMPLETE":
            pending.append({"assignment_id": assignment["assignment_id"], "reason": "SHARD_NOT_COMPLETE", "path": str(path)})
            continue
        if doc.get("assignment_id") != assignment["assignment_id"] or doc.get("protocol_sha256") != sha(protocol_path):
            raise ValueError(f"Shard assignment/protocol changed: {path}")
        rows = expand(doc, protocol["axes"])
        if len(rows) != assignment["cells"]:
            raise ValueError(f"Completed shard has wrong cardinality: {path}")
        docs.append((assignment, doc, rows))
        protected[str(path)] = sha(path)
    if pending:
        return {"status": "PENDING", "pending": pending, "completed_shards": len(docs), "results_written": 0}
    observations, seen, image_cache, owners = [], set(), {}, {}
    for assignment, doc, rows in docs:
        page_ids = set(assignment["page_ids"])
        for row in rows:
            key = (row.get("request_id"), row.get("label"), row.get("stage"))
            if key in seen:
                raise ValueError(f"Duplicate cell {key}")
            if key not in expected or expected[key]["review_assignment"] != assignment["assignment_id"]:
                raise ValueError(f"Wrong shard for {key}")
            if row.get("page_id") not in page_ids:
                raise ValueError(f"Wrong page for {key}")
            r = attest(row, doc, expected, protocol["axes"], image_cache, protected)
            frozen = datetime.fromisoformat(protocol["frozen_at_utc"].replace("Z", "+00:00"))
            viewed = datetime.fromisoformat(r["viewed_at_utc"].replace("Z", "+00:00"))
            if viewed < frozen:
                raise ValueError("Historical views cannot count as current-experiment observations")
            if r["request_id"] in owners and owners[r["request_id"]] != r["reviewer"]:
                raise ValueError("A request was split across reviewers")
            owners[r["request_id"]] = r["reviewer"]
            seen.add(key)
            observations.append(r)
    if seen != set(expected) or len(image_cache) != 138:
        raise ValueError("Missing cells/pages")
    label_order = {v: i for i, v in enumerate(protocol["labels"])}
    stage_order = {v: i for i, v in enumerate(protocol["stages"])}
    observations.sort(key=lambda r: (r["request_index"], stage_order[r["stage"]], label_order[r["label"]]))
    return {"status": "COMPLETE", "protocol": protocol, "rows": observations, "protected": protected}


def build_tables(rows, protocol):
    axes = {**protocol["axes"], JOINT: ["YES", "NO", "UNCERTAIN"]}
    parents = sorted({r["parent_id"] for r in rows})
    groups = ["ALL"] + sorted({r["report_group"] for r in rows})
    parent_counts, group_counts = [], []
    for label in protocol["labels"]:
        for stage in protocol["stages"]:
            panel = [r for r in rows if r["label"] == label and r["stage"] == stage]
            for parent in parents:
                rr = [r for r in panel if r["parent_id"] == parent]
                for axis, values in axes.items():
                    for value in values:
                        n = sum(r[axis] == value for r in rr)
                        parent_counts.append(dict(label=label,stage=stage,parent_id=parent,report_group=rr[0]["report_group"],axis=axis,value=value,count=n,requests=len(rr),fraction=n/len(rr)))
            for group in groups:
                rr = panel if group == "ALL" else [r for r in panel if r["report_group"] == group]
                by_parent = defaultdict(list)
                for r in rr:by_parent[r["parent_id"]].append(r)
                for axis, values in axes.items():
                    for value in values:
                        n = sum(r[axis] == value for r in rr)
                        macro = statistics.mean(sum(r[axis] == value for r in pp)/len(pp) for pp in by_parent.values())
                        group_counts.append(dict(label=label,stage=stage,report_group=group,axis=axis,value=value,count=n,requests=len(rr),parents=len(by_parent),request_fraction=n/len(rr),parent_macro_fraction=macro))
    index = {(r["label"],r["stage"],r["request_id"]):r for r in rows}
    request_transitions = []
    base_requests = [r for r in rows if r["label"] == protocol["labels"][0] and r["stage"] == protocol["stages"][0]]
    for baseline,candidate in CONTRASTS:
        for stage in protocol["stages"]:
            for q in base_requests:
                a,b = index[baseline,stage,q["request_id"]],index[candidate,stage,q["request_id"]]
                for axis in axes:
                    request_transitions.append(dict(baseline=baseline,candidate=candidate,stage=stage,request_id=q["request_id"],parent_id=q["parent_id"],condition_id=q["condition_id"],noise_index=q["noise_index"],report_group=q["report_group"],axis=axis,from_value=a[axis],to_value=b[axis],changed=a[axis]!=b[axis],uncertain_in_pair="UNCERTAIN" in (a[axis],b[axis])))
    parent_transitions,group_transitions=[],[]
    for baseline,candidate in CONTRASTS:
        for stage in protocol["stages"]:
            for axis,values in axes.items():
                rr=[r for r in request_transitions if r["baseline"]==baseline and r["candidate"]==candidate and r["stage"]==stage and r["axis"]==axis]
                for parent in parents:
                    pp=[r for r in rr if r["parent_id"]==parent]
                    counts=Counter((r["from_value"],r["to_value"]) for r in pp)
                    for a in values:
                        for b in values:
                            n=counts[a,b]
                            parent_transitions.append(dict(baseline=baseline,candidate=candidate,stage=stage,parent_id=parent,report_group=pp[0]["report_group"],axis=axis,from_value=a,to_value=b,count=n,requests=len(pp),fraction=n/len(pp)))
                for group in groups:
                    gg=rr if group=="ALL" else [r for r in rr if r["report_group"]==group]
                    by_parent=defaultdict(list)
                    for r in gg:by_parent[r["parent_id"]].append(r)
                    counts=Counter((r["from_value"],r["to_value"]) for r in gg)
                    for a in values:
                        for b in values:
                            n=counts[a,b]
                            macro=statistics.mean(sum((r["from_value"],r["to_value"])==(a,b) for r in pp)/len(pp) for pp in by_parent.values())
                            group_transitions.append(dict(baseline=baseline,candidate=candidate,stage=stage,report_group=group,axis=axis,from_value=a,to_value=b,count=n,requests=len(gg),parents=len(by_parent),request_fraction=n/len(gg),parent_macro_fraction=macro))
    return {"actual_observations.csv":rows,"parent_counts.csv":parent_counts,"five_axis_and_joint_counts.csv":group_counts,"paired_request_transitions.csv":request_transitions,"paired_parent_transitions.csv":parent_transitions,"paired_group_transitions.csv":group_transitions}


def historical_differences(rows, protocol):
    """Historical records are read only after validation of all current observations."""
    with HISTORICAL.open(encoding="utf-8-sig",newline="") as f:
        old=[r for r in csv.DictReader(f) if r["label"]=="EXPOSURE"]
    oldindex={(r["request_id"],r["stage"]):r for r in old}
    current=[r for r in rows if r["label"]=="EXPOSURE"]
    if len(old)!=138 or len(oldindex)!=138 or set(oldindex)!={(r["request_id"],r["stage"]) for r in current}:
        raise ValueError("Historical EXPOSURE panel differs; do not silently pool")
    differences=[]
    for r in current:
        h=oldindex[r["request_id"],r["stage"]]
        for axis,values in protocol["axes"].items():
            if h[axis] not in values:raise ValueError("Historical axis schema changed")
        h[JOINT]=joint(h)
        for axis in [*protocol["axes"],JOINT]:
            differences.append(dict(request_id=r["request_id"],parent_id=r["parent_id"],report_group=r["report_group"],stage=r["stage"],axis=axis,old_value=h[axis],current_value=r[axis],changed=h[axis]!=r[axis],uncertain_in_pair="UNCERTAIN" in (h[axis],r[axis]),old_reviewer=h.get("reviewer",""),current_reviewer=r["reviewer"],old_image_sha256=h.get("image_sha256",""),current_image_sha256=r["image_sha256"],old_note=h.get("note",""),current_note=r["note"]))
    return differences


def summarize():
    loaded=load_current()
    if loaded["status"]!="COMPLETE":return loaded
    rows,protocol,protected=loaded["rows"],loaded["protocol"],loaded["protected"]
    tables=build_tables(rows,protocol)
    tables["baseline_review_differences.csv"]=historical_differences(rows,protocol)
    protected[str(HISTORICAL)]=sha(HISTORICAL)
    protected[str(Path(__file__))]=sha(__file__)
    payloads={DEST/name:csv_bytes(table) for name,table in tables.items()}
    summary=dict(schema=SCHEMA,status="COMPLETE",cells=414,pages=138,requests=46,parents=23,reviewer_kind="AI_PROXY",blind=False,joint_definition=JOINT_NOTE,uncertain_policy="All UNCERTAIN labels are preserved in counts and full transition matrices. Joint is NO if any of its three axes is definitely negative; otherwise UNCERTAIN if any is uncertain. Strict joint positives are YES only; denominator remains all reviewed requests.",comparison_pairs=[list(c) for c in CONTRASTS],files={str(p):hashlib.sha256(data).hexdigest() for p,data in payloads.items()},protected=protected,observations_per_file={name:len(table) for name,table in tables.items()},score_files_read=0,labels_inferred_from_metrics=0,historical_labels_overwritten=0)
    payloads[DEST/'completion.json']=json_bytes(summary)
    for path,data in payloads.items():
        if path.exists() and path.read_bytes()!=data:
            raise ValueError(f"Existing completed report differs; no overwrite: {path}")
    for path,h in protected.items():
        if sha(path)!=h:raise ValueError(f"Input changed during aggregation: {path}")
    DEST.mkdir(parents=True,exist_ok=True)
    for path,data in payloads.items():
        if not path.exists():
            with path.open('xb') as f:f.write(data)
    return {"status":"COMPLETE","cells":414,"pages":138,"destination":str(DEST),"joint_definition":JOINT_NOTE}


def self_test():
    global OUT
    good={"overall_shape":"REASONABLE","C_use":"PLAUSIBLE","support_retention":"PRESENT"}
    assert joint(good)=="YES"
    assert joint({**good,"C_use":"UNCERTAIN"})=="UNCERTAIN"
    assert joint({**good,"C_use":"UNCERTAIN","support_retention":"MISSING_OR_DAMAGED"})=="NO"
    protocol={"labels":["EXPOSURE","PLAIN1500","MULTILOD1500"],"stages":["RAW"],"axes":{"overall_shape":["REASONABLE","INCOMPLETE","UNCERTAIN"],"C_use":["PLAUSIBLE","REPLACED_OR_IGNORED","UNCERTAIN"],"support_retention":["PRESENT","MISSING_OR_DAMAGED","UNCERTAIN"],"visible_fragment_or_gap":["YES","NO_OBVIOUS","UNCERTAIN"],"layering_or_fold":["YES","NO_OBVIOUS","UNCERTAIN"]}}
    rows=[]
    for label in protocol['labels']:
        for i in range(3):
            r=dict(label=label,stage='RAW',request_id=str(i),parent_id='a' if i<2 else 'b',condition_id=str(i),noise_index=i,report_group='TEST',**good,visible_fragment_or_gap='YES',layering_or_fold='UNCERTAIN')
            if label=='MULTILOD1500' and i==2:r['C_use']='UNCERTAIN'
            r[JOINT]=joint(r);rows.append(r)
    t=build_tables(rows,protocol)
    check=next(r for r in t['five_axis_and_joint_counts.csv'] if r['label']=='MULTILOD1500' and r['report_group']=='ALL' and r['axis']==JOINT and r['value']=='YES')
    assert check['count']==2 and check['request_fraction']==2/3 and check['parent_macro_fraction']==.5
    assert any(r['uncertain_in_pair'] for r in t['paired_request_transitions.csv'])
    assert len(t['paired_request_transitions.csv'])==3*3*6
    for table in t.values():assert csv_bytes(table)
    saved_out=OUT
    try:
        with tempfile.TemporaryDirectory(prefix='synthetic_visual_summary_test_') as tmp:
            OUT=Path(tmp)
            unit=dict(request_index=0,request_id='synthetic',parent_id='synthetic_parent',condition_id='synthetic_condition',report_group='SYNTHETIC',noise_index=0,label='EXPOSURE',stage='RAW',page_id='RAW__synthetic',review_assignment='synthetic_shard')
            image=OUT/'gallery'/unit['page_id']/'comparison.png';image.parent.mkdir(parents=True)
            image.write_bytes(b'SYNTHETIC HASH FIXTURE - NOT AN ACTUAL MESH IMAGE')
            receipt=image.with_name('comparison.json')
            receipt.write_bytes(json_bytes(dict(page_id=unit['page_id'],image_sha256=sha(image),identity=dict(request_id='synthetic',stage='RAW',cells=[dict(label=v) for v in protocol['labels']]))))
            row={**unit,**good,'visible_fragment_or_gap':'NO_OBVIOUS','layering_or_fold':'UNCERTAIN','note':'Synthetic validation fixture, not an actual review.','actually_viewed':True,'views_and_sections_checked':True,'tool':'view_image','view_event_id':'SYNTHETIC_ONLY','viewed_at_utc':datetime.now(timezone.utc).isoformat(),'image_sha256':sha(image),'image_receipt_sha256':sha(receipt)}
            doc=dict(reviewer='SYNTHETIC_ONLY',reviewer_kind='AI_PROXY',blind=False)
            expected={('synthetic','EXPOSURE','RAW'):unit}
            assert attest(row,doc,expected,protocol['axes'],{},{})[JOINT]=='YES'
            for change in ({'actually_viewed':False},{'image_sha256':'0'*64},{'C_use':None},{'tool':'inferred_from_score'}):
                try:attest({**row,**change},doc,expected,protocol['axes'],{},{})
                except ValueError:pass
                else:raise AssertionError('Invalid actual-view record passed')
    finally:OUT=saved_out
    print(json.dumps({'status':'PASS_SYNTHETIC_ONLY','checks':13,'actual_images_viewed':0,'actual_ratings_generated':0,'score_files_read':0}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:print(json.dumps(summarize(),ensure_ascii=False,indent=2))
