"""Finalize already computed source-mesh geometry audit; metadata/table arithmetic only.

No geometry scoring, model, OT, random sampling, or modifications of existing evidence.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))
def immutable(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        assert path.read_text(encoding="utf-8") == text, str(path) + " immutable mismatch"
    else:
        path.write_text(text, encoding="utf-8")
def csv_rows(name):
    with (ROOT/name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))
def quantiles(values):
    values=sorted(float(x) for x in values)
    def q(p):
        i=(len(values)-1)*p; a=math.floor(i); b=math.ceil(i)
        return values[a] + (values[b]-values[a])*(i-a)
    return dict(zip(["min","q25","median","q75","max"],[q(p) for p in [0,.25,.5,.75,1]]))
def table_write(name, rows):
    columns=list(rows[0])
    import io
    f=io.StringIO(newline="")
    w=csv.DictWriter(f,fieldnames=columns,lineterminator="\n")
    w.writeheader();w.writerows(rows)
    payload="\ufeff"+f.getvalue()
    path=ROOT/name
    if path.exists(): assert path.read_text(encoding="utf-8") == payload
    else: path.write_text(payload,encoding="utf-8",newline="")

def main():
    receipts = {s:read(ROOT/("receipt_"+s+".json")) for s in ["basic","intersections"]}
    assert receipts["basic"]["status"]=="BASIC_COMPLETE"
    assert receipts["intersections"]["status"]=="COMPLETE"
    for r in receipts.values():
        assert r["parents"]==26 and r["conditions"]==411
        assert all(r[k]==0 for k in ["GPU","OT_calls","model_calls","new_meshes"])
        assert not r["torch_imported"]
        for kind in ["input","source"]:
            before=r[kind+"_hashes_before"];after=r[kind+"_hashes_after"]
            assert before==after
            assert all(sha(Path(p))==h for p,h in before.items())
    parents=csv_rows("parents.csv"); conditions=csv_rows("conditions.csv")
    assert len(parents)==26 and len({r["uid"] for r in parents})==26
    assert len(conditions)==411 and len({r["task_id"] for r in conditions})==411
    assert Counter(r["ratio"] for r in conditions)=={"0.2":208,"0.4":203}
    by_parent={r["uid"]:r for r in parents}
    cache_rows=[]; unresolved=[]; class_totals=Counter()
    for row in parents:
        uid=row["uid"];N=int(row["N"])
        f=ROOT/"parent_intersections"/(uid+".json"); p=read(f)
        assert p["uid"]==uid and p["tested_pairs"]==N*(N-1)//2==p["expected_full_pairs"]
        assert sum(p["classification_counts"].values())==p["tested_pairs"]
        class_totals.update(p["classification_counts"])
        ids=[tuple(c["pair"]) for c in p["contacts"]]
        assert len(ids)==len(set(ids)) and all(0<=a<b<N for a,b in ids)
        counters=Counter(c["classification"] for c in p["contacts"])
        expected={k:v for k,v in p["classification_counts"].items() if k!="disjoint"}
        assert dict(counters)==expected
        for c in p["contacts"]:
            if c["classification"]=="UNRESOLVED":
                unresolved.append({"uid":uid,**c})
        cache_rows.append({
            "uid":uid,"N":N,"input_npz_sha256":p["input_npz_sha256"],
            "relative_path":str(f.relative_to(ROOT)).replace("\\","/"),
            "absolute_path":str(f),"sha256":sha(f),"bytes":f.stat().st_size,
            "L":p["L"],"tested_pairs":p["tested_pairs"],
            "classification_counts":p["classification_counts"],"status":p["status"],
        })
    assert sum(p["tested_pairs"] for p in cache_rows)==517381
    pair_label={"proper":"proper_transverse_penetration","coplanar_overlap":"coplanar_positive_area_overlap","unresolved":"UNRESOLVED"}
    for row in conditions:
        c=read(ROOT/"conditions"/(row["task_id"]+".json"))
        assert c["uid"]==row["uid"] and c["task_id"]==row["task_id"]
        N,K=int(row["N"]),int(row["K"]);ids=set(c["source_face_ids"]);free=set(c["free_source_ids"])
        assert len(ids)==K and len(free)==N-K and ids.isdisjoint(free) and ids|free==set(range(N))
        for detector in ["verified","primary","independent"]:
            for typ in pair_label:
                assert sum(int(row[f"{detector}_{s}_{typ}_pair_count"]) for s in ["CC","CF","FF"])==int(by_parent[row["uid"]][f"{detector}_{typ}_pair_count"])
            partitions=c["intersection_partitions"][detector]
            sums=Counter()
            for scope,expected in [("CC",K*(K-1)//2),("CF",K*(N-K)),("FF",(N-K)*(N-K-1)//2)]:
                p=partitions[scope]
                assert p["expected_pairs"]==sum(p["classification_counts"].values())==expected
                sums.update(p["classification_counts"])
                for key in ["proper_pairs","coplanar_positive_area_pairs","unresolved_pairs"]:
                    for a,b in p[key]:
                        assert (scope=="CC" and a in ids and b in ids) or (scope=="CF" and a in ids and b in free) or (scope=="FF" and a in free and b in free)
            if detector=="verified":
                parent_cache=next(p for p in cache_rows if p["uid"]==row["uid"])
                assert dict(sums)==parent_cache["classification_counts"]
        assert int(row["source_index_C_component_count"])==int(row["exact_coordinate_C_component_count"])==1

    parent_summary={"registered_parents":26,"retained_source_faces":sum(int(r["N"]) for r in parents),"unique_pair_tests":517381,
        "classification_counts":dict(class_totals),
        "finite_parents":sum(r["geometry_status"]=="PASS" for r in parents),
        "exact_zero_area_faces":sum(int(r["exact_zero_area_count"]) for r in parents),
        "exact_duplicate_extra_faces":sum(int(r["exact_duplicate_extra_faces"]) for r in parents),
        "near_degenerate_area_faces":sum(int(r["near_degenerate_area_count"]) for r in parents),
        "min_angle_lt5_faces":sum(int(r["minimum_angle_lt5_count"]) for r in parents),
        "min_angle_lt5_parents":sum(int(r["minimum_angle_lt5_count"])>0 for r in parents),
        "proper_parents":sum(int(r["verified_proper_pair_count"])>0 for r in parents),
        "coplanar_overlap_parents":sum(int(r["verified_coplanar_overlap_pair_count"])>0 for r in parents),
        "any_confirmed_conflict_parents":sum(int(r["verified_proper_pair_count"])+int(r["verified_coplanar_overlap_pair_count"])>0 for r in parents),
        "unresolved_parents":sum(int(r["verified_unresolved_pair_count"])>0 for r in parents),
        "source_index_component_count_distribution":dict(Counter(int(r["source_index_component_count"]) for r in parents)),
        "source_index_nonmanifold_parents":sum(int(r["source_index_nonmanifold_edge_count"])>0 for r in parents),
        "source_index_nonmanifold_edges":sum(int(r["source_index_nonmanifold_edge_count"]) for r in parents)}
    groups={}
    for name,rows in [("ALL",conditions),("C20",[r for r in conditions if r["ratio"]=="0.2"]),("C40",[r for r in conditions if r["ratio"]=="0.4"])]:
        g={"registered_conditions":len(rows),"K_quantiles":quantiles(r["K"] for r in rows),"area_fraction_parent_quantiles":quantiles(r["C_area_fraction_parent"] for r in rows),
            "one_edge_connected_C_count":sum(int(r["source_index_C_component_count"])==1 for r in rows),
            "C_with_boundary_branch_vertex":sum(int(r["source_index_C_boundary_branch_count"])>0 for r in rows),
            "C_with_nonmanifold_internal_edge":sum(int(r["source_index_C_nonmanifold_edge_count"])>0 for r in rows),
            "C_with_minimum_angle_lt5":sum(int(r["C_minimum_angle_lt5_count"])>0 for r in rows),
            "F_with_component_not_edge_attached_to_C":sum(int(r["source_index_unattached_F_component_count"])>0 for r in rows),
            "near_duplicate_NA":len(rows),"partition_stats":{}}
        for scope in ["CC","CF","FF"]:
            s={"conditions_with_confirmed_conflict":sum(int(r[f"verified_{scope}_proper_pair_count"])+int(r[f"verified_{scope}_coplanar_overlap_pair_count"])>0 for r in rows)}
            for typ in pair_label:
                key=f"verified_{scope}_{typ}_pair_count"
                s[typ+"_condition_count"]=sum(int(r[key])>0 for r in rows)
                s[typ+"_pair_sum_repeated_across_conditions"]=sum(int(r[key]) for r in rows)
            g["partition_stats"][scope]=s
        groups[name]=g
    same_parent=[]
    for uid in sorted(by_parent):
        for ratio in ["0.2","0.4"]:
            rows=[r for r in conditions if r["uid"]==uid and r["ratio"]==ratio]
            assert rows
            vals=[float(r["C_area_fraction_parent"]) for r in rows]
            assert len(set(r["K"] for r in rows))==1
            q=quantiles(vals)
            same_parent.append({"uid":uid,"ratio":ratio,"K":int(rows[0]["K"]),"registered_conditions":len(rows),
              **{"C_area_fraction_"+k:v for k,v in q.items()},"max_minus_min":max(vals)-min(vals),
              "max_div_min":max(vals)/min(vals),
              "minimum_task_id":min(rows,key=lambda r:float(r["C_area_fraction_parent"]))["task_id"],
              "maximum_task_id":max(rows,key=lambda r:float(r["C_area_fraction_parent"]))["task_id"]})
    table_write("same_parent_area_ranges.csv",same_parent)
    graph_comparison={}
    for graphscope,rows,prefix in [("parent",parents,""),("C",conditions,"C_"),("F",conditions,"F_")]:
        fields=[prefix+s for s in ["component_count","nonmanifold_edge_count","boundary_branch_count","boundary_edge_count","boundary_length","smallest_component_area_fraction"]]
        graph_comparison[graphscope]={s:sum(r["source_index_"+s]!=r["exact_coordinate_"+s] for r in rows) for s in fields}
    limitations=[
        "Static source geometry audit, not evaluation of generated outputs or proof of the cause of training failures.",
        "All C are edge-connected; one component does not establish semantic usefulness, compactness, a simple disk, or a user-like intended surface.",
        "Multiple complete-parent components and F components unattached by a complete edge can be valid separately assembled parts; counts are not rejection criteria.",
        "Five-degree minimum-angle diagnostic is inherited and not a validity threshold; zero area and skinny triangles are different.",
        "Verified conflict requires detector agreement. Three contact-subtype disagreements remain UNRESOLVED even though both classifiers label each illegal=False.",
        "Near-duplicate query metric is NA for all 411 conditions: no registered query bank exists; no query points were sampled.",
        "L is fixed sqrt(3)*2/.3762 for every parent and condition, not re-estimated from C or each parent.",
        "CC/CF/FF pair sums across conditions repeatedly count source pairs; they are not counts of independent defects.",
        "Exact coordinate connectivity uses stored float32 model coordinates without tolerance welding; source-index connectivity preserves original vertex IDs.",
        "The C20 and C40 source selector pools have known construction-policy differences; this descriptive audit does not turn their comparison into a pure ratio causal experiment.",
    ]
    summary={"schema":"chair_user_condition_geometry_audit_summary_v1","status":"COMPLETE_WITH_3_UNRESOLVED_PAIRS",
      "parents":parent_summary,"conditions":groups,"index_vs_exact_coordinate_field_differences":graph_comparison,
      "same_parent_largest_absolute_area_ranges":sorted(same_parent,key=lambda r:r["max_minus_min"],reverse=True)[:6],
      "same_parent_largest_area_ratios":sorted(same_parent,key=lambda r:r["max_div_min"],reverse=True)[:6],
      "unresolved_pairs":unresolved,"limitations":limitations,
      "resource_contract":{"processes":1,"numerical_threads":2,"GPU":0,"model_calls":0,"OT_calls":0,"new_meshes":0},
      "wall_seconds":{k:r["wall_seconds"] for k,r in receipts.items()},
      "peak_rss_kib":max(r["max_rss_kib"] for r in receipts.values()),
      "validation":"26 parent and 411 condition identities; all three detector partitions and pair counts; source/coordinate mapping was verified during scan; protected hashes rechecked during finalization",
      "summary_source_sha256":sha(Path(__file__)) }
    immutable(ROOT/"summary.json",summary)
    immutable(ROOT/"pair_cache_manifest.json",{
      "schema":"chair_source_parent_triangle_pair_cache_v1","status":"COMPLETE","parent_count":26,"tested_unique_pairs":517381,
      "receipt_path":"receipt_intersections.json","receipt_sha256":sha(ROOT/"receipt_intersections.json"),
      "face_id_contract":"pair entries use retained source-face row IDs (not raw original-face IDs); original_face_ids is included separately.",
      "contact_contract":"Each unique pair actually tested once by each detector. Contacts excludes only verified disjoint; primary and independent classifications preserved. UNRESOLVED is not confirmed illegal.",
      "no_rescan_required_for_new_face_subset":True,"parents":cache_rows})
    lines=["# Source geometry audit", "",
      "Completed: 26 retained training parents / 5122 faces, 411 existing conditions (208 C20 + 203 C40).",
      "This report summarizes existing numerical results only. No model, training, OT, mesh repair or new mesh was run.", "",
      f"All 411 C are one complete-edge-connected component under source-index and exact-coordinate definitions. Branching C boundaries: {groups['ALL']['C_with_boundary_branch_vertex']}/411; C internal nonmanifold edges: {groups['ALL']['C_with_nonmanifold_internal_edge']}/411.",
      "No exact zero-area or duplicate retained faces. Minimum angle below 5 degrees: 1367/5122 faces, distributed over all 26 parents. This diagnostic does not by itself make a face invalid.",
      "The 517381 unique source-parent triangle pairs contain 157 verified proper penetration pairs (10 parents), 153 verified positive-area coplanar overlap pairs (6 parents), and 3 unresolved contact-subtype disagreements. There are 14 parents with either verified conflict.",
      "Conditions with a verified conflict: CC 20/411, CF 101/411, FF 178/411; these subsets overlap and cannot be added. F with at least one component not connected to C by a complete edge: 112/411.", "",
      "| Pool | Conditions | K min / median / max | C area fraction min / q25 / median / q75 / max |",
      "|---|---:|---|---|"]
    for name in ["C20","C40"]:
        g=groups[name]; k=g["K_quantiles"]; a=g["area_fraction_parent_quantiles"]
        lines.append(f"| {name} | {g['registered_conditions']} | {k['min']:g} / {k['median']:g} / {k['max']:g} | "+" / ".join(f"{a[x]:.6f}" for x in ["min","q25","median","q75","max"])+" |")
    lines+=["", "Full retained face IDs, component memberships, per-face areas/angles and all classifier details remain in the parent/condition JSON files. CSV summaries preserve all 26/411 registered cases.",
      "See same_parent_area_ranges.csv for all 52 parent/ratio groups; fixed K can correspond to very different fractions of surface area.", "",
      "## Limitations", ""]
    lines+=["- "+x for x in limitations]
    lines+=["", "## Reuse and execution", "",
      "pair_cache_manifest.json binds all 26 full pair caches by SHA256 and records the face-ID convention; reuse it instead of rescanning for subsets.",
      "data_audit.py is the resumable entry with --stage basic, intersections, or all. Existing results are immutable and inputs/source hashes must still match.",
      "12 CPU synthetic tests passed; tests.stdout.txt and tests.xml contain the original run. Actual scoring source is preserved under source_snapshot/.",
      "Source/input hashes before and after match. Basic stage 11.865 seconds; pair stage 82.972 seconds. Peak RSS 127568 KiB; one CPU process with two numerical threads.", ""]
    report="\n".join(lines)
    if (ROOT/"SUMMARY.md").exists():assert (ROOT/"SUMMARY.md").read_text()==report
    else:(ROOT/"SUMMARY.md").write_text(report,encoding="utf-8")
    files=[]
    for f in sorted(ROOT.rglob("*")):
        if f.is_file() and f.name!="artifact_manifest.json" and "__pycache__" not in f.parts and not f.name.endswith(".tmp"):
            files.append({"path":f.relative_to(ROOT).as_posix(),"bytes":f.stat().st_size,"sha256":sha(f)})
    immutable(ROOT/"artifact_manifest.json",{
      "schema":"chair_user_condition_geometry_artifact_manifest_v1","status":"COMPLETE",
      "parents":26,"conditions":411,"pair_cache_parents":26,"files":files,
      "excluded":"Manifest itself excluded to avoid circular hash; no original files modified."})
    print(json.dumps({"status":summary["status"],"file_count":len(files),"summary_sha256":sha(ROOT/"summary.json"),"pair_cache_manifest_sha256":sha(ROOT/"pair_cache_manifest.json"),
      "artifact_manifest_sha256":sha(ROOT/"artifact_manifest.json"),"parent_summary":parent_summary,
      "ratio_summaries":{k:{"K":v["K_quantiles"],"area":v["area_fraction_parent_quantiles"]} for k,v in groups.items()}},ensure_ascii=False))
if __name__=="__main__":main()
