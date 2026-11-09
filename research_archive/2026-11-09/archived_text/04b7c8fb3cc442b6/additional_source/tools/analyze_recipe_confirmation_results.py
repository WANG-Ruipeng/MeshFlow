#!/usr/bin/env python3
"""Fixed selected-pair independent-input repeat: saved tables and hashes only.

No Torch, model, geometry scorer, sampler, RNG, automatic selection or metric
redefinition. The two experiments use different noise sets: cross-experiment
comparisons are parent/four-noise effects, never output-level noise matches.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import importlib.util
import itertools
import hashlib
import json
from pathlib import Path
import statistics

_spec = importlib.util.spec_from_file_location("_confirmation_course_tables", Path(__file__).with_name("analyze_recipe_curriculum_results.py"))
course = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(course)
base = course.base
VERSION = "recipe_confirmation_table_analysis_v1"
CANDIDATE, CONTROL = "C20_40_MIX", "R2_MIX_H_ALL"
LABELS = (CANDIDATE, CONTROL)
PAIRS = ((CANDIDATE, CONTROL),)
NOISES, FIRST_NOISES = (8, 9, 10, 11), (0, 1, 6, 7)
EXPERIMENT = "CHAIR_CONDITION_RECIPE_FACTORIAL_V1"
NOISE_NAMESPACE = EXPERIMENT + "/confirmation_noise/v1"
SCOPES = ("ALL", "ORIGINAL6", "EXTENDED2")
FROZEN_TABLE_SOURCES = {
    "analyze_recipe_factorial_results.py":"f8750d36d240256966abbb56bedddabdae61e4a1f61ff8c94bce73fe80598957",
    "analyze_recipe_curriculum_results.py":"9176a352f3c75c55dec6bcfa9989826bde151ac5a1e8f34840b9753558ad6256",
}
METRICS = (
"CF_coplanar_pair_count","CF_proper_affected_free_fraction","CF_proper_pair_count","CF_unresolved_or_degenerate_pair_count",
"C_exact_zero_area_count","C_near_degenerate_count","C_total_surface_area","ConflictCov_C","ConflictCov_C_coplanar_illegal_pairs",
"ConflictCov_C_cross","ConflictCov_C_cross_and_dup","ConflictCov_C_dup","ConflictCov_C_interior_samples",
"ConflictCov_C_interior_weight_fraction","ConflictCov_C_proper_pairs","ConflictCov_C_segment_extraction_failures",
"ConflictCov_C_status","ConflictCov_C_uncertain_fraction","ConflictCov_C_unresolved_pairs","ConflictCov_C_upper",
"EXACT_COORD_CF_shared_edges","EXACT_COORD_boundary_edges","EXACT_COORD_closed_two_faces_per_edge","EXACT_COORD_edges",
"EXACT_COORD_face_components","EXACT_COORD_nonmanifold_edges","EXACT_COORD_same_direction_two_face_edges","EXACT_COORD_vertices",
"FF_coplanar_pair_count","FF_illegal_affected_free_fraction","FF_illegal_pair_count","FF_proper_affected_free_fraction",
"FF_proper_pair_count","FF_unresolved_or_degenerate_pair_count","RAW_SOUP_CF_shared_edges","RAW_SOUP_face_components",
"free_angle_p05_degrees","free_exact_duplicate_excess_fraction","free_exact_zero_area_count","free_exact_zero_area_fraction",
"free_minimum_angle_lt5_fraction","free_near_degenerate_count","free_near_degenerate_fraction","free_q_gt10_fraction",
"free_q_mean","free_total_surface_area","whole_minimum_angle_lt5_fraction","whole_near_degenerate_count","whole_total_surface_area")
NONNUMERIC = {"ConflictCov_C_status", "EXACT_COORD_closed_two_faces_per_edge"}
number, distribution, summarize = base.is_number, base.distribution, course.summarize
read_json, read_csv, digest = base.read_json, base.read_csv, base.digest
write_json, write_csv = base.write_json, base.write_csv

def require(ok, message):
    if not ok:
        raise ValueError(message)

def fixed_catalog(rows):
    require(not ({k for r in rows for k in r["metrics"]} - set(METRICS)), "Unregistered metric field")
    present = {r["metric"]:r for r in base.metric_catalog(rows)}
    result = []
    for metric in METRICS:
        item = present.get(metric, dict(metric=metric, registered_outputs=len(rows), present=0,
            missing_field=len(rows), value_status_counts={}, direction=base.DIRECTIONS.get(metric,"descriptive_no_universal_direction")))
        # Keep the fixed numeric denominator even if the entire field is NA.
        item["numeric_summary"] = metric not in NONNUMERIC
        item["declared_status_counts"] = dict(Counter(str(r.get("metric_status",{}).get(metric,"UNSPECIFIED")) for r in rows))
        result.append(item)
    return result

def validate_requests(requests, noises):
    require(len(requests)==64, "Exactly64 requests required")
    by = {q["request_id"]:q for q in requests}
    require(len(by)==64, "Duplicate request")
    groups = defaultdict(list)
    for q in requests:
        groups[q["parent_id"],q["ratio"]].append(q)
    parents={q["parent_id"] for q in requests}
    require(len(parents)==8 and set(groups)=={(p,r)for p in parents for r in (.2,.4)}, "Eight parents at both ratios")
    for (parent,ratio),group in groups.items():
        require(len(group)==4 and {q["candidate"]for q in group}==set(noises), "Wrong registered noise set")
        for field in ("condition_id","condition_cohort","N","K","C_sha256"):
            require(len({q[field]for q in group})==1, "Condition differs across noises: "+field)
        require(all(q["N"]>q["K"]>0 for q in group), "Invalid free denominator")
    course.membership(requests)
    return by

def validate_scored_rows(rows, requests, query_registration, request_sha, noises):
    qby=validate_requests(requests,noises)
    require(query_registration.get("status")=="FROZEN","Query registration not frozen")
    require(query_registration.get("identity",{}).get("requests_sha256")==request_sha,"Query request hash mismatch")
    qq=query_registration["rows"]
    queries={q["condition_id"]:q for q in qq}
    require(len(qq)==16 and set(queries)=={q["condition_id"]for q in requests},"Exactly16 query conditions")
    expected={(label,rid)for label in LABELS for rid in qby}
    seen=set()
    for row in rows:
        key=row["label"],row["request_id"]
        require(key in expected and key not in seen,"Unknown or duplicate scored cell")
        seen.add(key);q=qby[key[1]];query=queries[q["condition_id"]]
        for field in ("parent_id","condition_id","candidate","ratio","condition_cohort","N","K"):
            require(row.get(field)==q[field],"Scored request differs: "+field)
        require(row.get("status")=="PASS" and bool(row.get("contracts")) and all(v is True for v in row["contracts"].values()),"Failed scoring/output contract")
        for field in ("condition_id","parent_id","ratio","C_sha256"):
            require(query[field]==q[field],"Query C binding differs: "+field)
        require(query.get("query_count")==2048 and query.get("all_four_noises_share_same_bank")is True,"Query support/noise sharing differs")
        identity=row["identity"]
        for field,value in (("request_id",q["request_id"]),("label",row["label"]),("input_sha256",q["input_sha256"]),("query_sha256",query["query_sha256"])):
            require(identity.get(field)==value,"Inner scored identity differs: "+field)
        for field,value in (("query_sha256",query["query_sha256"]),("registered_C_queries",2048),
                            ("registered_free_faces",q["N"]-q["K"]),("query_weight_denominator",query["weight_denominator"]),("L0",query["identity"]["L0"])):
            require(row.get(field)==value,"Scored denominator/query identity differs: "+field)
    require(seen==expected and len(rows)==128,"Complete128 selected-pair outputs required")
    fixed_catalog(rows)
    return rows

def validate_holdout(requests, old_requests, plan, collision, authorization, registration):
    validate_requests(requests,NOISES);validate_requests(old_requests,FIRST_NOISES)
    require(authorization.get("schema")=="chair_recipe_confirmation_training_authorization_v1" and authorization.get("stage")=="independent_confirmation","Independent confirmation authorization required")
    require(authorization.get("experiment")==EXPERIMENT,"Wrong experiment")
    require(authorization.get("selected_pair")==list(LABELS) and authorization.get("authorized_recipes")==list(LABELS),"Wrong explicitly selected pair")
    require(authorization.get("hybrid_schedules")=={r:"all"for r in LABELS},"H_ALL confirmation required")
    require(authorization.get("updates_per_recipe")==6000 and authorization.get("effective_batch")==8 and authorization.get("microbatch")==1,"Training budget differs")
    require(str(authorization.get("namespace","")).startswith(EXPERIMENT+"/confirmation/"),"Independent input namespace absent")
    require(registration.get("kind")=="confirm" and registration.get("labels")==list(LABELS) and registration.get("expected_outputs")==128,"Wrong confirmation registration")
    require(registration.get("new_parent_generalization")is False,"SameDEV limitation missing")
    require(plan.get("namespace")==NOISE_NAMESPACE and plan.get("candidate_model_identity_in_seed")is False and plan.get("expected_outputs")==128,"Frozen holdout plan differs")
    require(collision.get("status")=="PASS","Actual holdout collision audit must pass")
    planned={(q["parent_id"],q["ratio"],q["candidate"]):q for q in plan["rows"]}
    require(len(planned)==64 and len(plan["rows"])==64,"Holdout plan coverage")
    old={(q["parent_id"],q["ratio"]):q for q in old_requests}
    audit={q["request_id"]:q for q in collision["rows"]}
    require(len(audit)==64 and len(collision["rows"])==64,"Collision audit coverage")
    old_gaussian={q.get("Gaussian_sha256")for q in old_requests}
    old_input={q["input_sha256"]for q in old_requests}
    for q in requests:
        key=q["parent_id"],q["ratio"],q["candidate"]
        require(key in planned and key[:2]in old and q["request_id"]in audit,"Unknown holdout identity")
        p=planned[key];o=old[key[:2]];a=audit[q["request_id"]]
        for field in ("condition_id","N","K","C_sha256"):
            require(q[field]==p[field]==o[field],"Changed holdout condition: "+field)
        require(q["condition_cohort"]==o["condition_cohort"],"Changed parent membership")
        seed_text = NOISE_NAMESPACE+"/"+q["parent_id"]+"/n"+str(q["candidate"]-8)
        seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4],"little")
        require(p["seed_text"]==seed_text and p["seed"]==seed,"Noise namespace not bound to actual seed")
        require(q.get("noise_seed")==seed and q.get("noise_seed_text")==seed_text,"Holdout seed differs")
        for hash_field in ("Gaussian_sha256","noise_full_N_Gaussian_sha256","input_sha256"):
            value=q.get(hash_field)
            require(isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef"for c in value),"Invalid holdout hash: "+hash_field)
        require(q.get("new_parent_generalization")is False,"Holdout falsely claims new parents")
        require(a.get("status")=="PASS" and a.get("Gaussian_sha256")==q.get("Gaussian_sha256")and q.get("Gaussian_sha256")not in old_gaussian,"Actual holdout Gaussian identity differs")
        require(q["input_sha256"]not in old_input,"Holdout reuses old input bytes")
    require({q["request_id"]for q in requests}==set(audit),"Collision extra request")
    actual_old={q["request_id"]:q for q in collision.get("main_input_files",[])}
    require(len(actual_old)==64 and len(collision.get("main_input_files",[]))==64,"Collision original input coverage")
    for q in old_requests:
        require(q["request_id"]in actual_old,"Collision original request absent")
        a=actual_old[q["request_id"]]
        require(a["input_sha256"]==q["input_sha256"]and a["Gaussian_sha256"]==q["Gaussian_sha256"],"Collision original hash binding")
    full=defaultdict(set)
    for q in requests:
        full[q["parent_id"],q["candidate"]].add(q["noise_full_N_Gaussian_sha256"])
        require(audit[q["request_id"]].get("full_N_Gaussian_sha256")==q["noise_full_N_Gaussian_sha256"],"Collision full N hash differs")
    require(len(full)==32 and all(len(v)==1 for v in full.values()),"Full N draw must be shared across ratios")
    require(len({next(iter(v))for v in full.values()})==32,"Duplicate full N holdout draw")
    return True

def check_receipt(receipt, count):
    require(receipt.get("status")=="COMPLETE" and receipt.get("registered")==count and receipt.get("scored")==count,"Scoring receipt incomplete")

def checked_file(path, expected, files, name):
    p=base.resolve(path)
    require(p.is_file(),"Registered file missing: "+str(p))
    require(digest(p)==expected,"Actual registered file hash differs: "+name)
    files[name]=p
    return p

def validate_diversity(condition_rows, pair_rows, scored_rows, noises):
    expected={(r["label"],r["condition_id"]):(r["parent_id"],r["ratio"],r["condition_cohort"])for r in scored_rows}
    seen=set();pairs=defaultdict(set)
    for row in condition_rows:
        key=row["label"],row["condition_id"]
        require(key in expected and key not in seen,"Diversity condition duplicate/unknown")
        parent,ratio,cohort=expected[key]
        require((row["parent_id"],float(row["ratio"]),row["condition_cohort"])==(parent,ratio,cohort),"Diversity condition metadata")
        require(int(row["expected_pairs"])==6,"Diversity fixed six-pair denominator")
        seen.add(key)
    for row in pair_rows:
        key=row["label"],row["condition_id"];pair=int(row["candidate_a"]),int(row["candidate_b"])
        require(key in expected and pair not in pairs[key],"Diversity pair duplicate/unknown")
        require((row["parent_id"],float(row["ratio"]))==expected[key][:2],"Diversity pair condition metadata")
        pairs[key].add(pair)
    wanted=set(itertools.combinations(noises,2))
    require(seen==set(expected) and set(pairs)==set(expected) and all(v==wanted for v in pairs.values()),"Incomplete or incorrect six noise pairs")


def tables(rows, requests, cohort, source="primary_per_output", noises=NOISES):
    catalog = fixed_catalog(rows)
    for r in catalog:
        r["source_table"] = source
        r["source_GT_used_for_promotion"] = False
        if source != "primary_per_output":
            r["direction"] = "reference_diagnostic_not_a_promotion_gate"
    metrics = [r["metric"] for r in catalog if r["numeric_summary"]]
    by = {(r["label"], r["request_id"]): r for r in rows}
    groups = defaultdict(list)
    for q in requests:
        groups[q["parent_id"], q["ratio"]].append(q)
    parents = []
    for label in LABELS:
        for (pid, ratio), qs in sorted(groups.items()):
            qs = sorted(qs, key=lambda q: q["candidate"])
            for metric in metrics:
                values = [by[label, q["request_id"]]["metrics"].get(metric) for q in qs]
                parents.append(dict(label=label, parent_id=pid, ratio=ratio,
                    condition_cohort=qs[0]["condition_cohort"], parent_membership=cohort[pid],
                    metric=metric, noises=list(noises), noise_values=values, **distribution(values, 4)))
    summary = [dict(key, aggregation="four noises then equal parents",
                    **distribution([r["mean"] for r in group], len(group)))
               for key, group in summarize(parents, ("label", "ratio", "metric"))]
    pby = {(r["label"], r["parent_id"], r["ratio"], r["metric"]): r for r in parents}
    noise, paired = [], []
    for first, second in PAIRS:
        for q in requests:
            a, b = by[first, q["request_id"]], by[second, q["request_id"]]
            for metric in metrics:
                av, bv = a["metrics"].get(metric), b["metrics"].get(metric)
                noise.append(dict(first=first, second=second, request_id=q["request_id"],
                    parent_id=q["parent_id"], ratio=q["ratio"], condition_cohort=q["condition_cohort"],
                    parent_membership=cohort[q["parent_id"]], candidate=q["candidate"], metric=metric,
                    first_value=av, second_value=bv, delta=av-bv if number(av) and number(bv) else None))
        for (pid, ratio), qs in sorted(groups.items()):
            for metric in metrics:
                a, b = pby[first,pid,ratio,metric], pby[second,pid,ratio,metric]
                dd = [av-bv if number(av) and number(bv) else None for av,bv in zip(a["noise_values"], b["noise_values"])]
                paired.append(dict(first=first, second=second, parent_id=pid, ratio=ratio,
                    condition_cohort=qs[0]["condition_cohort"], parent_membership=cohort[pid], metric=metric,
                    first_mean=a["mean"], second_mean=b["mean"], noise_deltas=dd,
                    delta=a["mean"]-b["mean"] if number(a["mean"]) and number(b["mean"]) else None,
                    **{"noise_"+k:v for k,v in distribution(dd,4).items()}))
    paired_summary = []
    for key, group in summarize(paired, ("first", "second", "ratio", "metric")):
        dd = [r["delta"] for r in group]
        direction = base.DIRECTIONS.get(key["metric"], "descriptive") if source == "primary_per_output" else "reference_diagnostic"
        item = dict(key, delta_direction="first minus second", nominal_preferred_direction=direction,
                    parent_deltas=dd, **distribution(dd,len(group)))
        if all(number(x) for x in dd):
            item.update(minimum_parent_id=min(group,key=lambda r:r["delta"])["parent_id"],
                        maximum_parent_id=max(group,key=lambda r:r["delta"])["parent_id"])
            leave = [statistics.fmean(dd[:i]+dd[i+1:]) for i in range(len(dd))]
            item.update(leave_one_parent_out_min=min(leave), leave_one_parent_out_max=max(leave))
            if direction in ("lower", "lower_uncertainty", "higher"):
                benefit = sorted((max(0,(-1 if direction.startswith("lower") else 1)*v) for v in dd),reverse=True)
                total = sum(benefit)
                item.update(top1_share_of_gross_improvement=benefit[0]/total if total else None,
                            top2_share_of_gross_improvement=sum(benefit[:2])/total if total else None)
        paired_summary.append(item)
    noise_summary = []
    for key, group in summarize(noise, ("first", "second", "ratio", "metric")):
        values = [r["delta"] for r in group]
        record = dict(key, delta_direction="first minus second",
                      **distribution(values, len(group)))
        if all(number(v) for v in values):
            record.update(minimum_request_id=min(group,key=lambda r:r["delta"])["request_id"],
                          maximum_request_id=max(group,key=lambda r:r["delta"])["request_id"])
        noise_summary.append(record)
    return dict(metric_catalog=catalog, per_parent_metrics=parents, model_summary=summary,
                paired_noise_metrics=noise, paired_noise_summary=noise_summary,
                paired_parent_metrics=paired, paired_parent_summary=paired_summary)


def diversity_tables(conditions, pairs, rows, cohort, noises=NOISES):
    validate_diversity(conditions,pairs,rows,noises)
    grouped = defaultdict(list)
    for r in pairs:
        grouped[r["label"],r["condition_id"]].append(r)
    checked = []
    for r in conditions:
        pp = grouped[r["label"],r["condition_id"]]
        valid = [float(p["D_free_pair"]) for p in pp if p.get("status")=="PASS" and p.get("D_free_pair") not in (None,"") and number(float(p["D_free_pair"]))]
        if int(r["available_pairs"]) != len(valid):
            raise ValueError("Diversity available-pair count differs")
        saved = float(r["D_free_mean"]) if r.get("D_free_mean") not in (None,"") else None
        if len(valid)==6:
            if not number(saved) or abs(statistics.fmean(valid)-saved)>max(1e-12,abs(saved)*1e-10):
                raise ValueError("Diversity six-pair mean differs")
        elif saved is not None:
            raise ValueError("Incomplete diversity must not use a survivor mean")
        checked.append(dict(r, ratio=float(r["ratio"]), parent_membership=cohort[r["parent_id"]], value=saved))
    summary = [dict(key, metric="D_free_mean", direction="variation_not_larger_is_better",
                    **distribution([r["value"] for r in group],len(group)))
               for key,group in summarize(checked,("label","ratio"))]
    by = {(r["label"],r["parent_id"],r["ratio"]):r for r in checked}
    parents = []
    for first,second in PAIRS:
        for (label,pid,ratio),a in by.items():
            if label!=first: continue
            b=by[second,pid,ratio];av,bv=a["value"],b["value"]
            parents.append(dict(first=first,second=second,parent_id=pid,ratio=ratio,
                condition_cohort=a["condition_cohort"],parent_membership=cohort[pid],
                first_D_free=av,second_D_free=bv,delta=av-bv if number(av) and number(bv) else None,
                relative_change=av/bv-1 if number(av) and number(bv) and bv!=0 else None,
                direction="variation_not_larger_is_better"))
    paired = [dict(key, direction="variation_not_larger_is_better",
                   **distribution([r["delta"] for r in group],len(group)))
              for key,group in summarize(parents,("first","second","ratio"))]
    return dict(diversity_per_condition=conditions,diversity_pairs=pairs,diversity_summary=summary,
                diversity_paired_parents=parents,diversity_paired_summary=paired)



def flattened(rows):
    result=course.flattened(rows)
    for row in result:
        for metric in METRICS:
            row.setdefault("metric."+metric,None)
    return result

def complete_loo(paired):
    result=[]
    for key,group in summarize(paired,("first","second","ratio","metric")):
        for omitted in group:
            remaining=[r for r in group if r["parent_id"]!=omitted["parent_id"]]
            result.append(dict(key,omitted_parent_id=omitted["parent_id"],
                remaining_parent_ids=[r["parent_id"]for r in remaining],
                **distribution([r["delta"]for r in remaining],len(group)-1)))
    return result

def sign_relation(first,repeat):
    if not number(first) or not number(repeat):return "NA"
    if first==0 or repeat==0:
        return "both_equal_zero" if first==repeat else "one_equal_zero"
    if first<0 and repeat<0:return "both_lower"
    if first>0 and repeat>0:return "both_higher"
    return "direction_reversal"

def compare_effects(first,repeat,keys,value):
    """Compare matched parent/group effects, never individual noises across runs."""
    old={tuple(r[k]for k in keys):r for r in first}
    new={tuple(r[k]for k in keys):r for r in repeat}
    require(len(old)==len(first) and len(new)==len(repeat) and old.keys()==new.keys(),
            "Cross-repeat effect coverage differs")
    result=[]
    for key in sorted(old):
        a,b=old[key],new[key];av,bv=a[value],b[value]
        result.append(dict(dict(zip(keys,key)),first=CANDIDATE,second=CONTROL,
            first_stage_effect=av,independent_repeat_effect=bv,
            effect_change=bv-av if number(av)and number(bv)else None,
            direction_relation=sign_relation(av,bv),first_stage_noises=list(FIRST_NOISES),
            repeat_noises=list(NOISES),cross_experiment_output_pairing=False,
            first_stage_expected=a.get("expected",a.get("noise_expected")),
            first_stage_available=a.get("available",a.get("noise_available")),
            repeat_expected=b.get("expected",b.get("noise_expected")),
            repeat_available=b.get("available",b.get("noise_available")),
            interpretation="Descriptive repeat with new training input and evaluation noises; same eight DEV parents"))
    return result

def status_tables(rows,cohort):
    grouped=defaultdict(list)
    for row in rows:
        for scope in SCOPES:
            if scope=="ALL" or cohort[row["parent_id"]]==scope:
                grouped[row["label"],row["ratio"],scope].append(row)
    result=[]
    for (label,ratio,scope),group in sorted(grouped.items()):
        for metric in METRICS:
            statuses=Counter()
            for row in group:
                if metric not in row["metrics"]:v="MISSING_FIELD"
                elif row["metrics"][metric]is None:v="NA_NULL"
                elif number(row["metrics"][metric]):v="NUMERIC"
                else:v=json.dumps(row["metrics"][metric],sort_keys=True)
                statuses[v]+=1
            result.append(dict(label=label,ratio=ratio,parent_stratum=scope,metric=metric,
                registered_outputs=len(group),value_status_counts=dict(statuses),
                declared_status_counts=dict(Counter(str(r.get("metric_status",{}).get(metric,"UNSPECIFIED"))for r in group))))
    return result

def verify_sources():
    for name,sha in FROZEN_TABLE_SOURCES.items():
        require(digest(Path(__file__).with_name(name))==sha,"Frozen table arithmetic source changed: "+name)

def metadata_bindings(out,files,regs):
    """Read/hash registrations, small receipts and input/query files; no scores."""
    for phase,reg in regs.items():
        files[phase+"_requests"]=checked_file(reg["requests_path"],reg["requests_sha256"],files,phase+"_requests")
    require(regs["main"]["requests_sha256"]==regs["course"]["requests_sha256"],"First-stage request mismatch")
    conf=regs["confirm"];p=out/"phases/confirm"
    binding=read_json(checked_file(p/"preparation_binding.json",conf["preparation_binding_sha256"],files,"preparation_binding"))
    require(binding.get("selected_pair")==list(LABELS) and binding.get("phase")=="confirm","Preparation pair/phase differs")
    for key,item in binding["files"].items():
        checked_file(item["path"],item["sha256"],files,"bound_"+key)
    repo=Path(__file__).resolve().parent.parent
    for relative,sha in binding["sources"].items():
        checked_file(repo/relative,sha,files,"evaluation_source_"+relative)
    auth=read_json(files["bound_authorization"])
    require(binding["training_namespace"]==auth["namespace"],"Preparation training namespace differs")
    for relative,sha in auth["source_hashes"].items():
        checked_file(repo/relative,sha,files,"training_source_"+relative)
    requests_doc=read_json(files["confirm_requests"]);requests=requests_doc["rows"]
    old_requests=read_json(files["main_requests"])["rows"]
    require(digest(files["bound_main_requests"])==regs["main"]["requests_sha256"],"Bound original requests changed")
    plan=read_json(files["bound_noise_plan"])
    require(requests_doc.get("noise_plan_sha256")==digest(files["bound_noise_plan"]) and
            requests_doc.get("authorization_sha256")==digest(files["bound_authorization"]),"Request plan/authorization binding differs")
    require(requests_doc.get("holdout_noise_collision_audit_passed")is True and
            requests_doc.get("Gaussian_full_N_draws")==32,"Frozen holdout draw/collision declaration differs")
    collision=read_json(checked_file(p/"evaluation/holdout_noise_collision_audit.json",conf["collision_audit_sha256"],files,"collision_audit"))
    validate_holdout(requests,old_requests,plan,collision,auth,conf)
    queries={}
    for phase in ("main","confirm"):
        path=out/"evaluation/query_registrations"/(regs[phase]["requests_sha256"]+".json")
        expected=conf["query_registration_sha256"]if phase=="confirm"else digest(files["bound_main_queries"])
        files[phase+"_queries"]=checked_file(path,expected,files,phase+"_queries")
        queries[phase]=read_json(path)
    require(queries["confirm"]["rows"]==queries["main"]["rows"],"New query arrays/registration rows prohibited")
    for i,query in enumerate(queries["main"]["rows"]):
        checked_file(query["query_path"],query["query_sha256"],files,"query_bank_"+str(i))
    for phase,qs in (("main",old_requests),("confirm",requests)):
        for q in qs:
            checked_file(q["input_path"],q["input_sha256"],files,phase+"_input_"+q["request_id"])
    models=conf.get("models",[])
    require(len(models)==2 and [m["label"]for m in models]==list(LABELS),"Confirmation model registration differs")
    for m in models:
        require(m.get("recipe")==m["label"] and m.get("loader")=="recipe_confirmation","Confirmation model recipe/loader differs")
        expected=base.resolve(auth["out"])/"inference"/(m["label"]+"_generator_step6000.pt")
        require(base.resolve(m["checkpoint_path"]).resolve()==expected.resolve(),"Confirmation endpoint path differs")
    return requests,old_requests,queries,auth

def checkpoint_bindings(out,files,regs,rows_by_phase):
    # Weights are not reopened: validate the pre-sampling frozen small receipts.
    for phase,rows in rows_by_phase.items():
        reg=regs[phase]
        for label in {r["label"]for r in rows}:
            path=out/"phases"/phase/"checkpoints"/(label+".json")
            binding=read_json(path);files[phase+"_checkpoint_"+label]=path
            model=next(m for m in reg["models"]if m["label"]==label)
            require(all(binding.get(k)==v for k,v in model.items()),"Checkpoint model registration differs")
            require(binding["registration_sha256"]==digest(files[phase+"_registration"]),"Checkpoint registration hash differs")
            require(all(r["identity"].get("checkpoint_sha256")==binding["checkpoint_sha256"]for r in rows if r["label"]==label),
                    "Scored checkpoint identity differs")

def readme():
    return """# Independent input-stream confirmation: saved scalar evidence

Fixed candidate C20_40_MIX minus matched control R2_MIX_H_ALL. This analyzer does not choose or promote a model.
The confirmation panel is 128 outputs: two models, eight previously observed DEV parents, two ratios, and n8/n9/n10/n11. The first-stage reference uses the same conditions with n0/n1/n6/n7. Neither this repeat nor the noise split establishes new-parent generalization.

All 49 registered geometry fields are retained, including I components and two nonnumeric fields. Forty-seven fixed numeric columns have per-noise, per-parent, paired and stratum tables, including all-NA columns. Four noises are averaged first, then parents are equally weighted. ALL/ORIGINAL6/EXTENDED2 are reported separately at both ratios; C20 inherits C40 parent membership. A missing observation preserves the registered denominator and makes the complete mean NA; available_mean is explicitly separate.

Differences are always candidate minus control. Lower/equal/higher counts are descriptive and are not wins or significance. Parent tails, complete leave-one-parent-out rows and the range of leave-one-parent-out means do not constitute confidence intervals. D_free preserves all six within-condition noise pairs; variation is not automatically better.

cross_repeat_parent_effects.csv and cross_repeat_group_effects.csv compare the first-stage and repeat candidate-minus-control effects after averaging the four noises within each experiment. They never pair different noise indices or claim a pure training-seed effect: training inputs and evaluation noises both change. All metrics are included regardless of sign. No composite score, ranking, automatic promotion or GT reconstruction gate is computed.

C coordinates are hard-preserved but do not certify original vertex connectivity, usable chairs, semantic placement, seat support or clean attachment. CF/FF pair counts, exact-coordinate topology diagnostics, degenerate faces and sampled ConflictCov_C (I) remain their frozen geometric definitions. I uses the reused 2048-point C query banks, with lower/upper and support/uncertainty components retained; it does not resolve every thin-wall ambiguity. Duplicate excess is the existing exact-coordinate diagnostic, not a newly invented near-duplicate detector. Geometry and diversity require the full separate visual review for interpretation.

The inherited condition-pool selection-policy confound remains: these are existing-pool recipe comparisons, not a pure ratio causal study. The repeat changes the input random stream; Geo initialization remains 1010 and this is not a new initialization-seed trial. Exact checkpoint training provenance and completion accounting remain separate audits.

This tool reads saved tables, registered small receipts, actual input/query file hashes and source hashes. It performs zero scoring/model/geometry/RNG calls and never reopens large weights. Preserved metric status fields and auxiliary reference documents are not used for model selection.
"""

def analyze(out=base.DEFAULT_OUT):
    out=base.resolve(out).resolve()
    require(out.name==base.ROOT_NAME and out.parent.name=="experiments","Wrong experiment root")
    verify_sources()
    files={}
    for phase in ("main","course","confirm"):
        root=out/"phases"/phase
        files[phase+"_registration"]=root/"registration.json"
        for name in ("per_output.json","scoring_receipt.json","diversity_pairs.csv","diversity_per_condition.csv"):
            files[phase+"_"+name]=root/"evaluation"/name
    files["condition_pool_audit"]=out/"audit/condition_construction_audit.json"
    missing=[str(p)for p in files.values()if not p.is_file()]
    if missing:return dict(status="PENDING",missing_files=missing,new_model_calls=0,new_scoring_calls=0,automatic_promotion=False)
    regs={p:read_json(files[p+"_registration"])for p in ("main","course","confirm")}
    requests,old_requests,queries,auth=metadata_bindings(out,files,regs)
    raw={p:read_json(files[p+"_per_output.json"])["rows"]for p in regs}
    # Preserve full prior receipts/panel validation, then select the fixed pair.
    first_all=course.validate_panel(regs["main"],regs["course"],old_requests,raw["main"],raw["course"],
        read_json(files["main_scoring_receipt.json"]),read_json(files["course_scoring_receipt.json"]))
    first=[r for r in first_all if r["label"]in LABELS]
    repeat=raw["confirm"]
    check_receipt(read_json(files["confirm_scoring_receipt.json"]),128)
    validate_scored_rows(first,old_requests,queries["main"],regs["main"]["requests_sha256"],FIRST_NOISES)
    validate_scored_rows(repeat,requests,queries["confirm"],regs["confirm"]["requests_sha256"],NOISES)
    checkpoint_bindings(out,files,regs,dict(main=[r for r in first if r["label"]==CONTROL],
        course=[r for r in first if r["label"]==CANDIDATE],confirm=repeat))
    # Optional original auxiliary diagnostics are preserved, never promoted.
    auxiliaries={}
    for phase in ("main","course","confirm"):
        path=out/"phases"/phase/"evaluation/reference_A20_per_output.json"
        if path.is_file():
            files[phase+"_reference"]=path
            auxiliaries[phase]=[r for r in read_json(path)["rows"]if r["label"]in LABELS]
    cohort=course.membership(requests)
    result=tables(repeat,requests,cohort,noises=NOISES)
    old_result=tables(first,old_requests,cohort,noises=FIRST_NOISES)
    def diversity(stage):
        phases=("main","course")if stage=="first"else("confirm",)
        cc=[];pp=[]
        for phase in phases:
            label=CONTROL if phase=="main"else CANDIDATE if phase=="course"else None
            for name,target in (("diversity_per_condition.csv",cc),("diversity_pairs.csv",pp)):
                target.extend(r for r in read_csv(files[phase+"_"+name])if label is None or r["label"]==label)
        return diversity_tables(cc,pp,first if stage=="first"else repeat,cohort,
                                noises=FIRST_NOISES if stage=="first"else NOISES)
    d,od=diversity("repeat"),diversity("first")
    for pack in (d,od):
        pack["diversity_leave_one_parent_out"]=complete_loo(
            [dict(r,metric="D_free_mean")for r in pack["diversity_paired_parents"]])
    cross=dict(
        cross_repeat_parent_effects=compare_effects(old_result["paired_parent_metrics"],result["paired_parent_metrics"],
            ("parent_id","ratio","metric"),"delta"),
        cross_repeat_group_effects=compare_effects(old_result["paired_parent_summary"],result["paired_parent_summary"],
            ("ratio","metric","parent_stratum"),"mean"),
        cross_repeat_diversity_parent_effects=compare_effects(od["diversity_paired_parents"],d["diversity_paired_parents"],
            ("parent_id","ratio"),"delta"),
        cross_repeat_diversity_group_effects=compare_effects(od["diversity_paired_summary"],d["diversity_paired_summary"],
            ("ratio","parent_stratum"),"mean"))
    result["leave_one_parent_out"]=complete_loo(result["paired_parent_metrics"])
    result["metric_status_summary"]=status_tables(repeat,cohort)
    old_result["leave_one_parent_out"]=complete_loo(old_result["paired_parent_metrics"])
    sources={str(Path(p).resolve()):digest(p)for p in (__file__,course.__file__,base.__file__)}
    provenance=dict(version=VERSION,input_files={k:dict(path=str(p),sha256=digest(p))for k,p in files.items()},
        sources=sources,selected_pair=list(LABELS),training_namespace=auth["namespace"],
        operations="Saved tables, metadata and hashes only",new_model_calls=0,new_scoring_calls=0,
        new_geometry_calls=0,new_RNG_calls=0,weights_opened=0)
    identity=base.canonical_hash(provenance)
    dest=out/"analysis/confirmation"/(VERSION+"_"+identity[:16])
    receipt_path=dest/"analysis_receipt.json"
    if receipt_path.exists():
        old=read_json(receipt_path)
        require(old["input_identity"]==identity,"Immutable analysis identity differs")
        for relative,sha in old["output_sha256"].items():require(digest(dest/relative)==sha,"Immutable analysis output changed")
        return old
    for item in provenance["input_files"].values():require(digest(item["path"])==item["sha256"],"Input changed during read")
    for path,sha in sources.items():require(digest(path)==sha,"Analysis source changed during read")
    dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/"provenance.json",provenance)
    write_json(dest/"all_metrics.json",dict(rows=repeat,preserved_outputs=128))
    write_csv(dest/"per_output.csv",flattened(repeat))
    write_json(dest/"first_stage_pair_all_metrics.json",dict(rows=first,preserved_outputs=128))
    write_csv(dest/"first_stage_pair_per_output.csv",flattened(first))
    write_json(dest/"preserved_reference_diagnostics.json",dict(rows_by_phase=auxiliaries,promotion_gate=False))
    for name,rr in {**result,**d,**cross}.items():write_csv(dest/(name+".csv"),rr)
    for name,rr in {**old_result,**od}.items():write_csv(dest/("first_stage_"+name+".csv"),rr)
    write_json(dest/"condition_context.json",dict(parent_membership=cohort,
        inherited_condition_pool_audit=read_json(files["condition_pool_audit"]),
        training_completion_proven_by_analysis=False,new_parent_generalization=False,
        independent_input_repeat=True,independent_initialization_seed=False,Geo_initialization_seed=1010))
    write_json(dest/"decision_evidence.json",dict(model_summary=result["model_summary"],
        paired_parent_summary=result["paired_parent_summary"],diversity=d["diversity_paired_summary"],
        cross_repeat=cross,automatic_promotion=False,overall_score=None,statistical_significance=None,
        visual_review="SEPARATE_COMPLETE_ACTUAL_IMAGE_REVIEW_REQUIRED"))
    (dest/"README.md").write_text(readme(),encoding="utf8")
    receipt=dict(status="COMPLETE",version=VERSION,input_identity=identity,analysis_dir=str(dest),
        registered_outputs=128,first_stage_comparison_outputs=128,registered_requests=64,registered_conditions=16,
        preserved_metric_fields=len(METRICS),fixed_numeric_columns=len(METRICS)-len(NONNUMERIC),
        diversity_pairs_preserved=len(d["diversity_pairs"]),delta_direction="C20_40_MIX minus R2_MIX_H_ALL",
        new_model_calls=0,new_scoring_calls=0,automatic_promotion=False,
        cross_experiment_output_pairing=False,new_parent_generalization=False,
        output_sha256={p.name:digest(p)for p in sorted(dest.iterdir())if p.is_file()})
    write_json(receipt_path,receipt)
    write_json(out/"analysis/confirmation/latest.json",dict(analysis_dir=str(dest),receipt_path=str(receipt_path),
        receipt_sha256=digest(receipt_path),input_identity=identity))
    return receipt

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",default=base.DEFAULT_OUT)
    result=analyze(parser.parse_args(argv).out);print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result["status"]=="COMPLETE"else 2

if __name__=="__main__":raise SystemExit(main())
