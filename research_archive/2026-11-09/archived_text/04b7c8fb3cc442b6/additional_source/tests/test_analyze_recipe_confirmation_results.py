from __future__ import annotations
import copy
import hashlib
import importlib.util
import itertools
from pathlib import Path
import statistics
import pytest

_spec=importlib.util.spec_from_file_location("confirmation_analysis",Path(__file__).parents[1]/"tools/analyze_recipe_confirmation_results.py")
m=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(m)
H=lambda text:hashlib.sha256(str(text).encode()).hexdigest()

def panel(noises=m.NOISES):
    requests=[];queries=[];rows=[]
    for i in range(8):
        pid="p"+str(i)
        for ratio in (.2,.4):
            cid=pid+"__C"+str(round(ratio*100))
            cohort=("ORIGINAL6"if i<6 else"EXTENDED2")if ratio==.4 else"A20_REFERENCE"
            query=dict(condition_id=cid,parent_id=pid,ratio=ratio,C_sha256=H(cid),
                query_count=2048,all_four_noises_share_same_bank=True,query_sha256=H("query"+cid),
                weight_denominator=1.25,identity=dict(L0=7.0))
            queries.append(query)
            for n in noises:
                rid=cid+"__n"+str(n)
                q=dict(request_id=rid,parent_id=pid,ratio=ratio,condition_id=cid,candidate=n,
                    condition_cohort=cohort,N=200,K=int(200*ratio),C_sha256=H(cid),
                    input_sha256=H("input"+rid),Gaussian_sha256=H("z"+rid),
                    noise_full_N_Gaussian_sha256=H("full"+pid+str(n)),new_parent_generalization=False)
                seed_text=m.NOISE_NAMESPACE+"/"+pid+"/n"+str(n-8)
                q.update(noise_seed_text=seed_text,noise_seed=int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4],"little"))
                requests.append(q)
                for label in m.LABELS:
                    metrics={name:0.0 for name in m.METRICS}
                    metrics["ConflictCov_C_status"]="PASS"
                    metrics["EXACT_COORD_closed_two_faces_per_edge"]=False
                    metrics["free_q_mean"]=10.0+i+(i-3.5 if label==m.CANDIDATE else 0)
                    rows.append(dict(label=label,**{k:q[k]for k in("request_id","parent_id","ratio","condition_id","candidate","condition_cohort","N","K")},
                        identity=dict(request_id=rid,label=label,input_sha256=q["input_sha256"],query_sha256=query["query_sha256"]),
                        query_sha256=query["query_sha256"],registered_C_queries=2048,registered_free_faces=q["N"]-q["K"],
                        query_weight_denominator=1.25,L0=7.0,status="PASS",contracts={"exact_C":True},
                        metrics=metrics,metric_status={"ConflictCov_C":"PASS"}))
    registration=dict(status="FROZEN",identity={"requests_sha256":H("requests")},rows=queries)
    return requests,rows,registration

def holdout():
    requests,rows,queries=panel();old,_,_=panel(m.FIRST_NOISES)
    plan=dict(namespace=m.NOISE_NAMESPACE,candidate_model_identity_in_seed=False,expected_outputs=128,
        rows=[dict(**{k:q[k]for k in("parent_id","ratio","candidate","condition_id","N","K","C_sha256")},
            seed=q["noise_seed"],seed_text=q["noise_seed_text"])for q in requests])
    collision=dict(status="PASS",rows=[dict(request_id=q["request_id"],Gaussian_sha256=q["Gaussian_sha256"],
        full_N_Gaussian_sha256=q["noise_full_N_Gaussian_sha256"],status="PASS")for q in requests],
        main_input_files=[{k:q[k]for k in("request_id","input_sha256","Gaussian_sha256")}for q in old])
    auth=dict(schema="chair_recipe_confirmation_training_authorization_v1",stage="independent_confirmation",
        experiment=m.EXPERIMENT,selected_pair=list(m.LABELS),authorized_recipes=list(m.LABELS),
        hybrid_schedules={r:"all"for r in m.LABELS},updates_per_recipe=6000,effective_batch=8,microbatch=1,
        namespace=m.EXPERIMENT+"/confirmation/input_repeat1")
    reg=dict(kind="confirm",labels=list(m.LABELS),expected_outputs=128,new_parent_generalization=False)
    return requests,old,plan,collision,auth,reg

def diversity(rows,noises=m.NOISES):
    cc=[];pp=[];conditions={}
    for r in rows:conditions[r["label"],r["condition_id"]]=r
    for (label,cid),row in conditions.items():
        for a,b in itertools.combinations(noises,2):
            pp.append(dict(label=label,condition_id=cid,parent_id=row["parent_id"],ratio=str(row["ratio"]),
                candidate_a=str(a),candidate_b=str(b),D_free_pair="0.25",status="PASS"))
        cc.append(dict(label=label,condition_id=cid,parent_id=row["parent_id"],ratio=str(row["ratio"]),
            condition_cohort=row["condition_cohort"],expected_pairs="6",available_pairs="6",D_free_mean="0.25",
            direction="variation_not_larger_is_better",not_original_TMD=True))
    return cc,pp

def test_complete_panel_and_registered_metric_count():
    qs,rs,qr=panel()
    assert m.validate_scored_rows(rs,qs,qr,H("requests"),m.NOISES)==rs
    assert len(m.fixed_catalog(rs))==49
    assert sum(r["numeric_summary"]for r in m.fixed_catalog(rs))==47
    assert m.validate_holdout(*holdout())

@pytest.mark.parametrize("field,value",[
    ("input_sha256",H("shared-but-wrong")),("query_sha256",H("wrong-query")),("request_id","wrong"),
    ("label",m.CONTROL)])
def test_common_wrong_inner_binding_rejected(field,value):
    qs,rs,qr=panel()
    for r in rs[:2]:r["identity"][field]=value
    with pytest.raises(ValueError):m.validate_scored_rows(rs,qs,qr,H("requests"),m.NOISES)

@pytest.mark.parametrize("field,value",[
    ("registered_C_queries",2047),("registered_free_faces",999),("query_weight_denominator",2.5),
    ("L0",3.0),("candidate",0),("ratio",.3),("parent_id","other"),("status","PARTIAL")])
def test_outer_identity_and_denominator_rejected(field,value):
    qs,rs,qr=panel()
    rs[0][field]=value
    with pytest.raises(ValueError):m.validate_scored_rows(rs,qs,qr,H("requests"),m.NOISES)

@pytest.mark.parametrize("field,value",[
    ("C_sha256",H("wrong-C")),("parent_id","wrong-parent"),("ratio",.3),
    ("query_count",1000),("all_four_noises_share_same_bank",False)])
def test_query_binding_rejected(field,value):
    qs,rs,qr=panel();qr["rows"][0][field]=value
    with pytest.raises(ValueError):m.validate_scored_rows(rs,qs,qr,H("requests"),m.NOISES)

def test_wrong_noise_and_missing_cell_rejected():
    qs,rs,qr=panel(m.FIRST_NOISES)
    with pytest.raises(ValueError,match="noise"):m.validate_scored_rows(rs,qs,qr,H("requests"),m.NOISES)
    qs,rs,qr=panel()
    with pytest.raises(ValueError,match="Complete128"):m.validate_scored_rows(rs[:-1],qs,qr,H("requests"),m.NOISES)
    with pytest.raises(ValueError,match="duplicate"):m.validate_scored_rows(rs+[rs[0]],qs,qr,H("requests"),m.NOISES)

def test_old_noise_not_valid_independent_holdout():
    args=holdout();args[0][0]["Gaussian_sha256"]=args[1][0]["Gaussian_sha256"]
    args[3]["rows"][0]["Gaussian_sha256"]=args[0][0]["Gaussian_sha256"]
    with pytest.raises(ValueError,match="Gaussian"):m.validate_holdout(*args)

def test_namespace_must_determine_seed_not_just_metadata():
    args=holdout();q=args[0][0];p=args[2]["rows"][0]
    q["noise_seed"]=p["seed"]=123
    with pytest.raises(ValueError,match="namespace"):m.validate_holdout(*args)

def test_no_original_C_or_pool_change_on_holdout():
    args=holdout()
    # Keep new plan and requests mutually equal, but differ from original C.
    for q in args[0][:4]:q["C_sha256"]=H("changed-C")
    for p in args[2]["rows"][:4]:p["C_sha256"]=H("changed-C")
    with pytest.raises(ValueError,match="condition"):m.validate_holdout(*args)

def test_collision_old_input_receipt_bound_to_main():
    args=holdout();args[3]["main_input_files"][0]["input_sha256"]=H("wrong")
    with pytest.raises(ValueError,match="original hash"):m.validate_holdout(*args)

def test_cross_ratio_full_N_binding():
    args=holdout()
    args[0][0]["noise_full_N_Gaussian_sha256"]=H("other")
    args[3]["rows"][0]["full_N_Gaussian_sha256"]=H("other")
    with pytest.raises(ValueError,match="shared across ratios"):m.validate_holdout(*args)

def test_missing_and_all_NA_keep_denominators():
    qs,rs,_=panel()
    for r in rs:r["metrics"]["free_q_gt10_fraction"]=None
    rs[0]["metrics"]["free_q_mean"]=None
    result=m.tables(rs,qs,m.course.membership(qs))
    metric="free_q_gt10_fraction"
    p=next(r for r in result["per_parent_metrics"]if r["metric"]==metric)
    assert (p["expected"],p["available"],p["mean"])==(4,0,None)
    group=next(r for r in result["paired_parent_summary"]if r["metric"]==metric and r["parent_stratum"]=="ALL")
    assert(group["expected"],group["available"],group["mean"])==(8,0,None)
    broken=next(r for r in result["model_summary"]if r["label"]==m.CANDIDATE and r["ratio"]==.2 and r["metric"]=="free_q_mean"and r["parent_stratum"]=="ALL")
    assert(broken["expected"],broken["available"],broken["mean"])==(8,7,None)
    assert broken["available_mean"]is not None

def test_parent_weighting_6_2_strata_and_LOO():
    qs,rs,_=panel()
    result=m.tables(rs,qs,m.course.membership(qs))
    rows=[r for r in result["paired_parent_summary"]if r["metric"]=="free_q_mean"and r["ratio"]==.2]
    by={r["parent_stratum"]:r for r in rows}
    assert by["ALL"]["expected"]==8 and by["ALL"]["mean"]==0
    assert by["ORIGINAL6"]["expected"]==6 and by["ORIGINAL6"]["mean"]==-1
    assert by["EXTENDED2"]["expected"]==2 and by["EXTENDED2"]["mean"]==3
    assert(by["ALL"]["negative_count"],by["ALL"]["exact_zero_count"],by["ALL"]["positive_count"])==(4,0,4)
    loo=m.complete_loo(result["paired_parent_metrics"])
    p0=next(r for r in loo if r["metric"]=="free_q_mean"and r["ratio"]==.2 and r["parent_stratum"]=="ALL"and r["omitted_parent_id"]=="p0")
    assert p0["expected"]==7 and p0["mean"]==.5
    assert "p0"not in p0["remaining_parent_ids"]

def test_frozen_course_arithmetic_unchanged():
    qs,rs,_=panel(m.FIRST_NOISES);cohort=m.course.membership(qs)
    extra=[dict(copy.deepcopy(r),label="C40_20_MIX")for r in rs if r["label"]==m.CONTROL]
    frozen=m.course.tables(rs+extra,qs,cohort)
    adapted=m.tables(rs,qs,cohort,noises=m.FIRST_NOISES)
    for name in ("paired_noise_metrics","paired_parent_metrics","paired_parent_summary","paired_noise_summary"):
        selected=[r for r in frozen[name]if(r["first"],r["second"])in m.PAIRS]
        assert selected==adapted[name]

def test_all49_fields_in_csv_even_absent():
    qs,rs,_=panel()
    for r in rs:del r["metrics"]["ConflictCov_C_upper"]
    catalog=m.fixed_catalog(rs)
    assert next(r for r in catalog if r["metric"]=="ConflictCov_C_upper")["missing_field"]==128
    assert all("metric."+metric in m.flattened(rs)[0]for metric in m.METRICS)

def test_all_six_diversity_pairs_and_incomplete_denominator():
    qs,rs,_=panel();cc,pp=diversity(rs)
    result=m.diversity_tables(cc,pp,rs,m.course.membership(qs))
    assert len(result["diversity_pairs"])==192
    assert len(result["diversity_paired_parents"])==16
    pp[0]["D_free_pair"]="";pp[0]["status"]="BLOCKED";cc[0]["available_pairs"]="5";cc[0]["D_free_mean"]=""
    result=m.diversity_tables(cc,pp,rs,m.course.membership(qs))
    incomplete=next(r for r in result["diversity_summary"]if r["label"]==cc[0]["label"]and r["ratio"]==float(cc[0]["ratio"])and r["parent_stratum"]=="ALL")
    assert(incomplete["expected"],incomplete["available"],incomplete["mean"])==(8,7,None)

@pytest.mark.parametrize("mode",["missing","old_noise","mean","survivor_mean","metadata"])
def test_diversity_binding_negatives(mode):
    qs,rs,_=panel();cc,pp=diversity(rs)
    if mode=="missing":pp.pop()
    elif mode=="old_noise":pp[0]["candidate_a"]="0"
    elif mode=="mean":cc[0]["D_free_mean"]="0.8"
    elif mode=="survivor_mean":pp[0]["status"]="BLOCKED";pp[0]["D_free_pair"]="";cc[0]["available_pairs"]="5"
    else:pp[0]["parent_id"]="other"
    with pytest.raises(ValueError):m.diversity_tables(cc,pp,rs,m.course.membership(qs))

def test_cross_repeat_only_parent_effects_with_no_noise_matching():
    old=[dict(parent_id="p0",ratio=.2,metric="I",delta=-2.0)]
    new=[dict(parent_id="p0",ratio=.2,metric="I",delta=3.0)]
    r=m.compare_effects(old,new,("parent_id","ratio","metric"),"delta")[0]
    assert r["direction_relation"]=="direction_reversal"and r["effect_change"]==5
    assert r["cross_experiment_output_pairing"]is False
    assert "request_id"not in r and "candidate"not in r
    assert r["first_stage_noises"]==[0,1,6,7]and r["repeat_noises"]==[8,9,10,11]
    new[0]["delta"]=None
    assert m.compare_effects(old,new,("parent_id","ratio","metric"),"delta")[0]["direction_relation"]=="NA"

def test_actual_file_bytes_must_match(tmp_path):
    p=tmp_path/"query.npz";p.write_bytes(b"frozen bytes");files={}
    m.checked_file(p,m.digest(p),files,"query")
    p.write_bytes(b"changed bytes")
    with pytest.raises(ValueError,match="file hash"):m.checked_file(p,H("frozen bytes"),{},"query")

def test_missing_results_pending_writes_nothing(tmp_path):
    root=tmp_path/"experiments"/m.base.ROOT_NAME;root.mkdir(parents=True)
    result=m.analyze(root)
    assert result["status"]=="PENDING"and result["new_model_calls"]==result["new_scoring_calls"]==0
    assert list(root.iterdir())==[]

def test_diversity_loo_fixed_parent_denominator():
    qs,rs,_=panel();cc,pp=diversity(rs)
    result=m.diversity_tables(cc,pp,rs,m.course.membership(qs))
    loo=m.complete_loo([dict(r,metric="D_free_mean")for r in result["diversity_paired_parents"]])
    assert len(loo)==32
    assert all(r["expected"]==7 for r in loo if r["parent_stratum"]=="ALL")
    assert all(r["expected"]==1 for r in loo if r["parent_stratum"]=="EXTENDED2")

@pytest.mark.parametrize("wrong",[False,True])
def test_checkpoint_small_receipt_binding_without_weights(tmp_path,wrong):
    label=m.CANDIDATE
    model=dict(label=label,recipe=label,loader="recipe_confirmation",checkpoint_path="/absent-large-weight.pt")
    reg=dict(models=[model])
    registration=tmp_path/"phases/confirm/registration.json";m.write_json(registration,reg)
    files={"confirm_registration":registration}
    binding=dict(model,registration_sha256=m.digest(registration),checkpoint_sha256=H("weights"))
    m.write_json(tmp_path/"phases/confirm/checkpoints"/(label+".json"),binding)
    rows={"confirm":[dict(label=label,identity=dict(checkpoint_sha256=H("wrong")if wrong else H("weights")))]}
    if wrong:
        with pytest.raises(ValueError,match="checkpoint identity"):m.checkpoint_bindings(tmp_path,files,{"confirm":reg},rows)
    else:
        m.checkpoint_bindings(tmp_path,files,{"confirm":reg},rows)
        assert "confirm_checkpoint_"+label in files
