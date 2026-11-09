from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import pytest

_spec=importlib.util.spec_from_file_location("confirm_visual_merge",Path(__file__).parents[1]/"tools/merge_recipe_confirmation_review.py")
m=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(m)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False),encoding="utf8")

def setup_panel(tmp_path):
    run=tmp_path/"experiments/chair_condition_recipe_factorial_v1"
    safe_root=run/"audit/visual_review/confirm";gallery=run/"phases/confirm/gallery"
    pairs=[("root","33ecx"),("root","86ecx"),("schedule40_data","341bx"),("schedule40_data","bddx"),
           ("schedule40_data","866ex"),("schedule40_train","534bx"),("schedule40_train","c1bx"),
           ("schedule40_train","f596x")]
    rows=[];requests=[]
    for owner,parent in pairs:
        for ratio in(.2,.4):
            condition=parent+"_C"+str(round(100*ratio))
            for noise in(8,9,10,11):
                rid=condition+"__n"+str(noise);images=[]
                for mode in("colored","neutral"):
                    path=gallery/mode/(rid+".png");path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(b"\x89PNG\r\n\x1a\nmock-"+rid.encode()+mode.encode())
                    row=dict(request_id=rid,parent_id=parent,condition_id=condition,ratio=ratio,
                        condition_cohort="ORIGINAL6",candidate=noise,reviewer=owner,mode=mode,
                        path=str(path),sha256=m.digest(path),supplemental=False,display_aliases=list(m.ALIASES))
                    rows.append(row)
                    images.append(dict(mode=mode,path=str(path),sha256=m.digest(path),actually_viewed=True,
                        all_five_views_examined=True,view_tool="synthetic-test-only",viewed_at_utc="2000-01-01T00:00:00Z",
                        view_evidence="Synthetic metadata fixture; no image was visually reviewed"))
                doc=dict(schema=m.REQUEST_SCHEMA,reviewer=owner,request_id=rid,parent_id=parent,condition_id=condition,
                    ratio=ratio,candidate=noise,prior_main_and_course_knowledge=True,AI_reviewer=True,
                    private_mapping_read=False,confirmation_quality_read=False,images=images,
                    request_extra={"verbatim":"original common context"},notes=[
                        dict(display_alias=alias,observation="Original observation "+alias,uncertainty="Cannot certify topology",
                            original_extra={"text":"Keep this exactly"})for alias in m.ALIASES])
                write(safe_root/owner/"requests"/(rid+".json"),doc)
                requests.append(dict(request_id=rid))
    for owner in m.REVIEWERS:
        write(safe_root/owner/"cross_noise_initial.json",dict(schema=m.CROSS_SCHEMA,reviewer=owner,
            prior_main_and_course_knowledge=True,private_mapping_read=False,confirmation_quality_read=False,
            observations=[dict(parent_id=parent,observation="Original across-noise text",uncertainty="Views only")
                for who,parent in pairs if who==owner]))
    write(gallery/"gallery.json",dict(status="COMPLETE",pages=rows))
    request_path=run/"phases/confirm/evaluation/requests.json";write(request_path,dict(rows=requests))
    write(run/"phases/confirm/registration.json",dict(requests_path=str(request_path),requests_sha256=m.digest(request_path)))
    safe=dict(schema=m.MANIFEST_SCHEMA,registered_requests=64,registered_unique_outputs=128,primary_pages=128,
        supplemental_pages=0,display_aliases=list(m.ALIASES),source_gallery_sha256=m.digest(gallery/"gallery.json"),
        source_requests_sha256=m.digest(request_path),rows=rows)
    write(safe_root/"review_manifest.json",safe)
    write(gallery/"private_method_mapping.json",{"C20_40_MIX":"Method 2","R2_MIX_H_ALL":"Method 1"})
    return run,safe

def freeze_all(run):
    return[m.freeze_review(run,r)for r in m.REVIEWERS]

def note_path(run,reviewer="root"):
    return next((run/"audit/visual_review/confirm"/reviewer/"requests").glob("*.json"))

def test_full_freeze_merge_exact_coverage_and_verbatim_preservation(tmp_path):
    run,safe=setup_panel(tmp_path);freezes=freeze_all(run)
    result=m.merge(run);dest=Path(result["output_dir"])
    payload=m.read(dest/"per_output.json");coverage=m.read(dest/"coverage.json")
    assert result["unique_outputs"]==len(payload["rows"])==128
    assert coverage["primary_pngs"]==128 and coverage["requests"]==64
    assert coverage["per_reviewer"]==m.COUNTS
    assert len(payload["original_request_documents"])==64
    assert len(payload["original_cross_noise_sources"])==3
    assert payload["rows"][0]["original_note"]["original_extra"]["text"]=="Keep this exactly"
    assert payload["rows"][0]["original_request_document"]["request_extra"]["verbatim"]=="original common context"
    assert payload["rows"][0]["model_label"]=="R2_MIX_H_ALL"
    assert coverage["human_independent_blind_study"]is False
    assert coverage["numeric_subjective_scores_created"]is False
    assert coverage["quality_tables_read_by_merge"]is False
    assert len(m.read(dest/"image_coverage.json")["rows"])==128
    assert m.merge(run)==result
    assert m.freeze_review(run,"root")==freezes[0]

@pytest.mark.parametrize("change",["missing_freeze","changed_note","changed_image","extra_peer_request",
                                   "changed_manifest","changed_cross","wrong_freeze_hash"])
def test_failures_block_before_private_mapping_open(tmp_path,monkeypatch,change):
    run,safe=setup_panel(tmp_path);freeze_all(run)
    folder=run/"audit/visual_review/confirm"
    if change=="missing_freeze":(folder/"root/initial_freeze_receipt.json").unlink()
    elif change=="changed_note":
        p=note_path(run);doc=m.read(p);doc["notes"][0]["observation"]="Changed after freeze";write(p,doc)
    elif change=="changed_image":Path(safe["rows"][0]["path"]).write_bytes(b"changed")
    elif change=="extra_peer_request":
        p=note_path(run);write(folder/"schedule40_train/requests"/p.name,m.read(p))
    elif change=="changed_manifest":
        p=folder/"review_manifest.json";doc=m.read(p);doc["extra"]="changed after initial freeze";write(p,doc)
    elif change=="changed_cross":
        p=folder/"root/cross_noise_initial.json";doc=m.read(p);doc["observations"][0]["observation"]="changed";write(p,doc)
    else:
        p=folder/"root/initial_freeze_receipt.json";doc=m.read(p);doc["request_file_sha256"]["cross_noise_initial.json"]="wrong";write(p,doc)
    original=m.read;mapping_reads=[]
    def guarded(path):
        if Path(path).name=="private_method_mapping.json":mapping_reads.append(str(path))
        return original(path)
    monkeypatch.setattr(m,"read",guarded)
    with pytest.raises((ValueError,OSError)):m.merge(run)
    assert not mapping_reads
    assert not(run/"audit/visual_review/confirm_complete").exists()

@pytest.mark.parametrize("change",["alias_duplicate","view_false","all_views_false","missing_evidence",
    "future_view","wrong_image","missing_uncertainty","mapping_seen","quality_seen","false_prior","true_label"])
def test_initial_claim_and_note_negatives(tmp_path,change):
    run,safe=setup_panel(tmp_path);p=note_path(run);doc=m.read(p)
    if change=="alias_duplicate":doc["notes"][1]["display_alias"]=doc["notes"][0]["display_alias"]
    elif change=="view_false":doc["images"][0]["actually_viewed"]=False
    elif change=="all_views_false":doc["images"][0]["all_five_views_examined"]=False
    elif change=="missing_evidence":doc["images"][0]["view_evidence"]=""
    elif change=="future_view":doc["images"][0]["viewed_at_utc"]="2999-01-01T00:00:00Z"
    elif change=="wrong_image":doc["images"][0]["path"]=doc["images"][1]["path"]
    elif change=="missing_uncertainty":doc["notes"][0]["uncertainty"]=""
    elif change=="mapping_seen":doc["private_mapping_read"]=True
    elif change=="quality_seen":doc["confirmation_quality_read"]=True
    elif change=="false_prior":doc["prior_main_and_course_knowledge"]=False
    else:doc["notes"][0]["model_label"]="C20_40_MIX"
    write(p,doc)
    with pytest.raises(ValueError):m.freeze_review(run,"root")
    assert not(run/"audit/visual_review/confirm/root/initial_freeze_receipt.json").exists()

def test_changed_mapping_rejected_after_valid_initial_freezes(tmp_path):
    run,safe=setup_panel(tmp_path);freeze_all(run)
    write(run/"phases/confirm/gallery/private_method_mapping.json",{"other":"Method 1","C20_40_MIX":"Method 2"})
    with pytest.raises(ValueError,match="mapping"):m.merge(run)

def test_cross_noise_parent_missing_rejected(tmp_path):
    run,safe=setup_panel(tmp_path);p=run/"audit/visual_review/confirm/root/cross_noise_initial.json"
    doc=m.read(p);doc["observations"].pop();write(p,doc)
    with pytest.raises(ValueError,match="Cross-noise"):m.freeze_review(run,"root")

def test_supplement_or_duplicate_png_not_silently_counted(tmp_path):
    run,safe=setup_panel(tmp_path);safe["rows"][0]["supplemental"]=True
    with pytest.raises(ValueError):m.validate_manifest(safe)
    safe["rows"][0]["supplemental"]=False;safe["rows"][1]["path"]=safe["rows"][0]["path"]
    with pytest.raises(ValueError,match="Duplicate"):m.validate_manifest(safe)

def test_refuse_changed_completed_delivery(tmp_path):
    run,safe=setup_panel(tmp_path);freeze_all(run);result=m.merge(run)
    output=Path(result["output_dir"])/"per_output.csv";output.write_text("changed",encoding="utf8")
    with pytest.raises(ValueError,match="immutable"):m.merge(run)
