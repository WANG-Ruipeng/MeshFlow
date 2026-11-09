import copy
import importlib.util
import json
from pathlib import Path
import pytest
_spec=importlib.util.spec_from_file_location("confirm_review",Path(__file__).parents[1]/"tools/prepare_recipe_confirmation_review.py")
m=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(m)

def fixture(tmp_path):
    qs=[];pages=[]
    parents=["33ec0","86ec0","341b0","bdd0","866e0","534b0","c1b0","f5960"]
    for i,pid in enumerate(parents):
        for ratio in(.2,.4):
            cid=pid+"_"+str(ratio)
            for noise in(8,9,10,11):
                rid=cid+"_n"+str(noise)
                qs.append(dict(request_id=rid,parent_id=pid,ratio=ratio,candidate=noise,condition_id=cid,
                    condition_cohort="ORIGINAL6"if i<6 else"EXTENDED2",secret_model="MUST_NOT_COPY"))
                for mode in("colored","neutral"):
                    p=tmp_path/mode/(rid+".png");p.parent.mkdir(parents=True,exist_ok=True)
                    p.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-"+rid.encode()+mode.encode())
                    pages.append(dict(request_id=rid,condition_id=cid,candidate=noise,mode=mode,path=str(p),
                        sha256=m.digest(p),private_mapping={"MUST_NOT_COPY":"Method 1"}))
    gallery=dict(status="COMPLETE",page_count=128,expected_pages=128,all_registered_outputs_shown=True,
        five_views_per_model=True,model_labels_blinded=True,pages=pages,secret_mapping={"MUST_NOT_COPY":"Method 2"})
    return qs,gallery

def test_safe_allowlist_coverage_and_disclosure(tmp_path):
    qs,gallery=fixture(tmp_path)
    manifest=m.build_manifest(qs,gallery,tmp_path,"requesthash","galleryhash")
    assert len(manifest["rows"])==128 and manifest["registered_unique_outputs"]==128
    assert manifest["reviewer_primary_pages"]=={"root":32,"schedule40_data":48,"schedule40_train":48}
    assert manifest["prior_main_and_course_knowledge"]is True
    assert manifest["human_independent_blind_study"]is False
    assert manifest["private_mapping_opened"]is False
    assert all(not r["actually_viewed"]for r in manifest["rows"])
    assert "MUST_NOT_COPY"not in json.dumps(manifest)

@pytest.mark.parametrize("case",["missing","duplicate","wrong_noise","hash","unblinded","unknown_parent","outside"])
def test_manifest_binding_negatives(tmp_path,case):
    qs,gallery=fixture(tmp_path)
    if case=="missing":gallery["pages"].pop()
    elif case=="duplicate":gallery["pages"][-1]=copy.deepcopy(gallery["pages"][0])
    elif case=="wrong_noise":gallery["pages"][0]["candidate"]=0
    elif case=="hash":Path(gallery["pages"][0]["path"]).write_bytes(b"changed")
    elif case=="unblinded":gallery["model_labels_blinded"]=False
    elif case=="unknown_parent":qs[0]["parent_id"]="unknown"
    else:gallery["pages"][0]["path"]=str(tmp_path.parent/"outside.png")
    with pytest.raises(ValueError):m.build_manifest(qs,gallery,tmp_path,"requesthash","galleryhash")

def test_pending_does_not_write_or_read_mapping(tmp_path):
    out=tmp_path/"experiments/chair_condition_recipe_factorial_v1";out.mkdir(parents=True)
    result=m.prepare(out)
    assert result["status"]=="PENDING"and list(out.iterdir())==[]
