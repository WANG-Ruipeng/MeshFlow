"""No-model regression of the preregistered conservative screen boundaries."""
import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import contextual_alignment_analysis as a
import contextual_alignment_gallery as gallery

def fixture():
    return [dict(contrast="C_KNOWN_ALIGN_1000-A_FM_1000",panel="P1",cohort=scope,parent_id=str(i),metric=k,delta=v,delta_lower=v,delta_upper=v)
            for scope in ("OLD","NEW","ALL") for i in range(8) for k,v in (("SeamCov",.02),("ConflictCov_C",.005),("ExposedRetain",-.005))]
def result(rows):return a.screen(rows,[],"C_KNOWN_ALIGN_1000")
def criterion(r,name):return next(v for v in r["criteria"] if v["name"]==name)
def run():
    checks=[]
    def passed(name,assertion):
        if not assertion:raise AssertionError(name)
        checks.append(dict(name=name,status="PASS"))
    passed("complete_positive_fixture",result(fixture())["status"]=="PASS")
    for metric,lo,hi,name in (("SeamCov",.009,.011,"ALL_S_mean_gain"),("ConflictCov_C",.009,.011,"ALL_I_mean_cost"),("ExposedRetain",-.011,-.009,"ALL_E_mean_change")):
        rr=fixture()
        for r in rr:
            if r["metric"]==metric:r.update(delta=(lo+hi)/2,delta_lower=lo,delta_upper=hi)
        v=result(rr);passed(metric+"_threshold_crossing_INCOMPLETE",v["status"]=="INCOMPLETE" and criterion(v,name)["status"]=="INCOMPLETE")
    rr=fixture()
    for r in rr:
        if r["metric"]=="SeamCov":r.update(delta=.008,delta_lower=.007,delta_upper=.009)
    passed("S_certain_failure",result(rr)["status"]=="FAIL")
    rr=fixture()
    for r in rr:
        if r["metric"]=="SeamCov":r.update(delta_lower=-.001,delta_upper=.04)
    v=result(rr);passed("median_not_point_estimate",criterion(v,"ALL_S_parent_median")["status"]=="INCOMPLETE")
    rr=fixture()
    for r in rr:
        if r["metric"]=="SeamCov" and int(r["parent_id"])>=4:r.update(delta=.01,delta_lower=-.001,delta_upper=.03)
    v=result(rr);c=criterion(v,"ALL_S_improving_parents")
    passed("four_guaranteed_eight_possible_is_INCOMPLETE",c["lower"]==4 and c["upper"]==8 and c["status"]=="INCOMPLETE")
    rr=fixture()
    for r in rr:
        if r["metric"]=="SeamCov":
            vv=.05 if int(r["parent_id"])<4 else 0.;r.update(delta=vv,delta_lower=vv,delta_upper=vv)
    v=result(rr);passed("at_most_four_improving_is_FAIL",criterion(v,"ALL_S_improving_parents")["status"]=="FAIL" and v["status"]=="FAIL")
    rr=fixture()
    for r in rr:
        if r["metric"]=="SeamCov" and int(r["parent_id"])>=5:r.update(delta=0,delta_lower=0,delta_upper=0)
    passed("exactly_five_improving_can_PASS",result(rr)["status"]=="PASS")
    passed("one_missing_noise_does_not_shrink_denominator",a.mean([.02,None]) is None)
    rr=fixture();rr=[r for r in rr if r["parent_id"]!="7"]
    passed("one_missing_parent_is_INCOMPLETE",result(rr)["status"]=="INCOMPLETE")
    passed("E_uses_global_eligibility_bounds",a.interval("ExposedRetain",dict(ExposedRetain=.95,ExposedRetain_upper=1.,ExposedRetain_global_lower=.6,ExposedRetain_global_upper=1.))==(.6,1.))
    passed("zero_median_is_not_positive",a.criterion(0.,0.,">",0)=="FAIL")
    passed("detector_bounds_not_CI_no_extra_epsilon",a.criterion(.01,.01,">=",.01)=="PASS")
    passed("anonymous_labels_only",set(gallery.mapping().values())=={"Method 1","Method 2","Method 3","Method 4"})
    passed("no_torch_import","torch" not in sys.modules)
    return dict(status="PASS",checks=checks,count=len(checks),model_calls=0,geometry_scoring=0)
if __name__=="__main__":print(json.dumps(run()))
