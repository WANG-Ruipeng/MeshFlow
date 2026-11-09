"""CPU-only exhaustive multipart packing audit; no neural imports or forwards."""
from __future__ import annotations
import argparse
import collections
import json
import time
import traceback
import numpy as np
from . import data as D


def verify(out=D.DEFAULT_OUT):
    out=D.resolve(out);target=out/"cpu_verification.json"
    if target.exists():
        old=D.read(target)
        if old["manifest_sha256"]!=D.sha(out/"manifest.json") or old["verification_code_sha256"]!=D.sha(__file__):
            raise ValueError("Completed verification identity differs")
        return old
    start=time.perf_counter();before=D.resources();manifest=D.read(out/"manifest.json")
    calls=0;rows=[];fixtures=[];selected_positive=False;selected_extended=False;selected_closed=False
    try:
        for index,row in enumerate(manifest["rows"]):
            if D.sha(row["case_path"])!=row["case_sha256"]:raise ValueError("Consolidated case changed")
            case=D.load_case(row);n,k=int(case["N"]),int(case["K"])
            known,free=case["known_ids"],case["free_ids"]
            assert np.array_equal(case["pre_ot_full"]*np.float32(2),case["full_target"])
            assert np.array_equal(case["full_target"][known],case["C"])
            assert np.array_equal(case["full_target"][free],case["Y"])
            assert np.array_equal(case["C_vertices"][case["C_faces_model"]],case["C"])
            assert np.array_equal(np.sort(np.r_[known,free]),np.arange(n))
            assert np.all(case["source_face_vertex_ids"][known]==case["C_source_vertex_ids"][case["C_faces_model"]])
            base=5000000+index*100
            spec=dict(alpha=1.037,noise_seed=base,epsilon_seed=base+1,time_seed=base+2,permutation_seed=base+3)
            originals={key:value.copy() for key,value in case.items()}
            samples={}
            for arm in ("PLAIN","SEAM"):
                sample=D.sample_for_training(case,spec,arm);calls+=1;samples[arm]=sample
                expected=case["full_target"].copy();expected[...,0]*=np.float32(spec["alpha"])
                ids,corner=sample["source_face_ids"],sample["source_corner_permutations"]
                assert np.array_equal(np.sort(ids),np.arange(n))
                assert np.array_equal(expected[ids[:,None],corner],sample["x1"])
                fixed=sample["anchor_ids"]>=0
                assert np.array_equal(sample["free_corner_mask"],~fixed)
                assert not np.any(sample["u"][fixed])
                assert np.array_equal(sample["x0"][fixed],sample["xt"][fixed])
                assert np.array_equal(sample["x1"][fixed],sample["xt"][fixed])
                assert int(sample["valid_mask"].sum())==n and int(sample["known_mask"].sum())==k
                owners=int(case["occupied_labels"].sum()) if arm=="SEAM" else 0
                assert int(sample["free_coordinate_count"])==9*(n-k)-6*owners==3*int((~fixed).sum())
                assert all(np.isfinite(sample[key]).all() for key in ("x0","x1","xt","u"))
                assert np.all(np.sum(fixed,axis=1)[sample["known_mask"]]==3)
                assert int(np.sum(np.sum(fixed,axis=1)==2))==owners
                assert np.sum(np.sum(fixed,axis=1)==0)==n-k-owners
            assert all(np.array_equal(case[key],originals[key]) for key in case)
            # All inference layout fields depend only on C/N and supplied occupancy.
            edges,features=D.edge_features(case["C_vertices"],case["C_faces_original"],n)
            for occupancy in (np.zeros(len(edges),bool),np.ones(len(edges),bool)):
                layout=D.inference_layout(case["C_vertices"],case["C_faces_original"],case["C_faces_model"],n,occupancy)
                assert len(layout["anchor_ids"])==n
                assert layout["free_coordinate_count"]==9*(n-k)-6*int(occupancy.sum())
                assert np.array_equal(layout["edge_records"],case["edge_records"])
            if not len(edges):
                assert features.shape==(0,24)
                for key in samples["PLAIN"]:
                    if key!="arm":assert np.array_equal(samples["PLAIN"][key],samples["SEAM"][key])
                if not selected_closed:
                    fixtures.append(dict(condition_id=row["condition_id"],kind="CLOSED",result="PLAIN_SEAM_ALL_ARRAYS_EQUAL_EXCEPT_ARM_LABEL",neural_forward_equivalence="STATIC_DIRECT_DELEGATION_IN_EXISTING_seam_forward"))
                    selected_closed=True
            special=row.get("seam_reasons",[])
            wanted=(not selected_positive and case["occupied_labels"].any()) or (not selected_extended and len(edges) and special)
            if wanted:
                aug={key:value.copy() for key,value in case.items()}
                for key in ("full_target","pre_ot_full","C_vertices"):aug[key][...,0]*=np.float32(spec["alpha"])
                z=np.random.RandomState(spec["noise_seed"]).standard_normal((n-k,3,3)).astype(np.float32)
                eps=np.random.RandomState(spec["epsilon_seed"]).standard_normal(z.shape).astype(np.float32)
                t=float(samples["SEAM"]["t"])
                # The production helper casts time to FP32. The original same
                # scalar is reconstructed here to retain exact call identity.
                normal=float(np.random.RandomState(spec["time_seed"]).standard_normal());t=float(1/(1+np.exp(-normal)))
                reference=D.owner_sample(aug,z,eps,t,spec["permutation_seed"]);calls+=1
                assert set(reference)==set(samples["SEAM"])
                assert all(np.array_equal(reference[key],samples["SEAM"][key]) for key in reference)
                ordinary=D.legacy.supervised_sample(aug,z,eps,t,spec["permutation_seed"],"FULL4_C200");calls+=1
                assert all(np.array_equal(ordinary[key],samples["PLAIN"][key]) for key in ordinary)
                fixtures.append(dict(condition_id=row["condition_id"],kind="OWNER_ALIAS_THIRD_C" if special else "POSITIVE_OWNER",
                                     result="ALL_ARRAYS_BITWISE_EQUAL_EXISTING_CHAIR_OWNER_AND_PLAIN_IMPLEMENTATIONS",source_reasons=special))
                selected_positive=True
                if special:selected_extended=True
            rows.append(dict(condition_id=row["condition_id"],group=row["group"],N=n,K=k,boundary_edges=len(edges),
                             owners=int(case["occupied_labels"].sum()),plain_denominator=int(samples["PLAIN"]["free_coordinate_count"]),
                             seam_denominator=int(samples["SEAM"]["free_coordinate_count"]),status="PASS"))
            if (index+1)%50==0:print(json.dumps(dict(verified=index+1,OT_calls=calls)),flush=True)
        assert selected_positive and selected_extended and selected_closed
        schedule=D.read(out/"input_schedule.json")["steps"];train=manifest["train"]
        assert len(schedule)==24000
        counters=collections.Counter();cc=collections.Counter();pc=collections.Counter();prefix=[]
        for position,entry in enumerate(schedule,1):
            assert entry["step"]==position and len(entry["samples"])==8
            assert len({r["parent_id"] for r in entry["samples"]})==4
            for offset in range(0,8,2):
                first,second=entry["samples"][offset:offset+2]
                assert first["condition_id"]==second["condition_id"] and first["alpha"]==1. and .9<=second["alpha"]<=1.1
            for spec in entry["samples"]:
                row=train[spec["index"]]
                assert all(spec[key]==row[key] for key in ("condition_id","parent_id","part_type"))
                counters[row["parent_id"]]+=1;cc[row["condition_id"]]+=1;pc[(row["parent_id"],row["part_type"])]+=1
            if position in (3000,6000,12000,24000):
                prefix.append(dict(step=position,parents_seen=len(counters),conditions_seen=len(cc),
                                   parent_min=min(counters.values()),parent_max=max(counters.values()),
                                   condition_min=min(cc.values()),condition_max=max(cc.values())))
        assert len(counters)==64 and len(cc)==277 and max(counters.values())-min(counters.values())<=2
        for parent in counters:
            values=[value for (p,part),value in pc.items() if p==parent];assert max(values)-min(values)<=2
        for parent,part in pc:
            values=[cc[row["condition_id"]] for row in train if row["parent_id"]==parent and row["part_type"]==part]
            assert max(values)-min(values)<=2
        for path,expected in manifest["protected_inputs"].items():
            if D.sha(path)!=expected:raise ValueError("Protected original changed")
        result=dict(status="PASS",manifest_sha256=D.sha(out/"manifest.json"),verification_code_sha256=D.sha(__file__),
                    rows=rows,fixtures=fixtures,schedule_prefix_exposures=prefix,verified_conditions=len(rows),
                    CPU_OT_calls=calls,model_forwards=0,model_constructions=0,training_updates=0,GPU_jobs=0,
                    generated_test_noise_arrays=4*len(rows)+2*(len(fixtures)-1),
                    test_noise_scope="CPU algebra fixtures only, never evaluation or training sample identity",
                    source_inputs_unchanged=True,part_and_variant_round_robin_verified=True,
                    wall_seconds=time.perf_counter()-start,resources_before=before,resources_after=D.resources())
        D.write_new(target,result);return result
    except BaseException as error:
        stamp=str(time.time_ns())
        D.write_new(out/("cpu_verification_failure_"+stamp+".json"),dict(error=repr(error),traceback=traceback.format_exc(),
                    completed_conditions=len(rows),CPU_OT_calls=calls,model_forwards=0,GPU_jobs=0))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",default=str(D.DEFAULT_OUT));args=parser.parse_args()
    result=verify(args.out);print(json.dumps({key:result[key] for key in ("status","verified_conditions","CPU_OT_calls","wall_seconds")},indent=2))


if __name__=="__main__":main()
