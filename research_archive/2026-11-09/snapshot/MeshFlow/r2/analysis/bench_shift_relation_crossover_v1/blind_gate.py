"""Validate a complete anonymous review, then unblind exactly once on CPU.

No review inputs are read without --execute. Never reads a partial review to
make a pass/fail decision. Failure means STOP BEFORE GPU, not failed sampling.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from . import run as A
from ..bench_joint_shift_v1.closeout import AXES, table

C=A.core
REF="RELATION_V1_S1500"
SHIFT="JOINT_SHIFT_S1500"
MODELS=(REF,SHIFT)
SCHEMA="BENCH_CROSSOVER_ANONYMOUS_ADOPTION_GATE_V1"


def gate_from_pairs(pairs):
    """Only the pre-registered two-part gate; UNCERTAIN is never strict gain."""
    assert len(pairs)==32 and len({p["request_id"] for p in pairs})==32
    counts={model:sum(p[model+"__C_use"]=="PLAUSIBLE" for p in pairs) for model in MODELS}
    unique=[p for p in pairs if p[SHIFT+"__C_use"]=="PLAUSIBLE" and p[REF+"__C_use"]!="PLAUSIBLE"]
    strict=[p for p in pairs if p[SHIFT+"__C_use"]=="PLAUSIBLE" and p[REF+"__C_use"]=="REPLACED_OR_IGNORED"]
    both=[p for p in pairs if p[SHIFT+"__C_use"]==p[REF+"__C_use"]=="PLAUSIBLE"]
    parents=sorted({p["parent_id"] for p in strict})
    checks=dict(SHIFT_PLAUSIBLE_strictly_exceeds_RELATION=counts[SHIFT]>counts[REF],
                strict_REPLACED_to_PLAUSIBLE_in_at_least_2_parents=len(parents)>=2)
    identity=lambda p:{k:p[k] for k in ("request_id","parent_id","condition_id","part_type")}
    return dict(status="PASS" if all(checks.values()) else "FAIL",counts=counts,checks=checks,
        strict_gain_parents=parents,
        anonymous_shift_unique_positives=[identity(p) for p in sorted(unique,key=lambda p:p["request_id"])],
        anonymous_strict_gains=[identity(p) for p in sorted(strict,key=lambda p:p["request_id"])],
        anonymous_both_positive=[identity(p) for p in sorted(both,key=lambda p:p["request_id"])],
        later_candidate_rule="Retain at least 2/3 anonymous SHIFT unique positives, and every anonymous both-positive; other quality requirements remain in PROTOCOL. This gate does not evaluate a candidate.")


def validate_and_decode(exp,review_path):
    base=exp/"blind_review"; private=base/"private_identity"; public=base/"reviewer"
    # Count gate happens before accessing the private identity mapping.
    review=C.read(review_path)
    records=review.get("records",[])
    if len(records)!=64:
        return dict(status="WAITING_FOR_ALL_64_RECORDS",records=len(records),required=64)
    assert review["reviewer_kind"]=="AI_PROXY" and review["blind"] is True
    protocol=public/"review_protocol.md"
    assert review["protocol_sha256"]==C.sha(protocol)
    keys=[(r["page_id"],r["column"]) for r in records]
    expected={(f"page_{i:03d}",col) for i in range(1,33) for col in ("A","B")}
    assert len(set(keys))==64 and set(keys)==expected
    for r in records:
        assert r["actually_viewed"] is True and isinstance(r["note"],str) and r["note"].strip()
        for axis,allowed in AXES.items():assert r[axis] in allowed,(r["page_id"],axis)
    key_path=private/"identity_key.json"; mapping=C.read(key_path); key_hash=C.sha(key_path)
    done_path=private/"completion.json"; done=C.read(done_path)
    assert done["status"]=="ANONYMOUS_RENDER_COMPLETE_PENDING_REVIEW"
    assert done["pages"]==32 and done["cells"]==64
    assert done["identity_key_sha256"]==key_hash
    assert mapping["version"]=="existing_dev32_final_anonymous_pair_v1"
    assert mapping["requests"]==32 and mapping["cells"]==64 and mapping["models"]==list(MODELS)
    assert mapping["left_counts"]=={REF:16,SHIFT:16}
    A.verify_hashes(done["public_hashes"])
    assert set(p.name for p in public.iterdir())=={"review_protocol.md",*[f"page_{i:03d}.png" for i in range(1,33)]}
    hashes={str(p):C.sha(p) for p in (review_path,protocol,key_path,done_path,exp/"PROTOCOL.md")}
    mapping_jobs={j["opaque_id"]:j for j in mapping["jobs"]}
    assert set(mapping_jobs)=={k[0] for k in expected}
    receipts={}
    for page_id,job in mapping_jobs.items():
        rp=private/(page_id+".json"); receipt=C.read(rp)
        assert receipt["opaque_id"]==page_id and receipt["identity_key_sha256"]==key_hash
        image_path=public/(page_id+".png")
        assert C.resolve(receipt["image_path"]).resolve()==image_path.resolve()
        assert receipt["image_sha256"]==C.sha(image_path)
        assert C.resolve(job["source_png"]).is_file()
        assert C.sha(C.resolve(job["source_png"]))==job["source_png_sha256"]
        assert [d["column"] for d in receipt["display"]]==["A","B"]
        assert [d["model"] for d in receipt["display"]]==[c["model"] for c in job["cells"]]
        assert set(c["model"] for c in job["cells"])==set(MODELS)
        source=job["source_page"]
        assert source["split"]=="DEV" and source["sheet"]=="CLEAN_FINAL_POST"
        hashes[str(rp)]=C.sha(rp); hashes[str(image_path)]=receipt["image_sha256"]
        receipts[page_id]=receipt
    assert Counter(j["cells"][0]["model"] for j in mapping_jobs.values())==Counter({REF:16,SHIFT:16})
    decoded=[]
    for r in records:
        job=mapping_jobs[r["page_id"]]; page=job["source_page"]
        assert r["image_sha256"]==receipts[r["page_id"]]["image_sha256"]
        model=job["cells"][0 if r["column"]=="A" else 1]["model"]
        row=dict(r,label=model,stage="CLEAN_FINAL_POST",split="DEV",reviewer_kind="AI_PROXY",blind=True,
                 prior_exposure=r.get("prior_exposure",review.get("prior_exposure","NOT_REPORTED")))
        for field in ("request_id","parent_id","condition_id","part_type"):
            row[field]=page[field]
        decoded.append(row)
    assert len({r["request_id"] for r in decoded})==32
    assert len({r["parent_id"] for r in decoded})==8
    return dict(status="VALIDATED",rows=decoded,hashes=hashes,review_sha256=C.sha(review_path),
        reviewer_metadata={k:v for k,v in review.items() if k!="records"})


def tables(decoded):
    by={(r["label"],r["request_id"]):r for r in decoded}
    pairs=[]
    for req in sorted({r["request_id"] for r in decoded}):
        ref,shift=by[(REF,req)],by[(SHIFT,req)]
        p={k:ref[k] for k in ("request_id","parent_id","condition_id","part_type")}
        for axis in AXES:
            p[REF+"__"+axis]=ref[axis];p[SHIFT+"__"+axis]=shift[axis]
            p[axis+"__transition"]=ref[axis]+" -> "+shift[axis]
        p.update(reference_column=ref["column"],shift_column=shift["column"],page_id=ref["page_id"],
                 reference_note=ref["note"],shift_note=shift["note"])
        pairs.append(p)
    summary=[]
    for label in MODELS:
        selected=[r for r in decoded if r["label"]==label]
        groups=[("ALL_DEV32",selected)]+[(pid,[r for r in selected if r["parent_id"]==pid]) for pid in sorted({r["parent_id"] for r in selected})]
        for group,rs in groups:
            for axis,values in AXES.items():
                for value in values:
                    summary.append(dict(label=label,group=group,axis=axis,value=value,
                                        count=sum(r[axis]==value for r in rs),requests=len(rs)))
    prior_path=A.PRIOR/"visual_review/combined.json"
    prior={(r["label"],r["stage"],r["request_id"]):r for r in C.read(prior_path)["records"]}
    diff=[]
    for r in decoded:
        old=prior[(r["label"],r["stage"],r["request_id"])]
        for axis in AXES:
            diff.append(dict(label=r["label"],request_id=r["request_id"],parent_id=r["parent_id"],
                condition_id=r["condition_id"],part_type=r["part_type"],axis=axis,
                prior_nonblind_value=old[axis],current_anonymous_value=r[axis],changed=old[axis]!=r[axis],
                prior_note=old["note"],current_note=r["note"]))
    return pairs,summary,diff,{str(prior_path):C.sha(prior_path)}


def close_gate_failure(exp,gate):
    assert gate["status"]=="FAIL"
    # A gate stop must never conceal a partial model attempt or real output.
    if (exp/"workflow/state.json").exists():
        raise ValueError("Workflow was dispatched; cannot claim a zero-GPU gate stop")
    candidate=exp/"evaluation"/A.CANDIDATE
    if candidate.exists():
        forbidden=[]
        # prepare creates frozen, CPU-only layouts/<condition>/receipt.json.
        # Those receipts are not evidence of a generator dispatch. All other
        # candidate execution receipts/attempts remain disallowed, including
        # empty shard directories conservatively marking a started dispatch.
        for child in candidate.iterdir():
            if child.name=="layouts" and child.is_dir():
                continue
            if child.name.startswith("sampling_shard_"):
                forbidden.append(child)
                continue
            patterns=("calls.jsonl","trajectory.npz","raw.npz","receipt.json","attempt.json","failure.json")
            if child.is_file() and child.name in patterns:
                forbidden.append(child)
            elif child.is_dir():
                for pattern in patterns:
                    forbidden.extend(child.rglob(pattern))
        if forbidden:raise ValueError("Candidate artifacts exist; independently audit rather than declare zero calls")
    for p in (exp/"results/per_output.csv",exp/"gallery/completion.json"):
        if p.exists():raise ValueError("Candidate evaluation exists; gate-stop branch is not applicable")
    text="\n".join(["# 固定交叉采样：匿名复核门禁收口", "",
        "**采用正信号未通过本轮预先固定的有限匿名复核门槛，因此在GPU派发前停止。** "
        "这不是交叉采样失败，也不是零输出被记作负生成结果；候选根本没有执行。", "",
        f"全部32对/64列完成实际AI_PROXY匿名评阅。C_use=PLAUSIBLE：RELATION {gate['counts'][REF]}，SHIFT {gate['counts'][SHIFT]}。"
        f"严格REPLACED_OR_IGNORED→PLAUSIBLE覆盖{len(gate['strict_gain_parents'])}个DEV父。", "",
        "门槛为SHIFT采用总数严格更多，且严格改善覆盖至少2父；不以UNCERTAIN→PLAUSIBLE替代第二项。"
        "这是继续有限实验的工程门，不是统计显著或整体质量认证。原非盲评阅不覆盖，"
        "历史标签差异单独保存；匿名和既有非盲结果不强行合成一个成功率。", "",
        "本候选训练0、反向0、新生成器前向0、D0前向0、RAW/POST输出0、新评分0。"
        "核对无workflow派发状态、无候选调用/轨迹/结果产物。先前模型与其旧输出不属于本次新调用。", "",
        "逐请求和完整五轴见blind_review/gate_analysis/；原匿名记录及图像hash封存，未改标、补评或改门槛。"
        "未改变默认模型、未训练、未清理权重、未commit/push。任务在此停止。", ""])
    C.write_bytes(exp/"REPORT.md",text.encode("utf-8"))
    value=dict(status="COMPLETE_GATE_STOP_BEFORE_GPU",experiment="R2_BENCH_SHIFT_RELATION_CROSSOVER_V1",
        gate_status="FAIL",gate_sha256=C.sha(exp/"blind_review/gate.json"),
        actual_review_pairs=32,actual_review_cells=64,new_training_updates=0,new_generator_forwards=0,
        new_D0_forwards=0,new_RAW=0,new_POST=0,new_score_rows=0,
        candidate_sampled=False,scientific_interpretation="Anonymous adoption gate not passed; crossover not executed",
        report_sha256=C.sha(exp/"REPORT.md"),default_changed=False,commit_or_push=False)
    C.write_json(exp/"COMPLETE.json",value)
    return value


def run(exp=A.DEFAULT,execute=False,close_on_fail=False):
    if not execute:return dict(status="WAITING_FOR_EXPLICIT_EXECUTION",review_read=False,model_calls=0,scoring_calls=0)
    exp=C.resolve(exp).resolve(); base=exp/"blind_review"
    review_path=base/"review_records/review.json"
    if not review_path.exists():return dict(status="WAITING_FOR_COMPLETE_REVIEW",review_read=False)
    gate_path=base/"gate.json"
    if gate_path.exists():
        old=C.read(gate_path)
        assert old["script_sha256"]==C.sha(__file__)
        A.verify_hashes(old["sealed_input_hashes"]);A.verify_hashes(old["analysis_files"])
        assert old["review_sha256"]==C.sha(review_path)
        if close_on_fail and old["status"]=="FAIL":close_gate_failure(exp,old)
        return old
    decoded=validate_and_decode(exp,review_path)
    if decoded["status"]!="VALIDATED":return decoded
    pairs,summary,diff,prior_hashes=tables(decoded["rows"])
    decision=gate_from_pairs(pairs)
    protected=dict(decoded["hashes"]);protected.update(prior_hashes)
    reg_path=exp/"registration.json";reg=C.read(reg_path)
    protocol=exp/"PROTOCOL.md"
    registered=next((h for p,h in reg["protected"].items() if C.resolve(p).resolve()==protocol.resolve()),None)
    assert registered==C.sha(protocol),"Experiment gate protocol is not the frozen registered version"
    protected[str(reg_path)]=C.sha(reg_path)
    dest=base/"gate_analysis"
    exports=dict(decoded_records=decoded["rows"],paired_per_request=pairs,axis_summary=summary,
                 historical_label_differences=diff)
    for name,rows in exports.items():C.write_csv(dest/(name+".csv"),rows)
    seal=dict(status="SEALED_COMPLETE_ANONYMOUS_REVIEW",review_sha256=decoded["review_sha256"],
        input_hashes=protected,record_count=64,page_count=32,original_review_modified=False,
        reviewer_metadata=decoded["reviewer_metadata"],scope="Actual review assertions plus complete identity/hash validation; hashes alone cannot prove visual attention")
    C.write_json(base/"review_seal.json",seal)
    files={str(p):C.sha(p) for p in sorted(dest.iterdir()) if p.is_file()}
    files[str(base/"review_seal.json")]=C.sha(base/"review_seal.json")
    A.verify_hashes(protected)
    result=dict(schema=SCHEMA,**decision,completed_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=C.sha(protocol),review_protocol_sha256=decoded["reviewer_metadata"]["protocol_sha256"],
        review_sha256=decoded["review_sha256"],sealed_input_hashes=protected,analysis_files=files,
        script_sha256=C.sha(__file__),actual_review_pairs=32,actual_review_cells=64,
        source_review_modified=False,new_model_calls=0,new_scoring_calls=0,
        blind_limit="AI_PROXY anonymous review, not independent human ratings; recognition/prior exposure may remain",
        criterion_scope="Continue-or-stop gate only; no candidate quality or significance claim")
    C.write_json(gate_path,result)
    if close_on_fail and result["status"]=="FAIL":close_gate_failure(exp,result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment",default=str(A.DEFAULT));p.add_argument("--execute",action="store_true")
    p.add_argument("--close-on-fail",action="store_true")
    a=p.parse_args(); d=run(a.experiment,a.execute,a.close_on_fail)
    print(json.dumps({k:v for k,v in d.items() if k not in {"sealed_input_hashes","analysis_files",
        "anonymous_shift_unique_positives","anonymous_strict_gains","anonymous_both_positive"}},ensure_ascii=False,indent=2))
