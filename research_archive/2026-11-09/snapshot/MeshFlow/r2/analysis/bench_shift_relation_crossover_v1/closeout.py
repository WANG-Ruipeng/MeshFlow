"""CPU-only crossover delivery audit and fresh visual-review closeout.

Until all new images, actual reviews and a result-grounded decision note exist,
this command returns WAITING and writes no report or completion statement.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from . import run as A
from ..bench_joint_shift_v1.closeout import AXES, table, fmt

C=A.core


def readiness(exp):
    required=["results/completion.json","gallery/completion.json",A.DEST+"/completion.json",
              "visual_review/assignments.json","visual_review/protocol.md",
              "analysis_notes/decision_notes.md"]
    missing=[str(exp/p) for p in required if not (exp/p).exists()]
    pending=[]
    assignments=exp/"visual_review/assignments.json"
    if assignments.exists():
        d=C.read(assignments)
        for group,task in d["tasks"].items():
            path=exp/"visual_review"/task["record_file"]
            if not path.exists(): missing.append(str(path))
            elif len(C.read(path).get("records",[]))!=144: pending.append(group)
    return dict(status="WAITING" if missing or pending else "READY",missing=missing,pending=pending,
                model_calls=0,scoring_calls=0,report_written=False,completion_written=False)


def audit_calls(exp,rows):
    reg=C.read(exp/"registration.json")
    for key in ("protected","source_hashes"):
        A.verify_hashes(reg[key])
    qs=C.read(exp/"eval_manifest.json")["requests"]
    folder=exp/"evaluation"/A.CANDIDATE
    source_hashes={}; seen=set(); new_forwards=0
    for shard in range(2):
        p=folder/f"sampling_shard_{shard}"/"complete.json"
        d=C.read(p)
        assert d["status"]=="COMPLETE" and d["outputs"]==24
        assert d["new_forwards"]==600 and d["cached_prefix_forwards"]==600
        assert d["new_head_forwards"]==d["new_D0_forwards"]==0
        assert d["call_ledger"]==dict(started=600,completed=600)
        assert d["request_ids"]==[q["request_id"] for q in qs[24*shard:24*(shard+1)]]
        assert not seen & set(d["request_ids"])
        seen.update(d["request_ids"]); new_forwards+=d["new_forwards"]
        A.verify_hashes(d["files"]); source_hashes[str(p)]=C.sha(p)
    assert len(seen)==48 and new_forwards==1200
    d0=0
    for q in qs:
        req=folder/q["request_id"]
        raw_path=req/"raw/receipt.json"; raw=C.read(raw_path)
        assert raw["status"]=="COMPLETE" and raw["generator_forwards"]==25
        assert raw["cached_prefix_forwards"]==25 and raw["total_effective_generator_steps"]==50
        assert raw["D0_forwards"]==raw["head_forwards"]==0
        assert raw["prefix_model"]=="JOINT_SHIFT_S1500" and raw["suffix_model"]=="RELATION_V1_S1500"
        assert raw["prefix_state_index"]==25 and raw["source_labels_read"] is False
        assert raw["audit"]["anchored_all26_exact"] and raw["audit"]["RNG_unchanged"]
        A.verify_hashes(raw["files"])
        for field in ("prefix_trajectory_path","prefix_trajectory_sha256","prefix_receipt_sha256","noise_sha256"):
            assert raw[field]==q[field]
        prefix=C.resolve(q["prefix_trajectory_path"])
        assert C.sha(prefix)==q["prefix_trajectory_sha256"]
        with np.load(prefix,allow_pickle=False) as old, np.load(req/"raw/trajectory.npz",allow_pickle=False) as new:
            assert np.array_equal(old["state"][25],new["state"][0])
            assert len(new["state"])==26
            assert np.array_equal(new["absolute_steps"],np.arange(25,51))
        calls=[json.loads(line) for line in (req/"raw/calls.jsonl").read_text().splitlines() if line.strip()]
        assert len(calls)==50
        for event in ("FORWARD_STARTED","FORWARD_COMPLETE"):
            assert [r["absolute_step"] for r in calls if r["event"]==event]==list(range(25,50))
        post_path=req/"post/receipt.json"; post=C.read(post_path)
        assert post["D0_forwards"]==1 and post["request_id"]==q["request_id"]
        A.verify_hashes(post["files"]); d0+=1
        source_hashes[str(raw_path)]=C.sha(raw_path); source_hashes[str(post_path)]=C.sha(post_path)
    assert d0==48
    source_hashes[str(exp/"registration.json")]=C.sha(exp/"registration.json")
    return dict(status="PASS",training_updates=0,training_forwards=0,backwards=0,
                new_generator_forwards=1200,reused_prefix_steps=1200,new_RAW=48,new_D0=48,
                exact_cached_state25_checked=48,actual_score_rows=len(rows),source_hashes=source_hashes)


def visual_data(exp,numeric):
    vr=exp/"visual_review"; assignment=C.read(vr/"assignments.json")
    assert assignment["labels"]==list(A.LABELS) and assignment["stages"]==list(A.STAGES)
    assert assignment["protocol_sha256"]==C.sha(vr/"protocol.md")
    assert assignment["eval_manifest_sha256"]==C.sha(exp/"eval_manifest.json")
    gallery=C.read(exp/"gallery/completion.json")
    pages={p["page_id"]:p for p in gallery["images"]}; assert len(pages)==144
    numeric={(r["label"],r["stage"],r["request_id"]):r for r in numeric}
    reviews=[]; hashes={str(vr/"assignments.json"):C.sha(vr/"assignments.json"),
                        str(vr/"protocol.md"):C.sha(vr/"protocol.md")}
    for group,task in assignment["tasks"].items():
        path=vr/task["record_file"]; assert path.resolve().parent==vr.resolve()
        doc=C.read(path); assert doc["reviewer_kind"]=="AI_PROXY" and doc["blind"] is False
        expected={p["page_id"]:p for p in task["pages"]}
        assert len(doc["records"])==144 and len(expected)==48
        assert {r["page_id"] for r in doc["records"]}==set(expected)
        for r0 in doc["records"]:
            r=dict(r0); key=(r["label"],r["stage"],r["request_id"])
            p=pages[r["page_id"]]; n=numeric[key]; want=expected[r["page_id"]]
            assert r["actually_viewed"] is True and r["note"].strip()
            assert r["stage"]==p["sheet"]==want["stage"]
            for field in ("request_id","condition_id"):
                assert r[field]==p[field]==want[field]==n[field]
            assert r["parent_id"]==n["parent_id"] and r["parent_id"] in task["parents"]
            assert [c["model"] for c in p["cells"]]==list(A.LABELS)
            assert C.resolve(r["image_path"]).resolve()==C.resolve(p["path"]).resolve()
            assert r["image_sha256"]==p["image_sha256"]
            for axis,values in AXES.items(): assert r[axis] in values
            r.update(review_group=group,reviewer_kind="AI_PROXY",blind=False,
                     split=n["split"],part_type=n["part_type"],noise_index=n["noise_index"])
            reviews.append(r)
        hashes[str(path)]=C.sha(path)
    keys=[(r["label"],r["stage"],r["request_id"]) for r in reviews]
    assert len(keys)==len(set(keys))==432 and set(keys)==set(numeric)
    for p in pages.values():
        assert C.sha(C.resolve(p["path"]))==p["image_sha256"]
    hashes[str(exp/"gallery/completion.json")]=C.sha(exp/"gallery/completion.json")
    return reviews,hashes


def review_tables(reviews,sets):
    summaries=[]; pairs=[]; fixed=[]
    by={(r["label"],r["stage"],r["request_id"]):r for r in reviews}
    for label in A.LABELS:
        for stage in A.STAGES:
            rs=[r for r in reviews if r["label"]==label and r["stage"]==stage]
            groups=list(C.panel_rows(rs))+[(dict(panel=k,part_type="ALL_PARTS"),[r for r in rs if r["request_id"] in ids]) for k,ids in sets.items()]
            for panel,group in groups:
                for axis,values in AXES.items():
                    for value in values:
                        summaries.append(dict(label=label,stage=stage,**panel,axis=axis,value=value,
                            count=sum(r[axis]==value for r in group),requests=len(group)))
    for b in reviews:
        if b["label"]!=A.CANDIDATE: continue
        for ref in A.REFERENCES:
            a=by[(ref,b["stage"],b["request_id"])]
            p={k:b[k] for k in ("request_id","condition_id","parent_id","part_type","split","stage")}
            p.update(reference=ref,candidate=A.CANDIDATE,reference_note=a["note"],candidate_note=b["note"])
            for axis in AXES:
                p[axis+"__reference"]=a[axis]; p[axis+"__candidate"]=b[axis]
                p[axis+"__transition"]=a[axis]+" -> "+b[axis]
            pairs.append(p)
            for name,ids in sets.items():
                if p["request_id"] in ids: fixed.append(dict(fixed_set=name,**p))
    return summaries,pairs,fixed


def reference_review_changes(reviews):
    """Disclosure only, after the new review is complete; never a label source."""
    path=A.PRIOR/"visual_review/combined.json"
    prior={(r["label"],r["stage"],r["request_id"]):r for r in C.read(path)["records"]}
    changed=[]
    for r in reviews:
        if r["label"] not in A.REFERENCES:continue
        old=prior[(r["label"],r["stage"],r["request_id"])]
        for axis in AXES:
            changed.append(dict(label=r["label"],stage=r["stage"],request_id=r["request_id"],
                parent_id=r["parent_id"],part_type=r["part_type"],axis=axis,
                previous=old[axis],current=r[axis],changed=old[axis]!=r[axis]))
    return changed,{str(path):C.sha(path)}


def report(exp,data,reviews,summary,audit,history):
    decision=(exp/"analysis_notes/decision_notes.md").read_text(encoding="utf-8")
    assert decision.strip(),"A result-grounded decision note is required"
    lines=["# SHIFT→RELATION：固定t=0.5零训练交叉采样", "",decision,"",
        "## 方法、预算和证据边界", "",
        "仅一个固定候选SHIFT_RELATION_T050：前25步复用冻结SHIFT真实state[25]，后25步用冻结RELATION，"
        "标准Euler50总时长、C/N、噪声、C-only布局和后处理保持固定。没有线性混权、重抽噪声、扫描切换点或新训练。"
        "新增48条后半轨迹共1200生成前向、48单次D0；复用的1200前半步不能算新增前向。", "",
        "参考为RELATION_V1_S1500与JOINT_SHIFT_S1500；全部48请求、RAW/BASE_POST/FINAL均保留。"
        "历史TRAIN4父16请求、DEV8父32请求是既有开发面板，不是独立论文测试。"
        "BASE包含D0、恢复C/seam、GEO_C和定向，不能把阶段变化只归D0。", "",
        "所有177原生数值字段和NA原因原样保存。主聚合是噪声→条件→父等权；I主表只用PASS_ONLY父宏，"
        "LOW_SUPPORT及双方PASS配对另存。绝对FF/CF面对、保留/删面和面积同时保留。", "",
        "## 三阶段完整对照", ""]
    table_rows=[]
    for stage in A.STAGES:
        for panel in ("ALL","TRAIN","DEV"):
            for label in A.LABELS:
                rs=[r for r in data["rows"] if r["stage"]==stage and r["label"]==label and (panel=="ALL" or r["split"]==panel)]
                macro=lambda m:C.statistics_for(rs,m)["parent_macro"]
                valid=[r for r in rs if r["I_status"]=="PASS"]
                table_rows.append([stage,panel,label,fmt(macro("FF_illegal_affected_free_fraction"),True),
                    fmt(macro("CF_illegal_affected_free_fraction"),True),fmt(C.statistics_for(valid,"ConflictCov_C")["parent_macro"],True),
                    f"{len(valid)}/{len(rs)}",sum(r["FF_illegal_pair_count"] for r in rs),sum(r["CF_illegal_pair_count"] for r in rs),
                    fmt(macro("actual_free_faces")),fmt(macro("deleted_free_faces"))])
    lines += [table(["阶段","面板","模型","FF%","CF%","I PASS父宏%","I支持","FF对数","CF对数","free面","删面"],table_rows),"",
        "部位/逐父/逐请求、两参考配对及全部177字段见analysis_crossover_v1；单项下降不自动等同整体改进。", "",
        "## 本轮实际图审与固定采用集合", "",
        "新144页、432模型单元均实际查看；非盲AI_PROXY，五轴分别标注并保留UNCERTAIN。"
        "当前参考也重新评阅，不复制旧标签；图像不能认证精确相交/索引连接。"
        "固定gain9和original_positive8来自上轮DEV FINAL，选择先于本候选输出；它们只是预先固定子集，"
        "不新增成功定义，也不能用新的主观标签重选成员。", ""]
    vr=[]
    for s in summary:
        if s["part_type"]=="ALL_PARTS" and s["stage"]=="CLEAN_FINAL_POST":
            vr.append([s["panel"],s["label"],s["axis"],s["value"],s["count"],s["requests"]])
    lines += [table(["面板/固定集","模型","轴","枚举","数量","请求"],vr),"",
        "完整三阶段五轴、逐请求转移与固定集每个请求的采用/质量关系见visual_review。"
        "旧RELATION/SHIFT匿名两列复核若另行完成，是不同评阅证据；不替代新候选的三模型实际图审。", "",
        f"相同参考网格重新评阅，{len(history)}个五轴单项中有{sum(r['changed'] for r in history)}项"
        "与上一轮非盲标签不同；这不是参考模型变化。逐项披露见prior_reference_review_changes.csv。"
        "模型间比较只能使用本轮同页评阅，不把旧报告成功数与当前候选拼接，也不将UNCERTAIN自动算成功。", "",
        "## 调用账本、交付与恢复", "",
        f"CPU核对48份prefix state[25]逐值相同、每份后25步调用闭合；新增生成前向{audit['new_generator_forwards']}，"
        f"D0={audit['new_D0']}，训练/反向=0。收口自身模型调用0、评分0。未切默认、未commit/push。", "",
        "完整数值：results/per_output.csv及analysis_crossover_v1；图集：gallery；实际图审：visual_review；"
        "科学冻结：registration.json、PROTOCOL.md、eval_manifest.json；调用审计：analysis_notes/delivery_audit.json。"
        "只读完成核对：python -B -m r2.analysis.bench_shift_relation_crossover_v1.closeout --check-only。"
        "不要重新启动科学workflow重复采样。本轮完成后停止，不从结果自动选择另一个切换时间。", ""]
    return "\n".join(lines)


def run(exp=A.DEFAULT,check_only=False):
    exp=C.resolve(exp).resolve(); ready=readiness(exp)
    if ready["status"]!="READY": return ready
    data=A.load_inputs(exp)
    if data["status"]!="READY": return data
    analysis=C.read(exp/A.DEST/"completion.json"); A.verify_hashes(analysis["files"])
    A.verify_hashes(analysis["collector_input_hashes"]); A.verify_hashes(analysis["analysis_code_hashes"])
    reviews,hashes=visual_data(exp,data["rows"])
    summaries,pairs,fixed=review_tables(reviews,data["fixed_sets"])
    history,old_hashes=reference_review_changes(reviews); hashes.update(old_hashes)
    notes=exp/"analysis_notes/decision_notes.md"; hashes[str(notes)]=C.sha(notes)
    audit=audit_calls(exp,data["rows"])
    text=report(exp,data,reviews,summaries,audit,history)
    if check_only:return dict(status="PASS",pages=144,review_cells=432,numeric_rows=432,report_written=False,model_calls=0,scoring_calls=0)
    vr=exp/"visual_review"
    tables={"combined":reviews,"summary":summaries,"paired_per_output":pairs,
            "fixed_set_transitions":fixed,"prior_reference_review_changes":history}
    outputs=[]
    for name,rows in tables.items():
        path=vr/(name+".csv"); C.write_csv(path,rows); outputs.append(path)
    C.write_json(vr/"combined.json",dict(reviewer_kind="AI_PROXY",blind=False,records=reviews)); outputs.append(vr/"combined.json")
    C.write_json(vr/"validation.json",dict(status="PASS",pages=144,cells=432,source_hashes=hashes)); outputs.append(vr/"validation.json")
    C.write_json(exp/"analysis_notes/delivery_audit.json",audit); outputs.append(exp/"analysis_notes/delivery_audit.json")
    C.write_bytes(exp/"REPORT.md",text.encode("utf-8")); outputs.append(exp/"REPORT.md")
    A.verify_hashes(hashes); A.verify_hashes(data["source_hashes"])
    result=dict(status="COMPLETE",experiment="R2_BENCH_SHIFT_RELATION_CROSSOVER_V1",
        completed_utc=datetime.now(timezone.utc).isoformat(),training_updates=0,new_generator_forwards=1200,new_D0=48,
        numeric_rows=432,numeric_fields=177,actual_review_pages=144,actual_review_cells=432,
        source_review_hashes=hashes,source_data_hashes=data["source_hashes"],
        files={str(p):C.sha(p) for p in outputs},closeout_sha256=C.sha(__file__),
        model_calls_by_closeout=0,score_calls_by_closeout=0,default_changed=False,commit_or_push=False)
    done=exp/"COMPLETE.json"
    if done.exists():
        old=C.read(done)
        for k in ("files","source_review_hashes","source_data_hashes","closeout_sha256"):assert old[k]==result[k]
        return old
    C.write_json(done,result); return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--experiment",default=str(A.DEFAULT));p.add_argument("--check-only",action="store_true")
    args=p.parse_args(); d=run(args.experiment,args.check_only)
    print(json.dumps({k:v for k,v in d.items() if not k.endswith("hashes") and k!="files"},indent=2))
