"""Aggregate immutable A-panel scores only; no ML, OT or geometry imports."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]=""
import csv,json,math,hashlib,statistics,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
R=Path("/mnt/e/MeshFlow")
S=R/"native_t1/runs/chair_domain_control_v1"
O=R/"native_t1/analysis/chair_control_followup_v1/coverage_audit"
FILES=["A_context_per_case.csv","A_context_per_parent.csv","matched_budget_per_case.csv","matched_budget_per_parent.csv","shared_start_endpoint_per_case.csv","shared_start_endpoint_per_parent.csv","panels.json"]
M={
"coverage_pp":("fixed_R12_coverage_within_1pct_fraction",100,1),
"gt_to_free_RMS_pctL":("R12_to_free_RMS_bbox_pct",1,-1),
"fixed_GT_to_free_RMS_pctL":("fixed_R12_coverage_RMS_bbox_pct",1,-1),
"symmetric_RMS_pctL":("remainder_symmetric_RMS_bbox_pct",1,-1),
"free_to_GT_RMS_pctL":("free_to_R12_RMS_bbox_pct",1,-1),
"Gamma_edge_RMS_pctL":("Gamma_to_free_edges_RMS_bbox_pct",1,-1),
"Gamma_edge_p95_pctL":("Gamma_to_free_edges_p95_bbox_pct",1,-1),
"strict_Gamma_coverage_pp":("Gamma_to_free_edges_coverage_0.001_fraction",100,1),
"CF_affected_pp":("CF_proper_affected_free_fraction",100,-1),
"FF_affected_pp":("FF_illegal_affected_free_fraction",100,-1),
"small_angle_lt5_pp":("free_minimum_angle_lt5_fraction",100,-1),
"free_area_ratio":("free_area_over_GT_free_area",1,0)}
COMP={"Geo_minus_T1":("T1","Geo","Native+Geo_minus_Native"),"Full_minus_Geo":("Geo","Full","Native+Geo+Joint_minus_Native+Geo")}
ARMS={"T1":"Native","Geo":"Native+Geo","Full":"Native+Geo+Joint"}
FLAGS=["coverage_up_gt_RMS_worse","coverage_up_fixed_GT_RMS_worse","coverage_up_interface_better",
"coverage_up_interface_better_FF_worse","coverage_interface_FF_all_improve","coverage_interface_FF_all_nonworse",
"coverage_strictGamma_FF_all_improve","coverage_GT_interface_FF_all_improve","coverage_up_FF_worse",
"symmetric_RMS_better_FF_worse","interface_better_strictGamma_worse"]
CHECK=defaultdict(int)
EPS=1e-12
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def num(v):
    if v in (None,"","NA","None","nan"): return None
    f=float(v)
    assert math.isfinite(f),v
    return f
def avg(v):
    v=[x for x in v if x is not None]
    return statistics.fmean(v) if v else None
def equal(a,b,label):
    a,b=num(a),num(b)
    assert (a is None)==(b is None),(label,a,b)
    if a is not None: assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(label,a,b)
    CHECK[label]+=1
def index(rows,keys):
    out={tuple(r[k] for k in keys):r for r in rows}
    assert len(out)==len(rows),keys
    CHECK["unique_keys"]+=len(rows)
    return out
def write(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (O/name).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:("NA" if row.get(k) is None else row.get(k)) for k in keys})
def add_flags(r):
    ds={m:r[m+"_delta"] for m in M}
    if any(v is None for v in ds.values()):
        r.update({f:None for f in FLAGS});return r
    c,g,e,ff=[ds[k] for k in ("coverage_pp","gt_to_free_RMS_pctL","Gamma_edge_RMS_pctL","FF_affected_pp")]
    vals=[c>EPS and g>EPS,c>EPS and ds["fixed_GT_to_free_RMS_pctL"]>EPS,c>EPS and e<-EPS,
          c>EPS and e<-EPS and ff>EPS,c>EPS and e<-EPS and ff<-EPS,c>=-EPS and e<=EPS and ff<=EPS,
          c>EPS and ds["strict_Gamma_coverage_pp"]>EPS and ff<-EPS,
          c>EPS and g<-EPS and e<-EPS and ff<-EPS,c>EPS and ff>EPS,
          ds["symmetric_RMS_pctL"]<-EPS and ff>EPS,e<-EPS and ds["strict_Gamma_coverage_pp"]<-EPS]
    r.update(dict(zip(FLAGS,vals)));return r
def pair(r,b,a):
    for m,(field,scale,_) in M.items():
        x,y=num(b.get(field)),num(a.get(field))
        r[m+"_before"]=x*scale if x is not None else None
        r[m+"_after"]=y*scale if y is not None else None
        r[m+"_delta"]=(y-x)*scale if x is not None and y is not None else None
    return add_flags(r)
def groupkey(r):return (r["comparison"],r["split"],r["ratio"])
def parents(cases):
    groups=defaultdict(list)
    for r in cases:groups[(*groupkey(r),r["uid"])].append(r)
    out=[]
    for (comp,split,ratio,uid),rs in sorted(groups.items()):
        valid=[r for r in rs if r["status"]=="PASS"]
        p=dict(analysis=rs[0]["analysis"],comparison=comp,split=split,ratio=ratio,uid=uid,
               status="PASS" if valid else "NOT_APPLICABLE",reason=rs[0]["reason"],
               requested_case_pairs=len(rs),valid_case_pairs=len(valid),case_ids=";".join(r["case_id"] for r in rs))
        for m in M:
            for s in ("before","after","delta"):p[m+"_"+s]=avg(r[m+"_"+s] for r in valid)
        for f in FLAGS:p[f+"_case_count"]=sum(r[f] is True for r in valid)
        out.append(add_flags(p))
    return out
def summarize(cases,pars):
    out=[]
    for key in sorted({groupkey(r) for r in pars}):
        cs=[r for r in cases if groupkey(r)==key];ps=[r for r in pars if groupkey(r)==key]
        levels=[("case_all_noise",cs),("parent_mean",ps)]
        levels += [("noise_"+n,[r for r in cs if r["noise_index"]==n]) for n in sorted({r["noise_index"] for r in cs})]
        for level,rs in levels:
            valid=[r for r in rs if r["status"]=="PASS"]
            q=dict(analysis=rs[0]["analysis"],comparison=key[0],split=key[1],ratio=key[2],level=level,
                   requested_count=len(rs),valid_count=len(valid),NA_count=len(rs)-len(valid))
            for m,(_,_,d) in M.items():
                ds=[r[m+"_delta"] for r in valid]
                for suffix in ("before","after","delta"):q[m+"_"+suffix+"_mean"]=avg(r[m+"_"+suffix] for r in valid)
                q[m+"_delta_median"]=statistics.median(ds) if ds else None
                q[m+"_delta_min"]=min(ds) if ds else None;q[m+"_delta_max"]=max(ds) if ds else None
                q[m+"_improved_count"]=sum(v*d>EPS for v in ds) if d else None
                q[m+"_worse_count"]=sum(v*d<-EPS for v in ds) if d else None
                q[m+"_tie_count"]=sum(abs(v)<=EPS for v in ds)
            for f in FLAGS:q[f+"_count"]=sum(r[f] is True for r in valid)
            out.append(q)
    return out
def contributions(pars):
    out=[]
    for p in pars:
        peers=[r for r in pars if groupkey(r)==groupkey(p) and r["status"]=="PASS"]
        for m,(_,_,direction) in M.items():
            ds=[r[m+"_delta"] for r in peers];d=p[m+"_delta"];a=p[m+"_after"]
            positive=sum(max(v,0) for v in ds);negative=sum(max(-v,0) for v in ds);net=sum(ds)
            others=[r[m+"_delta"] for r in peers if r["uid"]!=p["uid"]]
            out.append(dict(analysis=p["analysis"],comparison=p["comparison"],split=p["split"],ratio=p["ratio"],
                uid=p["uid"],status=p["status"],reason=p["reason"],metric=m,direction=direction,
                valid_parent_count=len(peers),before=p[m+"_before"],after=a,delta=d,
                contribution_to_group_mean=d/len(peers) if d is not None else None,
                signed_share_of_net_delta=d/net if d is not None and abs(net)>EPS else None,
                share_of_positive_deltas=max(d,0)/positive if d is not None and positive>EPS else None,
                share_of_negative_magnitude=max(-d,0)/negative if d is not None and negative>EPS else None,
                delta_descending_rank=1+sum(v>d+EPS for v in ds) if d is not None else None,
                endpoint_descending_rank=1+sum(r[m+"_after"]>a+EPS for r in peers) if a is not None else None,
                group_mean_delta=avg(ds),leave_one_parent_out_mean_delta=avg(others) if d is not None else None,
                group_mean_sign_flips_without_parent=avg(ds)*avg(others)<0 if d is not None and others else None))
    return out
def fmt(x):return "NA" if x is None else f"{x:+.4f}"
def report(summ,pars,ss,sp,registered):
    lines=["# A 面板覆盖、接口与相交取舍复核","",
    "本报告只聚合冻结 CSV 中的已有分数，没有训练、模型前向、生成、OT、距离计算或几何重评分。终点比较为各臂新增 3000 更新后，起终点为各自旧 step1000 到累计 step4000；差值均为后者减前者。",
    "",
    "覆盖为固定 GT 自由面采样点在 1% bbox 对角线阈值内的比例，转为百分点。反馈中的 GT→free RMS 对应 R12_to_free_RMS_bbox_pct，单位 %L；固定覆盖点另有 fixed_R12_coverage_RMS_bbox_pct，已单列，两种既存采样口径不混用。接口为 Gamma→自由边 RMS（%L），并另列严格 Gamma 覆盖。FF 为 FF_illegal_affected_free_fraction，CF 为 CF_proper_affected_free_fraction，均转为百分点。",
    "",
    "逐父先平均两份噪声，再对有效父等权平均。每组登记 8 父/16 请求；A20 全部有效，A40 训练 7 父/14 请求、验证 6 父/12 请求。全部父及 NA 保留，NA 不补零、不进入数值均值。两个噪声不等于独立训练重复。方向判据为绝对差大于 1e-12，只有浮点相等判断意义，不代表统计显著性。",
    "",
    "三项同改善：覆盖增加、接口 RMS 降低、FF 比例降低。严格接口三项同改善：覆盖增加、严格 Gamma 覆盖增加、FF 降低。另列四项同改善（还需 GT RMS 降低）。这些计数不表示整体质量已合格。",
    "",
    "|比较|划分 / C|有效父 / 登记|Δ覆盖 pp|Δ GT RMS %L|Δ接口 RMS %L|ΔFF pp|覆盖升而 GT RMS 变差：请求；父|三项同改善：请求；父|",
    "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for s in summ:
        if s["level"]!="parent_mean":continue
        c=next(r for r in summ if groupkey(r)==groupkey(s) and r["level"]=="case_all_noise")
        lines.append(f"|{s['comparison']}|{s['split']} / {int(float(s['ratio'])*100)}%|{s['valid_count']}/{s['requested_count']}|{fmt(s['coverage_pp_delta_mean'])}|{fmt(s['gt_to_free_RMS_pctL_delta_mean'])}|{fmt(s['Gamma_edge_RMS_pctL_delta_mean'])}|{fmt(s['FF_affected_pp_delta_mean'])}|{c['coverage_up_gt_RMS_worse_count']}/{c['valid_count']}；{s['coverage_up_gt_RMS_worse_count']}/{s['valid_count']}|{c['coverage_interface_FF_all_improve_count']}/{c['valid_count']}；{s['coverage_interface_FF_all_improve_count']}/{s['valid_count']}|")
    lines += ["","逐父均值贡献与尾部","","均值贡献为该父平均差值除以有效父数。去掉该父后的均值揭示对单父的敏感性，不是另一个筛选实验。尾部指父对象聚合指标的最大值，不能定位到某块表面。"]
    for s in summ:
        if s["level"]!="parent_mean":continue
        ps=[r for r in pars if groupkey(r)==groupkey(s) and r["status"]=="PASS"]
        lines += ["",f"- {s['comparison']} / {s['split']} / {int(float(s['ratio'])*100)}%："]
        for p in sorted(ps,key=lambda r:r["gt_to_free_RMS_pctL_delta"],reverse=True)[:2]:
            loo=avg(r["gt_to_free_RMS_pctL_delta"] for r in ps if r["uid"]!=p["uid"])
            lines.append(f"  - GT RMS 差值最高父 {p['uid']}：Δ={fmt(p['gt_to_free_RMS_pctL_delta'])}%L，均值贡献={fmt(p['gt_to_free_RMS_pctL_delta']/len(ps))}%L，去掉此父后均值={fmt(loo)}%L；同父 Δ覆盖={fmt(p['coverage_pp_delta'])} pp、Δ接口={fmt(p['Gamma_edge_RMS_pctL_delta'])}%L、ΔFF={fmt(p['FF_affected_pp_delta'])} pp。")
        tail=max(ps,key=lambda r:r["gt_to_free_RMS_pctL_after"])
        lines.append(f"  - 终点 GT RMS 最大父 {tail['uid']}：{tail['gt_to_free_RMS_pctL_after']:.4f}%L，比较前臂 {tail['gt_to_free_RMS_pctL_before']:.4f}%L。覆盖差值最高两父："+ "；".join(f"{p['uid']} {fmt(p['coverage_pp_delta'])} pp" for p in sorted(ps,key=lambda r:r["coverage_pp_delta"],reverse=True)[:2])+"。")
        for flag,label in [("coverage_interface_FF_all_improve","三项同改善"),("coverage_up_interface_better_FF_worse","覆盖/接口改善但 FF 恶化"),("coverage_up_gt_RMS_worse","覆盖升而 GT RMS 恶化"),("coverage_strictGamma_FF_all_improve","严格接口三项同改善")]:
            lines.append("  - "+label+"父："+("；".join(r["uid"] for r in ps if r[flag]) or "无")+"。")
    lines += ["","共有 A20 第一噪声起终点","","各臂、各划分都是 8 父/8 对。第二噪声没有起点对应，不混入学习增量。每臂从各自起点恢复，不能由各自变化的绝对大小直接推出结构优势。",
    "","|模型|划分|Δ对称 RMS %L|Δ GT RMS %L|Δ覆盖 pp|Δ接口 RMS %L|Δ严格 Gamma pp|ΔFF pp|覆盖升 / 对称 RMS 改善 / FF 恶化|三项同改善|",
    "|---|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for s in ss:
        if s["level"]!="parent_mean":continue
        lines.append(f"|{s['comparison']}|{s['split']}|{fmt(s['symmetric_RMS_pctL_delta_mean'])}|{fmt(s['gt_to_free_RMS_pctL_delta_mean'])}|{fmt(s['coverage_pp_delta_mean'])}|{fmt(s['Gamma_edge_RMS_pctL_delta_mean'])}|{fmt(s['strict_Gamma_coverage_pp_delta_mean'])}|{fmt(s['FF_affected_pp_delta_mean'])}|{s['coverage_pp_improved_count']}/8；{s['symmetric_RMS_pctL_improved_count']}/8；{s['FF_affected_pp_worse_count']}/8|{s['coverage_interface_FF_all_improve_count']}/8|")
    lines += ["","起终点逐父代价与反例（全部父见 start_endpoint_parent.csv）："]
    for s in ss:
        if s["level"]!="parent_mean":continue
        ps=[r for r in sp if groupkey(r)==groupkey(s)]
        desc="；".join(f"{r['uid']} ΔFF {fmt(r['FF_affected_pp_delta'])} pp、Δ覆盖 {fmt(r['coverage_pp_delta'])} pp、Δ接口 {fmt(r['Gamma_edge_RMS_pctL_delta'])}%L" for r in sorted(ps,key=lambda r:r["FF_affected_pp_delta"],reverse=True)[:2])
        lines.append(f"- {s['comparison']} / {s['split']}：覆盖升且接口变好但 FF 变差 {s['coverage_up_interface_better_FF_worse_count']}/8；接口 RMS 变好但严格 Gamma 覆盖变差 {s['interface_better_strictGamma_worse_count']}/8。FF 差值最高两父：{desc}。")
    lines += ["","NA 与解释边界",""]
    for p in registered:
        if p["status"]!="READY":lines.append(f"- {p['split']} / {int(float(p['ratio'])*100)}% 父 {p['uid']}：{p['reason']}。两噪声和三模型均 NA。")
    lines += ["","这次拆解只定位到父对象和噪声，不定位表面区域，不给出因果解释。覆盖阈值内比例上升可以与 GT RMS、FF 或严格接口变差并存。",
    "20% 和 40% 改变 C、自由补集与有效父集合，不能从横向差异推断增加条件面比例的因果效应。训练父完整形状已见，评价 C 未作为本轮训练任务；收益不等于新对象泛化。单训练种子和固定小面板不能支持统计显著性或通用质量主张。",
    "","文件说明","",
    "- comparison_case.csv / comparison_parent.csv：所有登记请求、父格的配对原值、差值、联合标记及 NA。",
    "- tradeoff_summary.csv：按比较、划分、比例和逐请求 / 各噪声 / 逐父统计均值、方向与联合计数。",
    "- parent_contributions.csv：每父每指标对均值的贡献、正负贡献份额、终点/差值排序、去一父均值。净变化接近零时有符号占比不稳定，应优先读绝对贡献和正负份额。",
    "- start_endpoint_case.csv / start_endpoint_parent.csv / start_endpoint_summary.csv / start_endpoint_parent_contributions.csv：A20 第一噪声的配对学习与代价。",
    "- registered_parents.csv：16 父、两比例，共 32 登记格。",
    "- audit.json：输入和产物 SHA、与原父表/差分表的独立核对、实际执行次数及零实验资源计数。",""]
    (O/"COVERAGE_AUDIT_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
def main():
    clock=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    before={n:sha(S/n) for n in FILES}
    data={}
    for n in FILES:
        if n.endswith(".csv"):
            with (S/n).open(encoding="utf-8-sig",newline="") as f:data[n]=list(csv.DictReader(f))
    panels=json.loads((S/"panels.json").read_text())
    pi=index(panels["A"],["case_id"]);ai=index(data[FILES[0]],["arm","case_id"])
    assert len(pi)==64 and len(ai)==192
    expected=set(panels["A_train_parents"]+panels["A_validation_parents"]);assert len(expected)==16
    for arm in ARMS:assert {k[1] for k in ai if k[0]==arm}=={k[0] for k in pi}
    for (arm,cid),r in ai.items():
        p=pi[(cid,)]
        for k in ("uid","split","noise_index"):assert str(r[k])==str(p[k])
        assert float(r["ratio"])==p["ratio"] and r["phase_step"]=="3000"
        assert (r["status"]=="PASS")==(p["status"]=="READY")
        if r["status"]=="PASS":assert r["input_sha256"]==p["input_sha256"]
        else:assert r["reason"]==p["reason"]
        CHECK["registration_identity"]+=1
    ap=index(data[FILES[1]],["arm","split","ratio","uid"])
    for key,p in ap.items():
        rs=[r for r in ai.values() if (r["arm"],r["split"],r["ratio"],r["uid"])==key];assert len(rs)==2
        valid=[r for r in rs if r["status"]=="PASS"]
        equal(len(valid),p["completed_cases"],"A_parent_count")
        for field,_,_ in M.values():equal(avg(num(r[field]) for r in valid),p[field],"A_parent_metric")
    mi=index(data[FILES[2]],["comparison","phase_step","table","case_id"])
    mpi=index(data[FILES[3]],["comparison","phase_step","table","split","ratio","uid"])
    cases=[]
    for comp,(ba,aa,oldcomp) in COMP.items():
        for (cid,),p in sorted(pi.items()):
            b,a=ai[(ba,cid)],ai[(aa,cid)];assert b["status"]==a["status"]
            r=dict(analysis="matched_endpoint",comparison=comp,before_arm=ba,after_arm=aa,
                   before_phase_step=3000,after_phase_step=3000,table="A",split=a["split"],ratio=a["ratio"],
                   uid=a["uid"],case_id=cid,noise_index=a["noise_index"],status=a["status"],reason=a["reason"],
                   input_sha256=a["input_sha256"],before_raw_sha256=b["raw_sha256"],after_raw_sha256=a["raw_sha256"],
                   before_score_sha256=b["score_sha256"],after_score_sha256=a["score_sha256"])
            if a["status"]=="PASS":assert a["input_sha256"]==b["input_sha256"]
            pair(r,b,a);old=mi[(oldcomp,"3000","A",cid)]
            assert (old["status"]=="PASS")==(r["status"]=="PASS")
            for m,(field,scale,_) in M.items():
                for suffix in ("before","after","delta"):
                    v=num(old[field+"_"+suffix])
                    equal(r[m+"_"+suffix],v*scale if v is not None else None,"matched_case_metric")
            cases.append(r)
    pars=parents(cases)
    for p in pars:
        old=mpi[(COMP[p["comparison"]][2],"3000","A",p["split"],p["ratio"],p["uid"])]
        equal(p["valid_case_pairs"],old["valid_case_pairs"],"matched_parent_count")
        for m,(field,scale,_) in M.items():
            v=num(old[field+"_delta"]);equal(p[m+"_delta"],v*scale if v is not None else None,"matched_parent_metric")
    starts=[]
    for old in data[FILES[4]]:
        if old["table"]!="A":continue
        assert old["ratio"]=="0.2" and old["noise_index"]=="0" and old["same_input"]=="True"
        assert old["before_phase_step"]=="0" and old["after_phase_step"]=="3000" and old["status"]=="PASS"
        a=ai[(old["arm"],old["case_id"])];assert a["raw_sha256"]==old["after_raw_sha256"]
        b={field:old[field+"_before"] for field,_,_ in M.values()}
        r=dict(analysis="shared_start_endpoint",comparison=old["arm"]+"_endpoint_minus_start",
            before_arm=old["arm"],after_arm=old["arm"],before_phase_step=0,after_phase_step=3000,table="A",
            split=old["split"],ratio=old["ratio"],uid=old["uid"],case_id=old["case_id"],noise_index="0",
            status="PASS",reason="NA",input_sha256=a["input_sha256"],
            before_raw_sha256=old["before_raw_sha256"],after_raw_sha256=old["after_raw_sha256"])
        pair(r,b,a)
        for m,(field,scale,_) in M.items():
            for suffix in ("after","delta"):equal(r[m+"_"+suffix],num(old[field+"_"+suffix])*scale,"start_endpoint_metric")
        starts.append(r)
    assert len(starts)==48 and len(pars)==64 and len(cases)==128
    sp=parents(starts)
    spi=index([r for r in data[FILES[5]] if r["table"]=="A"],["comparison","split","ratio","uid"])
    for p in sp:
        arm=p["comparison"].split("_")[0];old=spi[(ARMS[arm]+"_endpoint_minus_start",p["split"],p["ratio"],p["uid"])]
        equal(p["valid_case_pairs"],old["valid_case_pairs"],"start_parent_count")
        for m,(field,scale,_) in M.items():equal(p[m+"_delta"],num(old[field+"_delta"])*scale,"start_parent_metric")
    registered=[]
    for uid in sorted(expected):
        for ratio in ("0.2","0.4"):
            ps=[p for p in panels["A"] if p["uid"]==uid and str(p["ratio"])==ratio]
            assert len(ps)==2 and {p["noise_index"] for p in ps}=={0,1}
            registered.append(dict(uid=uid,split=ps[0]["split"],ratio=ratio,status=ps[0]["status"],
                reason=ps[0]["reason"],registered_cases=2,models=3,case_ids=";".join(p["case_id"] for p in ps)))
    summ=summarize(cases,pars);ss=summarize(starts,sp)
    outputs={"registered_parents.csv":registered,"comparison_case.csv":cases,"comparison_parent.csv":pars,
        "tradeoff_summary.csv":summ,"parent_contributions.csv":contributions(pars),
        "start_endpoint_case.csv":starts,"start_endpoint_parent.csv":sp,"start_endpoint_summary.csv":ss,
        "start_endpoint_parent_contributions.csv":contributions(sp)}
    for name,rs in outputs.items():write(name,rs)
    report(summ,pars,ss,sp,registered)
    after={n:sha(S/n) for n in FILES};assert before==after
    audit=dict(status="PASS",started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-clock,source=str(S),output=str(O),
        input_hashes_before=before,input_hashes_after=after,source_unchanged=True,script_sha256=sha(Path(__file__)),
        independent_numeric_checks=dict(CHECK),input_row_counts={n:len(rs) for n,rs in data.items()},
        output_row_counts={n:len(rs) for n,rs in outputs.items()},
        registered_A_parents=16,registered_parent_ratio_cells=32,original_endpoint_model_requests=192,
        original_endpoint_PASS=174,original_endpoint_NA=18,
        metric_mapping={m:dict(source_column=f,scale=s,improvement_direction=d) for m,(f,s,d) in M.items()},
        zero_resource_counts=dict(training_updates=0,model_forwards=0,complete_generations=0,OT_calls=0,
                                  geometry_score_calls=0,distance_queries=0,GPU_tasks=0),
        execution=dict(stdlib_only=True,thread_limit=1,CUDA_VISIBLE_DEVICES="",analysis_script_executions=1,
            failed_analysis_executions=0,pre_script_tool_orchestration_failures=1),
        direction_tolerance=EPS,tail_definition="Existing per-parent aggregate RMS ranking; no surface localization",
        outputs=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.iterdir()) if p.is_file() and p.name!="audit.json"])
    (O/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="PASS",elapsed_seconds=audit["elapsed_seconds"],checks=sum(CHECK.values()),output_rows=audit["output_row_counts"])))
if __name__=="__main__":main()
