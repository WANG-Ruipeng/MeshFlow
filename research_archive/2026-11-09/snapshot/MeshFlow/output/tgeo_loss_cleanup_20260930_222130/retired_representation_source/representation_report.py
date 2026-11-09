"""Presentation of saved analytic-geometry representation alignment results.

Consumes scalar records and untouched raw arrays for fixed-view figures only.
No model, auxiliary head, OT, autograd or geometry scoring is invoked here.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np

MODELS=("Geo_initial","G0","G_Angular","G_AngularScale")
PAIRINGS=(("G0","Geo_initial"),("G_Angular","G0"),("G_AngularScale","G_Angular"),("G_AngularScale","G0"))
SEEDS=tuple(range(9901,9909))
FREE="remainder_symmetric_RMS_bbox_pct"
TO="free_to_R12_RMS_bbox_pct"
FROM="R12_to_free_RMS_bbox_pct"
GAMMA="boundary_RMS_bbox_pct"
P95="boundary_p95_bbox_pct"
COVER="boundary_coverage_fraction"
CF="CF_proper_affected_free_fraction"
FF="FF_illegal_pair_count"
LABELS={FREE:"自由面对称RMS (% L)",TO:"free→目标 RMS (% L)",FROM:"目标→free RMS (% L)",
    "fixed_R12_coverage_RMS_bbox_pct":"固定目标点→free RMS (% L)",
    "fixed_R12_coverage_within_1pct_fraction":"固定目标1%L内覆盖",
    GAMMA:"Gamma RMS (% L)",P95:"Gamma p95 (% L)",COVER:"Gamma覆盖",
    "whole_symmetric_RMS_bbox_pct":"完整表面对称RMS (% L)",
    CF:"CF穿刺free比例","CF_proper_pair_count":"CF穿刺pair数",FF:"FF非法pair数",
    "CF_coplanar_pair_count":"CF共面重叠pair数","FF_coplanar_pair_count":"FF共面重叠pair数",
    "free_minimum_angle_lt5_fraction":"最小角<5°比例","free_q_gt10_fraction":"q>10比例",
    "free_total_surface_area":"free总面积（无好坏方向）"}
SURFACE=(FREE,TO,FROM,"fixed_R12_coverage_RMS_bbox_pct","fixed_R12_coverage_within_1pct_fraction",GAMMA,P95,COVER,"whole_symmetric_RMS_bbox_pct")
QUALITY=(CF,"CF_proper_pair_count",FF,"CF_coplanar_pair_count","FF_coplanar_pair_count","free_minimum_angle_lt5_fraction","free_q_gt10_fraction","free_total_surface_area")


def _fmt(v):
    if v is None:return "NOT_RECORDED"
    if isinstance(v,bool):return str(v)
    if isinstance(v,(int,np.integer)):return str(int(v))
    if isinstance(v,(float,np.floating)):
        if not np.isfinite(v):return "NONFINITE"
        return f"{v:.5g}"
    if isinstance(v,(dict,list)):v=json.dumps(v,ensure_ascii=False,separators=(",",":"))
    return str(v).replace("|",chr(92)+"|").replace(chr(10)," ")


def _table(headers,rows):
    return ["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |"]+[
        "| "+" | ".join(_fmt(x) for x in row)+" |" for row in rows]


def _record(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def _dist(value):
    if not value or not value.get("finite_count"):return "NOT_RECORDED"
    worst=value.get("worst")
    if isinstance(worst,dict):worst=worst.get("value")
    return " / ".join(_fmt(value.get(k)) for k in ("mean","median"))+" / "+("N/A" if worst is None else _fmt(worst))


def _paired_cell(block,key):
    d=block.get("metrics",{}).get(key,{})
    c=block.get("improvement_counts",{}).get(key,{})
    return _dist(d)+f"; {c.get('better','?')}/{c.get('available','?')}"


def _rows(r):
    if r.get("status")!="PASS_EXECUTION":raise ValueError("Report requires completed execution")
    expected={(m,p,j,s) for m in MODELS for p in (0,1) for j in (0,1,2) for s in (SEEDS if j==0 else SEEDS[:2])}
    def identity(x):return x["model"],int(x["parent"]),int(x["patch"]),int(x["seed"])
    rows=r.get("scores",[]); generated=r.get("generated",[])
    if len(rows)!=96 or len(generated)!=96 or {identity(x) for x in rows}!=expected or {identity(x) for x in generated}!=expected:
        raise ValueError("Expected exact96 model-parent-patch-seed rows")
    raw={identity(x):x for x in generated}
    for row in rows:
        if row["patch_group"]!=("original" if row["patch"]==0 else "new"):raise ValueError("Patch group differs")
        if Path(row["raw_path"]).resolve()!=Path(raw[identity(row)]["raw_path"]).resolve():raise ValueError("Score/raw identity differs")
        if any(row.get("scalars",{}).get(k) is None or not np.isfinite(row["scalars"][k]) for k in (FREE,TO,FROM,GAMMA,P95,COVER)):
            raise ValueError("Required saved geometric score missing")
    if set(r.get("summary",{}).get("patches",{}))!={"0","1","2"}:raise ValueError("Separate patch summaries missing")
    return rows


def _figures(r,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    selected={(x["model"],x["parent"],x["patch"]):x for x in r["generated"] if x["seed"]==SEEDS[0]}
    if len(selected)!=24:raise ValueError("Fixed first-noise plot matrix incomplete")
    arrays={}
    for key,row in selected.items():
        if _record(row["raw_path"])["sha256"]!=row["raw_sha256"]:raise ValueError("Raw plot file changed")
        with np.load(row["raw_path"],allow_pickle=False) as z:value=z["output"].copy()
        if value.dtype!=np.float32 or value.size!=112*9 or not np.isfinite(value).all():raise ValueError("Raw plot geometry invalid")
        arrays[key]=value.reshape(112,3,3)
    bounds={}
    for parent in (0,1):
        points=np.concatenate([v.reshape(-1,3) for (m,p,j),v in arrays.items() if p==parent])
        lo=points.min(0);hi=points.max(0);bounds[parent]=((lo+hi)/2,max(float((hi-lo).max())*.55,1e-6))
    records=[]
    for patch in (0,1,2):
        fig=plt.figure(figsize=(15,13))
        for parent in (0,1):
            for free_only in (False,True):
                for col,model in enumerate(MODELS):
                    v=arrays[model,parent,patch]; ax=fig.add_subplot(4,4,(2*parent+int(free_only))*4+col+1,projection="3d")
                    ax.add_collection3d(Poly3DCollection(v[12:],facecolors="#7eabc7",edgecolors="#405266",linewidths=.24,alpha=.9))
                    if not free_only:ax.add_collection3d(Poly3DCollection(v[:12],facecolors="#f0a348",edgecolors="#804800",linewidths=.5,alpha=1.))
                    center,radius=bounds[parent]
                    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
                    ax.set_box_aspect((1,1,1));ax.set_proj_type("ortho");ax.view_init(elev=20,azim=-55);ax.set_axis_off()
                    ax.set_title(f"P{parent} | {model} | {'free100' if free_only else 'full112'}",fontsize=9)
        fig.suptitle(f"Fixed seed{SEEDS[0]} | patch{patch} | raw full/free, no repair\nIdentical per-parent bounds across all three patches",fontsize=12)
        fig.subplots_adjust(left=.01,right=.99,bottom=.005,top=.94,wspace=.01,hspace=.06)
        path=Path(out)/f"comparison_seed{SEEDS[0]}_patch{patch}_full_free.png"
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=170,facecolor="white");plt.close(fig)
        records.append(dict(**_record(path),patch=patch,seed=SEEDS[0],panels=16,visual_review="PENDING_CALLER_REVIEW"))
    return records


def _facts(summary):
    lines=[]
    for candidate,reference in PAIRINGS:
        for patch in (0,1,2):
            block=summary["patches"][str(patch)]["paired"][candidate+"_minus_"+reference]
            text=[]
            for parent in ("0","1"):
                q=block["parents"][parent]; d=q["metrics"]; c=q["improvement_counts"]
                text.append(f"P{parent} RMS均差{_fmt(d[FREE]['mean'])}（改善{c[FREE]['better']}/{c[FREE]['available']}）、Gamma均差{_fmt(d[GAMMA]['mean'])}、覆盖均差{_fmt(d[COVER]['mean'])}、CF比例均差{_fmt(d[CF]['mean'])}、FF均差{_fmt(d[FF]['mean'])}")
            lines.append(f"{candidate}−{reference} / patch{patch}："+"；".join(text)+"。")
    return lines


def _json_scalar_table(mapping):
    return [[k,v] for k,v in mapping.items() if not isinstance(v,(dict,list))]


def _loader_commands(r):
    """Independent POSIX processes; pass each checkpoint as a quoted argv."""
    import shlex
    program = ("import sys; from native_t1.representation_checkpoint import load_representation_generator; "
               "model, audit = load_representation_generator(sys.argv[1], device='cuda'); print(audit)")
    return [(model,shlex.join(["python","-c",program,str(r["arms"][model]["checkpoints"]["500"]["path"])]))
            for model in MODELS[1:]]


def render_report(r,figures=()):
    rows=_rows(r); summary=r["summary"]; protocol=r.get("protocol",{}); arms=r.get("arms",{})
    if any(arms.get(m,{}).get("completed_updates")!=500 or arms[m].get("status")!="PASS" for m in MODELS[1:]):
        raise ValueError("Three fixed500-update endpoints required")
    lines=["# GEOMETRY_REP_ALIGNMENT_REPORT","","方法身份：**analytic-geometry representation alignment pilot**。本轮新增三臂各500次生成器更新，共1500次；相同T_geo累计2000步起点，终点累计2500步。", "",
        "借鉴 [TORA](https://arxiv.org/html/2604.04050v1) 的训练期中间特征监督与 [Geometry Forcing](https://arxiv.org/html/2507.07982v2) 的方向/幅值对齐思路；未复现完整方法、Uni3D/VGGT教师或CKA。教师是固定解析GT几何描述，无外部教师权重。", ""]
    questions=("方向对齐是否改善同预算纯FM？","加Scale是否进一步受益或引入代价？","属性学习的改善是否进入完整生成？","原片尾部、新片覆盖与Gamma是否改善？","推理是否仍是原A+Geo、50次前向且无需教师？","当前值得保留哪一臂，哪些结论仍不成立？")
    interpretation=r.get("interpretation",{});answers=interpretation.get("answers",[])
    for i,q in enumerate(questions):
        answer=answers.get(str(i+1)) if isinstance(answers,dict) else answers[i] if i<len(answers) else None
        lines += [f"{i+1}. **{q}** "+(str(answer) if answer is not None else "尚未登记综合解释；下列完整生成配对表是证据，不自动据单一均值选赢家。"),""]
    if interpretation.get("next_step"):lines += [str(interpretation["next_step"]),""]
    lines += ["## 设计、对应与解释边界","",
        "G0只训练原masked FM；G_Angular加投影头P的描述方向监督；G_AngularScale再加S的完整13维幅值恢复。Angular是描述向量方向，不是直接网格法向角；Scale不是只预测缩放、边长或面积。",
        "读取第9个block完成、真正送入第10层的corner hidden；辅助分支单独FP32转换和三角角均值，不替换主干BF16 hidden。解析目标来自干净GT自由面，源ID仅用于监督对应。输入Geo特征与权重含义未改。",
        "三臂恢复同一seed1010输入流，fresh AdamW，effective8/micro1，原FM逐自由坐标分母。辅助损失用整批自由面分母，Scale再除13；生成器/辅助头分别clip1。辅助预热按新增optimizer步min(s/50,1)，不是Flow时间门控。",
        "G_AngularScale−G_Angular同时增加监督项与总辅助强度，不能单独归因幅值机制。无错配教师训练臂、无外部3D教师、无独立训练seed复现。额外片和完整父对象以前见过，只有本轮噪声新增；不是新对象泛化。",""]
    streams=[arms[m].get("stream_audits",[]) for m in MODELS[1:]]
    lines += [f"500步输入流逐条相同：{len(streams[0])==500 and streams[0]==streams[1]==streams[2]}。",
        "旧BF16主干面重排FAIL沿用baseline inheritance边界，未改容差或重新声称已修复。新增特征返回/标签/梯度/保存重载门禁单独记录。",""]
    lines += _table(["臂","新增更新","生成器参数","辅助参数","训练秒"],[[m,arms[m].get("completed_updates"),arms[m].get("generator_parameters",arms[m].get("parameters")),arms[m].get("auxiliary_parameters"),arms[m].get("training_seconds")] for m in MODELS[1:]])
    lines += ["","## 零更新标定与冻结目标","","量级标定读真实H9的梯度，最多8次前向与24次autograd.grad；零更新不等于零反向。lambda在看生成结果前固定，两辅助臂共享lambda_A。初始化的5%隐藏梯度目标不是全模型参数梯度保证，也不是后续每步保证。",""]
    calibration=r.get("calibration",{})
    lines += _table(["标定记录","值"],_json_scalar_table(calibration)) if isinstance(calibration,dict) else ["标定记录见results.json引用。"]
    lines += [""]+_table(["样本","父","K","t","g_FM(H9)","g_A(H9)","g_S(H9)","L_FM","L_A","L_S"],
        [[v.get(k) for k in ("sample_index","parent_index","K","t","g_FM","g_A","g_S","L_FM","L_A","L_S")] for v in calibration.get("rows",[])])+[""]
    medians=calibration.get("gradient_medians",calibration.get("medians",{}))
    lines += [f"H9范数中位数：{_fmt(medians)}。单状态标定使用自身损失分母；正式训练使用effective8累计分母，不能将其称为一个完整训练batch的梯度重放。", "",
        "冻结目标统计、去重训练父对象/源面身份、样本hash及每个g_FM/g_A/g_S/t保留在results.json或其calibration/target-statistics引用；不从生成输出拟合统计。",""]
    stats=r.get("target_statistics",{})
    lines += [f"统计去重自由面数：{_fmt(stats.get('unique_faces'))}；统计SHA256：{_fmt(stats.get('statistics_sha256'))}。",""]
    lines += _table(["描述组","维数","逐维mu","中心RMS s","归一化分母"],[[name,v.get("dimension"),v.get("mean"),v.get("centered_RMS"),v.get("denominator")] for name,v in stats.get("groups",{}).items()])+[""]
    lines += _training_section(r)+[""]
    lines += ["## 原始完整生成：分patch与父对象","","每格为“均值 / 输出中位 / 最差”。RMS、Gamma为% bbox对角线；coverage与质量比例为0–1。面积无改善方向。原片每父8噪声，额外每片每父仅2噪声；两额外片从不混成4个独立噪声。",""]
    for patch in (0,1,2):
        block=summary["patches"][str(patch)]
        lines += [f"### patch{patch}（{'原片，n=8/父' if patch==0 else '既有额外片，n=2/父'}）",""]
        lines += _table(["父","表面/接口指标",*MODELS],[[p,LABELS[k],*[_dist(block["models"][m]["parents"][p]["metrics"].get(k,{})) for m in MODELS]] for p in ("0","1") for k in SURFACE])
        lines += [""]+_table(["父","质量指标",*MODELS],[[p,LABELS[k],*[_dist(block["models"][m]["parents"][p]["metrics"].get(k,{})) for m in MODELS]] for p in ("0","1") for k in QUALITY])+[""]
    lines += ["## 同输入配对效果","","差=候选−参考；每格为“配对差均值 / 配对差中位 / 最差配对差；改善数/n”。先按同parent/patch/seed形成差，再求中位，绝不相减边际中位。未删除任何坏seed、未构造总分。",""]
    for candidate,reference in PAIRINGS:
        name=candidate+"_minus_"+reference
        lines += [f"### {candidate} − {reference}",""]
        lines += _table(["patch","父","自由RMS","free→目标","目标→free","Gamma RMS","Gamma p95","Gamma覆盖","CF比例","FF非法对"],[[j,p,*[_paired_cell(summary["patches"][str(j)]["paired"][name]["parents"][p],k) for k in (FREE,TO,FROM,GAMMA,P95,COVER,CF,FF)]] for j in (0,1,2) for p in ("0","1")])+[""]
    lines += ["## 逐seed与绑定","","紧凑逐输出CSV保留全部指标；JSON保留全部配对差及极端seed。以下列出三项增量的RMS逐seed差，正值为变差。",""]
    for patch in (0,1,2):
        block=summary["patches"][str(patch)]
        names=[a+"_minus_"+b for a,b in PAIRINGS[1:]]
        lines += _table(["patch","父","seed","Angular−G0","AngularScale−Angular","AngularScale−G0"],[[patch,p,seed,*[next(v[FREE] for v in block["paired"][name]["parents"][p]["per_seed_candidate_minus_reference"] if v["seed"]==seed) for name in names]] for p in ("0","1") for seed in block["seeds"]])+[""]
    binding=[]
    for j in (0,1,2):
        for m in MODELS:
            b=summary["patches"][str(j)]["models"][m]["binding"]
            binding.append([j,m,*[_dist(b["metrics"][k]) for k in ("Delta0","Delta1","B")],b["both_positive_count"]])
    lines += _table(["patch","模型","Delta0","Delta1","B","双向正次数"],binding)+["","绑定四格E00/E01/E10/E11及每seed均保留在summary.patches；固定100个free槽不重新匹配。正向绑定不是辅助loss的新成功判据，C精确保持来自clamping。",""]
    lines += ["## Teacher诊断：属性学习与完整生成分开","","G0没有训练辅助头，属性指标NOT_APPLIED，不能给它随机头后据此宣称其表示更差。辅助臂自己的头预测更准不等于主干更通用。错配标签评分只是同一预测的只读对应敏感性检查。",""]
    lines += _teacher_section(r)
    lines += ["","## 实际执行、审计与资产","","模型/训练、H9标定梯度、teacher、采样、CPU审计及绘图分别计账；预算不是实际调用次数。",""]
    lines += _table(["总账字段","实际值"],_json_scalar_table(r.get("counts",{})))+[""]
    lines += _table(["诊断调用","实际值"],_json_scalar_table(r.get("diagnostic_counts",{})))+[""]
    lines += _table(["阶段耗时秒","实际值"],_json_scalar_table(r.get("timing",{})))+[""]
    lines += _table(["新增门禁","记录"],[[key,value.get("status") if isinstance(value,dict) else value] for key,value in r.get("gates",{}).items()])+[""]
    infer=[]
    for model in MODELS:
        generated=[x for x in r["generated"] if x["model"]==model]
        attempts=[x for x in r.get("sampling_attempts",[]) if x["model"]==model and not x.get("replay")]
        infer.append([model,len(generated),sum(x.get("counts",{}).get("model_forward_attempts",0) for x in attempts),
            sum(x.get("counts",{}).get("model_forward_returns",0) for x in attempts),
            sum(x.get("auxiliary_modules_constructed",0) for x in generated),sum(x.get("auxiliary_calls",0) for x in generated),
            all(x.get("C_all51_bitwise") is True for x in generated)])
    lines += _table(["正式采样模型","完整输出","前向尝试","前向返回","P/S构造合计","P/S调用","C全部51态恒定"],infer)+[""]
    ledger=r.get("evaluation_ledger",{})
    lines += _table(["离线审计","实际值"],[[k,ledger.get(k)] for k in ("status","scored_outputs","seconds","complete_pairs_per_output","counts","all_reference_hashes_verified")])+[""]
    numeric=["nan_coordinate_count","inf_coordinate_count","free_near_degenerate_count","free_exact_zero_area_count","CF_unresolved_or_degenerate_pair_count","FF_unresolved_or_degenerate_pair_count","exact_C_free_shared_edge_count","C_inherited_nonmanifold_vertex_count","new_free_related_nonmanifold_vertex_count"]
    lines += _table(["诊断（输出计数求和）",*MODELS],[[k,*[sum(x["scalars"][k] for x in rows if x["model"]==m and x["scalars"].get(k) is not None) if any(x["model"]==m and x["scalars"].get(k) is not None for x in rows) else "NOT_RECORDED" for m in MODELS]] for k in numeric])+["",
        "完整独立CF/FF静态判交按旧阈值，CC按固定上下文缓存；精确FP32索引区分输入继承与free相关异常。计数求和不是源缺陷去重；当前绕序不等于不可重定向。未检出相交、自由描述学习或FF减少均不是水密/共享连接保证。精确谓词与连续碰撞安全NOT_RUN。",""]
    lines += _table(["臂/步","checkpoint","bytes","文件SHA256","生成器state SHA256"],[[m+"/"+step,cp.get("path"),cp.get("bytes"),cp.get("sha256",cp.get("checkpoint_sha256")),cp.get("generator_state_sha256",cp.get("state_sha256"))] for m in MODELS[1:] for step,cp in arms[m].get("checkpoints",{}).items()])+[""]
    lines += _table(["输入资产","路径","bytes","SHA256"],[[key,r["assets"][key].get("path"),r["assets"][key].get("bytes"),r["assets"][key].get("sha256")] for key in ("initial","inputs","evaluation_inputs","evaluation_noise","target_statistics") if key in r.get("assets",{})])+[""]
    lines += [f"协议SHA256：{_fmt(r.get('protocol_sha256'))}；源码版本/修改与冻结记录见protocol、execution_sources及code.diff，未重置或提交仓库。", ""]
    lines += ["严格推理加载：从已激活MeshFlow环境的WSL仓库根目录逐条执行。每条命令启动独立进程，仅加载一臂原T1+Geo，打印严格加载审计后退出；不构造P/S，也不启动采样。", "", "```bash"]
    for model,command in _loader_commands(r):
        lines += [f"# {model}",command]
    lines += ["```",""]
    lines += ["## 固定首噪声图","","预定seed9901，patch0/1/2各16格；四模型同父对象使用跨三图共同范围、同视角。仅原始FP32 full112/free100，不参考输出改选seed、片或相机。",""]
    for figure in figures:lines += [f"![seed9901 patch{figure['patch']} full/free]({Path(figure['path']).name})",""]
    lines += ["达到预算后停止，不自动延长、调lambda、换层、加入CKA/M10或进入下一阶段。若没有收益，仅限本次解析教师、固定第9层与500步预算未建立增量；不否定其他外部教师或表示对齐方案。",""]
    return chr(10).join(lines)


def _get(value,*keys):
    for key in keys:
        if not isinstance(value,dict):return None
        value=value.get(key)
    return value


def _mean(rows,*keys):
    values=[_get(row,*keys) for row in rows]
    valid=[float(v) for v in values if isinstance(v,(int,float)) and not isinstance(v,bool) and np.isfinite(v)]
    if not valid:return "NOT_APPLIED" if any(v=="NOT_APPLIED" for v in values) else None
    if len(valid)!=len(values):raise ValueError("Partially missing teacher values; do not silently change denominator")
    return float(np.mean(valid))


def _training_section(r):
    records=[(m,d) for m in MODELS[1:] for d in r["arms"][m].get("diagnostics",[])]
    lines=["## 预定训练日志与实际更新","","仅step1/10/50/100/250/500的不同训练batch日志；不据此宣布收敛，也不将辅助loss下降替代完整生成。G0辅助项NOT_APPLIED；Angular臂Scale为NOT_APPLIED。",""]
    lines += _table(["臂","步","FM","Angular","Scale","实际加权Angular","实际加权Scale","总loss","ramp","gen裁剪前L2","aux裁剪前L2"],[
        [m,d["step"],d.get("losses",{}).get("FM"),d.get("losses",{}).get("Angular") if m!="G0" else "NOT_APPLIED",
         d.get("losses",{}).get("Scale") if m=="G_AngularScale" else "NOT_APPLIED",
         d.get("losses",{}).get("weighted_Angular") if m!="G0" else "NOT_APPLIED",
         d.get("losses",{}).get("weighted_Scale") if m=="G_AngularScale" else "NOT_APPLIED",d.get("losses",{}).get("total"),d.get("ramp"),
         d.get("generator_grad_norm_before_clip"),d.get("auxiliary_grad_norm_before_clip") if m!="G0" else "NOT_APPLIED"] for m,d in records])+[""]
    lines += _table(["臂","步","组","梯度RMS","更新RMS","梯度非零数","更新非零数","均有限"],[
        [m,d["step"],name,v.get("RMS"),d.get("parameter_updates",{}).get(name,{}).get("RMS"),v.get("nonzero"),
         d.get("parameter_updates",{}).get(name,{}).get("nonzero"),v.get("finite") and d.get("parameter_updates",{}).get(name,{}).get("finite")]
        for m,d in records for name,v in d.get("gradients",{}).items() if name in ("input_geo","P","S")])+[""]
    lines += _table(["臂","步","预测","平均向量范数","均维方差","元素RMS","自由面数"],[
        [m,d["step"],name,v.get("norm_mean"),v.get("variance_mean"),v.get("element_RMS"),v.get("count")]
        for m,d in records for name,v in d.get("prediction_statistics",{}).items()])+[""]
    lines += ["role/主干输入/输出组的完整梯度与更新亦保留在原始日志。辅助loss对第10–12层没有直接梯度符合第9层读出设计；这些层仍由FM训练。"]
    return lines


def _teacher_section(r):
    """Read existing fixed-state diagnostics; no model or loss recomputation."""
    teacher=r.get("teacher",{})
    labels=("Geo_initial","common_aux_initial","G0","G_Angular","G_AngularScale")
    if not teacher:return ["Teacher汇总尚未登记：NOT_RECORDED。"]
    lines=["每个(parent,t)格是两份固定OT噪声的均值；同一对只做一次OT、复用五个t。五时间格是诊断网格，不是训练logit-normal分布期望。common_aux_initial是单独标记的共同初始P/S，绝不当作G0。",""]
    prepared={}
    for label in labels:
        group=teacher.get(label,{})
        rows=group.get("rows",[]) if isinstance(group,dict) else []
        if len(rows)!=20 or group.get("status")!="PASS":raise ValueError("Missing fixed20-state teacher records: "+label)
        prepared[label]=rows
    table=[]
    for parent in (0,1):
        for ti,t in enumerate((.1,.3,.5,.7,.9)):
            for label in labels:
                selected=[x for x in prepared[label] if x["parent_index"]==parent and x["time_index"]==ti]
                if len(selected)!=2:raise ValueError("Teacher parent/time pairing missing")
                table.append([parent,t,label,_mean(selected,"FM_velocity_MSE"),_mean(selected,"Angular_cosine"),_mean(selected,"Scale_MSE")])
    lines += _table(["父","t","模型/头身份","FM速度MSE","Angular cosine","Scale MSE"],table)+[""]
    scale=[]
    for parent in (0,1):
        for ti,t in enumerate((.1,.3,.5,.7,.9)):
            for label in ("common_aux_initial","G_AngularScale"):
                selected=[x for x in prepared[label] if x["parent_index"]==parent and x["time_index"]==ti]
                scale.append([parent,t,label,*[_mean(selected,"Scale_group_MSE",key) for key in ("centroid","log_edges","log_area","normal_outer")],
                    _mean(selected,"constant_baseline","Scale_MSE"),_mean(selected,"prediction_statistics","q_hat","mean_variance"),
                    _mean(selected,"target_statistics","mean_variance")])
    lines += ["Scale组内误差各按自身维数平均；常量基线用冻结训练描述均值，未按teacher目标重新拟合。方差用于检查近常量预测，不是几何质量评分。",""]
    lines += _table(["父","t","头身份","质心MSE","log边长MSE","log面积MSE","法向外积MSE","常量基线全MSE","q_hat均维方差","q均维方差"],scale)+[""]
    comparison=[]
    for parent in (0,1):
        for ti,t in enumerate((.1,.3,.5,.7,.9)):
            for label in ("G_Angular","G_AngularScale"):
                selected=[x for x in prepared[label] if x["parent_index"]==parent and x["time_index"]==ti]
                comparison.append([parent,t,label,_mean(selected,"Angular_cosine"),_mean(selected,"shuffled_target","Angular_cosine"),
                    _mean(selected,"Scale_MSE"),_mean(selected,"shuffled_target","Scale_MSE")])
    lines += ["同一预测对同样本内打乱目标的评分：不新运行网络、不训练错配对照。Angular只有P，Scale相关项NOT_APPLIED。",""]
    lines += _table(["父","t","臂","正确cosine","打乱cosine","正确Scale MSE","打乱Scale MSE"],comparison)
    return lines


def write_report(r,out):
    tick=time.perf_counter();out=Path(out).resolve();rows=_rows(r)
    for row in rows:
        if _record(row["metrics_path"])["sha256"]!=row["metrics_sha256"]:raise ValueError("Saved metric artifact changed")
    figures=_figures(r,out);text=render_report(r,figures)
    path=out/"GEOMETRY_REP_ALIGNMENT_REPORT.md"
    if path.exists():raise FileExistsError(path)
    path.write_text(text,encoding="utf-8")
    return dict(status="PASS",report=_record(path),figures=figures,source=_record(__file__),seconds=time.perf_counter()-tick,
        model_forwards=0,model_backwards=0,optimizer_updates=0,OT_calls=0,geometry_rescoring=0,
        figure_seed=SEEDS[0],visual_review="PENDING_CALLER_REVIEW")


def run_cpu_tests():
    """Five presentation/schema checks from synthetic saved scalars only."""
    assert _fmt(None)=="NOT_RECORDED" and _fmt(0.)=="0"
    d=dict(finite_count=2,mean=2.,median=2.,worst=dict(seed=1,value=3.))
    assert _dist(d)=="2 / 2 / 3"
    d["worst"]=None
    assert _dist(d)=="2 / 2 / N/A"
    rows=[]
    for model in MODELS:
        for parent in (0,1):
            for patch in (0,1,2):
                for seed in (SEEDS if patch==0 else SEEDS[:2]):
                    rows.append(dict(model=model,parent=parent,patch=patch,seed=seed,
                        patch_group="original" if patch==0 else "new",raw_path=f"/fixture/{model}_{parent}_{patch}_{seed}.npz",
                        scalars={k:1. for k in (FREE,TO,FROM,GAMMA,P95,COVER)}))
    fake=dict(status="PASS_EXECUTION",scores=rows,generated=rows,summary=dict(patches={str(j):{} for j in (0,1,2)}))
    assert len(_rows(fake))==96
    for bad in (dict(fake,scores=rows[:-1]),dict(fake,status="TRAINING")):
        try:_rows(bad)
        except ValueError:pass
        else:raise AssertionError("Partial/uncompleted report input accepted")
    return dict(status="PASS",checks=5,model_calls=0,geometry_calls=0,autograd_calls=0,
        real_assets_read=0,figures_rendered=0)
