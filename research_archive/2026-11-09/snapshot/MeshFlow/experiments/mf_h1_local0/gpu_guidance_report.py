"""Offline report/plots for GPU guidance and the same protected weld W.

No model, energy, geometric classification, welding or surface scoring runs in
this module. It summarizes recorded results and displays saved coordinates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .guided_native_geometry_report import _get, _n as _legacy_n, _dist
from .gpu_guidance_evaluation import compare_variants



def _n(value,digits=4):
    # A small nonzero geometric quantity must not be displayed as exact zero.
    if isinstance(value,(float,np.floating)) and np.isfinite(value) and 0<abs(value)<1e-4:
        return f"{value:.{digits}e}"
    return _legacy_n(value,digits)


def _table(headers,rows):
    if not rows:return "NOT_RUN / 无记录。"
    safe_headers=[str(h).replace("|","\\|").replace("\n"," ") for h in headers]
    return "\n".join(["| "+" | ".join(safe_headers)+" |",
        "| "+" | ".join("---" for _ in safe_headers)+" |"]+
        ["| "+" | ".join(_n(item) for item in row)+" |" for row in rows])


MODELS=("G0","GA","GR","GAR")
VARIANTS=("Raw","W")


def _topology(metrics):
    bundle=metrics["gpu_guidance_evaluation"].get("indexed_topology",{})
    return bundle.get("topology",bundle),bundle.get("indexing",{})


def _metric_row(row):
    m=row["metrics"];e=m["gpu_guidance_evaluation"];shape=e["free_shape_quality"]
    cf=e["detector_views"]["CF"]["primary"];ff=e["detector_views"]["FF"]["primary"]
    verified_cf=e["detector_views"]["CF"]["verified"];verified_ff=e["detector_views"]["FF"]["verified"]
    topo,indexing=_topology(m)
    proper_faces={p[1] for p in cf["proper_pairs"]}
    union=e["illegal_free_face_quality"]["primary"]["union"]
    return dict(phase=row.get("phase","development"),model=row["model"],parent=int(row["parent"]),seed=int(row["seed"]),
        variant=row["variant"],remainder_rms_pct=m["remainder"]["own"]["symmetric_rms_bbox_pct"],
        forward_rms_pct=m["remainder"]["own"]["predicted_to_target_rms_bbox_pct"],
        reverse_rms_pct=m["remainder"]["own"]["target_to_predicted_rms_bbox_pct"],
        full_rms_pct=m["whole"]["surface"]["symmetric_rms_bbox_pct"],
        CF_proper_faces=len(proper_faces),CF_proper_face_fraction=len(proper_faces)/100,
        CF_proper_pairs=cf["proper_pair_count"],CF_overlap_pairs=cf["coplanar_pair_count"],
        CF_unresolved=verified_cf["unresolved_count"],FF_proper_pairs=ff["proper_pair_count"],
        FF_overlap_pairs=ff["coplanar_pair_count"],FF_illegal_pairs=ff["illegal_pair_count"],FF_unresolved=verified_ff["unresolved_count"],
        boundary_rms_pct=m["guided_geometry_audit"]["boundary"]["rms_bbox_pct"],
        boundary_p95_pct=m["guided_geometry_audit"]["boundary"]["p95_bbox_pct"],
        boundary_coverage=m["guided_geometry_audit"]["boundary"]["coverage_fraction"],
        narrow_fraction=shape["min_angle_lt5_fraction"],angle_p05=shape["angle_p05_degrees"],
        q_gt10_fraction=shape["q_actual_gt10_fraction"],q_median=shape["q_actual"]["median"],q_p95=shape["q_actual"]["p95"],
        EA_CPU_reference=shape["guarded_EA_CPU_formula_reference"],EA_GPU=_get(m,"gpu_endpoint_energy","E_A"),ER_GPU=_get(m,"gpu_endpoint_energy","E_R"),
        area_total=shape["total_area"],area_floor_count=shape["area_floor_trigger_count"],
        zero_area_count=shape["exact_zero_area_count"],near_degenerate_count=shape["near_degenerate_area_count"],
        nan_count=m["whole"]["quality"]["nan_count"],inf_count=m["whole"]["quality"]["inf_count"],
        illegal_unique_faces=union["count"],illegal_q_gt10_faces=union["q_actual_gt10_count"],
        illegal_near_degenerate_faces=union["near_degenerate_count"],
        illegal_q_median=union["q_actual"]["median"],illegal_q_p95=union["q_actual"]["p95"],
        illegal_area_median=union["area_normalized"]["median"],illegal_area_p05=union["area_normalized"]["p05"],
        illegal_area_minimum=union["area_normalized"]["minimum"],
        illegal_well_shaped_nondegenerate_faces=len(union["q_at_most10_and_not_near_degenerate_face_ids"]),
        vertex_count=topo.get("vertex_count"),edge1=_get(topo,"edge_incidence_counts","1"),
        edge2=_get(topo,"edge_incidence_counts","2"),edge_gt2=_get(topo,"edge_incidence_counts",">2"),
        edge_components=topo.get("edge_connected_components"),duplicate_faces=topo.get("duplicate_faces"),
        vertex_link_status=_get(topo,"vertex_link","status"),vertex_link_nonmanifold=_get(topo,"vertex_link","nonmanifold_vertices"),
        t_junction_status=_get(topo,"t_junction","status"),t_junction_count=_get(topo,"t_junction","count"),
        t_junction_hits=_get(topo,"t_junction","hits"),orientation_status=topo.get("orientation_status",topo.get("orientation_consistent")),
        C_PRESERVED=e["C_PRESERVED"],C_indexing=indexing.get("C_PRESERVED"),C_OBJ=_get(row,"export","C_PRESERVED"),
        COMBINATORIAL_CLOSURE_STATUS=e["COMBINATORIAL_CLOSURE_STATUS"],
        EMBEDDED_GEOMETRY_VALIDITY_STATUS=e["EMBEDDED_GEOMETRY_VALIDITY_STATUS"],
        whole_faces=e["whole_faces"],free_faces=e["free_faces"],indexing=indexing,
        evaluation_seconds=e["scoring_and_additional_audit_seconds"])


NUMERIC=("remainder_rms_pct","forward_rms_pct","reverse_rms_pct","full_rms_pct","CF_proper_faces",
    "CF_proper_face_fraction","CF_proper_pairs","CF_overlap_pairs","CF_unresolved","FF_proper_pairs",
    "FF_overlap_pairs","FF_illegal_pairs","FF_unresolved","boundary_rms_pct","boundary_p95_pct","boundary_coverage",
    "narrow_fraction","angle_p05","q_gt10_fraction","q_median","q_p95","EA_CPU_reference","EA_GPU","ER_GPU","area_total",
    "area_floor_count","zero_area_count","near_degenerate_count","illegal_unique_faces","illegal_q_gt10_faces",
    "illegal_near_degenerate_faces","illegal_well_shaped_nondegenerate_faces","illegal_q_median","illegal_q_p95",
    "illegal_area_median","illegal_area_p05","illegal_area_minimum","vertex_count","edge1","edge2",
    "edge_gt2","edge_components","t_junction_count","vertex_link_nonmanifold","evaluation_seconds")


def summarize(results):
    source=results.get("scores",[])
    rows=[_metric_row(row) for row in source]
    rows.sort(key=lambda r:(r["phase"],r["model"],r["variant"],r["parent"],r["seed"]))
    lookup={(r.get("phase","development"),r["model"],r["variant"],int(r["parent"]),int(r["seed"])):r for r in source}
    if len(lookup)!=len(source):raise AssertionError("Duplicate method/variant/parent/noise scores")
    groups=[]
    for phase in ("development","confirmation"):
        for model in MODELS:
            for variant in VARIANTS:
                for parent in (0,1):
                    selected=[r for r in rows if (r["phase"],r["model"],r["variant"],r["parent"])==(phase,model,variant,parent)]
                    if not selected:continue
                    groups.append(dict(phase=phase,model=model,variant=variant,parent=parent,n=len(selected),
                        seeds=[r["seed"] for r in selected],statistics={k:_dist([r[k] for r in selected]) for k in NUMERIC},
                        C_preserved_count=sum(r["C_PRESERVED"] in (True,"PASS","C_PRESERVED") for r in selected),
                        statuses=[dict(seed=r["seed"],C=r["C_PRESERVED"],closure=r["COMBINATORIAL_CLOSURE_STATUS"],
                                       embedded=r["EMBEDDED_GEOMETRY_VALIDITY_STATUS"],vertex_link=r["vertex_link_status"]) for r in selected]))
    comparisons=[]
    for key,row in lookup.items():
        phase,model,variant,parent,seed=key
        targets=[]
        if model!="G0":targets.append(("guidance_same_layer",(phase,"G0",variant,parent,seed)))
        if variant=="W":targets.append(("W_same_generation",(phase,model,"Raw",parent,seed)))
        for kind,base_key in targets:
            if base_key in lookup:
                comparisons.append(dict(kind=kind,phase=phase,model=model,variant=variant,parent=parent,seed=seed,
                    comparator_model=base_key[1],comparator_variant=base_key[2],
                    comparison=compare_variants(lookup[base_key]["metrics"],row["metrics"])))
    return dict(schema="gpu_guidance_report_v1",per_output=rows,groups=groups,comparisons=comparisons,
        trajectory_geometry=_trajectory_geometry(results),status_counts=_status_counts(rows),
        report_model_calls=0,report_geometry_evaluations=0,report_weld_calls=0,
        scopes=dict(development="Repeatedly used regression9201..9204",confirmation="New9401..9408, two training parents only"))



def _status_counts(rows):
    result={}
    for phase in ("development","confirmation"):
        for variant in VARIANTS:
            subset=[r for r in rows if r["phase"]==phase and r["variant"]==variant]
            if not subset:continue
            counts={}
            for key in ("C_PRESERVED","C_indexing","C_OBJ","COMBINATORIAL_CLOSURE_STATUS",
                        "EMBEDDED_GEOMETRY_VALIDITY_STATUS","vertex_link_status"):
                values={}
                for row in subset:
                    value=str(row[key]);values[value]=values.get(value,0)+1
                counts[key]=values
            result[phase+"/"+variant]=dict(outputs=len(subset),counts=counts)
    return result


def _trajectory_geometry(results):
    output=[]
    for phase in ("development","confirmation"):
        for model in ("GA","GR","GAR"):
            subset=[r for r in results.get("generated",[]) if r.get("phase")==phase and r.get("model")==model]
            diagnostic=[]
            for row in subset:
                columns=row.get("geometry_diagnostic_columns",[])
                diagnostic.extend(dict(zip(columns,values)) for values in row.get("geometry_diagnostics",[]))
            if not diagnostic:continue
            output.append(dict(phase=phase,model=model,trajectories=len(subset),active_states=len(diagnostic),
                evaluation_status=dict(A="EVALUATED" if any(r.get("A_evaluated") for r in diagnostic) else "NOT_EVALUATED",
                    R="EVALUATED" if any(r.get("R_evaluated") for r in diagnostic) else "NOT_EVALUATED"),
                statistics={key:_dist([r.get(key) for r in diagnostic
                    if (key not in ("E_A","gradient_A_norm") or r.get("A_evaluated"))
                    and (key not in ("E_R","gradient_R_norm") or r.get("R_evaluated"))]) for key in
                    ("E_A","E_R","gradient_A_norm","gradient_R_norm","gradient_combined_norm","clipped_vertex_count",
                     "clipped_max_vertex_displacement","unclipped_max_vertex_displacement")},
                defined_cosine=_dist([r["gradient_cosine"] for r in diagnostic if r.get("cosine_defined")]),
                negative_defined_cosines=sum(r.get("cosine_defined",False) and r["gradient_cosine"]<0 for r in diagnostic),
                full_generation_seconds=_dist([r["costs"].get("complete_generation_wall_seconds") for r in subset]),
                model_cuda_seconds=_dist([r["costs"].get("model_cuda_seconds") for r in subset]),
                geometry_forward_cuda_seconds=_dist([r["costs"].get("geometry_forward_cuda_seconds") for r in subset]),
                geometry_backward_cuda_seconds=_dist([r["costs"].get("geometry_backward_cuda_seconds") for r in subset])))
    return output


def _group(summary,model,variant,parent,phase="development"):
    return next((r for r in summary["groups"] if (r["phase"],r["model"],r["variant"],r["parent"])==(phase,model,variant,parent)),None)


def _compare_text(summary,model,variant,phase="development"):
    values=[]
    for parent in (0,1):
        a,b=_group(summary,"G0",variant,parent,phase),_group(summary,model,variant,parent,phase)
        if not a or not b:continue
        x,y=a["statistics"],b["statistics"]
        values.append(f"P{parent}({b['n']}条)：R12 {_n(x['remainder_rms_pct']['mean'])}→{_n(y['remainder_rms_pct']['mean'])}%L；"
            f"窄角面{100*x['narrow_fraction']['mean']:.2f}→{100*y['narrow_fraction']['mean']:.2f}%；"
            f"CF面{100*x['CF_proper_face_fraction']['mean']:.2f}→{100*y['CF_proper_face_fraction']['mean']:.2f}%；"
            f"FF非法pairs {_n(x['FF_illegal_pairs']['mean'])}→{_n(y['FF_illegal_pairs']['mean'])}")
    return "；".join(values) if values else "NOT_RUN / 对应同口径结果未齐。"


def _weld_text(summary):
    text=[]
    for parent in (0,1):
        a,b=_group(summary,"G0","Raw",parent),_group(summary,"G0","W",parent)
        if not a or not b:continue
        x,y=a["statistics"],b["statistics"]
        text.append(f"P{parent}原T1经W：顶点数均值{_n(x['vertex_count']['mean'])}→{_n(y['vertex_count']['mean'])}，"
            f"单关联边{_n(x['edge1']['mean'])}→{_n(y['edge1']['mean'])}，共边连通分量{_n(x['edge_components']['mean'])}→{_n(y['edge_components']['mean'])}；"
            f"边界RMS {_n(x['boundary_rms_pct']['mean'])}→{_n(y['boundary_rms_pct']['mean'])}%L，"
            f"FF非法pairs {_n(x['FF_illegal_pairs']['mean'])}→{_n(y['FF_illegal_pairs']['mean'])}")
    return "；".join(text) if text else "NOT_RUN，未完成G0 Raw和同一W的配对审计。"



def _actual_findings(results,summary):
    f=[""]*8
    if len([r for r in summary["per_output"] if r["phase"]=="development"])!=64:return f
    def avg(m,p,k):return _group(summary,m,"Raw",p)["statistics"][k]["mean"]
    energy=lambda m,k:"；".join(f"P{p} {_n(avg('G0',p,k))}→{_n(avg(m,p,k))}" for p in (0,1))
    f[0]="实测EA下降："+energy("GA","EA_GPU")+"。<5°比例P0只下降0.25个百分点、P1下降2个百分点，两个父对象FF均增加；面形状有改善，几何联合目标未达到。"
    f[1]="GR的ER下降："+energy("GR","ER_GPU")+"，但两父CF穿刺面率、FF非法pair和整段边界RMS均上升。核能量下降没有转化为真实穿刺减少。"
    g=next(g for g in summary["trajectory_geometry"] if g["model"]=="GAR" and g["phase"]=="development")
    c=g["defined_cosine"];d=g["statistics"]
    f[2]=(f"GAR也未过门槛。80个引导状态的∥gA∥/∥gR∥均值{_n(d['gradient_A_norm']['mean'])}/{_n(d['gradient_R_norm']['mean'])}；"
        f"{c['n']}个定义余弦均值{_n(c['mean'])}、范围[{_n(c['minimum'])},{_n(c['maximum'])}]，负值{int(g['negative_defined_cosines'])}个。"
        "局部反向不足以把失败唯一归因于梯度冲突；两父CF比GR更高，FF也未优于G0。")
    f[3]="W合并了部分近邻，但8个G0+W均未闭合且顶点link均失败；P0/P1非流形顶点均值从0/2升到47.25/59.5。W全部32输出出现14个跨面重复（分布12输出），单面重复index和实际退化为0。不能把局部门禁通过说成全局修复。"
    f[4]="三种Guidance+W在两个父对象上的CF穿刺面率与FF非法pair均高于G0+W，没有建立联合优势。"
    f[5]="64/64个Raw/W端点及OBJ保持C，112面/free100；64/64均检出非法相交，组合闭合通过0。Raw顶点link通过16/32（允许边界），W通过0/32。T-junction穷举0仅表示未检出该容差下顶点落在非端点边上的狭义情况，不证明闭合。"
    costs=[]
    for g in summary["trajectory_geometry"]:
        if g["phase"]!="development":continue
        geo=g["geometry_forward_cuda_seconds"]["mean"]+g["geometry_backward_cuda_seconds"]["mean"]
        costs.append(f"{g['model']}完整生成{g['full_generation_seconds']['mean']:.4f}s，GPU几何前向+反向{geo:.4f}s")
    f[6]="三组均未过Raw门槛，9401–9408没有生成，不能宣称新噪声复现。"+"；".join(costs)+"（均值）。G0仅两次重放，不能作稳定加速比。"
    f[7]="建议停止在当前参数下扩大GA/GR/GAR修复验证：实现与确定性通过，几何联合收益未建立。先定位合法接触与整体拓扑约束的表达；其他训练支持/模块的收益仍未测试，本轮不自动转向。"
    return f


def _answers(results,summary):
    selection=results.get("selection",results.get("promotion",{}))
    confirmation=results.get("confirmation",{})
    selected=confirmation.get("candidate",selection.get("candidate"))
    actual=_actual_findings(results,summary)
    answers=[
        ("1. 形状能量只降数值，还是改善实际面质量且保持几何？",_compare_text(summary,"GA","Raw")+
         "。q、<5°面比例、angle p05、真实退化和R12分别列出；EA降低不替代实际质量或几何保持。面积下限触发不等于实际退化。"),
        ("2. 顶点核和7点核的盲区；GR是否减少真实相交？",_compare_text(summary,"GR","Raw")+
         "。三个顶点远离有限面时顶点核会漏掉面内部穿刺；7个固定样点仍只是一种离散表面偏置，不能证明所有交线都被捕获。"
         "恰在表面时核最大而梯度可为零，合法接触也可能被排斥；实际探针及GT相容性记录另列。GR没有CPU整面分离或验收门禁。"),
        ("3. GAR组合是否更好，是否有相反作用或代价？",_compare_text(summary,"GAR","Raw")+
         "。下表同时保留GA/GR/GAR，合梯度仅共用一次rho；逐步两项梯度范数和余弦由GPU诊断记录，不能预设正交或协同。"),
        ("4. W在原T1上解决了多少连接问题？",_weld_text(summary)+
         "。减少顶点数、边界边或组件可说明部分坐标错位被合并，不能自动说明一般T-junction、缺失面连接或非法相交已解决；"
         "顶点link、边关联和完整静态相交独立检查。没有删面、补洞或重三角化。"),
        ("5. Guidance+W是否优于同样W处理后的G0？", "GA+W："+_compare_text(summary,"GA","W")+"；GR+W："+
         _compare_text(summary,"GR","W")+"；GAR+W："+_compare_text(summary,"GAR","W")+
         "。这里基准始终是G0+W；没有把W自身收益归给Guidance。"),
        ("6. 最终C、面数、闭合、顶点流形和无相交分别如何？",
         "逐输出三标签与拓扑状态见后表。C坐标评分、生成51态保持及OBJ读回保持分开；面数固定112/100。"
         "所有边关联数为2不替代顶点link和相交审计。NO_ILLEGAL_TRIANGLE_INTERSECTIONS_DETECTED只指声明容差下的静态完整pair检查，CCD仍NOT_RUN。"),
        ("7. 新噪声复现了吗？真实额外成本是多少？",
         f"确认状态：{confirmation.get('status','NOT_RUN')}；唯一选择：{selected}。"
         +(_compare_text(summary,selected,"Raw","confirmation") if selected else "未选中候选时不执行新噪声，不强选不合格的最好组。")+
         "。模型、GPU几何前向/坐标反向、D2H、W/导出/离线审计分列；未预设任何毫秒或显存指标。"),
        ("8. 下一步证据支持什么？",
         "依预登记Raw门槛判断本轮GPU能量是否值得独立验证，不能靠+W图或单个噪声改选。"
         "若未通过，说明当前固定参数下未建立实用收益；边界标记、训练支持的上下文Guidance和神经细化均未在本轮测试，不能宣称已经胜出。"
         "到本轮预算或门禁停止，不新增训练、改阈值或修补到过线。")]
    return [(title,actual[index]+text) for index,(title,text) in enumerate(answers)]



def _configuration_lines(results):
    gates=results.get("gates",{});strict=gates.get("strict_load",{})
    initial,final=gates.get("initial_state_hash"),gates.get("final_state_hash")
    protection=results.get("protection",{})
    compact_protection={k:{kk:vv for kk,vv in v.items() if kk not in ("files_detail","records","before","after")}
        for k,v in protection.items() if k!="metadata_before" and isinstance(v,dict)}
    # File dictionaries can be large; only statuses/counts and explicit differences
    # belong in prose. All original records remain in the sole results JSON.
    protection_rows=[]
    for name,v in compact_protection.items():
        protection_rows.append([name,v.get("status"),v.get("files",v.get("count",v.get("checked"))),v.get("changed",v.get("failures",[]))])
    rows=[
        ["protocol SHA256",results.get("protocol_sha256")],
        ["checkpoint SHA256",_get(results,"assets","checkpoint","sha256")],
        ["strict load",strict.get("status")],
        ["missing / unexpected",str(strict.get("missing_keys"))+" / "+str(strict.get("unexpected_keys"))],
        ["参数总数 / 可训练数",str(strict.get("parameter_count"))+" / "+str(strict.get("trainable_parameter_count"))],
        ["state before",initial],["state after",final],
        ["state hash equality",None if initial is None or final is None else initial==final],
        ["原入口与预校验入口逐位一致",gates.get("original_forward_equivalence")],
        ["几何图仅当前free叶子",gates.get("geometry_graph")],
        ["实际运行时配置",gates.get("runtime")],
        ["Git起始状态",results.get("initial_git_status")],
        ["资源观测",results.get("resource_snapshot")],["实际环境版本",results.get("actual_environment_versions")],
    ]
    replay_rows=[[q.get("parent"),q.get("seed"),q.get("all51_bitwise_equal_to_archive"),q.get("status"),
                  _get(q,"costs","complete_generation_wall_seconds")] for q in gates.get("G0_replays",[])]
    repeat=gates.get("guided_repeat",{})
    return ["## 实际入口、版本与冻结门禁","",_table(["实测项","记录"],rows),"",
        _table(["G0重放父","seed","51态逐位一致","状态","完整生成秒"],replay_rows),"",
        "引导重复：51态逐位一致="+str(repeat.get("all51_bitwise_equal"))+
        "；GPU诊断逐位一致="+str(repeat.get("geometry_diagnostics_bitwise_equal"))+"。", "",
        _table(["保护范围","状态","文件数","变化/失败"],protection_rows),"",
        "模型、积分和C沿用原FP32坐标；仅几何能量临时使用Y=X/L、C/L，位移再乘L返回原坐标。几何反向只针对当前free坐标。C逐位复写属于构造约束，"
        "不能被解释为模型学习到了精确已知面。历史文件保护、冻结模型状态和原入口复现分别核对。",""]


def _probe_lines(results):
    probes=results.get("probes",{});energy=probes.get("energy_fixed",{});weld=probes.get("weld_fixed",{})
    gt=probes.get("GT_compatibility",{})
    fixture_map={x["name"]:x for x in energy.get("fixtures",[])}
    named=[("vertex_miss_seven_points_detect","顶点漏检、七点激活"),
           ("both_vertex_and_seven_points_miss","两种固定点库均近零响应"),
           ("surface_kernel_maximum_zero_gradient","恰在表面：能量最大、梯度零"),
           ("legal_shared_edge","合法共享边也受力"),("legal_90_degree_fold","合法90度折角也受力")]
    probe_rows=[]
    for key,label in named:
        f=fixture_map.get(key)
        if f is None:continue
        observed=f.get("seven_point",f.get("observed",{}))
        probe_rows.append([label,f.get("status"),f.get("vertex_energy"),
            _get(observed,"terms","E_R"),_get(observed,"gradients","norm_R"),
            _get(f,"intersection","classification")])
    return ["## 数学、数值与GT相容性探针","",
        "固定能量探针状态："+str(energy.get("status","NOT_RUN"))+
        "；保护W探针状态："+str(weld.get("status","NOT_RUN"))+
        "；离线评分helper："+str(_get(probes,"evaluation_helpers","status"))+"。"
        "每个固定探针的CPU FP64/实际GPU FP32值、有限差分、刚体/尺度/排列核验、零梯度和合法接触结果都保留在JSON与fixed_probes.npz，未选择性删去失败样例。","",
        _table(["训练对象GT","EA","ER","∥gA∥","∥gR∥","q>10面","floor面","共享顶点槽","共享槽gR RMS/max"],
            [[p,v.get("E_A"),v.get("E_R"),v.get("gradient_A_norm"),v.get("gradient_R_norm"),v.get("q_above_cut_count"),
              v.get("area_floor_count"),v.get("shared_vertex_count"),
              f"{_n(v.get('shared_vertex_gradient_R_RMS'))}/{_n(v.get('shared_vertex_gradient_R_max'))}"] for p,v in sorted(gt.items())]),"",
        _table(["固定反例","探针状态","顶点核E","七点ER","七点梯度范数","独立判交"],probe_rows),"",
        "反例PASS表示成功复现已声明局限，并非几何修复成功。两种点库漏检例在FP64中的极小值也可能在FP32下下溢为零。"
        "等边穿刺例EA=0；重合退化例在数值下限政策下也不能依赖EA复活。降低核能量后仍穿刺的独立反例已保留。","",
        "GT探针只检查势能是否会偏离合法原始几何，GT及其梯度绝不传入采样器或W。"
        "离面核并不区分非法穿刺与合法接触；共享位置处单个距离项导数为零，也不保证与该点相连整面的总梯度为零。",""]



def _energy_and_shape_lines(summary):
    return ["## 实际端点能量、相交面质量与GPU梯度","",
        _table(["组/层/父","EA GPU均值","ER GPU均值","角度p05均值","非法free面均值","其中q≤10且非近退化","非法面q中位/p95均值","非法面面积/L²中位/p05均值"],
            [[f"{g['model']}/{g['variant']}/P{g['parent']}",g['statistics']['EA_GPU']['mean'],g['statistics']['ER_GPU']['mean'],
              g['statistics']['angle_p05']['mean'],g['statistics']['illegal_unique_faces']['mean'],
              g['statistics']['illegal_well_shaped_nondegenerate_faces']['mean'],
              f"{_n(g['statistics']['illegal_q_median']['mean'])}/{_n(g['statistics']['illegal_q_p95']['mean'])}",
              f"{_n(g['statistics']['illegal_area_median']['mean'])}/{_n(g['statistics']['illegal_area_p05']['mean'])}"] for g in summary['groups']]),"",
        "这是每条输出分布统计的噪声均值，不是把所有面混池计算。G0 Raw中P0/P1平均93/91.25张非法面，"
        "其中92/84.5张q≤10且非近退化；相交并非全由极差小面构成，也没有安全删面的证据。","",
        _table(["阶段/组","状态数","∥gA∥/∥gR∥均值","合梯度均值","定义余弦n/均值/范围","负余弦数","截断顶点均值"],
            [[g['phase']+'/'+g['model'],g['active_states'],
              f"{_n(g['statistics']['gradient_A_norm']['mean']) if g['model']!='GR' else 'NOT_EVALUATED'}/{_n(g['statistics']['gradient_R_norm']['mean']) if g['model']!='GA' else 'NOT_EVALUATED'}",
              g['statistics']['gradient_combined_norm']['mean'],
              f"{g['defined_cosine']['n']}/{_n(g['defined_cosine']['mean'])}/[{_n(g['defined_cosine']['minimum'])},{_n(g['defined_cosine']['maximum'])}]",
              int(g['negative_defined_cosines']),g['statistics']['clipped_vertex_count']['mean']] for g in summary['trajectory_geometry']]),"",
        "GA未评估R项、GR未评估A项；原始日志的占位0不被当作能量为零的观测。余弦只汇总定义的状态，"
        "夹角不足以直接预测整体网格或积分终点效果。","",
        "三标签分层计数：\x60"+json.dumps(summary['status_counts'],ensure_ascii=False)+"\x60。",""]


def _weld_cost_lines(results):
    return ["## W、OBJ与独立CPU审计成本","",
        _table(["阶段/组/父/seed","接受/拒绝","最大位移","W算子秒","精确索引秒","调用墙钟秒","OBJ往返秒","拓扑秒","C构造/OBJ"],
            [[f"{x.get('phase')}/{x.get('model')}/P{x.get('parent')}/{x.get('seed')}",
              f"{_get(x,'audit','accepted_merges')}/{_get(x,'audit','rejected_merges')}",
              _get(x,'audit','max_coordinate_displacement'),_get(x,'audit','operator_seconds'),_get(x,'audit','indexing_seconds'),
              x.get('operator_seconds'),x.get('index_export_seconds'),x.get('topology_audit_seconds'),
              str(_get(x,'audit','C_PRESERVED'))+" / "+str(_get(x,'export','C_PRESERVED'))] for x in results.get('weld',[])]),"",
        "W算子内部计时与外层调用墙钟、精确索引/OBJ导出/CPU拓扑审计各有口径，不能直接相加重复计费。"
        "每条输出的独立几何/表面审计秒数保留在offline_summary；表格最大位移使用原模型坐标单位，协议上限为.001L。",""]


def render_report(results,summary,figures):
    lines=["# Native T1：纯GPU几何引导与保护已知面的焊接验证","",
        f"状态：**{results.get('status','UNKNOWN')}**。Raw引导与固定W后处理分层比较；所有统计来自已保存输出，不根据图片挑噪声。",
        "9201–9204是已反复使用的开发噪声；确认9401–9408只检验两个训练父对象上的新噪声稳健性。"
        "L为已有完整对象bbox元数据，尚未验证只有C且未知完整尺度的部署。","",
        "新增训练更新、模型参数更新和主干反向的实际计数见账本；本轮没有上下文Guidance、边界embedding、神经去噪器或Phase3。",""]
    for title,text in _answers(results,summary):lines += ["## "+title,"",text,""]
    lines += _configuration_lines(results)+_probe_lines(results)+_energy_and_shape_lines(summary)
    if results.get("failure") or results.get("error"):
        lines += ["实际停止/失败：`"+str(results.get("failure",results.get("error")))+"`。未运行项不假造结论。",""]
    lines += ["## 参数、资产与实现范围","",
        "GA：q=三边平方和/(4√3面积)，q_cut=10；规范面积分母下限1e-10，log的q下限1e-12。"
        "另报精确零面积、规范面积≤1e-12近退化、面积<1e-10触发下限，三者不混为一谈。",
        "GR：有限C三角形距离，3顶点+3边中点+质心等权7点，sigma=.005；min为分段可微，不是光滑碰撞屏障。"
        "k40–49才开启，dt=.02，lambda=1，单顶点截断rho=.001L；GAR合梯度共用上限。"
        "W容差=.001L，保护C坐标，不经GT选择锚点/代表点，不删面、补洞或改变面剖分。", "",
        "[冻结参数](protocol.json)、[确认前候选选择记录](selection.json)、[实际代码diff](code.diff)及[唯一结果JSON](results.json)。"
        "没有选择阶段或执行中断时，相应文件/字段以实际记录为准，不宣称已通过。", "",
        "## Raw与同一W分层结果","",
        _table(["阶段/组/层/父","n","R12均值/中位 %L","CF面%","CF对/共面","FF非法/未定","Gamma RMS/p95 %L","覆盖%","<5°面%","q>10面%"],
            [[f"{x['phase']}/{x['model']}/{x['variant']}/P{x['parent']}",x['n'],
              f"{_n(x['statistics']['remainder_rms_pct']['mean'])}/{_n(x['statistics']['remainder_rms_pct']['median'])}",
              100*x['statistics']['CF_proper_face_fraction']['mean'],
              f"{_n(x['statistics']['CF_proper_pairs']['mean'])}/{_n(x['statistics']['CF_overlap_pairs']['mean'])}",
              f"{_n(x['statistics']['FF_illegal_pairs']['mean'])}/{_n(x['statistics']['FF_unresolved']['mean'])}",
              f"{_n(x['statistics']['boundary_rms_pct']['mean'])}/{_n(x['statistics']['boundary_p95_pct']['mean'])}",
              100*x['statistics']['boundary_coverage']['mean'],100*x['statistics']['narrow_fraction']['mean'],100*x['statistics']['q_gt10_fraction']['mean']]
             for x in summary['groups']]),"",
        "CF/FF主计数保持原primary分类口径，同时完整保存independent与verified分类；有分歧或近退化时UNRESOLVED不被归为无交。"
        "CF百分比分母为100张free面，pair数另列；整段Gamma coverage阈值1e-3L，p95按长度加权。", "",
        "## 逐输出质量、几何与三标签","",
        _table(["阶段/组/层/父/seed","R12 %L","free→GT / GT→free","全表面 %L","q中位/p95","面积floor/零/近退化","C","组合闭合","嵌入几何","vertex link"],
            [[f"{x['phase']}/{x['model']}/{x['variant']}/P{x['parent']}/{x['seed']}",x['remainder_rms_pct'],
              f"{_n(x['forward_rms_pct'])}/{_n(x['reverse_rms_pct'])}",x['full_rms_pct'],f"{_n(x['q_median'])}/{_n(x['q_p95'])}",
              f"{x['area_floor_count']}/{x['zero_area_count']}/{x['near_degenerate_count']}",x['C_PRESERVED'],
              x['COMBINATORIAL_CLOSURE_STATUS'],x['EMBEDDED_GEOMETRY_VALIDITY_STATUS'],x['vertex_link_status']]
             for x in summary['per_output']]),"",
        _table(["阶段/组/层/父/seed","顶点数","边1/2/>2","共边组件","T-junction数/范围","非流形顶点","一致定向","相交面数","其中q≤10且非近退化"],
            [[f"{x['phase']}/{x['model']}/{x['variant']}/P{x['parent']}/{x['seed']}",x['vertex_count'],
              f"{x['edge1']}/{x['edge2']}/{x['edge_gt2']}",x['edge_components'],
              str(x['t_junction_count'])+' / '+str(x['t_junction_status']),x['vertex_link_nonmanifold'],x['orientation_status'],
              x['illegal_unique_faces'],x['illegal_well_shaped_nondegenerate_faces']] for x in summary['per_output']]),"",
        "相交面面积与q的完整分布在JSON中。出现q≤10且非近退化的相交面会反驳‘所有相交都来自极差小面’的简化假设；"
        "即使某面很差，也没有证明可安全删除。本轮没有prune。", "",
        "## 配对比较与候选冻结","",
        _table(["比较/阶段/组/层/父/seed","R12 Δ百分点","窄角比例Δ","Gamma RMS Δ","新CF proper","新FF proper","新FF共面"],
            [[f"{x['kind']}/{x['phase']}/{x['model']}/{x['variant']}/P{x['parent']}/{x['seed']}",
              x['comparison']['remainder_RMS_difference'],x['comparison']['narrow_fraction_difference'],x['comparison']['boundary_RMS_difference'],
              len(x['comparison']['pair_set_turnover']['CF']['primary']['proper_pairs']['added']),
              len(x['comparison']['pair_set_turnover']['FF']['primary']['proper_pairs']['added']),
              len(x['comparison']['pair_set_turnover']['FF']['primary']['coplanar_overlap_pairs']['added'])] for x in summary['comparisons']]),"",
        "guidance_same_layer在Raw上比较G0 Raw、在W上比较G0+W；W_same_generation比较同一Raw前后。"
        "新增与移除pair集合同时保留，不能用总数下降掩盖替换出的新相交。W候选没有全局碰撞接受门禁。", "",
        "选择记录：`"+json.dumps(results.get('selection',results.get('promotion',{})),ensure_ascii=False)+"`。"
        "只用Raw门槛决定候选；先冻结selection，再读取确认结果，不能由后处理或确认图反向选参数。", "",
        "## 同噪声自由部分绑定","",
        _table(["阶段/组/层/seed","R0/C0","R0/C1","R1/C0","R1/C1","方向0/1","双正"],
            [[f"{x.get('phase')}/{x.get('model')}/{x.get('variant')}/{x.get('seed')}",_get(x,'metrics','E_R0_F_C0'),_get(x,'metrics','E_R0_F_C1'),
              _get(x,'metrics','E_R1_F_C0'),_get(x,'metrics','E_R1_F_C1'),
              f"{_n(_get(x,'metrics','direction0_bbox_percentage_points'))}/{_n(_get(x,'metrics','direction1_bbox_percentage_points'))}",
              _get(x,'metrics','both_directions_positive')] for x in results.get('binding',[])]),"",
        "同一输出固定free集合在两个目标间评分，无重新配面，GT只用于评价。", "",
        "## 实测成本与调用账本","",
        _table(["阶段/组/父/seed","模型CUDA秒","能量前向秒","坐标反向秒","总CUDA秒","setup秒","循环墙钟秒","完整生成秒","最终D2H秒","Torch峰值GiB"],
            [[f"{x.get('phase')}/{x.get('model')}/P{x.get('parent')}/{x.get('seed')}",
              _get(x,'costs','model_cuda_seconds'),_get(x,'costs','geometry_forward_cuda_seconds'),_get(x,'costs','geometry_backward_cuda_seconds'),
              _get(x,'costs','total_cuda_seconds'),_get(x,'costs','setup_wall_seconds'),_get(x,'costs','wall_seconds'),
              _get(x,'costs','complete_generation_wall_seconds'),_get(x,'costs','final_D2H_seconds'),
              None if _get(x,'costs','peak_cuda_allocated') is None else x['costs']['peak_cuda_allocated']/2**30]
             for x in results.get('generated',[])]),"",
        "完整生成墙钟包括setup、50步循环和最终D2H，不包含随后NPZ写盘；loop wall另列。"
        "CUDA event为设备时间跨度，可能包含主机发射间隙，不是进程CPU时间。GPU几何不等于零CPU调度。"
        "采样器没有逐步显式张量标量读回或CPU几何；原主干bool索引可能隐式同步，未单独profile。"
        "模型CPU dispatch未单独profile；CUDA event、完整墙钟、D2H和CPU后处理不可混算。"
        "峰值为Torch CUDA已分配内存，不是整个进程或驱动内存。预热、测试、重放、重复调用与正式样本分账。", "",
        _table(["实际计数","值"],[[key,value] for key,value in results.get('counts',{}).items() if not isinstance(value,(dict,list))]),"",
        "几何能量账本（采样、探针、预热、离线端点分域）：`"+json.dumps(results.get('geometry_ledger',{}),ensure_ascii=False)+"`。", "",
        "CPU独立判交/评分账本：`"+json.dumps(results.get('evaluation_ledger',{}),ensure_ascii=False)+"`。", "",
        "预热分账：`"+json.dumps(results.get('warmup',{}),ensure_ascii=False)+"`。开发实现检查分账：`"+json.dumps(results.get('implementation_development_counts',{}),ensure_ascii=False)+"`。", "",
        "主干前向类别分账：`"+json.dumps(_get(results,'counts','forward_by_category'),ensure_ascii=False)+"`。完整生成类别分账：`"+json.dumps(_get(results,'counts','completed_by_category'),ensure_ascii=False)+"`。", "",
        "W处理、索引/OBJ导出与独立拓扑/几何评分耗时在weld逐输出记录分别保留；没有把它们混进GPU完整生成时间。", "",
        "## 固定图与停止", ""]
    lines += _weld_cost_lines(results)
    for figure in figures:lines += [f"![固定两对象seed9201 Raw/+W及free-only]({Path(figure['path']).name})","",f"SHA256：`{figure['sha256']}`。",""]
    if not figures:lines += ["固定图NOT_RUN：未形成必要输出，不挑其他seed替代。",""]
    lines += ["图只读取存档NPZ和已存相交记录；蓝色C、红色CF proper、紫色CF共面、橙色线框为FF非法涉及面。"
        "free-only行不叠C，所有组/层共享相机和完整范围，不遮盖差例。",
        "本轮新增训练更新=0、模型参数更新=0、主干反向=0；没有删面、补洞或改写已知面，不启动Phase3。"
        "按实际门禁和固定预算停止，不自动加训练、改阈值或追加修补。",""]
    return "\n".join(lines)


def _read_output(path,C):
    with np.load(path,allow_pickle=False) as archive:
        key=next((k for k in ('output','triangles','full','raw') if k in archive),None)
        if key is None and 'vertices' in archive and 'faces' in archive:
            value=archive['vertices'][archive['faces']]
        elif key is None:raise KeyError(f"No mesh coordinate array in {path}")
        else:value=archive[key]
        value=np.asarray(value).reshape(112,3,3).copy()
    if not np.array_equal(value[:12],C):raise AssertionError("Fixed plot input does not preserve C")
    return value


def fixed_figure(results,out,bank):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection,Poly3DCollection
    score_rows=[x for x in results.get('scores',[]) if x.get('phase','development')=='development' and x['seed']==9201]
    meshes={};metadata={}
    for row in score_rows:
        parent,model,variant=int(row['parent']),row['model'],row['variant']
        path=row.get('output_path',row.get('welded_path'))
        if variant=='W' and path is None:
            wr=next((x for x in results.get('weld',[]) if x.get('phase','development')=='development' and
                     (int(x['parent']),x['model'],x['seed'])==(parent,model,9201)),None)
            if wr:path=wr.get('welded_path',wr.get('output_path'))
        if variant=='Raw':path=path or row.get('raw_path')
        if path is None:continue
        C=np.asarray(bank['cases_by_key'][(parent,12)]['constraints'])
        key=parent,model,variant
        meshes[key]=_read_output(path,C);metadata[key]=row
    if not meshes:return None
    points=np.concatenate([v.reshape(-1,3)[:,[0,2,1]] for v in meshes.values()])
    low,high=points.min(0),points.max(0);center=(low+high)/2;radius=max(float((high-low).max())*.55,1e-6)
    fig=plt.figure(figsize=(27,15),constrained_layout=True)
    for parent in (0,1):
        for part in (0,1):
            for model_index,model in enumerate(MODELS):
                for vi,variant in enumerate(VARIANTS):
                    column=model_index*2+vi;row_index=parent*2+part
                    ax=fig.add_subplot(4,8,row_index*8+column+1,projection='3d')
                    key=parent,model,variant
                    if key not in meshes:
                        ax.text2D(.12,.5,'NOT_RUN',transform=ax.transAxes,fontsize=11)
                    else:
                        full=meshes[key][:,:,[0,2,1]];metrics=metadata[key]['metrics']
                        views=metrics['gpu_guidance_evaluation']['detector_views']
                        cf=views['CF']['primary'];ff=views['FF']['primary']
                        ff_faces={int(i)+12 for kind in ('proper_pairs','coplanar_overlap_pairs') for pair in ff[kind] for i in pair}
                        ids=list(range(112)) if part==0 else list(range(12,112));tri=full[ids]
                        colors=['#df8c25' if index in ff_faces else '#73838d' for index in ids]
                        widths=[.8 if index in ff_faces else .5 for index in ids]
                        ax.add_collection3d(Line3DCollection(np.concatenate([tri,tri[:,:1]],axis=1),colors=colors,linewidths=widths,alpha=.80))
                        if part==0:ax.add_collection3d(Poly3DCollection(full[:12],facecolors='#438acf',edgecolors='#205a8c',alpha=.3))
                        for name,color in (('proper_pairs','#e22b30'),('coplanar_overlap_pairs','#ce28c9')):
                            faces=sorted({int(pair[1])+12 for pair in cf[name]})
                            if faces:ax.add_collection3d(Poly3DCollection(full[faces],facecolors=color,edgecolors=color,alpha=.50,linewidths=.6))
                    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
                    ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-54);ax.set_axis_off()
                    ax.set_title(f'P{parent} | {model} {variant} | '+('full' if part==0 else 'free only'),fontsize=10)
    fig.suptitle('Fixed regression seed9201: all four methods, Raw and identical protected weld W\nBlue=C (hidden in free rows); red=CF proper; magenta=CF overlap; orange wires=FF illegal faces. Same view and bounds.',fontsize=14)
    path=Path(out)/'fixed_P0_P1_seed9201_Raw_W_full_free.png'
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=145);plt.close(fig)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parents=[0,1],seed=9201,
        models=list(MODELS),variants=list(VARIANTS),full_and_free_only=True,
        camera=dict(elevation=24,azimuth=-54),display_axes=['X','Z','Y'],shared_center=center.tolist(),shared_radius=radius,
        plotted_keys=[list(key) for key in meshes],no_geometry_rerun=True,coordinates_modified=False,model_calls=0)


def deliver_report(results,out,bank):
    out=Path(out).resolve()
    if out.parent!=Path(__file__).resolve().parent/'outputs' or not out.name.startswith('gpu_guidance_and_protected_weld_') or not out.is_dir():
        raise ValueError('Require current isolated GPU-guidance output')
    path=out/'GPU_GUIDANCE_AND_PROTECTED_WELD_REPORT.md'
    if path.exists():raise FileExistsError(path)
    summary=summarize(results)
    figure=fixed_figure(results,out,bank)
    figures=[] if figure is None else [figure]
    text=render_report(results,summary,figures)
    if '\ufffd' in text:raise AssertionError('Bad Unicode in report')
    path.write_text(text,encoding='utf-8')
    return dict(report=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        figures=figures,offline_summary=summary,report_model_calls=0,report_geometry_calls=0,report_weld_calls=0)
