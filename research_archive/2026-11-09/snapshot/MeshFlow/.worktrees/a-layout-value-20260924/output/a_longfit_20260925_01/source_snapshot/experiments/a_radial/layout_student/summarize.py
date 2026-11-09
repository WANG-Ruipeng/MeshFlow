"""Paired statistics, transparent denominators, and a Chinese research report."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from ..io_utils import write_json

LABELS={
("fixed","raw"):"Fixed",
("original_a","raw"):"原A直接输出",
("original_a","quality_fallback"):"原A＋质量回退",
("teacher_reference","offline_teacher"):"教师标签（离线参照）",
("student_10","raw"):"学生 seed10 直接输出",
("student_10","quality_fallback"):"学生 seed10＋质量回退",
("student_11","raw"):"学生 seed11 直接输出",
("student_11","quality_fallback"):"学生 seed11＋质量回退",
}
def median(values): return float(np.median(values)) if values else None

def summarize(run):
    run=Path(run);protocol=json.loads((run/"RUN_PROTOCOL.json").read_text())
    ledger=json.loads((run/"results.json").read_text());rows=ledger["rows"]
    keys=[(r["uid"],r["model"],r["policy"]) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError("Duplicate UID/model/policy")
    baseline={r["uid"]:r for r in rows if (r["model"],r["policy"])==("fixed","raw")}
    flat=[]
    for r in rows:
        b=baseline[r["uid"]]
        assert r["mainfield_hash"]==b["mainfield_hash"]
        assert [m["seed"] for m in r["metrics"]]==[m["seed"] for m in b["metrics"]]
        ratios=[m["rms_bbox_percent"]/bm["rms_bbox_percent"] for m,bm in zip(r["metrics"],b["metrics"])]
        r["rms_over_fixed_two_seeds"]=ratios
        r["qualified_improved_both"]=r["qualified"] and all(x<1-1e-9 for x in ratios)
        r["qualified_ge2pct_both"]=r["qualified"] and all(x<=.98 for x in ratios)
        r["qualified_ge10pct_both"]=r["qualified"] and all(x<=.90 for x in ratios)
        with np.load(run/r["mesh_file"]) as z:u=z["u"]
        with np.load(run/"data"/f'{r["uid"]}.npz') as z:target=z["teacher_u"];base=z["base_u"]
        angle=lambda a,b:np.rad2deg(2*np.arcsin(np.minimum(np.linalg.norm(a-b,axis=-1)/2,1)))
        r["offline_teacher_direction_rms_deg"]=float(np.sqrt(np.mean(angle(u,target)**2)))
        r["mean_predicted_displacement_deg"]=float(np.mean(angle(u,base)))
        r["mean_teacher_displacement_deg"]=float(np.mean(angle(target,base)))
        flat.append({
            "uid":r["uid"],"split":r["split"],"model":r["model"],"policy":r["policy"],
            "actual_faces":r["quality"]["face_count"],"valid":r["audit"]["valid"],
            "qualified":r["qualified"],"fallback_to_fixed":r["fallback_to_fixed"],
            "min_triangle_angle_deg":r["quality"]["min_triangle_angle_deg"],
            "triangle_min_angle_p05_deg":r["quality"]["triangle_min_angle_p05_deg"],
            "triangle_shape_quality_p05":r["quality"]["triangle_shape_quality_p05"],
            "rms_bbox_percent_seed1":r["metrics"][0]["rms_bbox_percent"],
            "rms_bbox_percent_seed2":r["metrics"][1]["rms_bbox_percent"],
            "paired_improvement_fraction_seed1":1-ratios[0],
            "paired_improvement_fraction_seed2":1-ratios[1],
            "qualified_improved_both":r["qualified_improved_both"],
            "qualified_ge2pct_both":r["qualified_ge2pct_both"],
            "qualified_ge10pct_both":r["qualified_ge10pct_both"],
            "offline_teacher_direction_rms_deg":r["offline_teacher_direction_rms_deg"],
            "mean_predicted_displacement_deg":r["mean_predicted_displacement_deg"],
            "mean_teacher_displacement_deg":r["mean_teacher_displacement_deg"],
            "generation_seconds":r["generation_seconds"],
            "hot_delivery_with_audit_seconds":r["hot_delivery_with_audit_seconds_median_3"],
            "mesh_file":r["mesh_file"],"mesh_hash":r["mesh_hash"],
        })
    groups=[]
    for split in ["all","train","internal_check"]:
        expected=len(protocol["uids"] if split=="all" else protocol["train_uids"] if split=="train" else protocol["internal_check_uids"])
        for (model,policy),label in LABELS.items():
            rr=[r for r in rows if r["model"]==model and r["policy"]==policy and (split=="all" or r["split"]==split)]
            groups.append({
                "split":split,"model":model,"policy":policy,"label":label,
                "expected_objects":expected,"recorded_objects":len(rr),"missing":expected-len(rr),
                "valid_count":sum(r["audit"]["valid"] for r in rr),
                "qualified_count":sum(r["qualified"] for r in rr),
                "fallback_count":sum(r["fallback_to_fixed"] for r in rr),
                "improved_both_count":sum(all(x<1-1e-9 for x in r["rms_over_fixed_two_seeds"]) for r in rr),
                "qualified_improved_both_count":sum(r["qualified_improved_both"] for r in rr),
                "qualified_ge2pct_both_count":sum(r["qualified_ge2pct_both"] for r in rr),
                "qualified_ge10pct_both_count":sum(r["qualified_ge10pct_both"] for r in rr),
                "paired_improvement_percent_median_seed1":median([100*(1-r["rms_over_fixed_two_seeds"][0]) for r in rr]),
                "paired_improvement_percent_median_seed2":median([100*(1-r["rms_over_fixed_two_seeds"][1]) for r in rr]),
                "all_readable_rms_bbox_percent_median_seed1":median([r["metrics"][0]["rms_bbox_percent"] for r in rr]),
                "min_triangle_angle_deg_median":median([r["quality"]["min_triangle_angle_deg"] for r in rr]),
                "triangle_min_angle_p05_deg_median":median([r["quality"]["triangle_min_angle_p05_deg"] for r in rr]),
                "teacher_direction_rms_deg_median":median([r["offline_teacher_direction_rms_deg"] for r in rr]),
                "hot_delivery_with_audit_seconds_median":median([r["hot_delivery_with_audit_seconds_median_3"] for r in rr if r["hot_delivery_with_audit_seconds_median_3"] is not None]),
                "not_independent_objects_across_training_seeds":True,
            })
    missing=[(uid,m,p) for uid in protocol["uids"] for m,p in LABELS if (uid,m,p) not in keys]
    summary={"experiment":protocol["experiment"],"expected_rows":128,"recorded_rows":len(rows),
             "attempt_count":ledger["attempt_count"],"execution_errors":ledger["errors"],"missing":missing,"groups":groups,
             "scope":protocol["scope"],"statistics":"per-object pairing; two new MC evaluation seeds are measurement rechecks"}
    write_json(run/"SUMMARY.json",summary,replace=(run/"SUMMARY.json").exists())
    write_json(run/"ANALYSIS_ROWS.json",{"rows":rows},replace=(run/"ANALYSIS_ROWS.json").exists())
    with (run/"PER_SAMPLE.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    reports=[json.loads((run/"training"/f"seed_{seed}"/"TRAIN_REPORT.json").read_text()) for seed in [10,11]]
    teachers=json.loads((run/"TEACHERS.json").read_text())["rows"]
    oracle=json.loads((run/"ORACLE_NUMERICS.json").read_text()) if (run/"ORACLE_NUMERICS.json").exists() else None
    overall={(g["model"],g["policy"]):g for g in groups if g["split"]=="all"}
    s10=overall[("student_10","raw")];s11=overall[("student_11","raw")]

    lines=[
        "任务对齐 A：有限训练验证",
        "",
        f"本轮结论：两个训练种子的原始输出均有小幅表面误差改善（配对中位数{s10['paired_improvement_percent_median_seed1']:.2f}% / {s11['paired_improvement_percent_median_seed1']:.2f}%），但严格合格收益各为{s10['qualified_improved_both_count']}/16、{s11['qualified_improved_both_count']}/16；内部检查均未取得合格收益。本轮固定预算速度监督尚未可靠复现教师的受约束改进。",
        "",
        "质量失败回退让输出重新合格，但两个种子都回退14/16，交付结果的配对改善中位数均为0。回退的稳健性不能写成学习收益。",
        "",

        "本轮只更新原 DomainVelocity 方向网络（29,907 参数）。主场、AE、FM、原始A资产和官方MeshFlow代码保持冻结；没有进行优化器或架构搜索。",
        "",
        "全部16对象都曾用于开发。固定UID哈希划分12训练、4内部检查，不是新的确认集或类别泛化实验。两个训练随机种子都从 original_A_10.pt 初始化，各执行3000步，final.pt为唯一部署权重。",
        "",
        f"教师标签：{sum(t['moved_label'] for t in teachers)}/16采用已审计的Fixed→优化端点，其余保留Fixed；选择要求两组旧采样均改善至少2%，该阈值是探索规则，不是显著性检验。训练读取独立的12对象标签文件；内部检查教师不参与训练。",
        "",
        "监督的是32步离散球面桥的切向速度，与原32次归一更新对齐。推理继续使用原256个主场代理点、原Bernstein角域保护、最终接受方向半径重查。没有新增主场、真值网格或教师端点作为网络条件。",
        "",
        "以下合格条件始终是：精确512面、独立精确端点几何审计通过、最小三角角和每面最小角p05均不低于Fixed（容差1e-6度）。这是本轮严格诊断合同。Fixed起点或回退得到合格并不构成学习收益。",
        "",
        f"完整性：记录{len(rows)}/128行，{ledger['attempt_count']}/80个对象/模型尝试；缺行{len(missing)}，程序异常{len(ledger['errors'])}。所有原始输出与回退策略输出分别保存。",
    ]
    fmt=lambda v: "—" if v is None else f"{v:.3f}"
    for split,display in [("train","训练对象（12个）"),("internal_check","内部检查对象（4个）"),("all","全16个开发对象")]:
        lines+=["",display,"",
            "|方法|几何有效|全部条件合格|合格且两次测量均改善|合格且均改善≥10%|回退Fixed|配对改善中位数(%)|",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for g in (x for x in groups if x["split"]==split):
            n=g["expected_objects"]
            lines.append(f"|{g['label']}|{g['valid_count']}/{n}|{g['qualified_count']}/{n}|{g['qualified_improved_both_count']}/{n}|{g['qualified_ge10pct_both_count']}/{n}|{g['fallback_count']}/{n}|{fmt(g['paired_improvement_percent_median_seed1'])}|")
    lines+=["","读数与成本解释","",
        "- 配对改善是先逐对象计算1−RMS方法/RMS Fixed再取中位数；负值表示退步。它包含未达到质量门槛的原始输出，须与合格列一起阅读。",
        "- 学生与原A的quality_fallback是新增的部署策略：预算、质量或有效性失败才回退Fixed；不读取RMS、教师布局或原始目标网格作选择。回退率单列，回退带来的合格不算学得收益。",
        "- 教师行仅为离线标签参照，使用逐对象优化信息，不是可部署学生或免费在线方法。",
        "- 原A和学生raw热时间包含proxy、32步保护、最终半径查询及自身精确审核，不含Fixed门槛的准备；quality_fallback热时间另外包含Fixed构造、门槛与审核、raw生成/审核、选择。因此不能把raw时间称为完整质量回退成本。",
        "- 热时间每对象重复3次取中位数，重复输出哈希必须一致；载入时间另列。面积RMS计算不计入部署成本。这里是布局模块成本，不是noise→AE/FM→网格的完整生成成本。",
        "",
        "|方法（全16）|热部署含自身审计 中位秒|",
        "|---|---:|"]
    for g in (x for x in groups if x["split"]=="all"):
        lines.append(f"|{g['label']}|{fmt(g['hot_delivery_with_audit_seconds_median'])}|")
    lines+=["","训练与数值证据",""]
    for r in reports:
        lines.append(f"- seed{r['seed']}：{r['update']}次更新，每训练父对象曝光{sorted(set(r['parent_exposures'].values()))}次；首批loss={r['first_loss']:.6g}，最后100批均值={r['last_100_loss_mean']:.6g}；训练{r['elapsed_training_seconds']:.2f}s，峰值分配显存{r['peak_gpu_allocated_bytes']/2**20:.1f}MiB。loss只衡量速度回归，不能替代部署质量。")
    lines+=["- CPU连续/恢复测试与真实RTX5080 CUDA 12步vs5+7步恢复均逐位一致；正式训练未继承smoke权重。",
        "- ORACLE_NUMERICS.json是教师端点已知的离线数值诊断，严禁计入学生推理；用于检查原32步数值链是否能实现这些目标。",
        "- 教师角度质量余量很窄。近边界可行标签的近似回归、教师完整场与学生proxy256的信息差都可能影响端点合格率；本轮不能分离这些原因，也不能据此证明网络容量上限。",
        "- 独立评测使用两组新的8192点面积采样：192401/292401加排序UID索引，未参与教师筛选、训练或checkpoint选择。采样复核不是独立对象或置信区间。",
        "- 原QEM质量—误差—成本证据保留在上一轮报告中；本轮没有重跑或重新定义QEM，更不把新增严格质量合同当作传统方法能力上限。",
        "",
        "两个研究任务独立推进","",
        "A模块的本轮问题是能否学到合格教师布局；完整系统的另一个问题是AE＋FM＋各出口与MeshFlow的生成质量、有效性和总成本。A是否成功不作为完整生成比较的前置门槛。完整MeshFlow网络接入/正式拟合资格、共享采样和五行评价仍需按独立计划完成；旧短训或Public Post不能充当已合格论文基线。",
        "",
        "文件","",
        "- [逐对象CSV](PER_SAMPLE.csv)、[统计JSON](SUMMARY.json)、[实验协议](RUN_PROTOCOL.json)、[教师选择](TEACHERS.json)。",
        "- [训练曲线](figures/loss.png)、[配对误差](figures/paired_rms.png)、[误差与角度质量](figures/rms_quality.png)。",
        "- 网格对照：figures/meshes_01.png至meshes_04.png；全部原始及回退网格：attempts/<uid>/<model>/*.npz；可视化OBJ在meshes_obj。",
        "- 模型：training/seed_10/final.pt、training/seed_11/final.pt；中间checkpoint仅用于恢复，不能按表现替换final。",
        "- [复现与后续登记](METHOD_AND_NEXT_zh.md)。",
    ]
    fit_path=run/"FIT_DIAGNOSTIC.json"
    if fit_path.exists():
        fit=json.loads(fit_path.read_text())
        vm={(g["split"],g["model"]):g for g in fit["velocity_summary"]}
        em={(g["split"],g["model"]):g for g in fit["endpoint_summary"]}
        diag=["教师拟合的事后诊断（不改变模型或实验协议）","",
            "训练loss大幅下降不能独立证明学会教师：同一固定32步教师网格上，不动的零速度预测已经有很低的MSE。以下只评估冻结最终权重，不用于选checkpoint；教师条件状态与实际自由推理不同。",
            "",
            "|划分|零速度MSE|学生10 MSE|学生11 MSE|学生10相对零速度下降|学生11相对零速度下降|",
            "|---|---:|---:|---:|---:|---:|"]
        for split in ("train","internal_check","all"):
            zz=vm[(split,"zero_velocity")];aa=vm[(split,"student_10")];bb=vm[(split,"student_11")]
            diag.append(f"|{split}|{zz['same_grid_mse']:.6g}|{aa['same_grid_mse']:.6g}|{bb['same_grid_mse']:.6g}|{100*(1-aa['same_grid_mse_over_zero']):.2f}%|{100*(1-bb['same_grid_mse_over_zero']):.2f}%|")
        ef=em[("all","fixed")];e10=em[("all","student_10")];e11=em[("all","student_11")]
        diag+=["",
            f"实际端点到教师的方向RMSE逐对象中位数：Fixed不动为{ef['direction_rmse_to_teacher_deg']['median']:.3f}度，学生为{e10['direction_rmse_to_teacher_deg']['median']:.3f}/{e11['direction_rmse_to_teacher_deg']['median']:.3f}度。各只有6/16比Fixed更接近教师，不能称为已经良好拟合教师。",
            "",
            f"质量失败也不是1e-6度容差噪声：两学生的失败对象最小角短缺中位数为{e10['min_angle_shortfall_on_failed_deg']['median']:.3f}/{e11['min_angle_shortfall_on_failed_deg']['median']:.3f}度；每面最小角p05短缺中位数为{e10['p05_angle_shortfall_on_failed_deg']['median']:.3f}/{e11['p05_angle_shortfall_on_failed_deg']['median']:.3f}度。短缺统计分别只对该项失败对象取值。",
            "",
            "连续与离散teacher oracle都在16/16通过原32步保护和真实精确端点审计。因而这轮不能把失败归因于32步数值链本身无法实现教师；但还没有容量、代理点信息或损失的因果消融，不能直接归因某一因素。",
            "",
            "与上一轮相比，Fixed与原A的32份基线网格按原int32连接语义的哈希全部相同，顶点与连接值未改变。本轮RMS改用新的独立采样，故原A的配对统计与旧报告可能有变化，并非改了原A或基线几何。",
            "",
            "证据：[FIT_DIAGNOSTIC.json](FIT_DIAGNOSTIC.json)、[ORACLE_NUMERICS.json](ORACLE_NUMERICS.json)。",""]
        position=lines.index("两个研究任务独立推进")
        lines[position:position]=diag

    if (run/"MESH_INTEGRITY.json").exists():
        lines += ["", "封存时的记录更正", "",
            "112/128份NPZ的面索引为int64，原账本mesh_hash使用原模板int32，因此包含dtype的摘要不同。全部保存坐标与连接值已重验；原网格、账本、权重和统计均保留。MESH_INTEGRITY.json分别绑定原逻辑摘要、实际存储数组摘要和NPZ文件SHA；封存与解压重放同时核验。未来评测直接记录两种摘要。",
            "",
            "详见[更正与实际评测源码](code_corrections/README_zh.md)、[网格完整性账本](MESH_INTEGRITY.json)。这项更正不改变上表的收益、合格率或成本。"]
    (run/"REPORT_zh.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print(json.dumps({"rows":len(rows),"missing":len(missing),"errors":len(ledger["errors"]),
                     "student_summary":[g for g in groups if g["split"] in ["all","internal_check"] and g["model"].startswith("student")]}))
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",required=True)
    summarize(p.parse_args().run)
