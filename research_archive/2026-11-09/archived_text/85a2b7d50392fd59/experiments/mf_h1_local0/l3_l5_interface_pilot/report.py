"""Saved-output presentation only: no model, loss, OT or geometry evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from .evaluation import MODELS,PAIRS,EFFECTS,_path,_seeds,_sha

RMS="remainder_symmetric_RMS_bbox_pct";GAMMA="boundary_RMS_bbox_pct"
SURFACE=(RMS,"free_to_R12_RMS_bbox_pct","R12_to_free_RMS_bbox_pct",
 "fixed_R12_coverage_RMS_bbox_pct","fixed_R12_coverage_within_1pct_fraction",
 GAMMA,"boundary_p95_bbox_pct","boundary_coverage_fraction","whole_symmetric_RMS_bbox_pct")
QUALITY=("CF_proper_affected_free_fraction","CF_proper_pair_count","FF_illegal_pair_count",
 "CF_coplanar_pair_count","FF_coplanar_pair_count","CF_unresolved_or_degenerate_pair_count",
 "FF_unresolved_or_degenerate_pair_count","free_minimum_angle_lt5_fraction","free_q_gt10_fraction",
 "free_near_degenerate_count","free_exact_zero_area_count","free_total_surface_area",
 "nan_coordinate_count","inf_coordinate_count","exact_C_free_shared_edge_count",
 "exact_C_free_shared_vertex_count","C_inherited_nonmanifold_vertex_count",
 "new_free_related_nonmanifold_vertex_count")
CORE=(RMS,"R12_to_free_RMS_bbox_pct","fixed_R12_coverage_within_1pct_fraction",
 GAMMA,"boundary_p95_bbox_pct","boundary_coverage_fraction","CF_proper_pair_count","FF_illegal_pair_count")
NAMES={"L0":"FM","L3":"FM+L3","L5":"FM+L5","J35":"FM+L3+L5","JEdge5":"FM+D_edge+L5"}

def _fmt(x):
    if x is None:return "NOT_RECORDED"
    if isinstance(x,bool):return str(x)
    if isinstance(x,(int,np.integer)):return str(int(x))
    if isinstance(x,(float,np.floating)):return f"{x:.6g}" if np.isfinite(x) else "NONFINITE"
    if isinstance(x,(dict,list,tuple)):x=json.dumps(x,ensure_ascii=False,separators=(",",":"))
    return str(x).replace("|","\\|").replace("\n"," ")

def _table(headers,rows):
    return ["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |"]+[
        "| "+" | ".join(_fmt(x) for x in row)+" |" for row in rows]

def _dist(x):
    if not x or not x.get("finite_count"):return "NOT_RECORDED"
    worst=x.get("worst");worst=worst.get("value") if isinstance(worst,dict) else worst
    return " / ".join(_fmt(x.get(k)) for k in ("mean","median"))+" / "+("N/A" if worst is None else _fmt(worst))

def _paired(cell,key):
    c=cell.get("improvement_counts",{}).get(key,{}) or {}
    return _dist(cell.get("metrics",{}).get(key))+f"; {c.get('better','N/A')}/{c.get('available','N/A')}"

def _record(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=_sha(p),bytes=p.stat().st_size)

def _scalars(value,prefix="",depth=0):
    if not isinstance(value,dict):return [(prefix,value)]
    rows=[]
    for k,v in value.items():
        name=prefix+"."+k if prefix else k
        if isinstance(v,dict) and depth<3:rows.extend(_scalars(v,name,depth+1))
        elif not isinstance(v,(dict,list,tuple)):rows.append((name,v))
    return rows

def _validate(r):
    s=r["summaries"]["pilot"]
    if tuple(s["models"])!=MODELS or len(s["seeds"])!=4:raise ValueError("Fixed five-model four-seed summary required")
    if len(r["scores"])!=120 or len(r["generated"])!=120:raise ValueError("All120 formal outputs required")
    for j in (0,1,2):
        for m in MODELS:
            for p in ("0","1"):
                if len(s["patches"][str(j)]["models"][m]["parents"][p]["per_seed"])!=4:raise ValueError("Four noises per cell required")
    return s

def _outcomes(s):
    lines=["## 本批匹配的完整生成","",
        "所有终点累计3000；旧三臂仅重载，两个组合臂各新增500。各parent/patch均4个新噪声，先单元内平均再六单元等权；没有排名/选择或补位。误差单位为% bbox对角线L，比例为[0,1]。表格为均值 / 中位数 / 最差值；面积无预设好坏方向。",""]
    lines+=_table(["模型"]+list(CORE),[[m]+[s["macros"][m][k]["macro_mean"] for k in CORE] for m in MODELS])+[""]
    for j in (0,1,2):
        block=s["patches"][str(j)]
        for p in ("0","1"):
            lines += [f"### P{p} / patch{j} / n=4",""]
            for title,keys in (("表面",SURFACE),("独立质量",QUALITY)):
                lines+=_table([title]+list(MODELS),[[k]+[_dist(block["models"][m]["parents"][p]["metrics"].get(k)) for m in MODELS] for k in keys])+[""]
            lines += ["逐seed配对差先计算，再求均值/中位数/最差及改善次数；不是边际中位数相减。负误差/计数差为改善，正coverage差为改善。",""]
            lines+=_table(["比较"]+list(CORE),[[a+"−"+b]+[_paired(block["paired"][a+"_minus_"+b]["parents"][p],k) for k in CORE] for a,b in PAIRS])+[""]
    return lines

def _factorial(s):
    lines=["## 固定系数L3/L5四格与交互","",
        "D3_when5off=L3−L0；D3_when5on=J35−L5；D5_when3off=L5−L0；D5_when3on=J35−L3；I=J35−L3−L5+L0。全部先逐seed计算，再按单元汇总；六cell宏值为单元均值的等权平均。负I只描述当前固定系数下的非加和性，没有等总监督强度对照，不能单凭它宣称纯机制协同。",""]
    f=s["factorial"]
    lines+=_table(["效应宏均值"]+list(CORE),[[name]+[f["macro"][name][k]["macro_mean"] for k in CORE] for name in EFFECTS])+[""]
    for j in (0,1,2):
        for p in (0,1):
            lines += [f"### P{p}/patch{j} 逐seed效应统计",""]
            lines+=_table(["效应"]+list(CORE),[[name]+[_paired(f["per_cell"][name][f"P{p}_patch{j}"],k) for k in CORE] for name in EFFECTS])+[""]
    lines += ["D_edge没有独立单项臂；JEdge5−J35仅评价固定L5背景下以有限边目标替换L3，不能构造完整D_edge×L5交互或宣布其单独优于L3。",""]
    return lines

def _binding(s):
    lines=["## 同噪声条件绑定","",
        "四格E00/E01/E11/E10复用输出的独立交叉评分，不新增采样。Delta0=E01−E00，Delta1=E10−E11，B为两者平均；基线已有正向绑定，重点看增量和逐seed集中度。",""]
    for j in (0,1,2):
        block=s["patches"][str(j)];rows=[]
        for m in MODELS:
            for x in block["models"][m]["binding"]["per_seed"]:rows.append([m,x["seed"]]+[x[k] for k in ("E00","E01","E11","E10","Delta0","Delta1","B")])
        lines += [f"### patch{j}",""]+_table(["模型","seed","E00","E01","E11","E10","Delta0","Delta1","B"],rows)+[""]
        lines+=_table(["模型","B均值/中位/最差","两方向正"],[[m,_dist(block["models"][m]["binding"]["metrics"]["B"]),block["models"][m]["binding"]["both_positive_count"]] for m in MODELS])+[""]
        lines+=_table(["比较","B增量均值/中位/最差","B增加次数"],[[name,_dist(v["binding"]["metrics"]["B"]),v["binding"]["higher_B_count"]] for name,v in block["paired"].items()])+[""]
    return lines

def _teacher(r):
    lines=["## 同一固定状态的终点诊断","",
        "五个累计3000终点共用16个保存状态，最多80次单步前向；只读计算FM/L3/L5/D_edge，没有随机辅助头。16状态来自8个训练配对各两个时间，不是16独立对象。此处与训练step500更新前（累计2999）的日志分开；状态损失改善不保证完整自由生成改善。",""]
    teacher=r.get("teacher",{});models=teacher.get("models",{})
    lines+=_table(["模型","状态","记录数","FM","L3","L5","D_edge","seconds"],[[m,models.get(m,{}).get("status"),len(models.get(m,{}).get("rows",[]))]+[models.get(m,{}).get("means",{}).get(k) for k in ("FM","L3","L5","D_edge")]+[models.get(m,{}).get("seconds")] for m in MODELS])+[""]
    for a,b in PAIRS:
        aa=models.get(a,{}).get("rows",[]);bb=models.get(b,{}).get("rows",[])
        ai={x.get("state_index"):x for x in aa};bi={x.get("state_index"):x for x in bb}
        if not ai or set(ai)!=set(bi):continue
        rows=[]
        for k in ("FM","L3","L5","D_edge"):
            dif=[ai[i][k]-bi[i][k] for i in sorted(ai) if isinstance(ai[i].get(k),(int,float)) and isinstance(bi[i].get(k),(int,float))]
            rows.append([k,float(np.mean(dif)) if dif else None,float(np.median(dif)) if dif else None,sum(v<0 for v in dif),len(dif)])
        lines += [a+"−"+b+"：固定状态配对诊断（非训练分布期望）",""]+_table(["目标","均差","配对中位","负差数","状态数"],rows)+[""]
    return lines


def _protocol(r):
    lines=["## 固定协议与边界","",
        "仅新增J35/JEdge5各500更新，总1000；共同G0累计2500，复用seed1010起点batch1000/sample8000的原500批输入流，终点累计3000。L0/L3/L5旧终点只读，不从旧3000继续训练，不新增纯D_edge臂。","",
        "原FM整批自由坐标分母不变；每项辅助先按样本内部定义归一，再按effective8平均，inactive仍在分母。g(t)=I[t≥.5]，ramp(s)=min(s/50,1)。fresh AdamW lr1e−5、betas(.9,.95)、wd0、clip1；原主干/role/Geo全量训练，模块eval且保留autograd。主干BF16、参数FP32、几何FP32。250 checkpoint只归档，完整生成固定500终点。","",
        "lambda3=3.6819396517801244，lambda5=0.3266104383520167，沿用保存精确值。J35不重新整体缩放；JEdge5只标定lambda_edge=lambda3·median(||g3||)/median(||gE||)，L5系数不变。联合监督总量可高于单项，固定系数交互不排除总强度影响。","",
        "L3仍为GT接口对应端点的各向异性误差，垂直边界线权重1、沿线.05。L5仍为三点求积、A_pred/(3A_GT,total)、空间核×无向Q=nnᵀ核，sigma_x=.02/.05/.10、sigma_Q=.5，位置/面积/法向均保留梯度。","",
        "D_edge仅用GT中恰好一known一free的真实共享边，每条登记一次：在s=0/.25/.5/.75/1，w=1/8,1/4,1/4,1/4,1/8上，GT边长加权双向有限线段距离²/(2·L_Gamma·L²)。坐标先除L，预测短边分母epsilon=1e−12；记录触发，不删边或修复。GT对应只用于监督/固定状态诊断，自由生成Gamma仍沿用独立无对应评分。","",
        "原50步Euler、N112/K12/free100，同seed共享原始Gaussian，不做GT OT采样初始化；C全部51状态逐位保持。采样无loss/GT/辅助头/几何梯度/修补，旧BF16重排FAIL按baseline-inheritance保留。无共享顶点建立、水密或全局无交保证。",""]
    return lines

def _accounting(r):
    lines=["## 资产、检查与实际成本",""]
    assets=r.get("assets",{});rows=[]
    for name,rec in assets.get("endpoints",{}).items():
        rows.append([name,rec.get("path"),rec.get("bytes"),rec.get("sha256"),rec.get("state_sha256",rec.get("model_state_sha256")),rec.get("cumulative_updates")])
    lines+=_table(["终点","路径","bytes","文件SHA","stateSHA","累计更新"],rows)+[""]
    rows=[]
    for m,a in r.get("arms",{}).items():
        for step,rec in a.get("checkpoints",{}).items():rows.append([m,step,rec.get("path"),rec.get("bytes"),rec.get("sha256")])
    lines+=_table(["新臂","新增步","本地路径","bytes","SHA"],rows)+[""]
    for name,rec in assets.items():
        if isinstance(rec,dict) and name!="endpoints":
            lines+=_table(["资产","路径","bytes","SHA"],[[name,rec.get("path"),rec.get("bytes"),rec.get("sha256")]])+[""]
    for name in ("environment","CPU_fixtures","counts","diagnostic_counts","timing","timing_summary","protection","actual_accounting"):
        if r.get(name):lines += [f"### {name}",""]+_table(["字段","记录"],_scalars(r[name]))+[""]
    cal=r.get("calibration",{})
    lines += ["### 一次标定与联合梯度","",
        "lambda_edge仅使用同一组有效初始化状态匹配接口项输出速度梯度；没有动态权重、梯度投影或联合重标定。下面范数/夹角只用于诊断，不触发规则变化。",""]
    lines+=_table(["标定字段","记录"],_scalars({k:v for k,v in cal.items() if k not in ("rows","states","strict_initial")}))+[""]
    if cal.get("rows"):
        lines+=_table(["state","parent","K","t","FM","L3","L5","D_edge"],
            [[x.get("state_index"),x.get("parent"),x.get("K"),x.get("t")]+
              [x.get("values",{}).get(k) for k in ("FM","L3","L5","D_edge")] for x in cal["rows"]])+[""]
        lines+=_table(["state","||gFM||","||g3||","||g5||","||gE||","cos(3,5)","cos(E,5)",
            "J35 norm","J35/FM","cos(J35,FM)","JEdge5 norm","JEdge5/FM","cos(JEdge5,FM)"],
            [[x.get("state_index")]+[x.get("gradient",{}).get("norms",{}).get(k) for k in ("FM","L3","L5","D_edge")]+
             [x.get("gradient",{}).get("cos_L3_L5"),x.get("gradient",{}).get("cos_edge_L5")]+
             [x.get("gradient",{}).get("joint",{}).get(m,{}).get(k) for m in ("J35","JEdge5") for k in ("L2","ratio_to_FM","cos_to_FM")]
             for x in cal["rows"]])+[""]
        lines += ["标定状态/velocity/标签hash、旧L3/L5全梯度逐位一致检查及细项诊断完整保存在results；正文只展开分项/联合范数与方向，避免复制大字典。",""]
    for m,a in r.get("arms",{}).items():
        lines += [f"### {m} 训练诊断","",
            "预登记日志来自不同batch，不能当固定状态收敛；step500 forward是更新前累计2999权重，与上述终点teacher分开。100/500取当批首个active微批，无active就记NONE，不换批。",""]
        logs=a.get("logs",a.get("diagnostics",[]))
        lines+=_table(["step","FM","L3 (gated/8)","L5 (gated/8)","D_edge (gated/8)","weighted geometry","total","active/8","ramp","实际联合/分项速度梯度"],
            [[x.get("step"),x.get("FM"),x.get("geometry_gated_means",{}).get("L3"),
              x.get("geometry_gated_means",{}).get("L5"),x.get("geometry_gated_means",{}).get("D_edge"),
              x.get("weighted_geometry"),x.get("total"),x.get("active_samples"),x.get("ramp"),x.get("velocity_gradient")] for x in logs])+[""]
        if a.get("timing"):lines+=_table(["timing字段","seconds"],_scalars(a["timing"]))+[""]
    lines += ["相同更新次数不代表相同算力。模型更新时间可能包含几何loss、反向及预登记诊断；CPU nested OT、保存、teacher、完整采样和离线审计分别计账，不把纯标量检查计入模型调用。",""]
    ledger=r.get("evaluation_ledger",{})
    lines+=_table(["独立审计字段","记录"],_scalars({k:v for k,v in ledger.items() if k!="input_audits"}))+[""]
    lines += ["每个正式输出独立CF1200对/FF4950对，C-only66对按六个固定上下文缓存。NaN/Inf、退化、共面重叠与UNRESOLVED单列，不以距离替代判交，不重评任何历史输出。精确FP32索引保留继承异常；焊接、水密修补：NOT_RUN。",""]
    return lines+_execution_details(r)

def render_report(r,figures=()):
    s=_validate(r)
    lines=["# L3_L5_INTERFACE_REPORT","",
        "这是五个固定终点、六个已见parent/patch单元、每单元4个新噪声的匹配探针；没有开发排名、选择后补位或历史分数拼接。只新增J35/JEdge5各500更新，不自动切换默认模型。",""]
    questions=("原L3/L5组合是否比单项更实用？","L5减FF的方向在新噪声上是否保留？",
        "D_edge替换L3是否真正改善接口？","改善有没有转化为覆盖/CF/FF/尾部的新代价？",
        "哪个结果值得保留，哪些只是局部信号？","有无必要再做一个独立问题，而不是继续扫loss？")
    answers=r.get("interpretation",{}).get("answers",[])
    for i,q in enumerate(questions):
        a=answers.get(str(i+1)) if isinstance(answers,dict) else answers[i] if i<len(answers) else None
        lines += [f"{i+1}. **{q}** "+(str(a) if a is not None else "待登记综合解释；不依据最低宏RMS自动宣布赢家。"),""]
    if r.get("interpretation",{}).get("next_step"):lines += [str(r["interpretation"]["next_step"]),""]
    lines+=_protocol(r)+_outcomes(s)+_factorial(s)+_binding(s)+_teacher(r)+_accounting(r)
    lines += ["## 固定首噪声、同视角原始full/free","",
        "固定注册首seed，三patch分别展示；Y向上，正交相机elev18/azim−60，每父对象跨五模型和三片共享坐标范围。只画原始输出，没有后处理或挑seed。首张图不代表整体或最差情况。",""]
    for f in figures:lines += [f"### patch{f['patch']} / seed{f['seed']}","",f"![raw full/free]({Path(f['path']).name})","",f"可视检查：{f.get('visual_review','PENDING_CALLER_REVIEW')}。",""]
    return "\n".join(lines+["完整逐输出41项标量及八配对/五四格效应逐seed表随results保存；额外噪声仍不等于新对象、新patch或独立训练seed泛化。完成后停止，不组合新赢家、不改lambda/核/门控、不延长训练。",""])

def _figures(r,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    seed=_seeds(r)[0]
    rows={(x["model"],int(x["parent"]),int(x["patch"])):x for x in r["generated"] if x["seed"]==seed}
    if len(rows)!=30:raise ValueError("Fixed first seed five-model matrix incomplete")
    arrays={}
    for key,row in rows.items():
        path=_path(row["raw_path"])
        if _sha(path)!=row["raw_sha256"]:raise ValueError("Plot raw SHA differs")
        with np.load(path,allow_pickle=False) as z:v=z["output"].copy()
        if v.dtype!=np.float32 or v.size!=112*9 or not np.isfinite(v).all():raise ValueError("Plot raw geometry invalid")
        arrays[key]=v.reshape(112,3,3)
    bounds={}
    for p in (0,1):
        points=np.concatenate([v.reshape(-1,3) for (m,parent,j),v in arrays.items() if parent==p])
        lo=points.min(0);hi=points.max(0);bounds[p]=((lo+hi)/2,max(float((hi-lo).max())*.55,1e-6))
    records=[]
    for j in (0,1,2):
        fig=plt.figure(figsize=(17,12))
        for p in (0,1):
            for free in (False,True):
                for col,m in enumerate(MODELS):
                    ax=fig.add_subplot(4,5,(p*2+int(free))*5+col+1,projection="3d");v=arrays[m,p,j]
                    ax.add_collection3d(Poly3DCollection(v[12:],facecolors="#7eabc7",edgecolors="#405266",linewidths=.24,alpha=.9))
                    if not free:ax.add_collection3d(Poly3DCollection(v[:12],facecolors="#f0a348",edgecolors="#804800",linewidths=.5))
                    center,radius=bounds[p]
                    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
                    ax.set_box_aspect((1,1,1));ax.set_proj_type("ortho");ax.view_init(elev=18,azim=-60,vertical_axis="y");ax.set_axis_off()
                    ax.set_title(f"P{p} | {m} | {'free100' if free else 'full112'}",fontsize=8)
        fig.suptitle(f"Fixed seed {seed} | patch {j} | raw, no repair | Y-up, elev18 azim-60\nIdentical per-parent bounds across models and patches",fontsize=11)
        fig.subplots_adjust(left=.01,right=.99,bottom=.005,top=.94,wspace=.01,hspace=.06)
        path=Path(out)/f"seed{seed}_patch{j}_full_free.png"
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=170,facecolor="white");plt.close(fig)
        records.append(dict(**_record(path),patch=j,seed=seed,panels=20,vertical_axis="y",elevation=18,azimuth=-60,visual_review="PENDING_CALLER_REVIEW"))
    return records

def write_report(r,out):
    started=time.perf_counter();_validate(r);out=_path(out);path=out/"L3_L5_INTERFACE_REPORT.md"
    if path.exists():raise FileExistsError(path)
    figures=_figures(r,out);text=render_report(r,figures)
    if "????" in text:raise ValueError("Corrupted UTF8 report")
    with path.open("x",encoding="utf-8") as f:f.write(text)
    r["report"]=dict(**_record(path),figures=figures,seconds=time.perf_counter()-started,
        model_calls=0,geometry_calls=0,OT_calls=0,autograd_calls=0)
    return r["report"]


def _execution_details(r):
    lines=["### 真实检查、身份与分账","",
        "原实验只读索引、共同起点和候选配方分开登记；L3/L5/J35/JEdge5均非自动默认。旧三臂仅复用权重，所有本轮120输出重新在同一批4新噪声上生成，旧开发/确认评分不混入。",""]
    lines+=_table(["字段","记录"],[
        ["experiment_index",r.get("experiment_index")],
        ["initial_state_sha256",r.get("protocol",{}).get("initial_state_sha256")],
        ["initial_stream_sha256",r.get("protocol",{}).get("training",{}).get("initial_stream_sha256")],
        ["historical500_stream_sha256",r.get("protocol",{}).get("training",{}).get("old500_batch_stream_sha256")],
        ["registered_seeds",r.get("protocol",{}).get("evaluation",{}).get("seeds")],
        ["seed_audit_status",r.get("seed_audit",{}).get("status")],
        ["protocol_sha256",r.get("protocol_sha256")]])+[""]
    lines+=_table(["旧臂","状态","本轮更新","旧终点累计","旧500批流SHA"],
        [[m,a.get("status"),a.get("new_optimizer_updates"),a.get("checkpoint",{}).get("cumulative_updates"),
          a.get("old500_batch_stream_sha256")] for m,a in r.get("reused",{}).items()])+[""]
    lines+=_table(["新臂","完成更新","fresh optimizer","起点stateSHA","起点streamSHA","500批streamSHA"],
        [[m,a.get("completed_updates"),a.get("fresh_optimizer"),a.get("initial_identity",{}).get("state_sha256"),
          a.get("initial_stream_sha256"),a.get("paired_stream_sha256")] for m,a in r.get("arms",{}).items()])+[""]
    strict=r.get("calibration",{}).get("strict_initial",{})
    lines+=_table(["共同初始化实际严格加载字段","记录"],[(k,strict.get(k)) for k in
        ("path","checkpoint_sha256","state_sha256","cumulative_updates","strict","missing_keys","unexpected_keys",
         "parameters","input_sha256","stream_seed","config_sha256","config_file_sha256","optimizer_restored")])+[""]
    lines+=_table(["Geo实际加载元数据","记录"],[(k,strict.get("context",{}).get(k)) for k in
        ("context_mode","branch_dtype","initialization_seed","additional_parameters")])+[""]
    timing_keys=("CPU_OT_and_stream_seconds","supervision_prepare_seconds","model_update_seconds","checkpoint_seconds","wall_seconds")
    lines+=_table(["新臂"]+list(timing_keys),[[m]+[a.get("timing",{}).get(k) for k in timing_keys] for m,a in r.get("arms",{}).items()])+[""]
    lines += ["相同500更新不等于同算力。model_update_seconds包括模型前向、实际几何loss、预登记叶子梯度诊断、反向、裁剪与更新；CPU_OT_and_stream_seconds、监督准备、checkpoint保存另列，wall还含日志与其他开销。不要把该时间作为纯主干吞吐。",""]
    lines+=_table(["CPU检查记录","状态","checks/tests","坐标autograd","记录SHA"],
        [[name,rec.get("result",{}).get("status"),rec.get("result",{}).get("checks",rec.get("result",{}).get("tests")),
          rec.get("result",{}).get("counts",{}).get("CPU_coordinate_autograd_calls",rec.get("result",{}).get("counts",{}).get("autograd_calls",rec.get("result",{}).get("autograd_calls"))),
          rec.get("artifact",{}).get("sha256")] for name,rec in r.get("CPU_fixtures",{}).items()])+[""]
    lines += ["这些CPU检查和已保存velocity上的叶子梯度不新增模型前向；wiring有真实前向/反向但0更新。终点teacher模型调用另计，不能混入训练更新。",""]
    rows=[]
    for stage,group in r.get("loss_accounting",{}).items():
        for name,rec in group.items():
            counts=rec.get("counters",{})
            rows.append([stage,name,rec.get("calls_attempted"),rec.get("calls_returned"),
                counts.get("short_predicted_edges",0),counts.get("zero_predicted_edges",0),
                counts.get("kernel_matrix_evaluations",0),counts.get("GTGT_kernel_cache_hits",0)])
    lines+=_table(["阶段","loss","attempts","returns","短预测边累计触发","零预测边累计触发","kernel矩阵","GTGT缓存命中"],rows)+[""]
    lines+=_table(["进程阶段","分布loss工作快照"],list(r.get("stage_distribution_counters",{}).items()))+[""]
    lines += ["loss_accounting为各调用的累积工作量；进程快照是交叉核账，不能把累计快照重复相加。GT-GT可缓存，可学习预测位置/面积/法向不detach。",""]
    for m,a in r.get("arms",{}).items():
        logs=a.get("logs",[])
        lines += [f"### {m} 梯度/更新与实际运行配置",""]
        lines+=_table(["step","参数组","梯度L2","梯度有限","更新L2","更新有限"],
            [[x.get("step"),g,v.get("L2"),v.get("finite"),x.get("parameter_updates",{}).get(g,{}).get("L2"),
              x.get("parameter_updates",{}).get(g,{}).get("finite")] for x in logs for g,v in x.get("gradients",{}).items()])+[""]
        if logs:
            runtime=logs[-1].get("runtime",{})
            keys=("cublas_workspace_config","deterministic_algorithms","deterministic_warn_only",
                "tf32_matmul","tf32_cudnn","sdpa_math_enabled","sdpa_flash_enabled","sdpa_mem_efficient_enabled","cuda_autocast_enabled","cuda_autocast_dtype","parameter_dtype","gradient_enabled",
                "native_precision","all_modules_eval","trainable_parameters")
            lines+=_table(["实际入口字段","记录"],[(k,runtime.get(k)) for k in keys])+[""]
    lines += ["### 有限线段目标的已知退化限制","",
        "中点对称投影保证端点交换对称及零长度边的有限值。已登记CPU fixture中，预测边恰缩在GT边中点时D_edge=0.046875但坐标梯度为0。正loss检测到未覆盖长度，却不保证该对称退化状态能靠D_edge自身梯度展开；离轴塌缩例梯度非零是不同情况。此限制原样保留，没有为消除它改公式、增模块或推理修补。","",
        "有限epsilon处为预登记的稳定近似；极短预测边保留并计数，GT异常边停止定位。D_edge即使很小也只表示几何接近，不产生共享顶点ID或水密证明。",""]
    return lines
