"""Deterministic report tables from finished CSVs; no experimental actions."""
import sys,csv,json,statistics,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'r2/src')]
from r2.pilots.bench_loop_motif_mini_v1 import run as X
from r2.pilots.bench_loop_motif_mini_v1.analyze import savecsv
O=X.OUT
def csvread(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def main():
    rows=csvread(O/'analysis/per_output_normalized177.csv');mac=csvread(O/'analysis/metric_parent_macro_all177.csv');vis=csvread(O/'analysis/visual_summary.csv')
    means={(r['stage'],r['method'],r['metric']):float(r['parent_macro']) if r['parent_macro'] else None for r in mac}
    J={(r['stage'],r['method']):int(r['J']) for r in vis}
    fields=X.E._fields();parent=[]
    for method in ('R','G0','G1','G3','L0','L3'):
        for stage in X.E.STAGES:
            for pid in sorted({r['parent_id'] for r in rows}):
                group=[r for r in rows if r['method']==method and r['stage']==stage and r['parent_id']==pid]
                row=dict(method=method,stage=stage,parent_id=pid,outputs=len(group),I_status=group[0]['I_status'])
                for f in fields:
                    try:row[f]=statistics.mean(float(r[f]) for r in group if r[f]!='')
                    except (ValueError,statistics.StatisticsError):row[f]=None
                parent.append(row)
    savecsv('per_parent_all177.csv',parent)
    metrics={'CF':'CF_illegal_affected_free_fraction','FF':'FF_illegal_affected_free_fraction','I':'ConflictCov_C','小角':'free_minimum_angle_lt5_fraction','有效free面':'actual_free_faces','自由面积':'free_total_surface_area','单面岛':'hinge_diag_isolated_1face_edge_components','合法接口请求':'C_F_has_valid_edge'}
    def table(stage):
        s=['| 方法 | J /8 | CF % | FF % | I % | 小角 % | 有效free面 | 自由面积 | 单面岛 | 有接口 /8 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for m in ('R','G0','G1','G3','L0','L3'):
            vals=[means[stage,m,metrics[k]] for k in metrics]
            s.append('| '+m+' | '+str(J.get((stage,m),'NA'))+' | '+' | '.join(f'{v*100:.3f}' if i<4 else f'{v*8:.0f}' if i==7 else f'{v:.3f}' for i,v in enumerate(vals))+' |')
        return '\n'.join(s)
    deltas=[];tails=[]
    for a,b in [('L0','G0'),('L3','G3'),('L3','L0'),('L3','G1'),('L3','R')]:
        for stage in X.E.STAGES:
            r=dict(comparison=a+'-'+b,stage=stage)
            for k in ('CF','FF','I'):
                f=metrics[k];r[k+'_delta_pp']=100*(means[stage,a,f]-means[stage,b,f]);limit={'CF':5,'FF':10,'I':10}[k]
                for rid in sorted({z['request_id'] for z in rows}):
                    ra=next(z for z in rows if z['method']==a and z['stage']==stage and z['request_id']==rid);rb=next(z for z in rows if z['method']==b and z['stage']==stage and z['request_id']==rid)
                    if k=='I' and ra['I_status']!='PASS':continue
                    d=100*(float(ra[f])-float(rb[f]));tails.append(dict(comparison=a+'-'+b,stage=stage,request_id=rid,metric=k,delta_pp=d,limit_pp=limit,exceeds_severe_tail=d>limit))
                r[k+'_mean_guard_pass']=r[k+'_delta_pp']<=2
            deltas.append(r)
    savecsv('quality_mean_guard.csv',deltas);savecsv('quality_request_tail_guard.csv',tails)
    dtab=['| 比较（FINAL） | ΔCF pp | ΔFF pp | ΔI pp |','|---|---:|---:|---:|']
    for r in deltas:
        if r['stage']=='CLEAN_FINAL_POST':dtab.append(f"| {r['comparison']} | {r['CF_delta_pp']:+.3f} | {r['FF_delta_pp']:+.3f} | {r['I_delta_pp']:+.3f} |")
    tlines=[]
    for r in tails:
        if r['stage']=='CLEAN_FINAL_POST' and r['comparison'] in ('L0-G0','L3-G3','L3-L0') and r['exceeds_severe_tail']:
            rid=r['request_id'];tlines.append(f"- {r['comparison']}，{rid[:4]}/{rid.rsplit('__',1)[-1]}：{r['metric']} {r['delta_pp']:+.3f}pp，超过预登记尾部门限 {r['limit_pp']}pp。")
    diagnostics=csvread(O/'analysis/guidance_per_output.csv');dt=['| 方法 | shadow RMS %L | 限幅步 | 累计 S RMS均值 | 累计 F RMS均值 |','|---|---:|---:|---:|---:|']
    for m in ('G0','G3','L0','L3'):
        rs=[r for r in diagnostics if r['method']==m]
        dt.append(f"| {m} | {statistics.mean(float(r['shadow_rms_percent_L']) for r in rs):.5f} | {sum(int(r['clipped_steps']) for r in rs)} | {statistics.mean(float(r['cumulative_correction_S_rms_sum']) for r in rs):.6f} | {statistics.mean(float(r['cumulative_correction_F_rms_sum']) for r in rs):.6f} |")
    phases=[X.read(p) for p in O.glob('workflow/attempts/*/completion.json')]
    pt=['| 阶段/worker | 秒 | 整机显存峰值 MiB | 最小空闲 MiB |','|---|---:|---:|---:|']
    for r in phases:pt.append(f"| {r['stage']}/{r['worker']} | {r['seconds']:.2f} | {r['gpu_peak_used_mib']:.0f} | {r['gpu_min_free_mib']:.0f} |")
    report='''# R2_BENCH_LOOP_MOTIF_MINI_V1 实际实验报告

2026-10-29。已完成固定两臂、8请求、零训练实验。**本版冻结loop没有让motif guidance获得新的条件完成成功，不晋级。** L0与L3在RAW和FINAL均为0/8联合正例；同面板G1为1/8，EXPOSURE为RAW8/8、FINAL4/8。没有通过预登记的“相对L0净增至少2例、覆盖至少2父”的开发信号门槛。

仍保留有限数值收益：L3相对G3的FINAL FF下降1.835pp、小角率下降7.289pp；但I上升8.162pp、CF上升1.297pp，且没有采用收益。L0相对G0的FINAL FF反而上升2.049pp。不能把这些混合结果讲成loop改善了整体质量，也不能把无训练loop的负结果推广到所有经过专门训练的循环模型。

## 1. 实验问题和实现

检验的是改变冻结官方Bench模型的内部函数，能否让同一引导更有用；不是已经确认“思考不足”。官方Bench EMA及D0沿用上一轮，生成器不是EXPOSURE，不添加Native/Geo/SEAM头。G0/G1/G3为同一官方权重机制对照，R=EXPOSURE是不同训练历史的实用流程参照。

零基层4、5、6、7各自执行两次整层半残差，顺序为0,1,2,3,4,4,5,5,6,6,7,7,8,9,10,11。每次 `h <- h + (L_j(h,c,mask)-h)/2`；完整执行attention和MLP后再缩放净残差，不是分别缩放子块。两次使用同一时间/count/mask，不detach，CFG两个分支一致。它改变模型函数和输入导数，冻结参数不意味着保留原采样分布。

- L0：loop＋G0，完整状态生成后覆盖C。
- L3：loop＋G3，全状态输入梯度，lambda=0.3、角点最大修正0.005L，保持原引导时序。
- r=1直接旁路到原层；在原bcb/n0、state025上，真实输出与全输入VJP均逐位相等。独立CPU算子检查亦通过。

每份Euler50，BF16前向、FP32参数/积分，TF32关闭，确定性算法严格、SDPA math。后端统一为单次官方D0→GEO_C/来源角序→HINGE_OPP015→ALPHA→HOLE。最终C精确保留全部通过，没有训练、额外噪声、阈值/层段扫描或中途取消L3。

## 2. 固定输入、评价范围

复用旧校准面板：6b(K4/N133)、a8(K39/N189)、28b9(K10/N384)、bcb(K2/N520)，各两份原噪声；只新增L0/L3共16份。该面板已多次用于开发，不是新父泛化或独立确认集。

原177评分字段及检测器不变：48行新增RAW/BASE/FINAL＋96行缓存=144行。CF/FF为受影响free面比例，父等权4父8输出；I保留原2048查询及权重分母，仅原PASS的3父6输出进入主均值，28b9 LOW_SUPPORT单列。a8仅69/2048内部支持点的解释限制继续保留。不改变域来争取通过。RAW triangle soup的索引孤岛不能直接当成可见碎片。

联合J要求同一输出：主体REASONABLE、C采用PLAUSIBLE、支撑PRESENT。32个新增图审单元逐张查看两个固定方位和原位置前/侧截面；是非匿名AI_PROXY，不是独立人类认证。旧64个参照评级原样复用。6b/n0采用UNCERTAIN未算成功；即使宽松计入，也只涉及一个父且L0/L3同时获益，无法满足交互收益门槛。

## 3. 完整主表

RAW（百分比越低越好；J及合法接口只是各自维度）：

'''+table('RAW')+'''

FINAL：

'''+table('CLEAN_FINAL_POST')+'''

R的极低FINAL冲突伴随大量删面及J从8到4，不能把它理解成无代价的优质成品。L0/L3的RAW FF均超过90%，原G0/G3也在92–93%，故高绝对值不是loop新制造的全部代价；跨阶段必须分别读。

## 4. 配对收益、损失和交互

'''+ '\n'.join(dtab)+'''

L0-G0、L3-G3、L3-L0在RAW/FINAL的联合J净变化都为0；描述性交互 `(L3-L0)-(G3-G0)` 的J亦为0，没有guidance独有的新成功。L3相比G1各阶段失去1例，相比R分别失去8个RAW、4个FINAL原联合正例。逐请求转移并未合并噪声掩盖。

FINAL严重尾部：

'''+ ('\n'.join(tlines) if tlines else '无超过预登记门限的尾部。')+'''

所有逐输出、逐父、全177字段差值、交互差值及均值/尾部门限均在analysis中。I均值代价已足以使L3相对G3不通过质量升级筛查，即使不考虑采用无收益。

## 5. 实际图审看到什么

- 6b/n0两臂都有相对完整外形，但C与灰座的轮廓/层差仍不明确；6b/n1蓝C与下方座框分离，大缺口保留。
- a8/n0蓝条板高于灰色折层主体；a8/n1有完整支撑但灰座覆盖蓝条板并有抬起折片。L3未改变用途关系。
- 28b9两噪声的C均高于另一层灰座；n1经过后端后灰主体/支撑大量消失，只剩细框残片。
- bcb/n0蓝C穿在灰条板内部，支撑间有薄片；n1蓝C悬于靠背与空座框之间。两臂没有补出符合C的完整座面。

这些判断来自图与截面，不由shadow误差、低CF或低I反推。看图没有获得新的联合正例，不能将几何数值局部下降解释为条件控制升级。

## 6. 引导是否真的改变了计算

'''+ '\n'.join(dt)+'''

L3的360/360引导步均触发限幅，且F修正非零。shadow误差未下降。由保存的前态、后态及终点预测还原实际修正，最大范数核对误差仅1.67e-7模型单位；与G3比较，逐步S/F方向余弦均值约0.00019/-0.00107。**这是各自已经分离轨迹上的修正比较，不是同状态Jacobian因果归因。** 它只说明实际修正方向发生变化，不能证明变得有用；任务评价没有兑现收益。

## 7. 执行、资源与工程记录

实际完成800采样向量场、360采样VJP；r1检查另2向量场＋2VJP；16单次D0；零训练更新、零新噪声、零模型重试。每采样向量场16层调用，共12800层调用，门禁另24；CFG双分支对应1600采样网络样本等价，不能称为原模型800调用的等算力。

RTX5080总16303MiB，WSL24核、约64GiB RAM。初始用户显存占用较高，以第一份已授权N520 L3测量，保留至少2GiB余量；第一阶段整机峰值11397MiB、最低空闲4583MiB。规划3800MiB为包含上下文/非分配器余量的估计，不冒充WDDM单进程实测。后续用户占用下降，但已有任务队列不拆分抢写。D0首批完成测量时8份已结束，再启动第二批；实际GPU并发1，CPU评分/渲染与GPU及彼此并行，进程均限1线程。没有停止用户进程。

'''+ '\n'.join(pt)+'''

表中是整机显存，CPU阶段非零GPU占用来自其他并行进程，不是CPU评分调用了模型。时间包含加载/哈希/并行争用，不能作为隔离算力基准；L3采样field/VJP累计23.94/18.92秒，而旧G3为18.49/15.01秒，仅描述真实运行。最慢CPU几何链为bcb/n1，L0/L3分别194.16/231.30秒，未跳过或改后端。

一次CPU绘图标签映射失败已修复：旧标签G3_L0p3显示为G3，原cell_id和lambda保留；修复前无图像产物，没有任何模型重算。旧代码/资产/评分只读，独立分析脚本及实际源码在archive，差异和hash附全。

## 8. 决定与交付

关闭本次固定loop+motif配置，不训练循环修正器、不蒸馏G3、不据此启动新扫描。部分FF/小角下降作为有限观察保留，但未改善采用且I代价显著；EXPOSURE现状参照与默认模型均不变。

入口：`RECOVERY.md`。完整原字段：`analysis/per_output_original177.csv`；规范方法名/固定I域数字表：`per_output_normalized177.csv`；逐父：`per_parent_all177.csv`；全部配对：`paired_deltas_all177.csv`；真实图审：`actual_visual_new32.csv`及`actual_visual_all96.csv`；方向/限幅：`guidance_per_step.csv`、`applied_direction_own_trajectory.csv`；固定截面：`gallery/index.html`。本地保留51状态、连续proposal、全部RAW/BASE/FINAL与评分；小包带实际代码、hash、diff和逐例CSV。完成后停止，无commit/push/默认切换。
'''
    (O/'REPORT.md').write_text(report,encoding='utf-8')
    print('\n'.join(dtab));print('\n'.join(tlines));print('REPORT_WRITTEN')
if __name__=='__main__':main()
