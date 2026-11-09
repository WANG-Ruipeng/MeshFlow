"""Write the closeout from completed local evidence, never launch work."""
from __future__ import annotations
import collections, json
import numpy as np
from . import common as C
from .analysis import rows


def table(headers,data):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x).replace('|',r'\|') for x in row)+' |' for row in data])


def run():
    a=C.read(C.OUT/'analysis/COMPLETE.json');d=C.read(C.OUT/'decision.json')
    vis=C.read(C.OUT/'visual_review/COMPLETE.json')
    assert d['status']!='PENDING' and vis['status']=='COMPLETE'
    summary=rows(C.OUT/'analysis/per_request.csv')
    scores=C.read(C.OUT/'scores/metric_rows.json')
    construction=C.read(C.OUT/'targets/COMPLETE.json')
    ss=[]
    for r in summary:
        ss.append([r['parent_id'][:6]+'/n'+r['noise_index'],r['seeds'],r['accepted_patches']+'/'+r['patches'],
                   r['EDIT_corners'],r['KEEP_corners'],r['post_surviving_patches'],r['FINAL_additional_repair_patches'],r['FINAL_lost_original_legal_edges'],r['numerical_local_pair_ready']])
    fields=['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C',
            'actual_free_faces','C_F_valid_edge_count','hinge_diag_isolated_1face_edge_components']
    metric=[]
    for parent in [None]+list(dict.fromkeys(r['parent_id'] for r in summary)):
        for label in ('RAW_IDENTITY','LOCAL_REPAIR'):
            rs=[r for r in scores if r['label']==label and r['stage']=='CLEAN_FINAL_POST' and (parent is None or r['parent_id']==parent)]
            assert len(rs)==(12 if parent is None else 2)
            vals=[float(np.mean([r[f] for r in rs])) for f in fields]
            metric.append(['全组6父' if parent is None else parent[:6],label]+[f'{v*(100 if i<3 else 1):.4f}' for i,v in enumerate(vals)])
    original=C.read(C.OUT/'data/assets_COMPLETE.json')['conflict_counts']
    attrs=[[key,count] for key,count in original.items()]
    rj=[[reason,count] for reason,count in sorted(a['rejection_reasons'].items())]
    resource=[]
    for arm in ('identity','target'):
        x=C.read(C.OUT/f'scores/{arm}/COMPLETE.json')
        resource.append([arm,x['workers'],f"{x['seconds']:.2f}",f"{x['worker_seconds_sum']:.2f}",f"{x['peak_worker_RSS_KiB']/1024:.2f}"])
    report=f'''# Bench：真实 RAW 局部修复目标试点

实验：`R2_BENCH_RAW_LOCAL_REPAIR_V1`；2026-10-24。

**{d['headline']}**

{d['explanation']}

这次实际制作了固定12份候选目标、逐角点状态和全部失败提案，完成原GEO及完整后处理配对评分和真实图审；没有把脚本交回后停止。数据阶段模型调用、训练、生成、D0前向均为0。这里的12请求按已知现象选出，只是开发试点，不估计一般成功率。

## 1. 实际检验的问题

RELDEF500相对SHAPE500没有新增严格视觉正例，并存在DEV FF代价。本轮结束该loss配置，转而检验：能否从真实生成的RAW中，制作有明确局部几何依据、在现有同槽坐标输出和后处理合同下仍有效的修复监督。

六父为a8、27、838c、91e、fd355、2d，每父固定n0/n1，不换样本、不重抽噪声、不借来源完整物体规定唯一正确款式。已有RAW、官方/SHAPE/RELDEF提议与BASE/FINAL均只读。48份BASE的顶点、槽位与角序对应核验通过，481个原缓存由hash保护。

最接近历史R05有界锚定、R27有界协调恢复及SHAPE/RELDEF保形。本次不同之处是建立真实RAW局部监督对和UNKNOWN掩码；没有扩大旧焊接半径，也没有将某个低FF后端自动当真值。

## 2. 冻结的唯一修复规则

在所有12请求上使用同一规则，从原有限三角面检测确认的非法C/F面对出发，仅尝试座面上表面附近的侧壁顶沿小幅穿刺：C局部法向朝+Y且Y分量≥0.7；自由面近直立，法向绝对点积≤0.5；有稳定下侧角点及1—2个仅小幅穿出的顶角。

采用**原GEO_ONLY真实角点簇**，把同簇的所有free角点赋到有限C表面最近点；不自造半径簇，不只拉一个角点。每个成员相对原RAW的位移≤0.005L，L=9.2081382646。所有簇成员、指定要消除的CF面对、受影响面和拒绝理由逐项保存。C、N、槽位、原RAW角序固定。

每个patch只尝试一次。要求指定非法关系真正消除、没有新增CF/FF或UNKNOWN、原合法边不丢、面积比处于[0.8,1.25]、法向变化≤30°，不得移动KEEP。失败回滚并保留提案，不扫描更大半径或换一种算法挑最好结果。+Y规则仅适用于这12份已审座面输入，不宣称是通用修复器。

这是一种有限的关系投影，不解决缺腿、重复主体、任意错层、删面或增面。局部目标仍可包含UNKNOWN区域；它不是完整无缺陷GT。

完整合同：[PROTOCOL.md](PROTOCOL.md)、[strategy.json](targets/strategy.json)。

## 3. 先补齐了同处理链对照

原GEO_ONLY只做到RAW→GEO/来源定向，没有后续HINGE/ALPHA/HOLE。若直接比较新目标与官方D0 FINAL，就会把省去D0本身的收益算给本轮编辑。

因此预先登记了必要CPU对照：RAW_IDENTITY与LOCAL_REPAIR使用同一GEO和HINGE_OPP015→ALPHA→HOLE链；RAW_IDENTITY复用原GEO缓存，不重新生成。其BASE的12×177=2,124个字段复算与原缓存精确相等。两臂共48行原始评分，24条同阶段逐输出配对，完整保留177字段、分母及NA。

**主要编辑增量是LOCAL_REPAIR−RAW_IDENTITY。** 官方、SHAPE、RELDEF仅作历史参照。目标不再送D0一次。

见[CPU_CONTROL_ADDENDUM.md](CPU_CONTROL_ADDENDUM.md)、[完整逐输出指标](scores/per_output.csv)、[配对差值](scores/paired_per_output.csv)。

## 4. 冲突归属复算

SHAPE→RELDEF的FF新增91对中，62对发生在共同保留的原RAW槽位之间，29对涉及新增保留的面；另有54对共同保留面的旧冲突消失。不能把全部代价归因于“多保住了面”。面被删掉后的面对不再可观察，也不当作几何修复。

{table(['类别（原RAW面身份）','面对数'],attrs)}

这张表描述面身份和冲突的转移，不单独证明形成冲突的动力学原因。逐面对记录：[conflict_attribution.csv](data/conflict_attribution.csv)。

## 5. 实际目标与后处理存活

构造得到{a['accepted_patches']}个通过局部检查的patch、{a['rejected_patches']}个拒绝patch，共{a['actual_EDIT_corners']}个EDIT角点，覆盖{a['requests_with_EDIT']}/12请求。下面的“额外FINAL修复”只计原RAW_IDENTITY FINAL仍有指定冲突、而目标FINAL保留相关原面并消除冲突的patch；原GEO已经解决的关系不记为新增收益。

{table(['请求','CF候选种子','接受/尝试patch','EDIT角点','自动KEEP角点','经BASE/FINAL存活patch','额外FINAL修复patch','对照FINAL连接未保留','数值局部资格'],ss)}

56个接受patch的指定关系在真实BASE/FINAL中均保持消除，相关原面均保留。但**4/12满足无新增冲突、无对照合法边代价的数值资格**（a8与838c各两噪声），它们的指定关系原GEO也能解决。真正多修复了FINAL指定关系的是91e两噪声（1与16个patch）；它们没有新增非法CF/FF，但相对对照FINAL少了1与4条合法F/F共边。

独立索引核查确认：这5条边在RAW、IDENTITY_BASE、TARGET_BASE都没有合法共边，相关面始终保留。**这是没有获得identity后续处理本来会新增的连接，不是撕裂RAW/GEO既有共边，也不是通过删面降低冲突。** 所以局部修复有几何正证据，却还不满足完整链下无连接代价的保守准入。见[实际共边配对账本](analysis/paired_original_legal_edges.csv)和[独立拓扑审计](audits/independent_topology_audit.json)。

拒绝理由允许一patch同时触发多个条件，所以各行之和不是拒绝patch数：

{table(['拒绝理由','出现次数'],rj)}

原自动KEEP规则要求已审a8/27的完整下侧支撑面同时满足局部位置和无已检冲突条件。若自动KEEP为空，代表此规则没有认证出KEEP，不代表RAW没有正确支撑；未认证角点仍是UNKNOWN，未偷偷作为identity监督。后续人工标注如有，保留为独立版本，不覆盖原mask或伪造通过门槛。

实际补充图审在a8/27四份输出中标注了**12个支撑下端KEEP角点**（每份a8四点、每份27两点），保存完整RAW slot/corner/坐标与8张已查看图的hash。只认证局部支撑端点位置，不认证整张面的自交或整物质量。它们另存于[manual_KEEP_annotations.json](visual_review/manual_KEEP_annotations.json)，没有修改冻结target的UNKNOWN状态和loss_mask；也不把KEEP-only的27变成EDIT正例。

数据：[逐请求](analysis/per_request.csv)、[逐patch存活](analysis/patch_post_survival.csv)、[KEEP/EDIT面存活](analysis/KEEP_EDIT_slot_survival.csv)、[原面对净进出](analysis/paired_illegal_original_faces.csv)。每个`targets/<request>/target.npz`含真实候选坐标、原RAW、KEEP/EDIT/UNKNOWN/C状态、loss_mask及角点身份；`proposals/`保留拒绝候选。

## 6. 同链FINAL质量

CF/FF为受非法关系影响的free面比例；I沿用原ConflictCov_C支持域/查询定义。面数、非法面对数和完整原字段仍在CSV中，不能只根据百分比判断改善。每父两噪声，表中父均值；全组为六父等权。

{table(['父','同链方法','CF %','FF %','I %','有效free面','合法C/F边','单面孤岛'],metric)}

RAW级修复与GEO后的增量分开记录。某个指定关系虽然在原始坐标上被消除，但若原GEO已经会消除它，就不能宣称本轮提升了最终质量。被处理链删去的指定面也不算成功。

全组FINAL的CF为3.4299%→2.3159%，FF为24.5781%→24.4546%，I为1.5340%→1.5055%；有效free面数均为149.8333/输出。按原面身份，FINAL消除28对CF与1对FF，0新增非法原面对，改善不是多删面得到的。收益集中在91e，不能据全组均值声称六父普遍提升；它与上面的5条最终连接机会损失需要同时报告。

## 7. 实际图审与训练决定

图册使用同相机、原截面，并补两侧及支撑处截面；包含完整目标、状态叠加、接受与拒绝patch局部放大，以及RAW_IDENTITY FINAL对LOCAL_REPAIR FINAL的直接对照。真实评阅标为非盲AI_PROXY，不能当作独立人类评价。

{d['visual_summary']}

冻结的数据准入为至少7/12有真实可信EDIT并通过原后处理及结构保护；KEEP-only、UNKNOWN占位或因为省掉D0而保住结构均不计为编辑正例。非空EDIT上限已只有**6/12**，数值资格进一步为**4/12**；图审无法把其余未修复样本变成合格训练对。决定：**{d['status']}**。这不是按结果追加了一个门槛：即使将全部6份候选都算合格，仍未达到事先冻结的7份。

{d['training_decision']}

本次12请求涉及的6个开发父不能一边进入训练一边宣称后端未见。只读盘点还确认，旧SHAPE16个后端训练父没有现成同一Coverage模型的RAW缓存；其他历史模型的18份RAW不具有相同生成分布身份，不能直接混作本轮训练数据。若将来准备新训练池，需要独立冻结父、RAW与修复标签。

这次门槛的结果仅约束这项具体修复规则与这版数据；不证明所有真实RAW局部修复都不可行，不证明D0存在理论上限，也不把所有残余错误统一判成必须删面或必须重新生成。

## 8. 实际执行、工程记录与恢复

构造CPU时间约{construction['seconds']:.2f}秒，原CF/FF检测器累计调用数按账本为{construction['detector_pair_scan_calls']}（一个全网格扫描含CF与FF两项）。评分资源：

{table(['阶段','CPU workers','墙钟秒','worker阶段秒之和','峰值RSS MiB'],resource)}

并行时间不能和单进程时间当作相同计算基准。初始资源为RTX5080、16,303MiB显存、24逻辑CPU、约61GiB可用RAM；GPU在数据阶段没有任务进程。

构造初次在预检发现RAW模型角序与导出来源角序不同，**在0候选、0检测器、0模型调用处停止**。原失败和源码保留；用已保存排列的逆映射修复索引账本后明确恢复，12份逐位往返验证通过，没有重放科学计算。target的source_corner_ids是canonical identity，旧导出排列另存input_export_source_corner_ids；原GEO映射不变。

另有91e/n1的patch033使用无GEO映射的−1标识，被构造为一个无效提案；它在位移/面积/法向预检即遭拒，未执行该提案的检测器或解码，也未贡献目标坐标。独立补充将其明确列为UNMAPPABLE工程失败：可解释的真实簇候选为78个（56接受、22拒绝），另有1个无映射失败；没有偷偷删掉日志或把它当物理不可修的证据。

渲染时新增支撑截面触发旧显示函数仅接受三个固定索引的限制。保留失败账本和已生成页面后，仅修正显示索引的调用，真实截面位置不变，随后完成图册；没有重算目标、评分或模型。

本阶段总计：12新目标GEO、12复用identity GEO、24完整cascade、48完整评分；0训练、0新权重、0生成、0D0、0新噪声、0GPU调用。原资产hash保持；无清理、默认切换、commit或push。

可恢复总入口（已完成阶段只做hash验证，不静默覆盖半成品）：

```powershell
wsl.exe -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.bench_raw_local_repair_v1.run verify
```

数据/目标/原链评分/渲染/分析分别可用`assets`、`targets`、`ready`、`evaluate`、`render`、`analysis`子命令。实际图审和训练准入不会由总入口自动伪造。源码与原评分代码归档、code.diff、源hash及独立审计随本目录交付。

## 9. 下一步的证据边界

{d['next_priority']}

当前任务在上述数据与评价交付后收口。未自动延长RELDEF、调loss权重、扩大吸附阈值或切换默认后端。
'''
    p=C.OUT/'REPORT.md'
    with p.open('x',encoding='utf-8') as f:f.write(report)
    print('REPORT_WRITTEN',p)

if __name__=='__main__':run()
