"""Read completed CPU exports and write a report; no scientific replay."""
from pathlib import Path
import statistics
from . import common as C


def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|',
                      *('|'+'|'.join(map(str,r))+'|' for r in rows)])


def main():
    done=C.read(C.OUT/'execution_completion.json')
    assert done['status']=='PASS' and done['scientific_stage_stopped']
    independent=C.read(C.OUT/'independent_audit/completion.json')
    assert independent['status']=='PASS' and independent['outputs']==36
    totals=C.rows(C.OUT/'topology_summary.csv')
    pairs=C.rows(C.OUT/'paired_per_parent_noise.csv')
    ori=C.rows(C.OUT/'orientation_per_output.csv')
    quality=C.rows(C.OUT/'quality_inherited_per_output.csv')
    vm=C.rows(C.OUT/'AVMR_PMR_per_output.csv')
    methods=('D0_BASE','RBF','BLEND')
    stages=('MODEL_POST','RESTORED_C_ONLY','ORIENTED')
    direction=[]
    for method in methods:
        for stage in stages:
            r=next(r for r in totals if r['group']=='ALL' and r['method']==method and r['stage']==stage)
            direction.append([method,stage,r['valid_edge_total'],r['opposed_edge_total'],r['outputs_with_opposed_edges']+'/12',
                              f"{100*float(r['opposite_winding_reached_free_area_fraction_mean']):.4f}%",
                              f"{100*float(r['reached_free_area_fraction_mean']):.4f}%"])
    examples=[]
    for r in pairs:
        if int(r['valid_edges_unchanged']):
            examples.append([r['method'],r['parent_id'][:4]+' n'+r['candidate'],r['valid_edges_unchanged'],
                             ' → '.join(r['opposed_'+s] for s in stages),
                             ' → '.join(f"{100*float(r['opposed_reach_'+s]):.3f}%" for s in stages)])
    inherited=[]; summary_rows=[]
    for method in methods:
        q=[r for r in quality if r['method']==method and r['stage']=='MODEL_POST']
        v=[r for r in vm if r['method']==method and r['stage']=='MODEL_POST' and r['setting']=='raw_tau_1e-6']
        cols=['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction']
        means={k:statistics.mean(float(r[k]) for r in q) for k in cols}
        means['AVMR_strict_model_1e-6']=statistics.mean(float(r['VMR']) for r in v)
        means['PMR_strict_model_1e-6']=statistics.mean(float(r['perfect']) for r in v)
        summary_rows.append(dict(method=method,n=12,value_status='INHERITED_AFTER_GEOMETRY_EQUIVALENCE',**means))
        inherited.append([method,*[f'{100*x:.3f}%' for x in means.values()]])
    C.csvwrite(C.OUT/'review/quality_inherited_summary.csv',summary_rows)
    conflicts=[]
    for method in methods:
        sub=[r for r in ori if r['method']==method]
        conflicts.append([method,
            str(sum(r['fixed_C_constraints_satisfiable_on_eligible_graph']=='True' for r in sub))+'/12',
            sum(int(r['fixed_C_anchor_conflict_components']) for r in sub),
            sum(int(r['parity_conflict_components']) for r in sub),
            sum(int(r['nonmanifold_edges']) for r in sub),
            str(sum(int(r['nonmanifold_edges'])>0 for r in sub))+'/12',
            sum(int(r['flipped_free_faces']) for r in sub)])
    report=f'''# 来源C角序恢复与free定向导出实验

版本：{C.VERSION}；日期：2026-10-11；状态：实际执行完成，科学阶段已停止。

**附件提出的表示问题得到验证和修正。恢复来源C角序、再固定C整理free后，RBF原有9条合法C/F共边全部获得一致方向，三份输出的方向可达面积恢复到各自无方向上限。没有新增几何连接，也没有改善座面错层或相交代价。** D0仍为比较基准，RBF保留为局部协调候选，BLEND不替代；没有修改正式默认流程。

## 1. 本轮实际做了什么

固定上一轮36份POST：同6条USER条件×2噪声×D0_BASE/RBF/BLEND。所有方法使用同一规则；不选图、不改面板。按每个条件保存的来源角点排列取逆，恢复来源C，再运行一次原固定C定向求解器。保存三阶段：

|阶段|含义|
|---|---|
|MODEL_POST|只读原规范化C的缓存POST|
|RESTORED_C_ONLY|仅用已保存来源映射恢复C，free原样|
|ORIENTED|固定恢复后的C，仅对可解分量重排free角点|

模型加载、前向、训练更新、新采样、新噪声、GEO_C、完整质量评分、新图均为0。实际36次来源恢复、36次方向求解；工程合成检查8项。72份新NPZ及72份配套PLY已落盘，原POST保留。

## 2. 新的C导出合同

保存映射满足 `canonical_corner[j] = original_corner[P[j]]`，因此恢复使用 `P_inverse = argsort(P)`。循环置换与反向置换分开登记，不用生成结果或得分决定C朝向。

这里恢复的是**原来源顶点ID及角序在现有模型坐标系中的表达**，没有反归一化坐标、没有加载原几何替换现有POST。原顶点坐标字节、面行身份、每张面的无序顶点ID集合、共享顶点/边关系及面数均不变。恢复后的C有序数组精确等于来源映射对应的模型坐标；它与旧规范化C有序数组不同，所以另设导出版本。模型输入及旧“规范化C逐位保持”结果未修改。

六条唯一条件原C有45条双面共享边，方向原本全部相反；规范化后28条同向，涉及5/6条件。36份实例重复计算为168条同向C/C边，恢复后全部为0。完整逐面来源ID、正逆排列和反向标志见 [source_C_face_mapping.csv](source_C_face_mapping.csv)，去重条件统计见 [来源恢复清单](review/source_restoration_summary.csv)。

逐角追溯采用两个明确索引域：`input_corner_*`保存旧处理前N面记录；`output_corner_*`按最终保留面及角序映射。`source_face_ids`不变，`source_corner_ids`与最终角序同步。边界最近点保留原先选中的几何见证并重映射角点编号，没有重新做最近点搜索或改变并列最小值的选择。本批没有UV、颜色或逐角法线，相关项NA；不宣称已支持任意未登记属性。

## 3. 方向结果：已有连接终于可以正确表达

下表每组12份。可达面积以每份保留free总面积为分母、再作输出宏平均；不是接合成功率，也不是全部网格都已定向。合法图沿实际索引共边传播，排除原检测器标记的非法面对；方向图额外要求共享边两侧绕序相反。C面作为已知起点。

{table(['方法','阶段','合法C/F边数','方向一致C/F边数','有方向一致接口的输出','方向路径可达free面积','无方向可达free面积'],direction)}

RBF宏平均方向可达面积 **0.2643% → 19.9459%**，BLEND **0.1584% → 6.8409%**；D0维持0.0137%。最终36/36输出的方向可达面积都等于各自原无方向可达面积，其中很多输出两者均为0，不能读成36份都接合成功。

**只恢复C还不够。** RBF在RESTORED_C_ONLY阶段，方向一致接口从3条降到2条、方向可达面积降到0.0958%；随后固定正确来源C整理free才达到9条和19.9459%。这解释了为何不能单独翻回C后就评价最终效果。

全部已有合法接口的逐例结果如下，箭头顺序为 MODEL_POST → RESTORED_C_ONLY → ORIENTED：

{table(['方法','父/噪声','合法接口边数（不变）','方向一致接口边数','方向路径可达面积'],examples)}

RBF的86ec n0从3.047%到85.333%，n1从0.124%到77.262%；bdd2 n0从0到76.756%。这些大幅变化说明**此前被方向合同遮蔽的是已有连接的可传播性**。RBF的合法接口依然只存在于3/12输出、2/6父；BLEND为2/12输出、1/6父，D0为1/12输出、1/6父。没有新顶点合并、添面或接缝生成。

## 4. 剩余冲突没有隐藏或修掉

求解器沿全部合格双面索引边建立异或约束，和用于评价的“排除非法面对”图不是同一个图。恢复后的所有C方向固定；可解分量按确定规则定向，冲突分量整体保持，不删面、拆点或放开C。

{table(['方法','合格双面约束全可解输出','已判定固定C冲突分量','奇偶环冲突分量','非流形边总数','含非流形边输出','翻转free面总数'],conflicts)}

旧合同每组10/12输出含直接固定C方向矛盾；来源恢复后，这类C/C矛盾消失。三组各剩同一个 **f596 n0** 的177面纯free分量，213条合格边约束含奇偶环矛盾。该分量不含C，所以不能再归因于C锚点；求解器将其锚点兼容性记为 `NA_PARITY_CONFLICT`，而非通过。整分量未翻，仍有101条合格边同向；101是保守保留后的未满足边数，不是最少要改的边数。

非流形边不纳入两面异或求解，三组原80/84/80条全部保留；每组10/12输出存在。全体还保留开放边界。无C分量的全局正反选择只是确定性约定，**没有认证外向法线、水密、流形或实体体积**。原输出均未发现精确退化面或索引完全重复面，但这不否定近退化、近重叠及原非法相交。

## 5. 几何质量原值继承，不冒称重新评分

36/36通过逐位顶点、无序索引面、来源面ID及角点追溯检查，72新NPZ与PLY互相一致。方向之外的几何等价成立后，以下原指标按原定义和分母继承；108阶段行及864条A-VMR/PMR记录均有 `INHERITED` 标签及原评分身份。本轮完整scorer调用为0，不宣称重新运算后浮点值逐位相同。

{table(['方法','CF非法影响free比例','FF非法影响free比例','I冲突覆盖','free小角<5°','严格A-VMR','严格PMR'],inherited)}

A-VMR/PMR严格档为模型单位1e-6，边界域仍是全部来源C开放边界，不是预先指定的必接接口，不能混作论文原尺度复现。8档全部保留，PMR始终0。逐顶点距离、无方向合法共边/图、面积、小角及删面情况未变。

**每组四份座面仍无合法C/F共边。** 固定2%L容差下，RBF仅866e n0有1/4边界顶点匹配（该例VMR25%，四份座面宏均6.25%）；其余RBF座面及D0/BLEND座面均为0。这个原有的单点接近不等于共边接合，三阶段数值不变。 来源恢复不解决主座面错层、生成另一层座面或背板并存；这些视觉观察引用上一轮同一几何的图审，本轮未生成新图、也未做新的视觉评分。RBF相对D0的FF约+0.530pp、I约+0.704pp代价仍在；BLEND的FF约+1.391pp仍在。

## 6. 执行与独立审计

入口：`r2/tools/source_winding_export_v1/run.py`。登记、逐例started/complete、文件hash和总账均保存；恢复只读复用完整PASS收据，遇到不完整已尝试例不会自动重放。科学实现冻结后未改。

实际Windows/WSL资源检查已记录：RTX5080共16303MiB，WSL派发前使用2600MiB、空闲13380MiB、利用率2%，无CUDA计算进程；24逻辑CPU，WSL可用内存约61.4GiB。先执行1个已授权真实例，完整峰值RSS58256KiB、耗时0.537秒，再派发2个CPU worker，每个最多2线程；没有GPU worker或模型探针。

记录的执行阶段墙钟 **{done['phase_seconds']:.3f}秒**，包含末尾保护文件hash复核，未包含最初准备和首次保护核验；36例任务内部耗时之和 **{sum(r['seconds'] for r in done['receipts']):.3f}秒**，不是并行墙钟。单worker观测最大RSS **{max(r['peak_RSS_KiB'] for r in done['receipts'])/1024:.2f}MiB**。未以并发时间作隔离性能比较。

2314份原数据、资产、权重、旧代码、结果和报告前后SHA256一致。源恢复8项合成正反检查通过，另有独立审计直接逆映射来源并用不同的异或并查集校验定向结果；审计回执见 [independent_audit/completion.json](independent_audit/completion.json)，实际源码随包快照保留。审计对36例、72NPZ/PLY、108图状态及864继承VM行逐项核验，不调用本轮恢复函数或求解器，也不重跑质量评分。实际通过432个逐角属性字段、11160个最近角点见证、540项配对和990项汇总检查。独立审计首次因只支持ASCII的读取器遇到二进制PLY而停止；失败版本、原因与修正diff保存在 independent_audit/engineering_001/，增加独立二进制读取后CPU审计通过，未重跑任何来源恢复、定向、GEO或模型步骤。

## 7. 下一步优先项与检验假说

**优先把“来源C恢复→固定来源C整理free”的独立导出合同接入维护流程，作为可选、可追溯的导出路径。** 本轮只完成独立实验，并未自动改默认入口。检验假说：只要保留正确来源映射且相应分量可定向，此导出能消除规范化带来的方向矛盾，并让已有合法连接的方向可达性达到无方向上限，同时逐位保持顶点和无序连接；无来源映射或存在奇偶环/非流形时必须明确标记，不能猜方向或默默修拓扑。

模型研究的未解问题仍是座面等条件的真实几何接续及RBF相交代价；不能因本轮方向指标变好晋级RBF、追加loss/训练或扩大搜索。本固定任务完成后停止，未commit/push。

## 8. 交付索引

|文件|内容|
|---|---|
|[逐父噪声配对](paired_per_parent_noise.csv)|36例三阶段方向变化及几何上限|
|[三阶段逐输出](topology_per_output.csv)|108行实际共边/方向/可达与非流形结果|
|[完整方向约束](orientation_constraints.csv)|逐边约束、排除原因和修复后满足情况|
|[分量分类](orientation_components.csv)|792分量的固定C/奇偶冲突与翻转记录|
|[几何等价](geometry_equivalence.csv)|36例原几何和缓存评分身份|
|[逐顶点距离](boundary_vertices.csv)|三阶段距离继承与角点见证重映射|
|[完整A-VMR/PMR](AVMR_PMR_per_output.csv)|8阈值×36例×3阶段，864行|
|[原质量继承](quality_inherited_per_output.csv)|原CF/FF/I/角度/面积与分母，108行|
|[执行收据](execution_completion.json)|实际预算、时间、hash和停止状态|
|outputs/方法/请求ID/|restored_C、oriented的NPZ/PLY和完整逐例账本|
|independent_audit/ 与 review/|独立审计、来源清单、实际源码、code.diff和hash|
|R2_SOURCE_WINDING_EXPORT_REVIEW.zip|小于10MB审阅包；没有大权重|

本报告中的变化仅适用于这6条件、2份噪声及三套既有POST，不外推为全体用户输入或未见对象的验证。
'''
    C.textfile(C.OUT/'REPORT.md','\ufeff'+report)
    C.write(C.OUT/'review/report_receipt.json',dict(status='PASS',report_sha256=C.sha(C.OUT/'REPORT.md'),
        source_inputs={str(C.OUT/n):C.sha(C.OUT/n) for n in ['execution_completion.json','topology_summary.csv','paired_per_parent_noise.csv','orientation_per_output.csv','quality_inherited_per_output.csv','AVMR_PMR_per_output.csv','independent_audit/completion.json']},
        scientific_calls=0))
    print('REPORT_WRITTEN',flush=True)


if __name__=='__main__':main()
