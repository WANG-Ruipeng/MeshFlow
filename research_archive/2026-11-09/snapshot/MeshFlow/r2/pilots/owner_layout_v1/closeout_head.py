"""Close a failed frozen gate honestly; never runs or retries a model."""
import csv,datetime,difflib,json,shutil
from pathlib import Path
from . import head as H

def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def pct(v):return 'NA' if v in (None,'','None') else f'{100*float(v):.2f}%'

def main():
    gate=H.read(H.OUT/'gate.json')
    if gate['status']=='PASS':raise SystemExit('Gate passed: finish authorized geometry before final closeout.')
    if (H.OUT/'completion.json').exists():raise SystemExit('Already complete; no overwrite')
    r=H.contract();done={a:H.read(H.OUT/'heads'/a/'complete.json') for a in r['head_arms']}
    assert all(x['updates']==2000 for x in done.values())
    ev=H.read(H.OUT/'evaluation_head/complete.json');assert ev['head_batch_forwards']==4
    assert not (H.OUT/'generation_registration.json').exists()
    source=H.read(H.OUT/'prepared_code_checks.json');assert H.sha(source['feature_source'])==source['feature_source_sha256']
    summary=rows(H.OUT/'evaluation_head/summary.csv');parents=rows(H.OUT/'evaluation_head/per_parent.csv')
    lines=['| 头/分组 | owner召回 | 真Null拒识 | alias拒识 | TP/FN/FP/TN | Brier↓ | ECE10↓ |',
           '|---|---:|---:|---:|---|---:|---:|']
    for x in summary:
        lines.append(f"| {x['arm']} / {x['split']} | {pct(x['owner_recall'])} | {pct(x['null_specificity'])} | {pct(x['alias_specificity'])} | {x['TP']}/{x['FN']}/{x['FP']}/{x['TN']} | {float(x['brier']):.6f} | {float(x['ECE10']):.6f} |")
    plines=['| 父/分组 | owner正确/总数 | 真Null误激活/总数 | alias误激活/总数 |','|---|---:|---:|---:|']
    peredge=rows(H.OUT/'evaluation_head/per_edge.csv')
    for x in parents:
        if x['arm']!='CONTEXT' or not (x['split']=='DEV' or x['parent_id'].startswith(('73b369','3f7417','85b169','8c9b20'))):continue
        z=[q for q in peredge if q['arm']=='CONTEXT' and q['parent_id']==x['parent_id']]
        null=sum(q['role']=='NULL' and float(q['logit'])>=0 for q in z)
        alias=sum(q['role']=='ALIAS' and float(q['logit'])>=0 for q in z)
        plines.append(f"| {x['parent_id'][:8]} / {x['split']} | {x['TP']}/{x['owner']} | {null}/{x['Null']} | {alias}/{x['alias']} |")
    gates='\n'.join(f'- `{k}`：'+('PASS' if v else '**FAIL**') for k,v in gate['gates'].items())
    text=f'''# R2 C-only owner 布局头独立实验

2026-10-17。固定实验完成；**预定 CONTEXT 候选未通过准入门槛，按冻结协议停止后续生成。** 两头均真实训练2000步，并完成一次固定终点评价；不是只写方案，也不是根据DEV换候选或调阈值。

这次回答的是“新C-only头能否同时恢复owner召回、保留Null/alias拒识”。它没有生成新mesh，因此没有本轮FF/I/主体接续的改善或退化结论。上轮LONG+绑定16/16及B换布局12/16仍是原记录，不移作本轮成绩。

## 1. 实际完成的比较

同一TRAIN47：400条边，360 owner、34真Null、6 alias。MLP897参数；CONTEXT17985参数、两层32宽四头注意力。两者共享原C-only24维特征＋边序rank与边数两维，仅TRAIN标准化。C/N是模型可观察输入；父ID、Y、来源面ID和标签没有进入forward。

各2000个full47批更新，fresh AdamW、LR0.001、weight_decay0.0001，未加权BCE按父先平均再47父平均；固定初始化种子与最终checkpoint，阈值0.5，无调温/阈值扫描、无early stopping、无事后选best。每头94000次父曝光（每父2000），数据没有新增父、几何或噪声。CONTEXT预定为生成候选，MLP只作对照。

训练全程CPU：BF16前向，FP32参数、优化器和BCE。两个进程并行，各1计算线程。MLP {done['MLP']['seconds']:.2f}秒，CONTEXT {done['CONTEXT']['seconds']:.2f}秒；是并发运行的各阶段墙钟，不作为孤立硬件对比。两份最终权重及完整学习曲线保留。

终态每个头对TRAIN47一个packed batch、DEV3一个packed batch，共4次前向。总实际占用头前向4004、反向4000、更新4000；生成器/D0/GPU调用均0。每条件receipt中的batch参与标记不能再累加成额外前向。

## 2. 全量分类与校准结果

{chr(10).join(lines)}

TRAIN为47父400边；DEV为3父24边（20 owner、1真Null、3 alias）。表中召回/拒识按对应角色计数，Brier/ECE按有效边micro统计；训练loss按父平均，分母不同。TRAIN校准好不代表跨父概率校准或开放生成成功。完整NLL、逐边概率和固定10-bin可靠性表另附。

## 3. 主候选的关键逐父结果

{chr(10).join(plines)}

全部47训练父均保留在逐父CSV；此处只突出原七C面板，未按结果剔除任何输入。0/0表示该父没有此类标签，不称100%拒识。

本次CONTEXT的5条owner漏判全部来自2d（边1/2/11/14/15）；2d同时误激活alias边9/12。608的唯一真Null边1以约0.99999999996概率被激活；610四条owner均正确。这个错误分布说明不能只给75%总体召回，也不能把三父一概称为失败。

固定门槛结果：

{gates}

DEV没有参与更新、标准化、阈值或模型选择。它是已经多轮观察的DEV，不能称全新测试。协议原文“各DEV父≥90%意味着20/20”有算术错误，已在**终点评价前**独立勘误：实际代码执行逐父≥90%，2d允许12/13，故最低19/20；没有事后改门槛。见[勘误](PROTOCOL_ERRATA.md)。

## 4. 应怎样解释

这说明本版固定特征、容量、监督和预算没有同时满足跨父owner/Null/alias的准入要求，不能靠更高的总激活率宣布修复。400条边中90%为owner，全部打开就能取得100%owner召回，却会误开所有负类；所以二者必须一起看。

**复核旧缓存还纠正了本次提议中的一个关键前提：LONG在这三个DEV上实际已经召回20/20个来源owner；其16/4/4布局是全部24边打开，另外误开了3条alias和1条真Null，并不是owner漏开。** 因此，之前LONG的DEV主体可达仍0/12，不能归因成“还没把该开的owner打开”。布局误激活可能有代价，但它与seam侧边、bulk几何未接通是不同问题。本轮没有新的生成来量化二者。

对熟悉四父，新CONTEXT确有分类层面的正结果：62/62 owner召回且4/4真Null正确关闭，优于旧LONG的48/62及旧B的28/62召回；但这些父参加了训练，不能代替DEV。MLP也召回62/62，却误开其中2条真Null。注意力新增容量帮助了训练池分离，但DEV并未优于MLP；不能把“加上下文”自动当成泛化修复。

这不是证明C-only预测在数学上不可能，也不是证明新布局会使生成质量变差。来源标签对应某一个完整网格的分配规范，不必是所有合理完成的唯一角色答案。alias还含同一来源面的owner优先顺序：当前二分类只决定是否占独立槽，**没有预测alias伙伴或恢复第三点引用**。边序rank依赖C局部顶点编号，不宣称任意重编号等变。

旧B占用头并非只训练了200步：它继承旧头后又跑800更新；保留四C旧类别权重与47父新分布不一致是可检验因素，不能称实现已被证实训崩。此次同时改变了loss权重、独立训练预算、特征/标准化和fresh优化器，因此新旧头变化不单独归因于其中一项。两种新头之间的同特征同预算对照用于观察上下文增量。

若继续这一方向，优先把“是否需要一张新的面”和“是否与另一边属于同一张面”分开定义，并处理C单独不能决定的多解性；不要自动加大生成器训练或把DEV加入头的监督。本次不执行下一版本。

## 5. 下游阶段与停止条件

| 阶段 | 状态 | 实际预算 |
|---|---|---:|
| MLP固定2000步 | COMPLETE | 2000更新 |
| CONTEXT固定2000步 | COMPLETE | 2000更新 |
| 两头固定终点评价 | COMPLETE | 4 packed前向 |
| LONG/B新布局采样 | NOT_RUN_GATE_FAILED | 0/56 RAW |
| 官方D0与原GEO_C | NOT_RUN_GATE_FAILED | 0/56 |
| 1.5%L绑定 | NOT_RUN_GATE_FAILED | 0/56单元 |
| 新mesh评分/图审 | NOT_RUN_GATE_FAILED | NA |

后续生成/评分/绑定适配器已准备，但没有绕过门槛运行。绑定保护仅做7个微型CPU断言，零实验网格；没有把这7项当作新生成结果。未自动换用MLP、扫描阈值、延长训练、添加A损失、清理权重、修改默认或commit/push。

## 6. 文件与可恢复性

- [逐边预测848行](evaluation_head/per_edge.csv)、[逐父100行](evaluation_head/per_parent.csv)、[两头完整汇总](evaluation_head/summary.csv)。
- [原七C与旧LONG/B头的配对](classification_analysis/seven_C_parents.csv)、[旧新头汇总](classification_analysis/seven_C_summary.csv)、[全部分类错误边](classification_analysis/all_failed_edges.csv)、[概率可靠性分箱](classification_analysis/reliability_bins.csv)。旧头只读缓存，没有重跑。
- [学习曲线](classification_analysis/learning_curve.png)、[MLP最终权重](heads/MLP/step2000.pt)、[CONTEXT最终权重](heads/CONTEXT/step2000.pt)。
- [固定门槛](gate.json)、[冻结合同](PROTOCOL.md)、[独立设计核对](DESIGN_REVIEW.md)、[源码核对](CODE_REVIEW.md)。
- [实际代码与hash](evidence/)、[完整完成账本](completion.json)。原数据、评分器、权重及历史报告只读保留。

只读入口：`python -B -m r2.pilots.owner_layout_v1.resume --stage status`。已完成训练不自动重跑；失败门槛阻止生成登记，根完成账本封存后不自动继续。
'''
    (H.OUT/'REPORT.md').write_text(text,encoding='utf-8')
    evidence=H.OUT/'evidence';evidence.mkdir(exist_ok=True);sources=[];diff=[]
    paths=sorted(Path(__file__).parent.glob('*.py'))+[Path(source['feature_source'])]
    for i,p in enumerate(paths):
        t=evidence/f'{i:02d}_{p.name}';shutil.copyfile(p,t)
        sources.append(dict(source=str(p),sha256=H.sha(p),snapshot=str(t)))
        if p.parent==Path(__file__).parent:
            diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8-sig').splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+str(p.relative_to(H.ROOT))))
    H.csvwrite(evidence/'source_hashes.csv',sources);(evidence/'code.diff').write_text(''.join(diff),encoding='utf-8')
    files=[H.OUT/'REPORT.md',H.OUT/'gate.json',H.OUT/'PROTOCOL.md',H.OUT/'PROTOCOL_ERRATA.md',H.OUT/'registration.json',
           H.OUT/'evaluation_head/complete.json',evidence/'source_hashes.csv',evidence/'code.diff']
    files+=list((H.OUT/'evaluation_head').glob('*.csv'))+list((H.OUT/'heads').glob('*/step2000.pt'))
    H.write(H.OUT/'completion.json',dict(status='COMPLETE',outcome='HEAD_GATE_FAILED_NO_GENERATION',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),head_updates=4000,head_forwards=4004,head_backwards=4000,
        generator_forwards=0,D0_forwards=0,GPU_calls=0,new_RAW=0,new_POST=0,new_binding_cells=0,
        per_edge_rows=len(peredge),per_parent_rows=len(parents),no_automatic_continuation=True,
        protected_inputs_unchanged=True,files={str(p):H.sha(p) for p in files}))
    print(json.dumps(H.read(H.OUT/'completion.json'),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
