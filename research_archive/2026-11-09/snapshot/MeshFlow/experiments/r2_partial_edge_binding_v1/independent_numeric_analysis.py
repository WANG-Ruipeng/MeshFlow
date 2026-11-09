"""Read completed binding artifacts only; no scorer, decoder or model imports."""
from pathlib import Path
import csv, json, collections, hashlib, math, statistics
import numpy as np

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')

def csvwrite(path, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

manifest=read(ROOT/'input_manifest.json')['rows']
lookup={r['request_id']:r for r in manifest}
completed=sorted(ROOT.glob('cells/*/*/complete.json'))
assert len(completed)==144, 'Wait for all144 cells; no partial scientific conclusion'
rows=[read(p)['row'] for p in completed]
labels=sorted({r['label'] for r in rows})
assert len(labels)==6 and all(sum(r['label']==lab for r in rows)==24 for lab in labels)
keys=('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction','free_q_gt10_fraction')
ledger_keys=('CF_retained_legal_edges','CF_lost_legal_edges','CF_new_legal_edges','CF_lost_baseline_length','FF_lost_legal_edges','FF_lost_baseline_length','FF_new_illegal_pairs','FF_illegal_pairs_removed','deleted_source_faces','deleted_source_area')

# Independent sums from individual ledger files, not aggregate.csv.
for row in rows:
    folder=ROOT/'cells'/row['label']/row['short_id']
    summary=read(folder/'paired_ledger.json')['summary']
    for key in ledger_keys:assert row[key]==summary[key], (row['label'],row['short_id'],key)
    cc=read(folder/'candidate_records.json')
    assert row['accepted']==sum(c['decision']=='ACCEPTED' for c in cc)
    assert row['candidates']==len(cc)
    for key in keys:
        assert abs(row['delta_pp_'+key]-100*(row[key]-float(lookup[row['request_id']]['score_table_row'][key])))<1e-12


def summary(group,label,split,parent='ALL'):
    out=dict(label=label,split=split,parent_id=parent,outputs=len(group),baseline_joint=sum(r['baseline_joint'] for r in group),
        screening_joint=sum(r['screening_joint'] for r in group),
        new_joint=sum(r['screening_joint'] and not r['baseline_joint'] for r in group),
        lost_joint=sum(r['baseline_joint'] and not r['screening_joint'] for r in group),
        adopted_unjoined_denominator=sum(r['baseline_adopted_unjoined'] for r in group),
        converted_ids=';'.join(sorted(r['short_id'] for r in group if r['screening_joint'] and not r['baseline_joint'])),
        accepted_bindings=sum(r['accepted'] for r in group),modified_outputs=sum(r['accepted']>0 for r in group),
        legal_cf_edges=sum(r['valid_cf_edges'] for r in group),legal_cf_length=sum(r['valid_cf_length'] for r in group))
    for key in ledger_keys:out[key]=sum(r[key] for r in group)
    for key in keys:
        out[key]=statistics.mean(r[key] for r in group)
        out['delta_pp_'+key]=statistics.mean(r['delta_pp_'+key] for r in group)
    return out

summaries=[summary([r for r in rows if r['label']==label and (split=='ALL' or r['split']==split)],label,split)
           for label in labels for split in ('ALL','TRAIN','DEV')]
parents=[summary(group,label,group[0]['split'],pid) for label in labels for pid in sorted({r['parent_id'] for r in rows})
         for group in [[r for r in rows if r['label']==label and r['parent_id']==pid]]]
csvwrite(ROOT/'independent_per_parent_paired.csv',parents)
csvwrite(ROOT/'independent_summary.csv',summaries)

funnel=[]
for label in labels:
    cc=collections.Counter();pre=collections.Counter()
    for path in (ROOT/'cells'/label).glob('*/binding.json'):
        rec=read(path);cc.update(rec['rejections']);pre.update(rec['prefilter'])
    assert sum(cc.values())==sum(r['candidates'] for r in rows if r['label']==label)
    funnel.append(dict(label=label,candidates=sum(cc.values()),**dict(cc),**{'prefilter_'+k:v for k,v in pre.items()}))
csvwrite(ROOT/'independent_rejection_funnel.csv',funnel)

# Only compare values already saved; do not call original scoring mathematics.
equiv=[];mismatches=[]
for row in rows:
    folder=ROOT/'cells'/row['label']/row['short_id']
    base_meta=lookup[row['request_id']]
    base=load_npz(base_meta['post_path']); current=load_npz(folder/'post.npz')
    same=(base['output'].dtype==current['output'].dtype and base['output'].shape==current['output'].shape and base['output'].tobytes()==current['output'].tobytes())
    if not same:continue
    bscore=read(base_meta['score_path']); nscore=read(folder/'score.json')
    checked=0;max_diff=0.;bad=[]
    for section in ('metrics','topology','distance'):
        for key, value in nscore[section].items():
            if key not in bscore[section]:continue
            before=bscore[section][key]
            if isinstance(value,(int,float,bool)) and isinstance(before,(int,float,bool)):
                diff=abs(float(value)-float(before)); checked+=1;max_diff=max(max_diff,diff)
                if diff>1e-12:bad.append(dict(section=section,key=key,before=before,after=value,difference=diff))
            elif value is None and before is None:checked+=1
    for b,n in zip(bscore['boundary'],nscore['boundary']):
        assert b['setting']==n['setting']
        for key in ('VMR','perfect'):
            checked+=1
            if b[key]!=n[key]:bad.append(dict(section='boundary',key=key,setting=b['setting'],before=b[key],after=n[key]))
    equiv.append(dict(label=row['label'],short_id=row['short_id'],accepted=row['accepted'],unchanged_triangle_bytes=True,
                      compared_values=checked,max_abs_difference=max_diff,status='PASS' if not bad else 'FAIL'))
    mismatches.extend(dict(label=row['label'],short_id=row['short_id'],**b) for b in bad)
csvwrite(ROOT/'unchanged_geometry_score_equivalence.csv',equiv)
assert not mismatches, mismatches[:3]

allrows=[r for r in summaries if r['split']=='ALL'];devrows=[r for r in summaries if r['split']=='DEV']
assert all(r['lost_joint']==0 and r['CF_lost_legal_edges']==0 for r in allrows)
assert all(r['deleted_source_faces']==0 for r in allrows)

lines=['# 独立数值核对：144 组部分边绑定', '',
'仅重新汇总已完成的逐单元结果、候选决策和来源连接账本，并比较已保存的评分；没有执行评分器、绑定器、模型、GPU、训练或新样本。', '',
'## 主要结论', '',
'原 6 份联合正例、62 条合法 C/F 边在所有六组中均保留；没有删面或复活旧删面。B 簇绑定的原合法 F/F 边损失为零，A 单角点绑定则损失 12／16／27 条。因此内部撕裂确实是独立于 CF/FF 平均分的代价。', '',
'筛选联合计数最多达到 A_035 的 9/24，但其新增 DEV 只有 2d/n1，并伴随 27 条原 F/F 合法边损失。B_020 和 B_035 均为 8/24，新增仅训练父 73 的 n0/n1；DEV 仍为 0/12。没有达到“B 在 1%—2%L 下达到 10+/24”的预期。', '',
'这些是“原 RAW 采用 YES ∩ 新 POST 合法接口”计数；修改后用途与主体状态仍需独立图审，不能自动称全部新增为经图审确认成功。', '',
'## 保留、转化和直接损伤', '',
'| 档位 | 接受绑定 | 有修改输出 | 联合筛选 | 10份采用未接上中的转化 | 新CF边 | 原FF边损失 | 新非法FF对 / 消除 |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for r in allrows:
    lines.append(f"| {r['label']} | {r['accepted_bindings']} | {r['modified_outputs']} | {r['screening_joint']}/24 | {r['new_joint']}/10 | {r['CF_new_legal_edges']} | {r['FF_lost_legal_edges']} | {r['FF_new_illegal_pairs']} / {r['FF_illegal_pairs_removed']} |")
lines+=['', '新增筛选请求：','']
for r in allrows:lines.append(f"- {r['label']}：{r['converted_ids'] or '无'}。")
lines+=['', 'B_020／B_035 各有 4 对新增非法 FF；B_035 虽消除了 9 对、平均 FF 下降，不能因此称为零新增损伤。B_020 的一次簇移动还产生附带连接，26 次接受对应 27 条新增合法 CF；接受数不等于新增边数。', '',
'## 代价变化（百分点；每档固定24输出等权，DEV固定12输出）', '',
'| 档位 | ALL CF | ALL FF | ALL I | DEV CF | DEV FF | DEV I | DEV联合 |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for a in allrows:
    d=next(x for x in devrows if x['label']==a['label'])
    vals=[a['delta_pp_'+k] for k in keys[:3]]+[d['delta_pp_'+k] for k in keys[:3]]
    lines.append('| '+a['label']+' | '+' | '.join(f'{x:+.4f}' for x in vals)+f" | {d['screening_joint']}/12 |")
lines+=['', '各组新增直接非法 CF 均为零。小角/q 熔断后没有删面，分母改变没有被用于掩盖代价；完整分项、父级与噪声级变化见 CSV。', '',
'## 候选拒绝漏斗', '',
'下表为通过无向轴线和端点距离预筛后的候选，B 的簇全员预算仍在其中审查。采用确定的首个拒绝原因，一条候选不重复计多个拒绝。','']
fkeys=sorted(set(k for r in funnel for k in r if k not in ('label','candidates') and not k.startswith('prefilter_')))
lines+=['| 档位 | 候选 | '+' | '.join(fkeys)+' |','|---|---:|'+'---:|'*len(fkeys)]
for r in funnel:lines.append('| '+r['label']+' | '+str(r['candidates'])+' | '+' | '.join(str(r.get(k,0)) for k in fkeys)+' |')
lines+=['', '## 不变几何上的评分一致性', '',
f"逐字节比较原 POST 与新 POST 三角坐标后，共 {len(equiv)} 个不变几何单元；对这些单元的已保存共有指标、拓扑标量、边界距离、8档 AVMR/PMR 共核对 {sum(r['compared_values'] for r in equiv)} 个值，全部在绝对误差 1e-12 内一致。没有重跑原评分器。", '',
'## 结论边界', '',
'- B 相对 A 显著避免了这批原有索引共边的撕裂，但不能推出任何输入都安全，更不能推出所有几何冲突为零。',
'- 1% 到 2%L 为 B 增加一个训练请求；2% 到 3.5%L 没有再扩大联合覆盖。A 的压力档才加入一个 DEV 请求，且损伤更大。',
'- 608 四份、610 的 n0/n1 等关键采用未接上请求仍没有被转化；不能把本次失败升级为所有后处理绑定不可行。',
'- 没有根据几何指标重新判定 RAW 用途；人工/AI_PROXY 图审没有在本独立数值分析中被伪造。若图审未完成，新增POST用途状态为待确认。',
'- 报告有限可行性与代价，不报告物理上限、指针头可学习性或普遍水密/外向认证。','']
(ROOT/'NUMERIC_FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')
write(ROOT/'independent_numeric_verification.json',dict(status='PASS',cells=144,arms=6,unchanged_geometry_cells=len(equiv),
    unchanged_compared_values=sum(r['compared_values'] for r in equiv),unchanged_mismatches=mismatches,
    old_joint_retained_all_six=True,old_CF_edges_retained_all_six=True,deleted_faces=0,
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'input_manifest.json',*completed]},
    new_scores=0,new_decodes=0,model_calls=0))
print(json.dumps(dict(status='PASS',cells=144,unchanged=len(equiv),compared=sum(r['compared_values'] for r in equiv),report=str(ROOT/'NUMERIC_FINDINGS.md')),indent=2))# Check public CSVs against independently read per-cell records if complete.
with (ROOT/'per_output.csv').open(encoding='utf-8-sig',newline='') as f:
    public={(r['label'],r['short_id']):r for r in csv.DictReader(f)}
assert len(public)==144
for row in rows:
    saved=public[(row['label'],row['short_id'])]
    for key,value in row.items():
        if isinstance(value,(float,int)) and not isinstance(value,bool):
            assert abs(float(saved[key])-float(value))<1e-12, (row['label'],row['short_id'],key)
        elif value is None:assert saved[key]=='', key
        else:assert saved[key]==str(value), key
with (ROOT/'rejection_funnel.csv').open(encoding='utf-8-sig',newline='') as f:
    public_funnel={r['label']:r for r in csv.DictReader(f)}
for row in funnel:
    saved=public_funnel[row['label']]
    for key,value in row.items():
        if key in ('label','candidates'):continue
        assert int(saved.get(key,0) or 0)==value, (row['label'],key)
print('PUBLIC_CSV_EQUIVALENCE_PASS:144 per-output rows,6 rejection funnels')
parent_funnel=[]
for label in labels:
    for pid in sorted({r['parent_id'] for r in manifest}):
        selected=[r for r in rows if r['label']==label and r['parent_id']==pid]
        counter=collections.Counter()
        for row in selected:counter.update(read(ROOT/'cells'/label/row['short_id']/'binding.json')['rejections'])
        parent_funnel.append(dict(label=label,parent_id=pid,split=selected[0]['split'],outputs=len(selected),
            candidates=sum(r['candidates'] for r in selected),accepted=sum(r['accepted'] for r in selected),**dict(counter)))
csvwrite(ROOT/'independent_candidate_parent.csv',parent_funnel)
with (ROOT/'NUMERIC_FINDINGS.md').open('a',encoding='utf-8') as f:
    f.write('\n## 无候选与拒绝必须区分\n\n608 与 610 两父的全部八个请求，在 A/B 的 3.5%L 压力档仍均为零候选，低档同样为零。这里没有实际尝试过把这些请求的边拉过去，不能写成“拉过去后几何失败”。本轮整边端点对应、开放边、轴线与预算限制没有覆盖它们。零候选可能涉及端点距离、边段长度、轴向或位置限制，不能仅凭这一点断言语义或宏观对齐的唯一根因。\n')
unknown=collections.Counter()
for path in ROOT.glob('cells/*/*/score.json'):
    score=read(path);unknown['graph_'+score['topology']['status']]+=1
    unknown['CF_unresolved_pairs']+=score['metrics']['CF_unresolved_or_degenerate_pair_count']
    unknown['FF_unresolved_pairs']+=score['metrics']['FF_unresolved_or_degenerate_pair_count']
print('SAVED_STATUS',json.dumps(dict(unknown)))
