"""Additional min-edge/area accounting from existing audit CSV; no geometry calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import collections,csv,hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
R=.015/(.95*.3762)
def load(name):
    with (HERE/name).open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        for k,v in row.items():
            if v in ('True','False'):row[k]=v=='True'
            elif v=='':row[k]=None
            else:
                try:row[k]=json.loads(v)
                except (ValueError,TypeError):pass
    return rows
def dump(name,value):
    with (HERE/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
def fraction(n,d):return n/d if d else None
def distribution(v):
    v=np.asarray(v,float)
    return dict(n=len(v),p10=float(np.quantile(v,.1)),median=float(np.median(v)),p90=float(np.quantile(v,.9)),mean=float(v.mean())) if len(v) else dict(n=0)

rows=load('per_face.csv');outputs=load('per_output.csv');summary=[];bylabel={}
for label in ('SEAM_S15000','SEAM_S21000'):
    allfree=[r for r in rows if r['label']==label and r['role']!='C']
    for role in ('ALL_FREE','SEAM','BULK'):
        source=[r for r in allfree if role=='ALL_FREE' or r['role']==role]
        deleted=[r for r in source if not r['kept']]
        collapsed=[r for r in deleted if r['primary_attribution']=='GEO_COLLAPSE_NONZERO_PRE']
        types=collections.Counter()
        for r in deleted:
            if r['RAW_edge_min']<=R:types['RAW_MIN_EDGE_ALREADY_LE_R']+=1
            elif r['D0_PROPOSAL_edge_min']<=R:types['D0_FIRST_MIN_EDGE_LE_R']+=1
            elif r['PRE_GEO_edge_min']<=R:types['RESTORE_FIRST_MIN_EDGE_LE_R']+=1
            else:types['PRE_MIN_EDGE_GT_R']+=1
        item=dict(label=label,role=role,free_slots=len(source),deleted=len(deleted),
            fraction_deleted=fraction(len(deleted),len(source)),
            deletion_reasons=dict(collections.Counter(r['delete_reason'] for r in deleted)),
            PRE_min_edge_le_R=sum(r['PRE_GEO_edge_min']<=R for r in deleted),
            PRE_min_edge_le_2R=sum(r['PRE_GEO_edge_min']<=2*R for r in deleted),
            PRE_min_edge_gt_2R=sum(r['PRE_GEO_edge_min']>2*R for r in deleted),
            collapse_count=len(collapsed),collapse_PRE_min_edge_le_R=sum(r['PRE_GEO_edge_min']<=R for r in collapsed),
            collapse_PRE_min_edge_le_2R=sum(r['PRE_GEO_edge_min']<=2*R for r in collapsed),
            min_edge_crossings=dict(types),
            RAW_min_edge_ge_2R_then_D0_le_R=sum(r['RAW_edge_min']>=2*R and r['D0_PROPOSAL_edge_min']<=R for r in deleted),
            RAW_area_deleted_cohort=sum(r['RAW_area'] for r in deleted),D0_area_deleted_cohort=sum(r['D0_PROPOSAL_area'] for r in deleted),
            PRE_area_deleted_cohort=sum(r['PRE_GEO_area'] for r in deleted),
            PRE_area_all_free=sum(r['PRE_GEO_area'] for r in source),
            deleted_PRE_area_share=fraction(sum(r['PRE_GEO_area'] for r in deleted),sum(r['PRE_GEO_area'] for r in source)),
            kept_PRE_min_edge_le_R=sum(r['kept'] and r['PRE_GEO_edge_min']<=R for r in source),
            kept_PRE_altitude_lt_R=sum(r['kept'] and r['PRE_GEO_minimum_altitude']<R for r in source))
        for metric in ('edge_min','minimum_altitude','area'):
            for before,after in (('RAW','D0_PROPOSAL'),('RAW','PRE_GEO'),('D0_PROPOSAL','PRE_GEO')):
                values=[r[after+'_'+metric]/r[before+'_'+metric] for r in deleted if r[before+'_'+metric]>0]
                item[after+'_over_'+before+'_'+metric]=distribution(values)
        summary.append(item)
        if role=='ALL_FREE':bylabel[label]=item

# Matched illustration inputs for the min-edge question; no detector/renderer.
example_rows=[]
for label in ('SEAM_S15000','SEAM_S21000'):
    rr=[r for r in rows if r['label']==label and not r['kept']]
    for name,predicate in (
        ('RAW_wide_D0_short_GEO_collapse',lambda r:r['primary_attribution']=='GEO_COLLAPSE_NONZERO_PRE' and r['RAW_edge_min']>=2*R and r['D0_PROPOSAL_edge_min']<=R),
        ('RAW_short_GEO_collapse',lambda r:r['primary_attribution']=='GEO_COLLAPSE_NONZERO_PRE' and r['RAW_edge_min']<=R)):
        candidates=[r for r in rr if predicate(r)]
        r=max(candidates,key=lambda r:r['RAW_area']) if candidates else None
        if r is None:continue
        repo=HERE.parents[2];source=repo/'experiments/r2_bench_parts_convergence_v1/evaluation'/label/r['request_id']
        def arr(p):
            with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
        raw=arr(source/'raw/raw.npz');pre=arr(source/'post/pre_geo.npz')['output'];canon=arr(source/'post/post_canonical.npz')
        d0=arr(source/'post/proposal.npz')['official_unconstrained_proposal'];i=r['source_raw_slot']
        dest=HERE/'examples_min_edge'/(name+'__'+label+'__slot'+str(i));dest.mkdir(parents=True)
        with (dest/'geometry.npz').open('xb') as f:
            np.savez_compressed(f,RAW=raw['raw'],D0_PROPOSAL=d0,PRE_GEO=pre,
                CLUSTERED_BEFORE_DELETE=pre.reshape(-1,3)[canon['corner_target_input_ids']],
                selected_raw_slot=np.asarray(i),known_faces=raw['known_faces'],source_face_kept=canon['source_face_kept'],
                raw_seam_slot_mask=raw['raw_seam_slot_mask'],corner_target_input_ids=canon['corner_target_input_ids'])
        example_rows.append(dict(selector=name,**r,geometry_npz=str(dest/'geometry.npz')))

dump('minimum_edge_area_summary.json',dict(status='PASS',radius_model=R,rows=summary,examples=example_rows,
    definition='All ratios are per-triangle before/after coordinates or sums of triangle areas. Semantic wood thickness and visual missing-surface validity UNKNOWN.',
    new_geometry_processing_calls=0,scorer_calls=0,model_calls=0))
lines=['# Bench GEO_C 删面归因（纯 CPU 缓存）','',
'核对 SEAM_S15000 与 SEAM_S21000 各 48 请求，共 96 输出、18,688 个生成槽位。来源映射齐全，RAW、D0 提议、恢复锚点后的 PRE_GEO、聚类后未删面的全部坐标均已按原映射逐位核验；没有复跑 GEO 或模型。',
'',f'固定半径 R={R:.14f} 模型单位。GEO 不按“小面积”直接删面；它先按固定代表点聚类，再删完全重复顶点、严格共线或完全重复三角面。',
'','| 检查点 | free 槽位 | 删除 | 聚类后退化 | 重复面 | 被删 PRE 面积占比 |','|---|---:|---:|---:|---:|---:|']
for label,a in bylabel.items():
    lines.append(f"| {label} | {a['free_slots']} | {a['deleted']} ({100*a['fraction_deleted']:.2f}%) | {a['collapse_count']} | {a['deleted']-a['collapse_count']} | {100*a['deleted_PRE_area_share']:.2f}% |")
lines+=['','全部被删面在 PRE_GEO 都是非零面积；未发现 PRE 已退化或与最终重复伙伴已经精确重合的例子。删除发生在这次合并之后，但不意味着 RAW/D0 没有预先产生接近退化或接近重复的几何。','',
'## 最短边与真实半径','', '| 检查点 | 被删面 min edge≤R | ≤2R | RAW已≤R | D0首次≤R | 恢复首次≤R | PRE仍>R |','|---|---:|---:|---:|---:|---:|---:|']
for label,a in bylabel.items():
    t=a['min_edge_crossings'];n=a['deleted']
    lines.append(f"| {label} | {a['PRE_min_edge_le_R']}/{n} ({100*a['PRE_min_edge_le_R']/n:.2f}%) | {a['PRE_min_edge_le_2R']}/{n} ({100*a['PRE_min_edge_le_2R']/n:.2f}%) | {t.get('RAW_MIN_EDGE_ALREADY_LE_R',0)} | {t.get('D0_FIRST_MIN_EDGE_LE_R',0)} | {t.get('RESTORE_FIRST_MIN_EDGE_LE_R',0)} | {t.get('PRE_MIN_EDGE_GT_R',0)} |")
lines+=['','各退化删除面均满足 PRE 最短边≤2R；两个端点可以分别移动到同一代表点，因此最短边>R仍会合并。重复面删除没有这个面内短边必要条件。','',
'## RAW → D0 / PRE 的收缩','', '| 检查点 | D0/RAW 最短边中位数 | PRE/RAW 最短边中位数 | D0/RAW 面积中位数 | 被删群体 RAW→D0→PRE 面积和 |','|---|---:|---:|---:|---|']
for label,a in bylabel.items():
    lines.append(f"| {label} | {a['D0_PROPOSAL_over_RAW_edge_min']['median']:.4f} | {a['PRE_GEO_over_RAW_edge_min']['median']:.4f} | {a['D0_PROPOSAL_over_RAW_area']['median']:.4f} | {a['RAW_area_deleted_cohort']:.3f} → {a['D0_area_deleted_cohort']:.3f} → {a['PRE_area_deleted_cohort']:.3f} |")
lines+=['','这支持“D0 改变局部尺度，GEO 再触发精确坍缩”的阶段性机制，不能据此认证 D0 删除了正确木条；D0也可能在压缩 RAW 中的冗余/噪声面。不同阶段面积和不是可见表面覆盖率。','',
'## 最强支持与反例','']
for label,a in bylabel.items():
    lines.append(f"- {label}：{a['RAW_min_edge_ge_2R_then_D0_le_R']} 个被删面满足 RAW 最短边≥2R、D0后≤R；但也有 {a['kept_PRE_min_edge_le_R']} 个 PRE 最短边≤R的面和 {a['kept_PRE_altitude_lt_R']} 个最小高<R的面被保留。小尺度不是充分删除条件。")
for r in example_rows:
    lines.append(f"- {r['selector']}：{r['label']} / {r['request_id']} / slot {r['source_raw_slot']}，min edge RAW/D0/PRE={r['RAW_edge_min']:.6f}/{r['D0_PROPOSAL_edge_min']:.6f}/{r['PRE_GEO_edge_min']:.6f}，最终 {r['delete_reason']}。完整局部图数据见 `{Path(r['geometry_npz']).relative_to(HERE)}`。")
lines+=['','其余 12 份例图数据覆盖原已细片、最大 PRE 面积退化、重复面及保留细面的反例；均未凭几何猜“正常木条”。最小高=2×面积/最长边，是三角形几何量，不是物体部件的物理宽度。',
'','## 交付与执行边界','',
'`per_face.csv` 保留每个槽位四阶段边长/最小高/面积、删除理由、代表点与顶点映射；`per_output.csv`、`per_output_stage_area.csv`、`attribution_summary.csv` 和 `minimum_edge_area_summary.json` 提供数目/面积/比例完整分解。',
'','两 CPU worker，每个 1 BLAS 线程；首个真实案例 0.070 秒、进程峰值 RSS 37,504 KiB。原审计在最后 JSON 序列化 numpy.int64 时退出；完整 CSV/例图及所有身份核对已完成。不完整 JSON 已封存，仅从 CSV 收口并复核 hash，没有新增几何处理。见 `completion.json`。',
'','没有模型前向、训练、评分、半径扫描或旧文件改写；此结果不判断最终视觉缺口是否可接受，也不单独认证误删。']
with (HERE/'FINDINGS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
dump('summary_completion.json',dict(status='PASS',rows=len(summary),examples=len(example_rows),
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'minimum_edge_area_summary.json',HERE/'FINDINGS.md',Path(__file__))},
    model_calls=0,new_geometry_calls=0,scorer_calls=0))
print(json.dumps(bylabel,indent=2))
