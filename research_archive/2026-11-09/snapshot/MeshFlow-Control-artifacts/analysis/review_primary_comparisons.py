import csv,json,hashlib,math,statistics,datetime
from pathlib import Path
R=Path('/mnt/e/MeshFlow-Control-artifacts/analysis'); D=R/'comparisons'
def read(f): return list(csv.DictReader((D/f).open(encoding='utf-8-sig')))
ss=read('contrast_distributions.csv'); pp=read('paired_parent.csv'); pn=read('paired_noise.csv')
contrasts={'B1000-A1000','C1000-B1000'}
metrics={'SeamCov','ConflictCov_C','ConflictCov_C_cross','ConflictCov_C_dup','ExposedRetain','D_sym2048','GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','fixed_query_MSE','fixed_query_p95_L','coverage_fraction','CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction'}
checks=[];summary=[]
for row in ss:
 if row['contrast'] not in contrasts or row['metric'] not in metrics or not row['mean']: continue
 key=(row['contrast'],row['panel'],row['cohort'],row['metric'])
 parents=[x for x in pp if tuple(x[k] for k in ['contrast','panel','cohort','metric'])==key]
 ds=[]
 for p in parents:
  noise=[x for x in pn if (x['contrast'],x['panel'],x['metric'])==(key[0],key[1],key[3]) and x['cohort'] in (('OLD','NEW') if key[2]=='ALL' else (key[2],)) and x['parent_id']==p['parent_id']]
  valid=[float(x['candidate_value'])-float(x['reference_value']) for x in noise if x['candidate_value'] and x['reference_value']]
  assert len(valid)==int(p['expected_noises']),key
  calc=statistics.mean(valid)
  assert math.isclose(calc,float(p['delta']),abs_tol=1e-12),key
  ds.append(calc)
 calcmean=statistics.mean(ds);med=statistics.median(ds)
 assert math.isclose(calcmean,float(row['mean']),abs_tol=1e-12),key
 assert math.isclose(med,float(row['median']),abs_tol=1e-12),key
 checks.append(dict(contrast=key[0],panel=key[1],cohort=key[2],metric=key[3],parent_count=len(ds),noise_count=sum(int(x['expected_noises']) for x in parents),noise_to_parent_to_mean_verified=True))
 if row['cohort']=='ALL':
  summary.append({k:row[k] for k in ['contrast','panel','cohort','metric','candidate_parent_mean','reference_parent_mean','mean','median','mean_relative_delta_percent','positive_count','negative_count','zero_count','noise_disagreement_parents','mean_delta_lower','mean_delta_upper','mean_delta_global_lower','mean_delta_global_upper','leave_one_parent_out']})
res=dict(status='PASS',scope='Read-only saved scalar cross-check, no geometry or model calls',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checked_contrast_metric_groups=len(checks),source_hashes={f:hashlib.sha256((D/f).read_bytes()).hexdigest() for f in ['contrast_distributions.csv','paired_parent.csv','paired_noise.csv','diversity_contrasts.csv']},checks=checks,summary=summary)
(R/'primary_comparison_scoring_review.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
md="""# 主比较的独立数字复核

已从 paired_noise.csv 的两侧数值重新相减，先均值噪声、再父等权，对照 paired_parent.csv 和 contrast_distributions.csv；仅读取已保存标量，不重评分。所核验主比较及指标分组全部一致。完整核验键与源哈希见 primary_comparison_scoring_review.json。

与主会话已读结论没有数值分歧，补充下列解释边界：

- P1 ALL，C1000−B1000 的 E 均值为 +0.31736 个百分点，但仅 2 父正、4 父负、2 父零，中位为 −0.01258 个百分点。去掉 f596 后变为 −0.15368 个百分点；这只是描述性敏感性，不能删除该父另作结论。OLD E +0.65858 个百分点、NEW −0.02386 个百分点，不能说旧/新噪声都改善。全参考可见域保守区间仍应保留。
- C1000−B1000 的 I 总值增加 +0.09308 个百分点，其中 cross +0.08087、dup −0.02136；I 是集合并，二项不能直接相加。该结果不支持把代价统一称为重复覆盖增多。相应 FF 受影响比例下降 0.71114 个百分点（7/8 父降低），CF proper 却增加 0.13690 个百分点；不能用 FF 下降替代条件接口改善。
- B1000−A1000 的 S 均值仅 +0.08451 个百分点，中位 −0.02460 个百分点，3 正、4 负、1 零；OLD −0.75249，NEW +0.92151 个百分点。去掉 bdd 后 S 均值 −0.45739 个百分点。微小总体正均值不是普遍增益。
- B1000−A1000 的 Dsym −0.55358% 来自 F→GT −1.45028%，GT→F 反而 +0.20107%；8 父内部都有噪声方向不一致。B 相对 A 的 FF 受影响比例 +0.79532 个百分点，7/8 父上升。双向均值下降不能写成所有距离/局部缺陷均改善。
- P2 C1000−B1000 的 I 均值降低 0.10681 个百分点，但 3 个条件增加、仅 back2 降低；四个手工条件也不是四个独立来源父。P2 没有 S/E/GT 距离资格。两比较的精确坐标 C–F 共享边均为零，组件统计未变，不能从视觉接近或 S 变化推断焊接改善。

S/E 数值越高越好，I/距离/受影响面比例越低越好；计数仅表原差分方向。P1 source GT 是一个来源形状，不是开放式补全的唯一正确目标。LOPO、噪声差异和 AI 图审均为描述证据，不引入晋级门槛或人类可用率。
"""
(R/'primary_comparison_scoring_review.md').write_text(md,encoding='utf8')
print(json.dumps({'status':'PASS','groups':len(checks),'summary':str(R/'primary_comparison_scoring_review.json')}))
