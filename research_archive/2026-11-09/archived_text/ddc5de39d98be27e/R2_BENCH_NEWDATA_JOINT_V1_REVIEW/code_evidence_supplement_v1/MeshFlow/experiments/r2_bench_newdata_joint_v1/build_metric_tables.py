"""Write display tables from frozen summaries; no model or scoring calls."""
from pathlib import Path
import json,csv,hashlib
O=Path(__file__).resolve().parent
s=json.loads((O/'results/key_metrics_summary.json').read_text(encoding='utf-8'))
axes=[('CF','CF_illegal_affected_free_fraction',100),('FF','FF_illegal_affected_free_fraction',100),
('small_angle_lt5','free_minimum_angle_lt5_fraction',100),('free_faces','actual_free_faces',1),
('deleted_free_faces','deleted_free_faces',1),('nonmanifold_edges','nonmanifold_edges',1),
('C_F_valid_edges','C_F_valid_edge_count',1),('C_F_valid_length','C_F_valid_edge_total_length',1),
('bulk_reached_area','free_only_reached_bulk_area_fraction',100),('seam_bulk_edges','legal_seam_bulk_edge_count',1),
('isolated_1face_components','hinge_diag_isolated_1face_edge_components',1),('free_area','free_total_surface_area',1)]
p=['# 三阶段完整重点指标','',
'由冻结168行评分的36组汇总直接列出。百分数为CF/FF/小角/I/可达面积；长度、面积为模型单位。先噪声、再同父条件、最后父等权。NA保留。开放C的共边和可达性仅描述，不等同用途成功；闭合C无接口要求。RAW的连续角点表示与POST索引拓扑不同，应分阶段比较。',
'','| group | stage | model | requests | parents | '+' | '.join(x[0] for x in axes)+' | I_PASS | I_PASS_requests |','| '+' | '.join(['---']*(7+len(axes)))+' |']
def f(x,scale=1):return 'NA' if x is None else f'{x*scale:.4f}'
for r in s['summaries']:
 p.append('| '+' | '.join([r['group'],r['stage'],r['label'],str(r['requests']),str(r['parents'])]+[f(r['parent_macro'][k],m) for _,k,m in axes]+[f(r['I_pass_parent_macro'],100),str(r['I_pass_support']['requests'])])+' |')
p+=['','完整177评分字段、逐请求分母及失败/NA状态见 [per_output.csv](per_output.csv)。所有I均PASS，但不能把它等同whole-mesh Ri。新增桥面可使最终free面数与原预算减删面数略有差别。原始固定父面口径也保存在完整CSV，不以删面后分母替代所有统计。','',
'## 实际图审全类别','',
'非盲AI_PROXY；所有168 cells实际查看。以下逐输出计数与父等权比例同时保留，UNCERTAIN不丢弃。',
'','| model | stage | group | axis | value | count | requests | parent_macro_% |','| --- | --- | --- | --- | --- | ---: | ---: | ---: |']
v=O/'visual_review/aggregates/5cc5f7e7ba6271d6/summary.csv'
for r in csv.DictReader(v.open(encoding='utf-8-sig')):
 p.append('| '+' | '.join([r[k] for k in ('label','stage','eval_group','axis','value','count','output_denominator')]+[f(float(r['parent_macro_fraction']),100)])+' |')
out=O/'results/METRIC_TABLES.md'
text='\n'.join(p)+'\n'
if out.exists():assert out.read_text(encoding='utf-8')==text
else:out.write_text(text,encoding='utf-8')
print(json.dumps({'status':'COMPLETE','rows':len(s['summaries']),'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
