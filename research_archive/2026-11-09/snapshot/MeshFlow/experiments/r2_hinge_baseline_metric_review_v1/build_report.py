"""Build a compact decision report from independently recomputed tables."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path

OUT=Path(__file__).resolve().parent
def load(name):
    with (OUT/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
S={(r['group'],r['metric']):r for r in load('summary_all_metrics.csv')}
ST=['POST','POST_HINGE015','POST_HINGE025']
def val(group,key,stage,stat='mean'):
    return float(S[group,key][stage+'_'+stat])
def formatted(group,key,stage,scale=1,dec=3):
    return f'{val(group,key,stage)*scale:.{dec}f}'
def table(group,keys):
    lines=['| 指标 | 原 POST | HINGE 1.5%L | HINGE 2.5%L |','|---|---:|---:|---:|']
    for name,key,scale,dec in keys:
        lines.append('| '+name+' | '+' | '.join(formatted(group,key,s,scale,dec) for s in ST)+' |')
    return '\n'.join(lines)
KEYS=[
('A-VMR，严格模型单位 1e-6 % ↑','A_VMR__raw_tau_1e-6',100,3),
('PMR，严格模型单位 1e-6 % ↑','PMR__raw_tau_1e-6',100,2),
('合法 C/F 共边条数 ↑','C_F_valid_edge_count',1,3),
('合法 C/F 共边长度 ↑','C_F_valid_edge_total_length',1,3),
('可达 bulk 面积 % ↑','free_only_reached_bulk_area_fraction',100,3),
('方向一致可达 free 面积 % ↑','opposite_winding_reached_free_area_fraction',100,3),
('CF 非法影响 free 面比例 % ↓','CF_illegal_affected_free_fraction',100,3),
('FF 非法影响 free 面比例 % ↓','FF_illegal_affected_free_fraction',100,3),
('I，C 冲突覆盖 % ↓','ConflictCov_C',100,5),
('I cross % ↓','ConflictCov_C_cross',100,5),
('I dup % ↓','ConflictCov_C_dup',100,5),
('free 小角 <5° % ↓','free_minimum_angle_lt5_fraction',100,3),
('free 最小角 p05，度 ↑','free_angle_p05_degrees',1,3),
('free q>10 % ↓','free_q_gt10_fraction',100,3),
('free q 均值 ↓','free_q_mean',1,4),
('近退化 free % ↓','free_near_degenerate_fraction',100,3),
('精确零面积 free % ↓','free_exact_zero_area_fraction',100,3),
('精确重复 free % ↓','free_exact_duplicate_excess_fraction',100,3),
('非流形边 ↓','nonmanifold_edges',1,3),
('精确坐标图同向双面边 ↓','EXACT_COORD_same_direction_two_face_edges',1,3),
('实际共边图单面分量 ↓','hinge_diag_isolated_1face_unfiltered_edge_components',1,3),
('仅共点关联 C 的 free 面积 %','hinge_diag_vertex_attached_only_unfiltered_free_area_fraction',100,3),
('完全与 C 不连通的 free 面积 %','hinge_diag_fully_C_disconnected_free_area_fraction',100,3),
('bulk 内部合法共边数 ↑','hinge_diag_bulk_only_valid_shared_edge_count',1,3),
('开放边数，描述量','boundary_edges',1,3),
('保留 free 面数','actual_free_faces',1,3),
('累计从 RAW 删除 free 面数','deleted_free_faces',1,3),
('C 精确保留 %','C_preserved_exact',100,1)]
lines=['# HINGE 与 S2000 完整指标复算及保留判断','',
'**建议保留 HINGE 1.5%L 为可选后处理候选，2.5%L 保留为压力档；当前证据不足以直接替换统一 baseline。** 两档在原 S2000 上降低 FF 和实际单面分量，增加合法共边，且没有额外删面；但 A-VMR/PMR 已饱和且不变，DEV 主体接续仍未达到 30% 门槛，同向双面边有少量增加。','',
'本次只用同一 S2000 生成器的原 POST 与两个独立 HINGE 终点，不混入 CLEAN、SMOOTH 或 BOUND。固定 28 请求：4 个 TRAIN 父×4 噪声，3 个已多次使用的 DEV 父×4 噪声。完整复算原124数值字段和11拓扑扩展，共135字段、84行、7560个逐请求增量；没有空缺数值，没有重新训练、生成、D0或几何重评分。','',
'## 完整关键结果','',
'下表为逐输出宏均值，TRAIN 每父4噪声、DEV 每父4噪声，因此也等于各组内先噪声后父等权。比例统一显示百分数，长度为原模型单位；原始 CSV 保持0至1比例和原单位。CF/FF分母是实际保留free面数，I是冻结C查询上的冲突覆盖，不是整椅自交面积。','',
'### TRAIN 16 请求','',table('TRAIN',KEYS),'',
'### DEV 12 请求','',table('DEV',KEYS),'',
'全部135项（包括8档A-VMR/PMR、边界距离、原始/精确/实际拓扑、分母和UNKNOWN字段）见[可筛选完整表](index.html)与[完整CSV](summary_all_metrics.csv)。单面分量是共边图概念，有些仍通过顶点与其他面连接，不应一律称为物理悬浮片。','',
'## 配对改善及代价','',
'| 分组和指标 | 1.5%L 改善/持平/退化 | 2.5%L 改善/持平/退化 |','|---|---:|---:|']
for g in ('TRAIN','DEV'):
    for name,key in [('FF','FF_illegal_affected_free_fraction'),('CF','CF_illegal_affected_free_fraction'),('I','ConflictCov_C'),('小角比例','free_minimum_angle_lt5_fraction'),('实际单面分量','hinge_diag_isolated_1face_unfiltered_edge_components'),('q均值','free_q_mean'),('同向双面边','EXACT_COORD_same_direction_two_face_edges')]:
        r=S[g,key]
        cs=['/'.join(r[s+'_'+c] for c in ('improved','unchanged','worsened')) for s in ST[1:]]
        lines.append('| '+g+' '+name+' | '+' | '.join(cs)+' |')
lines += ['',
'- 两档 FF、CF、小角比例、q>10比例、实际单面分量没有逐请求恶化。全部16个A-VMR/PMR字段、边界距离、近退化、零面积、重复面、非流形边数和删面数逐例不变。匹配不变不意味着连接不变。','- I唯一上升请求为 `610/n1`，增加0.048828个百分点，DEV均值增加0.004069个百分点；变化来自cross覆盖，dup不变。','- 1.5%L 新增同向双面边的请求是 `2d/n3`、`608/n0`、`85/n3`，各1条；2.5%L另在`8c/n2`增加1条。旧合法边保留并不保证新增边绕序全部相反。这次没有额外重定向来隐藏该代价。','- q均值存在少量退化，1.5%L最大单例`608/n2` +0.009010，2.5%L最大单例`3f/n1` +0.020713；后者最小角p05还下降0.106664°。这不改变“新增/恶化<5°面”守卫通过，但排除了“全部形状指标无损”的表述。','',
'## 实际新增连接与分母核账','',
'| 28请求合计 | HINGE 1.5%L | HINGE 2.5%L |','|---|---:|---:|',
'| 接受簇合并次数 | 84 | 99 |','| 新增合法 FF 共边 | 108 | 129 |','| 新增合法 CF 共边 | 3 | 3 |','| 消除原非法 FF 面对 | 22 | 36 |','| 消除原非法 CF 面对 | 4 | 4 |','| 丢失原合法共边 | 0 | 0 |','| 新增非法 CF/FF 面对 | 0 | 0 |','| HINGE额外删除面 | 0 | 0 |','| 新增可达 bulk 面 | 52 | 56 |','',
'FF下降伴随非法面对真实消除，且保留free面数固定，因此不是删面或分母膨胀造成的表观改善。TRAIN ≥30% bulk可达为13/16→14/16→14/16，DEV均为0/12。','',
'原4624条合法边全部保留，旧边绕序关系变化为0。1.5%L新增111条合法边，其中108条反向、3条同向；2.5%L新增132条，其中128条反向、4条同向。两档的合法边增量含CF与FF，不能全部称为方向一致接续。','',
'`73/n3` 两档新增7条合法FF边和32张可达bulk面，可达比例14.9935%→52.5957%；固定原bulk面积分母仍为52.8381%，证明存在真实连通增量。DEV的`608/n3`和`610/n0`在2.5%L下FF分别下降5.30、4.48个百分点，但并未接入足够大的C可达主体。','',
'HINGE改变局部坐标和面积：平均free总面积相对变化约+0.1602%/+0.1921%，个别原本极小三角形的面积变化可较大。仅共点挂接面积比例的小变化可能含面积重分配，不能全部记作新增连通。','',
'## 图审与结论边界','',
'既有非盲AI_PROXY截面复查支持局部闭口：S2000 `8c/n0`、CLEAN `73/n0`、SMOOTH `85/n2`都有可见改善。但本次数值主比较严格只使用S2000。基线`8c/n3`两档都未接受修改，旧图审中的疑似闭口不算几何正证据。`608/n3`指定缺口仍存在。','',
'保留的理由是本面板内低成本、无额外删面的局部共边与FF收益。不能称为完整接续方案、普遍视觉修复或独立泛化提升；DEV只有三个反复观察的父。1.5%L保持预定主档，不能看完结果后把2.5%L改称预注册主结果。没有切换默认、合入代码或自动扩大阈值。','',
'## 指标适用性和复算入口','',
'GT重建、S和E保持原NA：本任务没有唯一目标款式、没有登记必须接续的接口、也没有外露要求。FID/KID/TMD等未在该冻结面板实现的指标不补造；135字段完整不等于35类指标全部实施。RAW由同一生成结果继承，HINGE仅改变POST，不能把POST收益记在RAW生成器上。','',
'既有HINGE计时为56格、2个CPU worker合计墙钟212.2543秒，范围包含绑定及评分，不能当作纯解码延迟或与串行生成直接相加。没有新增GPU前向。','',
'[字段定义与适用性](metric_definitions.json) · [逐输出](per_output.csv) · [逐噪声](paired_per_noise.csv) · [逐父](paired_per_parent.csv) · [全部退化项](all_regressions.csv) · [评分状态](status_coverage.csv)。','',
'独立实现从原504行表复算，核对7560个逐噪声增量、405组汇总、945组逐父汇总及116条描述性退化记录，差异为0；135字段无缺失。1532个登记资产和源码hash一致。见[独立验证](independent_validation.json)。','',
'运行 `recompute.py` 可重新读取原缓存、核对冻结hash并复算，再运行 `build_report.py` 更新本独立目录。原文件不写入。资源为单汇总进程加两个独立只读核验，零GPU调用；已有评分缓存不重复检测。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('REPORT.md written; all tables derived from recomputed summary')
