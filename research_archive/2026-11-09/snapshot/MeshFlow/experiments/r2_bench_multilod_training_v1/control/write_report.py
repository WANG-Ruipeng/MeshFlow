"""Compose the final report exclusively from completed logs and actual review rows."""
from pathlib import Path
import csv
import hashlib
import json

OUT=Path('/mnt/e/MeshFlow/experiments/r2_bench_multilod_training_v1')
MODELS=('EXPOSURE','PLAIN1500','MULTILOD1500')
STAGES=('RAW','BASE_POST','CLEAN_FINAL_POST')

def rows(p):
    with (OUT/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def table(headers, data):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in data]])+'\n'

def num(r,key,pct=False):
    if not r.get(key):return 'NA'
    return f"{float(r[key])*(100 if pct else 1):.3f}"

def main():
    vpath=OUT/'visual_review/summary_v1/completion.json'
    assert json.loads(vpath.read_text())['status']=='COMPLETE'
    metrics=rows('results/report_metrics_v1/primary_parent_macro.csv')
    counts=rows('visual_review/summary_v1/five_axis_and_joint_counts.csv')
    transitions=rows('visual_review/summary_v1/paired_request_transitions.csv')
    observed=rows('visual_review/summary_v1/actual_observations.csv')
    historical=rows('visual_review/summary_v1/baseline_review_differences.csv')
    # All observations are first read after the three actual-view shards pass validation.
    def count(label,stage,axis,value,group='ALL'):
        rr=[r for r in counts if r['label']==label and r['stage']==stage and r['report_group']==group and r['axis']==axis and r['value']==value]
        assert len(rr)==1,(label,stage,axis,value,group)
        return int(rr[0]['count'])
    joint_axis=next(r['axis'] for r in counts if r['axis'] not in ('overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold'))
    parts=['''# Bench 固定 C 多 LOD 增强：两臂各 1500 步实测

本轮完成了数据接入、两条真实训练、92 份新生成、统一后处理、完整177字段评分和138页实际图审。**多LOD有局部接口与部分几何代价的改善，但没有形成能够替换EXPOSURE的全面升级。** 对等预算普通续训而言，FINAL合法接口输出从17/46增加到21/46，CF降低0.905个百分点，小角面降低2.335个百分点；同时FF增加0.360个百分点，有效支持域I从2.506%升到6.344%，新增接口没有覆盖DEV或留出父。实际图审的三轴交集，RAW为EXPOSURE/PLAIN/MULTILOD=33/35/34，FINAL为15/17/16，增强没有超过普通续训。普通续训也不能因I较低直接晋级：部分低分伴随大量表面被后处理删除。

本轮检验的是“同一物理C、同一批父对象的合格低面数目标，混入训练是否改善生成”。不是扩独立父，不是重新三角化C，也不是SEAM或预测头实验。固定任务到此结束；未延训、未改默认、未清理、未commit/push。

## 1. 数据究竟增强了多少

从EXPOSURE实际训练的30父/30条物理C出发，原自然N为128—776、K为2—84。独立CPU数据阶段只增加9份合格LOW，总39个几何版本，新增独立父和物理C均为0。每父最多一份LOW；无LOW的父继续使用原版。9父受N128下限阻塞、1父减面不足、11候选未通过质量准入，没有为了凑数放松检查。

参考官方MeshFlow离线QEM及几何筛选思路，但增加了本任务的固定C合同：C坐标、来源顶点身份/角序和原C/F接口邻面不变，只简化free区域。不是声称逐行复制官方预处理。LOW入选版本的面积比例为0.997632—1.000127，最大双向有限表面探针偏差为0.149034%L；这是每面13探针的采样检查，不能当作连续Hausdorff上界。9份均完成实际AI_PROXY三视图检查，未新增检测到的非法CF/FF。

数据阶段共记录41条实际简化路径。v1额外冻结所有锐边端点导致大多数父无法减面；独立engineering_v2取消额外的非C锐边点锁定，允许自由端并入坐标不动的固定端，质量门槛不变。原v1、所有失败与原数据保留。详见[数据报告](../r2_bench_multilod_data_v1/REPORT.md)。

| 父ID前缀 | 原N→LOW N | K |
| --- | --- | --- |
| 35948 | 215→173 | 2 |
| 4cd23 | 256→206 | 2 |
| 943dd | 256→206 | 24 |
| b6b10 | 256→206 | 2 |
| c97683 | 176→142 | 2 |
| ef207 | 248→200 | 3 |
| 28b9 | 384→352 | 10 |
| bcb4 | 520→416 | 2 |
| 77aa | 776→622 | 68 |

两臂父曝光完全一致：OLD9000、FOCUS1500、HIGH1500，共12000实例。MULTILOD实际LOW为1596/12000，即13.30%，其余为NATURAL；FOCUS a8/27没有LOW。不能把39版本描述成39个独立训练对象，也不能声称全池每次都有增强。

## 2. 固定训练与配对合同

同一保留EXPOSURE live生成器、匹配AdamW和RNG分叉，从global25500/Native13500起，每臂新增1500步，固定终点global27000/Native15000。不是从官方EMA重新训练，不恢复其他分支动量，不挑best checkpoint。

保留Native/T1＋role＋Geo、free-only FM、先OT后HYBRID0.25、batch8/micro1、BF16前向/FP32参数与积分、lr1e-5、AdamW betas(0.9,0.95)、wd0、clip1、无EMA。CUBLAS确定性配置、TF32关闭、SDPA math及严格确定性运行沿用。无新参数、新loss、SEAM槽或预测头调用。

PLAIN保持原目标，MULTILOD在同父合法版本中按完整CLEAN/X_WIDTH对交替NATURAL/LOW。两臂12000实例的父、条件、增强参数与随机流身份逐项配对；N变化时重新构造实际噪声、mask、OT匹配，不能把不同大小数组称为逐元素相同噪声。没有复用历史训练噪声来冒充新训练。

原OLD/FOCUS/HIGH的组MSE系数固定。若原微批计数为c_i、gain为q_i、分母为D_old，则a_g=Σ(c_i q_i)/D_old；本臂实际计数c'_i=9(N_i−K_i)，gain'_i=a_g D_arm/Σ_g c'_i。因此组权不随LOW面数改变，组内仍按实际free坐标数平均，不等于每父梯度等权。PLAIN与原gain最大误差4.44e-16。39版本×两种增强共78次实际CPU OT合同核验通过。

训练曲线见[PNG](training_summary/training_loss.png)与[PDF](training_summary/training_loss.pdf)。两臂无数值发散，固定预算内没有清楚的持续下降优势；不同目标上的loss不作为生成能力结论，也不宣称已到收敛平台。完整实现和系数定义见[METHODS](METHODS.md)。

## 3. 评价口径

复用原23父、46请求，C、N、两份噪声、查询、相机和位置截面固定。两新模型各46份Euler50 RAW；92份RAW全部齐全后才执行D0。后端完全一致：单次官方Bench D0→GEO_C/来源定向→HINGE_OPP015→ALPHA→HOLE。保留RAW、BASE_POST和CLEAN_FINAL_POST（下文FINAL）。EXPOSURE138行旧评分只读复用，新增276行，共414行/177指标。

数值按噪声→条件→父→组聚合、父等权。CF/FF表示非法面对影响的free面比例，不是所有面对的相交概率。I采用原ConflictCov_C；主表限定三模型三阶段共同PASS的19父/38请求。28b9、945、abf、af为LOW_SUPPORT，原值和原因保留，不补0。NEW_WIDE的I只有2/4父，HIGH_TRAIN为NA。RAW soup单面索引孤岛与POST共边孤岛不是同一拓扑阶段，不能直接把其差值当“修好碎面数”。

C坐标检查与真实索引共边分别记录。free总面积比=当前阶段free总面积/该模型同请求RAW的free总面积，含变形、焊接与新增桥面；它不是原表面覆盖保留率。表面破损需要同时查看删面账本和图片。来源完整款式不作为唯一正确答案。以下数值不包含统计显著性或新父泛化认证。
''']
    allrows=[r for r in metrics if r['scope']=='ALL']
    cols=[('CF%','CF_illegal_affected_free_fraction',True),('FF%','FF_illegal_affected_free_fraction',True),('I%','I_common_PASS',True),('小角<5°%','free_minimum_angle_lt5_fraction',True),('非流形边均数','nonmanifold_edges',False),('单面孤岛均数','hinge_diag_isolated_1face_edge_components',False),('free面积/RAW%','free_area_ratio_to_RAW',True)]
    parts.append('## 4. 全组三阶段数值\n\n'+table(['阶段','模型',*[x[0] for x in cols],'合法CF输出'],[[r['stage'],r['label'],*[num(r,k,p) for _,k,p in cols],f"{r['legal_CF_output_count']}/46"] for r in allrows]))
    parts.append('''
FINAL MULTILOD相对PLAIN：合法CF输出增加4份且0份丢失，但平均合法边长度2.626→2.479模型坐标单位，方向一致可达free面积10.013%→9.797%，因此“更多输出出现至少一条接口”不等于接口普遍更长或连接到更大主体。单面孤岛28.065→27.783，差异很小；不能说碎面已解决。原RAW槽删除均数218.109→213.587、free总面积比55.290%→57.253%，处理损失有局部减轻；总面积比本身不能定位保住了哪些原表面。

新增接口身份为943/n0、943/n1、9e/n0（OLD）和abf/n1（HIGH_TRAIN），只在熟悉父出现；其中只有943实际有LOW。DEV、HOLDOUT与HIGH_HOLDOUT均未新增合法C/F共边，不能把本轮记成接续泛化成功。

## 5. 分组代价，不能用熟悉父收益抵消DEV
''')
    grouped=[r for r in metrics if r['scope']=='report_group' and r['stage']=='CLEAN_FINAL_POST']
    parts.append(table(['组','模型','父数','CF%','FF%','I%','I父数','面积/RAW%','合法CF输出'],[[r['group'],r['label'],r['parents'],num(r,'CF_illegal_affected_free_fraction',True),num(r,'FF_illegal_affected_free_fraction',True),num(r,'I_common_PASS',True),r['I_common_PASS__parents'],num(r,'free_area_ratio_to_RAW',True),f"{r['legal_CF_output_count']}/{r['requests']}"] for r in grouped]))
    parts.append('''
DEV相对PLAIN，MULTILOD的FINAL CF从3.996%升到8.721%，FF从8.960%升到12.509%，I从2.942%升到16.821%；HIGH_HOLDOUT的FF与I也升高。另一方面，PLAIN相对EXPOSURE同样有代价：全组小角44.706%→47.652%，free总面积比58.814%→55.290%。不能只从这三模型中挑各项最有利数字，拼成不存在的流程。

## 6. 实际图审与逐请求转移

三位AI_PROXY分别逐张打开不重叠请求的全部三阶段对比页，138页/414个模型阶段单元，均检查两斜视图与原前/侧截面。模型名可见，属于非盲AI代理评阅，不是独立人类认证。五轴独立记录，不由分数推标签；UNCERTAIN保留。相同EXPOSURE资产也重新观看，不复制旧标签。

严格三轴交集J=整体REASONABLE ∩ C用途PLAUSIBLE ∩ 支撑PRESENT；这是功能/结构观察，不包含“无裂缝、无相交、水密”。有确定负项记NO，无确定负项但有不确定项记UNCERTAIN，只有三项确定正面才记YES。
''')
    vt=[]
    for stage in STAGES:
        for label in MODELS:
            vt.append([stage,label,count(label,stage,'overall_shape','REASONABLE'),count(label,stage,'C_use','PLAUSIBLE'),count(label,stage,'support_retention','PRESENT'),count(label,stage,joint_axis,'YES'),count(label,stage,joint_axis,'UNCERTAIN'),count(label,stage,'visible_fragment_or_gap','YES'),count(label,stage,'layering_or_fold','YES')])
    parts.append(table(['阶段','模型','合理主体/46','合理采用/46','支撑保留/46','J YES/46','J不确定/46','可见碎片或裂隙/46','可见重层折叠/46'],vt))
    gt=[]
    for stage in STAGES:
        for group in ('OLD','DEV','FOCUS','HOLDOUT','HIGH_TRAIN','HIGH_HOLDOUT','NEW_WIDE_TRAIN'):
            cells=[r for r in counts if r['stage']==stage and r['report_group']==group and r['axis']==joint_axis and r['value']=='YES']
            if cells:gt.append([stage,group,cells[0]['requests'],*[count(l,stage,joint_axis,'YES',group) for l in MODELS]])
    parts.append('\n'+table(['阶段','组','请求数','EXPOSURE J','PLAIN J','MULTILOD J'],gt))
    tt=[]
    for stage in STAGES:
        for ref,cand in (('PLAIN1500','MULTILOD1500'),('EXPOSURE','PLAIN1500'),('EXPOSURE','MULTILOD1500')):
            subset=[r for r in transitions if r['baseline']==ref and r['candidate']==cand and r['stage']==stage and r['axis']==joint_axis]
            assert len(subset)==46
            gain=[r for r in subset if r['from_value']!='YES' and r['to_value']=='YES']
            loss=[r for r in subset if r['from_value']=='YES' and r['to_value']!='YES']
            tt.append([stage,ref+'→'+cand,len(gain),len(loss),'; '.join(r['request_id'].split('__')[0][:6]+'/'+r['request_id'].split('__')[-1] for r in gain) or '—','; '.join(r['request_id'].split('__')[0][:6]+'/'+r['request_id'].split('__')[-1] for r in loss) or '—'])
    parts.append('\n'+table(['阶段','比较','进入J数','离开J数','进入身份','离开身份'],tt))
    parts.append('\nMULTILOD相对PLAIN在三阶段均为2d/n0从NO→YES，fd355/n0从YES→NO，617/n0从YES→UNCERTAIN。DEV的1/8总数未变，但成功身份更换；不能称跨父净改善。进入/离开J含UNCERTAIN边界变化，不能都叫“明确失败修复/明确退化”；完整三态转移见CSV。三模型RAW支撑均PRESENT46/46，但这不认证无缺陷；FINAL分别只剩22/19/21份支撑PRESENT。增强没有解除现有后端的广泛支撑损失。\n')
    dif=[]
    for stage in STAGES:
        for axis in ('overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold',joint_axis):
            rr=[r for r in historical if r['stage']==stage and r['axis']==axis]
            dif.append([stage,axis,sum(r['changed']=='True' for r in rr),len(rr)])
    parts.append('\n同一EXPOSURE资产的历史/本轮评阅差异也完整保留；它表示评阅不确定性，不能解释为模型变化。历史主报告数字不被改写。本轮配对结论统一使用本轮同页判断。\n\n'+table(['阶段','轴','相对历史标签改变数','单元数'],dif))
    parts.append('''
## 7. 两个尾部与两个接口正例

**f495/n1：低I可能伴随删掉主体。** PLAIN的FINAL I为1.465%、FF为0，但RAW到FINAL的free总面积比只剩13.728%，原155张free槽删除106张；图中座面/支撑大幅变成窄残条。MULTILOD总面积比56.498%、删除64张，但I为67.822%、FF为19.780%，灰座与蓝C仍存在明显层叠关系。这里不能把PLAIN低I称为全面质量胜出，也不能因MULTILOD保面更多忽略冲突。

**2d/n0：采用与冲突可能同时增加。** MULTILOD的蓝C更明显参与座面，保留宽端架，free总面积比89.692%，但RAW I已有26.611%，FINAL达40.625%；PLAIN FINAL I为0、free总面积比62.500%，侧截面仍有灰座与蓝C高度关系不理想的现象。生成组织、局部冲突和后端表面损失必须分开解释。

上述两请求约占MULTILOD相对PLAIN全组I净增量的73.35%。这是尾部分布描述，不是剔除它们的理由。其余请求、全部尾部和NA仍保留。

**943/n0与9e/n0：新增接口不自动等于视觉跨级。** 943三模型都呈现相近的完整背板/端架组织，MULTILOD出现合法共边是有意义的局部几何收益，但图像没有明显的整体级别跃迁。9e三模型的蓝色曲座用途较明确，灰色端架仍有明显破损/残片；增加接口未消除这个视觉问题。以上为主会话额外查看的补充说明，不叠加进414单元的正式计数。

[完整图集与原位置截面](gallery/index.html)。每页保留对应输入、模型、N/K、阶段和hash，不只展示精选成功例。

## 8. 资源、真实调用与工程记录

Windows/WSL实测RTX5080，16303MiB；24 CPU，可用RAM约60GiB。先启动一条已授权正式训练，观察前25次真实更新（含高N776、反向和AdamW），随后按空闲显存与利用率逐步启动第二条。保留至少2048MiB余量，使用20%峰值系数；没有增加合成GPU探针。两臂并行、各自模型/优化器/随机流/输出目录，不改变科学配置来适配并行。

单worker实测allocator峰值allocated约3.76GiB、reserved约4.07GiB；用于调度的保守完整进程估计4800MiB。WDDM未提供可靠逐进程显存，不能把allocator或设备总占用当单worker完整峰值。并行期间设备总占用峰值约13147MiB，包含用户后台。训练墙钟PLAIN2644.44秒、MULTILOD2647.85秒，时间互相重叠，不能与旧孤立运行直接比较吞吐。

| 实际阶段 | PLAIN | MULTILOD | 口径 |
| --- | --- | --- | --- |
| optimizer更新 | 1500 | 1500 | 无失败/重算 |
| 训练前向/反向 | 12000/12000 | 12000/12000 | 累计真实调用 |
| Euler50 RAW | 46 | 46 | 各2300逻辑向量场调用 |
| 采样墙钟秒 | 176.90 | 174.20 | 有并行竞争 |
| 单次D0/GEO | 46 | 46 | 92 RAW barrier后执行 |
| D0/GEO墙钟秒 | 58.81 | 49.52 | 实际分阶段计时 |
| 预测头/额外probe | 0 | 0 | 无新增头训练 |

CPU两worker的RAW92评分约261.64秒；两臂各46条POST链约310.95/335.12秒。138页渲染约41.25秒。CPU/GPU有独立阶段重叠，以上时间不可相加当实验总墙钟。图审人工式查看时间另计，不混入模型吞吐。

正式评分冻结前修复了一处工程序列化问题：私有克隆函数不能直接作为ProcessPool spawn入口，改为新模块顶层wrapper；实际spawn合同测试通过后才开始评分。原评估源码未改，修复没有模型调用或失败评分重算。正式科学阶段失败、静默重试、额外训练均0。

## 9. 决策与可复查交付

保留多LOD数据构造器、9份合格版本、两臂真实终点与完整记录，但**不把MULTILOD1500或PLAIN1500替换为新的统一默认/全面SOTA**。本轮只支持有限的接口与部分几何收益，DEV冲突、支撑损失与碎面问题仍在。9/30父、13.3%曝光是本次增强的真实覆盖，不能据此否定所有官方式增强，也不能未经检验继续加大比例或延长。

按照“先增强、再看生成、最后考虑预测头”的顺序，增强训练阶段已完成；**预测头尚未启动**。若继续结构预测路线，应把本轮作为独立数据对照保留，先定义C-only可获得的关系与推理解码合同，不把本轮未解决的问题写成已由增强修复。

- [完整逐输出177字段](results/per_output.csv)、[父等权主表](results/report_metrics_v1/primary_parent_macro.csv)、[逐父配对](results/report_metrics_v1/paired_primary_per_parent.csv)、[尾部分布](results/report_metrics_v1/tail_distribution.csv)。
- [实际414单元图审](visual_review/summary_v1/actual_observations.csv)、[逐请求转移](visual_review/summary_v1/paired_request_transitions.csv)、[历史EXPOSURE评阅差异](visual_review/summary_v1/baseline_review_differences.csv)。
- [训练账本](training_summary/complete.json)、[PLAIN完整终点](training/PLAIN1500/checkpoint_step1500.pt)、[MULTILOD完整终点](training/MULTILOD1500/checkpoint_step1500.pt)。对应generator_step1500.pt为推理导出，完整终点含匹配AdamW/游标/随机状态。
- [冻结注册](training_registration.json)、[恢复入口](RECOVERY.md)、[文件身份终检](closeout/identity_audit.json)、[实际源码及hash](closeout/source_hashes.json)、[总完成回执](completion.json)。

所有旧评分、原数据和官方资产保持不变。任务完成后停止。
''')
    report='\n'.join(parts)
    path=OUT/'REPORT.md'
    if path.exists():
        assert path.read_text(encoding='utf-8')==report,'Report exists with different content; require explicit versioned correction'
    else:path.write_text(report,encoding='utf-8')
    print(json.dumps(dict(status='REPORT_WRITTEN',bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),joint_axis=joint_axis),ensure_ascii=False))

if __name__=='__main__':main()
