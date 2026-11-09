"""Combine actual reviews and completed tables; no model, scoring or geometry calls."""
import csv, hashlib, json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_seam_pool47_couple_v1'
LABELS=('B_LONG_LAYOUT','PLAIN3000','COUPLE3000')
STAGES=('RAW','POST','BOUND015')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def table(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writecsv(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def mean(rr,k):return statistics.mean(float(r[k]) for r in rr)
def md(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    protocol=read(OUT/'protocol.json');qs=protocol['evaluation']['requests'];qmap={q['request_id']:q for q in qs}
    comp=read(OUT/'results/completion.json');assert comp['rows']==252 and comp['numeric_fields']==124
    reviews=[];sources={}
    for who,n in [('root',36),('binding',72),('review',72),('sampling',72)]:
        p=OUT/f'visual_review_endpoint_{who}.csv';rr=table(p);assert len(rr)==n
        reviews+=rr;sources[str(p)]=sha(p)
    expected={(q,l,s) for q in qmap for l in LABELS for s in STAGES}
    assert len(reviews)==252 and {(r['request_id'],r['label'],r['stage']) for r in reviews}==expected
    reviews.sort(key=lambda r:(list(qmap).index(r['request_id']),STAGES.index(r['stage']),LABELS.index(r['label'])))
    assert all(r['actually_viewed'].lower() in ('true','1','yes') and r['note'].strip() for r in reviews)
    assert len({r['page_path'] for r in reviews})==84
    writecsv(OUT/'visual_review_endpoint.csv',reviews)
    (OUT/'visual_review_endpoint.json').write_text(json.dumps(dict(status='COMPLETE_ACTUAL_AI_PROXY_REVIEW',rows=252,pages=84,
        blinding='Nonblind model names; observations are actual view-based, not metric-filled',sources=sources,
        limit='Finite views and central sections do not certify topology or absence of intersection.'),ensure_ascii=False,indent=2),encoding='utf-8')
    vr=[]
    flags=('body_complete','C_adopted_as_seat','obvious_fragments_or_folded_sheets','visible_cracks')
    for l in LABELS:
        for s in STAGES:
            for split in ('TRAIN','DEV'):
                rr=[r for r in reviews if r['label']==l and r['stage']==s and qmap[r['request_id']]['split']==split]
                row=dict(label=l,stage=s,split=split,n=len(rr))
                for f in flags:
                    for value in ('YES','NO','UNCERTAIN'):row[f+'_'+value]=sum(r[f]==value for r in rr)
                vr.append(row)
    writecsv(OUT/'results/visual_review_summary.csv',vr)
    data=table(OUT/'results/per_output.csv');keys=table(OUT/'results/key_results.csv');ledger=table(OUT/'results/binding_ledger.csv')
    grouped={(l,s,sp):[r for r in data if r['label']==l and r['stage']==s and r['split']==sp] for l in LABELS for s in STAGES for sp in ('TRAIN','DEV')}
    postrows=[];rawrows=[]
    for s in ('POST','BOUND015'):
        for sp in ('TRAIN','DEV'):
            for l in LABELS:
                rr=grouped[l,s,sp]
                postrows.append([s,sp,l,f"{sum(float(r['free_only_reached_bulk_area_fraction'])>=.3 for r in rr)}/{len(rr)}",
                    *[f'{100*mean(rr,k):.3f}' for k in ('free_only_reached_bulk_area_fraction','FF_illegal_affected_free_fraction','CF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction')],
                    f"{mean(rr,'actual_free_faces'):.2f}",f"{mean(rr,'deleted_free_faces'):.2f}"])
    for sp in ('TRAIN','DEV'):
        for l in LABELS:
            rr=grouped[l,'RAW',sp];rawrows.append([sp,l,*[f'{100*mean(rr,k):.3f}' for k in ('FF_illegal_affected_free_fraction','CF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction')]])
    bindrows=[];bindsumm=[]
    for l in LABELS:
        for sp in ('TRAIN','DEV'):
            rr=[r for r in ledger if r['label']==l and r['split']==sp]
            row=dict(label=l,split=sp,n=len(rr))
            for k in ('CF_lost_legal_edges','FF_lost_legal_edges','deleted_source_faces','CF_new_illegal_pairs','FF_new_illegal_pairs','FF_illegal_pairs_removed','accepted_tip','accepted_edge'):
                row[k]=sum(int(r[k]) for r in rr)
            bindsumm.append(row);bindrows.append(list(row.values()))
    writecsv(OUT/'results/binding_split_totals.csv',bindsumm)
    visualrows=[]
    for r in vr:
        if r['stage']!='POST':continue
        visualrows.append([r['split'],r['label'],r['n'],*[f"{r[f+'_YES']} / {r[f+'_UNCERTAIN']}" for f in flags]])
    ta=table(OUT/'training_analysis/first_last_all.csv');tg=table(OUT/'training_analysis/time_binned_geometry.csv')
    trows=[]
    for l in LABELS[1:]:
        for w in ('FIRST100','LAST100'):
            def metric(k):return next(float(r['mean']) for r in ta if r['arm']==l and r['window']==w and r['metric']==k)
            def geom(k):return next(float(r['velocity_mse']) for r in tg if r['arm']==l and r['window']==w and r['time_bin']=='-1' and r['region']==k)
            trows.append([l,w,f"{metric('FM'):.6f}",f"{geom('source_interface_third'):.6f}",f"{geom('bulk'):.6f}",f"{metric('coupling'):.6f}" if l=='COUPLE3000' else '未启用/未计算'])
    pairrows=[]
    for sp in ('TRAIN','DEV'):
        for s in ('POST','BOUND015'):
            a,b=grouped['PLAIN3000',s,sp],grouped['COUPLE3000',s,sp]
            pairrows.append([sp,s,*[f'{100*(mean(b,k)-mean(a,k)):+.3f}' for k in ('free_only_reached_bulk_area_fraction','FF_illegal_affected_free_fraction','CF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction')],f"{mean(b,'actual_free_faces')-mean(a,'actual_free_faces'):+.2f}"])
    report='''# R2_SEAM_POOL47_COUPLE_V1 实验收口报告

本轮三项工作已实际执行：保存输出的双档 CPU 绑定、纯 C 角部规则诊断，以及同 B_POOL47 起点的两臂各 3000 步训练和固定面板完整评价。

**主要结论：没有得到可直接晋级的 couple 增量。** 更长训练和 couple 都降低了训练接口误差，但原 POST 的 DEV bulk可达≥30%的请求仍是 0/12。相同 1.5%L 绑定后，普通续训在 `608/n3` 得到一份主体可达新正例，couple 没有复现；这是有界的局部正结果，不是普遍 DEV 接续能力。两条续训的 TRAIN FF 更低，但 DEV FF 相比同布局 B 起点更高。不能把“之前只欠训练”当作已经证实的唯一解释。

## 1. 冻结合同和实际预算

- TRAIN47，原几何/原 N/原来源 owner 训练布局；没有去重、混合新数据或加入 DEV。
- 同一 `B_POOL47_step1000.pt` 活生成器及其 AdamW/RNG，global 13000 开始；PLAIN3000 与 COUPLE3000 各到 global 16000。两臂都不是从另一臂继续。
- 共享 24000 实例 OT 缓存（3000×8），逐位置父对象、增强、OT、噪声、t 和角点排列一致；47 父实际曝光 508—512 次/父。两臂初始化与实际曝光流 hash 完全一致。
- 原 free-corner FM、OT 后 HYBRID0.25、batch8/micro1、BF16 前向/FP32 参数与积分。COUPLE 仅增加既有 A 的 λ=0.1 同源 seam-tip/bulk 一步终点 MSE；两侧反传，不改原定义。头冻结且未调用。
- 每臂实际 24000 前向、24000 反向、3000 AdamW 更新；无失败/重放。预定 step3000 是唯一主终点，1000/2000 只作恢复。
- 同 LONG C-only 推理布局、原 C/N 和固定噪声，7 父×4 噪声=28 请求/臂。先保存全部56 RAW，再做56次单次官方 D0、原 GEO_C 与来源角序定向。实际2800次生成器前向、56次D0、0头前向。
- 两终点112份 RAW/POST 原评分，另56份固定1.5%L绑定。加入只读复用的B起点后，保存252逐输出行、124数值字段和全部逐噪声/逐父配对；没有仅保留有利指标。
- 本轮不测试新占用头、GT推理布局、去重、额外训练轮数或新loss；未改默认模型、commit或push。

## 2. CPU 保存输出绑定与 DEV 几何诊断

在28份 `B_LONG_LAYOUT` 已保存 POST 上运行1.5%和2.0%L，共56格，零模型调用。使用真实GEO_C角点簇及原pre_geo全簇成员位移，保留C锚点、来源面身份和逐操作账本。

两档 TRAIN 主体可达≥30%均从12/16到14/16，宏均可达62.490%；DEV达到≥30%的请求均仍0/12，宏均可达0.391%。1.5%L的TRAIN FF变化−0.195pp，DEV反而+0.681pp；2.0%的DEV为+0.735pp。旧合法CF/FF边全部保留，无新增非法CF，无额外删面；新增/移除非法FF必须分开看。独立重放通过，当前根C锚点保护实际拒绝9个风险候选。

DEV不是简单“第三点离bulk太远”：96个保留seam tip中95个到有限bulk开放线段≤2%L，但近线段/共点不等于整条侧边共边。608/610在两档都没有可接受的整侧边候选，主要受边轴向、端点和全簇位移限制；2d少量几何候选又被坍缩/小角/形状门槛拒绝。每请求保留seam面实际为2d16张、610和608各4张，不能把平均8张当作每份身份。

2d三条来源alias在四噪声中形成的12个槽位全部保留；只有n0的一条alias槽有2对非法FF，全部12槽均无非法CF。不能据此称所有alias都堵住了bulk。详见 `cpu_binding/dev_diagnostic/` 的逐tip、逐侧边、筛选漏斗和alias去向表。

## 3. C-only 角部规则

先冻结只读C的规则，再比来源标签；来源Y/父ID/DEV标签没有进入提议排序或正式推理。50条件共424个共点边对提议；47条TRAIN提议通过单候选几何门槛，33条经容量竞争最终选中，DEV选中0。

来源的6条TRAIN alias与3条DEV alias **全部已被初始枚举覆盖**，但9条都被预先固定的5°最小角门槛拒绝（实际约3.67°—4.36°，部分也有q代价）。所以这次规则没有成为可用的alias预测器：召回0/6、0/3；这也不能解释成C无法枚举这些三角形。33个已选提议均不匹配来源alias，其中2个匹配其他来源面，31个不匹配；不匹配唯一来源完成并不自动意味着其开放生成结果非法。没有看到标签后调阈值，也未把此规则接入正式采样。

## 4. 训练拟合与真实生成分开看

'''+md(['臂','窗口','FM','seam第三点速度MSE','bulk速度MSE','couple项'],trows)+'''

COUPLE末100的seam第三点速度MSE比PLAIN低约26%，但bulk误差几乎不变。couple原始项由0.133413降到0.056738；λ·couple/FM输出速度梯度范数和比由0.462降到0.236，**不是参数梯度占比**。PLAIN该项0是未启用而非几何误差为零。这些都是正常监督插值态统计，不是额外rollout，不能替代以下终点结果。

## 5. 全面板终点数值

以下均为逐请求宏均值；TRAIN16与DEV12分开。FF/CF是非法面对涉及的free面比例；I是原固定C查询上的 `ConflictCov_C`，不是全物体冲突覆盖。可达指原合法free共边路径的bulk面积比例；≥30%只是预登记描述门槛，不认证整圈封闭或整物体可用。百分数单位为%，差值为百分点pp。

'''+md(['阶段','组','模型','bulk≥30%','bulk可达%','FF%','CF%','I%','小角%','保留free面','删除free面'],postrows)+'''

仅比较couple的增量（COUPLE−PLAIN）：

'''+md(['组','阶段','bulk可达pp','FF pp','CF pp','I pp','小角pp','保留面差'],pairrows)+'''

原POST里，两新臂TRAIN均11/16、DEV均0/12；加同一绑定后PLAIN为13/16与1/12，COUPLE为14/16与0/12。B起点同绑定已是14/16与0/12。COUPLE在原POST平均FF较PLAIN低0.548pp（TRAIN）和1.016pp（DEV），但CF/I和DEV小角率更高，DEV平均少保留10.42张free面，不能把它概括成整体质量全面提升。相较B起点，PLAIN/COUPLE的DEV原POST FF分别升约3.955/2.940pp。

唯一新DEV ≥30%请求是普通续训 `608b46b0_n3`，绑定后bulk可达61.799%。逐操作核验显示只接受1条 LATERAL：seam来源面2与bulk面99形成长2.022633模型单位的合法侧边，tip端此前已经共点，仅另一bulk簇并到固定C簇，最大实际位移1.12722%L。旧4条CF及155条FF合法边全部保留，0删面、0新增非法CF/FF，分别消除1对旧CF和FF；I从1.2695%降到1.1230%。COUPLE同一来源面配对需要2.48402%L，超过冻结阈值，其接受的TIP只增加bulk内部边而未接到seam。详见 `endpoint_binding/DEV_SINGLE_CASE_AUDIT.json`。

原POST没有达到该门槛，couple同请求也没有；因此收益属于“该续训终点＋冻结绑定”组合的一个请求，不是couple特有效果。原FF/I的尾部也保留，例如COUPLE `610/n0` POST FF38.168%、I8.105%，不能只引用整体平均。

RAW表也完整保留；soup阶段与索引整理后不可混成一个口径：

'''+md(['组','模型','RAW FF%','RAW CF%','RAW I%','RAW小角%'],rawrows)+'''

## 6. 绑定的直接代价账本

'''+md(['模型','组','n','失CF合法边','失FF合法边','删面','新增非法CF','新增非法FF','消除非法FF','TIP接受','整边接受'],bindrows)+'''

这些数来自同一POST前后、按来源面/真实索引精确配对；没有用总分变化代替撕裂检查，也没有通过删面掩盖损伤。具体候选与拒绝原因仍在每格账本。绑定的几何正结果与训练本身收益分别报告。

## 7. 实际图审

84页（两角度＋同尺度正侧剖面）全部实际查看，逐模型共252条记录。评阅者为AI_PROXY，模型名可见；不是独立人类盲评。YES与UNCERTAIN分开保存，来源款式不作唯一答案；有限视图不能认证无交/水密/精确共边。

下表为POST图审，格内为 YES数 / UNCERTAIN数；碎片与裂口列YES代表观察到问题。

'''+md(['组','模型','n','主体完整 Y/U','C座面采用 Y/U','碎片/折片 Y/U','裂口 Y/U'],visualrows)+'''

具体例子：73/n1两条续训都比B起点更清楚地以蓝C为座面，但仍有背板裂口；85/n1,n2整体视觉较好，开放式靠背不自动判为缺失。2d/n3的COUPLE在RAW、POST和绑定后均保留脱离主体的支撑条及不完整底座。608/n3的COUPLE也有外张/撕开的腿板。2d部分噪声存在蓝色上座面与较低灰色弯曲壳层，用途是否真正整合保留UNCERTAIN；不能用“C坐标没动”代替采用成功。完整逐例理由见 `visual_review_endpoint.csv`，各评阅者原始表与实际PNG hash另存。

## 8. 对原假说的判断与收口

1. 在B+LONG布局上追加固定局部绑定：有TRAIN局部正收益，但DEV没有在原终点上普遍转化，2%L也没有突破；保留其有限适用范围，不称物理上限。
2. 仅凭C凹角确定来源alias：当前固定规则未成功。规则能枚举来源alias，却与来源小角几何不兼容；也没有证明C决定唯一正确完成。
3. TRAIN47延长训练并加couple：确实进一步拟合了接口监督，但没有给出couple稳定提升DEV接续的证据。普通续训+绑定出现1份DEV局部正例，同时两新臂DEV FF高于B起点，说明额外训练不是无代价的修复；固定3000步也不等于已经证明所有部分充分收敛。

本轮优先保留的是**普通续训608/n3与其绑定前后的可复查局部正例**，以及两臂完整配对证据；不把couple晋级为默认，也不自动追加训练。若今后继续，优先问题应是“为何同样更低的监督接口误差未稳定变成生成时的整侧边相容”，而不是继续把一个MSE阈值等同于已收敛或把所有DEV失败归因于占用漏判。此处仅给出证据边界，本轮固定任务到此结束。

## 9. 资源、工程事件和产物

RTX5080 16GiB、24逻辑CPU。开始遵守合计约6GiB训练偏好，测得一worker allocator reserved峰3184MiB并保留CUDA/背景余量，先单臂。用户随后明确允许使用当前空闲显存；重新测资源后在PLAIN末段增量开启COUPLE。短并发窗口设备used峰9034MiB、free最低6946MiB；PLAIN/COUPLE各训练墙钟4820.46/4883.06秒，包含短重叠，不能相加当纯串行总墙钟或作为隔离性能benchmark。WDDM进程VRAM不可得，设备总量包含背景进程，没有把allocator当作完整进程显存。

RAW阶段先测真实请求，再增量2worker；D0按自身首例测量，因第一短阶段已完成，实际两臂先后执行。CPU OT/评分/绑定/渲染均按实测RAM和负载分配单线程worker。没有额外探针模型调用、OOM或科学参数降配。

工程事件保留：角部规则首次CSV导出字段不足，原几何结果保留，仅恢复导出；用户解除显存限制后主动停止旧串行协调器并由唯一新协调器派发，未中断训练；独立CPU资源观察器遇到状态文件原子替换竞态，内置GPU观察账本完整，未重跑训练；采样首例资源日志曾误取无序glob的第5条，随后追加精确更正，不影响实际样本或预算。没有静默覆盖旧科学结果。

- 最终权重：`checkpoints/PLAIN3000_step3000.pt`、`checkpoints/COUPLE3000_step3000.pt`（含匹配AdamW）；`inference/*_generator.pt`为独立生成器导出，非EMA，不含头。
- 训练：`training_analysis/`逐更新、五个t区间、曝光与学习曲线；`training/*/calls.jsonl`实际调用账本。
- 数值：`results/per_output.csv`、`paired_per_noise.csv`、`paired_per_parent.csv`、完整124字段汇总与删除/角色/绑定账本。
- 图像：`gallery/training/index.html`（56页RAW/POST）、`gallery/endpoint_binding/index.html`（28页同绑定）；`summary_figures/`科学图及PNG/PDF。
- CPU诊断：`cpu_binding/`、`corner_rule/REPORT.md`及`export_recovery/`；原报告/数据/模型只读。
- 逐例图审：`visual_review_endpoint.csv`与四份原始评阅记录。来源与预算最终核验见 `final_audit.json`；源码、code.diff和hash索引见 `source_archive/`。
- 恢复入口：`python -B -m r2.pilots.seam_pool47_couple_v1.resume status`，默认只读，失败调用不会自动重放。实际完成后写入`completion.json`停止自动派发。
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    (OUT/'report_completion.json').write_text(json.dumps(dict(status='REPORT_WRITTEN_FINAL_AUDIT_PENDING',model_calls=0,scorer_calls=0,
        visual_rows=252,visual_pages=84,report_sha256=sha(OUT/'REPORT.md'),input_review_hashes=sources),indent=2),encoding='utf-8')
    print('REPORT.md and 252 actual visual records written; final audit still required.')
if __name__=='__main__': main()
