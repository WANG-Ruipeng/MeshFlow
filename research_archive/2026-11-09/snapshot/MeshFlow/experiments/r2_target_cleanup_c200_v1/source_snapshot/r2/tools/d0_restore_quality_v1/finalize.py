"""Record actual AI_PROXY review and close the completed CPU diagnostic."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from . import run as R
from r2.pilots.seam_slot_c200_v1 import common as C


def main():
    reg=R.registration();summary=C.read(R.OUT/'summary.json');gallery=C.read(R.OUT/'gallery/manifest.json');execution=C.read(R.OUT/'score_completion.json')
    observations={
      '73b369b9': '底座与主体间的可见开口在RAW和无约束D0中已有；刚体列保留同样开口。第二视角D0的座面与靠背附近白缝更明显，硬恢复后部分收窄；不能据此认证精确共边。',
      '3f741759': '靠背、座面和支撑腿在五阶段均可辨；刚体列与无约束D0无可见局部修复。未见硬恢复后主体全面撕裂。n2的蓝色座面覆盖变化与C被移动有关，不能将蓝色遮挡当作无重层。',
      '85b16941': '四份椅子主要座面、背板、支撑保留；刚体列没有可见消裂收益。部分座面边沿细白线与点状栅格不能直接判定为真实裂缝或孤立片。',
      '8c9b2023': '横档靠背、座面与腿整体保留；n2低横档/座面附近不整齐关系跨阶段存在。刚体变换未改变这些局部关系，未见硬恢复新增大块主体缺失。'}
    reviews=[]
    for q in reg['requests']:
        images=[x for x in gallery['images'] if x['parent_id']==q['parent_id']]
        assert len(images)==2
        for stage in ('RAW','UNCONSTRAINED','RIGID','HARD_RESTORE','POST'):
            note=observations[q['parent_id'][:8]]
            if q['request_id']=='73b369b9_n2':note+='该噪声上背板的横向开口在所有列持续存在。'
            reviews.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],stage=stage,
                reviewer='AI_PROXY / occupancy_sampling_review',blind=False,status='ACTUALLY_VIEWED',
                review_scope='Two complete contact sheets, same scale; all four noise rows and five columns inspected.',
                visible_major_parts='PRESENT_IN_VIEWS',exact_join='UNKNOWN_NOT_CERTIFIED_FROM_IMAGES',
                tiny_fragments_or_hidden_overlap='UNKNOWN_RENDER_RESOLUTION_AND_OCCLUSION',
                rigid_local_healing='NO_VISIBLE_CHANGE' if stage=='RIGID' else 'NA',notes=note,
                image_paths=[x['path'] for x in images],image_sha256=[x['sha256'] for x in images]))
    C.csvwrite(R.OUT/'gallery/actual_visual_review.csv',reviews)
    C.write(R.OUT/'gallery/review_completion.json',dict(status='COMPLETE',actual_images_viewed=8,review_rows=80,reviewer='AI_PROXY',blind=False,
        basis='Images emitted through view_image and inspected before this record. No independent human review.',files={x['path']:x['sha256'] for x in gallery['images']}))
    r=summary['summary'];before,rigid,hard=r
    reduction=100*(1-summary['mean_C_RMS_after']/summary['mean_C_RMS_before'])
    text=f'''# R2 D0 恢复与刚体诊断 V1

状态：COMPLETE，2026-10-16。四条已见TRAIN条件×四份原噪声，16个缓存请求。**零模型加载、零神经网络前向、零训练、零新噪声、零新GEO。**

**结论：D0的原始连续提议并非已经完美连接的整体；全局刚体拟合保留了所有原有内部相交，且不能精确恢复输入C。硬恢复确实改变了局部CF/FF关系，但没有制造主要bulk内部冲突。** 这项结果拒绝“刚体恢复即可100%消裂”的强结论，不否定更有针对性的局部协调恢复。

## 固定合同与评分域

只读 `SEAM_SLOT_C200` 原PREDICTED布局16个RAW、完整单次D0 proposal、恢复后的PRE_GEO与原POST，未使用新ALL_ON/SOURCE输出。UNCONSTRAINED使用实际D0的C_hat/F_hat；HARD_RESTORE复用现有C及seam前两角点写回结果。RIGID仅根据C角点对应求一个proper Kabsch R,t（detR=+1），同源C顶点副本总权重1，对整个proposal施加相同变换，不重新写回C、不进行焊接/删面。

刚体计算及输出FP64；原缓存保持FP32。固定L={C.L}，不依输出bbox重定尺度。实际内部CF使用每个阶段自身的C侧三角形与F；FF只看F。旧HARD评分与原RAW/POST参考读取缓存，新增UNCONSTRAINED/RIGID共32套实际内部扫描，固定原C冲突仅作额外诊断。

**固定原C上的I诊断不是前两方法的任务成功评分。** 它用原2048查询及原I实现，回答方法F与原用户C的相容程度；前两方法实际导出的C已改变。实际C_hat上的I为NA_NO_FROZEN_QUERY_FOR_MOVED_C，没有重抽或偷偷映射查询。CF、FF、I不可混同；无唯一GT重建排名。

## 真实D0内部质量与恢复代价

|阶段|实际内部CF受影响面|FF受影响面|非法CF对|非法FF对|bulk–bulk非法对|
|---|---:|---:|---:|---:|---:|
|UNCONSTRAINED|{100*before['CF']:.4f}%|{100*before['FF']:.4f}%|{before['CF_pairs_sum']}|{before['FF_pairs_sum']}|{before['bulk_bulk_pairs_sum']}|
|单一RIGID|{100*rigid['CF']:.4f}%|{100*rigid['FF']:.4f}%|{rigid['CF_pairs_sum']}|{rigid['FF_pairs_sum']}|{rigid['bulk_bulk_pairs_sum']}|
|现有HARD_RESTORE|{100*hard['CF']:.4f}%|{100*hard['FF']:.4f}%|{hard['CF_pairs_sum']}|{hard['FF_pairs_sum']}|{hard['bulk_bulk_pairs_sum']}|

比例先逐输出再宏平均，面对数为16请求总和；不能将面对计数当受影响面比例。RIGID与UNCONSTRAINED的全部172个非法CF、1925个非法FF面对身份和分类逐一完全相同，没有新增/消除。全角点两两距离最大数值差为{summary['max_rigid_distance_invariance_error']:.3g}，符合刚体距离不变量。

UNCONSTRAINED→HARD_RESTORE：CF保留36、消除136、新增177，净增41；FF保留1889、消除36、新增32，净减4。不能把“总量差不多”解释为没有局部副作用。新增/消除面对均保存在pair_changes.csv，未删任何面来掩盖代价。

**三阶段bulk–bulk均为1852对，且逐对身份完全相同。** 硬恢复仅改C与seam固定角点；bulk坐标逐字节不变。因而这部分冲突确实在原D0提议中已有，不是将C写回才造成的。原POST FF约14.563%是已执行GEO之后的参考，不与未GEO的66.669%直接当方法优劣竞赛；本轮没有替RIGID再运行GEO。

## C是否只是整体移动

跨16输出的C加权RMS均值为{summary['mean_C_RMS_before']:.8f}→{summary['mean_C_RMS_after']:.8f}模型单位，均值之比改善{reduction:.2f}%。刚体后的最大C角点残差{summary['max_C_residual']:.8f}，约{100*summary['max_C_residual']/C.L:.4f}%L，16/16仍未精确恢复。

共{summary['duplicate_C_groups']}个重复来源C顶点组，{summary['split_C_groups']}个在D0提议中出现不同角点位置；最大组内距离{summary['max_C_copy_split']:.8f}，约{100*summary['max_C_copy_split']/C.L:.4f}%L。刚体保持这些副本间距离，不能将它们同时送回一个原C点。这是当前16份真实缓存中的非刚体证据，不是从旧RBF面板借来的推测。

## 固定原C诊断与3f/n2

固定原C冲突宏均值：UNCONSTRAINED {100*before['I_against_fixed_original_C']:.4f}%，RIGID {100*rigid['I_against_fixed_original_C']:.4f}%，HARD {100*hard['I_against_fixed_original_C']:.4f}%。再次强调前两者不是保C方法的可比任务分数。

3f/n2分别为42.8711%、50.8301%、43.2129%。D0原F已经与原C附近的大面积区域冲突，不能将42%尾部全部归给恢复；单一刚体拟合C还可能使F相对固定原C的冲突增大。这不证明实际C_hat本身的I为这些数值。

## 连接、散片与实际图审

8张同尺度双视角联系图均已实际查看；80行（16请求×5阶段）非盲AI_PROXY记录已保存。蓝色在UNCONSTRAINED/RIGID中表示移动后的实际C侧，而非已经保真的原C。

- 73：底座/主体开口跨RAW、原D0与刚体列存在；n2上背开口也持续存在。第二视角D0座面/靠背附近白缝较明显，硬恢复后部分收窄，不能说只有恢复才出现裂缝。
- 3f：背部、座面和支撑腿保留；n2蓝色覆盖差异须结合C位移与遮挡解释，不能根据蓝色遮住就判无重层。
- 85：主要外形保留；刚体没有可见局部修复。座面细白线/点状边界不能直接认证为真实散片。
- 8c：横档、座面和腿保留；n2低横档附近不整齐关系在多阶段持续。未观察到硬恢复才造成的大块主体缺失。

精确共享边、小裂缝和隐藏重叠统一保留UNKNOWN。无约束与刚体的精确坐标共边诊断平均CF边0、面分量187；硬恢复平均CF边9、面分量163.5。此处是未GEO独立角点表示的严格拓扑诊断，**不能把187个索引分量说成187块肉眼碎片**，也不能以白缝像素代替有限几何判定。

## 执行、工程记录与交付

按实际资源先跑首个授权请求，再使用2个CPU worker、每worker单线程。科学计算阶段墙钟{execution['wall_seconds']:.3f}秒；含资源准备、图审与报告的总时间另计。GPU仅资源查询，没有模型调用；未停止其他用户程序。

全部科学单元和score_completion已成功后，首次CSV汇总遇到重复request_id关键字TypeError。冻结run.py及结果保留，独立finish.py仅修正字典合并完成汇总，没有重评分、重拟合或重试科学单元。见aggregation_recovery.json。

完整逐例/逐父表：stage_metrics.csv、per_parent.csv、rigid_fit_per_output.csv、C_copy_groups.csv、role_pairs.csv；因果面对账本：pair_transitions.csv、pair_changes.csv；旧参考：reference_RAW_POST.csv。实际内部原评分明细、固定C逐点向量和rigid几何位于outputs/。图集gallery/index.html，真实图审gallery/actual_visual_review.csv。输入和源码hash均保留，最终保护复核通过。

与历史RBF/BLEND区别：本轮实际读取无约束D0本体，并首次在此16面板上做单一proper rigid诊断；没有传播局部恢复场。结论不把以前局部RBF正结果抹掉，也不自动改变C合同或默认后处理。

本项固定任务完成并停止；没有自动commit/push或启用新后处理。
'''
    p=R.OUT/'REPORT.md'
    if p.exists() and p.read_text(encoding='utf8')!=text:raise ValueError('Frozen report differs')
    if not p.exists():p.write_text(text,encoding='utf8')
    resume='''# 恢复入口\n\n已完成，不重复计算。使用WSL既有环境，PYTHONPATH=/mnt/e/MeshFlow/r2/src:/mnt/e/MeshFlow。\n\n1. python -B -m r2.tools.d0_restore_quality_v1.run --stage run --workers 2 （命中完整receipt只核验）\n2. python -B -m r2.tools.d0_restore_quality_v1.finish （独立汇总恢复；不要用旧run --stage aggregate/all）\n3. python -B -m r2.tools.d0_restore_quality_v1.render （读取既有图）\n4. python -B -m r2.tools.d0_restore_quality_v1.finalize （核验已记录图审/报告）\n\n发现未完成cell保留attempt后停止，不隐式重评分；缺缓存BLOCKED，不模型重生成。\n'''
    p=R.OUT/'RUN_AND_RESUME.md'
    if not p.exists():p.write_text(resume,encoding='utf8')
    sources=dict(reg['source_hashes'],**gallery['source_hashes'])
    for src in Path(__file__).parent.glob('*.py'):sources[str(src)]=C.sha(src)
    snapshot=R.OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for src,digest in sources.items():
        name=src.replace('/','__').replace(':','_').replace('\\','__');target=snapshot/name
        if not target.exists():shutil.copyfile(C.resolve(src),target)
        if C.sha(target)!=digest:raise ValueError('Snapshot differs')
    C.csvwrite(R.OUT/'source_hashes.csv',[dict(path=p,sha256=h) for p,h in sources.items()])
    R.registration()
    final=dict(status='COMPLETE',requests=16,model_calls=0,new_GEO=0,training_updates=0,review_images=8,review_rows=80,
        protected_files=len(reg['protected']),protected_unchanged=True,
        files={str(p):C.sha(p) for p in R.OUT.rglob('*') if p.is_file() and p.name!='completion.json'})
    C.write(R.OUT/'completion.json',final);print(json.dumps({k:v for k,v in final.items() if k!='files'},indent=2));return final

if __name__=='__main__':main()
