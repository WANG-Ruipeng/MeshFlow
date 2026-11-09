from .common import *
import collections,math,shutil,ast

def table(rows,keys):
    return '| '+' | '.join(keys)+' |\n|'+'|'.join('---' for _ in keys)+'|\n'+'\n'.join('| '+' | '.join(str(r.get(k,'')) for k in keys)+' |' for r in rows)

def main():
    out=OUT/'analysis';out.mkdir(exist_ok=True)
    def csvread(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
    weights=csvread(OUT/'weights/per_parameter.csv');widx={(r['label'],r['parameter']):r for r in weights};summary=[]
    for label in ('A','COVERAGE','USER'):
        rows=[r for r in weights if r['label']==label];b=sum(float(r['base_norm'])**2 for r in rows);d=sum(float(r['delta_norm'])**2 for r in rows)
        summary.append(dict(label=label,parameters=sum(int(r['count']) for r in rows),relative_delta_norm=math.sqrt(d/b),delta_norm=math.sqrt(d)))
    cosrows=csvread(OUT/'weights/delta_cosine.csv');cos=[]
    for a,b in [('A','COVERAGE'),('A','USER'),('COVERAGE','USER')]:
        cross=na=nb=0
        for r in cosrows:
            if r['a']!=a or r['b']!=b:continue
            an=float(widx[a,r['parameter']]['delta_norm']);bn=float(widx[b,r['parameter']]['delta_norm']);na+=an*an;nb+=bn*bn
            if r['cosine']:cross+=float(r['cosine'])*an*bn
        cos.append(dict(a=a,b=b,global_delta_cosine=cross/math.sqrt(na*nb)))
    save(out/'weight_summary.json',dict(norms=summary,cosines=cos))
    qs=read(OUT/'evaluation/score_registrations/COMMON.json')['requests'];metrics=[];pairs=[];cache_refs=[]
    for q in qs:
        p=OUT/'evaluation/scores/COMMON'/q['request_id']/'RAW/row.json'
        if not p.exists():continue
        a=read(p);a['producer']='NEW_COMMON_RAW';metrics.append(a)
        if q['split']=='DEV':
            bpath=V2/'evaluation/scores/USER_V2_S1500'/q['request_id']/'RAW/row.json';b=read(bpath);b['producer']='VERIFIED_PREVIOUS_USER_RAW';metrics.append(b)
            pair=dict(request_id=q['request_id'],condition_id=q['condition_id'],part_type=q['part_type'],noise_index=q['noise_index'])
            for k,v in a.items():
                if isinstance(v,(int,float)) and not isinstance(v,bool) and isinstance(b.get(k),(int,float)):
                    pair[k+'__COMMON']=v;pair[k+'__USER']=b[k];pair[k+'__delta']=b[k]-v
            pairs.append(pair);cache_refs.append(dict(request_id=q['request_id'],path=str(bpath),sha256=sha(bpath),status='same checkpoint/noise/anchor layout verified in gallery; historical RAW, no new forward'))
    csvwrite(out/'available_per_output_RAW.csv',metrics);csvwrite(out/'paired_cached_DEV_RAW.csv',pairs);save(out/'cache_refs.json',cache_refs)
    source=csvread(OUT/'source_reference/metrics.csv');source_summary=[]
    for r in source:
        source_summary.append(dict(parent=r['parent_id'][:6],split=r['split'],K=int(r['K']),owner_legal_edges=int(r['C_F_valid_edge_count']),bulk_reach_percent=round(float(r['free_only_reached_bulk_area_fraction'])*100,3),FF_percent=round(float(r['FF_illegal_affected_free_fraction'])*100,3),small_angle_percent=round(float(r['free_minimum_angle_lt5_fraction'])*100,3)))
    csvwrite(out/'source_summary.csv',source_summary)
    notes={
      ('35948e',0):('NO','UNKNOWN','COMMON 的灰座面在蓝色C上方另起一层，侧剖面高度错位；USER当前训练父输出缺失。'),
      ('35948e',1):('YES','PLAUSIBLE','COMMON 座面位置与C基本一致，有背和支撑；不以复现来源款式为成功条件。USER缺失。'),
      ('943dde',0):('PARTIAL','UNCERTAIN','COMMON 的C大致位于背部，但灰背与C有分层，侧剖面局部不连续；不能凭截线端点认定真实裂口。'),
      ('943dde',1):('NO','PLAUSIBLE_DIFFERENT_USE','COMMON 生成了位于C上方的座台，竖直C留在下部，未明确作为靠背使用。'),
      ('bf504b',0):('NO','PLAUSIBLE_BUT_MISMATCHED','COMMON 的灰主体比蓝色支撑小且偏低，C水平臂突出，竖腿悬出；输入未融入主体。'),
      ('bf504b',1):('NO','PLAUSIBLE_BUT_MISMATCHED','与n0相近的宏观尺度/位置不相容；两噪声不能据此证明多样性坍缩。'),
      ('5dda4e',0):('NO','PLAUSIBLE_BUT_MISMATCHED','COMMON 另画较低灰座面；USER也把灰座面放到C下方且新增大背，未修复C与主体高度关系。'),
      ('5dda4e',1):('NO','PLAUSIBLE_BUT_MISMATCHED','COMMON C低于灰座台；USER C在灰座台上方，截面两者均存在错层，方向不同。'),
      ('9d5608',0):('UNCERTAIN','PLAUSIBLE','COMMON C靠近靠背顶沿；USER减少了外围支撑/板件，但C与下方主体仍存在空间分离。C为狭长顶条，图像用途判定有歧义。'),
      ('9d5608',1):('NO','PLAUSIBLE_BUT_MISMATCHED','COMMON 把灰平台放在C上方；USER转为靠背式主体但C仍浮在顶部。款式变化不等于采用成功。'),
      ('2d1e5e',0):('NO','PLAUSIBLE_BUT_MISMATCHED','COMMON与USER均将小蓝色L形支撑包在较大灰壳中，截面位置错误高度近似相同。'),
      ('2d1e5e',1):('NO','PLAUSIBLE_BUT_MISMATCHED','COMMON与USER均另生成完整大体量座背，蓝色短支撑留在内部；局部薄片略有差异，宏观未解决。')}
    visual=[];pages=read(OUT/'gallery/complete.json')['pages']
    for page in pages:
        q=next(q for q in qs if q['request_id']==page['request_id']);adoption,quality,note=notes[q['parent_id'][:6],q['noise_index']]
        visual.append(dict(request_id=q['request_id'],split=q['split'],part=q['part_type'],reviewer='AI_PROXY',blinded=False,actually_viewed=True,
            COMMON_RAW_adoption=adoption,COMMON_RAW_whole=quality,USER_current_train='BLOCKED' if q['split']=='CURRENT_TRAIN' else 'VERIFIED_CACHE_VIEWED',
            image=page['path'],image_sha256=page['sha256'],notes=note))
    csvwrite(out/'visual_review.csv',visual);save(out/'visual_review.json',dict(status='ACTUAL_REVIEW_COMPLETE_FOR12_AVAILABLE_PAGES',reviewer='AI_PROXY_NONBLIND',pages=visual,
        limitations='Source reference visible; not blinded, not human rating;18 generated cells viewed,12 new COMMON plus6 cached USER; no POST or current-TRAIN USER judgments'))
    layout=read(OUT/'layouts/complete.json')['records'];layagg=[]
    for split in ('CURRENT_TRAIN','DEV'):
        r=[r for r in layout if r['split']==split];layagg.append(dict(split=split,edges=sum(x['edges'] for x in r),source_owner=sum(x['source_owner'] for x in r),predicted=sum(x['predicted'] for x in r),false_negative=sum(x['false_negative'] for x in r),false_positive=sum(x['false_positive'] for x in r)))
    diag=read(OUT/'gpu_dispatch_blocked.json');validation=csvread(OUT/'source_reference/projection_validation.csv')
    modelrows=[]
    for label in ('COMMON','USER'):
        for q in qs:
            path=OUT/'evaluation'/label/q['request_id']/'raw/receipt.json'
            state='COMPLETE_NEW_RAW' if path.exists() else 'BLOCKED_GPU'
            if label=='COMMON' and q==qs[12]:state='FAILED_PARTIAL_2_RETURNED_CALLS_NO_SAVED_TRAJECTORY'
            modelrows.append(dict(label=label,request_id=q['request_id'],split=q['split'],status=state))
    csvwrite(out/'request_status.csv',modelrows)
    md=f'''# Bench 生成轨迹与条件作用诊断：执行中断与已完成证据

日期：2026-10-22。**状态：PARTIAL / GPU BLOCKED，不是完成实验，也不是方法负结果。**

本轮已实际执行，但在 COMMON 第13个请求的第3次前向之前发生 CUDA illegal memory access；之后显卡状态查询也阻塞。没有启动训练。没有足够证据回答“USER 是否只改善训练插值拟合、而不能改善自由生成”。缺失项目全部保留为 BLOCKED，不能用权重范数或几张 COMMON 图代替。

## 1. 已执行的范围与准确账本

- 固定8个父、8条C、每C两份噪声，32个原计划完整生成。按部位与K/N元数据配对，选择先于生成。
- 4个当前TRAIN父：35948e座面、943dde靠背、bf504b支撑、838c48组合；对应DEV：5dda4e、9d5608、2d1e5e、f495bd。旧ID中的SEAT字样不覆盖已核验的实际BACK标签。
- 当前TRAIN的4次历史小占用头前向完成；DEV布局复用缓存。生成器实际602次前向返回：12×50完整RAW，另有2次属于未保存完整轨迹的中断请求。USER生成器0次；D0 0次；反向/优化器更新0次。
- 12份RAW与51状态轨迹已保存、原版评分已完成，保留逐输出完整字段。12张固定视角/截面图已实际查看；其中6个DEV对照复用了身份匹配的既有USER RAW，不计作新生成。
- 8个来源的精确共享顶点索引参考与合法接续评分完成。20个CPU投影有限差分检查完成，最大绝对导数误差{max(float(x['abs_difference']) for x in validation):.6g}，目标投影残差最大{max(float(x['target_split_mse']) for x in validation):.6g}。
- A/COVERAGE/USER相对COMMON的194个浮点参数张量差分全部计算，形成582行；33个预登记早/中/晚注意力及Geo矩阵完成精确SVD。没有新权重产生。

## 2. 现在已经能确定什么

### 来源不是普遍的“bulk≥30%”示范

{table(source_summary,['parent','split','K','owner_legal_edges','bulk_reach_percent','FF_percent','small_angle_percent'])}

当前4个TRAIN来源的无向合法图均可达全部bulk。旧DEV座面5dda4e仅5.83%，且来源本身有较多非法FF。不能把统一30%阈值解释为每个来源应满足的结构真值；也不能把DEV来源当作唯一正确款式来评价生成。

此参考严格按来源顶点身份建图，未按距离焊接。free角序沿用训练规范化表达，**方向可达字段不用于解释来源本身的正确绕序**；本节只使用无向合法共边和几何冲突。源表面存在小角面也不自动意味着无用或必须删除。

### 布局差异真实存在，但还不是因果解释

{table(layagg,['split','edges','source_owner','predicted','false_negative','false_positive'])}

当前TRAIN漏11条、误开2条，DEV漏12条。这是来源owner布局与生产布局的差异；来源owner不是C唯一合法完成。原计划的来源布局自由生成对照尚未执行，不能据此断言漏开边造成宏观错层。TRAIN靠背943dde的布局完全一致，所以其两份oracle采样应省略；有效原计划为2380次生成器调用，核心上限仍为2480。

### 权重改变量更大，不等于过拟合或改坏某层

{table([dict(label=r['label'],relative_percent=round(100*r['relative_delta_norm'],4),delta_norm=round(r['delta_norm'],4)) for r in summary],['label','relative_percent','delta_norm'])}

{table([dict(a=r['a'],b=r['b'],cosine=round(r['global_delta_cosine'],4)) for r in cos],['a','b','cosine'])}

这是拼接全部参数后的整体范数/余弦，不是逐层百分比的简单平均。零或极小基值的相对值留空，另报绝对变化。详细每参数/模块/奇异值均在weights目录。USER整体变化约0.891%，大于A的0.512%和COVERAGE的0.494%；不能据此推出过拟合、形状流形维数或某模块的因果责任。

相同状态响应与ΔWh没有运行，因此**模块回退的证据门槛未满足，本轮不启动回退实验。**

### 实际看过的图像支持有限的现象判断

- COMMON在35948e座面n0另生成上方灰座面，n1则大致正确使用C；这说明同C仍有明显噪声差异。
- COMMON在bf504b支撑两个噪声均出现小灰主体和较大蓝色支撑空间不相容。
- 6个已有DEV配对RAW中，2d的两噪声仍是蓝色短L支撑被大灰壳包住；5dda的灰座面与蓝C高度不一致。9d5608款式变化明显，但C是否合理参与仍有歧义。
- 这些不是后处理比较。当前TRAIN的USER输出尚缺，不能判断USER训练父生成是否好于COMMON，也不能把COMMON失败归因于USER训练。

完整记录见analysis/visual_review.csv；18个生成单元已查看，非盲AI_PROXY，没有把来源样式相同当成功。

## 3. 未完成项

|项目|状态|
|---|---|
|剩余4个COMMON、16个USER新完整生成|BLOCKED_GPU；其中6个USER DEV旧缓存仅用于局部比较|
|80次固定训练状态前向、实际ΔWh|BLOCKED_GPU|
|304次中途续跑|BLOCKED_GPU|
|来源布局从Gaussian生成|BLOCKED_GPU；计划6请求300次，另2请求布局相同省略|
|96次C正负扰动响应|BLOCKED_GPU|
|32次D0与统一GEO/FINAL链|BLOCKED_GPU；RAW跨模型屏障未满足，不提前跑|
|模块回退|NOT_TRIGGERED，证据门槛未满足|

不能宣布“只有泛化失败”“晚期向量场破坏结构”“D0毁掉提升”或“已定位某层”。本轮没有新的训练结论。

## 4. 工程故障、并发与恢复

初始RTX5080 16303MiB，使用约2.7–3.2GiB，空闲12.5GiB以上，24逻辑CPU、可用RAM约61GiB。COMMON真实首轮后总显存约3697MiB，Windows采样计算占用24%，满足新增worker显存余量，因此曾派发独立USER进程。但该进程实际卡在NVML资源查询，未加载模型、未前向；没有两模型同时执行前向的证据。不能将故障归因于并行OOM。

COMMON前602次返回后在下一次CPU→CUDA传输报错；WSL nvidia-smi子进程480观察为D+不可中断等待。已终止本轮USER进程479及其查询子进程的运行意图，未杀用户任务、未重置显卡或重启系统。Windows事件日志权限不足，具体驱动/硬件根因UNKNOWN。

另有两次CPU工程读取修正：A使用实际checkpoint_step1500.pt而非不存在的generator导出；来源评分返回dict按row取值，已成功的评分缓存直接复用。它们没有模型重算。所有原始数据、权重与旧报告未修改。

恢复入口：`python -B -m r2.pilots.bench_trajectory_diagnostic_v1.resume` 默认只报告阻塞、不调用模型。GPU恢复并明确允许中断请求重算2次后，使用 `--gpu-recovered --allow-two-forward-replay`。入口保留12份完整RAW，只归档未完成请求的attempt，原噪声重放该请求；预计累计2382次，仍在2480上限内。故障恢复改为逐模型执行并重新检查资源。后续仍需补功能分析、全阶段图审与最终报告，不会自动训练或切模型。

**下一步优先补齐已冻结诊断，而不是基于本轮缺失数据再改架构或扩池。**

入口与产物：registration.json、evaluation_inputs/、fixed_states/、execution/、layouts/、weights/、source_reference/、analysis/、gallery/index.html。报告是可恢复中间报告，不是实验收口成功声明。
'''
    (OUT/'REPORT.md').write_text(md,encoding='utf-8-sig')
    sources={}
    for p in Path(__file__).parent.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'));target=OUT/'source_snapshot'/p.name;target.parent.mkdir(exist_ok=True)
        shutil.copyfile(p,target);sources[str(p)]=sha(p)
    save(OUT/'source_hashes.json',dict(recorded_at='after partial execution and recovery-wrapper repair; not a preexecution hash claim',files=sources))
    save(OUT/'PARTIAL_STATUS.json',dict(status='BLOCKED_GPU',new_generator_calls=602,new_head_calls=4,new_D0_calls=0,training_updates=0,new_complete_raw=12,
        CPU_original_scores=12,source_reference_conditions=8,actual_review_pages=12,verified_cached_USER_RAW_reviewed=6,
        module_rollback='NOT_TRIGGERED',main_hypothesis='UNRESOLVED',report_sha256=sha(OUT/'REPORT.md')))
    print(json.dumps(dict(weight_summary=summary,cosines=cos,raw_rows=len(metrics),paired_dev=len(pairs),status='BLOCKED_GPU'),ensure_ascii=False))
if __name__=='__main__':main()
