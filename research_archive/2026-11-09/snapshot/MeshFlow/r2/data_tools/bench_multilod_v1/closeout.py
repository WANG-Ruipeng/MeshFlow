"""Create the data-stage report, source snapshot and recoverable integrity receipt."""
from . import common as c
from .finalize import text_once
from collections import Counter
import csv
import json


def main():
    summary=c.read(c.OUT/'data/admission_summary.json')
    manifest=c.read(c.OUT/'data/manifest.json')
    with (c.OUT/'data/per_parent.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    accepted=[r for r in rows if r['status']=='ACCEPTED_LOW']
    phases=[]
    for label,directory in [('v1',c.OUT),('engineering_v2',c.OUT/'engineering_v2')]:
        events=[json.loads(x) for x in (directory/'ledger.jsonl').read_text('utf-8').splitlines()]
        current=None
        for e in events:
            if e['kind']=='CPU_DISPATCH':current=e
            if e['kind']=='CPU_DATA_BUILD_COMPLETE' and current is not None:
                phases.append(dict(revision=label,phase='proposal' if current['proposals_only'] else 'build',
                    workers=current['workers'],wall_seconds=e['unix']-current['unix'],
                    parents=e['parents'],numeric_pass=e['numeric_pass']))
                current=None
    counts=Counter(p['replay_group'] for p in manifest['parents'])
    ac=Counter(r['replay_group'] for r in accepted)
    table='\n'.join(f'| {r["parent_id"][:6]} | {r["replay_group"]} | {r["K"]} | {r["N_original"]} → {r["N_proposed"]} | {float(r["surface_probe_max_pct_L"]):.5f}% |' for r in accepted)
    report=f'''# Bench 固定C多面数增强：实际数据阶段报告

## 结论

已实际完成CPU构造、数值准入、三视图AI_PROXY检查和原件hash复核。原30父、30条物理C保持不变，新增9份合格的低面数几何版本，总计39份。没有新增独立父、没有新增物理条件，也没有证明训练或生成质量已经提高。

可以进入有限的“原池续训 vs 多LOD续训”比较；目前证据不足以直接将增强称为有效方法，关系预测头留在生成器比较之后。

## 借鉴原MeshFlow的哪一部分

[MeshFlow论文S5](https://arxiv.org/html/2606.23489v2#S5)描述离线二次误差边收缩、以几何距离过滤不同面数版本。本轮借鉴这条数据思路。官方[数据加载器](https://github.com/qiisun/MeshFlow/blob/master/datasets/mesh_dataset.py)中的在线排列与可选旋转/缩放是不同机制；本轮没有把Gaussian FM起点叫作目标几何增强。

本实现额外保护真实C及原CF共边邻面，采用确定性表面探针距离，不是官方脚本复现，也不是连续Hausdorff认证。原30父N128..776，C保持K2..84；新版本N142..622。

与已有路线的区别：自然N实验此前没有合格同父低N版本；JOINT_SHIFT改变C/Y坐标；C_EDGEFLIP改变C内部划分。本轮固定C，只给完整目标的free部分增加另一种三角划分与N。

## 实际准入

| 状态 | 父数 | 处理 |
|---|---:|---|
| 数值和实际图审通过 | 9 | 原版＋一份低N版 |
| 原N接近128，无法有意义减面 | 9 | 只保留原版 |
| 减面不足 | 1 | 只保留原版 |
| 几何质量门槛未通过 | 11 | 候选留档，不进训练池 |

| 父ID前缀 | 曝光组 | K | 总面数 | 双向表面探针最大距离 |
|---|---|---:|---:|---:|
{table}

9份的C坐标、顶点身份、内部角序及原CF邻面均精确保留；没有新增非法CF/FF、退化面、精确重复面、非流形边或同向双面边。9份总面积比例范围{summary['min_accepted_area_ratio']:.8f}—{summary['max_accepted_area_ratio']:.8f}，最大双向表面探针距离{summary['max_accepted_probe_distance_pct_L']:.6f}%L。

全部9份实际查看了前后斜视与俯视的原目标/候选/C对照，记录为非盲AI_PROXY。图审只覆盖可见范围；精确C合同由数组检查负责。图集是训练目标，不是模型生成结果。

## 固定门槛与失败边界

目标约减20%面；至少减max(8,ceil(原N×5%))面且N≥128。保持原坐标变换，不重新归一化。C及原CF邻面固定；边界、旧拓扑缺陷保护；边收缩受link条件、原/前一步面法向变化≤30°及不新增退化/重复/非流形约束。

独立准入要求：总面积98%—102%；每个原共边连通分量面积95%—105%；各分量身份一对一；全局/分量表面探针最大距离与bbox坐标偏移≤0.25%L。L=sqrt(3)×2/0.3762，为固定模型尺度。每面13个确定性探针，不是严格连续Hausdorff。非法面对采用封存的有限三角面检测器，按原面ID＋分类配对，不能用几何近邻冒充来源身份。

11份质量拒绝中，5f0e28新增两对非法FF：(15,31)、(16,31)；其余候选主要因分量内距离/面积与全局几何距离不合格。`COMPONENT_PRESERVATION`包含身份、面积和距离，失败不等于连通分量被删除。没有放宽阈值重试这些质量失败候选，也没有删除来源父来凑通过率。原30份仅继承冻结池资格，不借此次增量检查重命名成全部无缺陷数据。

## 明确记录的工程修正

第一版把所有非C锐边端点一律锁死，21个可尝试父中19个全部使用顶点被冻结，只有4cd得到有效版本。检查确认这是超出C/接口合同的额外限制。

保留v1全部代码、proposal和结果，另开engineering_v2：非C锐边依靠原QEM平面与30°法向约束，不再一律冻结端点；允许自由端收缩到原坐标不动的固定端。C/接口、边界、旧缺陷保护及独立质量门槛保持不变。只对20个减面不足父再执行一条确定性路径，新增8份通过；v1通过父和9个N下限父未重跑。实际共41条减面路径，不是30次单次成功构造。

## 训练接入与下一阶段

原池分组为{dict(counts)}；有LOW的父分组为{dict(ac)}。特别是FOCUS的a8/27本轮没有合格LOW版本。不能声称增强覆盖全部结构或解决了独立结构不足。

新清单按父组织variants。保留EXPOSURE原OLD/FOCUS/HIGH曝光政策（75%/12.5%/12.5%以及组内父日程），在父被选中之后再选NATURAL/LOW；没有LOW的父保持原样。同一CLEAN/X_WIDTH配对使用同一版本。不能把39行均匀抽样来替代原父政策，也不能称30父全局等权。

旧训练器明确拒同父多版本，并缓存旧N/coefficient，因此不能把新JSON直接塞进去。独立CPU选择器已准备；正式训练还须为选中版本重算valid/free mask、N/count bin、FM分母、OT目标和相应噪声数组，禁止复用旧N的coefficient或匹配。C保持不变不意味着不同free槽位的噪声数组可以逐元素照搬。

已用真实的历史3000步、24,000实例日程执行CPU选择合同核验（没有重跑训练）：39/39真实case的hash、N/K、版本及数组检查通过，原父/group曝光和四类随机种子逐行不变。LOW占3,202/24,000＝13.3417%；OLD为2,452/18,000，FOCUS为0/3,000，HIGH为750/3,000。这说明全池改变的实际曝光有限，不能把9/30个父直接解释成30%的训练实例增强。详见[data/sampler_contract_audit.json](data/sampler_contract_audit.json)。

后续顺序：①从同一保留起点分叉原池/多LOD两臂，固定更新预算、原父曝光与评测；先判断RAW与固定后端上的采用、完整性、支撑和CF/FF/I/碎片是否改善。②再固定生成器，比较几何距离、几何小头、几何＋隐藏特征的关系预测；来源拓扑只作训练标签或单列oracle，不进入部署输入。③关系预测过门后才考虑用预测关系解码。不能把增加数据版本和新增关系头同时合成一个无法归因的比较。

本轮未冻结新增生成器训练预算，所以执行范围止于数据和CPU适配准备：模型前向0、更新0、采样0。没有训练预测头、改默认模型、删除旧资产、commit或push。

## 产物

- [训练数据清单](data/manifest.json)：30父及39版本，完整路径与SHA256。
- [逐父准入表](data/per_parent.csv)：包括所有失败，非只列成功。
- [实际检查图集](REVIEW.html)与[图审记录](visual_review/)。
- `parents/`与`engineering_v2/parents/`：原始proposal、来源面谱系、逐项质量、Native packing检查及账本。
- [恢复入口](RUNBOOK.md)、`source_snapshot/`、`source_hashes.json`、`completion.json`。
'''
    text_once(c.OUT/'REPORT.md',report)
    runbook='''# CPU数据入口与恢复

已有完成结果优先校验，不重抽样、不调用模型。

在E:/MeshFlow工作区，用已有WSL环境：
```powershell
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.data_tools.bench_multilod_v1.run status
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.data_tools.bench_multilod_v1.run_v2 status
wsl.exe --cd /mnt/e/MeshFlow /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.data_tools.bench_multilod_v1.verify
```

从完整原输入构建的已执行顺序是inspect → run freeze → run propose/build → render → 工程v2 build/render → 实际view_image图审 → finalize → closeout。
`build`遇到已完成receipt只核hash并复用；遇到不完整proposal_attempt停止，不静默重算。工程v2只限原20个减面不足父，不对质量失败再次减面。

`review_root.py`仅封存本会话实际已看的三份观察，不是自动图审程序；不得用于为新图伪造人工或AI看图。其余六份由实际审阅代理保存。

`audit_sampler`仅以历史日程做CPU映射/身份核验，3000步是被读取的历史日程长度，不是新增训练；它不会写新训练日程或运行模型。

正式训练入口尚未修改。新manifest不能平铺传给旧trainer。见sampler.py及其接入说明：同父曝光不增加，CLEAN/X_WIDTH对齐，N/OT/FM分母重算。没有此阶段GPU预算或预测头权重。
'''
    text_once(c.OUT/'RUNBOOK.md',runbook)
    code=c.ROOT/'r2/data_tools/bench_multilod_v1'
    source_hashes=[]
    for p in sorted(code.glob('*.py')):
        dst=c.OUT/'source_snapshot'/p.name;dst.parent.mkdir(exist_ok=True)
        data=p.read_bytes()
        if dst.exists():
            if dst.read_bytes()!=data:raise RuntimeError('Snapshot differs')
        else:dst.write_bytes(data)
        source_hashes.append(dict(path=str(p),sha256=c.sha(p),snapshot=str(dst)))
    for name in ('intersections.py','seams.py'):
        p=c.ROOT/'r2/research/seam_slot_c200_v1/scoring_source/legacy/src/meshflow_control/evaluation'/name
        dst=c.OUT/'source_snapshot'/('frozen_detector_'+name)
        data=p.read_bytes()
        if dst.exists():
            if dst.read_bytes()!=data:raise RuntimeError('Detector snapshot differs')
        else:dst.write_bytes(data)
        source_hashes.append(dict(path=str(p),sha256=c.sha(p),snapshot=str(dst)))
    c.write(c.OUT/'source_hashes.json',dict(rows=source_hashes))
    c.write(c.OUT/'phase_timing.json',dict(phases=phases,scope='CPU dispatch wall times including scheduling; excludes implementation, fixture tests and image review.',model_calls=0))
    paths=[p for p in sorted(c.OUT.rglob('*')) if p.is_file() and p.name not in ('completion.json','verification_receipt.json')]
    c.write(c.OUT/'completion.json',dict(status='CPU_DATA_STAGE_COMPLETE',model_forwards=0,model_updates=0,
        source_manifest_sha256=c.sha(c.SOURCE),original_parents=30,added_geometry_versions=len(accepted),
        total_geometry_versions=manifest['total_versions'],original_case_hashes_verified=30,
        cpu_workers=2,numeric_threads_per_worker=1,gpu_workers=0,phases=phases,
        new_artifact_bytes=sum(p.stat().st_size for p in paths),
        files=[dict(path=str(p),sha256=c.sha(p),bytes=p.stat().st_size) for p in paths]))
    print(c.json.dumps(dict(status='CPU_DATA_STAGE_COMPLETE',parents=30,versions=manifest['total_versions'],phases=phases,new_bytes=sum(p.stat().st_size for p in paths)),ensure_ascii=False))


if __name__=='__main__':main()
