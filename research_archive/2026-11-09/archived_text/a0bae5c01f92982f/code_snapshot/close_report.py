"""Close the fixed quick pilot using completed receipts and actual image reviews."""
from pathlib import Path
import csv,json,hashlib,collections,shutil,difflib,time,zipfile
from . import eval_common as C
OUT=C.OUT;EV=OUT/'evaluation';G=EV/'gallery'

def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def main():
    reviews=rows(G/'AI_PROXY_review_half.csv')+rows(G/'AI_PROXY_review_full.csv')
    assert len(reviews)==48 and len({(r['label'],r['request_id'],r['stage']) for r in reviews})==48
    for r in reviews:
        assert r['review_status']=='REVIEWED' and r['reviewed_image_sha256']==r['image_sha256']==C.sha(r['image_path'])
    C.csvwrite(G/'AI_PROXY_review.csv',reviews)
    groups=collections.defaultdict(collections.Counter)
    for r in reviews:groups[r['label'],r['stage']][r['C_adoption']]+=1
    C.write(G/'review_completion.json',dict(status='ACTUALLY_REVIEWED',actual_pages=8,rows=48,review_type='nonblind AI_PROXY, not human rating',
        adoption=[dict(label=k[0],stage=k[1],counts=dict(v)) for k,v in sorted(groups.items())],limitations='Visual adoption and visible contact do not certify topology, legality or generalization.'))
    m=C.read(OUT/'inputs/manifest.json')
    for path,h in m['source_hashes'].items():
        assert C.sha(path)==h, path
    C.write(OUT/'protected_assets_final.json',dict(status='PASS',files=len(m['source_hashes']),models_called=0,sha256_unchanged=True))
    code={str(p):C.sha(p) for p in C.CODE.glob('*.py')}
    frozen=C.read(EV/'registration.json')['code_hashes']
    for p,h in frozen.items():assert C.sha(p)==h
    for p in C.CODE.glob('*.py'):
        target=OUT/'code_snapshot'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
    diff=''.join(''.join(difflib.unified_diff([],p.read_text().splitlines(keepends=True),fromfile='/dev/null',tofile=str(p.relative_to(C.ROOT)))) for p in sorted(C.CODE.glob('*.py')))
    (OUT/'code.diff').write_text(diff,encoding='utf-8')
    C.write(OUT/'source_file_hashes.json',dict(new_code=code,frozen_evaluation_and_core=frozen))
    report='''# R2_SEAT_QUICK_LEARNABILITY_V1

状态：已完成并停止。2026-10-13。两臂各200步、16份新RAW、16次单次D0、48个RAW/POST逻辑评分与48条实际图审。未增加实验、改变默认模型、commit或push。

**结论：200步集中学习已给出局部正信号，原架构并非完全学不会座面采用；尚未证明跨噪声稳定解决。** 相比相同预算全池续训，集中组在一个困难请求上改善RAW座面采用，并在两个请求上新增POST合法接续；同时丢失一个原有接口，部分错层仍在。宜保留这套快速筛查入口，不据此晋级默认模型或否定其他数据/架构假说。

## 固定比较

三模型是冻结EXP67、SMALL_C200（4父4条件集中续训）和FULL_C200（原67父194条件普通续训）。两臂严格恢复同一step6000生成器、AdamW及RNG，lr=1e-5，局部更新1—200，优化器累计6001—6200。batch8/micro1、4clean/4width、free-only FM、OT后HYBRID0.25、BF16前向/FP32参数及积分均不变。无新模块、loss、教师或额外前向探针。

集中组每批四父各2个独立随机实例，每父总曝光400次；全池每批8不同父，每父总曝光23或24次。同槽增强、时间和随机流种子身份配对，但C/free形状不同，不能称两组输入或OT匹配逐位相同。当前比较集中座面条件曝光这项整体策略，不单独证明减少父数或仅座面训练的因果。

面板固定4个TRAIN父，各1条C、2份旧评价噪声；原C/N/噪声/查询、来源角序不变，未见训练随机流并不等于未见父泛化。两份FULL来源仍有18/54对远端FF风险，不能称全目标干净。原几何不修复，不用唯一来源GT款式评分。100步仅恢复点，固定200步评价，无best选择。

## 数值结果

所有数值为同一官方单次D0→GEO_C→来源C角序恢复和free定向后的POST，4父等权（每父2噪声）。CF/FF为原非法面对定义；I保持原近重复/重叠候选定义，并非全面几何正确性认证。

|POST指标|冻结EXP67|集中4条件200步|全池200步|
|---|---:|---:|---:|
|有合法C/F共边的输出|3/8|4/8|2/8|
|严格A-VMR|28.846%|41.493%|23.611%|
|PMR|25.000%|25.000%|12.500%|
|方向可达free面积|17.427%|34.832%|15.335%|
|CF非法|2.282%|1.144%|2.378%|
|FF非法|21.110%|21.138%|23.158%|
|I|0.555%|0.458%|1.263%|
|小角<5°|31.354%|26.905%|32.800%|

集中组相对冻结起点并非所有项都提高：FF基本持平且略升0.028个百分点，PMR持平；RAW的CF从5.223%升至5.902%，I从2.808%升至3.308%。全部RAW的严格A-VMR和合法共边仍为0。新增精确接续是**生成位置变化与相同后处理共同作用**的结果，不能记成RAW已生成精确拓扑。

集中组新增接续来自3f7417/n0（2条合法边、方向可达99.01%）及85b169/n1（7条、44.17%）；73b369/n0从1条降为0。8c9b20双噪声保留15/17条；其中n0严格A-VMR从100%降至94.44%。大可达面积表示少数接口接入较大的free连通区域，不是C边界接好比例。

## 实际图审

8页全部实际查看，48条逐stage记录在[图审CSV](evaluation/gallery/AI_PROXY_review.csv)。这是非盲AI_PROXY，不是用户评价或独立人评。图中蓝色是C，灰色是free；完整视图与两个C中心切片共享尺度。

- **3f7417/n0**：集中组RAW灰座体抬到蓝C同层，POST也保留改善；冻结起点与全池对照仍有下方替代座层。这是本轮最清楚的采用正例。同父n1未复现。
- **73b369/n1**：三模型仍有蓝C低于灰色座面的错层。n0布局本已较合理，两个续训组都丢失了原有一条合法边。
- **8c9b20双噪声**：三模型的座面采用与整体款式近似，本来就是相对正例，不能把维持现状记成新增采用能力。
- **85b169双噪声**：集中组的椅背、腿和厚座框明显改变，两个噪声趋于相近款式；n1有真实新增POST接口。座面位置原本即可解释，不仅凭款式变化判采用成功或多样性坍缩。未进行正式多样性评估。

每个阶段的图审采用计数相同：冻结5 YES/3 NO，集中6 YES/2 NO，全池5 YES/2 NO/1 UNCERTAIN。这些只概括4个熟悉条件上的观察，不能当泛化成功率。模板UNKNOWN未用于冒充完成，最终记录保留真正不确定项。

## 耗时与资源

SMALL训练阶段320.961秒，FULL320.181秒，合计641.142秒（10分41秒），包含载入、输入及保存。两臂串行，共3200训练F、3200B、400更新；0失败、0重试、0悬空调用。16次Euler50生成循环合计51.276秒；两组D0/GEO_C/导出阶段合计66.111秒；32次新CPU评分40.528秒、2 worker。各阶段口径见账本，前向/反向是训练墙钟子项，不重复相加。首次实现、准备和图审另计；不把10分41秒声称为整个交付时间。

按用户约6GiB实验显存预算，使用单GPU worker，保持其他图形学进程。训练allocated峰约2768/2796MiB，reserved峰3120/3160MiB；整卡观测峰9853/10060MiB含用户程序。Windows/WDDM未提供单PID显存，进程峰记NA；观察相对启动前整卡增量约3.5—3.7GiB，不能视为精确进程测量。GPU完成后已释放，整卡回到约6323MiB背景占用。没有OOM或因资源改精度、batch、模型。

## 下一步判断与交付

本轮证明短训练足以出现可检验的几何和局部接续变化，因此后续可以先用这种小试筛方向。优先假说是：**现有结构对部分可靠座面条件的学习曝光不足，集中学习能改善采用，但对另一些噪声/形状仍存在障碍。** 若继续，仍应以剩余错层请求的跨噪声改善为判断，不只追求当前正例指标；200步结果不能证明架构已足够，也不能证明必须更换架构。本轮不自动追加更新。

- [完整图集](evaluation/gallery/index.html)
- [48条逐输出](evaluation/per_output.csv)、[逐父配对](evaluation/paired_per_parent.csv)、[补全接续配对](numerical_pairs.csv)
- [数值与调用复算](NUMERICAL_FINDINGS.md)、[来源与输入风险](inputs/PREPARATION.md)
- [恢复入口](README_RESUME.md)：`r2/pilots/seat_quick_learnability_v1/run.py`
- 两臂完整step200权重在`checkpoints/`，推理导出在`inference/`；SHA见相邻JSON。100恢复点保留，不上传权重。
- 实际新代码、code.diff、source_file_hashes.json、原始评分JSON/面对及逐点记录、资源与逐调用账本均在本目录。公共模型、原评分器、旧RAW与旧报告只读。
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    report_files=[OUT/'REPORT.md',OUT/'NUMERICAL_FINDINGS.md',OUT/'README_RESUME.md',OUT/'registration.json',OUT/'source_file_hashes.json',OUT/'code.diff',OUT/'summary.json',OUT/'numerical_pairs.csv',OUT/'implementation_review.json',OUT/'protected_assets_final.json',OUT/'inputs/PREPARATION.md',OUT/'inputs/manifest.json']
    report_files+=list((OUT/'code_snapshot').glob('*.py'))+list(EV.glob('*.csv'))+list(G.glob('*.csv'))+list(G.glob('*.json'))+[G/'index.html']+list((G/'images').glob('*.jpg'))
    package=OUT/'R2_SEAT_QUICK_LEARNABILITY_REVIEW.zip'
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(report_files)):z.write(p,p.relative_to(OUT).as_posix())
    assert package.stat().st_size<10_000_000
    with zipfile.ZipFile(package) as z:assert z.testzip() is None
    complete=dict(status='COMPLETE_STOPPED',training_updates=400,train_forward=3200,train_backward=3200,new_rollouts=16,generator_forwards=800,D0_forwards=16,
        raw_logical=24,post_logical=24,new_scored_stages=32,reused_scored_stages=16,actual_review_pages=8,actual_review_rows=48,
        protected_files=len(m['source_hashes']),unchanged_core_sources=len(frozen),failed_calls=0,automatic_replays=0,
        model_default_changed=False,commit_or_push=False,package_path=str(package),package_bytes=package.stat().st_size,package_sha256=C.sha(package),unix=time.time())
    C.write(OUT/'COMPLETE.json',complete)
    print(json.dumps(complete,ensure_ascii=False))
if __name__=='__main__':main()