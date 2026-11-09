"""Package a truthful DATA_SCALE_BLOCKED delivery. No model, RNG, OT or detector calls."""
from pathlib import Path
import collections,datetime,difflib,hashlib,io,json,os,shutil,subprocess,time,zipfile
ROOT=Path('/mnt/e/MeshFlow');OUT=ROOT/'experiments/r2_user_data_scaling_12h_v1';CODE=ROOT/'r2/pilots/user_data_scaling_12h_v1'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);s=json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if p.exists() and p.read_text('utf-8-sig')!=s:raise FileExistsError(str(p))
    if not p.exists():p.write_text(s,'utf-8')
def md(p,s):
    p=Path(p)
    if not p.exists():p.write_text(s,encoding='utf-8-sig')
def csvwrite(p,rows):
    import csv
    s=io.StringIO(newline='');keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(s,keys);w.writeheader();w.writerows(rows)
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():p.write_text(s.getvalue(),encoding='utf-8-sig')

def main():
    f=read(OUT/'data/data_freeze.json');assert f['status']=='DATA_SCALE_BLOCKED'
    assert read(OUT/'data/verification_data.json')['status'].startswith('PASS')
    reg=read(OUT/'registration.json');now=datetime.datetime.now(datetime.timezone.utc);elapsed=(now-datetime.datetime.fromisoformat(reg['t0_utc'])).total_seconds()
    frozen=read(OUT/'data/qualified_condition_manifest.json');train=read(OUT/'data/train_manifest.json');panel=read(OUT/'data/panel_manifest.json')
    tasks=frozen['conditions'];large=set(train['parents']);small=set(train['pools']['USER_SMALL']);dev={t['uid'] for t in panel['conditions']}
    coverage=[]
    for name,ids in [('USER_SMALL_CANDIDATE',small),('USER_LARGE_67_CANDIDATE',large),('NEW_DEV_ALL_AVAILABLE_CONDITIONS',dev)]:
        part=[t for t in tasks if t['uid'] in ids]
        for typ in ['SEAT_FULL','SEAT_HALF','BACK_FULL','BACK_HALF']:
            rr=[t for t in part if t['condition_type']==typ]
            coverage.append(dict(pool=name,condition_type=typ,parents=len({r['uid'] for r in rr}),conditions=len(rr),model_status='NOT_TRAINED_DATA_SCALE_BLOCKED'))
    csvwrite(OUT/'data/coverage_matrix.csv',coverage)
    write(OUT/'data_freeze.json',f)
    budget=dict(status='DATA_SCALE_BLOCKED_NOT_LOCKED',target_steps_per_arm=6000,allowed_fallback_steps_per_arm=4000,
        selected_steps_per_arm=None,preflight_updates=0,measured_seconds_per_update=None,
        reason='Task section 2: 67 qualified training parents <128. Do not initialize or measure a training task that cannot be formally dispatched.',
        training_deadline_unix=reg['training_deadline_unix'],formal_updates_SMALL=0,formal_updates_LARGE=0)
    write(OUT/'training_budget_lock.json',budget)
    missing=[]
    for stage in ['common_official_initialization','preflight_16_updates','SMALL_final_checkpoint','LARGE_final_checkpoint','inference_exports',
                  'fixed_training_diagnostics','new_eval_gaussian_arrays','420_RAW','420_D0_proposals','420_POST','RAW_POST_scores','model_comparison_gallery','output_visual_reviews']:
        missing.append(dict(artifact=stage,status='BLOCKED_NOT_RUN',reason='DATA_SCALE_BLOCKED before GPU and evaluation-input freeze'))
    csvwrite(OUT/'results/missing_artifacts.csv',missing)
    csvwrite(OUT/'results/paired_per_parent.csv',[dict(parent_uid=u,split='NEW_DEV',comparison='LARGE_MINUS_SMALL',status='NA_NOT_RUN',
        C_adoption='NA',attachment='NA',overall_quality='NA',CF='NA',FF='NA',I='NA',AVMR='NA',PMR='NA',reason='DATA_SCALE_BLOCKED') for u in sorted(dev)])
    csvwrite(OUT/'results/paired_per_condition.csv',[dict(parent_uid=r['uid'],condition_id=r['condition_id'],type=r['condition_type'],N=r['N'],K=r['K'],
        status='NA_NOT_RUN',difference='NA',noise_arrays='NOT_MATERIALIZED',reason='DATA_SCALE_BLOCKED') for r in panel['conditions']])
    csvwrite(OUT/'results/planned_output_status.csv',[dict(parent_uid=r['uid'],condition_id=r['condition_id'],noise_slot=i,label=label,
        status='NOT_GENERATED',raw_path='NA',post_path='NA',scores='NA',actual_noise_identity='NA_NOT_MATERIALIZED')
        for r in panel['conditions'] for i in [0,1] for label in ['USER_SMALL','USER_LARGE_67','HIST_USER_MIX_FROZEN']])
    write(OUT/'results/research_answers.json',dict(status='UNANSWERED_DATA_SCALE_BLOCKED',
        worth_scaling='NA: no comparable trained models or generated outcomes',adoption_seat='NA',adoption_back='NA',true_attachment='NA',
        CF_FF_I_cost='NA',historical_USER_MIX_change='NA',fit_vs_coverage_diagnostics='NA',
        next_priority='Obtain and validate enough existing low-face-count Chair parent meshes and source face regions before repeating the registered comparison.'))
    assets=[('OFFICIAL_GENERATOR_EMA',reg['official_generator_path'],'bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d','PRIOR_LOCAL_IDENTITY_RECEIPT_NOT_MODEL_LOADED'),
        ('HIST_USER_MIX_FROZEN',str(ROOT/'experiments/r2_user_mix_from_official_v1/inference/R2_USER_MIX_generator_step6000.pt'),'8c1287b13541b59fa7b1e70974a70b06fbe0d67160449c225e5bdc7fdd0897cd','THIS_RUN_HISTORICAL_CACHE_CHECK'),
        ('OFFICIAL_D0_EMA',str(ROOT/'experiments/r2_postprocess_avmr_pmr_v1/assets/chair_denoiser_0030000.pt'),'6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36','PRIOR_RECEIPT_AND_TASK_IDENTITY_NO_REHASH_7GB')]
    write(OUT/'weights_manifest.json',dict(new_weights_created=0,reason='DATA_SCALE_BLOCKED',references=[dict(role=role,path=path,
        exists=Path(path).exists(),bytes=Path(path).stat().st_size if Path(path).exists() else None,expected_sha256=h,verification_basis=b,embedded_in_zip=False) for role,path,h,b in assets]))
    audit=dict(status='DATA_SCALE_BLOCKED',elapsed_seconds_at_packaging=elapsed,formal_updates=0,discarded_preflight_updates=0,generator_F=0,generator_B=0,D0_F=0,
        OT_calls=0,new_gaussian_arrays=0,new_generated_meshes=0,new_weights=0,source_unique_pair_tests=3786998,source_primary_tests=3786998,source_independent_tests=3786998,
        source_variants=196,source_independent_groups=181,actual_v2_input_review_parents=181,additional_mapped_existing_reviews=25,
        cpu_quality_workers=2,threads_per_worker=2,quality_phase_wall_seconds=270.668,training_seconds=0,sampling_seconds=0,D0_seconds=0,output_scoring_seconds=0,
        note='CPU input-review/source scans are not model evaluation. Overlapping phases are not added as isolated wall time.')
    write(OUT/'call_accounting.json',audit)
    timing_note=dict(status='DISCLOSED_PREPARATION_ORDER_DEVIATION',requirement='Task section 2: split parents before deriving conditions',
        actual_order='Candidate ordering rules were registered first, all source proposals/reviews were prepared, then final 32-parent DEV selection and parent-exclusive manifests were made. Preliminary condition materialization preceded final role assignment.',
        impact='No model/noise/generator outcome existed; final identity/group disjointness is verified. This is a preparation-order deviation, not evidence of a completed fully compliant training run.',
        resolution='Current run remains DATA_SCALE_BLOCKED. A future authorized run must freeze parent roles first and derive formal conditions afterward; do not backdate or silently relabel these preparation timestamps.')
    write(OUT/'preparation_order_note.json',timing_note)
    # Navigation is an input-data gallery; it does not imply generated outputs or model evaluation.
    sheets=sorted((OUT/'data/review_sheets').glob('v2_*.png'))+sorted((OUT/'data/existing_region_review_sheets').glob('*.png'))
    html='<!doctype html><meta charset="utf-8"><title>Actual input reviews — DATA_SCALE_BLOCKED</title><h1>输入数据图审：未训练、未生成模型</h1><p>181个候选父已实际AI_PROXY审阅；语义审阅与CC/CF几何资格分别保留。下列均为来源网格，不是生成结果。</p>'
    for p in sheets:
        rel=p.relative_to(OUT).as_posix();html+=f'<h2>{p.name}</h2><a href="{rel}"><img src="{rel}" style="max-width:100%;height:auto"></a>'
    if not (OUT/'INPUT_REVIEW_GALLERY.html').exists():(OUT/'INPUT_REVIEW_GALLERY.html').write_text(html,'utf-8')
    report=f'''# R2 USER Data Scaling 12H V1 — DATA_SCALE_BLOCKED

**本轮不能回答“大池相对小池是否值得继续扩大”。两臂训练没有启动，因此不存在规模收益或失败的模型证据。** 实际执行了资产核验、来源几何扫描、区域构造、逐父输入图审和数据物化；任务书第2节要求训练父少于128时停止。本轮99个合格独立父中留出32个新DEV，训练池为67父，缺61父才达到最低门槛，缺189父才达到256目标。

## 实际数据与门禁

| 项目 | 实际数量 |
|---|---:|
| 本地官方Chair train / test | 2822 / 313 |
| 官方train内N=128..256原始候选父 | 462 |
| 无精确零面积面的原始train候选 | 186 |
| 结合现成版本、历史排除、同源去重后扫描 | 196版本 / 181独立组 |
| 按冻结规则所有面均涉及缺陷的组 | 12 |
| 仍有至少一张未被标记问题面的组 | 169（不是169个合格USER父） |
| 真正通过本轮语义图审与局部CC/CF检查 | 99父 / 286条去重条件 |
| 新DEV | 32父；可用92条，预选64条 |
| 训练候选大池 | 67父 / 194条 |
| 嵌套训练小池 | 32父 / 91条 |

四类总数：SEAT_FULL 71、SEAT_HALF 70、BACK_FULL 77、BACK_HALF 68。训练大池四类为50/48/52/44，小池为22/23/24/22。大池增加的是35个独立父，同时条件数增加103；这只是数据覆盖差异，没有训练后的效果差异。`USER_LARGE_67`仅为可恢复清单标签，不是获准执行的LARGE实验。

新DEV来自官方train的独立留出，没有使用封存test；全部排除已知历史条件训练与旧DEV父组。官方生成器/D0是否预训练见过这些对象仍为UNKNOWN。预选64条的类型数为SEAT_FULL14、SEAT_HALF15、BACK_FULL19、BACK_HALF16，未强凑每类16；尚未物化评测Gaussian，不能称420请求已经正式冻结或执行。

## 实际做了什么

1. 完整读取任务书并从首次本地执行 `{reg['t0_utc']}` 起计时；12小时截止 `{reg['deadline_utc']}`。本次在数据门触发后提前停止科学任务，不把“12小时窗口”解释为必须空跑12小时。
2. 核对本地官方Chair全部3135条来源。额外检查三个官方小归档，共59,823,851字节；通用shapenet包的3135个Chair与本地逐文件相同，两份rebuttal无Chair，因此新增版本为0。通用split与类别split有59个UID分歧，未借此将test改为train。未重新下载大权重。
3. 只使用已有N128..256网格，保留原始FP32点、面ID、顶点ID与角点排列；无QEM、焊接、删面或修补。新索引预处理在一个唯一面身份样本上与原实现6组数组逐位相同；重复几何面用原始索引身份区分，不因旧几何匹配器不唯一就擅自删Y。
4. 对196版本各扫一次完整M，总3,786,998个唯一面对；原主检测器和独立检测器均核验，再依面ID分解CC/CF/FF。151版本有确认非法面对、11版本有未决面对；这不是全部151父被排除。**182条已保留条件仍带Y内部缺陷或未决风险**，没有把整个Y要求成无交网格。
5. 初始光滑面聚合常选到座下、背后、腿或局部条带。保留这些失败提案，补充可见性候选，再实际看完181父的46张输入表；另对25个历史标签可逐位映射的父看完7页。所有标签为AI_PROXY，不是人工认证或自动语义真值；HALF缺失/别名不凑数。
6. 最终将图审接受与局部CC/CF无非法/未决条件取交集，按既定版本顺序、同源组与父级划分物化286份C。独立CPU检查确认99父/286C身份与split关系，并对稳定选择的3条逐位核对C原角序、原始顶点/面ID、Y精确补集和2倍模型尺度。

## 已执行与未执行的边界

训练初始化、正式更新、隔离预检更新、生成器前向/反向、OT、D0前向、新Gaussian、新RAW/POST、新权重全部为 **0**。6000/4000共同步数没有进入实测选择阶段，`training_budget_lock.json`明确为BLOCKED且selected_steps=null。没有用历史权重冒充新臂，没有训练67父的替代实验。

训练、采样、单D0常驻、GEO_C、来源角序导出与原评分的独立入口已实现；CPU协议检查和阻塞门实际执行。**GPU路径为IMPLEMENTED_NOT_GPU_VALIDATED**。历史USER_MIX的12份RAW、12份D0 proposal与6份查询缓存身份已核验，但仅是可复用资产，不是本轮新对照结果。大权重仅引用在 `weights_manifest.json`。

没有生成420份RAW/POST，因此模型图集、训练曲线、AVMR/PMR、CF/FF/I、采用/接续视觉判断均NA。`results/`中的逐父/逐条件表是明确标记NOT_RUN的缺失台账，不是虚构评分；输入图库 `INPUT_REVIEW_GALLERY.html` 也明确不是生成效果图。

资源为实测RTX5080（16303MiB）、24逻辑CPU；来源扫描首实际例后使用2个CPU worker，每worker2线程，扫描阶段270.668秒。没有GPU实验worker，没有为测步时额外做一个不合格训练探针。至整理时端到端约{elapsed/60:.1f}分钟；重叠的扫描、构造、代码和图审时间不相加当作独占耗时。

## 限制与准备顺序记录

- 67是**这次实际构造并核验出来的训练父数**，不是本地只有67把椅子，也不是证明所有潜在标注方法最多只能得到67父。部分候选仍是区域需要重选，不应统称为脏椅子；清单保存逐项原因。
- 本轮在准备前采用“完整M非有限/精确零面积均不入候选，Y内部非退化相交可保留”的解释，未为凑数放宽。这个来源资格口径与未标注区域共同限制当前数量。
- 存在一项明确时序偏差：预先冻结了父候选排序，但先准备全体候选区域/图审，随后才选出最终32父DEV，部分预备条件物化也先于最终角色分配。最终父/组互斥且没有模型输出泄漏，但这不完全符合第2节“先划父再派条件”的顺序。已在 `preparation_order_note.json` 单列；**不宣称已完成一轮完全合规的训练实验**。后续如重新授权，须先冻结父角色，再派生正式条件，不能回填时间戳。
- 本次没有启动模型，不能用数据筛选结果判断小池过拟合、大池学习慢、某loss无效或历史USER_MIX退化。

## 对七个研究问题的回答

1. 候选小/大池是32/67个独立父的嵌套关系；不是复制旧25父。正式规定最低128父未达成。
2. 新父采用率及seat/back差异：NA，未训练/生成。
3. 是否真接续、是否另生主座面：NA，不能用来源C可见性代替生成采用。
4. CF/FF/I、细长面、删面、支撑代价：NA，没有生成结果。
5. 相对历史USER_MIX：NA；缓存资格通过不等于完成对比。
6. 拟合或覆盖机制：NA，没有训练诊断，不作归因。
7. **优先补齐可核验的源面区域与现成低面数Chair资产，使留出32DEV后训练父达到至少128，再执行相同定义的规模对照。** 当前没有证据支持先改loss、接合模块或扩大面数范围；这些均未自动启动。

## 文件与恢复

关键结果：`data/data_freeze.json`、`data/qualified_conditions.csv`、`data/parent_decisions.csv`、`data/condition_decisions.csv`、`data/coverage_matrix.csv`、`data/conditions/`、`data/quality/`、`data/reviews/`、`reviews/reviewer_existing.json`、`data/verification_data.json`。

全部输入原图本地保留；review ZIP含报告、逐例CSV、C数组、实际图审证据和代码快照，不含历史大权重与完整父数组。未嵌入文件在路径/hash清单中列出；小包不能独立重训。`README_RESUME.md`给出只读status/verify入口；start已实际返回2并在导入Torch/派发模型之前拒绝运行。

旧数据、权重、RAW、评分代码和默认模型未改；未commit、未push。数据阻塞交付完成后停止。
'''
    md(OUT/'REPORT.md',report)
    md(OUT/'README_RESUME.md','''# 恢复说明

状态：DATA_SCALE_BLOCKED。当前67训练父低于任务书128门槛，不能强行继续训练。

在 PowerShell / E:\\MeshFlow 中：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/pilots/user_data_scaling_12h_v1/resume_all.py --stage status
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/pilots/user_data_scaling_12h_v1/resume_all.py --stage verify
```

`start` 已实际验证在数据门返回2，不导入Torch、不执行GPU。没有可恢复的本轮训练检查点。

源数据清单、286份条件与181父输入图审已经保存；不应重新下载权重或重新扫描相同来源。GPU实现未验证，不能把入口存在当作已完成实验。现有窗口、停止状态和冻结文件不能通过改时间戳、强行扩噪声或改父名单绕过。

若未来用户另行授权足量数据的新任务，先解决来源/区域缺口及先划父后派条件顺序，再登记新窗口与新冻结数据；不得静默将它写回本轮。
''')
    reg.update(status='DATA_SCALE_BLOCKED',scientific_status='STOPPED_BEFORE_MODEL_INITIALIZATION',training_steps=None,
        result_data_freeze='data/data_freeze.json',finalized_utc=now.isoformat(),elapsed_seconds_at_packaging=elapsed)
    (OUT/'registration.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n','utf-8')
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(dict(event='DATA_SCALE_BLOCKED_DELIVERY',utc=now.isoformat(),**audit),ensure_ascii=False)+'\n')
    # Snapshot only this pilot; protected sources are references with existing audited hashes.
    diff=[];code_hashes=[]
    for p in sorted(CODE.glob('*')):
        if not p.is_file() or p.suffix not in ('.py','.md'):continue
        dst=OUT/'code_snapshot'/p.name;dst.parent.mkdir(exist_ok=True)
        if not dst.exists():shutil.copyfile(p,dst)
        assert sha(dst)==sha(p)
        code_hashes.append(dict(path=str(p),sha256=sha(p),snapshot=str(dst)))
        diff+=difflib.unified_diff([],p.read_text('utf-8-sig').splitlines(True),fromfile='/dev/null',tofile='b/'+p.relative_to(ROOT).as_posix())
    write(OUT/'source_hashes.json',dict(base_commit=reg['head'],new_code=code_hashes,protected_dependency_receipts=[
        'inventory/input_hashes.json','training_implementation_review.json','evaluation/asset_inventory.json'],note='Large immutable weight hashes are referenced; no repeated 7GB rehash.'))
    if not (OUT/'code.diff').exists():(OUT/'code.diff').write_text(''.join(diff),'utf-8')
    # Existing full PNG audit sheets are included; large source meshes/archives/weights remain local.
    zip_path=OUT/'R2_USER_DATA_SCALING_12H_V1_REVIEW.zip';allfiles=[p for p in OUT.rglob('*') if p.is_file() and p!=zip_path and '.tmp.' not in p.name]
    include=[];excluded=[]
    for p in allfiles:
        rel=p.relative_to(OUT)
        omit=(rel.parts[0]=='assets' or (len(rel.parts)>1 and rel.parts[:2]==('data','parents')) or (rel.suffix=='.png' and rel.parts[:2]==('data','review_sheets') and not rel.name.startswith('v2_')))
        item=dict(path=str(p),relative_path=rel.as_posix(),bytes=p.stat().st_size,sha256=sha(p),embedded_in_zip=not omit)
        (excluded if omit else include).append(item)
    write(OUT/'review_file_manifest.json',dict(status='DATA_SCALE_BLOCKED',included=include,excluded=excluded,weights='weights_manifest.json',standalone_training_package=False))
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for r in include:z.write(r['path'],r['relative_path'])
        z.write(OUT/'review_file_manifest.json','review_file_manifest.json')
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        required=['REPORT.md','README_RESUME.md','code.diff','data/qualified_conditions.csv','data/parent_decisions.csv','data/verification_data.json','review_file_manifest.json']
        assert all(p in z.namelist() for p in required)
    receipt=dict(status='PASS_ZIP_CRC_AND_REQUIRED_CONTENTS',zip=str(zip_path),bytes=zip_path.stat().st_size,sha256=sha(zip_path),file_count=len(include)+1)
    write(OUT/'delivery_receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
