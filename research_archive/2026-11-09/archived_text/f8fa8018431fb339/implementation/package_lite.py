"""Build a <10,000,000-byte cloud review ZIP from small evidence only."""
from pathlib import Path
import csv
import hashlib
import io
import json
import time
import zipfile

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]
DELIVERY=ROOT/'deliverables'
DEST=DELIVERY/'CHAIR_CONTROL_USABILITY_QUICK_PILOT_V1_REVIEW_LITE.zip'

README='''**Chair Control Usability Quick Pilot V1：小于10MB的云端复核包**

本包按用户要求，只保留报告、完整逐输出/逐父CSV、实际评分源码、协议、身份及关键审计。前一版67.5MB完整包已删除；原实验数据仍在本地保留。

建议读取顺序：

1. results/QUICK_CONTROL_USABILITY_REPORT.md 和 task_contract.txt：本轮目标、真实执行账本、定义、完整结果与证据边界。
2. results/per_output_metrics.csv：全部80个唯一输出的完整标量指标，包括主面板64项及嵌套附加16项；两面板重合8项不重复计数。
3. results/per_condition_summary.csv、paired_parent_summary.csv：候选内均值、固定同winner的GT-oracle Best@2/4、全部8父配对分布。
4. results/scoring/primary_summary.csv、diversity_pairs.csv、sensitivity_checks.csv：配对均差/中位差/改善父/最差父、完整96候选对、固定4项加密复核。
5. results/multi_context_summary.csv、scoring/multi_context_global.csv、registration/conditions.json：2父嵌套条件、面积/片数/free预算，以及同一公共未观测区域的公平比较。
6. results/source_code/new/scoring.py、aggregation.py 和 source_code/original/：实际执行的新旧评分代码；定义、源hash和工程diff一并保留。

所有CSV保留完整行与列，没有只保留汇总或有利案例。包内另含请求登记、模型身份、调用账本、独立评分审计及空人评表。主要数值仍是HYBRID平均D下降11.70%但仅4/8父改善、F@1%下降、接口RMS改善而细容差贴合混合；Best@4不是部署选择器，也不是人类可用率。

**本精简包不含：** 权重、raw、NPZ/NPY、源网格、Gaussian数组、逐点距离向量、图库图片及HTML。原报告和代码保持原文，里面的绝对路径以及指向这些省略项的链接仅供本地追溯，不能在本包中直接打开。附带 omitted_materials.json 明确记录范围。本包支持标量结果/分母/汇总与代码定义复核，不支持重新计算原始几何距离或重新看图。不要假称已检查被省略的原始数据。

AI_PROXY文件只是上一轮实际看图后的定性记录；云端本次没有收到图片，不能把它当独立视觉核验或人工评阅。人工U_single/U_at4仍为NA，user_study=NOT_RUN。空表不是已执行的研究结果。

无需新增训练、模型前向、生成或重新评分。唯一优先建议仍待云端依据现有标量证据判断：先组织真实盲评，检验候选潜力能否转成用途收益，还是现有失败已足以改变下一步优先级。

bundle_manifest.json记录全部文件字节数和SHA256。文件已在ZIP内逐项核验；解压后运行 python verify_bundle.py 可再次只读校验。
'''

PROMPT='''请复核这个小于10MB的 CHAIR_CONTROL_USABILITY_QUICK_PILOT_V1 精简结果包。
先读 CLOUD_REVIEW_README.md、results/QUICK_CONTROL_USABILITY_REPORT.md，再读取完整逐输出、逐条件、逐父CSV及实际评分源码，不要只看报告摘要。

请判断：
1. 8父等权下HYBRID相对OT_BASE的平均质量、GT-oracle Best@2/4潜力、接口相容分别改善或退化多少？结合配对均差、中位差、改善父数与最差父。
2. 固定四候选的Best@4是否掩盖不稳定性？只根据D选择同一winner，检查它的F-score、接口、CF/FF，不能每列各选最佳。
3. 小角比例结合GT参照、候选多样性结合质量代价，应如何解读？不能把单一计数转成人类可用率。
4. C1⊂C2⊂C3在同一公共未观测区域上是否改善？确认排除复制C的计分捷径，并保留第二父C3的Gamma NA。
5. 现有证据更支持先开展真实用途盲评，还是先解决某类明确失败？只提出一个优先下一步及可证伪假说。

限制：本包没有原始数组、网格、逐点距离或图片；原报告引用的这些本地路径不在包内。可核对标量、汇总、定义和既有审计，不能声称重新算过几何或实际看过图片。AI_PROXY是此前定性记录，不是本次独立视觉证据。人工U_single/U_at4=NA。
不要新增训练、前向、生成，或重开MOMENT。新噪声不是新形状，本轮只是旧开发父对象上的探索性快测。
'''

VERIFY='''from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
manifest=json.loads((root/'bundle_manifest.json').read_text(encoding='utf-8'))
errors=[]
for row in manifest['files']:
    p=(root/row['path']).resolve()
    if not p.is_relative_to(root) or not p.is_file():
        errors.append(row['path']);continue
    data=p.read_bytes()
    if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
        errors.append(row['path'])
print(json.dumps(dict(status='PASS' if not errors else 'FAIL',files=len(manifest['files']),errors=errors)))
raise SystemExit(bool(errors))
'''


def main():
    started=time.perf_counter()
    assert DEST.resolve().is_relative_to(DELIVERY.resolve())
    if DEST.exists():raise FileExistsError(str(DEST))
    payload={}
    def add(relative):
        path=OUT/relative
        if not path.is_file():raise FileNotFoundError(str(path))
        payload['results/'+relative]=path.read_bytes()
    required=[
        'QUICK_CONTROL_USABILITY_REPORT.md','task_contract.txt','protocol.json','requests.csv',
        'per_output_metrics.csv','per_condition_summary.csv','paired_parent_summary.csv','multi_context_summary.csv',
        'sampling_accounting.json','human_ratings_template.csv','human_ratings_nested_template.csv',
        'blind_mapping.csv','report_ready_numbers.json','ai_proxy_visual_review.json','code.diff','delivery_manifest.json',
        'scoring/definitions.json','scoring/summary.json','scoring/primary_summary.csv','scoring/diversity_pairs.csv',
        'scoring/sensitivity_checks.csv','scoring/multi_context_global.csv','scoring/final_cpu_audit.json',
        'scoring/per_output.json','scoring/per_condition.json','scoring/cpu_ledger.jsonl',
        'scoring/tests/scoring_cpu_tests.json','gallery/qa.json','gallery/summary.json',
        'registration/checkpoints.json','registration/conditions.json','registration/construction_protocol.json',
        'registration/construction_trace.json','registration/cache_audit.json','registration/source_files.json',
        'registration/registration_summary.json','registration/independent_inputs_audit.json',
        'registration/final_scoped_source_audit.json','registration/generation_sources_before.json',
        'registration/new_code_before_generation.json','registration/generation_identity_during.json',
        'registration/revision_pre_sampling.json','source_code/source_hashes.json',
        'ledger/calls.jsonl','ledger/lifecycle.jsonl','ledger/stages.jsonl','ledger/reporting.jsonl',
        'resources/snapshots.jsonl','load_audits/OT_BASE.json','load_audits/OT_HYBRID.json',
        'engineering_revisions/resume_guard/receipt.json','engineering_revisions/resume_guard/validation.json',
        'engineering_revisions/resume_guard/code.diff','engineering_revisions/report_clarity/receipt.json',
        'engineering_revisions/report_clarity/code.diff',
    ]
    for relative in required:add(relative)
    for p in sorted((OUT/'source_code').rglob('*')):
        if p.is_file() and p.suffix in ('.py','.yaml','.yml'):
            add(p.relative_to(OUT).as_posix())
    for p in sorted((OUT/'outputs').glob('*/result.json')):
        add(p.relative_to(OUT).as_posix())
    for p in sorted((OUT/'replays').glob('*/result.json')):
        add(p.relative_to(OUT).as_posix())
    payload['CLOUD_REVIEW_README.md']=README.encode('utf-8')
    payload['CLOUD_REVIEW_PROMPT.txt']=PROMPT.encode('utf-8')
    payload['verify_bundle.py']=VERIFY.encode('utf-8')
    payload['implementation/package_lite.py']=Path(__file__).read_bytes()
    payload['replacement_cleanup.json']=(DELIVERY/'CHAIR_CONTROL_USABILITY_REVIEW_REPLACEMENT.json').read_bytes()
    omitted=dict(raw_arrays='OMITTED_BY_USER_REQUEST',weights='OMITTED',meshes='OMITTED',
        query_points_and_per_point_distances='OMITTED',images_and_gallery_html='OMITTED',
        original_local_data='RETAINED_UNCHANGED',geometry_recomputation_available=False,
        warning='Historical path references in preserved report/code/manifests do not promise these assets are in this lightweight package.')
    payload['omitted_materials.json']=json.dumps(omitted,ensure_ascii=False,indent=2).encode('utf-8')
    def count_csv(name):return len(list(csv.DictReader(io.StringIO(payload[name].decode('utf-8-sig')))))
    counts={k:count_csv('results/'+k) for k in ('requests.csv','per_output_metrics.csv','per_condition_summary.csv',
            'paired_parent_summary.csv','multi_context_summary.csv','scoring/diversity_pairs.csv','scoring/sensitivity_checks.csv')}
    assert counts['requests.csv']==counts['per_output_metrics.csv']==80
    assert counts['per_condition_summary.csv']==24 and counts['scoring/diversity_pairs.csv']==96
    assert counts['scoring/sensitivity_checks.csv']==4
    assert len(list((OUT/'outputs').glob('*/result.json')))==80
    forbidden={'.npz','.npy','.pt','.pth','.obj','.ply','.stl','.png','.jpg','.jpeg','.webp','.html'}
    assert not any(Path(name).suffix.lower() in forbidden for name in payload)
    manifest=dict(schema='chair_quick_cloud_review_lite_v1',scope='Full scalar tables, code and small audit evidence; no raw or images',
        byte_limit=10_000_000,table_rows=counts,human_usable=None,user_study='NOT_RUN',model_calls=0,
        files=[dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(payload.items())])
    payload['bundle_manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8')
    with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as bundle:
        for name,data in sorted(payload.items()):bundle.writestr(name,data)
    assert DEST.stat().st_size<10_000_000,DEST.stat().st_size
    with zipfile.ZipFile(DEST) as bundle:
        assert bundle.testzip() is None
        assert len(bundle.namelist())==len(set(bundle.namelist()))==len(payload)
        for name,data in payload.items():assert bundle.read(name)==data,name
    digest=hashlib.sha256(DEST.read_bytes()).hexdigest()
    DEST.with_suffix('.zip.sha256').write_text(digest+'  '+DEST.name+'\n',encoding='utf-8')
    DELIVERY.joinpath('CHAIR_CONTROL_USABILITY_CLOUD_REVIEW_README_LITE.md').write_text(README,encoding='utf-8')
    DELIVERY.joinpath('CHAIR_CONTROL_USABILITY_CLOUD_REVIEW_PROMPT_LITE.txt').write_text(PROMPT,encoding='utf-8')
    result=dict(status='PASS',zip_path=str(DEST),zip_bytes=DEST.stat().st_size,limit_bytes=10_000_000,
        sha256=digest,files=len(payload),table_rows=counts,raw_files=0,images=0,model_calls=0,
        all_zip_crc_and_payloads_verified=True,original_results_untouched=True,seconds=time.perf_counter()-started)
    DEST.with_suffix('.verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
