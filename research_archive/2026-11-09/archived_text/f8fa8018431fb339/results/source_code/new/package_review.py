"""Self-contained cloud review ZIP; copies real caches, excludes only checkpoint weights."""
from __future__ import annotations
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
from .quick_common import ROOT, OUT, CODE, resolve, read, write, sha


README = '''**Chair Control Usability Quick Pilot V1：云端完整复核包**

建议将本 ZIP 与 CLOUD_REVIEW_PROMPT.txt 一起提供给云端。先读 results/QUICK_CONTROL_USABILITY_REPORT.md，再看逐输出/逐父表与图库；不要仅依据本页摘要。

已经实际完成：80 个唯一输出（32 个旧缓存 + 48 个新生成），两条身份回放逐字节通过，2500 次模型前向；零训练、零反向、零 optimizer 更新，零采样失败。80 份 CPU 几何评分与旧512核对完成，96 个候选对齐全。人工评分未做，所有人类可用率为 NA。

数据入口：

|要复核的问题|包内实际文件|
|---|---|
|实验约束、模型身份与总账|results/task_contract.txt、protocol.json、sampling_accounting.json、ledger/、load_audits/|
|完整80输出和全部8父的分布|results/per_output_metrics.csv、per_condition_summary.csv、paired_parent_summary.csv|
|全部P/R/F、接口与GT-oracle同winner|上述CSV、results/scoring/primary_summary.csv、report_ready_numbers.json|
|全部候选实际网格、C、GT、噪声与mask|resolved_outputs.csv 中的 package_raw/package_condition/package_input；路径全部可在本包内解析|
|查询点与逐点距离|results/scoring/queries/、vectors/；NPZ字段见 array_inventory.csv|
|固定2父多片条件|results/multi_context_summary.csv、registration/construction_protocol.json、conditions.json、construction_trace.json|
|全量图库|results/gallery/technical/index.html、nested.html；包含C与GT|
|真实盲评|results/gallery/blind/ 整个目录可单独发送；不要发 blind_mapping.csv 或技术图库给评阅者|
|评分源码与完整原定义|results/source_code/new/、source_code/original/、scoring/definitions.json、code.diff|
|原缓存、原评分和源网格依据|reference_assets/ 下保留原目录结构；original_path_map.csv 连接原绝对路径与包内路径|
|轻量自检与最终独立核验|results/scoring/tests/、scoring/final_cpu_audit.json、registration/independent_inputs_audit.json、final_scoped_source_audit.json、gallery/qa.json|

results/ 保留实验实际输出，不改旧记录中的路径或哈希。旧记录中的 Windows/WSL 绝对路径只是来源身份；云端读取时请使用 original_path_map.csv 和 resolved_outputs.csv。32 个旧缓存 raw 和对应原评分/metadata 已实际打入 reference_assets/，不是只给路径。新 raw 包括51状态；原缓存的状态/运行合同以原metadata和两条本轮身份回放为证。

本包仅排除两个1.56GB冻结权重文件，不影响离线几何复核；完整文件SHA、模型状态SHA和严格加载记录均在包内，权重排除项列于 excluded_weights.json。未做自动训练、生成或模型重建。用于原始几何构造的8个repair_v2网格、冻结原512查询、旧噪声和本轮全部Gaussian也已包含。reference_assets 的历史任务清单可能包含本轮以外任务的路径，这些历史背景引用不代表本轮新增生成预算，也不保证那些额外历史任务全部装入本包。

当前主要结果：HYBRID四候选平均D降低11.70%，仅4/8父改善；F-score@1%由0.31636降到0.29005，接口RMS改善而细容差边贴合混合。GT-oracle Best@4比HYBRID自身平均D低44.93%，不是部署选择器；同winner的CF/FF也并非全面更好。增加片区时HYBRID公共RMS为0.04084→0.04822→0.03638，中间层退化。这些均不能转换成人类可用率。

AI_PROXY只是实际查看6个候选及4个GT的定性初筛，见 results/ai_proxy_visual_review.json；它没有给全量usable标签。gallery/summary.json中的AI_PROXY=NOT_RUN是渲染器当时的QA状态，后续定性初筛单独记录，并未修改渲染器原结果。

核验：先解压，再运行 `python verify_bundle.py`（仅Python标准库，读取/哈希文件，无模型或网络）。bundle_manifest.json 含每个载荷的SHA256及字节数，ZIP完整性也已实际验证。排除manifest自身的递归哈希。array_inventory.csv只列真实NPZ字段的shape/dtype，不自动运行模型。

这是旧开发父对象上的探索性快测，新噪声不是新形状。报告的唯一优先建议是对现有64候选做真实用途盲评，检验几何候选潜力是否转化成U_at4收益；尚未执行。不要将MOMENT重新纳入比较或把本包当新训练授权。
'''

PROMPT = '''请基于这个完整本地结果包，复核 CHAIR_CONTROL_USABILITY_QUICK_PILOT_V1。
先读取 CLOUD_REVIEW_README.md、results/QUICK_CONTROL_USABILITY_REPORT.md 和 task_contract.txt，再读取完整CSV及必要的实际网格/距离向量，而不是只看摘要或精选图。

请回答：
1. 8个父等权下，HYBRID相对同预算OT_BASE的平均质量、Best@2/4潜力和接口相容各有什么收益/代价？区分均值、中位差、改善父数、最差父。
2. 固定四候选中，Best@4是否掩盖候选质量不稳定？只按D选择同一winner，检查它的F、接口和CF/FF，不为每指标分别选最好候选。
3. 几何多样性是候选差异还是错误主导？小角应与同GT free并列，不把相交计数直接当用途结论。
4. C1⊂C2⊂C3对同一公共未观测区域的单向指标怎样变化？确认被复制的C没有进入距离目标；第二父C3的Gamma为NA。
5. 结合完整图库，哪些明确失败会妨碍“低多边形形状构思与后续选择/编辑”？实际看图才能给视觉结论，标记AI_PROXY，不伪造人评。
6. 当前更应做真实盲评，还是已经有足够证据优先修某类模型失败？只提一个优先下一步和可证伪假说，说明证据边界。

人工评分目前未执行，U_single/U_at4=NA；不使用距离阈值替代人类usable。
包内 resolved_outputs.csv 与 original_path_map.csv 是可移植路径入口；32旧缓存raw、48新raw、条件GT、噪声、冻结查询点/逐点距离、评分源码全部实际包含。大权重仅提供hash，本次云端复核只需离线分析，不应训练/前向/新增生成、不重开MOMENT。
'''

VERIFY = '''"""Validate all bundle payloads with standard library only; no model/network calls."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'bundle_manifest.json').read_text(encoding='utf-8'))
errors=[]
for row in manifest['files']:
    path=(root/row['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        errors.append((row['path'],'missing/unsafe'));continue
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    if path.stat().st_size!=row['bytes'] or h.hexdigest()!=row['sha256']:
        errors.append((row['path'],'size/hash mismatch'))
print(json.dumps(dict(status='PASS' if not errors else 'FAIL',files=len(manifest['files']),errors=errors),indent=2))
raise SystemExit(bool(errors))
'''


def csv_bytes(items):
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(items[0]) if items else [])
    writer.writeheader();writer.writerows(items)
    return stream.getvalue().encode('utf-8-sig')


def run(out=OUT):
    import numpy as np
    started=time.perf_counter();out=Path(out).resolve()
    target=(ROOT/'deliverables/CHAIR_CONTROL_USABILITY_QUICK_PILOT_V1_REVIEW.zip').resolve()
    if not target.is_relative_to(ROOT.resolve()):raise ValueError('Output outside workspace')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError('Preserve prior archive: '+str(target))
    payload={};origin={};excluded=[]
    def add(path,name,expected=None):
        path=resolve(path).resolve()
        if not path.is_file():raise FileNotFoundError(str(path))
        digest=sha(path)
        if expected and digest!=expected:raise ValueError('Source differs: '+str(path))
        if name.startswith('/') or '..' in Path(name).parts:raise ValueError('Unsafe archive path')
        payload[name]=dict(path=path,bytes=path.stat().st_size,sha256=digest)
        origin[str(path)]=name
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.suffix not in ('.tmp','.pyc') and '__pycache__' not in path.parts:
            add(path,'results/'+path.relative_to(out).as_posix())
    for row in read(out/'registration/source_files.json'):
        path=resolve(row['path'])
        if path.suffix.lower() in ('.pt','.pth','.ckpt'):
            excluded.append(dict(original_path=row['path'],sha256=row['sha256'],bytes=row['bytes'],
                                 reason='Frozen weights excluded from offline cloud analysis; full identity/load audit included'))
            continue
        add(path,'reference_assets/'+path.relative_to(ROOT).as_posix(),row['sha256'])
    # Include the exact newly added packaging implementation too.
    add(Path(__file__),'implementation/package_review.py')
    def package_path(value):
        path=resolve(value).resolve()
        if str(path) not in origin:raise ValueError('Required output dependency missing: '+str(path))
        return origin[str(path)]
    resolved=[]
    for request in read(out/'requests.json'):
        rid=request['request_id'];result=read(out/'outputs'/rid/'result.json')
        scored=read(out/'scoring/outputs'/(rid+'.json'))['metrics']
        resolved.append(dict(request_id=rid,uid=request['uid'],arm=request['arm'],candidate=request['candidate'],
            context_level=request['context_level'],panels=';'.join(request['panels']),generation_status=result['status'],
            N=request['N'],K=request['K'],checkpoint_sha256=request['checkpoint_sha256'],
            raw_sha256=result['raw_sha256'],package_raw=package_path(result['raw_path']),
            package_condition=package_path(request['condition_npz']),package_input=package_path(request['input_npz']),
            package_metrics=package_path(out/'scoring/outputs'/(rid+'.json')),
            package_vectors=package_path(scored['vectors_path']),
            package_result=package_path(out/'outputs'/rid/'result.json')))
    assert len(resolved)==80 and sum(r['generation_status']=='CACHED' for r in resolved)==32
    arrays=[]
    for name,item in payload.items():
        if name.endswith('.npz'):
            with np.load(item['path'],allow_pickle=False) as bank:
                for key in bank.files:
                    value=bank[key]
                    arrays.append(dict(package_file=name,array=key,shape=str(list(value.shape)),dtype=str(value.dtype)))
    mapping=[dict(original_path=path,package_path=name,sha256=payload[name]['sha256'],bytes=payload[name]['bytes'])
             for path,name in sorted(origin.items())]
    extras={'CLOUD_REVIEW_README.md':README.encode('utf-8'),
            'CLOUD_REVIEW_PROMPT.txt':PROMPT.encode('utf-8'),
            'verify_bundle.py':VERIFY.encode('utf-8'),
            'resolved_outputs.csv':csv_bytes(resolved),
            'original_path_map.csv':csv_bytes(mapping),
            'array_inventory.csv':csv_bytes(arrays),
            'excluded_weights.json':(json.dumps(excluded,indent=2,ensure_ascii=False)+'\n').encode('utf-8')}
    manifest=dict(schema='chair_control_usability_cloud_review_bundle_v1',status='COMPLETE',
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),unique_outputs=80,cached_raw_included=32,
        generated_raw_included=48,excluded_weights=2,model_calls_for_packaging=0,human_usable=None,user_study='NOT_RUN',
        files=[dict(path=name,sha256=item['sha256'],bytes=item['bytes']) for name,item in sorted(payload.items())]+
              [dict(path=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)) for name,data in sorted(extras.items())])
    extras['bundle_manifest.json']=(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
    pending=target.with_suffix('.pending.zip')
    if pending.exists():raise FileExistsError('Preserve unfinished archive: '+str(pending))
    with zipfile.ZipFile(pending,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as bundle:
        for name,item in sorted(payload.items()):bundle.write(item['path'],name)
        for name,data in sorted(extras.items()):bundle.writestr(name,data)
    with zipfile.ZipFile(pending,'r') as bundle:
        bad=bundle.testzip()
        if bad:raise ValueError('ZIP CRC failed: '+bad)
        for item in manifest['files']:
            data=bundle.read(item['path'])
            if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
                raise ValueError('Archived payload mismatch: '+item['path'])
        for row in resolved:
            for key,value in row.items():
                if key.startswith('package_'):assert value in bundle.namelist(),(key,value)
    pending.replace(target)
    digest=sha(target)
    target.with_suffix('.zip.sha256').write_text(digest+'  '+target.name+'\n',encoding='utf-8')
    # Small companion files allow the user to preview the handoff before upload.
    target.with_name('CHAIR_CONTROL_USABILITY_CLOUD_REVIEW_README.md').write_text(README,encoding='utf-8')
    target.with_name('CHAIR_CONTROL_USABILITY_CLOUD_REVIEW_PROMPT.txt').write_text(PROMPT,encoding='utf-8')
    result=dict(status='PASS',zip_path=str(target),sha256=digest,zip_bytes=target.stat().st_size,
                uncompressed_payload_bytes=sum(item['bytes'] for item in manifest['files']),
                files=len(manifest['files'])+1,raw_outputs=80,cached_raw_copied=32,new_raw_included=48,
                excluded_weight_files=excluded,all_crc_and_sha256_verified=True,
                all80_relative_dependencies_resolve=True,model_calls=0,seconds=time.perf_counter()-started)
    write(target.with_suffix('.verification.json'),result)
    return result


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False))
