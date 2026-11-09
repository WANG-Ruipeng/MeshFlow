"""Read completed receipts and hash retained files, without model imports."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def utc(t):return datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat()
def win(p):return 'E:/'+str(p).removeprefix('/mnt/e/')
def ref(p):return str(Path(p).relative_to(R))
OUT=R/'output';ART=R/'artifacts'
OUT.mkdir(exist_ok=True);ART.mkdir(exist_ok=True)
reg=read(R/'registration.json'); audit=read(R/'audits/training_completion.json')
weights=[]
for arm in ['PLAIN','LAYOUT']:
 status=read(R/'training'/arm/'status.json')
 assert status['status']=='COMPLETE' and status['calls']['failed']=={}
 for key in ['checkpoint','inference_export']:
  x=status[key];p=Path(x['path']);got=sha(p)
  assert got==x['sha256'] and p.stat().st_size==x['bytes']
  weights.append(dict(label=arm,role=key,path=str(p),windows_path=win(p),bytes=p.stat().st_size,sha256=got,
    local_step=x['local_step'],global_step=x['global_step'],retained=True,verification='Actual file size and streaming SHA256; no checkpoint deserialization',source_receipt=ref(R/'training'/arm/'status.json')))
src=Path(reg['source'])
for name,role in [(src.name,'source_full_recovery'),('generator_step3000.pt','source_inference_export')]:
 p=src.parent/name;meta=read(p.with_suffix('.json'));got=sha(p)
 assert got==meta['sha256']
 weights.append(dict(label='EXPOSURE',role=role,path=str(p),windows_path=win(p),bytes=p.stat().st_size,sha256=got,
  global_step=25500,retained=True,source_asset_read_only=True,verification='Actual file size and streaming SHA256; no checkpoint deserialization',source_receipt=str(p.with_suffix('.json'))))
assert next(w for w in weights if w['role']=='source_full_recovery')['sha256']==reg['source_sha256']
weights_doc=dict(schema='bench_layout_oracle_retained_weights_v1',status='VERIFIED',verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  weights=weights,new_endpoint_file_bytes=sum(x['bytes'] for x in weights if x['label']!='EXPOSURE'),
  scope='Fixed endpoints retained for reproducibility; no promotion/default switch. checkpoint_latest aliases/copies are not removed or counted as independent scientific endpoints.',
  existing_assets_modified=0,checkpoint_loads=0,model_calls=0)
(ART/'retained_weights.json').write_text(json.dumps(weights_doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

facts={'schema':'bench_layout_oracle_execution_ledger_v1','training':{},'sampling':{},'d0_geo':{},'cpu_batches':[],'source_receipts':{}}
for arm in ['PLAIN','LAYOUT']:
 s=read(R/'training'/arm/'status.json'); lines=[json.loads(x) for x in (R/'training'/arm/'calls.jsonl').read_text().splitlines()]
 facts['training'][arm]=dict(audit['arms'][arm],pid=s['pid'],attempt=s['attempt'],resources=s['resources'],
  first_formal_call_utc=utc(lines[0]['time_unix']),last_formal_call_utc=utc(lines[-1]['time_unix']),
  calls_wall_span_seconds=lines[-1]['time_unix']-lines[0]['time_unix'],
  calls=s['calls'])
for label in ['PLAIN1000','LAYOUT1000','LAYOUT_NULL']:
 raw=read(R/'evaluation'/label/'raw_complete_full.json');post=read(R/'evaluation'/label/'post_complete_full.json')
 sampdir=next((R/'evaluation'/label).glob('sampling_attempt*'))
 res=read(sampdir/'resources.json');dres=read(next((R/'evaluation'/label).glob('D0_resource_observer*/resource_observation.json')))
 receipts=[read(p) for p in (R/'evaluation'/label).glob('*/raw/receipt.json')]
 assert len(receipts)==12 and sum(x['new_generator_forwards'] for x in receipts)==600
 facts['sampling'][label]=dict(completion=raw,resources=res['summary'],observer_from_utc=utc(res['samples'][0]['time_unix']),
  observer_to_utc=utc(res['samples'][-1]['time_unix']),
  max_request_allocator_allocated_MiB=max(x['audit']['peak_allocated_bytes'] for x in receipts)/2**20,
  max_request_allocator_reserved_MiB=max(x['audit']['peak_reserved_bytes'] for x in receipts)/2**20)
 facts['d0_geo'][label]=dict(completion=post,resources=dres['summary'],observer_from_utc=utc(dres['samples'][0]['time_unix']),observer_to_utc=utc(dres['samples'][-1]['time_unix']),
  allocator_peak='NA_NOT_RECORDED_IN_COMPLETION',process_peak='NA_NVIDIA_SMI')
for p in sorted((R/'evaluation/completions').glob('*.json')):
 x=read(p);facts['cpu_batches'].append({k:v for k,v in x.items() if k!='files'})
facts['gallery']={k:v for k,v in read(R/'gallery/complete.json').items() if k!='protected'}
facts['score_collection']={k:v for k,v in read(R/'results/completion.json').items() if k not in ['source_rows','files','metric_fields']}
facts['engineering_events']=[read(p) for p in sorted((R/'engineering_events').glob('*.json'))]
for p in [R/'registration.json',R/'audits/training_completion.json',R/'results/completion.json',R/'gallery/complete.json',*list((R/'evaluation').glob('*/*complete_full.json')),*list((R/'evaluation/completions').glob('*.json')),*list((R/'dispatch').glob('*.json'))]:facts['source_receipts'][str(p)]=sha(p)
(OUT/'execution_ledger_evidence.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

lines=['# R2_BENCH_LAYOUT_ORACLE_V1 实际执行账本','',
'本账本仅从已完成调用日志、生产receipt及资源记录汇总。读取权重做流式SHA256核验，没有反序列化检查点、训练、模型调用或重评分。固定终点保留，不代表晋级。','',
'## 科学计算与阶段预算','',
'| 阶段 | 实际执行 | 原始记录 |','|---|---:|---|',
'| PLAIN训练 | 1000次更新、8000次前向、8000次反向、8000实例曝光 | training/PLAIN/status.json、calls.jsonl |',
'| LAYOUT训练 | 1000次更新、8000次前向、8000次反向、8000实例曝光 | training/LAYOUT/status.json、calls.jsonl |',
'| 新生成 | 3模式×12请求=36 RAW；Euler50共1800次逻辑向量场/生成器前向 | evaluation/*/raw_complete_full.json |',
'| 官方D0 | 36次单次D0，36次基础GEO_C；所有新RAW完成后才进入D0 | evaluation/*/post_complete_full.json |',
'| 新统一评分 | 36 RAW＋36 BASE_POST＋36 CLEAN_FINAL_POST=108条 | results/completion.json |',
'| 冻结参照 | EXPOSURE12请求×3阶段=36条历史评分，只读复用 | audits/reference_cache_audit.json |',
'| 图页 | 36页，每页4方法，共144方法阶段单元；CPU 2 worker | gallery/complete.json |','',
'两臂正式调用账本均无失败、无pending调用；没有生产重跑、额外更新或新噪声抽样。旧EXPOSURE没有重新生成或评分。LAYOUT_NULL使用同一个LAYOUT终点，并非第三条训练臂。布局支路包含在生成器前向内，未额外调用旧占用头。','',
'## 实际墙钟与吞吐','',
'以下为各阶段生产计时器的实际范围，不把多进程时间求和称为用户等待时间，也不将并行训练与单卡独占测速等同。数据构造、阅读、人工图审及全部收口没有统一启动/结束计时器，记为NA，不由文件时间猜算。','',
'| 阶段 | 秒 | 口径 |','|---|---:|---|']
for arm,x in facts['training'].items():
 lines.append(f"| {arm}训练 | {x['elapsed_seconds']:.3f} | 生产训练阶段；1000更新；实际重叠运行 |")
 lines.append(f"| {arm}正式更新计算累计 | {x['update_seconds_sum']:.3f} | 每次update计时之和；均值{x['update_seconds_mean']:.4f}秒/更新 |")
for label,x in facts['sampling'].items():lines.append(f"| {label} RAW12份 | {x['completion']['seconds']:.3f} | 含本阶段模型准备/采样的producer计时，{12/x['completion']['seconds']:.4f}输出/秒 |")
for label,x in facts['d0_geo'].items():lines.append(f"| {label} D0/GEO12份 | {x['completion']['seconds']:.3f} | 原producer D0/GEO阶段计时；不是纯D0 kernel时间 |")
for x in facts['cpu_batches']:lines.append(f"| CPU {x['mode']} {'+'.join(x['labels'])} | {x['seconds']:.3f} | {x['cells']}请求，{x['workers']} workers，包含相应评分/后端工作 |")
lines += [f"| 36页CPU渲染 | {facts['gallery']['seconds']:.3f} | 2 workers；不含后续实际图审 |",'',
'## 资源与并发的证据边界','',
'实际WSL设备为NVIDIA GeForce RTX 5080，nvidia-smi总显存16303MiB；24逻辑CPU，初始可用RAM约61.25GiB。首臂dispatch时整卡已用3503MiB、可用12477MiB、GPU利用率0%。PLAIN正式更新之后重新观察，第二臂dispatch时已用8057MiB、可用7923MiB、利用率25%、可用RAM约59.53GiB。新增worker的保守估计4750MiB，1.2倍余量加2048MiB安全边际，需要7748MiB，满足当时7923MiB可用量。该4750MiB是派发估计，不是已测进程显存峰。','',
'训练确实有两个独立PID（PLAIN491、LAYOUT761）重叠运行；每臂独立模型、优化器和冻结随机流。训练资源observer每2秒采样。nvidia-smi逐进程显存返回N/A，不能从整卡峰或PyTorch allocator峰推算独立进程峰。整卡观测含另一worker、CUDA/context和用户背景进程。未停止无关进程。','',
'| 训练臂 | allocator allocated峰 MiB | allocator reserved峰 MiB | observer整卡已用峰 MiB | 进程显存峰 | observer样本数 |','|---|---:|---:|---:|---|---:|']
for arm,x in facts['training'].items():lines.append(f"| {arm} | {x['peak_allocated_MiB']:.3f} | {x['peak_reserved_MiB']:.3f} | {x['observed_total_device_peak_MiB']:.0f} | NA（nvidia-smi不可得） | {x['resources']['sample_count']} |")
lines += ['',
'allocator为每个正式完整更新前重置后记录，再跨1000更新取最大；不是训练进程总VRAM。观察到的整卡最大12971MiB也不是连续采样保证捕获的硬上界。','',
'采样三个独立模式进程按PLAIN1000→LAYOUT1000→LAYOUT_NULL增量派发，单进程一次处理一份请求；前一组达到代表性输出后才观察/派发后续组。记录有前一进程退出与后一进程启动的短暂重叠，不能称三路全程同时采样或宣称三路并发峰。D0同样逐组增量派发，每个dispatch记录workers=1；记录包含前一进程收尾和下一进程准备重叠，但没有证据支持三路D0持续并发。CPU RAW评分可与D0独立工作重叠；CPU评分/后端每批最多2 workers，每worker1线程。','',
'| 模式 | RAW秒级observer整卡峰 MiB | RAW逐请求allocated/reserved峰 MiB | D0秒级observer整卡峰 MiB |','|---|---:|---:|---:|']
for label,x in facts['sampling'].items():
 lines.append(f"| {label} | {x['resources']['total_device_used_peak_mib']:.0f} | {x['max_request_allocator_allocated_MiB']:.3f}/{x['max_request_allocator_reserved_MiB']:.3f} | {facts['d0_geo'][label]['resources']['total_device_used_peak_mib']:.0f} |")
lines += ['',
'采样与D0 observer均1秒间隔，覆盖本次实际任务；逐进程显存全部NA。D0完成记录未提供独立allocator峰，记NA。RAW逐请求allocator值不是含模型加载及CUDA上下文的进程峰。各observer开始/结束UTC和全部计时字段保存在output/execution_ledger_evidence.json；不可将不同observer的峰相加。','',
'## 工程事件与保留资产','',
'CPU测试入口曾因维护包路径未插入而导入失败；只修测试harness后通过，生产模型/数据/评分未重跑。训练收口CPU汇总曾将paired[-50]/paired[-100]误写成元素而非尾部切片，引发TypeError；analysis/training_complete_v1保留partial，新版写training_complete_v2。该修正新增生产前向、反向、更新、评分均为0，不改冻结生产代码。事件详见engineering_events两份JSON。数据准备还修正过nullable review字段，均在冻结前且零模型调用。','',
'CPU reduced-width模型合同测试另列：最终15项测试计144次微型native block调用、14次layout模块调用、3次backward、2次输入VJP、2次微型AdamW更新；不是正式生产计算。此前14/15/15项CPU开发测试未逐项计数，明确NA；所有均为缩小CPU fixture，无生产检查点、无CUDA。见audits/model_cpu_gate.json。','',
'artifacts/retained_weights.json列出两臂各一个full恢复终点及一个推理导出，另列只读EXPOSURE full/export来源，实际读取文件size和流式SHA256核验通过；没有删除、覆盖任何权重。checkpoint_latest仍保留，不作为额外科学终点。','',
'本轮新增四个终点/导出文件总计 '+str(weights_doc['new_endpoint_file_bytes'])+' bytes。源码、原数据、原模型、旧RAW/POST与评分不动；未切默认、未commit/push。','',
'## 核验来源','',
'- audits/training_completion.json 与 training/*/status.json、calls.jsonl：正式训练计数及资源。',
'- dispatch/*_go.json：逐臂派发前实测与headroom。',
'- evaluation/*/raw_complete_full.json、post_complete_full.json：36 RAW/36 D0。',
'- evaluation/completions/*.json、results/completion.json：CPU实际批次、108新评分/36旧评分。',
'- evaluation/*/sampling_attempt*/resources.json、D0_resource_observer*/resource_observation.json：整卡资源监测。',
'- gallery/complete.json：渲染预算；visual_review/completed/*.json：单独实际图审记录。',
'- output/execution_ledger_evidence.json：本账本依赖的完成记录SHA256与细节。','']
(OUT/'EXECUTION_LEDGER.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'new_endpoint_bytes':weights_doc['new_endpoint_file_bytes'],'weights_verified':len(weights),'ledger':str(OUT/'EXECUTION_LEDGER.md'),'production_calls':0}))
