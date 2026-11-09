"""Append bounded completed evidence; preserve historical route counts/defaults."""
from pathlib import Path
import json,hashlib

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_three_mechanisms_v1'
def read(p):return json.loads(p.read_text('utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(OUT/'COMPLETION.json')['status']=='COMPLETE'
assert 'DRAFT / PENDING' not in (OUT/'REPORT.md').read_text('utf-8')
title='## 2026-10-30 Bench三机制快筛：未形成整体升级'
entry='''

## 2026-10-30 Bench三机制快筛：未形成整体升级

`R2_BENCH_THREE_MECHANISMS_V1`已完成用户授权的共享双面PATCH200、训练式LOOP200和局部RL16；同EXPOSURE/global25500起点、30 NATURAL父，普通续训PLAIN200/267对照。监督667更新；RL16机会、64策略轨迹、14更新/2跳过。6父各原2噪声，72新RAW、72单次官方D0、相同GEO_C/来源定向→HINGE_OPP015→ALPHA→HOLE；PATCH另有明示的D0后共享恢复。252行177指标、36对照页和固定截面、252实际非盲AI_PROXY记录齐全。

FINAL父宏均值：PLAIN200/PATCH200的FF9.785/21.009%、I(C)9.078/6.779%，方向C可达12.032/4.281%；PATCH平均30条RAW强制内部边，到FINAL仅11.917条合法，不能把结构保证算成学到的接续。LOOP相对近似等block PLAIN267，FF10.662→13.409%、CF5.078→2.992%、I7.373→2.360%，free面积12.560→9.213；新增a8两噪声7条合法C/F边，DEV/df6仍无合法C/F边。f495/n0后处理清除了替代灰座，RAW仍替代；不将删面后的低I冒称生成器RAW采用提升。RL16相对同随机过程RL_ZERO，FF10.204→10.632%、CF7.424→7.708%、I9.416→9.623%，合法C/F接口身份与条数不变。

三臂均不替换EXPOSURE或正式默认；保留LOOP的有限连接/后处理用途信号，不自动长训。PATCH不等于完整学习关系头，LOOP不等于旧冻结半残差或完整LiFT，局部RL不同于旧R29/R30但本配置无稳定收益。短预算、小开发面板不否定整个机制家族。CF/FF按受影响free面，I是C查询域；RAW soup单面分量不直接等于视觉碎片。原标签差异、支持域/NA、质量成本和工程失败均保留。

新增实际权重/AdamW/RNG、逐父配对、实际评分代码、code.diff、source_hashes及只读总核验入口齐全。一次PLAIN工程错误多1成功前向、0反向/更新，显式计账；评测资源结构和RL文件/状态hash修复均在模型调用前完成。无OOM、重抽、默认切换、清理、commit或push。固定范围已完成停止。

[完整报告](experiments/r2_bench_three_mechanisms_v1/REPORT.md) · [全图集与截面](experiments/r2_bench_three_mechanisms_v1/gallery/index.html) · [逐输出177字段](experiments/r2_bench_three_mechanisms_v1/results/per_output.csv) · [实际图审与独立不确定性修订](experiments/r2_bench_three_mechanisms_v1/visual_review/summary_v2/actual_observations.csv) · [恢复与核验](experiments/r2_bench_three_mechanisms_v1/RUNBOOK.md)。
'''
receipt=OUT/'memory_update.json'
if receipt.exists():
    print(json.dumps(dict(status='ALREADY_COMPLETE')))
    raise SystemExit
changes=[]
for path,content in ((ROOT/'WORKSPACE.md',entry),
    (ROOT/'r2/docs/research_route_memory.md',entry.replace('](experiments/','](../../experiments/'))):
    before=path.read_bytes();before_hash=sha(path)
    if title in before.decode('utf-8-sig'):raise RuntimeError('Unexpected prior memory entry')
    path.write_bytes(before+content.encode('utf-8'))
    changes.append(dict(path=str(path),before_sha256=before_hash,after_sha256=sha(path),operation='append_only'))
path=ROOT/'r2/docs/research_route_registry.json';before=sha(path);doc=read(path)
follows=doc.setdefault('bench_followup_experiments',[])
assert not any(x.get('id')=='R2_BENCH_THREE_MECHANISMS_V1' for x in follows)
follows.append(dict(id='R2_BENCH_THREE_MECHANISMS_V1',date='2026-10-30',status='COMPLETE_NO_OVERALL_PROMOTION',
    nearest_prior=['R07','R10','R29','R30','R2_BENCH_LOOP_MOTIF_MINI_V1'],
    substantive_change='PATCH explicit shared free states/restricted OT; trained final-block loop with two FM readouts; local action LoRA/RLOO with frozen-reference quality guards.',
    budget=dict(PLAIN=267,PATCH=200,LOOP=200,RL_opportunities=16,RL_updates=14,RL_skips=2),
    evaluation=dict(parents=6,requests=12,new_RAW=72,new_D0=72,metric_rows=252,fields=177,actual_AI_PROXY_cells=252),
    decision='No default promotion. PATCH/RL versions stop; retain bounded LOOP connection/postprocessing-use observations with FF/area/structural costs. No fresh-generalization claim.',
    report='experiments/r2_bench_three_mechanisms_v1/REPORT.md',report_sha256=sha(OUT/'REPORT.md'),
    authorization_boundary='Completed fixed scope only; no automatic extension, cleanup, default change, commit or push.'))
path.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
changes.append(dict(path=str(path),before_sha256=before,after_sha256=sha(path),operation='new_followup_key_historical_counts_unchanged'))
receipt.write_text(json.dumps(dict(status='COMPLETE',changes=changes,report_sha256=sha(OUT/'REPORT.md')),
    ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps(dict(status='COMPLETE',files=len(changes))))
