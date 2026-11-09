"""Validate completed evidence and write the bounded experiment report.

No neural imports, no new geometry/scoring, no scientific retries.
"""
from __future__ import annotations
import csv,difflib,hashlib,json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_chair_back_transfer_v1'
CODE=ROOT/'r2/pilots/chair_back_transfer_v1'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def write_text(p,s):
    with Path(p).open('x',encoding='utf-8') as f:f.write(s)
def pc(x):return f'{100*x:.2f}%'
def local(s):
    s=str(s)
    return ROOT/Path(s.split('/mnt/e/MeshFlow/',1)[1]) if s.startswith('/mnt/e/MeshFlow/') else Path(s)

def main():
    m=read(OUT/'eval_manifest.json');r=read(OUT/'registration.json')
    digest=read(OUT/'metrics_digest.json');ledger=read(OUT/'execution_ledger.json')
    vis=read(OUT/'visual_review/completion.json');accept=read(OUT/'cpu_post_acceptance.json')
    assert len(m['requests'])==16 and sha(OUT/'eval_manifest.json')==r['eval_manifest_sha256']
    assert ledger['generator_calls_returned']==800 and ledger['head_calls_returned']==8 and ledger['D0_calls_returned']==16
    assert ledger['training_updates']==ledger['backwards']==ledger['failed_worker_attempts']==0
    assert accept['status']=='PASS' and vis['actual_review_rows']==64
    for name,h in r['scientific_code_hashes'].items():assert sha(local(name))==h,name
    rows=list(csv.DictReader((OUT/'evaluation/tables/per_output.csv').open(encoding='utf-8-sig')))
    assert len(rows)==80 and len({(x['request_id'],x['stage']) for x in rows})==80
    visuals=list(csv.DictReader((OUT/'visual_review/per_stage.csv').open(encoding='utf-8-sig')))
    byvis={x['request_id']:x for x in visuals if x['stage']=='FINAL_POST'}
    outrows={}
    for q in m['requests']:
        rid=q['request_id']
        for stage in ('RAW','BASE_POST','HINGE_OPP015','ALPHA','FINAL_POST'):
            outrows[(rid,stage)]=read(OUT/'evaluation/scores/S2000'/rid/stage/'row.json')
        for path in (OUT/'evaluation/S2000'/rid/'raw/receipt.json',OUT/'evaluation/S2000'/rid/'post/receipt.json',
            OUT/'evaluation/S2000'/rid/'cpu/FINAL_POST/complete.json',OUT/'gallery/S2000_BACK_FULL'/f'{rid}.png'):
            assert path.exists(),path
    d={(x['split'],x['stage']):x for x in digest}
    conditions=[];seen=set()
    for q in m['requests']:
        if q['condition_id'] in seen:continue
        seen.add(q['condition_id']);conditions.append(q)
    table_conditions='\n'.join(f"| {q['split']} | {q['parent_id'][:8]} | {q['K']} | {q['N']} | {q['input_facts']['C_boundary_edges']} |" for q in conditions)
    table_stages=[]
    for stage in ('RAW','BASE_POST','HINGE_OPP015','ALPHA','FINAL_POST'):
        x=d['ALL',stage]
        table_stages.append(f"| {stage} | {pc(x['CF_illegal_affected_free_fraction'])} | {pc(x['FF_illegal_affected_free_fraction'])} | {pc(x['ConflictCov_C'])} | {pc(x['free_only_reached_bulk_area_fraction'])} | {x['bulk_ge30_outputs']}/16 | {pc(x['free_minimum_angle_lt5_fraction'])} |")
    table_groups=[]
    for split in ('FAMILIAR','DEV','ALL'):
        a=d[split,'BASE_POST'];b=d[split,'FINAL_POST'];v=vis['final_counts'][split]
        table_groups.append(f"| {split} | {v['back_role_adoption'].get('YES',0)} / {v['back_role_adoption'].get('UNCERTAIN',0)} | {a['bulk_ge30_outputs']} → {b['bulk_ge30_outputs']} / {b['outputs']} | {pc(a['free_only_reached_bulk_area_fraction'])} → {pc(b['free_only_reached_bulk_area_fraction'])} | {pc(b['FF_illegal_affected_free_fraction'])} | {pc(b['ConflictCov_C'])} |")
    table_outputs=[]
    for q in m['requests']:
        rid=q['request_id'];a=outrows[rid,'BASE_POST'];b=outrows[rid,'FINAL_POST']
        table_outputs.append(f"| [{rid}](gallery/S2000_BACK_FULL/{rid}.png) | {byvis[rid]['back_role_adoption']} | {pc(a['free_only_reached_bulk_area_fraction'])} → {pc(b['free_only_reached_bulk_area_fraction'])} | {pc(b['FF_illegal_affected_free_fraction'])} | {pc(b['ConflictCov_C'])} |")
    report=f'''# Chair BACK_FULL 冻结迁移检查

状态：完成。日期：2026-10-19。实验：`R2_CHAIR_BACK_TRANSFER_V1`。

**结论：当前 Chair SEAM 能在多数被检查的靠背输入上生成可辨认的椅身，并让 C 承担靠背用途；但座面强结果中的主体接续能力尚未稳定迁移。** 实际图审14/16靠背用途为 YES、2/16不确定；最终只有2/16达到预登记的 bulk 可达面积≥30%。这两类证据分开保存，不把“有合法 C/F 边”当成完整接续成功。

此次只检查完整靠背，没有运行椅腿、扶手或其他新部位。没有训练、选新权重或改变默认模型。

## 固定输入与证据边界

从现有 BACK_FULL 条件中，先核对 C 连贯性、来源拓扑/绕序、非退化与可逆身份，要求 K≥2。候选按 `(K, parent_id, condition_id)` 排序，取四分位固定秩；没有按生成结果或来源 Y 质量挑选。熟悉组可用31条，DEV可用18条，各取4条，每条预先保存2份新噪声。5条不连贯输入在生成前排除，原因见 `eligibility.json`。全部8条 C 已在生成前实际查看。

| 组别 | 父ID前缀 | K | N | C开放边数 |
| --- | --- | ---: | ---: | ---: |
{table_conditions}

FAMILIAR表示父对象属于TRAIN47；其BACK_FULL可能已经在早期EXP67训练中出现，不能称“全新条件泛化”。DEV未进入条件适配训练，但官方预训练是否见过这些父为UNKNOWN。每父一个C、两个噪声，规模只适合范围检查。完整来源模型不作为唯一应当生成的款式。

## 固定模型、处理与执行

生成器为保留的 `PLAIN_S2000_generator.pt`（这里的PLAIN仍是SEAM表示；live global15000），布局为冻结 `LONG_occupancy.pt`，logit阈值0。新8条C都实际经过C-only头，137条开放边全部激活；这不是owner准确率，也没有使用来源Y布局。

固定流程：标准Gaussian → Euler50 → 单次官方Chair D0 → 恢复C/seam锚点 → GEO_C和来源绕序恢复 → HINGE_OPP015 → ALPHA → HOLE。`FINAL_POST`是最后一个阶段。FP32参数/积分、BF16前向、严格确定性运行合同保持。

实际执行800次生成器前向、8次小头前向、16次D0前向；0训练更新、0反向、0失败工作进程、0科学重试。全部RAW保存后才开始D0。模型、输入、噪声、51个积分状态的锚点和导出索引经独立CPU验收PASS。

资源：RTX5080 16GiB、24 CPU核；先测首个授权任务再派第二个8请求分区。两个RAW进程的实际rollout时间区间没有重叠，不声称并行加速。D0一个进程加载一次处理16份，CPU评分/后处理2workers。稀疏整卡快照最高占用4379MiB、最低空闲11601MiB；不是连续进程峰值。RAW allocator最大reserved618MiB、D0为1812MiB，不包含完整驱动/其他进程显存。16次RAW积分合计51.45秒，D0阶段含加载/保存52.88秒；CPU后处理186.33秒、80阶段评分55.83秒、图库7.06秒（部分并行，不能直接当隔离benchmark相加）。

账本：[execution_ledger.json](execution_ledger.json)。工程预检修正记录在 `audits/score_preflight_before_execution/`，发生于正式评分前，原定义/输入/旧结果保留。

## 完整数值结果

以下为8父等权均值（本面板每父2份，亦等于逐输出均值）。CF、FF是非法面对所影响的free面占比；I是固定C查询上的ConflictCov，含横穿/近重复覆盖，不是精确相交面积。I两分项可能重叠，不可直接相加。≥30%仅为预登记的描述阈值，不能认证整个接缝封闭或物体可用。

| 阶段 | CF↓ | FF↓ | I↓ | seam→原bulk可达面积 | 可达≥30% | 小角<5°↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table_stages)}

RAW三角汤没有GEO建立的自由共享索引，RAW图可达为0不能单独当作几何没有相邻关系。RAW的FF实际为89.44%；BASE大幅降低包含D0及GEO的共同效果，不能归功于其中单独一个操作。

| 组别 | 图审用途YES / 不确定 | BASE→FINAL ≥30% | 平均bulk可达 BASE→FINAL | 最终FF | 最终I |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table_groups)}

全部16份BASE/FINAL均有合法且反向绕序的C/F接口，但多数仍只连到小片seam。两个≥30%的最终输出为 **DEV f199965d/n1（99.66%）和9d81bc69/n0（95.21%）**；后者在HINGE阶段新增达到阈值。其余14份均未达，熟悉组为0/8。不能由DEV均值较高反推DEV整体更易或已泛化，只是两个大连通分量抬高均值。

额外CPU后处理的实际收益：FF 21.06%→18.83%，CF 7.46%→6.73%，I 21.26%→21.18%；合法图单面孤岛13.31→9.88个/请求，原始边图单面孤岛8.31→5.31。非流形边11.13个/请求不变，GEO删free面14.19张/请求不变。

HINGE接受69次、ALPHA39次，HOLE添加4张桥面。BASE原有2567张面全部保留；原合法CF边234/234、FF边2136/2136保留；最终新增合法CF边1条、FF边148条。按原检测器定义，新增非法CF/FF面对均为0，旧非法CF面对248→229、FF443→400。固定BASE面分母的最终FF为18.855%，因此收益不是只用4张新增面稀释分母。4个请求平均形状因子略恶化，不能称“所有指标零退化”。桥面仅来源映射ID为−1，输出面ID仍唯一非负。

完整179个数值字段、80阶段行均保存在 [per_output.csv](evaluation/tables/per_output.csv)；另有逐条件、逐父、配对增量、面对和合法边记录。验收：[cpu_post_acceptance.json](cpu_post_acceptance.json)。

## 实际图审与截面

16页全部实际打开查看，每页4阶段、两个相对表面视角和前/侧固定位置截面；共64条阶段记录。评阅身份是非盲AI_PROXY，非人类盲评。颜色蓝=C、灰=生成；相同C跨噪声/阶段共用尺度和剖切位置。图片切线不自动等同真实裂缝，所有精确拓扑/水密判断在视觉记录中保留UNKNOWN，并用独立几何评分解释。

14份能明确辨认C的靠背角色，2份fb8c保留不确定：灰色弧背遮挡蓝色平板、上角突出，侧面存在分层关系。背板前后层也可能是合理厚度，本轮不凭截图宣布所有灰背层非法。16份都能辨认座面、腿/支撑与靠背；整体合理性13份YES、3份不确定。

熟悉组的整板、斜板、条栅和曲面靠背均有可辨认用途，但f4f/n1的蓝板伸到座下、b631/n1有横跨蓝背的灰条片/厚箱体，8c/n0有腿间小悬片候选。DEV的abbe/f199仍有背缘薄片和侧面间隔。9d81/n1填充了原C框的中心开口；没有给定空域禁止条件，因此不自动判作失败。

BASE→FINAL大部分页面在当前视图尺度下外观接近；f199/n1可见局部小缺口收拢。未观察到明显新主体破损，不等于没有任何局部形状代价。图审结果不由指标阈值生成。

[完整图库与截面](gallery/S2000_BACK_FULL/index.html) · [64条图审](visual_review/per_stage.csv) · [逐请求视觉变化](visual_review/paired.csv)

| 请求（点击看四阶段） | 最终靠背用途 | bulk可达 BASE→FINAL | 最终FF | 最终I |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(table_outputs)}

## 此轮决定与限制

**保留“Chair完整靠背有可辨认用途与少量真实主体接续”的结果，不能写成各种部位都具备稳定控制。** 后处理在这个新面板仍减少非法面对、保住原连接，但最终I仍高，连接覆盖仍有限。跨部位输入值得继续研究，现有证据不足以承诺和座面强结果一样。

如果下一轮继续，我优先检验“SEAM阶段的条件部位覆盖是否限制靠背接续”：将已有合法BACK_FULL加入固定训练任务，并保留本面板与座面回归作预先固定的比较。待检验假说是：在不加大后处理位移预算的情况下，背部任务曝光能提高seam与bulk的共同几何组织，而不只是增加被结构锁定的C/F边。此轮没有开始该训练，也没有由单终点检查证明原因就是遗忘或训练不足。

没有比较早期EXP67在相同新噪声上的结果，没有统计跨所有Chair父或椅腿等部位的成功率；官方预训练暴露UNKNOWN。没有把来源GT款式作为成功标准，没有重新抽噪声、调头或挑输出。

## 文件与恢复

- 输入/噪声：`eval_manifest.json`、`inputs/`、`queries/`；manifest SHA256 `{r['eval_manifest_sha256']}`。
- 所有产物：`evaluation/S2000/<request>/raw/`、`post/`、`cpu/`，原RAW/BASE保留。
- 数值：`evaluation/tables/`、`evaluation/scores/`、`key_per_output.csv`。
- 图像/审阅：`gallery/S2000_BACK_FULL/`、`visual_review/`。
- 实现：`r2/pilots/chair_back_transfer_v1/`；源码身份 `source_manifest.json`、新增代码 `code.diff`。
- 恢复入口：`python -m r2.pilots.chair_back_transfer_v1.resume` 默认只显示状态；`--execute`仅补已冻结任务的缺失阶段，未决神经调用禁止自动重放。本轮已完成，不需要重新执行。

此轮无新权重、无清理、无默认切换、无commit/push。完成固定任务后停止。
'''
    write_text(OUT/'REPORT.md',report)
    codefiles=sorted([p for p in CODE.rglob('*') if p.is_file() and p.suffix in ('.py','.md')])
    patch=''.join(''.join(difflib.unified_diff([],p.read_text(encoding='utf-8-sig').splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+p.relative_to(ROOT).as_posix())) for p in codefiles)
    write_text(OUT/'code.diff',patch)
    source=dict(experiment='R2_CHAIR_BACK_TRANSFER_V1',new_code={p.relative_to(ROOT).as_posix():sha(p) for p in codefiles},
        inherited_scientific_sources=r['scientific_code_hashes'],model_hashes=r['model_hashes'],
        score_registration_sha256=sha(OUT/'evaluation/score_registration.json'),code_diff_sha256=sha(OUT/'code.diff'),
        note='New-file diff only, not a Git commit. Full inherited scoring source identities remain in score_registration.json; no model copies.')
    write(OUT/'source_manifest.json',source)
    artifacts=['REPORT.md','eval_manifest.json','registration.json','data_registration.json','execution_ledger.json','metrics_digest.json',
       'key_per_output.csv','cpu_post_acceptance.json','visual_review/completion.json','visual_review/per_stage.csv','visual_review/paired.csv',
       'evaluation/tables/completion.json','evaluation/tables/per_output.csv','gallery/S2000_BACK_FULL/completion.json','source_manifest.json','code.diff']
    complete=dict(status='COMPLETE',experiment='R2_CHAIR_BACK_TRANSFER_V1',utc=datetime.now(timezone.utc).isoformat(),
        conditions=8,requests=16,RAW=16,BASE_POST=16,CPU_variants=48,scored_rows=80,gallery_pages=16,actual_visual_rows=64,
        generator_forwards=800,head_forwards=8,D0_forwards=16,training_updates=0,backwards=0,scientific_retry=0,
        model_changed=False,default_changed=False,cleanup=False,commit=False,push=False,
        scientific_source_hash_recheck='PASS',cached_cpu_acceptance='PASS',
        artifacts={s:sha(OUT/s) for s in artifacts},stop='Fixed check complete. No additional experiment dispatched.')
    write(OUT/'completion.json',complete)
    print(json.dumps({k:v for k,v in complete.items() if k!='artifacts'},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
