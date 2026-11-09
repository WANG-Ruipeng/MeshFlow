"""Finalize Session A's confirmed insufficient-data handoff; never publishes READY."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('control_data_prepare', HERE/'data_prepare.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
RUN,DATA,REG=prep.RUN,prep.DATA,prep.REG


def main():
    assert not (RUN/'DATA_READY.json').exists(), 'Refuse to change a ready experiment'
    old=prep.read_json(RUN/'DATA_BLOCKED.json')
    assert old['experiment_id']=='control_mvp48_v1' and old['status']=='BLOCKED'
    audit=prep.read_json(DATA/'candidate_audit.json')
    short=prep.read_json(DATA/'selection_shortfall.json')
    independent=prep.read_json(DATA/'independent_shortfall_validation.json')
    assert independent['status']=='CONFIRMED_BLOCKED'
    for rec in independent['checked_files']:
        path=RUN/rec['path']
        if rec['path']=='DATA_BLOCKED.json' and (REG/'data_blocked_initial.json').exists():
            path=REG/'data_blocked_initial.json'
        assert prep.sha(path)==rec['sha256'],rec['path']
    assert short['shortages']==[dict(available_groups=6,required=8,role='test',shortfall=2)]
    for name in ('manifest.json','splits.json','patches.json','train_plan.json','validation_plan.json','test_plan.json'):
        assert not (DATA/name).exists(),name
    assert not list((DATA/'meshes').glob('*.npz'))
    if not (REG/'data_resources_after.json').exists():
        prep.write_json(REG/'data_resources_after.json',prep.resources())
    before=(REG/'data_resources_before.json').stat().st_mtime
    scan_end=(DATA/'candidate_audit.json').stat().st_mtime
    approximate=max(0.,scan_end-before)
    runtime=dict(schema='control_mvp48_blocked_runtime_v1',status='BLOCKED',
        gpu_workers=0,cpu_geometry_workers=1,cpu_threads_per_worker=1,
        independent_audit_overlapped_with_report_work=True,
        scan_group_patch_write_wall_seconds_approx=approximate,
        timing_scope='Difference of data_resources_before.json and candidate_audit.json file mtimes; includes scan, grouping, BFS and audit serialization; not precise CPU time or isolated benchmark.',
        approximate_source_objects_per_second=3135/approximate if approximate else None,
        selected_object_preprocess_calls=0,mesh_archives_written=0,plans_written=0,
        model_forwards=0,optimizer_updates=0,OT_calls=0,
        VRAM_process_peak='not applicable: this session dispatched no CUDA work',
        memory_measurement_scope='GPU snapshots are shared system usage; no inference/training per-worker peak inferred',
        independent_audit_report='data/independent_shortfall_validation.json')
    prep.write_json(REG/'data_runtime_blocked.json',runtime)
    import numpy as np
    good={s:[x for x in audit['objects'] if x['official_split']==s and not x['rejection_reasons']] for s in ('train','test')}
    face_summary={s:dict(count=len(xs),min=min(x['N'] for x in xs),median=float(np.median([x['N'] for x in xs])),max=max(x['N'] for x in xs)) for s,xs in good.items()}
    report=f'''# Control MVP48 — Session A 数据报告

**BLOCKED：未凑齐48对象；官方test仅6个合格独立对象，要求8个，缺2个。未创建DATA_READY.json，下游不得开始训练或评价。**

| 检查阶段（逐级） | 官方train | 官方test |
|---|---:|---:|
| 本地官方划分 | 2822 | 313 |
| 自然三角面数128–256 | 462 | 50 |
| 无重复源三角形 | 125 | 20 |
| 再要求无零面积/数值退化面 | 74 | 6 |
| 再要求适用连通C | 65 | 6 |
| 合并可发现同源/重复后的独立组 | 59 | 6 |

范围内排除原因可重叠：train重复源三角形337个、零面积/数值退化276个；另9个几何合格对象不能取得规定连通条件。test重复30个、零面积34个，两类重叠20个（互斥分解：仅重复10、仅零面积14、两者20、合格6）；独立复核确认30个都有源顶点索引重复的三角形，34个都有精确零面积面，容差额外排除为0。剩余6个全部能取得4个适用C。test数量不足发生在条件片与家族排除之前。范围外train/test另有2360/263个。不以严格水密或无自交筛选，未检查自交，未做修复。

合格候选的面数分布：train 65个，{face_summary['train']['min']}/{face_summary['train']['median']:g}/{face_summary['train']['max']}（最小/中位/最大）；test 6个，128/187/228。测试候选自然面数为128、136、168、206、212、228。完整48对象分布不存在。

32/8/8未冻结。`selection_shortfall.json`中的32/8/6仅是固定seed排序的失败诊断，不是可用划分。128个训练条件、16个验证条件、16个测试条件均未交付；逐对象NPZ、最终任务流和manifest均未生成。已在CPU审计候选起点、共享完整边BFS与接口适用性，逐候选失败原因保存在`candidate_audit.json`。异常跨known/free边必须拒绝patch，不能成为正常Edge标签；无面数、类别、测试数或官方split替换。

数据来源只读：`experiments/mf_h1_local0/assets/shapenet-03001627`，3135个本地官方UID。先按UID家族代理、精确几何、规范化几何和登记的近重复规则分组，再尝试划分。共2448组；282个共享UID末16位组、9个精确几何重复组。71个后缀组跨官方train/test，因此文件名或官方split本身不构成完全无泄漏证明。末16位仅是保守同源线索，没有可验证的祖先元数据。

近重复实际范围：全部128–256面对象加历史开发对象，共528个、139128对；轴对齐bbox规范化后的顶点集及面质心双向Hausdorff均≤0.01时合组，共3对。未做旋转/镜像对齐、语义家族或全部3135对象的近重复穷举。历史20个训练/诊断父对象及其分组闭包禁止进入验证/测试；P0/P1均112面，本轮不入选。证据见`registration/data_historical_audit.json`。不能据此声称完全无潜在近重复泄漏。

坐标约定已按实际代码登记：官方`ObjaverseDataset.__getitem__`加Native T1 flags；保存捕获的OT前坐标及最终模型坐标，模型坐标为`(pre_OT*2).astype(float32)`；同一完整对象一次官方预处理，C/free共享变换。原始几何、角/面映射、bbox与标签仅用于监督或评分。`registration/data_contract.json`已发布逐对象NPZ和加载器字段提案；本次在选集处停止，尚未执行48对象规范化、真实NPZ集成或B/C最终绑定，不能宣称坐标联调已完成。`load_sample.py`的局部RNG规则已做CPU检查，但真实数据集成未运行。

第一波规则已预登记为1000×8，128个训练对象/patch组合循环近均衡，Gaussian/t/排列及训练、验证、测试、绘图和模型初始化各用独立随机域；实际8000条任务、32个测试case与8个teacher case尚未生成或冻结。原定测试32case×3模型=96条生成、验证8case×3终点=24次t=.5只读teacher，不使用验证选checkpoint；本次实际模型调用、生成、训练更新和OT均为0。

曝光边界：拟定训练和验证来自官方train，可能已被官方无条件预训练看过；验证只能称“本轮条件微调未见”。拟定测试来自官方test；具体checkpoint的真实预训练内容尚未独立验证。历史双对象调参结果不能充当独立测试证据。

独立复核见`independent_shortfall_validation.json`：CONFIRMED_BLOCKED；3135份源文件再次哈希一致，512个范围候选及139128对近重复检查独立复现。该报告记录的初始DATA_BLOCKED快照原样保存在`registration/data_blocked_initial.json`，当前根状态已补齐本报告及全部交付哈希。保留候选审计、缺口清单、协议、字段提案、历史排除、CPU构造/检查脚本与加载器。未下载数据、改公共代码/权重、复制大权重、训练、评分选数据、reset/clean、commit/push。Session A写入限定在data/、registration/data_*、experiments/.../data_*以及用户要求的根DATA_BLOCKED.json。

资源：准备前RTX5080总16303MiB、用2235MiB、空闲13745MiB、利用率1%；WSL可用24 CPU、约61.3GiB RAM。实际0 GPU worker，1个CPU几何worker/1数值线程，独立只读审计与报告并行。扫描/分组/BFS/写审计约{approximate:.1f}秒（文件时间差，非隔离benchmark）；模型规范化、任务计划和GPU阶段未执行。前后资源快照及计时范围见registration/data_resources_*.json、data_runtime_blocked.json。

按失败契约停止，不放宽限制、不自动换split、不创建空READY。
'''
    prep.write_bytes(DATA/'DATA_REPORT.md',report.encode('utf-8'))
    initial=REG/'data_blocked_initial.json'
    if not initial.exists():prep.write_bytes(initial,(RUN/'DATA_BLOCKED.json').read_bytes())
    files={str(p.relative_to(RUN)):dict(sha256=prep.sha(p),bytes=p.stat().st_size) for p in sorted(DATA.rglob('*')) if p.is_file() and '.pending_' not in p.name}
    files.update({str(p.relative_to(RUN)):dict(sha256=prep.sha(p),bytes=p.stat().st_size) for p in sorted(REG.glob('data_*')) if p.is_file()})
    blocked=dict(schema='control_mvp48_data_blocked_v1',experiment_id='control_mvp48_v1',status='BLOCKED',owner='Session A',
        reason='Official test has only 6 eligible independent objects; 8 required, shortfall 2. Confirmed independently before any family/patch reduction.',
        shortages=short['shortages'],counts=audit['counts_by_filter'],eligible_face_distribution=face_summary,
        frozen_object_count=0,required_object_count=48,frozen_split=False,
        delivered_patch_counts=dict(train=0,validation=0,test=0),task_plans_frozen=False,
        report=dict(path='data/DATA_REPORT.md',sha256=prep.sha(DATA/'DATA_REPORT.md')),
        independent_audit=dict(path='data/independent_shortfall_validation.json',sha256=prep.sha(DATA/'independent_shortfall_validation.json')),
        independent_audit_snapshot_remap={'DATA_BLOCKED.json':'registration/data_blocked_initial.json'},
        original_failure_record='registration/data_blocked_initial.json',files=files,
        scripts={str(p.relative_to(prep.REPO)):prep.sha(p) for p in sorted(HERE.glob('data_*.py'))},
        no_ready_marker=True,model_forwards=0,optimizer_updates=0,OT_calls=0,
        downstream_action='Wait; do not train, validate or generate from the diagnostic partial selection.',
        stopping_condition='Candidate shortage. No automatic fallback or changed scientific definition.')
    content=prep.json_bytes(blocked)
    # Preserve the initial failure above; replace only this session-owned live status.
    if (RUN/'DATA_BLOCKED.json').read_bytes()!=content:
        pending=DATA/'data_blocked_final.pending_A'
        prep.write_bytes(pending,content)
        os.replace(pending,RUN/'DATA_BLOCKED.json')
    verify=prep.read_json(RUN/'DATA_BLOCKED.json')
    for path,rec in verify['files'].items():assert prep.sha(RUN/path)==rec['sha256'],path
    assert not (RUN/'DATA_READY.json').exists()
    print(json.dumps(dict(status='CONFIRMED_BLOCKED',test_available=6,test_required=8,shortfall=2,
        report=str(DATA/'DATA_REPORT.md'),blocked_sha256=prep.sha(RUN/'DATA_BLOCKED.json'),evidence_files=len(files))))


if __name__=='__main__':main()
