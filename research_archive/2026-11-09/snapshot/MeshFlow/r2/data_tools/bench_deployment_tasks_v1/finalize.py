"""Seal actual data artifacts and blockers; never turns staging into training."""
from collections import Counter
import difflib
import json
import time
from .prepare import OUT, ROOT
from r2.data_tools.bench_user_v2 import build as b


def main():
    if (OUT/'COMPLETE.json').exists():
        from .run import verify
        verify();return
    d=OUT/'dataset/source_tasks_pending_adapter'
    receipt=b.read(d/'COMPLETE.json')
    summary=b.read(d/'summary.json');manifest=b.read(d/'manifest.json')
    rows=manifest['rows'];accepted=[r for r in rows if not r['hold_reasons']]
    groups=Counter(summary['accepted_staging_groups'])
    current=ROOT/'experiments/r2_bench_user_relation_rebuild_v1/proposed_merged_manifest.json'
    current_m=b.read(current)
    b.save(OUT/'dataset/reused_reference.json',dict(source_manifest=str(current),source_sha256=b.sha(current),
        train=current_m['train'],dev_regression=current_m['dev'],
        input_layout_ready=False,reason='Historical original records reused only; missing production caches and incompatibility remain.',
        no_new_parent_claim=True))
    status=dict(status='DATA_PREPARATION_CLOSED_WITH_TRAINING_BLOCKERS',training_ready=False,
        old_reference_parents=26,old_reference_conditions=91,new_source_candidates=16,
        new_source_plausible=15,new_whole_source_held=1,new_lowN_targets=0,
        new_source_task_candidates=len(rows),source_task_accepted=len(accepted),
        source_task_held=len(rows)-len(accepted),source_parents_with_accepted_tasks=len({r['parent_id'] for r in accepted}),
        accepted_staging_groups=dict(groups),source_task_N_range=summary['N_range'],source_task_K_range=summary['K_range'],
        production_train=[],new_model_ready_parents=0,new_model_ready_conditions=0,
        closed_C_existing_Bench_wrapper_supported=True,
        source_summary_closed_C_adapter_caveat_superseded_by=str(OUT/'source_task_verification/closure_caveat.json'),
        blocks=['No new complete target satisfies the unchanged N128..256 contract after the frozen conservative construction.',
                'Open-C deployment layout and faithful target packing not validated for all tasks; do not replace with source-owner. Closed C uses the existing empty Bench layout.',
                'Training initialization/ancestry, budget and generation noise remain unregistered; this turn authorizes data only.'],
        task_manifest=str(d/'manifest.json'),task_manifest_sha256=b.sha(d/'manifest.json'),
        parent_split=str(OUT/'split_audit/new_data_split.json'),parent_split_sha256=b.sha(OUT/'split_audit/new_data_split.json'))
    b.save(OUT/'DATASET_STATUS.json',status)
    b.save(OUT/'call_ledger.json',dict(neural_forward=0,generator_forward=0,occupancy_head_forward=0,D0_forward=0,
        backward=0,optimizer_updates=0,new_generation_noise=0,new_generated_meshes=0,
        CPU_target_proposals=16,CPU_source_task_pairs=len(rows),CPU_actual_review_pages=len(rows),
        CPU_source_figure_pages=32,source_figure_note='Initial16 plus legibility revision16, both preserved.',
        geometry_worker_count=4,source_task_render_workers=2,numeric_threads_per_worker=1,
        GPU_workers=0,resource_files=['preparation.json','rebuild_resources.json','rebuild_summary.json',
            'asset_inventory/source_gallery_manifest_v2.json','dataset/source_tasks_pending_adapter/execution_receipt.json'],
        geometry_construction_seconds=b.read(OUT/'rebuild_summary.json').get('elapsed_seconds'),
        geometry_independent_verification_seconds=b.read(OUT/'rebuild_verification/summary.json')['elapsed_seconds'],
        timing_note='CPU substage timing only, not total collaboration/visual review or isolated training benchmark.',
        source_task_engineering_receipt=str(d/'execution_receipt.json'),
        split_count_correction='split_audit/engineering_correction_01/NOTE.md'))
    protected=dict(b.read(OUT/'preparation.json')['protected_source_hashes'])
    for p,key in [(OUT/'layout_contract/COMPLETE.json','input_hashes'),
                  (OUT/'layout_contract/HANDOFF.json','source_hashes')]:
        for path,h in b.read(p)[key].items():
            if '/bench_deployment_tasks_v1/' not in path.replace('\\','/'):
                protected[path]=h
    protected.update(b.read(OUT/'split_audit/protected_input_hashes.json'))
    protected[str(current)]=b.sha(current)
    for path,h in protected.items():assert b.sha(path)==h,path
    b.save(OUT/'protected_source_hashes.json',protected)
    report=OUT/'REPORT.md'
    appendix='\n## 7. 实际数据交付计数\n\n'
    appendix+=f'30条候选全部实际图审并保存原始意见；最终源任务准入 **{len(accepted)}/30**，涉及 **{len({r["parent_id"] for r in accepted})}父**；其余{len(rows)-len(accepted)}条因语义或C拓扑HOLD。这里只是源分辨率任务准入，新增可直接训练父仍为 **0**。\n\n'
    appendix+='| 暂存评价用途 | 准入C数 |\n|---|---:|\n'
    for g,n in sorted(groups.items()):appendix+=f'| {g} | {n} |\n'
    appendix+='\n同父预留组要求该父也有已准入的训练C；只有预留C通过的父另外标PENDING，不混入有效同父迁移计数。\n\n'
    appendix+='[30条逐条件清单](dataset/source_tasks_pending_adapter/per_condition.csv) · [M/C/Y/C+Y图集](dataset/source_tasks_pending_adapter/gallery/index.html) · [数据状态](DATASET_STATUS.json)。每条含实际input_C_only.npz、paired_C_Y.npz、来源面ID、来源/分组/图片hash；不是仅交汇总表。\n'
    text=report.read_text('utf-8')
    if '## 7. 实际数据交付计数' not in text:report.write_text(text+appendix,'utf-8')
    # New scripts are isolated; include exact added-source diff without touching git.
    code=ROOT/'r2/data_tools/bench_deployment_tasks_v1'
    diff=''.join(''.join(difflib.unified_diff([],p.read_text('utf-8').splitlines(True),
        fromfile='/dev/null',tofile=str(p.relative_to(ROOT)).replace('\\','/'))) for p in sorted(code.glob('*.py')))
    (OUT/'code.diff').write_text(diff,'utf-8')
    all_files=sorted(p for p in OUT.rglob('*') if p.is_file())
    for p in all_files:
        if p.suffix.lower() in ('.md','.json','.py','.html'):
            assert '\ufffd' not in p.read_text('utf-8-sig'),p
    hashes={str(p):b.sha(p) for p in all_files}
    hashes.update({str(p):b.sha(p) for p in sorted(code.glob('*.py'))})
    b.save(OUT/'COMPLETE.json',dict(status=status['status'],training_ready=False,
        new_model_ready_parents=0,source_tasks_accepted=len(accepted),
        source_hashes_unchanged=True,protected_input_files=len(protected),protected_source_hashes=protected,
        delivery_hashes=hashes,delivery_files=len(all_files),bytes_before_receipt=sum(p.stat().st_size for p in all_files),
        neural_forwards=0,GPU_calls=0,training_updates=0,
        model_weights_modified=False,cleanup=False,commit=False,push=False,
        source_task_receipt=receipt,completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    marker='## 2026-10-22 Bench生产条件数据建设：源任务已产出，新增低面训练池未就绪'
    workspace=ROOT/'WORKSPACE.md'
    if marker not in workspace.read_text('utf-8-sig'):
        with workspace.open('a',encoding='utf-8') as f:
            f.write(f'\n\n{marker}\n\nSHIFT交叉保持关闭。纯CPU核对source-owner确实改变可见锚点/角色/轨迹；但旧DEV16C均可忠实表示，不把它认定为失败根因。当前26父91C仅4缓存+2闭合可核对，85缺生产预测。复用560来源供给记录，实际复核16干净高面来源，15结构可解释、1HOLD；保全部边界的共面内部重三角化16父8011面不减，独立核查内部顶点为0，不证明其他低面重建不可能。\n\n父/家族先划分后制作30条真实C/Y，{len(accepted)}条通过语义和C拓扑源任务准入，原面全覆盖逐位往返；全部仍超N128..256，新增可训练父为0。13父TRAIN_STAGING、2父适配未见暂存、1HOLD；同父预留没有已准入训练C时另列PENDING，官方预训练未见不认证。零模型/头/D0前向、训练、新噪声，不切默认/清理/commit/push。未把旧26父重新包装开训。\n\n[报告](experiments/r2_bench_deployment_task_data_v1/REPORT.md) · [数据状态](experiments/r2_bench_deployment_task_data_v1/DATASET_STATUS.json) · [源任务图集](experiments/r2_bench_deployment_task_data_v1/dataset/source_tasks_pending_adapter/gallery/index.html)。恢复入口 `python -B -m r2.data_tools.bench_deployment_tasks_v1.run --resume` 完成后仅核验hash。\n')
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
