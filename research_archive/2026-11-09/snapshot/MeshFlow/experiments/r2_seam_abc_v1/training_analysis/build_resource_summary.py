"""Read existing receipts only; no torch imports, subprocesses or fresh probes."""
from pathlib import Path
import hashlib
import json
import statistics

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
SEEN = {}

def read(path):
    path = Path(path)
    data = path.read_bytes()
    SEEN[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)

def jsonl(path):
    data = path.read_bytes()
    SEEN[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return [json.loads(x) for x in data.decode().splitlines() if x.strip()]

def stats(values):
    values = list(values)
    if not values:
        return {'count': 0, 'min': None, 'max': None, 'mean': None, 'sum': None}
    return dict(count=len(values), min=min(values), max=max(values),
                mean=statistics.mean(values), sum=sum(values))

def gpu(row):
    line = row.get('gpu', {}).get('stdout', '').strip().splitlines()
    if not line:
        return None
    try:
        name, total, used, free, util = [x.strip() for x in line[0].split(',')]
        return dict(name=name, total_mib=float(total), used_mib=float(used),
                    free_mib=float(free), utilization_percent=float(util))
    except (ValueError, TypeError):
        return None

def gpu_range(rows):
    rows = [gpu(x) for x in rows]
    rows = [x for x in rows if x is not None]
    return dict(snapshot_count=len(rows),
                used_mib=stats(x['used_mib'] for x in rows),
                free_mib=stats(x['free_mib'] for x in rows),
                utilization_percent=stats(x['utilization_percent'] for x in rows),
                scope='Observed total device values including other workers/background; sampled maximum, not per-process peak')

def snapshot_interval(folder):
    before = read(folder / 'resources_before.json')
    after = read(folder / 'resources_finished.json')
    return dict(before_unix=before['unix'], after_unix=after['unix'],
                observed_envelope_seconds=after['unix']-before['unix'],
                scope='Difference of saved before/finished snapshot timestamps; includes loading but does not replace the phase timer')

def main():
    result = dict(schema='seam_abc_saved_resource_summary_v1',
                  source_policy='Existing immutable phase receipts and logs only',
                  new_model_calls=0, new_probes=0,
                  process_VRAM_peak_mib=None,
                  process_VRAM_status='NA: WSL/WDDM nvidia-smi did not expose attributable process memory; allocator reserved memory is not a substitute',
                  concurrent_timing_warning='These are measured shared-device/concurrent wall times, not independent hardware or method-speed benchmarks. Worker walls and summed per-request times must not be added as elapsed experiment time.',
                  phases={})
    initial = read(BASE/'resources/initial.json')
    result['initial_hardware'] = dict(gpu=gpu(initial), cpu_logical_cores=initial['CPU_cores'],
                                      ram_available_bytes=initial['RAM_available_bytes'],
                                      recorded_utc=initial['utc'])
    observed_gpu = []
    for arm in ['A_COUPLE', 'B_POOL47', 'C_SHARED_TIP']:
        folder = BASE/'training'/arm
        status = read(folder/'status.json')
        logs = jsonl(folder/'training_log.jsonl')
        assert status['status'] == 'COMPLETE' and len(logs) == 800
        attempt = folder/'attempts'/status['attempt']
        resource = read(attempt/'resources.json')
        observed_gpu += resource['samples']
        timing = {key:stats(row[key] for row in logs if key in row) for key in
                  ['input_seconds','update_seconds','forward_seconds','backward_seconds',
                   'optimizer_seconds','head_forward_seconds','head_backward_seconds',
                   'head_optimizer_seconds','pointer_forward_seconds','pointer_backward_seconds','pointer_optimizer_seconds']}
        result['phases'][arm] = {'training': dict(
            status='COMPLETE', updates=800, wall_seconds=status['wall_seconds'],
            wall_minutes=status['wall_seconds']/60,
            wall_scope='Trainer timer begins before checkpoint verification/model loading and ends after final checkpoint/export/status work; includes periodic saves and CPU log/input overhead',
            training_threads_per_worker=2,
            observed_update_timer_seconds=timing,
            allocator_peak_allocated_bytes=stats(row['cuda_peak_allocated_bytes'] for row in logs),
            allocator_peak_reserved_bytes=stats(row['cuda_peak_reserved_bytes'] for row in logs),
            allocator_peak_scope='Per-formal-update reset; maximum over all 800 saved updates includes forward/backward and all optimizer steps. Not total process VRAM.',
            sampled_device_resources=gpu_range(resource['samples']),
            process_memory_status=status['resources']['process_memory_status'],
            process_peak_mib=status['resources']['process_peak_mib'],
            source=f'experiments/r2_seam_abc_v1/training/{arm}/status.json')}
    result['phases']['LONG'] = {'training': dict(status='REUSED_FROZEN_REFERENCE', new_updates=0,
                                                 wall_seconds=0, scope='No LONG training in this experiment; its historical cost is not counted here')}
    for arm in ['LONG','A_COUPLE','B_POOL47','C_SHARED_TIP']:
        folder = BASE/'evaluation'/arm
        raw = read(folder/'raw_complete.json')
        post = read(folder/'post_complete.json')
        raw_receipts = [read(p) for p in sorted(folder.glob('*/raw/receipt.json'))]
        new_raw = [r for r in raw_receipts if r.get('new_rollouts',1) == 1]
        new_post = []
        for p in sorted(folder.glob('*/post/receipt.json')):
            rr = read(p)
            if 'cache_origin' not in rr:
                new_post.append(read(p.parent/'proposal_receipt.json'))
        assert len(new_raw)==raw['new_rollouts'] and len(new_post)==post['new_D0_forwards']
        raw_stage = list(folder.glob('generation_attempt_*'))
        post_stage = list(folder.glob('postprocess_attempt_*'))
        assert len(raw_stage)==len(post_stage)==1
        phase_snapshots = {}
        for name, stage in [('RAW',raw_stage[0]),('POST',post_stage[0])]:
            snapshots = [read(p) for p in sorted(stage.glob('resources_*.json'))]
            per_output = [read(p) for p in sorted(folder.glob('*/'+('raw/resources_before.json' if name=='RAW' else 'post/resources_before_D0.json')))
                          if p.parent.parent.name in {r['request_id'] for r in new_raw}]
            snapshots += per_output
            observed_gpu += snapshots
            phase_snapshots[name] = gpu_range(snapshots)
        result['phases'][arm]['RAW'] = dict(
            status=raw['status'], outputs=raw['outputs'], new_rollouts=len(new_raw), reused_rollouts=raw['outputs']-len(new_raw),
            new_generator_forwards=raw['new_generator_forwards'], wall_seconds=raw['wall_seconds'],
            wall_scope='Generation phase timer starts after resident generator/pointer loading; includes layouts, registered rollouts, output writing and identity checks. Excludes initial model load.',
            loading_to_finish_snapshot_interval=snapshot_interval(raw_stage[0]),
            rollout_seconds=stats(r['audit']['seconds'] for r in new_raw),
            allocator_peak_allocated_bytes=stats(r['audit']['peak_allocated_bytes'] for r in new_raw),
            allocator_peak_reserved_bytes=stats(r['audit']['peak_reserved_bytes'] for r in new_raw),
            allocator_peak_scope='Per Euler50 rollout reset, resident generator included; not process VRAM',
            sampled_device_resources=phase_snapshots['RAW'])
        result['phases'][arm]['POST'] = dict(
            status=post['status'], outputs=post['outputs'], new_D0_forwards=len(new_post),
            reused_POST=post['outputs']-len(new_post), new_GEO_calls=post['new_GEO_calls'],
            wall_seconds=post['wall_seconds'],
            wall_scope='POST timer includes official D0 loading, new one-forward proposals, C/seam restoration, GEO_C, orientation, writes and identity checks; not D0-only time',
            loading_to_finish_snapshot_interval=snapshot_interval(post_stage[0]),
            D0_proposal_receipt_seconds=stats(r['seconds'] for r in new_post),
            D0_proposal_timer_scope='Synchronized D0 call plus event/proposal serialization up to proposal receipt; not kernel-only time',
            allocator_peak_allocated_bytes=stats(r['peak_allocated_bytes'] for r in new_post),
            allocator_peak_reserved_bytes=stats(r['peak_reserved_bytes'] for r in new_post),
            allocator_peak_scope='Per D0 proposal reset, resident model included; not process VRAM',
            sampled_device_resources=phase_snapshots['POST'])
    scoring = read(BASE/'evaluation/score_completion.json')
    rendering = read(BASE/'evaluation/gallery/render_completion.json')
    pages = [read(p) for p in sorted((BASE/'evaluation/gallery').glob('*_n[0-3]_*.json')) if '.started.' not in p.name]
    assert len(pages)==56
    result['CPU_scoring'] = dict(status=scoring['status'], cells=scoring['cells'], workers=scoring['workers'],
        threads_per_worker=1, wall_seconds=scoring['seconds'],
        cell_seconds=stats(r['seconds'] for r in scoring['receipts']),
        worker_peak_RSS_KiB=stats(r['peak_RSS_KiB'] for r in scoring['receipts']),
        memory_scope='Cumulative process peak RSS recorded at each cell; range is not independently reset per-cell peak',
        model_calls=0, new_queries=0, concurrent_with='CPU rendering during its initial portion')
    result['CPU_rendering'] = dict(status='RENDER_COMPLETE', pages=rendering['pages'],
        workers_after_first=rendering['workers_after_first'], threads_per_worker=rendering['threads_per_worker'],
        wall_seconds=rendering['wall_seconds'], page_seconds=stats(r['seconds'] for r in pages),
        worker_peak_RSS_KiB=stats(r['peak_RSS_KiB'] for r in pages),
        memory_scope='Cumulative worker process RSS through recorded page; includes imports',
        new_model_calls=0, concurrent_with='CPU scoring')
    result['observed_total_device_range'] = gpu_range(observed_gpu)
    result['dispatch_interpretation'] = [
        'Training A and B overlap; C starts after A finishes and overlaps the tail of B. Maximum concurrent training arms is two.',
        'Some generator sampling overlaps a training arm; per-arm RAW timing is not isolated.',
        'All registered RAW outputs were saved before the POST stage. Four POST arm dispatches were sequential.',
        'CPU score and render workers each expand to two after their first real authorized cell/page; each uses one thread.',
        'Training workers each use two CPU threads. Saved total-device maxima include concurrent workers and unrelated user/background GPU use.',
    ]
    result['source_file_count'] = len(SEEN)
    result['source_hash_set_sha256'] = hashlib.sha256(json.dumps(SEEN,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    result['source_key_files'] = {k:v for k,v in SEEN.items() if k.endswith(('status.json','raw_complete.json','post_complete.json','score_completion.json','render_completion.json','initial.json'))}
    target = BASE/'training_analysis/final_resource_summary.json'
    assert not target.exists(), 'Do not silently overwrite an earlier summary'
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines = ['# 已保存计时与资源汇总', '',
             '只读取本轮原始账本和 receipt；未运行模型、评分或新资源探针。以下为实测墙钟，期间存在并行竞争，不能作为独立速度基准。', '',
             '| 臂 | 本轮训练（800 步，秒） | RAW 阶段（秒） | 新 RAW / 复用 | POST 阶段（秒） | 新 D0 / 复用 POST |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in ['LONG','A_COUPLE','B_POOL47','C_SHARED_TIP']:
        p=result['phases'][arm]
        lines.append(f"| {arm} | {p['training']['wall_seconds']:.3f} | {p['RAW']['wall_seconds']:.3f} | {p['RAW']['new_rollouts']} / {p['RAW']['reused_rollouts']} | {p['POST']['wall_seconds']:.3f} | {p['POST']['new_D0_forwards']} / {p['POST']['reused_POST']} |")
    d0s = '，'.join(f"{a} {result['phases'][a]['POST']['D0_proposal_receipt_seconds']['sum']:.3f} 秒" for a in ['LONG','A_COUPLE','B_POOL47','C_SHARED_TIP'])
    lines += ['', 'LONG 为冻结历史参照，本轮没有训练。训练计时含加载、800 次更新、周期保存与最终导出；RAW 计时在模型加载完成后开始；POST 计时含官方 D0 加载、单次去噪、GEO_C、定向和写盘。JSON 同时保留逐 rollout、逐 D0 proposal 计时及加载前后快照时间，不把三种口径混用。', '',
              f'新 D0 proposal 的逐次计时累计为：{d0s}。这个短计时包含同步前向、事件记录及 proposal 写盘至 receipt 前，不是纯 CUDA kernel 时间；与上表完整 POST 墙钟分开保存。', '',
              f"CPU 评分：224 单元，2 worker、每 worker 1 线程，墙钟 **{scoring['seconds']:.3f} 秒**。渲染：56 页，首张实际任务后扩为 2 worker、每 worker 1 线程，墙钟 **{rendering['wall_seconds']:.3f} 秒**。两项有重叠，不能相加作为总历时。", '',
              '| 训练臂 | 800 次更新中的最大 allocated（MiB） | 最大 reserved（MiB） | 采样到的整卡 used 峰值（MiB） |',
              '| --- | ---: | ---: | ---: |']
    for arm in ['A_COUPLE','B_POOL47','C_SHARED_TIP']:
        t=result['phases'][arm]['training']
        lines.append(f"| {arm} | {t['allocator_peak_allocated_bytes']['max']/2**20:.3f} | {t['allocator_peak_reserved_bytes']['max']/2**20:.3f} | {t['sampled_device_resources']['used_mib']['max']:.0f} |")
    raw_alloc=[result['phases'][a]['RAW']['allocator_peak_allocated_bytes'] for a in result['phases']]
    raw_reserved=[result['phases'][a]['RAW']['allocator_peak_reserved_bytes'] for a in result['phases']]
    d0_alloc=[result['phases'][a]['POST']['allocator_peak_allocated_bytes'] for a in result['phases']]
    d0_reserved=[result['phases'][a]['POST']['allocator_peak_reserved_bytes'] for a in result['phases']]
    def span(entries):return f"{min(x['min'] for x in entries)/2**20:.3f}–{max(x['max'] for x in entries)/2**20:.3f}"
    lines += ['',f"96 次新 RAW 的逐次 allocator 峰值范围：allocated {span(raw_alloc)} MiB，reserved {span(raw_reserved)} MiB；96 次新 D0 proposal：allocated {span(d0_alloc)} MiB，reserved {span(d0_reserved)} MiB。均包含相应驻留模型，但不包含完整进程所有显存。", '',
              f"CPU worker 累计 peak RSS 记录范围：评分 {result['CPU_scoring']['worker_peak_RSS_KiB']['min']:.0f}–{result['CPU_scoring']['worker_peak_RSS_KiB']['max']:.0f} KiB；渲染 {result['CPU_rendering']['worker_peak_RSS_KiB']['min']:.0f}–{result['CPU_rendering']['worker_peak_RSS_KiB']['max']:.0f} KiB。这是进程累计高水位，未逐单元重置。", '',
              '**WDDM/WSL 未提供可归属的进程显存：各阶段 process VRAM 保留 NA。** allocated/reserved 仅是 PyTorch 分配器统计，不能替代进程峰值。整卡 used 含其他训练/采样进程与后台占用，且为离散快照峰值，不能归因到单臂。', '',
              f"硬件记录为 NVIDIA GeForce RTX 5080，整卡 16,303 MiB、24 逻辑 CPU；首次快照可用 RAM {initial['RAM_available_bytes']/2**30:.3f} GiB。训练最多两臂并行，部分采样与训练重叠；四个 POST 派发依次执行。", '',
              f"来源：{len(SEEN)} 个既有 JSON/JSONL；汇总 JSON 保留关键文件 hash 与完整来源 hash 集合摘要。"]
    (BASE/'training_analysis/final_resource_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','source_files':len(SEEN),
                      'training_wall_seconds':{a:result['phases'][a]['training']['wall_seconds'] for a in result['phases']},
                      'CPU_score_wall_seconds':scoring['seconds'], 'CPU_render_wall_seconds':rendering['wall_seconds'],
                      'output':str(target)}))

if __name__ == '__main__':
    main()
