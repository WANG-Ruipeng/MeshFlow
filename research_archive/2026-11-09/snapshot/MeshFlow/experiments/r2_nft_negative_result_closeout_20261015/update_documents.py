"""Append the completed NFT evidence to persistent memory without rewriting old results."""
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def read(rel):
    return (ROOT / rel).read_text(encoding='utf-8-sig')

def write(rel, content):
    (ROOT / rel).write_text(content, encoding='utf-8')

def main():
    assert (OUT / 'cleanup_receipt.json').is_file(), 'Cleanup first; do not claim planned deletion.'
    cleanup = json.loads((OUT / 'cleanup_receipt.json').read_text(encoding='utf-8-sig'))
    assert cleanup['status'] == 'COMPLETE'
    source_paths = ['experiments/r2_terminal_reward_nft_v1/REPORT.md',
                    'experiments/r2_nft_training_c_frozen_v1/REPORT.md']
    srcs = json.loads(read('r2/docs/research_history_sources.json'))
    assert len(srcs) == 83 and max(x['id'] for x in srcs) == 83, 'Already amended or unexpected source index.'
    for idx, rel in enumerate(source_paths, 84):
        p = ROOT / rel
        b = p.read_bytes()
        srcs.append(dict(id=idx, path='E:/MeshFlow/'+rel, workspace_relative_path=rel,
                         bytes=len(b), lines=len(b.decode('utf-8-sig').splitlines()),
                         sha256=hashlib.sha256(b).hexdigest()))
    write('r2/docs/research_history_sources.json', json.dumps(srcs, ensure_ascii=False, indent=2)+'\n')
    md = read('r2/docs/research_history_sources.md')
    md = md.replace('2026-10-14. Primary reports: 83.',
                    '2026-10-15. Primary reports: 85 (83 from the 2026-10-14 audit plus two NFT reports).', 1)
    md = md.replace('not 83 independent methods', 'not 85 independent methods', 1)
    for s in srcs[-2:]:
        rel = s['workspace_relative_path']
        md += f"| S{s['id']} | [{rel}](<../../{rel}>) | {s['lines']} | {s['bytes']} | `{s['sha256']}` |\n"
    write('r2/docs/research_history_sources.md', md)
    registry = json.loads(read('r2/docs/research_route_registry.json'))
    assert not any(r['id'] == 'R29' for r in registry['routes'])
    registry['version'] = 'research_route_memory_20261015_nft_closeout_v1'
    registry['counts'].update(not_stable_upgrade=29, primary_reports=85)
    registry['previous_audit'] = dict(date='2026-10-14', not_stable_upgrade=28, primary_reports=83)
    registry['routes'].append(dict(
        id='R29', name='终态奖励NFT LoRA与共享端点SELF_REFLOW', status='NOT_STABLE_UPGRADE',
        lifecycle='CLOSED_NEGATIVE_CURRENT_RECIPE',
        evidence='FULL_S6000起点，rank4 LoRA 589824参数，两臂各256更新。主面板36请求三模型明确联合均9；八条奖励TRAIN C各四新噪声的冻结检查联合FULL/NFT/SELF=11/12/11，RAW采用均26/32。唯一新增a7/n0也由SELF实现；NFT相对SELF的净优势来自保留a7/n2。共同已知28对/7C的原始R，NFT对FULL -0.003113，对SELF -0.002472；一条C奖励UNKNOWN保留。NFT FF略降，但I及局部尾部有代价，接口总长下降。',
        retry_boundary='不沿当前配置直接增加轮数、覆盖或挑checkpoint。若未来重开，需实质改变适配/优化或监督转化机制并另行受控验证；本证据不否定所有RL/NFT，也未证明奖励函数错误。SELF是共享NFT端点的off-policy对照，不是普通GT续训。',
        source_ids=[84,85],
        closeout_report='experiments/r2_nft_negative_result_closeout_20261015/NEGATIVE_RESULT_REPORT.md',
        cleanup_receipt='experiments/r2_nft_negative_result_closeout_20261015/cleanup_receipt.json',
        retained='FULL_S6000 and official models; NFT/SELF round15; all RAW/POST, metrics, reviews, source and logs',
        retired='NFT/SELF round00..14 .pt (30 files); historical checkpoint-by-checkpoint replay unavailable'))
    write('r2/docs/research_route_registry.json', json.dumps(registry, ensure_ascii=False, indent=2)+'\n')
    mem = read('r2/docs/research_route_memory.md')
    mem = mem.replace('更新：2026-10-14；覆盖截至 FULL_CONTOUR_S6000 完成的本地历史。',
                      '更新：2026-10-15；原FULL_CONTOUR历史盘点补入NFT训练与冻结新噪声检查的收口。', 1)
    mem = mem.replace('本次盘点 **83份主实验／诊断报告**',
                      '2026-10-14盘点 **83份主实验／诊断报告**，本次补入两份NFT报告，来源索引共 **85份**', 1)
    mem = mem.replace('**28条已运行但未形成稳定升级的候选路线（R01—R28）**',
                      '**29条已运行但未形成稳定升级的候选路线（R01—R29）**', 1)
    mem = mem.replace('“30种方法已被证明永远无效”', '“31种方法已被证明永远无效”', 1)
    entry = '''## R29：NFT终态奖励路线已收口为当前配方的负结果（2026-10-15）

训练后终点评价面板36个配对请求，FULL/NFT/SELF明确严格联合均9/36，没有新增成功覆盖。补充冻结检查保留原8条奖励TRAIN C，每条4份新噪声，三模型共96输出：RAW采用均26/32，严格联合FULL/NFT/SELF为11/12/11。唯一新增a7b82d/n0也由无奖励SELF实现；NFT相对SELF的净差来自保留a7/n2，并非独有新能力。其余7条C联合数量没有变化。[S84](../../experiments/r2_terminal_reward_nft_v1/REPORT.md) [S85](../../experiments/r2_nft_training_c_frozen_v1/REPORT.md)

固定原始奖励在共同已知28对/7C上，NFT−FULL=-0.003113、NFT−SELF=-0.002472；第8条C保持UNKNOWN，全32奖励差可行区间跨零，不声称统计显著或全组真实奖励已经下降。NFT的FF均值下降，但I和局部尾部仍有代价，合法接口总长70.2352→66.6572。SELF共享NFT端点池，是off-policy无奖励对照；不是原GT普通续训。图审为实际AI_PROXY，非人类金标准。

**决定：停止扩大当前NFT配方，不追加轮次、不挑中间权重、不切默认。** 保留局部正结果，不上升为“RL无效”或“奖励函数已被证伪”。FULL_S6000继续作为方法比较参照。以后若重开，应提出实质的适配/优化或监督转化变化及可检验预测；不能只改名或加步数重跑。

按用户授权清除NFT/SELF两臂round00..14的30个中间权重，保留round15两份、FULL/官方模型及全部RAW/POST、评分、图审、日志、代码和旧报告。中间checkpoint逐轮重放已退役；旧报告与审计描述历史完成状态，不代表全部中间权重仍在磁盘。[负结果报告](../../experiments/r2_nft_negative_result_closeout_20261015/NEGATIVE_RESULT_REPORT.md) · [清理账本](../../experiments/r2_nft_negative_result_closeout_20261015/cleanup_receipt.json)。本次仅文档与清理，零模型调用。

'''
    mem = mem.replace('## 先读这几条\n', entry+'## 先读这几条\n', 1)
    write('r2/docs/research_route_memory.md', mem)
    prefix = '''## 2026-10-15 NFT负结果收口与中间权重清理

当前NFT配置停止扩大。训练后终点评价面板36个配对请求三模型明确联合均9/36；八条奖励TRAIN C×四份新噪声的冻结检查，FULL/NFT/SELF严格联合11/12/11，RAW采用均26/32。唯一新增请求也由无奖励SELF实现，NFT相对SELF的净差是保住另一旧正例。共同已知28对/7C的NFT原始奖励相对FULL -0.003113；第8条C奖励UNKNOWN，不填零。FF均值局部改善与I/尾部代价同时记录，不否定全部RL，也不认为改奖励已被证明是首要答案。

用户授权删除30个NFT/SELF round00..14中间.pt，逻辑空间287,949,354字节（274.61 MiB）；保留两份round15、FULL_S6000、官方模型及全部RAW/POST/图审/评分/源码/日志/旧报告。历史逐轮权重重放已退役；不自动补回、不续训。实际磁盘变化、完整路径与SHA见[清理回执](experiments/r2_nft_negative_result_closeout_20261015/cleanup_receipt.json)。[负结果报告](experiments/r2_nft_negative_result_closeout_20261015/NEGATIVE_RESULT_REPORT.md)；研究记忆新增R29，来源索引83+2=85份。FULL仍为方法比较参照，维护CLI默认不变；零新模型调用，无commit/push。

'''
    write('WORKSPACE.md', prefix+read('WORKSPACE.md'))
    agents = read('AGENTS.md')
    agents = agents.replace('The 2026-10-14 review groups 28 tested candidates without a stable upgrade and two pre-training gate blocks;',
                            'The 2026-10-14 review plus the 2026-10-15 NFT closeout register 29 tested candidates without a stable upgrade and two pre-training gate blocks;', 1)
    agents += '\n## 2026-10-15 NFT negative-result closure\n\nR29 closes the current terminal-reward NFT/SELF_REFLOW recipe. The 36-request panel had no new joint positives; the eight training-C fresh-noise check had FULL/NFT/SELF=11/12/11 strict positives, with the only new request also gained by SELF. This does not establish a stable reward-specific upgrade or invalidate all RL. Do not automatically extend this recipe. NFT/SELF round00..14 weights were removed at user request; round15, FULL_S6000, official models and all evidence remain. Historical intermediate replay is retired. See the route memory and experiments/r2_nft_negative_result_closeout_20261015/NEGATIVE_RESULT_REPORT.md.\n'
    write('AGENTS.md', agents)
    retirement = '''# NFT current-recipe closure

2026-10-15: closed as a negative result for a stable reward-specific control upgrade.
This is a maintenance addendum, not a rewrite of the completed original experiment.

Only NFT/SELF_FM round00..14 checkpoint files were removed at the user's request.
The two round15 endpoints, external FULL_S6000 base, official D0, all saved RAW/POST,
metrics, actual image reviews, training logs, receipts, source and original reports remain.

The old execution audit describes historical completion. Re-running a full audit that
requires every intermediate checkpoint will now report those intentional removals.
Do not run the old training controller to reconstruct missing rounds. Final inference
still needs the retained FULL base and the appropriate final LoRA; the adapter alone
is not a standalone generator. GPU replay is not authorized by this note.

Closeout: ../r2_nft_negative_result_closeout_20261015/NEGATIVE_RESULT_REPORT.md
Deletion ledger: ../r2_nft_negative_result_closeout_20261015/cleanup_receipt.json
'''
    write('experiments/r2_terminal_reward_nft_v1/RETIREMENT.md', retirement)
    print('Updated six persistent documents, two source-index entries, R29 and one retirement addendum; no old report modified.')

if __name__ == '__main__':
    main()
