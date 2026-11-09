"""Report digest of existing attribution CSVs; no scoring or model access."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

DEFAULT_OUT = Path(__file__).resolve().parents[3] / 'native_t1/runs/chair_control_feedback_auto_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_table(info):
    path = Path(info['path'])
    assert sha(path) == info['sha256'], f'Changed source: {path}'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def run(out=DEFAULT_OUT, label='explore_TRAJECTORY_REPLAY_1000'):
    assert label == 'explore_TRAJECTORY_REPLAY_1000', 'This fixed interpretation digest is scoped to TR1000'
    out = Path(out)
    dev_index_path = out / 'derived/attribution/DEV/latest.json'
    transfer_index_path = out / 'derived/transfer_attribution/latest.json'
    dev_index, transfer_index = read_json(dev_index_path), read_json(transfer_index_path)
    dev = {key: read_table(dev_index['tables'][key]) for key in ('summary', 'per_parent', 'per_output', 'noise_discordance', 'leave_one_out')}
    transfer = {key: read_table(transfer_index['tables'][key]) for key in ('pool_metric_summary', 'per_parent', 'per_output')}
    roles = ('START', 'MATCHED_CONTROL')
    metrics = ('SeamCov', 'ConflictCov_C', 'ExposedRetain')
    select = lambda row: row['candidate_label'] == label and row['reference_role'] in roles and row['metric'] in metrics
    summaries = [row for row in dev['summary'] if select(row)]
    assert len(summaries) == 6, 'Both complete references and all three controls required'
    parents = [row for row in dev['per_parent'] if select(row)]
    outputs = [row for row in dev['per_output'] if select(row)]
    noises = [row for row in dev['noise_discordance'] if select(row)]
    lopo = [row for row in dev['leave_one_out'] if select(row)]
    assert (len(parents), len(outputs), len(noises), len(lopo)) == (48, 96, 48, 48)
    evidence = []
    for summary in summaries:
        group = [row for row in parents if row['metric'] == summary['metric'] and row['reference_role'] == summary['reference_role']]
        gains = [float(row['favorable_gain']) for row in group]
        assert len(group) == 8 and abs(sum(gains) / 8 - float(summary['favorable_gain'])) < 1e-14
        assert abs(statistics.median(gains) - float(summary['favorable_parent_median'])) < 1e-14
        worst = min(group, key=lambda row: float(row['favorable_gain']))
        value = dict(summary)
        value['most_adverse_share_of_net_gain_percent'] = float(worst['share_of_net_gain_percent'])
        value['mean_without_most_adverse_parent'] = sum(float(row['favorable_gain']) for row in group if row['uid'] != worst['uid']) / 7
        value['all_leave_one_out_negative'] = float(summary['leave_one_out_gain_max']) < 0
        value['all_leave_one_out_positive'] = float(summary['leave_one_out_gain_min']) > 0
        evidence.append(value)
    train_s = [row for row in transfer['per_parent'] if row['pool'] == 'TRAIN_FEEDBACK' and row['candidate_label'] == label and row['reference_role'] == 'MATCHED_CONTROL' and row['metric'] == 'SeamCov']
    assert len(train_s) == 8
    train_gains = [float(row['favorable_gain']) for row in train_s]
    top = sorted(train_s, key=lambda row: float(row['favorable_gain']), reverse=True)[:2]
    train = dict(mean_gain=sum(train_gains)/8, median_gain=statistics.median(train_gains),
                 improved_parents=sum(gain > 0 for gain in train_gains), adverse_parents=sum(gain < 0 for gain in train_gains),
                 unchanged_score_parents=sum(gain == 0 for gain in train_gains),
                 two_largest_positive_parents=[dict(uid=row['uid'], gain=float(row['favorable_gain']), share_of_net_gain_percent=float(row['share_of_net_gain_percent'])) for row in top],
                 two_largest_share_of_net_gain_percent=sum(float(row['share_of_net_gain_percent']) for row in top),
                 note='Equal metric scores do not imply identical geometry. Training and DEV use different parents; this is descriptive transfer evidence, not paired parent causality.')
    identity = dict(script_sha256=sha(__file__), label=label, dev_attribution=dict(path=dev_index['path'], sha256=dev_index['sha256']),
                    transfer_attribution=dict(path=transfer_index['path'], sha256=transfer_index['sha256']),
                    source_tables=dict(DEV={key:dev_index['tables'][key] for key in dev}, TRAIN_FEEDBACK={key:transfer_index['tables'][key] for key in transfer}))
    for source in (identity['dev_attribution'], identity['transfer_attribution']):
        assert sha(source['path']) == source['sha256']
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    target = out / 'derived/attribution/DEV/digests' / (label + '__' + token)
    target.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, rows in dict(DEV_per_parent=parents, DEV_per_output=outputs, DEV_noise_discordance=noises, DEV_leave_one_out=lopo, TRAIN_FEEDBACK_matched_S_per_parent=train_s).items():
        path = target / (name + '.csv'); write_csv(path, rows)
        tables[name] = dict(path=str(path), rows=len(rows), sha256=sha(path))
    result = dict(status='DESCRIPTIVE_ONLY', identity=identity, DEV=evidence, TRAIN_FEEDBACK_matched_S=train, tables=tables,
                  limitations=['Original registered gates are unchanged.', 'Contributions use point estimates; original conservative bounds remain in source CSVs.', 'Net contribution percentages can exceed 100% when other parents offset gains.', 'No model, geometry scoring, CONFIRM access, threshold or candidate selection was used.'])
    write_json(target / 'distribution_digest.json', result)
    lookup = {(row['reference_role'],row['metric']):row for row in evidence}
    lines = [f'# {label}：8 父、2 噪声的分布证据', '', '以下均为父内两噪声平均后、8 父等权的点值描述。原保守区间和门槛均保留；本摘要不替代晋级门槛。', '']
    for role, text in (('START','START'), ('MATCHED_CONTROL','同更新数 teacher')):
        e, s, i = (lookup[(role, metric)] for metric in ('ExposedRetain','SeamCov','ConflictCov_C'))
        lines += [f'相对 {text}：E 均值增益 {float(e["favorable_gain"]):+.8f}，其中 {e["most_favorable_parent"][:8]} 占净均值增益 {float(e["most_favorable_share_of_net_gain_percent"]):.4f}%；去除该父后剩余 7 父均值为 {float(e["gain_without_most_favorable_parent"]):+.8f}。父中位增益 {float(e["favorable_parent_median"]):+.8f}，{e["improved_parents"]}/8 父改善，{e["opposite_noise_parent_count"]}/8 父的两噪声方向相反。', '',
            f'S 均值差 {float(s["favorable_gain"]):+.8f}，{s["adverse_parents"]}/8 父下降、{s["improved_parents"]}/8 上升、{s["unchanged_parents"]}/8 同分。最差父 {s["most_adverse_parent"][:8]} 占净下降 {float(s["most_adverse_share_of_net_gain_percent"]):.4f}%，但去除任一父后的均值仍为负：范围 [{float(s["leave_one_out_gain_min"]):+.8f}, {float(s["leave_one_out_gain_max"]):+.8f}]。两噪声反向 {s["opposite_noise_parent_count"]}/8 父，因此下降并非仅由单父造成。', '',
            f'I 减少均值 {float(i["favorable_gain"]):+.8f}，{i["improved_parents"]}/8 父改善；两噪声反向 {i["opposite_noise_parent_count"]}/8 父，noise 0/1 平均有利增益分别 {float(i["noise0_mean_favorable_gain"]):+.8f} / {float(i["noise1_mean_favorable_gain"]):+.8f}。', '']
    lines += [f'训练内相对 teacher 的 S 均值增益 {train["mean_gain"]:+.8f}，中位数 {train["median_gain"]:+.8f}。分布为 {train["improved_parents"]} 升、{train["adverse_parents"]} 降、{train["unchanged_score_parents"]} 同分；最大的两个正增益父占净均值增益 {train["two_largest_share_of_net_gain_percent"]:.4f}%。两项同分使排序后的中间两项均为 0。不能将正均值写成多数父都有改善。', '',
              '训练内与 DEV 是不同的 8 父集合。两池差异仅为已见条件向开发集迁移的描述，不能据此单独认定具体机制因果。500 与 1000 步分别计算，不复用贡献比例。', '',
              '完整逐父、逐输出、噪声方向与留一法 CSV 见同目录；distribution_digest.json 保存全部来源路径和 SHA-256。']
    (target/'DISTRIBUTION_DIGEST.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    latest = dict(status='COMPLETE', path=str(target/'distribution_digest.json'), sha256=sha(target/'distribution_digest.json'), memo_path=str(target/'DISTRIBUTION_DIGEST.md'), memo_sha256=sha(target/'DISTRIBUTION_DIGEST.md'), tables=tables)
    write_json(out/'derived/attribution/DEV/distribution_digest_latest.json', latest)
    return latest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, default=DEFAULT_OUT); parser.add_argument('--label', default='explore_TRAJECTORY_REPLAY_1000')
    args = parser.parse_args(); print(json.dumps(run(args.out, args.label), indent=2))
