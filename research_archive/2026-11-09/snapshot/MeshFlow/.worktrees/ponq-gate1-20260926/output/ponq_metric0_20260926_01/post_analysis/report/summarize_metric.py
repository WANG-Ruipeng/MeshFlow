"""Frozen-control/fixed-metric comparison, post-analysis only."""
import argparse,collections,csv,json,hashlib,math,shutil
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf-8')
def stat(values):
    a=[float(x) for x in values if x is not None and math.isfinite(x)]
    return dict(n=len(a),mean=float(np.mean(a)),median=float(np.median(a)),min=min(a),max=max(a)) if a else dict(n=0)
def mid(s):return s.get('median',float('nan'))
def aggregate(rows):
    metrics=['rms_pct_bboxdiag','pred_to_gt_rms_pct','gt_to_pred_rms_pct','p95_pct_bboxdiag','faces']
    sem=['P_rms_pct_bbox','P_symmetric_nn_rms_pct_bbox','n_oriented_angle_mean_deg','n_magnitude_rms','n_reverse_rate',
        'n_nearzero_count','F_relative','C_relative','C_spectral_relative','P_target_coverage_1p0pct_fraction','P_near_duplicate_pair_count']
    return dict(count=len(rows),valid=sum(r['valid'] for r in rows),quality_pass=sum(r['quality_pass'] for r in rows),
        extracted=sum(r['extraction_completed'] for r in rows),
        all_metrics={k:stat([r.get(k) for r in rows]) for k in metrics},
        valid_RMS=stat([r.get('rms_pct_bboxdiag') for r in rows if r['valid']]),
        semantics={k:stat([(r.get('endpoint_semantic') or {}).get(k) for r in rows]) for k in sem},
        noncycle_samples=sum(r.get('strict_noncycle_vertex_count',0)>0 for r in rows),
        nonmanifold_edge_samples=sum(r.get('nonmanifold_edges',0)>0 for r in rows),
        disconnected_samples=sum(r.get('beta0',0)>1 for r in rows),
        selfintersection_samples=sum(r.get('exact_intersection_pairs',0)>0 for r in rows),
        degenerate_samples=sum(r.get('exact_degenerate_faces',0)>0 for r in rows),
        outside_domain_samples=sum(r.get('input_sites_outside_actual_domain',0)>0 for r in rows),
        betti=dict(collections.Counter(str([r.get('beta'+str(i)) for i in range(3)]) for r in rows)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);root=ap.parse_args().root
    assert read(root/'EXECUTION.json')['status']=='COMPLETE'
    out=root/'post_analysis/report';out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
    rows={arm:read(root/'evaluation'/arm/'FREE_LEDGER.json') for arm in ['B0','M0']}
    result=dict(arms={a:aggregate(r) for a,r in rows.items()},weights=read(root/'calibration/WEIGHTS.json'),decision=read(root/'DECISION.json'),
        training=read(root/'training/M0/RESULT.json'),control_historical_training=read(root/'control/RESULT.json'),
        preflight=read(root/'preflight/RESULT.json'),gradient_monitor={},costs={})
    result['gradient_monitor']['parent3000']=read(root/'calibration/weighted_verification/RESULT.json')['summary']
    for label in ['M0_5000','M0_7000','M0_9000','B0_9000']:
        result['gradient_monitor'][label]=read(root/'probes'/label/'RESULT.json')['summary']
    for arm in ['B0','M0']:
        roll=read(root/'evaluation'/arm/'ROLLOUT.json');ev=read(root/'evaluation'/arm/'SUMMARY.json')
        result['costs'][arm]=dict(integration_seconds=roll['integration_seconds'],posthoc_seconds=roll['posthoc_seconds'],
            extraction_evaluation_seconds=ev['extraction_evaluation_seconds'],count=32)
    x=sorted(rows['B0'],key=lambda r:r['sample_id']);y=sorted(rows['M0'],key=lambda r:r['sample_id'])
    delta=[b['rms_pct_bboxdiag']-a['rms_pct_bboxdiag'] for a,b in zip(x,y) if 'rms_pct_bboxdiag' in a and 'rms_pct_bboxdiag' in b]
    result['paired']=dict(RMS_M0_minus_B0=stat(delta),RMS_improved=sum(d<0 for d in delta),RMS_worsened=sum(d>0 for d in delta),RMS_tied=sum(d==0 for d in delta),
        both_valid=sum(a['valid'] and b['valid'] for a,b in zip(x,y)),
        B0_only_valid=sum(a['valid'] and not b['valid'] for a,b in zip(x,y)),
        M0_only_valid=sum(not a['valid'] and b['valid'] for a,b in zip(x,y)),
        neither_valid=sum(not a['valid'] and not b['valid'] for a,b in zip(x,y)),
        both_quality=sum(a['quality_pass'] and b['quality_pass'] for a,b in zip(x,y)),
        B0_only_quality=sum(a['quality_pass'] and not b['quality_pass'] for a,b in zip(x,y)),
        M0_only_quality=sum(not a['quality_pass'] and b['quality_pass'] for a,b in zip(x,y)))
    trace=[json.loads(s) for s in (root/'training/M0/TRAIN_TRACE.jsonl').read_text().splitlines() if s.strip()]
    result['candidate_last100']={k:stat([r[k] for r in trace[-100:]]) for k in ['new_weighted_loss','old22_loss','x_mse','n_mse','F_mse','gradient_norm_before_clip']}
    result['original_manifest_sha256']=sha(root/'MANIFEST.json');result['protocol_sha256']=sha(root/'RUN_PROTOCOL.json')
    dump(out/'RESULT_NUMBERS.json',result)
    cols=['arm','sample_id','valid','quality_pass','status','faces','beta0','beta1','beta2','nonmanifold_edges','strict_noncycle_vertex_count',
        'exact_intersection_pairs','rms_pct_bboxdiag','pred_to_gt_rms_pct','gt_to_pred_rms_pct','p95_pct_bboxdiag','input_sites_outside_actual_domain','wall_seconds']
    with (out/'FREE_LEDGER_64.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(x);w.writerows(y)
    lines=['# SINGLE_STATE_METRIC0 数值账本','',
        '原DeepSets、共同3000步起点、同6000次后续更新。B0使用已归档9000步权重，M0只改变一次标定后固定的字段loss系数；采样不加权。两模型使用本轮新32噪声，64Euler及原±1域。',
        '',
        '|方法|几何有效/32|有效且RMS≤0.20%/32|全样本RMS中位(%bbox)|预测→目标RMS中位|目标→预测RMS中位|位置RMS中位|法向角中位(度)|C相对误差中位|面数min/median/max|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for arm in ['B0','M0']:
        g=result['arms'][arm];m=g['all_metrics'];s=g['semantics'];f=m['faces']
        lines.append(f"|{arm}|{g['valid']}|{g['quality_pass']}|{mid(m['rms_pct_bboxdiag']):.6g}|{mid(m['pred_to_gt_rms_pct']):.6g}|{mid(m['gt_to_pred_rms_pct']):.6g}|{mid(s['P_rms_pct_bbox']):.6g}|{mid(s['n_oriented_angle_mean_deg']):.6g}|{mid(s['C_relative']):.6g}|{f.get('min')}/{f.get('median')}/{f.get('max')}|")
    w=result['weights'];p=result['paired']
    lines+=['','RMS百分数均以固定目标网格bbox对角线归一化，8192面积样本/方向。表中包括全部可评估输出，不删无效网格；有效条件统计及缺失数在JSON中。',
        '',
        '固定系数（乘字段内部平均MSE）：'+json.dumps(w['weights'],ensure_ascii=False),
        '起点未加权共享参数梯度RMS：'+json.dumps(w['unweighted_gradient_rms'],ensure_ascii=False),
        f"共同尺度c={w['common_scale']:.9g}；旧/新总梯度RMS={w['old_total_gradient_rms']:.9g}/{w['new_total_gradient_rms']:.9g}。",
        '',
        f"逐噪声配对：M0表面RMS改善{p['RMS_improved']}份，恶化{p['RMS_worsened']}份；差值中位{mid(p['RMS_M0_minus_B0']):.6g}%bbox。",
        f"工程决定：{result['decision']['status']}；联合门槛仍为至少24/32，不改变0.20%阈值。",
        '',
        '|权重检查点|旧总shared梯度RMS|新总shared梯度RMS|新位置/法向/F字段梯度RMS|',
        '|---|---:|---:|---|']
    for label,g in result['gradient_monitor'].items():
        fields=g['new_field_gradient_rms']
        lines.append(f"|{label}|{g['old_total_gradient_rms']:.6g}|{g['new_total_gradient_rms']:.6g}|{fields['x']:.6g}/{fields['n']:.6g}/{fields['F']:.6g}|")
    lines+=['','每行是在该模型、同24校准探针上，分别用旧/新loss度量观察参数梯度。共同尺度只匹配起点，不承诺后续保持平衡，也不匹配AdamW实际更新向量。没有在线调整。',
        '',f"M0实际新增训练耗时{result['training']['training_elapsed_seconds']:.3f}秒。历史B0训练耗时只作背景，不作同场速度比较。"]
    for arm,c in result['costs'].items():lines.append(f"- {arm}本轮32份积分{c['integration_seconds']:.3f}秒，后验诊断{c['posthoc_seconds']:.3f}秒，提取/审核/表面评估{c['extraction_evaluation_seconds']:.3f}秒。")
    lines+=['','完整裁定与边界见 VERDICT_zh.md；本文件不改变原清单或原有效性定义。']
    (out/'NUMERIC_REPORT_zh.md').write_text(chr(10).join(lines)+chr(10),encoding='utf-8')
    dump(out/'MANIFEST.json',dict(scope='numeric report supplement only',files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps({'arms':result['arms'],'paired':result['paired'],'decision':result['decision']['status']},indent=2))
if __name__=='__main__':main()