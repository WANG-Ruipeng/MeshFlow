"""Descriptive post-analysis only. Frozen primary/gate/diagnostic ledgers unchanged."""
import argparse,collections,csv,hashlib,json,math,shutil
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def stat(a):
    values=[float(x) for x in a if x is not None and math.isfinite(x)]
    return dict(n=len(values),min=min(values),median=float(np.median(values)),mean=float(np.mean(values)),max=max(values)) if values else dict(n=0)
def median(s):return s.get('median',float('nan'))
def aggregate(rows):
    fields=['P_rms_pct_bbox','n_oriented_angle_mean_deg','n_magnitude_rms','n_reverse_rate','n_nearzero_count',
        'F_relative','C_relative','C_spectral_relative','world_Q_relative','P_symmetric_nn_rms_pct_bbox',
        'P_target_coverage_0p1pct_fraction','P_target_coverage_0p5pct_fraction','P_target_coverage_1p0pct_fraction',
        'P_near_duplicate_pair_count','P_near_duplicate_point_fraction']
    return dict(count=len(rows),extracted=sum(r['extraction_completed'] for r in rows),valid=sum(r['valid'] for r in rows),
        quality_pass=sum(r.get('quality_pass',False) for r in rows),
        surface_RMS_all=stat([r.get('rms_pct_bboxdiag') for r in rows]),
        surface_RMS_valid=stat([r.get('rms_pct_bboxdiag') for r in rows if r['valid']]),
        faces=stat([r.get('faces') for r in rows]),
        semantic={k:stat([(r.get('endpoint_semantic') or {}).get(k) for r in rows]) for k in fields},
        noncycle_samples=sum(r.get('strict_noncycle_vertex_count',0)>0 for r in rows),
        nonmanifold_edge_samples=sum(r.get('nonmanifold_edges',0)>0 for r in rows),
        disconnected_samples=sum(r.get('beta0',0)>1 for r in rows),
        selfintersecting_samples=sum(r.get('exact_intersection_pairs',0)>0 for r in rows),
        degenerate_samples=sum(r.get('exact_degenerate_faces',0)>0 for r in rows),
        outside_domain_samples=sum(r.get('input_sites_outside_actual_domain',0)>0 for r in rows),
        outside_domain_points=sum(r.get('input_sites_outside_actual_domain',0) for r in rows),
        betti=dict(collections.Counter(str([r.get('beta'+str(k)) for k in range(3)]) for r in rows)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();root=a.root
    assert read(root/'EXECUTION.json')['status']=='COMPLETE'
    out=root/'post_analysis/report';out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
    arms=['B0','B1','B2'];allrows={a:read(root/'evaluation'/a/'FREE_LEDGER.json') for a in arms};result=dict(arms={},paired={})
    for arm in arms:
        rows=allrows[arm];field=read(root/'evaluation'/arm/'FIELD_LEDGER.json')
        train=read(root/'training'/arm/'RESULT.json');roll=read(root/'evaluation'/arm/'ROLLOUT.json');ev=read(root/'evaluation'/arm/'SUMMARY.json')
        trace=[json.loads(s) for s in (root/'training'/arm/'TRAIN_TRACE.jsonl').read_text().splitlines() if s.strip()]
        gradients=read(root/'probes'/arm/'GRADIENT_ROWS.json');probes=read(root/'probes'/arm/'PROBE_ROWS.json')
        g=dict(splits={s:aggregate([r for r in rows if r['split']==s]) for s in ['old','new']},
            field={c:aggregate([r for r in field if r['combination']==c]) for c in ['POO','OPO','OOP']},
            parameters=train['parameter_counts'],training_seconds=train['training_elapsed_seconds'],
            training_wall_seconds=train['total_wall_seconds_including_initialization_and_save'],
            sampling64_count=64,integration_seconds=roll['integration_seconds'],posthoc_seconds=roll['posthoc_seconds'],
            extraction_evaluation_seconds=ev['extraction_evaluation_seconds'],
            last100={k:stat([r[k] for r in trace[-100:]]) for k in ['loss','x_mse','n_mse','F_mse','gradient_norm_before_clip']},
            probes={},gradients={})
        g['advance']=g['splits']['new']['quality_pass']>=24
        for t in sorted(set(r['time'] for r in probes)):
            rr=[r for r in probes if r['time']==t];gg=[r for r in gradients if r['time']==t]
            velocity_keys=['velocity_mse','velocity_x_mse','velocity_n_mse','velocity_F_mse',
                'velocity_x_weighted_contribution','velocity_n_weighted_contribution','velocity_F_weighted_contribution']
            g['probes'][str(t)]=dict(n=len(rr),velocity={k:stat([r.get(k) for r in rr]) for k in velocity_keys},
                endpoint={k:stat([r['endpoint'].get(k) for r in rr]) for k in g['splits']['new']['semantic']},
                zero={k:stat([r['zero_velocity_endpoint'].get(k) for r in rr]) for k in g['splits']['new']['semantic']})
            keys=['gradient_x_unweighted_norm','gradient_n_unweighted_norm','gradient_F_unweighted_norm',
                'gradient_x_weighted_norm','gradient_n_weighted_norm','gradient_F_weighted_norm','gradient_total_norm',
                'gradient_cosine_x_n','gradient_cosine_x_F','gradient_cosine_n_F']
            g['gradients'][str(t)]={k:stat([r.get(k) for r in gg]) for k in keys}
        result['arms'][arm]=g
    for left,right in [('B0','B1'),('B0','B2'),('B1','B2')]:
        result['paired'][left+'_vs_'+right]={}
        for split in ['old','new']:
            x=sorted([r for r in allrows[left] if r['split']==split],key=lambda r:r['sample_id'])
            y=sorted([r for r in allrows[right] if r['split']==split],key=lambda r:r['sample_id'])
            result['paired'][left+'_vs_'+right][split]=dict(
                both_quality=sum(a['quality_pass'] and b['quality_pass'] for a,b in zip(x,y)),
                left_only_quality=sum(a['quality_pass'] and not b['quality_pass'] for a,b in zip(x,y)),
                right_only_quality=sum(not a['quality_pass'] and b['quality_pass'] for a,b in zip(x,y)),
                neither_quality=sum(not a['quality_pass'] and not b['quality_pass'] for a,b in zip(x,y)),
                right_minus_left_RMS=stat([b['rms_pct_bboxdiag']-a['rms_pct_bboxdiag'] for a,b in zip(x,y) if 'rms_pct_bboxdiag' in a and 'rms_pct_bboxdiag' in b]))
    result['advancing_arms']=[arm for arm in arms if result['arms'][arm]['advance']]
    result['scope']='single target,one common parent seed,6000 additional updates; old/new noise distinct; new noise is not new-object generalization'
    result['protocol_sha256']=sha(root/'RUN_PROTOCOL.json');result['original_manifest_sha256']=sha(root/'MANIFEST.json')
    dump(out/'RESULT_NUMBERS.json',result)
    cols=['arm','split','sample_id','split_id','combination','generated','oracle_assisted','valid','quality_pass','status','faces','beta0','beta1','beta2',
        'nonmanifold_edges','strict_noncycle_vertex_count','exact_intersection_pairs','rms_pct_bboxdiag','p95_pct_bboxdiag','input_sites_outside_actual_domain','wall_seconds']
    with (out/'FREE_LEDGER_192.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader()
        for arm in arms:w.writerows(allrows[arm])
    lines=['# SINGLE_STATE_INTERACTION0 数值报告','',
        '共同3000步checkpoint分叉，B0续训、B1逐点MLP、B2单层4头集合注意力，各追加6000步至9000；主结果固定为末端。',
        '固定目标、完整22通道、配对/loss/原始法向不变；64 Euler，固定±1域，不裁剪、不重抽、不换域。',
        '',
        '新32噪声的推进线为至少24份满足原完整几何有效合同且表面RMS≤0.20%目标bbox对角线。新噪声仍是同一目标的随机输入，不是新对象泛化。',
        '',
        '|臂|噪声|几何有效/32|有效且精度达标/32|全样本表面RMS中位(%bbox)|位置RMS中位(%bbox)|法向均角中位(度)|中心化C相对误差中位|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        for split in ['old','new']:
            g=result['arms'][arm]['splits'][split];s=g['semantic']
            lines.append(f"|{arm}|{split}|{g['valid']}|{g['quality_pass']}|{median(g['surface_RMS_all']):.5g}|{median(s['P_rms_pct_bbox']):.5g}|{median(s['n_oriented_angle_mean_deg']):.5g}|{median(s['C_relative']):.5g}|")
    lines+=['','表面RMS是到固定508面目标网格的双向面积采样距离，8192点/方向；所有可评估输出均计入（包括无效网格），有效条件统计及缺失数量在JSON中。实际面数不强制508。','',
        '|臂|总/新增参数|新增训练秒|64份积分秒|后验诊断秒|88份提取审核与误差秒|最后100步loss均值|',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm in arms:
        g=result['arms'][arm];p=g['parameters']
        lines.append(f"|{arm}|{p['total']}/{p['added']}|{g['training_seconds']:.3f}|{g['integration_seconds']:.3f}|{g['posthoc_seconds']:.3f}|{g['extraction_evaluation_seconds']:.3f}|{g['last100']['loss']['mean']:.6g}|")
    lines+=['','积分时间含原direct路径中的L诊断（不加到速度）；GPU同步计时。参数近似相同不代表同FLOPs或同墙钟。只有每臂更新数、曝光、训练流严格配对。','',
        '|臂|仅P预测有效/8|仅n预测有效/8|仅F预测有效/8|',
        '|---|---:|---:|---:|']
    for arm in arms:
        f=result['arms'][arm]['field'];lines.append(f"|{arm}|{f['POO']['valid']}|{f['OPO']['valid']}|{f['OOP']['valid']}|")
    lines+=['','上述24份/臂诊断仅使用旧ID0,4,8,12,16,20,24,28；其余字段来自真值，不进入主192份成绩。','',
        '工程推进线通过的实验臂：'+(', '.join(result['advancing_arms']) or '无')+'。','',
        '分时刻分块loss、共享隐藏层梯度、位置覆盖、近重复点、法向反向率与长度、有效输出条件RMS详见 RESULT_NUMBERS.json。梯度是在不同模型内部隐藏表示上的诊断，不等于各字段真实参数更新幅度，不能由通道数量或单个梯度范数断言因果主导。','',
        '完整解释与审核结果见 EXPERIMENT_REPORT_zh.md；此数值汇总不修改原实验清单。']
    (out/'NUMERIC_REPORT_zh.md').write_text(chr(10).join(lines)+chr(10),encoding='utf-8')
    dump(out/'MANIFEST.json',dict(scope='summary supplement only',files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps({a:{s:{k:result['arms'][a]['splits'][s][k] for k in ['valid','quality_pass']} for s in ['old','new']} for a in arms},indent=2))
if __name__=='__main__':main()