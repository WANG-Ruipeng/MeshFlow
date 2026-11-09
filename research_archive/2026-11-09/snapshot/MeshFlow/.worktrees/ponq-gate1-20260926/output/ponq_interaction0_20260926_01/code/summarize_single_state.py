"""Post-hoc descriptive summary; never changes frozen experiment artifacts."""
import argparse,collections,csv,hashlib,json,math,shutil
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(values):
    a=np.array([x for x in values if x is not None and math.isfinite(x)],float)
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),max=float(a.max()),mean=float(a.mean())) if len(a) else dict(n=0)
def group(rows):
    return dict(total=len(rows),valid=sum(bool(r['valid']) for r in rows),
        extracted=sum(bool(r.get('extraction_completed')) for r in rows),
        statuses=dict(collections.Counter(r['status'] for r in rows)),
        faces=stats([r.get('faces') for r in rows]),
        surface_RMS_pct_bbox_all=stats([r.get('rms_pct_bboxdiag') for r in rows]),
        surface_RMS_pct_bbox_valid=stats([r.get('rms_pct_bboxdiag') for r in rows if r['valid']]),
        betti=dict(collections.Counter(str([r.get('beta'+str(i)) for i in range(3)]) for r in rows)),
        nonmanifold_edges_samples=sum(r.get('nonmanifold_edges',0)>0 for r in rows),
        noncycle_vertex_samples=sum(r.get('strict_noncycle_vertex_count',0)>0 for r in rows),
        disconnected_samples=sum(r.get('beta0',0)>1 for r in rows),
        selfintersection_samples=sum(r.get('exact_intersection_pairs',0)>0 for r in rows),
        degenerate_samples=sum(r.get('exact_degenerate_faces',0)>0 for r in rows),
        actual_fixed_outside_samples=sum(r.get('input_sites_outside_actual_fixed_cube',0)>0 for r in rows))
def semantic(rows,key):
    fields=['P_rms_pct_bbox','n_oriented_angle_mean_deg','n_magnitude_rms','F_relative','C_relative','C_spectral_relative','world_Q_relative','world_Q_spectral_relative']
    return {f:stats([r.get(key,{}).get(f) for r in rows]) for f in fields}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();root=a.root
    execution=read(root/'EXECUTION.json')
    assert execution['status']=='COMPLETE',execution['status']
    dest=root/'post_analysis'/'report';dest.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,dest/'summarize_single_state.py')
    rows=read(root/'interventions_64/LEDGER.json');probe=read(root/'probes/PROBE_ROWS.json')
    combos=['PPP','PPO','POP','POO','OPP','OPO','OOP','OOO'];domains=['fixed','enclosing']
    res=dict(run=root.name,execution=execution,training=read(root/'training/direct/RESULT.json'),
        step_decision=read(root/'STEP_TEST_DECISION.json'),
        validity_definition='closed oriented 2-manifold + every vertex link simple cycle + Betti[1,0,1] + no exact intersections/degeneracy/duplicates + finite nonnegative capacities; no shape RMS or face-count threshold',
        comparison_reference='single canonical thin-U frame6 q_only 508-face mesh; not original analytic/PL surface',
        field_definition='P means actual extractor v; n is raw normal; F is complete centered factor; decode world Q at selected P. Pred/oracle fields all use one post-hoc P-Hungarian permutation.',
        metric_note='RMS is two-sided sampled surface distance, 8192 samples/direction, normalized by target bbox diagonal; percentages. No sample selected or discarded.',
        combinations={d:{c:group([r for r in rows if r['domain']==d and r['combination']==c]) for c in combos} for d in domains},
        probes_by_time={},free_semantics={})
    for t in sorted(set(r['time'] for r in probe)):
        rr=[r for r in probe if r['time']==t]
        res['probes_by_time'][str(t)]=dict(n=len(rr),endpoint=semantic(rr,'endpoint'),zero_velocity=semantic(rr,'zero_velocity_endpoint'),
            velocity_mse=stats([r.get('velocity_mse') for r in rr]),
            velocity_x_mse=stats([r.get('velocity_x_mse') for r in rr]),
            velocity_n_mse=stats([r.get('velocity_n_mse') for r in rr]),
            velocity_F_mse=stats([r.get('velocity_F_mse') for r in rr]))
    res['probe09_geometry']={d:group([r for r in read(root/'probes/GEOMETRY_ROWS.json') if r['domain']==d]) for d in domains}
    for nfe in [64,256]:
        p=root/f'sampling_{nfe}/ROLLOUT.json'
        if p.exists():
            roll=read(p)
            res['free_semantics'][str(nfe)]=dict(postmatched=semantic(roll['rows'],'endpoint_semantic'),
                initial_noise_coupling=semantic(roll['rows'],'initial_coupling_endpoint_semantic'))
    optional=root/'interventions_256/LEDGER.json'
    if optional.exists():
        rr=read(optional);res['free256']={d:group([r for r in rr if r['domain']==d]) for d in domains}
        res['paired64_256']={}
        for d in domains:
            aa=sorted([r for r in rows if r['domain']==d and r['combination']=='PPP'],key=lambda r:r['sample_id'])
            bb=sorted([r for r in rr if r['domain']==d],key=lambda r:r['sample_id'])
            res['paired64_256'][d]=dict(both_valid=sum(x['valid'] and y['valid'] for x,y in zip(aa,bb)),
                rescued=sum(not x['valid'] and y['valid'] for x,y in zip(aa,bb)),
                lost=sum(x['valid'] and not y['valid'] for x,y in zip(aa,bb)),
                neither_valid=sum(not x['valid'] and not y['valid'] for x,y in zip(aa,bb)),
                RMS_delta_pct_bbox=stats([y['rms_pct_bboxdiag']-x['rms_pct_bboxdiag'] for x,y in zip(aa,bb)]))
    trace=[json.loads(s) for s in (root/'training/direct/TRAIN_TRACE.jsonl').read_text().splitlines() if s.strip()]
    res['last100_loss']=stats([r['loss'] for r in trace[-100:]])
    res['alloracle_identity']={d:dict(valid=sum(r['valid'] for r in rows if r['domain']==d and r['combination']=='OOO'),
        canonical_surface_hash_count=len({r.get('surface_original_target_ids_sha256') for r in rows if r['domain']==d and r['combination']=='OOO'})) for d in domains}
    dump(dest/'RESULT_NUMBERS.json',res)
    cols=['sample_id','nfe','combination','domain','generated','oracle_assisted','status','valid','vertices','faces','beta0','beta1','beta2','nonmanifold_edges','strict_noncycle_vertex_count','exact_intersection_pairs','exact_degenerate_faces','rms_pct_bboxdiag','p95_pct_bboxdiag','wall_seconds']
    with (dest/'GEOMETRY_LEDGER.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
        if optional.exists():w.writerows(read(optional))
    txt=['# 单快照 direct FM：结果账本','',
        '固定 thin-U frame 6 的有效 q_only 状态；单目标训练 3,000 步，宽128 DeepSets。复用旧 chart 的 mean/scale，32 个预先固定噪声，64 步 Euler；本轮不是泛化或多样性实验。',
        '',
        '**有效率只表示本轮拓扑与几何有效合同，不包含形状误差阈值，也不要求恰好512面。目标网格508面。** 表面误差单独报告，参考是选定目标的完整网格。',
        '',
        'P/O 表示预测/真值；三位依次为提取位置、raw 法向、中心化 F。只有 PPP 是自由生成；其余全部是借用真值的离线诊断。OOO 的32次为同一对象重排身份检查。',
        '',
        '|位置/法向/F|fixed有效/32|enclosing有效/32|fixed表面RMS中位数(%bbox)|enclosing表面RMS中位数(%bbox)|',
        '|---|---:|---:|---:|---:|']
    for c in combos:
        x=res['combinations']['fixed'][c];y=res['combinations']['enclosing'][c]
        txt.append(f"|{c}|{x['valid']}|{y['valid']}|{x['surface_RMS_pct_bbox_all']['median']:.6g}|{y['surface_RMS_pct_bbox_all']['median']:.6g}|")
    txt+=['','所有误差汇总使用全部可评估输出，包括无效输出，不能解释为合格输出质量。失败类型和实际面数见 RESULT_NUMBERS.json / GEOMETRY_LEDGER.csv。','',
        f"训练实际完成 {res['training']['completed_steps']} 步，耗时 {res['training']['training_elapsed_seconds']:.3f} 秒；最后100步 loss均值 {res['last100_loss']['mean']:.6g}。",
        '',
        '|路径t|位置RMS(%bbox)|零速度位置RMS|法向均角(度)|零速度法向均角|局部C相对误差|零速度C相对误差|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for t,g in res['probes_by_time'].items():
        p,z=g['endpoint'],g['zero_velocity']
        txt.append(f"|{float(t):.2f}|{p['P_rms_pct_bbox']['median']:.4g}|{z['P_rms_pct_bbox']['median']:.4g}|{p['n_oriented_angle_mean_deg']['median']:.4g}|{z['n_oriented_angle_mean_deg']['median']:.4g}|{p['C_relative']['median']:.4g}|{z['C_relative']['median']:.4g}|")
    txt+=['','上表先逐噪声计算语义误差，再对32个噪声取中位数。条件路径端点诊断显式使用目标；t=.9 已含90%目标，不能记为生成。近端探针通过也不能单独证明积分器有错。','',
        f"t=.9 端点探针有效：fixed {res['probe09_geometry']['fixed']['valid']}/32，enclosing {res['probe09_geometry']['enclosing']['valid']}/32。",
        '',
        '自由采样的后验位置匹配与路径噪声耦合不同，数值对照为描述性诊断；不能把自由轨迹到某个单目标配对端点的距离直接叫作真实无条件速度误差。']
    if optional.exists():
        txt+=['','按训练前固定的同域门槛（近端探针≥24/32且自由64≤8/32）触发一次256步采样；权重、噪声均不变。']
        for d in domains:
            g=res['free256'][d];q=res['paired64_256'][d]
            txt.append(f"- {d}: {g['valid']}/32 有效；64→256救回{q['rescued']}个、丢失{q['lost']}个；配对表面RMS变化中位数 {q['RMS_delta_pct_bbox']['median']:.6g}%bbox。")
        txt+=['','此后未追加步数扫描或训练。']
    txt+=['','完整几何行在原 interventions_64/LEDGER.json（512行）、probes/GEOMETRY_ROWS.json（64行），以及如触发时 interventions_256/LEDGER.json（64行）。两个域分别报告，不取最优并集。','',
        '本文件只作描述性汇总。最终解释、零训练审计与历史角点统计更正见同目录 EXPERIMENT_REPORT_zh.md。独立复核见 ../independent_audit/。']
    (dest/'NUMERIC_REPORT_zh.md').write_text('\n'.join(txt)+'\n',encoding='utf-8')
    dump(dest/'MANIFEST.json',{'scope':'summary supplement only; original run MANIFEST unchanged','files':{str(p.relative_to(dest)):digest(p) for p in sorted(dest.rglob('*')) if p.is_file()}})
    print(json.dumps({'out':str(dest),'combination_counts':{d:{c:res['combinations'][d][c]['valid'] for c in combos} for d in domains},'free256':{d:g['valid'] for d,g in res.get('free256',{}).items()},'probe09':{d:g['valid'] for d,g in res['probe09_geometry'].items()}},indent=2))
if __name__=='__main__':main()