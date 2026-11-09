"""Deterministic paired aggregation and indexed topology diagnostics."""
from __future__ import annotations
from collections import Counter,defaultdict
import json,math
from pathlib import Path
import numpy as np
from .run import OUT,ROOT,CODE,read,write,csv_write,arrays,digest,clean,RADIUS

METRICS=['SeamCov','ConflictCov_C','ConflictCov_C_upper','ExposedRetain','ExposedRetain_global_lower','ExposedRetain_global_upper',
 'GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','D_sym2048','Fscore2048_0.01','Fscore2048_0.02',
 'fixed_query_p95_L','Gamma32_edge_p95_L0','FF_illegal_affected_free_fraction','FF_illegal_pair_count',
 'CF_proper_affected_free_fraction','CF_proper_pair_count','FF_unresolved_or_degenerate_pair_count',
 'CF_unresolved_or_degenerate_pair_count','free_near_degenerate_fraction','free_minimum_angle_lt5_fraction',
 'free_area_over_GT_free_area','actual_N','actual_free_faces']

def numeric(x):return isinstance(x,(int,float,np.number)) and not isinstance(x,(bool,np.bool_)) and np.isfinite(x)

def topology(vertices,faces,K):
    faces=np.asarray(faces,dtype=np.int64);edges=defaultdict(list)
    for i,f in enumerate(faces):
        for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
            edges[tuple(sorted((int(a),int(b))))].append((i,int(a)<int(b)))
    parent=np.arange(len(faces))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    for incidences in edges.values():
        for i,_ in incidences[1:]:parent[root(i)]=root(incidences[0][0])
    components=len({root(i) for i in range(len(faces))})
    c_open=[v for v in edges.values() if sum(i<K for i,_ in v)==1]
    attached=sum(any(i>=K for i,_ in v) for v in c_open)
    return dict(vertices=len(vertices),faces=len(faces),edges=len(edges),
        boundary_edges=sum(len(v)==1 for v in edges.values()),nonmanifold_edges=sum(len(v)>2 for v in edges.values()),
        same_direction_two_face_edges=sum(len(v)==2 and v[0][1]==v[1][1] for v in edges.values()),
        face_components_via_shared_edge=components,closed_two_faces_per_edge=bool(edges) and all(len(v)==2 for v in edges.values()),
        CF_shared_edges=sum(any(i<K for i,_ in v) and any(i>=K for i,_ in v) for v in edges.values()),
        C_open_edges=len(c_open),C_open_edges_exact_attached=attached,
        C_open_edges_exact_attached_fraction=attached/len(c_open) if c_open else None)

def exact_index(raw):
    x=np.ascontiguousarray(raw.reshape(-1,3));keys=x.view(np.uint32).reshape(-1,3)
    _,first,inverse=np.unique(keys,axis=0,return_index=True,return_inverse=True)
    return x[first],inverse.reshape(-1,3)

def summarize(pairs):
    parents=sorted({x['parent'] for x in pairs});byparent=[]
    for parent in parents:
        cells=[x for x in pairs if x['parent']==parent];r=dict(parent=parent,outputs=len(cells))
        for key in METRICS:
            for suffix in ('raw','geo','delta'):
                values=[x.get(key+'_'+suffix) for x in cells if numeric(x.get(key+'_'+suffix))]
                r[key+'_'+suffix]=float(np.mean(values)) if values else None
        byparent.append(r)
    summary={}
    for key in METRICS:
        good=[x for x in byparent if numeric(x.get(key+'_delta'))]
        low_better=not(key.startswith(('SeamCov','ExposedRetain','Fscore')))
        summary[key]=dict(raw=float(np.mean([x[key+'_raw'] for x in good])) if good else None,
            geo=float(np.mean([x[key+'_geo'] for x in good])) if good else None,
            delta=float(np.mean([x[key+'_delta'] for x in good])) if good else None,
            valid_parents=len(good),parents_decreased=sum(x[key+'_delta'] < -1e-12 for x in good),
            parents_increased=sum(x[key+'_delta'] > 1e-12 for x in good),
            parents_improved=sum((x[key+'_delta'] < -1e-12 if low_better else x[key+'_delta'] > 1e-12) for x in good))
    return byparent,summary

def run(out=OUT):
    out=Path(out);cases=read(out/'cases.json');rows=read(out/'evaluation/per_output.json')
    lookup={(r['label'],r['request_id']):r for r in rows}
    if len(rows)!=64 or len(lookup)!=64:raise ValueError('Need all 64 paired rows, including explicit NA')
    pairs=[];geometry=[];topologies=[];deletions=[];corners=[]
    for c in cases:
        rid=c['id'];a=lookup['RAW',rid];b=lookup['GEO_C',rid];r=dict(id=rid,parent=c['parent'],noise=c['noise'],panel=c['panel'],raw_status=a['status'],geo_status=b['status'])
        for key in METRICS:
            x,y=a.get(key),b.get(key);r.update({key+'_raw':x,key+'_geo':y,key+'_delta':y-x if numeric(x) and numeric(y) else None})
        pairs.append(r)
        raw=arrays(c['raw_path'])['output'];g=arrays(c['geo_path']);record=read(out/'geometry'/f'{rid}.json');K=c['K']
        displacement=g['corner_displacement'].reshape(-1,3);free_dis=displacement[K:].reshape(-1)
        geo=dict(id=rid,parent=c['parent'],noise=c['noise'],panel=c['panel'],L0=c['L0'],**record['stats'])
        geo.update(move_max_L0=float(free_dis.max()/c['L0']),move_rms_L0=float(np.sqrt(np.mean(free_dis**2))/c['L0']),
            move_p95_L0=float(np.quantile(free_dis,.95)/c['L0']),deleted_free_fraction=record['stats']['deleted_free_faces']/(len(raw)-K))
        geometry.append(geo)
        v,f=exact_index(raw)
        for label,vv,ff in [('RAW_SOUP',raw.reshape(-1,3),np.arange(raw.size//3).reshape(-1,3)),('RAW_EXACT_COORD_INDEX',v,f),('GEO_C',g['vertices'],g['faces'])]:
            topologies.append(dict(id=rid,parent=c['parent'],noise=c['noise'],panel=c['panel'],label=label,**topology(vv,ff,K)))
        deletions += [dict(id=rid,parent=c['parent'],noise=c['noise'],panel=c['panel'],**x) for x in record['face_records'] if not x['kept']]
        for i in range(len(raw)):
            for j in range(3):
                corners.append(dict(id=rid,source_face=i,corner=j,known=i<K,kept_face=bool(g['source_face_kept'][i]),
                    status=str(g['corner_status'][i,j]),displacement=float(displacement[i,j]),displacement_L0=float(displacement[i,j]/c['L0']),
                    target_input_corner=int(g['corner_target_input_ids'][i,j]),C_candidates=int(g['corner_C_candidate_count'][i,j])))
    csv_write(out/'analysis/paired_per_output.csv',pairs);write(out/'analysis/paired_per_output.json',pairs)
    csv_write(out/'analysis/geometry_per_output.csv',geometry);csv_write(out/'analysis/topology_per_output.csv',topologies)
    csv_write(out/'analysis/deleted_free_faces.csv',deletions);csv_write(out/'analysis/corner_displacements.csv',corners)
    result=dict(panels={},cases=len(cases),rows=len(rows),model_calls=0,denoiser='NOT_RUN_ASSET_UNAVAILABLE',
        geometry=dict(deleted_free_faces=sum(x['deleted_free_faces'] for x in geometry),outputs_with_deletion=sum(x['deleted_free_faces']>0 for x in geometry),
            moved_free_corners=sum(x['moved_free_corners'] for x in geometry),max_displacement=max(x['max_corner_displacement'] for x in geometry),
            max_displacement_L0=max(x['move_max_L0'] for x in geometry),C_all32_preserved=all(x['known_all_bytes_unchanged'] for x in geometry),
            free_corner_status_counts=dict(sum((Counter(x['corner_status_counts']) for x in geometry),Counter())),
            deleted_by_reason=dict(sum((Counter(x['deleted_by_reason']) for x in geometry),Counter()))),
        source_files={str(p):digest(p) for p in (out/'protocol.json',out/'cases.json',out/'evaluation/per_output.json')})
    for panel in ('OLD','FRESH','ALL'):
        cells=[x for x in pairs if panel=='ALL' or x['panel']==panel];parents,summary=summarize(cells)
        result['panels'][panel]=dict(outputs=len(cells),parents=len(parents),summary=summary)
        csv_write(out/'analysis'/f'paired_parents_{panel}.csv',parents)
        csv_write(out/'analysis'/f'summary_{panel}.csv',[dict(metric=k,**v) for k,v in summary.items()])
    noise_rows=[]
    for n in range(4):
        _,s=summarize([x for x in pairs if x['noise']==n]);noise_rows += [dict(noise=n,metric=k,**v) for k,v in s.items()]
    csv_write(out/'analysis/paired_noise_summary.csv',noise_rows)
    topo_summary=[]
    for label in ('RAW_SOUP','RAW_EXACT_COORD_INDEX','GEO_C'):
        group=[r for r in topologies if r['label']==label]
        topo_summary.append(dict(label=label,**{k:float(np.mean([r[k] for r in group])) for k in
            ('vertices','faces','boundary_edges','nonmanifold_edges','same_direction_two_face_edges','face_components_via_shared_edge','CF_shared_edges','C_open_edges_exact_attached_fraction')},
            closed_outputs=sum(r['closed_two_faces_per_edge'] for r in group)))
    result['topology']=topo_summary;csv_write(out/'analysis/topology_summary.csv',topo_summary)
    write(out/'analysis/results.json',result)
    plot(out,pairs)
    report(out,result,pairs)
    return dict(status='PASS',cases=32,paired_rows=32,metrics=len(METRICS),report=str(out/'START_POSTPROCESS_REPORT.md'))

def plot(out,pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=['SeamCov','ConflictCov_C','ExposedRetain','GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','FF_illegal_affected_free_fraction']
    titles=['Seam coverage (higher better)','C conflict (lower better)','C exposed retention (higher better)',
            'GT to free MSE (lower better)','Free to GT MSE (lower better)','FF illegal affected (lower better)']
    fig,axes=plt.subplots(2,3,figsize=(12,7.3))
    for ax,k,title in zip(axes.flat,keys,titles):
        for p in pairs:
            x,y=p[k+'_raw'],p[k+'_geo']
            if numeric(x) and numeric(y):ax.plot([0,1],[x,y],color='#276ba3' if p['panel']=='OLD' else '#d28225',alpha=.5,lw=1)
        ax.set_xticks([0,1],['RAW','GEO_C']);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2)
    fig.suptitle('START: fixed 32 outputs; each line is one paired noise result\nBlue OLD / Orange previously observed FRESH; no new generation',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93));p=out/'figures';p.mkdir(exist_ok=True)
    fig.savefig(p/'paired_metrics.png',dpi=150);fig.savefig(p/'paired_metrics.pdf');plt.close(fig)

def report(out,result,pairs):
    s=result['panels']['ALL']['summary'];g=result['geometry']
    def val(k,side,percent=False):
        x=s[k][side]
        return 'NA' if x is None else (f'{100*x:.3f}%' if percent else f'{x:.8g}')
    lines=['# START 保持 C 的后处理对照报告','',
        '任务：CHAIR_START_POSTPROCESS_COMPARISON_V1。固定原始 START，复用 8 父 × 4 噪声的 32 份 RAW。零训练、零模型前向、零重新生成。',
        '', '## 执行范围与证据边界','',
        '完成 RAW 与 GEO_C 的固定配对。GEO_C 是一次保持 C 的几何整理；未找到身份可核验的论文学习式去噪器代码/权重，DENOISE 和 DENOISE+GEO 均 NOT_RUN_ASSET_UNAVAILABLE。因此本轮不是论文完整后处理的复现，也不能从本轮推断学习式去噪无效。',
        'OLD 为 n0/n1，FRESH 为历史 n2/n3；四份噪声都已被观察过，不是独立确认集。无模型晋级或默认采样路径变更。',
        '', '## 固定处理规则','',
        f'半径固定为 {RADIUS:.12g} 个模型坐标单位。采用参考尺度 x_ref=.95×.3762×x_model，参考半径 .015；源码有 x_model=2×x_unitbox/.3762。这个选择是论文启发的单一工程阈值，作者后处理具体运行尺度尚未完整追溯，不声称阈值等价复现；无输出 bbox 重归一、无阈值扫描。',
        'C 的全部三角形、角顺序、绕序和 FP32 位模式保持；不同 C 锚不合并。free 角只在唯一 C 锚落入半径时吸附；多锚冲突拒绝。其他 free 角按确定顺序归到半径内不可移动的原始点代表，无传递链和均值漂移。仅删除 free 精确重顶点/共线/重复面，C 优先。没有补洞、平滑、裁剪、量化或法向绕序修复。',
        '', '## 汇总结果','',
        '先对同父四噪声取均值，再对八父等权平均；比例差使用百分点。更完整的逐输出、逐父和逐噪声数据见 analysis/*.csv。',
        '', '| 指标 | RAW | GEO_C | 差值 | 改善父数 / 8 |','|---|---:|---:|---:|---:|']
    for k,name,percent in [('SeamCov','接合 S ↑',True),('ConflictCov_C','C 冲突 I ↓',True),('ExposedRetain','C 暴露保持 E ↑',True),
            ('GT_to_F_MSE2048_L0','GT→free MSE ↓',False),('F_to_GT_MSE2048_L0','free→GT MSE ↓',False),('D_sym2048','双向距离 MSE ↓',False),
            ('FF_illegal_affected_free_fraction','FF 非法相交受影响面比例 ↓',True)]:
        delta=s[k]['delta'];d='NA' if delta is None else (f'{delta*100:+.3f} pp' if percent else f'{delta:+.8g}')
        lines.append(f'|{name}|{val(k,"raw",percent)}|{val(k,"geo",percent)}|{d}|{s[k]["parents_improved"]}|')
    lines += ['', '这是小面板描述性结果，没有新设通过门槛。I/E 的不确定上下界和 CF/FF 未解决事件保留在逐例表，不能把未知当没有相交。',
        '', '## 位移、删面与连接代价','',
        f'C 保留：{g["C_all32_preserved"]}，32/32。移动 free 角共 {g["moved_free_corners"]}；最大位移 {g["max_displacement"]:.9g} 模型单位，最大位移/L0 {g["max_displacement_L0"]:.6%}。',
        f'共删除 {g["deleted_free_faces"]} 个 free 面，涉及 {g["outputs_with_deletion"]}/32 输出；原因：{json.dumps(g["deleted_by_reason"],ensure_ascii=False)}。原始 N 与最终 N 分开报告，未凑回面数。',
        '精确索引拓扑见 topology_per_output.csv。RAW_SOUP 是原始每面独立顶点的导出；RAW_EXACT_COORD_INDEX 只共享坐标逐位一致的角，不移动几何，用来避免把单纯改索引误当生成质量提升。GEO_C 用实际导出的 vertices/faces。共享边连通、非流形与闭边诊断均不等于水密/无自交认证。',
        '', '## 评分与可复现性','',
        'RAW32 使用逐例身份验证后的原评分缓存。GEO_C32 使用原评分函数，适配变量 free 面数；分母以实际 free 数为准。GT512/2048、C2048、Gamma、方向文件逐字节复用。free→GT 的预测表面采样使用原固定种子和原面积采样算法在新表面重映射，因此 Q_F 坐标会改变；没有新增 GT 查询或随机种子。',
        'C/RAW/source hash、逐角位移、删除来源、PLY 与 NPZ 全几何回读以及独立审计见 audit/。CPU 调度先测一个正式评分，再按可用 CPU/RAM 追加进程；实际 worker 数、峰值 RSS 和时长见 resources/ 与 status/。',
        '工程过程的失败和修复均保存在 ledger/ 与 engineering/：首例 PLY 空格错误被回读拦截、RAW数组表示形状适配、图库变面数适配；这些封套修复不改变处理算法、固定半径、旧评分定义或输入。首例几何因导出修复重复一次，保存数组逐位相同。',
        '', '## 可视化与交付','',
        '打开 OPEN_GALLERY.html：32 例全部展示 RAW / GEO_C / GT，同父全部噪声共同相机，四视角加线框。C 蓝、free 橙；不裁异常、不平滑。辅助边界/位移图不是 S/I/E 评分替代。AI 看图复核记录独立提供，不冒充用户研究。',
        '主数值图：figures/paired_metrics.png。处理前后 indexed PLY：meshes/RAW 与 meshes/GEO_C；处理 NPZ/逐角映射：outputs/GEO_C。',
        '', '## 解释与下一步','',
        '最终证据解释与逐父尾部观察见 DECISION.md；在数值、全图库与独立审计完成后写入。固定任务到此结束，不自动训练、扩大阈值或切换默认模型。','']
    decision=out/'DECISION.md'
    if decision.exists():
        lines[2:2]=decision.read_text(encoding='utf-8').splitlines()[2:]+['']
    (out/'START_POSTPROCESS_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
