from pathlib import Path
import ast
p=Path('/mnt/e/MeshFlow/experiments/mf_h1_local0/chair_bounded_attachment_v1/reporting.py')
s=p.read_text();s=s.replace('import json\n','import json\nimport csv\n')
s=s.replace("        next_step='继续保留 RAW。若另行授权，唯一优先方向是提高生成端对 C 边双端位置的局部可匹配性；检验假说为：当前 small-budget 投影的适用率受同时落入两端局部半径的自由边稀缺限制，只有新增方法在同样位移预算内提高跨父候选率，并同时通过 strict 接缝与质量门禁，才值得继续。'", "        next_step='继续保留 RAW / START，停止本轮局部修正路线。本固定任务不追加失败分类研究、训练、采样或参数扫描。'")
s=s.replace("        next_step='保留 ENDPOINT 作为开发集上的可选局部插件。唯一后续是独立未观察父对象的冻结复核，检验同一预算与规则能否维持接缝收益且不增加冲突或主体误差。'", "        next_step='保留 ENDPOINT 作为本开发集上被门禁支持的可选局部插件，本固定任务停止，不启动额外研究。'")
s=s.replace("        next_step='仅保留本轮被门禁支持的方法。唯一后续是独立未观察父对象上复核同一固定方法的联合收益与总成本，检验开发集收益能否复现。'", "        next_step='仅保留本轮被门禁支持的方法，完成固定任务后停止，不自动启动后续实验。'")
s=s.replace("        '实际源码、code.diff、保护前后审计由最终 delivery 阶段封存。'", "        '实际源码、code.diff、保护前后审计由最终 delivery 阶段封存。'")
s=s.replace("        '- 原来源 hash 在 registration/source_hashes.json；实际源码、code.diff、保护前后审计由最终 delivery 阶段封存。','']", "        '- [质量成本紧凑表](quality_cost.csv)、[实际源码hash](source_hashes.json)、[代码diff](code.diff)、[保护依赖审计](audit/protected_dependencies.json)、[全部制品清单](artifact_manifest.json)。',\n        '- 实际新源码及旧评分依赖位于 source_snapshots/new 与 source_snapshots/dependencies；[恢复入口](README_RESUME.md)。','']")
s=s.replace("    report=out/'BOUNDED_ATTACHMENT_CORRECTION_REPORT.md';report.write_text", """    events=[]
    for record in modified:
        log=read_json(record['intervention_log_path']);event=log.get('geometry_events',{})
        remain=Counter((x[0],x[-1]) for x in event.get('after',[]))
        events.append(dict(uid=record['uid'],candidate=record['candidate'],scope=event.get('scope'),
            removed=len(event.get('removed',[])),added=len(event.get('added',[])),remaining={str(k):v for k,v in remain.items()}))
    if events:
        index=lines.index('## 全八父 strict 配对与两噪声')
        lines[index:index]=['接受检查仅覆盖实际改变的 free face 所触及的面片对，原非法交可保留；不是全局清理。修改例的实际事件：'+json.dumps(events,ensure_ascii=False)+'。','']
    quality_rows=[]
    for label in LABELS:
        item=dict(label=label,scored_outputs=counts[label],parents=8)
        for key,name in quality_keys:item[key]=mean(label,key)
        for key in ('raw_generation_seconds','project_seconds','deployment_seconds'):item[key]=mean(label,key)
        item['endpoint_CPU_wall_seconds']=ep.get('seconds') if label=='ENDPOINT' else None
        item['gallery_initial_render_seconds']=gallery.get('initial_render_seconds',gallery.get('seconds'))
        item['gallery_latest_run_seconds']=gallery.get('latest_run_seconds',gallery.get('seconds'))
        quality_rows.append(item)
    with (out/'quality_cost.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(quality_rows[0]));writer.writeheader();writer.writerows(quality_rows)
    report=out/'BOUNDED_ATTACHMENT_CORRECTION_REPORT.md';report.write_text""")
ast.parse(s);p.write_text(s);print('REPORT_UPDATED_AST_PASS')