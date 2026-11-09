import json,csv,hashlib,datetime
from pathlib import Path
R=Path('/mnt/e/MeshFlow-Control-artifacts');A=R/'analysis'
p=A/'visual_review_scoring_blind.json'; b=json.loads(p.read_text())
m=json.loads((R/'gallery/gallery_manifest.json').read_text())
pages={(x['condition_id'],int(x['candidate'])):x for x in m['pages']}
rows=[]; receipts=[]
for row in b['rows']:
    x=dict(row);uid=x['parent_id'];n=x['candidate'];g=pages[(uid+'_C1',n)]
    for kind in ['blind','reference']:
        q=Path(g[kind+'_page'])
        sha=hashlib.sha256(q.read_bytes()).hexdigest()
        assert sha==g[kind+'_sha256'],(q,sha,g[kind+'_sha256'])
        x[kind+'_page']='E:/'+str(q).split('/mnt/e/')[1]
        x[kind+'_sha256']=sha
    x['source_reference_seen']=True
    if uid.startswith('c1'):
        x['reference_observation']='来源为较细腿的直背无扶手椅，C是原上背片；C本身的三角边界不能归为新生成损坏。'
        if n==4:
            x['reference_observation']+='本噪声输出均带高脚和下横撑，属于替代款式；A1000下横撑长斜薄片仍是局部缺陷。'
        elif n==1:
            x['reference_observation']+='来源无所见上翘尖片/座下悬片；开口和厚薄差异不独立判错，薄片观察保留。'
        else:
            x['reference_observation']+='来源更简单，但不要求补全复刻来源；无参考观察中的橙色局部折片仍独立记录。'
    else:
        x['reference_observation']='来源为细长高背无扶手椅且有下横撑；实背/条栅、圆拱和扶手变化不是自动失败。'
        if n==1:
            x['reference_observation']+='C1000新增局部锯齿折片在参考页仍清楚，其椅子主体未消失；A/B1000也有座下长薄片。'
        elif n==4:
            x['reference_observation']+='C1000去扶手后的主体更简洁不能算缺失来源必需结构；细悬线仍保留为局部疑点。'
        elif n==5:
            x['reference_observation']+='1000三个模型侧面较500简洁；圆拱顶部仍有不齐开口，不能据此认证完整闭合。'
        else:
            x['reference_observation']+='1000三个模型都有较大款式变化及扶手薄片；不能把这种共同变化归因为C分支独有。'
    x['reference_changes_blind_assessment']='不撤销盲观测；加入来源款式与已知C边界限定。'
    x['review_status']='COMPLETE_ASSIGNED_SCOPE'
    x['image_readability']='四个实体视角与线框可读；细缝/遮挡下连接及精确相交仍不可据像素裁决。'
    x['view_scope']=json.dumps(x['view_scope'],ensure_ascii=False)
    rows.append(x)
fields=list(rows[0])
with (A/'visual_review_scoring.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
assert len(rows)==56 and len({(x['request_id'],x['label']) for x in rows})==56
for key in sorted({(x['parent_id']+'_C1',x['candidate']) for x in rows}):
 g=pages[key];receipts.append({k:g[k] for k in ['condition_id','candidate','blind_page','blind_sha256','reference_page','reference_sha256','labels','view_scope']})
receipt=dict(status='COMPLETE_ASSIGNED_SCOPE',review_type='AI_VISUAL_REVIEW',human_usability='NOT_RUN',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_count=2,request_count=8,model_output_rows=56,blind_pages_actually_viewed=8,reference_pages_actually_viewed=8,order='All blind pages viewed and notes saved before reference pages; no original blind observations overwritten.',blind_notes_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),csv_sha256=hashlib.sha256((A/'visual_review_scoring.csv').read_bytes()).hexdigest(),gallery_manifest_sha256=hashlib.sha256((R/'gallery/gallery_manifest.json').read_bytes()).hexdigest(),notes='f596 n1 reference page read in original-pixel overlapping vertical strips because full page exceeded tool text limit. No source image was modified.',pages=receipts)
(A/'visual_review_scoring_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
md="""# P1 后两父的实际 AI 视觉审阅

范围：c1b8c2f0c4f3cc43aff7c08b060f5ed6、f59627a077dcc2566c63b8b48495261a，各 n0/n1/n4/n5；每请求 START、A500、B500、C500、A1000、B1000、C1000，共 56 个模型输出。四个实体视角和线框均已实际看过。先看完 8 页 blind 并固定逐条观察，再看全部 8 页 reference。逐输出见 visual_review_scoring.csv，原盲观察保留在 visual_review_scoring_blind.json，源图哈希见 receipt。

这是 AI_VISUAL_REVIEW。人评为 NOT_RUN，不产生可用率，也不是对全 8 父的独立审阅结论。

c1b 的主体在这些输出中仍清楚。n0/n5 的三臂差异较小，局部上背折片持续存在。n1 的 START 大尖片和悬垂片在 500 步输出中减弱，但 1000 步三个模型又有上翘薄片与下背三角开口。n4 是完整的高脚凳式变体；A1000 的下横撑长斜薄片较突出，B1000/C1000 较简洁。高脚、下横撑、背部开口、腿粗细与来源不同不自动构成破碎。

f596 的变化更依赖噪声。n0 的三个 1000 步模型都转为圆拱扶手椅，扶手及座周有突兀三角薄片；款式变化不是缺陷，薄片是另一个问题。n1 的 C1000 背横条更交错和参差、扶手附近上翘片与细悬片更多，比 A/B1000 局部更紊乱，但椅子主体仍可辨认。相反，n4 的 C1000 是较简洁的无扶手圆拱椅，去掉扶手不能判为主体缺失。n5 三个 1000 步输出侧面均比各自 500 步更简洁，拱顶仍有不齐开口。因此不能据一页概括 C 一致改善或一致退化。

参考只提供一种来源补全。c1b 的蓝 C 自带不规则三角边界，不能把继承的蓝面边界计成新生成损坏。f596 的来源是细长高背无扶手椅，有下横撑；生成扶手、条栅或圆拱是可接受的款式维度，评价需与尖片和折面分开。蓝 C 与橙面邻近或遮挡，只能给视觉相容性线索，不能证明焊接、正确接缝、无重叠、拓扑连通或水密；字节保留由独立合同审计判断。
"""
(A/'visual_review_scoring.md').write_text(md,encoding='utf8')
print(json.dumps({'status':receipt['status'],'rows':len(rows),'csv_sha256':receipt['csv_sha256'],'receipt':str(A/'visual_review_scoring_receipt.json')},ensure_ascii=False))
