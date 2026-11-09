import csv,json,hashlib,statistics
from pathlib import Path
R=Path('/mnt/e/MeshFlow-Control-artifacts')
def readcsv(p): return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def table(headers,rows): return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
notes=json.loads((R/'analysis/visual_review_root_notes.json').read_text())
gallery=json.loads((R/'gallery/gallery_manifest.json').read_text())
pages={(r['condition_id'],r['candidate']):r for r in gallery['pages']}
reviews=[]
for n in notes:
    page=pages[(n['condition'],n['noise'])]
    p=Path(page['blind_page'])
    assert sha(p)==page['blind_sha256']
    for label,note in n['models'].items():
        reviews.append(dict(request_id='P2__'+n['condition']+'__n'+str(n['noise']),panel='P2',parent_id=n['condition'],condition_id=n['condition'],candidate=n['noise'],label=label,review_type='AI_VISUAL_REVIEW',human_usability='NOT_RUN',C_compatibility=n['common'],shape_and_local_defects=note,style_vs_fragmentation='款式差异不自动视为失败；明确尖片、错层、C外伸和双板位置按本行文字分别保留。',certainty=n['certainty'],blind_page=str(p),blind_sha256=page['blind_sha256'],reference_status='NA_NO_SOURCE_GT',reviewer='root',review_scope='front3q,back3q,front,side,wire; actual image viewing; method labels visible',image_transport='Original PNG decoded; lossless WebP in memory, displayed in tool; no mesh/image-file modifications. Small details remain resolution limited.'))
assert len(reviews)==64
with (R/'analysis/visual_review_root.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(reviews[0]));w.writeheader();w.writerows(reviews)
dump(R/'analysis/visual_review_root.json',dict(status='AI_VISUAL_REVIEW_COMPLETE',output_count=64,pages=16,human_usability='NOT_RUN',viewing_order='All P2 pages contain no source target. Root read metrics before images; not method-blind or score-blind.',rows=reviews))
profiles=readcsv(R/'analysis/comparisons/checkpoint_distributions.csv')
lookup={(r['panel'],r['cohort'],r['label'],r['metric']):r['mean'] for r in profiles}
div=readcsv(R/'analysis/comparisons/diversity_by_cohort.csv')
def v(panel,cohort,label,m):
    if m=='D_free_mean':
        return statistics.fmean(float(r[m]) for r in div if r['panel']==panel and r['cohort']==cohort and r['label']==label)
    return float(lookup[(panel,cohort,label,m)])
metrics=[('CF proper / free %','CF_proper_affected_free_fraction',100),('FF illegal / free %','FF_illegal_affected_free_fraction',100),('C冲突覆盖 I %','ConflictCov_C',100),('I crossing %','ConflictCov_C_cross',100),('I near-duplicate %','ConflictCov_C_dup',100),('参考 S %','SeamCov',100),('参考 E %','ExposedRetain',100),('GT→F MSE/L² ×1000','GT_to_F_MSE2048_L0',1000),('F→GT MSE/L² ×1000','F_to_GT_MSE2048_L0',1000),('Dsym ×1000','D_sym2048',1000),('固定查询 p95 / L %','fixed_query_p95_L',100),('固定查询 coverage %','coverage_fraction',100),('Diversity ×1000','D_free_mean',1000)]
tabs={}
for step in (500,1000):
    labels=['START']+[a+str(step) for a in 'ABC']
    tabs['p1_'+str(step)]=table(['指标']+labels,[[name]+[f'{v("P1","ALL",l,m)*s:.4f}' for l in labels] for name,m,s in metrics])
unit=readcsv(R/'analysis/comparisons/per_unit_metrics.csv')
u={(r['parent_id'],r['label'],r['metric']):r['value'] for r in unit if r['panel']=='P2'}
labels=['START','A1000','B1000','C1000']
rows=[]
for c in ['P2_seat2','P2_seat32','P2_back2','P2_seat_back4']:
    for name,m,s in metrics[:5]:
        rows.append([c,name]+[f'{float(u[(c,l,m)])*s:.3f}' for l in labels])
tabs['p2']=table(['条件','指标']+labels,rows)
tabs['diversity']=table(['面板/噪声']+['START','A1000','B1000','C1000'],[[p+'/'+c]+[f'{v(p,c,l,"D_free_mean")*1000:.5f}' for l in labels] for p,c in [('P1','OLD'),('P1','NEW'),('P1','ALL'),('P2','OPEN')]])
pairs=readcsv(R/'analysis/comparisons/paired_parent.csv')
rows=[]
for pid in sorted({r['parent_id'] for r in pairs if r['panel']=='P1'}):
    out=[pid[:8]]
    for contrast in ('B1000-A1000','C1000-A1000','C1000-B1000'):
        r=next(r for r in pairs if r['parent_id']==pid and r['panel']=='P1' and r['cohort']=='ALL' and r['contrast']==contrast and r['metric']=='D_sym2048')
        out.append(f'{float(r["relative_delta_percent"]):+.3f}%')
    rows.append(out)
tabs['p1_parent_Dsym']=table(['父ID前8位','B−A Dsym相对差','C−A','C−B'],rows)
dump(R/'analysis/report_tables.json',tabs)
print(json.dumps(dict(status='PASS',visual_rows=64,tables=list(tabs))))
print(tabs['diversity'])
print(tabs['p1_parent_Dsym'])
