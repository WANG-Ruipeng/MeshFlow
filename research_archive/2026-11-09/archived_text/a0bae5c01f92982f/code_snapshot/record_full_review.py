"""Root's actual review of the four FULL pages, viewed 2026-10-13."""
from pathlib import Path
import csv,hashlib,json
ROOT=Path('/mnt/e/MeshFlow/experiments/r2_seat_quick_learnability_v1/evaluation/gallery')
rows=list(csv.DictReader((ROOT/'AI_PROXY_review_template.csv').open(encoding='utf-8-sig')))
result=[]
for r in rows:
    if r['condition_type']!='SEAT_FULL':continue
    p=Path(r['image_path']);h=hashlib.sha256(p.read_bytes()).hexdigest()
    assert h==r['image_sha256']
    r.update(review_status='REVIEWED',reviewer='AI_PROXY_ROOT_ACTUAL_IMAGE',blind=False,reviewed_image_sha256=h,
             C_adoption='YES',C_floating='NO' if r['parent_id'].startswith('8c9b') else 'UNCERTAIN',
             C_replaced_by_other_surface='NO',C_wrong_layer='NO',whole_body_complete='YES',support_missing='NO',
             fragments_or_folded_layers='UNCERTAIN',visible_join='YES' if r['parent_id'].startswith('8c9b') else 'UNCERTAIN')
    if r['parent_id'].startswith('8c9b'):
        r['notes']='实际查看本噪声三模型RAW/POST整椅与双中心截面。三模型均将蓝C置于座体上表面，灰侧壁在端部接近C；椅背、座框和支撑布局相近，没有清楚的采用改善。图中三角条纹不自动判非法折叠；视觉接触不认证索引共边。'
    elif r['label']=='SMALL_C200':
        r['notes']='实际查看整椅与双截面。蓝C保留为座体上表面，灰座框变深、背板和腿形明显改变；两个噪声趋于相近厚座框款式。不能仅凭款式改变判采用提升或多样性坍缩。中心截面不足以认证接触，visible_join与悬空保留不确定；没有明显另造上层灰座面。'
    else:
        r['notes']='实际查看整椅与双截面。蓝C处于可解释的座面位置，灰色下层可解释为座框/底面，不据此强判替代座面；中央截面未能证明接续，接触与悬空记不确定。n1开放框架靠背可为合理款式，不按唯一来源GT判缺失。与另一非集中组的整体款式相近。'
    result.append(r)
assert len(result)==24
with (ROOT/'AI_PROXY_review_full.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
print(json.dumps(dict(actual_pages=4,reviewed_rows=24,geometry_or_score_changes=0)))