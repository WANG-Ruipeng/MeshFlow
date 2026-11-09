"""Persist the agent's actual eight-page visual reading; no scoring inference.

The judgments below were written after the displayed JPEGs were inspected.
All comparison labels were visible. Base rows are an independent second view,
not an inter-rater agreement estimate or a replacement for the root review.
"""
import json
from pathlib import Path
from . import common as C
from .rendering import REVIEW_FIELDS,GALLERY

# Field order: adoption, floating, replaced, wrong layer, complete, missing support,
# fragments/folded layers, visible join. U remains uncertain, never converted.
OBSERVATIONS=[
('PAIR01','DEV',0,'base','NUYYYNNU',
 '蓝色半座面在灰色独立座面下方；正面切线灰座面约Y=0.25、C约Y=-0.45，侧截面也有明显间隔。四腿和靠背存在，C还伸出框架。',
 'D0后灰座面与蓝C仍明显上下分层；框架与四腿保留。是否在未切中的位置碰到支架无法从这两刀确认。'),
('PAIR01','DEV',0,'sub4','UUUNYNNU',
 '整体变为圆柱底座加圆背形态，切线灰座面接近蓝C高度；但蓝片明显超出圆形主体轮廓，不能仅凭同层确认合理采用。',
 '圆柱底座形态保留，局部灰线与蓝片更同层，但突出主体的蓝片仍在，接缝及实际用途保留不确定。'),
('PAIR01','DEV',1,'base','NUYYYNNU',
 '箱式椅主体包围蓝片；切面灰色横向座层约Y=0.2而C约Y=-0.45，输入落在座面下方。箱式支撑可作为合法形态，不按缺四腿判坏。',
 '箱式形态和灰座层仍在C上方；蓝片被外壳遮挡但切面显示错层，未看到后处理修复其坐面用途。'),
('PAIR01','DEV',1,'sub4','NUYYYNNU',
 '形状转为细四腿椅，灰色座层明显高于蓝C；蓝片向侧方伸出，改变形态没有解决该噪声的错层。',
 '四腿椅保留，正面与侧面切线仍见蓝C低于灰色座层；无法把同一表面细分理解成这一例采用改善。'),
('PAIR01','TRAIN',0,'base','NUYYYNNU',
 '圆柱底座座面位于约Y=-0.45，蓝C约Y=0，灰座面另行形成。完整底座和背部存在，但蓝片处于偏高位置。',
 '灰色圆形座层仍低于蓝C，形态和错层基本保留；可见接合不足以判定，不能从一个投影的线相遇推出共边。'),
('PAIR01','TRAIN',0,'sub4','NUYYUNYU',
 '生成物变为带过长竖杆和层叠框架的椅形，切面出现灰色上层约Y=0.5、C约Y=0；蓝片更像座块下层，整体合理性不确定。',
 '灰色上层与蓝C之间仍有高度差，异常高竖杆和框架层保留；支撑存在但完整合理椅形无法确定。'),
('PAIR01','TRAIN',1,'base','NUYYYNNU',
 '四腿椅整体存在；正侧切面有灰色上座层高于C，蓝线夹在上下灰层之间，作为用户要求坐面的用途不能认定成立。',
 '后处理把灰色厚座轮廓整理得更清楚，蓝C仍处于其内部高度；没有消除输入被独立座层替代的迹象。'),
('PAIR01','TRAIN',1,'sub4','UUUUUUYU',
 '整体变为低矮箱式/折叠层外观，蓝面在中间露出，周围灰层与扶手折叠明显；用途和支撑结构均难以确定。',
 '低矮箱式外观和上缘折叠保持，蓝面用途仍不确定；不能因为没有原四腿就自动判支撑缺失，也不能据此称改善。'),
('PAIR03','DEV',0,'base','UUUUYNNU',
 '箱式椅包围输入，中央切面主要是蓝C而未见明显平行灰座层，存在合理座面解释；但全图遮挡较强，采用保留不确定。',
 '箱式椅及中央蓝C保留，切面没有明显大错层；蓝灰边界被侧壁遮挡，不把缺少反证当作已经采用或接上。'),
('PAIR03','DEV',0,'sub4','UUUUYNNU',
 '仍为箱式椅；蓝C上方出现很近的灰横线，可能是相邻厚度面或替代层，单凭当前截面不能区分。',
 '近邻灰层与蓝C仍并列，主体存在；没有足够证据将这条比base更明确地认定采用，也不能据薄层断定失败。'),
('PAIR03','DEV',1,'base','UUUUYNNU',
 '四腿椅中蓝C在座位高度并向右伸出，截面有上下邻近灰线；难区分合法厚度、局部支架与重复座面。',
 '四腿与靠背完整，蓝C与灰层近邻且略向外伸，接合和是否被其他层替代均保留不确定。'),
('PAIR03','DEV',1,'sub4','UUUUYNNU',
 '四腿椅仍成立，蓝片在座位上方/附近较明显地伸出；侧截面蓝线与下方灰线接近，不能把靠近直接当采用。',
 '灰线与蓝C之间的近层关系仍在，右侧突出蓝片可见；没有形成两噪声一致的更好采用证据。'),
('PAIR03','TRAIN',0,'base','YNNNYNNU',
 '大块蓝C清楚构成箱式椅的内侧坐面，靠背与扶手在其上方，中央截面未见另造的灰坐面覆盖；箱式支撑完整。',
 '蓝面继续作为箱式椅坐面，完整主体保留；图像支持用途但无法认证精确共享边，因此接合仍不确定。'),
('PAIR03','TRAIN',0,'sub4','NUYYYNYU',
 '变为四腿椅，灰色实际座层约Y=0.2而蓝C在约Y=0.8，蓝片明显悬高并超出椅背侧方；细分后用途反而变差。',
 '灰座层与高处蓝C仍明显分离，四腿和靠背在但高蓝片未纳入坐面；这是一条明确反向结果。'),
('PAIR03','TRAIN',1,'base','NUYYYNYU',
 '支撑和背部存在但灰背上缘折叠拥挤；蓝面高于独立灰坐面并大幅伸出，坐面用途失败，不能用主体存在掩盖。',
 '后处理后灰色实际座层仍远低于蓝C，背部折片仍可见；输入高悬关系未修复。'),
('PAIR03','TRAIN',1,'sub4','NUYYYNNU',
 '灰色四腿椅较规则，但蓝C仍在靠背上部高度并向外突出，另造灰座面明显更低；整体更整齐不等于控制成功。',
 '规则椅形与高悬蓝C同时保留，未见坐面采用改善；局部投影可能相遇但不认证真实接缝。'),
]

def main():
    reg=C.load_registration();mapping={(q['pair_id'],q['split'],int(q['candidate']),q['variant']):q for q in reg['requests']}
    dest=GALLERY/'review_sub4';dest.mkdir(parents=True,exist_ok=True)
    values=[];visited={}
    for pair,split,noise,variant,codes,nraw,npost in OBSERVATIONS:
        assert len(codes)==8,(pair,split,noise,variant,codes)
        q=mapping[(pair,split,noise,variant)];path=GALLERY/'images'/f'{pair}__{split}__n{noise}__BASE_SUB4.jpg';digest=C.sha(path);visited[str(path)]=digest
        for stage,note in [('RAW',nraw),('POST',npost)]:
            labels={key:{'Y':'YES','N':'NO','U':'UNCERTAIN'}[code] for key,code in zip(REVIEW_FIELDS,codes)}
            # Correct explicit completeness/support tuple for TRAIN sub4 n0.
            if (pair,split,noise,variant)==('PAIR03','TRAIN',0,'sub4'):
                labels.update(whole_body_complete='YES',support_missing='NO',fragments_or_folded_layers='YES')
            values.append(dict(**{k:q[k] for k in ('request_id','pair_id','condition_id','base_condition_id','parent_id','split','condition_type','region_type','tier','N','K','variant','candidate')},stage=stage,label='EXP67_FROZEN',reviewer='/root/local_scope_review',review_kind='AI_PROXY',review_status='REVIEWED',blind=False,image_path=str(path),image_sha256=digest,reviewed_image_sha256=digest,**labels,notes=note,
                review_scope='Actual whole view and both fixed central sections; source GT and new metric values not read. Central cuts may miss off-plane contacts.'))
    assert len(values)==32 and len(visited)==8 and len({(r['request_id'],r['stage']) for r in values})==32
    C.csvwrite(dest/'review_sub4.csv',values)
    C.write(dest/'review_sub4.json',dict(status='REVIEWED',rows=values,image_hashes=visited,images_actually_viewed=8,independent_base_rows=16,new_sub4_rows=16,
        independent_review_caution='Root reviews base TRAIN/DEV pages; retain disagreements, do not claim inter-rater reliability or vote by metrics.',
        intervention_caution='Same C surface and same free Gaussian, but K and total N change. Shape differences do not isolate token count.',
        summary='No consistent subdivision adoption benefit. PAIR01 DEV n0 changes from obvious wrong layer to closer same-level geometry but still uncertain use. PAIR03 TRAIN n0 loses a plausible adopted seat and generates a lower separate seat beneath C. Other cases retain failure or thin-layer uncertainty. RAW/POST judgments in these views agree.'))
    first=GALLERY/'images/PAIR00__n0__TRAIN_DEV.jpg'
    C.write(dest/'layout_QA.json',dict(status='PASS',actual_first_image_viewed=str(first),image_sha256=C.sha(first),all_sub4_pages_viewed=list(visited),title_and_panel_labels_readable=True,whole_and_local_views_readable=True,axis_contract='Y vertical',no_visible_label_clipping=True,judgments_are_not_topology_certificates=True))
    print(json.dumps(dict(status='REVIEWED',images=8,rows=32,rows_path=str(dest/'review_sub4.csv'))))
if __name__=='__main__':main()
