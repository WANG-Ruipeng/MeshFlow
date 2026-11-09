"""Transcribe the actually viewed 12 DEV contact sheets; no geometry/model work."""
import csv, hashlib, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
LABELS=('FULL_S6000','ORIGINAL_C200','DEDUP_C200')
# C-use, visible-crack flag, shared visual observation, RAW-specific, POST-specific.
observations={
('2d08a64e',0):[
 ('YES','NO','蓝色C呈座面，灰色弯曲侧裙、靠背和滑橇式支承均可辨。','边沿有细碎栅格纹理，不能据此认证散片。','主体和座面位置保留，边沿较规整。'),
 ('YES','NO','蓝色C仍是座面，靠背下段更整块；前侧灰色框/裙在C下方。','前缘局部伸出与叠层关系仅图中难分。','前缘仍有外伸和层次，不能认定整段已接上。'),
 ('YES','UNKNOWN','蓝色C仍呈座面，生成背部改为数条横向板带，支承仍在。','横向白带可能为款式开口，不能直接标为破裂。','横向板带和座面下曲面保留，未见大块丢失。')],
('2d08a64e',1):[
 ('YES','UNKNOWN','蓝色C处在箱形座框内；靠背、座框和支腿可辨。','靠背底部与座框之间有横向白带，是否设计开口不确定。','该横向开口仍可见，不能由外观推断合法共边。'),
 ('YES','UNKNOWN','与冻结模型相近的座框/靠背/支腿结构，C仍呈座面。','靠背底部与座框处有白带。','背部与座框的白带变窄但仍不足确认接续。'),
 ('UNCERTAIN','UNKNOWN','改成板状支承的窄体椅；蓝色C向前/侧方悬伸，是否被合理整合存在歧义。','款式与另两组明显不同，不能仅因面更少就判质量提高。','板状支承保留，C外伸仍在；未见回到另两组的围合座框。')],
('2d08a64e',2):[
 ('YES','UNKNOWN','蓝色C呈座面；薄背板、曲面侧裙、滑橇支架齐全。','座侧可见短白缝，接触关系不确定。','侧裙与C边缘仍有偏置，未见普遍新增接续的视觉证据。'),
 ('YES','UNKNOWN','主体与冻结模型相近，C仍呈座面，背部略更整块。','前侧C与灰色下框有局部错层外观。','该前缘层次仍可见，未见明显主体缺失。'),
 ('YES','UNKNOWN','同样的曲面座裙和滑橇支架，C视觉用途保留。','侧边/前缘仍有细缝，不能认证无裂。','整体较规整但C轮廓与灰色下框并非处处一致。')],
('2d08a64e',3):[
 ('YES','UNKNOWN','C呈座面；背板与折叠板状/框状底座可辨，座前缘有悬伸。','底座存在三角空区与斜板，是否不期望缺面不确定。','底座空区与斜板保留，不能认定已经修复。'),
 ('YES','NO','改成包围更厚的低矮块状椅体，C在座位区域。','相比冻结款式明显变化，外壳更包围。','包围式外壳保留，未见明显大缺片；这不是同款式恢复证据。'),
 ('YES','UNKNOWN','保持板状悬臂支承风格，C在座面位置，前缘仍悬伸。','底座较简洁，背/座/底座间仍有白色开口。','板状底座与白色开口保留；不能仅凭简洁判连接改善。')],
('610b1621',0):[
 ('YES','UNKNOWN','C位于扶手椅的座面区域，侧板/背/低脚齐全。','座前灰色层状框较复杂，是否重复覆盖未知。','前侧层状框保留，隐藏冲突不可从图排除。'),
 ('YES','NO','C仍呈座面，生成扶手椅前板较整块。','前板简化，C前缘略外伸。','前板保持整块；不能将平滑外观当作合法接口。'),
 ('YES','UNKNOWN','C仍呈座面，外壳/短脚完整。','背部上沿存在白色开口，可能是板件间隙。','背上开口仍在，前板较整块，未见主体大缺失。')],
('610b1621',1):[
 ('YES','UNKNOWN','蓝色C处在带厚侧板的椅体座位区域，主体完整可辨。','座前灰色斜板与背上沿可见层次/白带。','背上缘白带与座前层次仍在；是否裂缝或构造空隙未知。'),
 ('YES','UNKNOWN','C视觉座面用途保留；侧板/背板/底座完整可辨。','座前仍有折面层次，未显示一致消失。','表面较整块，座前下方仍有明显层次。'),
 ('YES','UNKNOWN','同类扶手椅，C仍呈座面，未见主要部件缺失。','背上缘和座前存在细白带/层次。','背顶白带仍可见，隐藏穿插与精确接触未知。')],
('610b1621',2):[
 ('NO','NO','生成的小椅架明显窄于蓝色C，背架从C内部位置竖起；C向多侧大幅外伸。','椅体可辨，但不是对这块完整座面轮廓的合理采用。','后处理后尺度/轮廓失配仍在。'),
 ('NO','NO','同样是小椅架配过大的蓝色C，背架仍位于C内部。','没有看到整体顺应C轮廓。','尺度失配保留，未见相对冻结模型的采用修复。'),
 ('NO','NO','仍是窄小四腿/条栅背架配大面积C，C多边大幅悬伸。','C与主体比例问题未消失。','后处理后仍保留相同失配。')],
('610b1621',3):[
 ('UNCERTAIN','NO','圆孔背板和四腿可辨，C比生成座框更宽并外伸；视觉用途与轮廓相容性存在歧义。','孔洞像背板款式，不能记为裂缝。','C外伸保留，图中不支持精确接续认证。'),
 ('UNCERTAIN','NO','圆孔背与粗腿保留，蓝色C轮廓仍超出生成座框。','没有看到明确的轮廓收敛收益。','外伸仍在，未见整体大缺失。'),
 ('UNCERTAIN','UNKNOWN','圆孔背/四腿仍在，C外伸仍明显。','背板孔下方局部白缝/细碎表面可见，但非流形性不可目测。','孔下局部表面更整块，整体轮廓关系未根本改变。')],
('608b46b0',0):[
 ('YES','UNKNOWN','C呈座面，灰色座框明显内缩于C外轮廓；圆顶背板与腿齐全。','腿与背顶有细缝/细碎纹理，无法排除栅格效应。','内缩座框仍在，局部腿缝未能确认消失。'),
 ('YES','YES','C仍呈座面且灰色座框内缩；主体可辨。','前腿有细长白色开口/分叉外观，并非明显独立漂浮碎片。','前腿细长开口仍可见，不能称为全面补裂。'),
 ('YES','YES','C视觉用途保留，但灰色内框仍未到C轮廓。','前腿有较明显纵向白色开口，背板底部亦有白隙。','纵向腿隙及背底白隙仍可见，主体未大块丢失。')],
('608b46b0',1):[
 ('YES','NO','蓝色C作为扶手椅座位区域，背/侧板/低脚完整。','座前下方有层状灰色板件；隐藏重叠未知。','整体轮廓保留，座前板层仍在。'),
 ('YES','NO','与冻结模型相近，C仍呈座面，主体完整可辨。','灰色前裙与C存在层次，无法判断隐藏覆盖。','形态接近冻结模型，未见大幅连接改变。'),
 ('YES','NO','改成较低背、较高侧板的方箱式椅，C仍处于座位区域。','款式明显变化，未见独立漂浮片。','低背方箱式形态保留，不能把改款式等同几何改善。')],
('608b46b0',2):[
 ('YES','NO','蓝色C是方箱式扶手椅内部座面；侧板与背板齐全。','未见明显独立散片。','整体形态和C用途保留。'),
 ('YES','YES','C用途保留，但一侧板出现明显三角开口，可透见内部蓝色。','该开口区别于冻结组整块侧板；不声称其必然为非法破裂。','三角开口保留，不能称为后处理修复了所有裂口。'),
 ('YES','NO','变成更高、更厚侧板与底脚的椅体，C仍呈内部座面。','体型/背部折面明显变化，未见孤立漂浮片。','高侧板形态保留，整体未见主要部件缺失。')],
('608b46b0',3):[
 ('YES','UNKNOWN','C呈座面，但内缩灰色座框与背底白色间隙仍可见。','腿有窄白缝/细碎纹理，物理裂缝与栅格效应难分。','背底仍有局部白隙，不能断言整个轮廓接上。'),
 ('YES','UNKNOWN','同类圆顶背与四腿，C用途保留；内缩灰色座框仍在。','背底横向白带较显眼，腿局部也有窄白缝。','背底白隙仍见，整体未缺大部件。'),
 ('YES','UNKNOWN','C呈座面，圆顶背和四腿保留，灰色座框仍在C内部。','背底横向白带可见；没有明显独立漂浮碎片。','局部白带较窄但仍不足认证真实共边，内框偏置未消失。')],
}

manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf8'))
pages={Path(p['path']).name:p for p in manifest['images']}
viewed=[]
for prefix,n in observations:
    name=f'{prefix}_n{n}.jpg';p=pages[name]
    actual=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    assert actual==p['sha256']
    viewed.append(dict(image_path=p['path'],image_sha256=actual,viewed=True,
                       views_reviewed=['RAW v0','RAW v1','POST v0','POST v1'],
                       labels_reviewed=list(LABELS)))
with (HERE/'review_template.csv').open(encoding='utf-8-sig',newline='') as f:
    template=list(csv.DictReader(f))
rows=[]
for row in template:
    key=(row['parent_id'][:8],int(row['noise']))
    if key not in observations:continue
    usage,cracks,note,raw,post=observations[key][LABELS.index(row['label'])]
    row.update(reviewer='AI_PROXY',review_status='ACTUALLY_VIEWED',review_mode='NON_BLIND_AI_PROXY',
        C_usage=usage,body_complete='YES',visible_fragments='NO',visible_cracks=cracks,
        visible_extra_surfaces='UNKNOWN',visible_join='UNKNOWN',
        notes=note+(' '+raw if row['stage']=='RAW' else ' '+post)+
        ' 仅两视角外观；NO表示未见，不认证无缺陷。隐藏重叠及精确拓扑接续UNKNOWN。该父未进入本次16父续训池。')
    assert pages[Path(row['image_path']).name]['sha256']==row['image_sha256']
    rows.append(row)
assert len(rows)==72 and len(viewed)==12
assert len({(r['request_id'],r['label'],r['stage']) for r in rows})==72
out=HERE/'actual_review_dev.csv'
assert not out.exists(),'Preserve previous review; do not silently overwrite'
with out.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
receipt=dict(status='COMPLETE',reviewer='AI_PROXY',review_mode='NON_BLIND_AI_PROXY',
    images_actually_viewed=12,rows=72,model_calls=0,new_render_calls=0,
    historical_split='DEV',continued_training_parent_exposure=False,
    source_manifest_sha256=hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest(),
    review_path=str(out),review_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),viewed=viewed,
    limitation='No score-derived visual labels. Both views inspected per RAW/POST. Hidden overlap and exact topology remain UNKNOWN. No-source-style ranking.')
(HERE/'actual_review_dev_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(status='COMPLETE',images=12,rows=72,review_sha256=receipt['review_sha256'])))
