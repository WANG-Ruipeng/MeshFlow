"""Transcribe completed, non-blind AI_PROXY inspection of twelve actual contact sheets.

This does not infer visual labels from metrics. Every named JPG was opened with
view_image, with all four labels and both RAW/POST opposite views inspected.
"""
import csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
G = ROOT / 'evaluation/gallery'
ORDER = ['ORIGINAL_ORIG_N','ORIGINAL_LOW_N','DEDUP_ORIG_N','DEDUP_LOW_N']

# Per column: C role, body completeness, RAW cracks, POST cracks, visible extra
# surfaces, RAW observation, POST observation. UNKNOWN is not a negative finding.
D = {
('73b369b9',0):[
 ('YES','YES','YES','YES','UNKNOWN','完整板式靠背和悬挑底座，蓝色C位于可见座面；背顶和板边有细裂。','板式主体保留，靠背顶缘与座面后缘细缝仍可见。'),
 ('YES','YES','YES','YES','YES','变为弯曲薄板/条带椅，数条灰色横带跨过蓝色座面，靠背与支承有较多缝。','主体是可辨认的弯板椅；座面横带覆盖C部分表面，背座转折仍分层。'),
 ('YES','YES','YES','YES','UNKNOWN','板式椅主体和蓝色座面清楚；背顶、底座转角有细缝。','板式主体保留，板边细裂仍见，未见整块主体丢失。'),
 ('YES','YES','YES','YES','YES','条带靠背左右边缘有参差碎边，灰色座面条带跨过C。','靠背整体连续一些，但侧缘参差和灰色座面横带仍在。')],
('73b369b9',1):[
 ('YES','YES','YES','YES','UNKNOWN','蓝色座面与整块上拱靠背、板式底座构成椅体，背顶拼接有缝。','椅体主体保持，蓝色座面可辨，背顶拼接线仍见。'),
 ('NO','YES','YES','YES','YES','灰色倾斜座板位于蓝色C上方并向前伸出，C主要在下方露边；弯板有多条裂缝。','灰色倾斜座板覆盖C主要区域，后背与底座可辨但座面用途未交给C。'),
 ('YES','YES','YES','YES','UNKNOWN','蓝色座面、上拱靠背及底座齐全，背顶边缘有小裂。','板式椅保留，蓝色座面清楚，边缘少量细裂。'),
 ('NO','YES','YES','YES','YES','灰色斜座板覆盖蓝色C，靠背条带与底座开缝较多。','灰色斜座板仍主导座面，靠背右侧与底座有可见缝。')],
('73b369b9',2):[
 ('NO','YES','YES','YES','YES','完整扶手框架椅的灰色座面明显位于蓝色C上方，C只在下缘出现。','框架和灰色主座面仍完整，蓝色C仍在下方，座面采用未成立。'),
 ('NO','YES','YES','YES','YES','弯板椅中灰色斜座面覆盖蓝色C，背部横缝与侧缘缺口可见。','斜座面覆盖C未改变，靠背边缘与底座仍有缝。'),
 ('NO','YES','YES','YES','YES','灰色扶手框架和高于C的灰色座面完整；C仅下方露边。','灰色座面仍替代C，框架小裂存在，整体椅体未缺失。'),
 ('NO','YES','YES','YES','YES','灰色斜板覆盖C；弯板靠背和底座有密集细缝。','倾斜灰色座板仍覆盖C，主体可辨，局部开缝仍在。')],
('73b369b9',3):[
 ('UNCERTAIN','YES','YES','YES','YES','框架椅有蓝色水平片和上方灰色座沿/斜片，层次使座面用途有歧义，扶手附近断缝。','框架主体保留，C上方的灰色跨片仍在；扶手与背侧有开缝。'),
 ('UNCERTAIN','YES','YES','YES','YES','完整箱式弯板椅，灰色斜条跨C，座面用途有歧义；圆卷边有细裂。','连续侧板改善主体组织，但灰色斜条仍压在C上方，不能确定C承担座面。'),
 ('UNCERTAIN','UNKNOWN','YES','YES','YES','几片大板构成高背与支承，灰色斜座板跨C；是否缺少结构或特殊极简款式不确定。','仍为稀疏大板构型，C与灰色斜座板关系不清，主体完整性不确定。'),
 ('UNCERTAIN','YES','YES','YES','YES','完整箱式弯板主体比稀疏板更清楚，灰色斜条仍跨过C。','完整侧板、底座与靠背可辨，主体表面更连续，但C上方横带未消失。')],
('3f741759',0):[
 ('YES','YES','YES','YES','UNKNOWN','C是明显座面，四腿和条背齐全；背顶缺口、左座裙三角开口可见。','背顶和左座裙的小开缝仍见，主体完整。'),
 ('YES','YES','YES','YES','UNKNOWN','C为座面，腿部完整；背帽参差、竖条有细缝，左座裙开口较原N小。','座裙较整齐，但背帽仍有不规则拼缝；并非全面无裂。'),
 ('YES','YES','YES','YES','UNKNOWN','蓝色座面与四腿条背清楚，左座裙和背帽有开缝。','主体保持，左座裙三角开口和背帽局部缝可见。'),
 ('YES','YES','YES','YES','UNKNOWN','方形条背椅，座裙较完整；背条底部有参差小片贴近C。','座裙开口减少，但背条底部仍有尖碎边和局部裂口。')],
('3f741759',1):[
 ('YES','YES','YES','YES','UNKNOWN','椅体完整，背帽中部可见白色裂口，C明确为座面。','背帽小白缝仍见，主体无明显缺失。'),
 ('YES','YES','YES','NO','UNKNOWN','主体与C用途保持，背帽有细缝但较原N小，腿部规整。','背帽和腿部表面更连续，两视角未见原N那样明显的背帽白裂口。'),
 ('YES','YES','YES','YES','UNKNOWN','C为座面；背帽有白裂，右腿底端有开裂/缺角。','背帽局部白缝与腿底边缺口仍可见。'),
 ('YES','YES','YES','NO','UNKNOWN','主体完整，C为座面，背帽细缝较小，腿部规整。','背帽与腿端更连续，原N明显裂口减弱，两视角未见明显大裂。')],
('3f741759',2):[
 ('YES','YES','YES','YES','UNKNOWN','桶状围背及前裙构成椅体，C在座面中央；围壁有细长竖裂。','圆围壁细裂仍在，C可见且位于座面，整体主体保留。'),
 ('YES','YES','YES','YES','UNKNOWN','变为方形扶手箱椅，侧板更连续，C可见；角柱与前裙仍有细裂。','方形外壁比原N竖裂少，C作为内部座面可见，转角拼缝仍存在。'),
 ('YES','YES','YES','YES','UNKNOWN','桶形围壁有多处较大不规则竖向裂缝及底边缺口，蓝色C仍为座面。','围壁下部与右侧裂口明显，C仍可见，未把裂口当作合法款式直接忽略。'),
 ('NO','YES','YES','YES','YES','方形外壁较连续，但灰色生成座板几乎完全盖住蓝色C，仅有细蓝线。','外壳连续性改善，但蓝色C在座面完全不可见，灰色座板占据用途位置，记采用代价。')],
('3f741759',3):[
 ('YES','YES','YES','NO','UNKNOWN','C为座面，四腿齐全；背帽局部小缝。','整体规整，两视角未见明显大裂，C用途保持。'),
 ('YES','YES','YES','NO','UNKNOWN','C与主体保持，背帽裂口和竖条细缝可见。','背帽经POST较连续，主体与原N接近；两视角未见明显大裂。'),
 ('YES','YES','YES','NO','UNKNOWN','C为座面，背帽有明显白裂，主体齐全。','背帽裂口已减弱，主体规整，未见明显大裂。'),
 ('YES','YES','YES','NO','UNKNOWN','C为座面，背帽有白裂与细缝，腿部齐全。','背帽裂口经POST减弱，主体与原N近似，无明确额外视觉收益。')],
('85b16941',0):[
 ('YES','YES','YES','YES','UNKNOWN','板背椅和蓝色座面完整；背板左竖边及上边有裂缝。','主体完整，背板左侧裂缝仍见。'),
 ('YES','YES','YES','YES','UNKNOWN','变成粗腿条背椅，但背帽有大白裂、竖条歪斜且断续，C仍是座面。','背帽大缺口及竖条破碎/歪斜仍然明显，较原N局部表面差。'),
 ('YES','YES','YES','YES','UNKNOWN','板背与座裙完整，背板左侧和上缘有开缝。','板背和座裙边缘细裂仍见，C用途清楚。'),
 ('YES','YES','YES','NO','UNKNOWN','变为较简洁方框背椅，C为座面；主体边缘较整齐。','方框背、腿部与C清楚，两视角未见原N背板竖裂；款式变化本身不计功。')],
('85b16941',1):[
 ('YES','YES','YES','YES','UNKNOWN','不对称板背/腿部，左侧腿边有分叉小片，另一侧出现大脚块；C座面清楚。','左腿细裂和不对称脚块保留，主体可辨。'),
 ('YES','YES','YES','YES','UNKNOWN','常规条背和四腿更规整；背条底部仍有小尖片，C座面清楚。','腿部表面更完整、原N分叉腿片减少；背条底部仍有局部参差。'),
 ('YES','UNKNOWN','YES','YES','UNKNOWN','极细开放框架，框角及腿部有断裂，既可能极简款式也可能缺面；C作为座面可辨。','背框上角和腿部仍断续，主体是否缺面不确定。'),
 ('YES','YES','YES','YES','UNKNOWN','较厚四腿和横梁靠背成形，C座面可辨；背柱顶和座沿有小缝。','四腿/横梁结构更连续，明显断续的细框减少，仍有边缘细裂。')],
('85b16941',2):[
 ('YES','YES','YES','YES','UNKNOWN','板背椅完整、C为座面；背板左边及上缘细裂。','板背/座裙细裂仍在，主体完整。'),
 ('YES','YES','YES','YES','UNKNOWN','变为高侧板扶手椅，C在内部座面；大侧壁连续，背座转折有条缝。','侧壁完整但背座转折与角柱细缝仍在，主要是款式变化而非明确净改善。'),
 ('YES','YES','YES','YES','UNKNOWN','C和板背椅完整，背板边缘、座裙细裂。','完整主体保留，板背边缘细裂存在。'),
 ('YES','YES','YES','YES','UNKNOWN','高侧板箱式椅，C在座面，背板条带和转角细缝可见。','外侧壁连续，内部背板与侧墙交角仍有缝，净视觉收益不明确。')],
('85b16941',3):[
 ('YES','YES','YES','YES','UNKNOWN','C为座面，板背/四腿完整，背板左侧有裂。','主体完整，背板左侧裂缝仍见。'),
 ('YES','YES','YES','YES','UNKNOWN','变为条背粗腿椅，C仍为座面，背条歪斜与底部小碎边可见。','背板裂缝换成条背底部参差，不足以说整体更好。'),
 ('YES','YES','YES','YES','UNKNOWN','C为座面，左座裙有较大开口，背板边缘细裂。','左座裙开口仍见，主体可辨。'),
 ('YES','YES','YES','YES','UNKNOWN','变为高侧板扶手椅，C作为内部座面；侧板和背座接角有裂线。','侧板主体连续但转角、前墙边缝仍在；不同款式不能直接算改善。')],
}

# Main comparison is POST only; RAW notes are kept separate.
J = {
('73b369b9',0):[('MIXED','YES','NO','板式变为弯板/条带，未见主体丢失，但灰条跨C和接角缝增加，不能定为净改善。'),('WORSE','YES','NO','原N较整齐板面变成背侧参差和多层座条，局部表面更破碎。')],
('73b369b9',1):[('WORSE','NO','NO','灰色倾斜座板替代原本清楚的蓝色C座面，裂缝也未消失。'),('WORSE','NO','NO','蓝色C从明确座面变成灰色斜板下的露边，背侧和底座开缝。')],
('73b369b9',2):[('MIXED','NO','NO','两N都由灰色座面替代C；款式从框架到弯板且仍有裂缝，无清楚净收益。'),('MIXED','NO','NO','两N均未采用C，低N板面更薄但开缝仍在；不能以换款式记改善。')],
('73b369b9',3):[('MIXED','UNKNOWN','NO','侧板主体更连续，但灰色跨座片仍使C用途含混。'),('IMPROVED','UNKNOWN','NO','相比原N稀疏大板，箱式完整侧板/靠背/底座更清楚；C上方灰条仍使用途不确定。')],
('3f741759',0):[('MIXED','YES','NO','座裙开口减少但背帽仍有不规则拼缝，局部收益伴随新边缘问题。'),('MIXED','YES','NO','座裙改善，背条底部仍出现参差尖片；不能定为全面更好。')],
('3f741759',1):[('IMPROVED','YES','NO','背帽原有白裂口减弱、腿端更连续，C用途与完整主体保持。'),('IMPROVED','YES','NO','背帽与腿端较原N连续，未见新增主体缺失，C座面保持。')],
('3f741759',2):[('IMPROVED','YES','NO','方形围壁减少原N细长竖裂，C仍是内部可见座面，主体保持。'),('MIXED','NO','NO','围壁裂缝明显减少，但灰色生成座面遮盖/替代C，是质量收益与采用退化并存。')],
('3f741759',3):[('UNCHANGED','YES','NO','POST两者主体与小裂观感接近，未看到清楚净改善。'),('UNCHANGED','YES','NO','POST背帽均已连续、C与四腿保持；差别主要是轮廓小变化。')],
('85b16941',0):[('WORSE','YES','NO','新增明显背帽白裂及条背歪斜，C和主体虽保持但局部表面更差。'),('IMPROVED','YES','NO','方框背和腿部更连续，原N背板左竖裂减弱；仅表面连续性计为收益。')],
('85b16941',1):[('IMPROVED','YES','NO','原N分叉腿片与不对称脚块减少，四腿更规整；背条底部小裂仍在。'),('IMPROVED','YES','NO','原N断续细框变为较连续四腿/横梁，C座面保持，主体完整性更明确。')],
('85b16941',2):[('MIXED','YES','NO','大侧板更连续，但背座接角仍有缝；大幅款式切换不能算无条件改进。'),('MIXED','YES','NO','外墙连续、内部接角仍裂，缺少清楚的总体优劣。')],
('85b16941',3):[('MIXED','YES','NO','板背裂缝换成条背底部参差，C保持但没有清楚净视觉改善。'),('MIXED','YES','NO','左座裙开口变化伴随高侧墙/转角缝，不把换款式当成净改善。')],
}

with (G/'review_template.csv').open(encoding='utf-8-sig',newline='') as f: template=list(csv.DictReader(f))
rows=[]; delta=[]; images={}
for row in template:
    key=(row['parent_id'][:8],int(row['noise']))
    if key not in D:continue
    c,body,rawcrack,postcrack,extra,rawnote,postnote=D[key][ORDER.index(row['label'])]
    path=G/Path(row['image_path']).name
    sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==row['image_sha256']
    images[str(path)]=sha
    r=dict(row);r.update(reviewer='AI_PROXY',review_status='REVIEWED',review_mode='NON_BLIND_AI_PROXY',
        C_usage=c,body_complete=body,visible_fragments='UNKNOWN',visible_cracks=rawcrack if row['stage']=='RAW' else postcrack,
        visible_extra_surfaces=extra,visible_join='UNKNOWN',notes=rawnote if row['stage']=='RAW' else postnote,
        scope_note='Fresh visible C seat role; does not replace historical strict adoption/joint labels. NO visible defect means not seen in these two views, not geometric certification. No score-derived labels.')
    rows.append(r)
    if row['stage']=='POST' and row['label'].endswith('LOW_N'):
        source=row['source_arm'];j=J[key][0 if source=='ORIGINAL_C200' else 1]
        delta.append(dict(request_id=row['request_id'],parent_id=row['parent_id'],split=row['split'],source_arm=source,
            stage='POST',judgment=j[0],C_role_retained=j[1],new_major_body_loss=j[2],raw_notes=rawnote,notes=j[3],
            image_path=row['image_path'],image_sha256=sha,reviewer='AI_PROXY',review_mode='NON_BLIND_AI_PROXY'))
assert len(rows)==96 and len(delta)==24 and len(images)==12
def writecsv(path,rs):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
writecsv(G/'actual_review_train.csv',rows);writecsv(G/'visual_delta_train.csv',delta)
receipt=dict(status='COMPLETE',reviewer='AI_PROXY',mode='NON_BLIND',actual_view_image_contact_sheets=12,reviewed_cells=96,post_comparisons=24,
    visible_join_policy='UNKNOWN: contact-sheet appearance does not certify shared indices or legality.',
    uncertain_vs_unknown='UNCERTAIN=visible role ambiguity; UNKNOWN=insufficient evidence.',
    notes='All four columns and two opposite RAW/POST views inspected. Geometric score tables were not used to assign labels. Models identify frozen weights; original-N reviews are fresh and do not overwrite earlier records.',
    images=[dict(path=p,sha256=h) for p,h in sorted(images.items())],
    outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [G/'actual_review_train.csv',G/'visual_delta_train.csv']})
(G/'actual_review_train_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(rows=len(rows),comparisons=len(delta),images=len(images))))
