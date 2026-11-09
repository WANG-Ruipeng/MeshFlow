"""Actual nonblind image observations: sorted requests16..31, both frozen arms."""
import datetime
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
KEY='NATIVE_S15000__SEAM_S15000_full'
GALLERY=ROOT/'gallery'/KEY
PRIOR=ROOT/'visual_review/NATIVE_S10000_process'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest_path=ROOT/'gallery_manifests'/(KEY+'.json')
manifest=read(manifest_path); completion=read(GALLERY/'completion.json')
assert len(manifest['pages'])==len(completion['images'])==48
assert completion['definition_sha256']==sha(GALLERY/'definition.json')
pages=sorted(manifest['pages'],key=lambda p:p['request_id'])[16:32]
schema=read(PRIOR/'review_schema.json')
schema.update(checkpoint='NATIVE_S15000 / SEAM_S15000',actual_view_count=16,
    scope='Part02: sorted request_id indices16..31, both arms;64 RAW/POST observations',
    prior_schema_sha256=sha(PRIOR/'review_schema.json'))
schema['limitations'][-1]='This16-request subset of the15000 full panel is not the entire evaluation, a final checkpoint decision, or proof of convergence.'
schema['limitations'].append('Both model labels and all four columns were visible: this is nonblind AI_PROXY review.')
C='COMPLETE_MACRO'; P='PARTIAL'; U='UNCERTAIN'; V='PLAUSIBLE'; X='NO_VISIBLE_COHERENT_USE'; Q='NO_OBVIOUS_IN_VIEWS'
# Each arm has separate RAW/POST body and thin-sheet labels and independent notes.
# Tuple: body, C-use, intended-role, layer, thin, RAW note, POST note, change.
notes=[
('3db1d3904f058b574a9ca2c84bced0dd__BACK_SURFACE_1__n0','完整靠背表面',
 ((C,P),V,'YES',U,(U,'YES'),
  '蓝竖板位于灰座板后沿，灰座板与两端板状支撑形成长凳。侧截面灰座板向蓝板附近延伸，靠背位置合理；不能由图像证明共享连接。',
  '蓝背板位置保留，但灰座板出现更长开口和窄条，侧截面变为许多短段；表面完整性较弱。短段不直接证明拓扑断片。',
  '靠背位置保持，POST座板更薄且长开口明显。'),
 ((C,P),U,'YES','YES',(U,'YES'),
  '蓝板处于背部，周围还出现大片斜向灰三角面和背侧灰层；座板及两端支撑形成主体。C附近有更多灰面，但是否自然组成同一背板不确定。',
  '背部灰三角层仍覆盖或贴近蓝板部分区域，灰座板条带化；侧截面仍不足以确认背部与座板的拓扑接续。',
  '背部灰层保留，POST同时使座板变薄和开口。')),
('3db1d3904f058b574a9ca2c84bced0dd__BACK_SURFACE_1__n1','完整靠背表面',
 ((C,P),X,'PARTIAL','YES',(U,'YES'),
  '另一整块灰高背板位于蓝板前方，蓝板主要在背侧露出；侧截面有前后分层。灰体自身有座条和支撑，未见蓝板成为连续灰背的一部分。',
  '双背层关系持续，灰座条和腿部进一步变薄，腿上出现三角形开口。未见蓝板采用改善。',
  '双背层不变，POST座面与腿部更薄。'),
 ((C,P),X,'PARTIAL','YES',(U,'YES'),
  '蓝板后侧与灰整背仍处于不同位置，边缘有灰斜片靠近蓝板；灰体座条与支撑完整可辨，但大片灰背仍替代了蓝板所在表面。',
  '靠近蓝板边缘的灰片仍在，座条和腿部出现更显著细条/开口；侧截面有斜线并不认证有效桥接。',
  '局部灰斜片保留，宏观双背层未消除。')),
('3db1d3904f058b574a9ca2c84bced0dd__COMBO_PART_ASSEMBLY__n0','背板加两端腿架组合',
 ((C,C),V,'YES','YES',(U,'YES'),
  '蓝腿架位于灰座面下方，蓝背板处于背部，宏观组合合理；另有灰背层在蓝板前方，背顶与座前缘略卷曲。',
  '主体与蓝组件位置保持，背顶和座前缘出现线状窄片；侧截面蓝灰背层仍有间隔。前截面未切到C，不能据此评价接口。',
  '宏观组合保持，POST更平但边缘更细。'),
 ((C,C),V,'YES','YES',(U,'YES'),
  '蓝腿架与灰座面相配，腿边有灰三角片；蓝背板前仍有灰表层。整体完整，C用途位置合理，但灰蓝多层关系存在。',
  '灰座与背面较平，背顶及前缘保留细线状片，蓝腿旁灰片仍可见；图像不认证这些片形成合法连接。',
  '组件采用位置保持，POST仍有边缘窄片。')),
('3db1d3904f058b574a9ca2c84bced0dd__COMBO_PART_ASSEMBLY__n1','背板加两端腿架组合',
 ((C,C),V,'YES','YES',(U,'YES'),
  '灰座面位于两组蓝腿上，蓝背板与灰背层并行；顶缘卷曲、座前沿有细长灰边。完整长凳宏观结构清楚。',
  '主要部件保留，顶缘变成几条细线，座前沿长开口更明显；灰背仍未与蓝板在视觉上合为单层。',
  '整体结构保持，背层与薄边持续。'),
 ((C,C),V,'YES','YES',(U,'YES'),
  '灰面组织成座背主体，蓝腿外侧及内侧有灰三角片，C部件位置合理；背层之间仍有可见差异。',
  '灰主体更平，蓝腿周围灰片仍在，背顶和座前沿细线/小缺口保留；不能以这些连接外观推断共享索引。',
  '宏观采用保持，POST未消除背层和窄片。')),
('459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n0','座板与弯曲端支撑组合',
 ((P,P),U,'PARTIAL','YES',('YES','YES'),
  '蓝宽座板上方另有带大开口的灰长板，远端灰支撑短于蓝弯曲支撑。前侧截面均显示蓝灰两层，灰面局部还靠近蓝板内部。',
  '上方灰长板更平却仍为窄条和破口，层差持续，灰端支撑变薄；无法从投影确认细片是否独立。',
  '平整收益伴随窄条，座面错层持续。'),
 ((P,P),U,'PARTIAL','YES',('YES','YES'),
  '灰长板与三角条片仍主要位于蓝座板上方，蓝板附近多出灰线/面；另一端灰支撑仍短。主体表面不连续的外观明显。',
  '上层灰板变成更平的条片，蓝板附近灰层保留，两种截面仍显示额外水平层。没有清楚的整体融入改善。',
  'POST保留双层和条片，不能只按C附近灰面更多认定改善。')),
('459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n1','座板与弯曲端支撑组合',
 ((P,P),U,'PARTIAL','YES',('YES','YES'),
  '蓝组合上方另有带长裂口和三角开口的灰薄板；远端灰支撑下垂。前侧截面的灰水平层高于蓝座板。',
  '灰薄板进一步条带化，蓝灰高度差不变；局部灰端面也变薄。',
  'POST条带化更显著，双层关系保持。'),
 ((P,P),U,'PARTIAL','YES',('YES','YES'),
  '灰上层板仍悬在蓝座板上方，部分灰面贴近蓝板端部和内部；有多处长开口，未见完整协调表面。',
  '上层灰板更窄、更平，远端灰支撑仍比蓝端短；多层座板关系并未被整理消除。',
  '表面变平但开口和层差持续。')),
('459f90aa2162b1f1d46c340938e2ff1c__SEAT_SURFACE_1__n0','完整矩形座面',
 ((C,C),X,'PARTIAL','YES',(Q,Q),
  '灰体形成开放靠背、座框及端支撑；蓝板位于座面区但高于灰座面。前侧截面均显示独立蓝层与下方灰层。',
  '灰表面较平，主体完整；蓝板依然高于灰座面，没有可见的连续侧面把其自然接入主体。',
  '主体保持，蓝灰座面高度差持续。'),
 ((C,C),U,'YES','YES',(U,U),
  '蓝板两侧有灰斜面/短边从其边缘延伸，外观比单独悬置更接近一块上层座面；但下方仍存在另一灰座层。',
  '两侧短灰边保留，蓝板仍高于主要灰座层；局部靠近不认证整个座面已接入，双层含义仍不确定。',
  'C边缘灰片保留，但主座层仍在下方。')),
('459f90aa2162b1f1d46c340938e2ff1c__SEAT_SURFACE_1__n1','完整矩形座面',
 ((C,P),U,'YES','YES',(U,'YES'),
  '蓝板位于有背长凳的座面区，灰座层在其下方，端部灰面靠近蓝板；两层是否构成合理厚度或错层，图像仍不确定。',
  '蓝灰座层保持，灰腿和边框明显变细，局部成为近线状结构；没有明确整体采用变化。',
  '座层关系保持，POST腿与框架削薄。'),
 ((C,P),U,'YES','YES',(U,'YES'),
  '蓝板处于座面高度，端部和边缘有灰片相邻；侧截面仍显示下方独立灰层，局部采用有迹象而整体关系不明。',
  '蓝板附近灰短边继续存在，灰支撑变为细线，双层座面未明显合一。',
  '局部灰边保留，POST仍有支撑削薄代价。')),
('5dda4ef4ab9add9411ee8243f54292d6__LEG_SUPPORT_1__n0','单腿竖直侧面',
 ((C,C),U,'PARTIAL','YES',(Q,U),
  '灰体是上下两层平台和腿组成的完整架体；蓝竖片位于端部外侧，但高度范围与旁边灰立柱不一致，未见灰腿围绕它完整展开。',
  '两层架体保持，部分自由腿更薄；蓝片仍位于灰立柱旁，截面中的平行竖线关系未消失。',
  '宏观架体保持，蓝腿面与灰立柱关系仍不明确。'),
 ((C,C),U,'PARTIAL','YES',(Q,U),
  '灰双层架体与蓝端部竖片并存，蓝片附近有窄灰面；主要灰立柱仍占据自己的位置，C用途只能部分辨认。',
  '灰面略平，蓝片附近仍有灰窄条；主体保持，但腿柱更薄，未见明确采用跃迁。',
  '局部邻近保持，未消除立柱与C的范围差异。')),
('5dda4ef4ab9add9411ee8243f54292d6__LEG_SUPPORT_1__n1','单腿竖直侧面',
 ((C,P),U,'PARTIAL','YES',(U,'YES'),
  '灰座板和两个板状端支撑构成简单长凳；蓝长竖条在外侧角部，上下均超出灰结构对应范围，支撑用途并不完全匹配。',
  '座板变为长窄条和开口，蓝竖条位置保持；侧截面短段增多，不能据此判拓扑独立。',
  'POST座板变薄，蓝竖片与支撑的差异保持。'),
 ((C,P),U,'PARTIAL','YES',(U,'YES'),
  '灰主体仍是低座板与短端支撑，蓝竖片更高更长；蓝片附近有灰边但未见整体腿形随其适配。',
  '座板条带化，前截面只余C并不能证明整个灰体断连；表面视图中灰座板仍在，但与长蓝片的用途关系不清。',
  '条片化与输入腿面范围失配仍在。')),
('5dda4ef4ab9add9411ee8243f54292d6__SEAT_SURFACE_1__n0','单条完整座面板条',
 ((C,P),X,'NO','YES',(U,'YES'),
  '灰体呈低箱式长凳和圆端支撑，蓝窄条悬在灰顶面上方；前侧截面显示清楚高度间隔，未见其成为灰顶面部分。',
  '灰主体上表面保持平，圆端支撑变成薄扇片外观；蓝条仍高于灰顶面。',
  '悬置关系保持，POST支撑薄片化。'),
 ((C,P),V,'PARTIAL','YES',(U,'YES'),
  '蓝条两侧有灰斜面向下延伸，形成屋脊状上盖；其下仍有一层灰平顶。C在上盖中的几何用途较可理解，但不等于来源座面角色或合法连接。',
  '屋脊状上盖与蓝条保留，下方灰平层仍在；圆端支撑被削成扇状薄片，整体质量有可见代价。',
  'C参与上盖的外观保持，POST支撑变薄。')),
('5dda4ef4ab9add9411ee8243f54292d6__SEAT_SURFACE_1__n1','单条完整座面板条',
 ((C,P),U,'NO','YES',(Q,'YES'),
  '灰体为带侧架、栏杆和座板的长凳，蓝条在座板上方靠近栏杆区；可能作为横向构件，未显示原座面用途。',
  '灰栏杆和腿成为明显细线，蓝条仍高于灰座板；主体表面完整性下降而用途歧义保持。',
  '蓝条位置保持，POST框架线状化。'),
 ((C,P),U,'PARTIAL','YES',(U,'YES'),
  '蓝条附近有较大的斜灰三角面向灰座板方向延伸，侧截面出现楔形关系；下面原灰座层仍在，是否自然融入仍不确定。',
  '斜灰片与蓝条保留，侧架和腿削成细线；有局部延伸不等于主体接续已获认证。',
  '楔形局部关系保持，POST仍削薄框架。')),
('9d56087c9f98bf90717cca22daef45e7__LEG_SUPPORT_1__n0','从基底到背顶的后墙表面',
 ((C,P),X,'PARTIAL','YES',(U,'YES'),
  '蓝大墙面在另一张灰窄背长凳后方，灰体自己有背、座和支脚；侧截面显示明显前后距离，未见自然采用。',
  '蓝墙与灰凳分层保持，灰支脚近线状；没有可见的主体融入改善。',
  '分层不变，POST支脚削薄。'),
 ((C,P),U,'YES','YES',('YES','YES'),
  '蓝墙外周出现大片灰三角边框和下沿灰片，原灰窄背长凳仍在其前方；C不再完全孤立外观，但宽片与主体关系不明确。',
  '周边灰片保留，底部及中间缺口附近有较大的斜灰片，灰支脚变细；两套背部/墙面位置仍并存。',
  'C周边灰片保持，尚不能说自然融合且有薄片代价。')),
('9d56087c9f98bf90717cca22daef45e7__LEG_SUPPORT_1__n1','从基底到背顶的后墙表面',
 ((C,P),X,'PARTIAL','YES',(U,'YES'),
  '灰窄背凳在蓝墙前方，自带的灰背与蓝墙位置不同；蓝墙更高更低，当前视图缺少主体使用它的证据。',
  '灰腿显著变细，灰背与蓝墙仍前后分开；宏观关系不变。',
  '分层持续，POST腿部更薄。'),
 ((C,P),U,'YES','YES',('YES','YES'),
  '灰三角边框沿蓝墙上沿、侧沿及底部展开，前方灰凳主体仍保留；周边相接的外观不能确定这些宽片是否接入主要连通体。',
  '蓝墙周边灰片保留，中间下方形成斜片，灰凳支脚细化；仍有前后双层关系。',
  '周边灰片与分层同时保留，POST支撑变薄。')),
('9d56087c9f98bf90717cca22daef45e7__SEAT_SURFACE_1__n0','高背块顶部水平盖条',
 ((C,P),X,'NO','YES',(U,'YES'),
  '蓝盖条在灰窄背上方，侧截面显示与灰背顶有明显间隔；灰凳自身有座、背和支脚。',
  '盖条悬置关系保持，灰腿变细，灰背截面更碎；不能由短线推断拓扑断片。',
  '盖条未自然融入，POST腿部削薄。'),
 ((C,P),V,'YES','YES',(U,'YES'),
  '蓝条下方出现灰侧面，形成较容易理解的背顶盖/上部块体；侧截面从C两端伸出灰边，但与更低的原灰背仍有层差。',
  '顶部盖形关系保留，灰支脚变细；从C延出的灰片仍不能证明其与下方主体合法共边。',
  '顶部盖用途较清楚的外观保持，仍有层差和支撑削薄。')),
('9d56087c9f98bf90717cca22daef45e7__SEAT_SURFACE_1__n1','高背块顶部水平盖条',
 ((C,C),V,'PARTIAL',U,(Q,Q),
  '蓝横条跨在两个外倾侧架上端，灰座板位于下方，整体像带上横梁的长凳；蓝条作为上部横向构件的用途可理解，不要求与来源高背块同款。',
  '上横条、外倾侧架和灰座面关系基本保持，表面略平整；未见大量细小悬片。真实连接仍未知。',
  '可理解的上横梁用途保持，宏观形状稳定。'),
 ((C,C),U,'NO','YES',(Q,Q),
  '灰体变为上下两层平台与立柱的架体，蓝条主要隐藏在上层灰顶板下方；前侧截面显示蓝条低于灰顶面，具体用途不明。',
  '双层架体保持，蓝条仍藏在上层下方，局部灰边变平；不能用不同款式本身判失败，但C采用并不清晰。',
  '主体保持，蓝条在上层下面的隐藏关系仍不明确。')),
]
assert len(notes)==len(pages)==16
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
rows=[];pairs=[]
for page,entry in zip(pages,notes):
    rid,part,*arms=entry
    assert page['request_id']==rid
    labels=[c['label'] for c in page['cells']]
    assert labels==['NATIVE_S15000_RAW','NATIVE_S15000_POST','SEAM_S15000_RAW','SEAM_S15000_POST']
    path=GALLERY/(page['page_id']+'.png');ir=read(path.with_suffix('.json'))
    assert sha(path)==ir['image_sha256'] and ir['definition_sha256']==sha(GALLERY/'definition.json')
    for arm,obs in zip(('NATIVE_S15000','SEAM_S15000'),arms):
        body,use,role,layer,thin,raw,post,change=obs
        for j,stage in enumerate(('RAW','POST')):
            rows.append(dict(review_version=schema['review_version'],reviewer='AI_PROXY',actual_image_viewed=True,human_reviewed=False,nonblind=True,
                reviewed_at_utc=now,label=arm,global_step=15000,request_id=rid,condition_id=page['condition_id'],parent_id=page['parent_id'],
                split=page['split'],stage=stage,K=page['K'],N=page['N'],part_description=part,image_path=str(path),image_sha256=sha(path),
                image_receipt_sha256=sha(path.with_suffix('.json')),gallery_definition_sha256=sha(GALLERY/'definition.json'),
                views_used=['surface_-60_23','surface_120_23','finite_front_section_at_C_bbox_center','finite_side_section_at_C_bbox_center'],
                reviewed_cell_label=arm+'_'+stage,other_model_cells_visible=True,body_recognizable_as_bench=True,
                body_completeness=body[j],C_visual_coherent_use=use,construction_intended_role_visible=role,
                visible_offset_or_redundant_layer=layer,thin_or_dangling_sheet_appearance=thin[j],
                exact_shared_connectivity='NOT_ASSESSED_FROM_IMAGES',topologically_disconnected_fragment='UNKNOWN_IMAGE_ONLY',
                notes=(raw,post)[j],metric_values_used_for_labels=False,source_full_mesh_shown=False,
                scope='Fixed15000 full-panel subset; image-only observation without checkpoint/data/seed selection or training-policy changes'))
        pairs.append(dict(label=arm,request_id=rid,split=page['split'],part_description=part,RAW_to_POST=change,
            image_sha256=sha(path),reviewer='AI_PROXY',actual_image_viewed=True))
assert len(rows)==64 and len(pairs)==32
def write(name,text):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(text)
write('review.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
write('raw_post_comparison.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs))
write('review_schema.json',json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
md=['# 15000双臂完整面板：中间16请求图审（AI_PROXY）','',
 '按request_id字典排序查看index16..31，每张联合图实际查看一次，共16页。逐页查看Native/SEAM各自RAW/POST、两个表面视角及固定前/侧截面，保存64条阶段观察与32条前后比较。非盲AI_PROXY，不使用数值metric或唯一来源完整网格决定标签。','',
 'SEAM在部分输入边缘形成了可见的灰侧面、斜片或包边，例如5dda板条n0的屋脊状上盖和9d盖条n0的背顶盖外观；这属于局部外观证据，不认证接入主体或合法拓扑。459f多层座板、3db双背层等问题仍在。9d盖条n1两臂生成款式不同，不能按来源款式排名；本页Native上横梁用途较清楚，SEAM蓝条则隐藏在上层板下。','',
 '两臂多例POST都有平整与削薄并存：座板长开口、细腿、薄条和宽灰片需要保留为代价。截面短线、空缺或投影相交均不直接等于拓扑碎片、孔洞或非法相交。所有实际共享连接与拓扑断片分别NOT_ASSESSED/UNKNOWN。','',
 '|请求|模型|RAW → POST观察|','|---|---|---|']
for p in pairs:md.append('|'+p['request_id'][:6]+'/'+p['request_id'].split('__')[1]+'/'+p['request_id'].split('__')[-1]+'|'+p['label']+'|'+p['RAW_to_POST']+'|')
md+=['','这只是固定完整面板的中间16请求，不代表整个48请求，更不据此选择checkpoint、暂停训练或宣布收敛。','']
write('REVIEW.md','\n'.join(md))
write('receipt.json',json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',reviewer='AI_PROXY',reviewed_at_utc=now,
    images_viewed=16,image_view_calls_per_page=1,sorted_request_indices=[16,31],models=['NATIVE_S15000','SEAM_S15000'],
    stage_rows=64,paired_rows=32,model_calls=0,new_samples=0,rescoring=0,data_audits=0,
    protected_gallery_definition_sha256=sha(GALLERY/'definition.json'),protected_manifest_sha256=sha(manifest_path),
    request_ids=[p['request_id'] for p in pages],files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',images=16,stage_rows=64,paired_rows=32,model_calls=0)))
