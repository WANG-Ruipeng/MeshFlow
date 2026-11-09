"""Save actual image-only observations after viewing sorted pages16..31 once each."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
LABEL = 'NATIVE_S12000'
GALLERY = ROOT / 'gallery/NATIVE_S12000_full'
PRIOR = ROOT / 'visual_review/NATIVE_S10000_process'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest_path = ROOT / 'gallery_manifests/NATIVE_S12000_full.json'
manifest = read(manifest_path)
completion = read(GALLERY / 'completion.json')
assert len(completion['images']) == 48
assert completion['definition_sha256'] == sha(GALLERY / 'definition.json')
pages = sorted(manifest['pages'], key=lambda x: x['request_id'])[16:32]
assert len(pages) == 16
schema = read(PRIOR / 'review_schema.json')
schema.update(checkpoint=LABEL, actual_view_count=16,
    scope='Part02 of the fixed48-request full panel: request_id sorted indices16..31;32 RAW/POST observations',
    prior_schema_sha256=sha(PRIOR / 'review_schema.json'))
schema['limitations'][-1] = 'This is the common Native12000 endpoint, not a SEAM comparison; these16 pages alone are not the whole48-request panel or a convergence claim.'

# Per request: body RAW/POST, C-use RAW/POST, intended-role, offset/layer,
# thin-sheet RAW/POST, then independent RAW and POST notes and their comparison.
notes = [
('3db1d3904f058b574a9ca2c84bced0dd__BACK_SURFACE_1__n0', '完整靠背表面',
 ('COMPLETE_MACRO','PARTIAL'), ('PLAUSIBLE','PLAUSIBLE'), 'YES','UNCERTAIN', ('UNCERTAIN','YES'),
 '蓝色竖板位于灰座面后沿，灰座板和两端支撑形成可辨认的长凳；侧截面中座板向蓝板附近延伸。用途位置合理，但图像不能证明共享连接。',
 '蓝背板与灰座板的总体位置保留，但座板出现更细长的条带和长开口，侧截面由较完整轮廓变为短段；宏观表面完整性较弱，不能由短段直接断言拓扑断片。',
 '靠背用途保持；POST使座面变薄且开口更明显。'),
('3db1d3904f058b574a9ca2c84bced0dd__BACK_SURFACE_1__n1', '完整靠背表面',
 ('COMPLETE_MACRO','PARTIAL'), ('NO_VISIBLE_COHERENT_USE','NO_VISIBLE_COHERENT_USE'), 'PARTIAL','YES', ('UNCERTAIN','YES'),
 '另有一整块灰色高背板位于蓝背板前方，蓝板只在背侧露出；侧截面显示前后分层。灰座板、支撑和高背构成主体，但未见蓝板成为连续灰背的一部分。',
 '双背层关系保持；灰座面成为更窄的条片，腿部出现明显三角开口与薄片外观。蓝板仍位于另一块完整灰背板后方。',
 'POST未消除双背层，并削薄座面与腿部表面。'),
('3db1d3904f058b574a9ca2c84bced0dd__COMBO_PART_ASSEMBLY__n0', '背板加两端腿架组合',
 ('COMPLETE_MACRO','COMPLETE_MACRO'), ('PLAUSIBLE','PLAUSIBLE'), 'YES','YES', ('UNCERTAIN','YES'),
 '蓝背板与蓝腿架都处于合理部位，灰座面将两端组织成完整凳体；侧截面可见灰背面与蓝背板并行但有间距，灰座前缘略卷曲。',
 '主要凳体和蓝腿架位置保留，灰背面仍与蓝板分层；背顶和座前沿出现线状细条及开口。前截面没有切到C，不能据此评价该处接续。',
 '宏观组合保持，POST较平但边缘细条更明显。'),
('3db1d3904f058b574a9ca2c84bced0dd__COMBO_PART_ASSEMBLY__n1', '背板加两端腿架组合',
 ('COMPLETE_MACRO','COMPLETE_MACRO'), ('PLAUSIBLE','PLAUSIBLE'), 'YES','YES', ('UNCERTAIN','YES'),
 '灰座面位于蓝腿架上方，蓝背板承担后侧区域；另一层灰背面在其前方，顶边有卷起的细长结构。整体仍可辨认为完整有背长凳。',
 '灰背与座板更平，蓝色组件位置保持；顶缘和座前沿残留细线状片，侧截面背面与蓝板的间距仍在。图像不认证合法连接。',
 '整体位置保持，分层未消失，边缘出现较细条片。'),
('459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n0', '座板与弯曲端支撑组合',
 ('PARTIAL','PARTIAL'), ('UNCERTAIN','UNCERTAIN'), 'PARTIAL','YES', ('YES','YES'),
 '蓝色宽座板及弯曲端支撑上方另有高一层的灰长板；灰板存在三角开口，远端灰支撑较短。前侧截面都显示灰长板高于蓝座板，另有灰线贴近蓝板内部。',
 '上方灰板变得更平但更加条片化，灰板与蓝座板之间的高度差未消除；远端支撑仍短，局部表面连续性较弱。',
 'POST平整了部分灰板，未解决额外座层和开口。'),
('459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n1', '座板与弯曲端支撑组合',
 ('PARTIAL','PARTIAL'), ('UNCERTAIN','UNCERTAIN'), 'PARTIAL','YES', ('YES','YES'),
 '蓝组合提供宽座板与一端支撑，灰几何在其上方另成一张带破口的长薄面，另一端灰支撑下垂；两组截面显示平行错层。',
 '上方灰面更细碎成条带，部分侧边消失；平行错层保持，灰端支撑也更薄。无法仅靠投影判定哪些条带拓扑上独立。',
 'POST条带和薄支撑更明显，蓝灰双层关系未改善。'),
('459f90aa2162b1f1d46c340938e2ff1c__SEAT_SURFACE_1__n0', '完整矩形座面',
 ('COMPLETE_MACRO','COMPLETE_MACRO'), ('NO_VISIBLE_COHERENT_USE','NO_VISIBLE_COHERENT_USE'), 'NO','YES', ('NO_OBVIOUS_IN_VIEWS','NO_OBVIOUS_IN_VIEWS'),
 '灰色高背、厚座和四腿主体完整，蓝板却悬在灰座面上方。前、侧截面均有清楚的高度间隔，未见灰主体围绕蓝板形成座面。',
 '灰主体保持完整且略平整；蓝板仍处于灰座面上方中空区域，明显高度错层没有消失。',
 '主体完整但蓝座板悬置的关系基本不变。'),
('459f90aa2162b1f1d46c340938e2ff1c__SEAT_SURFACE_1__n1', '完整矩形座面',
 ('COMPLETE_MACRO','COMPLETE_MACRO'), ('UNCERTAIN','UNCERTAIN'), 'YES','YES', ('NO_OBVIOUS_IN_VIEWS','NO_OBVIOUS_IN_VIEWS'),
 '灰体呈高背箱式长凳，蓝板位于灰底座顶面附近并向前伸出；截面仍显示蓝板下方有独立灰水平层，是否自然接为同一座面不确定。',
 '箱体表面变平，蓝板位置与其下灰层仍分开；未见大量小悬片，但也不能由接近的位置认证采用和连接。',
 '蓝板比同条件n0更接近座面位置；RAW到POST仍保留蓝灰层差。'),
('5dda4ef4ab9add9411ee8243f54292d6__LEG_SUPPORT_1__n0', '单腿竖直侧面',
 ('COMPLETE_MACRO','COMPLETE_MACRO'), ('UNCERTAIN','UNCERTAIN'), 'PARTIAL','YES', ('NO_OBVIOUS_IN_VIEWS','NO_OBVIOUS_IN_VIEWS'),
 '灰体形成厚座板与四个短腿，蓝竖片位于外侧端角，顶部高于灰座板、底部低于灰腿。作为端腿的位置部分合理，但灰腿并未明显围绕蓝片展开。',
 '灰座面略平，主体保留；蓝竖片仍比灰腿更长并处在外侧，前截面的小间距仍可见。',
 '宏观凳体保留；输入腿面与灰腿的尺度/位置关系未明显改变。'),
('5dda4ef4ab9add9411ee8243f54292d6__LEG_SUPPORT_1__n1', '单腿竖直侧面',
 ('COMPLETE_MACRO','PARTIAL'), ('UNCERTAIN','UNCERTAIN'), 'PARTIAL','YES', ('UNCERTAIN','YES'),
 '灰座板与两块端支撑构成简单长凳，蓝竖条在一端伸到座面上方和腿底下方；其范围与灰支撑不一致。',
 '灰座板变成带长开口的薄条，侧截面出现许多短线；蓝竖条位置不变，仍未看到生成支撑与其协调为一体。',
 'POST座板表面变薄、开口增加，蓝竖片关系没有明显改善。'),
('5dda4ef4ab9add9411ee8243f54292d6__SEAT_SURFACE_1__n0', '单条完整座面板条',
 ('COMPLETE_MACRO','COMPLETE_MACRO'), ('UNCERTAIN','UNCERTAIN'), 'NO','YES', ('NO_OBVIOUS_IN_VIEWS','NO_OBVIOUS_IN_VIEWS'),
 '灰体是有两条背板、座板和侧架的完整长凳；蓝水平板条处在靠背下沿而不是灰座面高度，并从两端伸出。可能被解释成横向构件，但当前视图未给出连续采用证据。',
 '灰主体几乎保持原状，蓝板仍位于灰座面上方靠背区；固定截面只切到C，不能以缺少灰交线推出整个模型断开。',
 'RAW到POST变化小，板条的构造座面用途仍未显现。'),
('5dda4ef4ab9add9411ee8243f54292d6__SEAT_SURFACE_1__n1', '单条完整座面板条',
 ('COMPLETE_MACRO','COMPLETE_MACRO'), ('UNCERTAIN','UNCERTAIN'), 'PARTIAL','YES', ('UNCERTAIN','UNCERTAIN'),
 '灰体形成上下双层的平台/凳架，蓝板大部分遮在上层灰面附近；前侧截面显示蓝板略低于上层灰面，另有下方灰水平层，实际用途仍有歧义。',
 '两层灰结构与蓝板的相对高度保持，细腿和侧边更规整但仍较薄。未见蓝条明显成为上层连续表面，隐藏关系无法从两视图确定。',
 '双层架体与蓝板高度关系保持，未见明确采用变化。'),
('9d56087c9f98bf90717cca22daef45e7__LEG_SUPPORT_1__n0', '从基底到背顶的后墙表面',
 ('COMPLETE_MACRO','PARTIAL'), ('NO_VISIBLE_COHERENT_USE','NO_VISIBLE_COHERENT_USE'), 'PARTIAL','YES', ('UNCERTAIN','YES'),
 '蓝色大墙面在一张独立外观的灰长凳后方，灰凳自己已有窄背板、座板和两支脚；侧截面有明显前后距离，未见灰体借蓝墙完成自身结构。',
 '灰背板和座板仍在蓝墙前方，支脚变得近线状；墙与灰凳的分层关系保持。支脚是否真实断开不能由图像确定。',
 '蓝墙分层保持，POST将自由支脚显著削薄。'),
('9d56087c9f98bf90717cca22daef45e7__LEG_SUPPORT_1__n1', '从基底到背顶的后墙表面',
 ('COMPLETE_MACRO','PARTIAL'), ('NO_VISIBLE_COHERENT_USE','NO_VISIBLE_COHERENT_USE'), 'PARTIAL','YES', ('UNCERTAIN','YES'),
 '灰色窄背长凳出现在蓝墙前方，灰背自身占据另一个平行位置；蓝墙更高且延伸至更低位置，当前视图中缺少自然融入迹象。',
 '灰凳仍与蓝墙前后分层，灰腿缩成很薄的条线，灰座和靠背保持；没有可见的C采用改善。',
 'POST保留分层并削薄腿部；不能将细线直接记作拓扑碎片。'),
('9d56087c9f98bf90717cca22daef45e7__SEAT_SURFACE_1__n0', '高背块顶部水平盖条',
 ('COMPLETE_MACRO','PARTIAL'), ('NO_VISIBLE_COHERENT_USE','NO_VISIBLE_COHERENT_USE'), 'NO','YES', ('UNCERTAIN','YES'),
 '蓝条位于灰长凳靠背上方，侧截面显示蓝条与倾斜灰背顶之间仍有明显间隔；灰凳自带座板、窄背和支脚，未见蓝条作为背顶盖的一部分。',
 '蓝条悬在背顶上方的关系不变；灰腿明显变细，背部截面也更零碎。不能把截面短线当作独立三维面片证据。',
 '输入盖条仍未可见地融入，POST支脚更薄。'),
('9d56087c9f98bf90717cca22daef45e7__SEAT_SURFACE_1__n1', '高背块顶部水平盖条',
 ('COMPLETE_MACRO','COMPLETE_MACRO'), ('NO_VISIBLE_COHERENT_USE','NO_VISIBLE_COHERENT_USE'), 'NO','YES', ('NO_OBVIOUS_IN_VIEWS','UNCERTAIN'),
 '灰体为高背、座面和弯曲底座的块状长凳；蓝水平条横在灰背前方中上部，灰背顶更高，侧截面两者有明显前后间隔，不像顶部盖面。',
 '灰宏观形状保持，右侧扶持面边缘出现细齿/开口外观；蓝条仍在灰背前方，不见恢复为背顶盖或其他明确连续部分。',
 '主体保留，蓝条前后错位持续，部分侧缘变薄。'),
]
assert len(notes) == len(pages)
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
rows, pairs = [], []
for page, entry in zip(pages, notes):
    rid, part, body, use, role, layer, thin, raw, post, change = entry
    assert rid == page['request_id']
    path = GALLERY / (page['page_id'] + '.png')
    ir = read(path.with_suffix('.json'))
    assert sha(path) == ir['image_sha256']
    assert ir['definition_sha256'] == sha(GALLERY / 'definition.json')
    for j, stage in enumerate(('RAW', 'POST')):
        rows.append(dict(review_version=schema['review_version'], reviewer='AI_PROXY', actual_image_viewed=True,
            human_reviewed=False, nonblind=True, reviewed_at_utc=now, label=LABEL, global_step=12000,
            request_id=rid, condition_id=page['condition_id'], parent_id=page['parent_id'], split=page['split'],
            stage=stage, K=page['K'], N=page['N'], part_description=part, image_path=str(path), image_sha256=sha(path),
            image_receipt_sha256=sha(path.with_suffix('.json')), gallery_definition_sha256=sha(GALLERY / 'definition.json'),
            views_used=['surface_-60_23','surface_120_23','finite_front_section_at_C_bbox_center','finite_side_section_at_C_bbox_center'],
            body_recognizable_as_bench=True, body_completeness=body[j], C_visual_coherent_use=use[j],
            construction_intended_role_visible=role, visible_offset_or_redundant_layer=layer,
            thin_or_dangling_sheet_appearance=thin[j], exact_shared_connectivity='NOT_ASSESSED_FROM_IMAGES',
            topologically_disconnected_fragment='UNKNOWN_IMAGE_ONLY', notes=(raw,post)[j], metric_values_used_for_labels=False,
            source_full_mesh_shown=False, scope='Fixed common12000 full-panel image observation; no checkpoint/data/seed selection or training-policy changes'))
    pairs.append(dict(request_id=rid, split=page['split'], part_description=part, RAW_to_POST=change,
                      image_sha256=sha(path), reviewer='AI_PROXY', actual_image_viewed=True))

def write(name, text):
    with (OUT / name).open('x', encoding='utf-8') as f:
        f.write(text)

write('review.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
write('raw_post_comparison.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in pairs))
write('review_schema.json', json.dumps(schema, ensure_ascii=False, indent=2) + '\n')
md = ['# NATIVE_S12000 中间16请求图审（AI_PROXY）', '',
 '按request_id字典排序审阅index16..31，共16张图页、32条RAW/POST记录。每页实际查看一次，两种表面视角与固定前/侧有限截面均查看。没有使用数值metric生成标签，没有读取来源完整网格。', '',
 '3db1组合输入在宏观位置上较合理，但背面仍有蓝灰分层；459f座面n0明显位于另一个灰座面上方，n1更接近座面位置而仍有层差。5dda的板条可被解释成横构件，不能仅凭其来源座面标签判错。9d的整墙和盖条多与灰凳分开呈现。', '',
 'POST有时变平，同时将座面、支脚变成更薄的条状外观。有限截面出现短线或空缺，不能单独证明拓扑碎片、三维孔洞或非法穿插。所有共享连接和拓扑断片继续NOT_ASSESSED/UNKNOWN。', '',
 '|请求|部件说明|RAW → POST观察|', '|---|---|---|']
for p in pairs:
    md.append('|'+p['request_id'][:6]+'/'+p['request_id'].split('__')[1]+'/'+p['request_id'].split('__')[-1]+'|'+p['part_description']+'|'+p['RAW_to_POST']+'|')
md += ['', '本分片不是完整48请求汇总，也不与尚未完成的SEAM终点作胜负判断；不据图更改训练协议。', '']
write('REVIEW.md', '\n'.join(md))
write('receipt.json', json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE', reviewer='AI_PROXY',
    reviewed_at_utc=now, images_viewed=16, image_view_calls_per_page=1, sorted_request_indices=[16,31],
    stage_rows=32, paired_rows=16, model_calls=0, new_samples=0, rescoring=0, data_audits=0,
    protected_gallery_definition_sha256=sha(GALLERY/'definition.json'), protected_manifest_sha256=sha(manifest_path),
    request_ids=[p['request_id'] for p in pages], files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}), ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE', images=16, stage_rows=32, model_calls=0)))
