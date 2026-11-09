"""Persist actual nonblind Native-only viewing of eight immutable 17000 pages."""
import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
LABEL = 'NATIVE_S17000'
KEY = 'NATIVE_S17000__SEAM_S17000_process'
GALLERY = ROOT / 'gallery' / KEY
PRIOR = ROOT / 'visual_review/NATIVE_S13000_process'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

manifest_path = ROOT / 'gallery_manifests' / (KEY + '.json')
manifest = read(manifest_path)
completion = read(GALLERY / 'completion.json')
assert len(manifest['pages']) == len(completion['images']) == 8
assert completion['definition_sha256'] == sha(GALLERY / 'definition.json')
schema = read(PRIOR / 'review_schema.json')
schema.update(checkpoint=LABEL, actual_view_count=8,
              prior_schema_sha256=sha(PRIOR / 'review_schema.json'),
              scope='Eight fixed process requests; Native cells only in the joint Native/SEAM gallery; 16 RAW/POST observations')
schema['limitations'] = [x.replace('at13000', 'at17000') for x in schema['limitations']]

# The following notes were written after eight actual original-resolution image views.
# Images also display SEAM; those columns are not scored in this Native-only record.
notes = [
    ('2d1e5eeb255e79dcab53bf7fa22f3636__BACK_SURFACE_1__n0', '靠背板',
     ('COMPLETE_MACRO', 'PARTIAL'), 'UNCERTAIN', 'YES', 'YES',
     ('UNCERTAIN', 'YES'),
     '灰体有靠背、座板及四腿，蓝板处于背部区域；灰背板高于蓝板并局部遮挡它。侧截面显示蓝板两侧另有灰轮廓，并未显示单一连续背面；端部支撑外观仍较完整。',
     '主体款式保持，背面蓝灰多层关系持续；两端自由腿明显削成近线状细片，背板上缘也更薄。截面若干灰线消失不足以证明三维孔洞或断片。',
     '背部多层未消失，POST明显削薄自由腿。'),
    ('459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n0', '座板加弯曲端支撑',
     ('PARTIAL', 'PARTIAL'), 'UNCERTAIN', 'PARTIAL', 'YES',
     ('YES', 'YES'),
     '蓝座板和弯曲端支撑之上有多条不规则灰三角长片，长片间有明显开口；另一端有短灰支撑。前、侧截面均显示上层灰片高于蓝板，同时少量灰轮廓处在蓝板厚度范围内。',
     '灰长片变平但仍跨在蓝板上方，长开口与错层保持；远端灰支撑进一步变薄。不能根据薄片外观断定其拓扑独立或非法穿插。',
     '灰片略平，蓝灰错层和不规则开口仍明显。'),
    ('459f90aa2162b1f1d46c340938e2ff1c__SEAT_SURFACE_1__n0', '完整矩形座面',
     ('COMPLETE_MACRO', 'COMPLETE_MACRO'), 'NO_VISIBLE_COHERENT_USE', 'PARTIAL', 'YES',
     ('NO_OBVIOUS_IN_VIEWS', 'NO_OBVIOUS_IN_VIEWS'),
     '灰体是具有开放靠背、灰座面及楔形两端支撑的完整长凳；蓝座板明显高于灰座面。前截面有两条分离的蓝灰水平层，侧截面也没有显示灰侧壁上接到蓝板。',
     '灰凳体更平整但蓝板仍高于下层灰座面，位置差未消失；楔形支撑仍有可见面积。此图未显示蓝板成为灰体连续座面的证据。',
     '宏观凳体保持，独立蓝座板与灰座面高度差持续。'),
    ('9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n0', 'U形端支撑',
     ('COMPLETE_MACRO', 'PARTIAL'), 'PLAUSIBLE', 'YES', 'UNCERTAIN',
     ('YES', 'YES'),
     '灰座板从蓝U形端架向另一端延伸，端架用途在位置上合理；另一端两条灰腿明显较细。固定截面显示灰座板在蓝横梁下方附近，仍不能由投影确定连接关系。',
     '灰座板更平，另一端自由腿被削成几乎线状；蓝U形端架保持。截面中的杂短线减少，不能据此认定接续改善。',
     '端架用途合理，POST进一步削薄远端自由腿。'),
    ('122a480cfcdd742650c626aa72455dae__LEG_SUPPORT_1__n0', '单条实心支撑腿',
     ('COMPLETE_MACRO', 'COMPLETE_MACRO'), 'PLAUSIBLE', 'YES', 'YES',
     ('UNCERTAIN', 'UNCERTAIN'),
     '灰体形成有多条靠背板、座板、腿和横撑的完整长凳；蓝腿位于支撑区域且上端靠近座板下方，位置合理。其旁边仍有相邻灰支撑，前截面可见两者横向错位，不能确认是否为冗余支撑。',
     '整体板条凳和蓝腿位置保持，板条更平且间隙更规整；蓝腿与旁边灰支撑的并置关系仍在。没有明显新漂浮片的图像证据，但共享连接仍未知。',
     '宏观完整性保持，支撑区域用途合理但相邻蓝灰构件关系仍有歧义。'),
    ('122a480cfcdd742650c626aa72455dae__SEAT_SURFACE_1__n0', '单条完整座面板条',
     ('COMPLETE_MACRO', 'COMPLETE_MACRO'), 'PLAUSIBLE', 'NO', 'UNCERTAIN',
     ('UNCERTAIN', 'YES'),
     '生成物是四腿板条凳，蓝窄条明显低于灰座面，在两腿之间可解释成低位横撑。此完成没有采用构造时的座面角色，但开放生成不要求该唯一角色；侧截面同样显示蓝条低于上方灰板。',
     '四腿与低位蓝横条保持，灰座面板条变薄且条间空隙变宽；前截面部分灰线消失不能独立证明外壳破洞。蓝条仍可作低位横向构件，真实连接未知。',
     '蓝条可作低位横撑，POST让上方灰板条更薄。'),
    ('20222f467d257022c8bbbd98dee48cb__BACK_SURFACE_1__n0', '开放矩形背框',
     ('COMPLETE_MACRO', 'COMPLETE_MACRO'), 'NO_VISIBLE_COHERENT_USE', 'PARTIAL', 'YES',
     ('NO_OBVIOUS_IN_VIEWS', 'NO_OBVIOUS_IN_VIEWS'),
     '灰体是高实心背板、厚座和四腿组成的完整长凳；蓝开放框位于整块灰背板前方。侧截面显示蓝框与灰背面前后分离，未见蓝框成为背部的连续结构。',
     '高背凳形状与框前后错层基本不变，灰表面稍平。截面不覆盖所有接点，仍不认证蓝框连接或断开。',
     '主体完整，蓝框前方另置及灰背重复层关系持续。'),
    ('20222f467d257022c8bbbd98dee48cb__COMBO_SEAT_SUPPORT__n0', '后座板条加楔形端支撑',
     ('COMPLETE_MACRO', 'PARTIAL'), 'UNCERTAIN', 'PARTIAL', 'YES',
     ('UNCERTAIN', 'YES'),
     '蓝后沿条与楔形端片位于座板及支撑区，位置大致合理；蓝端片旁仍有独立可见的灰腿。侧截面中蓝条与灰座板端部存在偏置，没有显示清楚的连续轮廓。',
     '灰背与座面更平，但自由腿明显变成细线，保留蓝楔片因此更突出；蓝后沿条与灰体的局部错位关系仍未清楚消失。',
     '表面更平，代价是自由腿显著削薄；蓝灰接合仍不确定。'),
]

now = datetime.datetime.now(datetime.timezone.utc).isoformat()
rows, pairs = [], []
for page, note in zip(manifest['pages'], notes):
    rid, part, body, use, role, layer, thin, raw, post, change = note
    assert page['request_id'] == rid
    assert [c['label'] for c in page['cells']][:2] == [LABEL + '_RAW', LABEL + '_POST']
    path = GALLERY / (page['page_id'] + '.png')
    ir = read(path.with_suffix('.json'))
    assert sha(path) == ir['image_sha256']
    assert ir['definition_sha256'] == completion['definition_sha256']
    for j, stage in enumerate(('RAW', 'POST')):
        rows.append(dict(
            review_version=schema['review_version'], reviewer='AI_PROXY', actual_image_viewed=True,
            human_reviewed=False, nonblind=True, reviewed_at_utc=now, label=LABEL, global_step=17000,
            request_id=rid, condition_id=page['condition_id'], parent_id=page['parent_id'], split=page['split'],
            stage=stage, K=page['K'], N=page['N'], part_description=part, image_path=str(path),
            image_sha256=sha(path), image_receipt_sha256=sha(path.with_suffix('.json')),
            gallery_definition_sha256=completion['definition_sha256'],
            views_used=['surface_-60_23', 'surface_120_23', 'finite_front_section_at_C_bbox_center',
                        'finite_side_section_at_C_bbox_center'],
            reviewed_cell_label=LABEL + '_' + stage, other_model_cells_visible=True,
            body_recognizable_as_bench=True, body_completeness=body[j], C_visual_coherent_use=use,
            construction_intended_role_visible=role, visible_offset_or_redundant_layer=layer,
            thin_or_dangling_sheet_appearance=thin[j], exact_shared_connectivity='NOT_ASSESSED_FROM_IMAGES',
            topologically_disconnected_fragment='UNKNOWN_IMAGE_ONLY', notes=(raw, post)[j],
            metric_values_used_for_labels=False, source_full_mesh_shown=False,
            scope='Fixed process checkpoint Native-only observation; no checkpoint/data/seed selection or training-policy changes'))
    pairs.append(dict(request_id=rid, label=LABEL, split=page['split'], part_description=part,
                      RAW_to_POST=change, image_sha256=sha(path), reviewer='AI_PROXY', actual_image_viewed=True))

def write(name, content):
    with (OUT / name).open('x', encoding='utf-8') as stream:
        stream.write(content)

write('review.jsonl', ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
write('raw_post_comparison.jsonl', ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in pairs))
write('review_schema.json', json.dumps(schema, ensure_ascii=False, indent=2) + '\n')
md = [
    '# NATIVE_S17000 固定过程图审（AI_PROXY）', '',
    '实际逐页查看8张联合图页，仅记录左两列 NATIVE_S17000_RAW/POST，含两个表面视角及固定前/侧截面。SEAM列可见，故为非盲图审。本记录保存16条阶段观察与8条前后比较，没有读取数值metric或完整来源网格。', '',
    '459f完整座板仍高于灰座面，20222背框仍位于整块灰背板前方。U形端架和单腿输入的位置较合理，但真实共享连接未知。122a窄条没有呈现来源座面用途，却可解释为低位横撑，不能因此直接判开放生成失败。多份POST表面更平，同时自由腿或板条明显变薄。', '',
    '|请求|部位|RAW → POST|', '|---|---|---|']
for pair in pairs:
    md.append('|' + pair['request_id'][:6] + '/' + pair['request_id'].split('__')[1] + '|' +
              pair['part_description'] + '|' + pair['RAW_to_POST'] + '|')
md += ['', '共享索引连接全部 NOT_ASSESSED；拓扑断片全部 UNKNOWN_IMAGE_ONLY。截面短线、空缺或投影交叉不直接认证断片、孔洞或非法穿透。8请求过程面板不用于挑checkpoint、更改训练或宣布收敛。', '']
write('REVIEW.md', '\n'.join(md))
write('receipt.json', json.dumps(dict(
    status='ACTUAL_IMAGE_REVIEW_COMPLETE', reviewer='AI_PROXY', reviewed_at_utc=now,
    images_viewed=8, image_view_calls_per_page=1, reviewed_model=LABEL, source_gallery_key=KEY,
    reviewed_cell_indices=[0, 1], stage_rows=16, paired_rows=8, model_calls=0, new_samples=0,
    rescoring=0, data_audits=0, protected_gallery_definition_sha256=sha(GALLERY / 'definition.json'),
    protected_manifest_sha256=sha(manifest_path),
    files={p.name: sha(p) for p in OUT.iterdir() if p.is_file()}), ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE', images=8, stage_rows=16,
                      paired_rows=8, model_calls=0)))
