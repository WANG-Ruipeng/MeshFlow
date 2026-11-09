"""Persist actual nonblind Native-only viewing of eight immutable 19000 pages."""
import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
LABEL = 'NATIVE_S19000'
KEY = 'NATIVE_S19000__SEAM_S19000_process'
GALLERY = ROOT / 'gallery' / KEY
PRIOR = ROOT / 'visual_review/NATIVE_S17000_process'

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
schema['limitations'] = [x.replace('at17000', 'at19000') for x in schema['limitations']]

# The following notes were written after eight actual original-resolution image views.
# Images also display SEAM; those columns are not scored in this Native-only record.
notes = [
    ('2d1e5eeb255e79dcab53bf7fa22f3636__BACK_SURFACE_1__n0', '靠背板',
     ('COMPLETE_MACRO', 'COMPLETE_MACRO'), 'UNCERTAIN', 'YES', 'YES',
     ('UNCERTAIN', 'YES'),
     '灰体由宽背板、水平座面和弧形裙板底座组成，整体可辨认为长凳；蓝板处于靠背区，另一视角可见其后方的大块灰背板。侧截面显示蓝板与灰背轮廓前后分离，弧形底座不能仅因不同于四腿款式就判为主体缺失。',
     '宽背板、座面和弧形底座保持，灰表面更平，蓝灰背板前后分层仍在。两端弧形裙板附近三角开口更明显、边缘更薄；截面短线减少不能单独证明三维删面或拓扑断片。',
     '弧形底座宏观结构保持，背部错层持续，POST端部开口更明显。'),
    ('459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n0', '座板加弯曲端支撑',
     ('PARTIAL', 'PARTIAL'), 'UNCERTAIN', 'PARTIAL', 'YES',
     ('YES', 'YES'),
     '蓝座板和弯曲端支撑之上覆盖着不规则灰三角条片，条片之间有长开口；另一端有短灰支撑。前、侧截面清楚显示上层灰片高于蓝板，同时另有少量灰线位于蓝板厚度范围内。',
     '上层灰条片略平，但仍高于蓝板且保留长开口；另一端灰支撑较薄。薄片外观与投影重合不能直接认证拓扑断开或非法穿插。',
     '灰片略平，蓝灰错层和不规则开口仍明显。'),
    ('459f90aa2162b1f1d46c340938e2ff1c__SEAT_SURFACE_1__n0', '完整矩形座面',
     ('COMPLETE_MACRO', 'COMPLETE_MACRO'), 'NO_VISIBLE_COHERENT_USE', 'PARTIAL', 'YES',
     ('NO_OBVIOUS_IN_VIEWS', 'NO_OBVIOUS_IN_VIEWS'),
     '灰体有开放矩形靠背、灰座面与楔形端部支撑；蓝完整座板明显悬在灰座面上方。前、侧截面均保留分离的蓝灰水平层，未见灰侧壁向上接到蓝板。',
     '宏观长凳保持，楔形支撑仍有可见面积，灰面略平；蓝座板与灰座面的高度差没有清楚缩小，仍未显示连续座面关系。',
     '宏观凳体保持，独立蓝座板与灰座面高度差持续。'),
    ('9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n0', 'U形端支撑',
     ('COMPLETE_MACRO', 'PARTIAL'), 'PLAUSIBLE', 'YES', 'UNCERTAIN',
     ('YES', 'YES'),
     '蓝U形架位于灰座板一端，整体相对位置支持端支撑用途；另一端两条自由腿很细。侧截面中灰座板轮廓在蓝横梁下方，真实共享连接仍不可由图像判断。',
     '座板更平，远端两条自由腿进一步变为近线状细构件；蓝U形架完整保持。截面短杂线较少，不能因此认证接合更好。',
     '端架用途合理，POST进一步削薄远端自由腿。'),
    ('122a480cfcdd742650c626aa72455dae__LEG_SUPPORT_1__n0', '单条实心支撑腿',
     ('COMPLETE_MACRO', 'COMPLETE_MACRO'), 'PLAUSIBLE', 'YES', 'YES',
     ('UNCERTAIN', 'UNCERTAIN'),
     '灰体有完整板条靠背、座面、支撑腿和低位横撑；蓝腿位于其中一个支撑位置且上端接近座面，使用位置合理。前截面仍显示蓝腿旁的灰支撑轮廓，二者是否组成合理厚度或冗余构件无法由图像确认。',
     '板条凳宏观完整，灰板条更平、间隙较清楚，蓝腿及旁边的灰支撑关系保留。没有明显新增漂浮片，但不能由表面外观认证共享顶点连接。',
     '宏观完整性保持，支撑区域用途合理但相邻蓝灰构件关系仍有歧义。'),
    ('122a480cfcdd742650c626aa72455dae__SEAT_SURFACE_1__n0', '单条完整座面板条',
     ('COMPLETE_MACRO', 'COMPLETE_MACRO'), 'PLAUSIBLE', 'NO', 'UNCERTAIN',
     ('UNCERTAIN', 'YES'),
     '四腿板条凳的灰座面高于蓝窄条，蓝条在两腿间可解释成低位横撑。侧截面显示蓝条与上方灰座板相距明显；它没有成为构造时的座面，但合理替代用途不应按唯一来源角色判错。',
     '低位蓝横条和四腿保持，灰座板条明显变薄且开缝增大。中央截面灰线减少不能直接等同于三维外壳破洞，蓝条的真实连接仍未知。',
     '蓝条可作低位横撑，POST让上方灰板条更薄。'),
    ('20222f467d257022c8bbbd98dee48cb__BACK_SURFACE_1__n0', '开放矩形背框',
     ('COMPLETE_MACRO', 'COMPLETE_MACRO'), 'NO_VISIBLE_COHERENT_USE', 'PARTIAL', 'YES',
     ('NO_OBVIOUS_IN_VIEWS', 'NO_OBVIOUS_IN_VIEWS'),
     '灰体有高实心背板、厚座面和四腿；蓝开放矩形框位于整块灰背板前方。侧截面中蓝框与灰背轮廓相距明显，未显示蓝框替代或自然融入连续背面。',
     '主体高背凳与蓝框的相对位置基本保持，蓝灰前后分层仍清楚。部分灰面更平不代表蓝框已经拓扑接入，有限截面也不能认证其断开。',
     '主体完整，蓝框前方另置及灰背重复层关系持续。'),
    ('20222f467d257022c8bbbd98dee48cb__COMBO_SEAT_SUPPORT__n0', '后座板条加楔形端支撑',
     ('COMPLETE_MACRO', 'PARTIAL'), 'UNCERTAIN', 'PARTIAL', 'YES',
     ('UNCERTAIN', 'YES'),
     '蓝后沿条及楔形端片处于座面和支撑附近，位置大致合理；端片旁另有灰腿。侧截面中蓝条与灰座面后端仍有局部偏置，未显示清楚连续的面轮廓。',
     '背面和座面灰表面更平，而自由腿变成细长线片，蓝楔片因此显得突出。蓝后沿条附近的灰轮廓未清楚与其汇合，不能认证接合。',
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
            human_reviewed=False, nonblind=True, reviewed_at_utc=now, label=LABEL, global_step=19000,
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
    '# NATIVE_S19000 固定过程图审（AI_PROXY）', '',
    '实际逐页查看8张联合图页，仅记录左两列 NATIVE_S19000_RAW/POST，含两个表面视角及固定前/侧截面。SEAM列可见，故为非盲图审。本记录保存16条阶段观察与8条前后比较，没有读取数值metric或完整来源网格。', '',
    '2d形成宽背、座面与弧形底座，蓝灰背面仍前后分层，POST端部开口更明显。459f完整座板仍高于灰座面，20222背框仍位于整块灰背板前方。U形端架及单腿位置较合理；122a窄条可解释为低位横撑，不能仅因不是来源座面用途就判失败。多份POST表面更平，同时自由腿或板条明显变薄。', '',
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
