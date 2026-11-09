"""Save the 16 actual terminal image reviews; no rendering, scoring or model calls."""
import datetime
import hashlib
import json
from pathlib import Path
from observations import notes

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
KEY = 'NATIVE_S21000__SEAM_S21000_full'
GALLERY = ROOT / 'gallery' / KEY
PRIOR = ROOT / 'visual_review/NATIVE_S15000__SEAM_S15000_full/part_02/review_schema.json'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

manifest_path = ROOT / 'gallery_manifests' / (KEY + '.json')
manifest = read(manifest_path)
completion = read(GALLERY / 'completion.json')
definition_hash = sha(GALLERY / 'definition.json')
assert completion['status'] == 'RENDER_COMPLETE_PENDING_REVIEW'
assert completion['definition_sha256'] == definition_hash
assert len(manifest['pages']) == len(completion['images']) == 48
all_pages = sorted(manifest['pages'], key=lambda row: row['request_id'])
assert len(set(page['request_id'] for page in all_pages)) == 48
pages = all_pages[:16]
assert len(notes) == len(pages) == 16
image_receipts = {row['request_id']: row for row in completion['images']}
schema = read(PRIOR)
schema.update(checkpoint='NATIVE_S21000 / SEAM_S21000', actual_view_count=16,
              scope='Part01: sorted request_id indices0..15, both terminal arms; 64 RAW/POST observations',
              prior_schema_sha256=sha(PRIOR))
schema['limitations'] = [
    'Two surface projections and finite central sections do not cover all surfaces.',
    'A section short line, empty interval or projected crossing is not proof of a disconnected triangle, hole or illegal intersection.',
    'Thin supports can be geometrically present but visibly weak; topology remains unassessed.',
    'A geometrically ambiguous part can have a reasonable alternative use; intended source role is diagnostic only.',
    'This16-request subset of the21000 final full panel is not the entire evaluation or proof of convergence.',
    'Both model labels and all four columns were visible: this is nonblind AI_PROXY review.'
]
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
rows, pairs, viewed = [], [], []
for index, (page, observation) in enumerate(zip(pages, notes)):
    request_id, part, *arms = observation
    assert page['request_id'] == request_id
    assert [cell['label'] for cell in page['cells']] == [
        'NATIVE_S21000_RAW', 'NATIVE_S21000_POST', 'SEAM_S21000_RAW', 'SEAM_S21000_POST']
    path = GALLERY / (page['page_id'] + '.png')
    image_hash = sha(path)
    ir = read(path.with_suffix('.json'))
    assert image_hash == ir['image_sha256'] == image_receipts[request_id]['image_sha256']
    assert ir['definition_sha256'] == definition_hash
    viewed.append(dict(sorted_request_index=index, request_ordinal_1_based=index+1,
                       request_id=request_id, page_id=page['page_id'], image_path=str(path), image_sha256=image_hash))
    for label, obs in zip(('NATIVE_S21000', 'SEAM_S21000'), arms):
        body, use, role, layer, thin, raw, post, change = obs
        for j, stage in enumerate(('RAW', 'POST')):
            rows.append(dict(
                review_version=schema['review_version'], reviewer='AI_PROXY', actual_image_viewed=True,
                human_reviewed=False, nonblind=True, reviewed_at_utc=now, label=label, global_step=21000,
                request_id=request_id, sorted_request_index=index, request_ordinal_1_based=index+1,
                condition_id=page['condition_id'], parent_id=page['parent_id'], split=page['split'],
                stage=stage, K=page['K'], N=page['N'], part_description=part, image_path=str(path),
                image_sha256=image_hash, image_receipt_sha256=sha(path.with_suffix('.json')),
                gallery_definition_sha256=definition_hash, gallery_manifest_sha256=sha(manifest_path),
                views_used=['surface_-60_23', 'surface_120_23', 'finite_front_section_at_C_bbox_center',
                            'finite_side_section_at_C_bbox_center'],
                reviewed_cell_label=label+'_'+stage, other_model_cells_visible=True,
                body_recognizable_as_bench=True, body_completeness=body[j], C_visual_coherent_use=use,
                construction_intended_role_visible=role, visible_offset_or_redundant_layer=layer,
                thin_or_dangling_sheet_appearance=thin[j], exact_shared_connectivity='NOT_ASSESSED_FROM_IMAGES',
                topologically_disconnected_fragment='UNKNOWN_IMAGE_ONLY', notes=(raw, post)[j],
                metric_values_used_for_labels=False, source_full_mesh_shown=False,
                scope='Fixed terminal21000 full-panel first16 requests; actual image-only review, no selection or training changes'))
        pairs.append(dict(label=label, request_id=request_id, sorted_request_index=index,
                          request_ordinal_1_based=index+1, split=page['split'], part_description=part,
                          RAW_to_POST=change, image_sha256=image_hash, reviewer='AI_PROXY', actual_image_viewed=True))
assert len(rows) == 64 and len(pairs) == 32
assert len(set((row['label'], row['request_id'], row['stage']) for row in rows)) == 64
assert [row['sorted_request_index'] for row in viewed] == list(range(16))

def write(name, value):
    with (OUT / name).open('x', encoding='utf-8') as file:
        file.write(value)

write('review.jsonl', ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
write('raw_post_comparison.jsonl', ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in pairs))
write('review_schema.json', json.dumps(schema, ensure_ascii=False, indent=2)+'\n')
write('viewed_images.json', json.dumps(viewed, ensure_ascii=False, indent=2)+'\n')
md = ['# 21000终点两臂图审：前16请求（AI_PROXY）', '',
      '固定48请求按request_id字典排序的index0..15（第1–16条）已逐页实际查看。每张联合页包含两臂RAW/POST、两个表面视角及固定前/侧截面；保存64阶段记录、32组前后比较。标签可见，故为非盲AI_PROXY。没有读取数值metric、完整来源网格或据此挑图。', '',
      '本分片较清楚的SEAM局部正例是1dd/SEAT/n0：蓝板在箱式长凳的上座面位置，POST后其上表面更清楚，而普通臂仍悬在高背凳的座面上方。1dd/LEG/n1、20222/COMBO/n1也可看到合理的输入部位用途，但并非SEAM独有收益。122a窄条和2d背板在若干输出中可解释为低位横撑、搁板或侧裙板；未采用唯一来源角色不直接判失败。', '',
      '保留相反证据：20222/COMBO/n0普通臂蓝座沿和端支撑的用途较清楚，SEAM则仍有蓝楔片与灰腿并置、POST自由腿变细。20222/BACK/n0的SEAM虽有更多灰斜片，却仍有背部层差和大三角开口。2d/LEG/n1两臂蓝小支撑均主要藏在灰架内部，缺少自然采用的可见证据。', '',
      '主要代价是POST的腿架、板条削薄，以及部分宽薄片、端部开口和蓝灰错层。所有真实共享连接均NOT_ASSESSED，拓扑断片均UNKNOWN_IMAGE_ONLY；截面线段不直接认证孔洞、非法相交或断开。', '',
      '|编号|请求|模型|RAW → POST|', '|---|---|---|---|']
for pair in pairs:
    rid = pair['request_id']
    md.append('|'+str(pair['request_ordinal_1_based'])+'|'+rid[:6]+'/'+rid.split('__')[1]+'/'+rid.split('__')[-1]+'|'+pair['label']+'|'+pair['RAW_to_POST']+'|')
md += ['', '这16请求包含12 TRAIN与4 DEV输出身份，只是完整48请求的一部分；不单独用于全组胜负、泛化或收敛结论。', '']
write('REVIEW.md', '\n'.join(md))
write('receipt.json', json.dumps(dict(
    status='ACTUAL_IMAGE_REVIEW_COMPLETE', reviewer='AI_PROXY', reviewed_at_utc=now,
    images_viewed=16, image_view_calls_per_page=1, sorted_request_indices=[0,15],
    request_ordinals_1_based=[1,16], exact_index_coverage_pass=True,
    models=['NATIVE_S21000','SEAM_S21000'], stage_rows=64, paired_rows=32,
    model_calls=0, new_samples=0, rescoring=0, data_audits=0,
    protected_gallery_definition_sha256=definition_hash, protected_manifest_sha256=sha(manifest_path),
    protected_completion_sha256=sha(GALLERY/'completion.json'),
    request_ids=[page['request_id'] for page in pages],
    files={path.name:sha(path) for path in OUT.iterdir() if path.is_file()}), ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE', images=16, stage_rows=64,
                      paired_rows=32, index_coverage=[0,15], model_calls=0)))
