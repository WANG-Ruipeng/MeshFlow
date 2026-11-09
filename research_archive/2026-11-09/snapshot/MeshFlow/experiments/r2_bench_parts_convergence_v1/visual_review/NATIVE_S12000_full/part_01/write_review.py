"""Write only manually supplied observations after assigned PNGs were viewed."""
import datetime
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
LABEL='NATIVE_S12000'
GALLERY=ROOT/'gallery'/f'{LABEL}_full'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest_path=ROOT/'gallery_manifests'/f'{LABEL}_full.json'
manifest=read(manifest_path)
complete=read(GALLERY/'completion.json')
assert len(complete['images'])==48
pages=sorted(manifest['pages'],key=lambda p:p['request_id'])[:16]
observations=read(OUT/'observations.json')
assert {r['request_id'] for r in observations}=={p['request_id'] for p in pages}
observations={r['request_id']:r for r in observations}
old_schema=ROOT/'visual_review/NATIVE_S10000_process/review_schema.json'
schema=read(old_schema)
schema.update(checkpoint=LABEL,scope='Full fixed 48-request baseline panel; part_01 assigned lexicographic requests 1..16 only',actual_view_count=16,
    prior_schema_sha256=sha(old_schema),assignment='sorted(request_id)[0:16]')
schema['limitations']=[x for x in schema['limitations'] if not x.startswith('No pretraining image baseline')]
schema['limitations'].append('This shard reviews only 16 assigned requests and does not represent the full-panel outcome by itself.')
schema['limitations'].append('Two fixed noises per condition do not establish broad diversity or generalization; no source-style ranking.')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
rows=[];pairs=[]
for p in pages:
    ob=observations[p['request_id']]
    assert ob['actual_image_viewed'] is True
    im=GALLERY/(p['page_id']+'.png');ir=read(im.with_suffix('.json'))
    assert sha(im)==ir['image_sha256'] and ir['definition_sha256']==sha(GALLERY/'definition.json')
    for stage in ('RAW','POST'):
        n=ob[stage]
        rows.append(dict(review_version=schema['review_version'],reviewer='AI_PROXY',actual_image_viewed=True,human_reviewed=False,nonblind=True,
            reviewed_at_utc=now,label=LABEL,global_step=12000,request_id=p['request_id'],condition_id=p['condition_id'],parent_id=p['parent_id'],
            split=p['split'],stage=stage,K=p['K'],N=p['N'],part_description=ob['part_description'],image_path=str(im),image_sha256=sha(im),
            image_receipt_sha256=sha(im.with_suffix('.json')),gallery_definition_sha256=sha(GALLERY/'definition.json'),
            views_used=['surface_-60_23','surface_120_23','finite_front_section_at_C_bbox_center','finite_side_section_at_C_bbox_center'],
            body_recognizable_as_bench=n['body_recognizable_as_bench'],body_completeness=n['body_completeness'],C_visual_coherent_use=n['C_visual_coherent_use'],
            construction_intended_role_visible=n['construction_intended_role_visible'],visible_offset_or_redundant_layer=n['visible_offset_or_redundant_layer'],
            thin_or_dangling_sheet_appearance=n['thin_or_dangling_sheet_appearance'],exact_shared_connectivity='NOT_ASSESSED_FROM_IMAGES',
            topologically_disconnected_fragment='UNKNOWN_IMAGE_ONLY',notes=n['notes'],metric_values_used_for_labels=False,source_full_mesh_shown=False,
            scope='Pre-SEAM COMMON12000 fixed full panel; assigned part_01 only; no checkpoint/data/seed selection or training-policy changes'))
    pairs.append(dict(request_id=p['request_id'],split=p['split'],part_description=ob['part_description'],RAW_to_POST=ob['RAW_to_POST'],
        image_sha256=sha(im),reviewer='AI_PROXY',actual_image_viewed=True))
def write(name,text):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(text)
write('review.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
write('raw_post_comparison.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs))
write('review_schema.json',json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
md=['# NATIVE_S12000 完整面板图审：part_01','',
 '已实际查看按request_id字典序排列的前16张图页；每页含RAW/POST、两个表面视角与C中心前/侧有限截面，共32条阶段观察。此处只覆盖完整48请求面板的分配子集，不代表完整面板或最终SEAM结论。非盲AI_PROXY，未以metric、来源款式或未打开图像填写标签。','',
 '|请求|部位|RAW → POST实际观察|','|---|---|---|']
for p in pairs:
    rid=p['request_id'].split('__')
    md.append('|'+rid[0][:6]+' / '+rid[1]+' / '+rid[2]+'|'+p['part_description']+'|'+p['RAW_to_POST']+'|')
md+=['','逐阶段记录见`review.jsonl`。C视觉用途与精确接续分开：全部共享索引连接、三维拓扑断片为NOT_ASSESSED/UNKNOWN。有限截面短段、投影交叉或规则板条间隙均不单独认证非法相交或断片；无部位名输入时保留合理替代用途的歧义。','',
 '这是COMMON12000分叉前基线的实际图审，不是Native与SEAM比较。未调用模型、重采样、重评分、审计新数据或改变生产代码；不按图选择checkpoint或修改训练。图片hash、图集定义hash及实际查看声明见receipt和逐条记录。','']
write('REVIEW.md','\n'.join(md))
write('receipt.json',json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',reviewer='AI_PROXY',reviewed_at_utc=now,
    images_viewed=16,image_view_calls_per_page=1,stage_rows=32,paired_rows=16,assignment='sorted(request_id)[0:16]',
    model_calls=0,new_samples=0,rescoring=0,data_audits=0,protected_gallery_definition_sha256=sha(GALLERY/'definition.json'),
    protected_manifest_sha256=sha(manifest_path),files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()}),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',images=16,stage_rows=32,model_calls=0)))
