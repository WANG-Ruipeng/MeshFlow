"""Persist observations only after all 16 assigned immutable pages are viewed."""
import datetime
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
KIND='NATIVE_S21000__SEAM_S21000_full'
LABELS=('NATIVE_S21000','SEAM_S21000')
GALLERY=ROOT/'gallery'/KIND
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,text):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(text)
manifest_path=ROOT/'gallery_manifests'/(KIND+'.json')
manifest=read(manifest_path)
complete=read(GALLERY/'completion.json')
assert len(complete['images'])==48
pages=sorted(manifest['pages'],key=lambda p:p['request_id'])[32:48]
obs=read(OUT/'observations.json')
assert len(obs)==16 and {r['request_id'] for r in obs}=={p['request_id'] for p in pages}
obs={r['request_id']:r for r in obs}
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
schema=read(ROOT/'visual_review/NATIVE_S15000__SEAM_S15000_full/part_01/review_schema.json')
schema.update(checkpoint=KIND,scope='Fixed paired 21000 full panel; sorted request_id indices 32..47',
    actual_view_count=16,actual_stage_rows=64,model_labels=list(LABELS),assignment='sorted(request_id)[32:48]')
keys=['body_recognizable_as_bench','body_completeness','C_visual_coherent_use','construction_intended_role_visible',
      'visible_offset_or_redundant_layer','thin_or_dangling_sheet_appearance','notes']
rows=[];pairs=[];cross=[]
for p in pages:
    ob=obs[p['request_id']]
    assert ob['actual_image_viewed'] is True
    im=GALLERY/(p['page_id']+'.png');ir=read(im.with_suffix('.json'))
    assert sha(im)==ir['image_sha256'] and ir['definition_sha256']==sha(GALLERY/'definition.json')
    for label in LABELS:
        for stage in ('RAW','POST'):
            values=ob[label][stage]
            assert len(values)==len(keys)
            n=dict(zip(keys,values))
            rows.append(dict(review_version=schema['review_version'],reviewer='AI_PROXY',review_agent_shard='part_03',
                actual_image_viewed=True,human_reviewed=False,nonblind=True,reviewed_at_utc=now,label=label,global_step=21000,
                request_id=p['request_id'],condition_id=p['condition_id'],parent_id=p['parent_id'],split=p['split'],stage=stage,
                K=p['K'],N=p['N'],part_description=ob['part_description'],image_path=str(im),image_sha256=sha(im),
                image_receipt_sha256=sha(im.with_suffix('.json')),gallery_definition_sha256=sha(GALLERY/'definition.json'),
                views_used=['surface_-60_23','surface_120_23','finite_front_section_at_C_bbox_center','finite_side_section_at_C_bbox_center'],
                **n,exact_shared_connectivity='NOT_ASSESSED_FROM_IMAGES',topologically_disconnected_fragment='UNKNOWN_IMAGE_ONLY',
                metric_values_used_for_labels=False,source_full_mesh_shown=False,
                scope='Registered paired21000 final endpoint image review; assigned16/48 only; no checkpoint/data/seed selection'))
        pairs.append(dict(label=label,request_id=p['request_id'],split=p['split'],part_description=ob['part_description'],
            RAW_to_POST=ob[label]['RAW_to_POST'],image_sha256=sha(im),reviewer='AI_PROXY',review_agent_shard='part_03',actual_image_viewed=True))
    cross.append(dict(request_id=p['request_id'],comparison=ob['paired_observation'],reviewer='AI_PROXY',
        actual_image_viewed=True,image_sha256=sha(im)))
write('review.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
write('raw_post_comparison.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs))
write('paired_model_observations.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cross))
write('review_schema.json',json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
md=['# 21000步Native／SEAM配对图审：part_03','',
 '已实际打开按request_id字典序排列的第33至48张组合页；每页查看两臂RAW/POST、两种表面视角和固定前/侧有限截面。保存64条阶段观察、32条同臂RAW→POST比较、16条模型间观察。非盲AI_PROXY，未由metric推标签，未展示唯一来源完整款式。','',
 '|请求|实际模型间观察|','|---|---|']
for p in cross:
    rid=p['request_id'].split('__')
    md.append('|'+rid[0][:6]+' / '+rid[1]+' / '+rid[2]+'|'+p['comparison']+'|')
md+=['','这里只覆盖完整48请求中的16请求，不单独形成全组胜负判断。薄片、错层、主体支撑变化与C用途分别记录；合理替代用途保留歧义。全部精确共边为NOT_ASSESSED_FROM_IMAGES、拓扑断片为UNKNOWN_IMAGE_ONLY。有限截面短线/空隙不能单独认证碎片、穿插或破洞。','',
 '这是固定最终21000检查点的观察，未按图改变数据、噪声或终点。未调用模型、渲染、重评分或改训练代码。','']
write('REVIEW.md','\n'.join(md))
write('receipt.json',json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',reviewer='AI_PROXY',review_agent_shard='part_03',reviewed_at_utc=now,
    images_viewed=16,image_view_calls_per_page=1,stage_rows=64,paired_rows=32,cross_model_rows=16,assignment='sorted(request_id)[32:48]',
    model_labels=list(LABELS),model_calls=0,new_samples=0,rescoring=0,new_rendering=0,
    protected_gallery_definition_sha256=sha(GALLERY/'definition.json'),protected_manifest_sha256=sha(manifest_path),
    files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()}),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',images=16,stage_rows=64,paired_rows=32,model_calls=0)))
