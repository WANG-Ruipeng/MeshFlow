"""Persist notes after all eight existing S6000 gallery pages were viewed once."""
import datetime
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
LABEL='NATIVE_S6000'
GALLERY=ROOT/'gallery'/f'{LABEL}_process'
MANIFEST=ROOT/'gallery_manifests'/f'{LABEL}_process.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=read(MANIFEST);complete=read(GALLERY/'completion.json')
assert len(complete['images'])==8 and len(manifest['pages'])==8
prior=ROOT/'visual_review/NATIVE_S2000_process'
schema=read(prior/'review_schema.json')
schema['checkpoint']=LABEL
schema['actual_view_count']=8
schema['prior_schema_sha256']=sha(prior/'review_schema.json')
observations=[
 dict(prefix='2d1e5',part='靠背板',body=('PARTIAL','PARTIAL'),use='UNCERTAIN',role='YES',layer='YES',thin=('UNCERTAIN','UNCERTAIN'),
 raw='蓝C位于靠背区域，灰色体可辨认为长凳；蓝板后面仍有近乎平行的另一层灰背板，侧截面可见前后偏移。下部为弯曲宽片，两端支撑较不规整。仅凭投影不能确定宽片是否独立悬浮。',
 post='下部曲线较平顺，蓝板与灰背板错层仍在；表面视角右端可见较窄的片状开口。没有看到蓝C与周围体明确形成连续表面的证据。',
 change='局部平顺；蓝灰靠背错层仍在。'),
 dict(prefix='459f90aa2162b1f1d46c340938e2ff1c__LEG',part='座板＋端支撑组合',body=('PARTIAL','PARTIAL'),use='UNCERTAIN',role='PARTIAL',layer='YES',thin=('YES','YES'),
 raw='蓝色组合包含座板及一个曲面端支撑；灰色长条片另铺在蓝座板上方。前、侧截面显示持续高度差，另一端灰支撑较短。支撑功能位置合理，但新增灰片像额外一层座面。',
 post='上方灰条部分变平，长条间的较大开口和缺损更明显；该灰层仍高于蓝座板。短灰端支撑仍在，不能把表面平整等同于整体协调。',
 change='额外灰座面层未消失；POST条带间开口更明显。'),
 dict(prefix='459f90aa2162b1f1d46c340938e2ff1c__SEAT',part='完整矩形座面',body=('COMPLETE_MACRO','COMPLETE_MACRO'),use='NO_VISIBLE_COHERENT_USE',role='NO',layer='YES',thin=('UNCERTAIN','UNCERTAIN'),
 raw='灰体形成可辨认的有靠背长凳；蓝色完整座板悬在灰座面上方，两种截面均显示明显间隔。蓝C没有作为可见灰座面的延续。',
 post='灰体稍平整，蓝灰座面高度差仍然清楚。背部截面短线不单独认证为漂浮面或孔洞；输入错层没有明显改变。',
 change='主体保持，蓝C与灰座面的分层持续。'),
 dict(prefix='9fd5',part='U形端支撑',body=('PARTIAL','PARTIAL'),use='PLAUSIBLE',role='YES',layer='UNCERTAIN',thin=('YES','YES'),
 raw='蓝U形架处于凳体一端，灰座面从其内侧向另一端伸出，端支撑用途在位置上合理。另一端灰腿较薄，蓝架内侧有细长灰片，精确连接无法从图像确定。',
 post='灰座面更平整，但另一端灰腿变得近线状，灰座面下沿也出现清楚的细长间隙；蓝架位置不变。没有充分证据说C接续改善。',
 change='座面平顺伴随对端支撑显著变薄。'),
 dict(prefix='122a480cfcdd742650c626aa72455dae__LEG',part='一条实心腿',body=('COMPLETE_MACRO','COMPLETE_MACRO'),use='PLAUSIBLE',role='YES',layer='YES',thin=('UNCERTAIN','YES'),
 raw='灰色体有完整座面、靠背和宽曲面侧支撑，蓝腿上端接近座面下侧，支撑用途在位置上合理。蓝腿多被宽灰支撑面遮挡；侧截面仍见蓝腿附近多条斜向灰线，不能由遮挡判定替代或相交。',
 post='主要座面和靠背保持；右侧曲面出现更分散的扇状细片，蓝腿附近的灰色层次仍可见。宏观完整不代表这些侧片已经拓扑连接或没有冲突。',
 change='宏观主体保持；POST右侧扇状细片更显眼。'),
 dict(prefix='122a480cfcdd742650c626aa72455dae__SEAT',part='一条完整座面板条',body=('COMPLETE_MACRO','COMPLETE_MACRO'),use='UNCERTAIN',role='NO',layer='YES',thin=('NO_OBVIOUS_IN_VIEWS','NO_OBVIOUS_IN_VIEWS'),
 raw='生成体呈双层平板架／凳体，有四腿、上层面和下层面；蓝色窄条位于两层之间，接近腿间横向拉条位置，而不是上方座面。主体可辨认且相对规整，但C是否真正接入无法凭图确定。输入没有部位标签，替代用途不能按唯一来源款式直接判错。',
 post='双层结构与中间蓝条位置保持，边缘略平整；没有看到蓝条改为构造时座面用途。此页未见明显小悬片，不代表所有表面均无断裂。',
 change='双层架体保持；蓝C仍是中间条状元素，用途存在歧义。'),
 dict(prefix='20222f467d257022c8bbbd98dee48cb__BACK',part='开放矩形靠背框',body=('COMPLETE_MACRO','COMPLETE_MACRO'),use='NO_VISIBLE_COHERENT_USE',role='PARTIAL',layer='YES',thin=('NO_OBVIOUS_IN_VIEWS','NO_OBVIOUS_IN_VIEWS'),
 raw='灰体为高背长凳，蓝矩形框位于整块灰背板前方。侧截面显示明显前后错层；虽然同处靠背区域，蓝框不像灰背板的连续组成部分。',
 post='高灰背板、座面与蓝框位置关系基本保持，未见明显接入变化。没有大量小碎片外观，仍不能凭表面投影认证拓扑。',
 change='灰背板前方的蓝框错层持续，整体变化小。'),
 dict(prefix='20222f467d257022c8bbbd98dee48cb__COMBO',part='后座板条＋端支撑侧面',body=('PARTIAL','PARTIAL'),use='UNCERTAIN',role='PARTIAL',layer='UNCERTAIN',thin=('YES','YES'),
 raw='蓝长条处于灰座面后缘附近，蓝楔形片位于一端支撑处，功能位置大致合理。侧截面里蓝段与灰座面端部仍有间隔，蓝端片旁有另一条灰腿；无法确认合成整体。',
 post='灰背板与座面平整，但自由端腿与支撑明显变成很细的条或线，蓝楔形片仍突出。没有看到间隔明确消失，主体支撑显得薄弱。',
 change='表面平顺伴随自由腿变薄，不能记为全面改善。'),
]
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
rows=[];pairs=[]
for page,n in zip(manifest['pages'],observations):
    assert page['request_id'].startswith(n['prefix'])
    im=GALLERY/(page['page_id']+'.png');ir=read(im.with_suffix('.json'))
    assert sha(im)==ir['image_sha256'] and ir['definition_sha256']==sha(GALLERY/'definition.json')
    for j,stage in enumerate(('RAW','POST')):
        rows.append(dict(review_version=schema['review_version'],reviewer='AI_PROXY',actual_image_viewed=True,human_reviewed=False,nonblind=True,
            reviewed_at_utc=now,label=LABEL,global_step=6000,request_id=page['request_id'],condition_id=page['condition_id'],parent_id=page['parent_id'],
            split=page['split'],stage=stage,K=page['K'],N=page['N'],part_description=n['part'],image_path=str(im),image_sha256=sha(im),
            image_receipt_sha256=sha(im.with_suffix('.json')),gallery_definition_sha256=sha(GALLERY/'definition.json'),
            views_used=['surface_-60_23','surface_120_23','finite_front_section_at_C_bbox_center','finite_side_section_at_C_bbox_center'],
            body_recognizable_as_bench=True,body_completeness=n['body'][j],C_visual_coherent_use=n['use'],construction_intended_role_visible=n['role'],
            visible_offset_or_redundant_layer=n['layer'],thin_or_dangling_sheet_appearance=n['thin'][j],exact_shared_connectivity='NOT_ASSESSED_FROM_IMAGES',
            topologically_disconnected_fragment='UNKNOWN_IMAGE_ONLY',notes=n[stage.lower()],metric_values_used_for_labels=False,source_full_mesh_shown=False,
            scope='Fixed process checkpoint observation; no checkpoint/data/seed selection or training-policy changes'))
    pairs.append(dict(request_id=page['request_id'],split=page['split'],part_description=n['part'],RAW_to_POST=n['change'],image_sha256=sha(im),reviewer='AI_PROXY',actual_image_viewed=True))
def write(name,text):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(text)
write('review.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
write('raw_post_comparison.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs))
write('review_schema.json',json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
md=['# NATIVE_S6000 过程图审（AI_PROXY）','',
 '已实际逐张查看固定process面板的8张既有图页，每张仅查看一次；每页含RAW/POST、两个表面视角与C中心位置前/侧有限截面。形成16条逐阶段记录与8条处理前后对照，沿用S2000图审字段。非盲AI_PROXY，非人工评分；未用数值metric推标签，未查看来源完整款式作为成功答案。','',
 '本轮8页都能辨认凳体或支撑架体，但C的自然采用仍有明显不确定项。459f90完整座面仍位于另一层灰座面上方；20222蓝背框仍处于整块灰背板前方。122a窄板输入处于双层架体中间，像横向拉条；因为输入没有部位语义标签，不能仅因不是构造时的座面用途就宣布开放生成失败。','',
 '后处理使部分表面更平，但9fd5腿架与20222组合请求的自由支撑显著变薄；459f90组合请求上方灰条之间开口更明显。图像中的细线、短段或扇状片只记录外观，未据此宣称三维分量断开。','',
 '|请求（n0）|部位|RAW → POST实际观察|','|---|---|---|']
for p in pairs:md.append('|'+p['request_id'].split('__')[0][:6]+' / '+p['request_id'].split('__')[1]+'|'+p['part_description']+'|'+p['RAW_to_POST']+'|')
md += ['',
 '逐项见`review.jsonl`与`raw_post_comparison.jsonl`。可见用途不等于合法共享边，宏观完整不等于几何无冲突；精确连接和拓扑断片继续保持NOT_ASSESSED/UNKNOWN。闭合C不因为没有开放接口而被额外惩罚。','',
 '本材料仅是6000步过程观察，不是最终SEAM结果或收敛判断。没有重新采样、调用模型、重新评分、修改指标、改变训练或挑选checkpoint；后续仍按冻结合同执行完整面板。当前页的两个投影与有限截面不能覆盖所有离截面的细节。','']
write('REVIEW.md','\n'.join(md))
receipt=dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',reviewer='AI_PROXY',reviewed_at_utc=now,images_viewed=8,image_view_calls_per_page=1,stage_rows=16,paired_rows=8,
    model_calls=0,new_samples=0,rescoring=0,protected_gallery_definition_sha256=sha(GALLERY/'definition.json'),protected_manifest_sha256=sha(MANIFEST),
    files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()})
write('receipt.json',json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'],images=8,stage_rows=16,model_calls=0),ensure_ascii=True))
