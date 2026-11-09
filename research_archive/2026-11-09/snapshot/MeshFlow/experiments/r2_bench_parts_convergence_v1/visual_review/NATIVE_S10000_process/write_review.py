"""Save image-only observations after viewing each of eight S10000 pages once."""
import datetime
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
LABEL='NATIVE_S10000'
GALLERY=ROOT/'gallery'/f'{LABEL}_process'
PRIOR=ROOT/'visual_review/NATIVE_S6000_process'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest_path=ROOT/'gallery_manifests'/f'{LABEL}_process.json'
manifest=read(manifest_path)
assert len(read(GALLERY/'completion.json')['images'])==8
schema=read(PRIOR/'review_schema.json')
schema.update(checkpoint=LABEL,actual_view_count=8,prior_schema_sha256=sha(PRIOR/'review_schema.json'))
notes=[
 ('2d1e5','靠背板',('PARTIAL','PARTIAL'),'UNCERTAIN','YES','YES',('UNCERTAIN','UNCERTAIN'),
  '灰体可辨认为长凳，蓝靠背位于靠背区域；蓝板后方另有平行灰板，侧截面显示前后错层。下部灰面呈弯曲兜状宽片，端部较不规整；不能仅据投影确定是否悬浮。',
  '下部曲线略平顺，右侧仍有薄片状开口；蓝板与灰背板间的偏移保持。未见明确的C连续接入证据。',
  '局部平顺，靠背错层仍在。'),
 ('459f90aa2162b1f1d46c340938e2ff1c__LEG','座板＋端支撑组合',('PARTIAL','PARTIAL'),'UNCERTAIN','PARTIAL','YES',('YES','YES'),
  '蓝组合为宽座板及弯曲端支撑；其上方仍有另一层灰色长条片，部分灰片也贴近蓝座板内部。前、侧截面都显示多层关系，另一端灰支撑较短。',
  '上方灰条更平但开口很明显，另一灰层仍靠近蓝座板；蓝灰多层关系没有消除。缺口和片状表面使整体连续性仍不确定。',
  '灰条略平整，多层座面及开口保留。'),
 ('459f90aa2162b1f1d46c340938e2ff1c__SEAT','完整矩形座面',('COMPLETE_MACRO','COMPLETE_MACRO'),'NO_VISIBLE_COHERENT_USE','NO','YES',('NO_OBVIOUS_IN_VIEWS','NO_OBVIOUS_IN_VIEWS'),
  '灰体是有厚座面、四腿与高靠背的完整凳体，蓝座板位于灰座面上方的中空区域。表面视角和两种截面均显示蓝板与下方灰座面存在很明显的高度差，没有作为灰座面的连续部分。',
  '主体依然完整且表面较规整；蓝板的悬置层关系基本不变。没有大量小悬片外观，不代表输入已经合理接入。',
  '高背凳主体保持；蓝座板仍明显高于生成座面。'),
 ('9fd5','U形端支撑',('PARTIAL','PARTIAL'),'PLAUSIBLE','YES','UNCERTAIN',('YES','YES'),
  '蓝U形架在凳体一端，灰座面从其内侧向远端延伸，相对位置支持端支撑用途。灰色远端腿较细，蓝架内部有细长灰片，是否共享拓扑不能靠图像确认。',
  '灰座面平顺，远端灰腿变成更细的近线状支撑；蓝架内的部分杂线减少，但不能据此称接续改善。',
  '表面平顺伴随自由端腿变薄。'),
 ('122a480cfcdd742650c626aa72455dae__LEG','一条实心腿',('COMPLETE_MACRO','COMPLETE_MACRO'),'PLAUSIBLE','YES','YES',('UNCERTAIN','UNCERTAIN'),
  '灰体有完整座面、靠背和宽曲面侧支撑；蓝腿上端接近座面下侧，支撑用途位置合理。蓝腿大部分被宽灰支撑面遮挡，侧截面有多条斜向灰线，不能由此直接判定重叠或替代。',
  '主要表面与蓝腿位置保持，前截面新增可见横向灰线穿过蓝腿投影区域；这是截面观察，不直接等于非法穿透。侧部细片与多层支撑关系仍不确定。',
  '宏观形状保持，蓝腿周围关系仍有遮挡和多层不确定性。'),
 ('122a480cfcdd742650c626aa72455dae__SEAT','一条完整座面板条',('COMPLETE_MACRO','COMPLETE_MACRO'),'UNCERTAIN','NO','YES',('NO_OBVIOUS_IN_VIEWS','NO_OBVIOUS_IN_VIEWS'),
  '生成体呈双层平板架／凳体，蓝窄条位于上下两层之间，视觉更像腿间横条。灰色主结构完整且较规整；模型没有部位名输入，因此这种不同用途解释不能单凭来源标签判错，但真实连接仍不确定。',
  '双层形状和中间蓝条保持，部分腿线在截面中略倾斜；未见蓝条变为构造时座面用途，也未见明显大量小悬片。',
  '双层架体保持；中间蓝条用途仍有歧义。'),
 ('20222f467d257022c8bbbd98dee48cb__BACK','开放矩形靠背框',('COMPLETE_MACRO','COMPLETE_MACRO'),'NO_VISIBLE_COHERENT_USE','PARTIAL','YES',('NO_OBVIOUS_IN_VIEWS','NO_OBVIOUS_IN_VIEWS'),
  '灰体为完整高背长凳，蓝矩形背框位于整块灰背板前方。侧截面清楚显示前后偏移，虽然处于靠背区域，仍不像灰背板的连续组成部分。',
  '整体灰体与蓝框关系几乎不变，未见明确采用或连接改善。截面有些线段消失不能单独解释成三维删面孔洞。',
  '蓝框前后错层持续，宏观形状基本不变。'),
 ('20222f467d257022c8bbbd98dee48cb__COMBO','后座板条＋端支撑侧面',('PARTIAL','PARTIAL'),'UNCERTAIN','PARTIAL','UNCERTAIN',('YES','YES'),
  '蓝长条在灰座面后沿附近，楔形蓝片在一端支撑处，用途位置大致合理。侧截面中蓝短段和灰座面端部仍有小间隔，旁边另有灰腿，无法判断是否真正接为整体。',
  '灰背与座面较平，但灰色自由腿明显细化成条线；蓝楔形端片保持。接合间隔没有明确消失，支撑薄弱仍是可见代价。',
  '表面更平，但自由端腿显著变薄。'),
]
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
rows=[];pairs=[]
for page,n in zip(manifest['pages'],notes):
    prefix,part,body,use,role,layer,thin,raw,post,change=n
    assert page['request_id'].startswith(prefix)
    path=GALLERY/(page['page_id']+'.png');ir=read(path.with_suffix('.json'))
    assert sha(path)==ir['image_sha256'] and ir['definition_sha256']==sha(GALLERY/'definition.json')
    for j,stage in enumerate(('RAW','POST')):
        rows.append(dict(review_version=schema['review_version'],reviewer='AI_PROXY',actual_image_viewed=True,human_reviewed=False,nonblind=True,
            reviewed_at_utc=now,label=LABEL,global_step=10000,request_id=page['request_id'],condition_id=page['condition_id'],parent_id=page['parent_id'],
            split=page['split'],stage=stage,K=page['K'],N=page['N'],part_description=part,image_path=str(path),image_sha256=sha(path),
            image_receipt_sha256=sha(path.with_suffix('.json')),gallery_definition_sha256=sha(GALLERY/'definition.json'),
            views_used=['surface_-60_23','surface_120_23','finite_front_section_at_C_bbox_center','finite_side_section_at_C_bbox_center'],
            body_recognizable_as_bench=True,body_completeness=body[j],C_visual_coherent_use=use,construction_intended_role_visible=role,
            visible_offset_or_redundant_layer=layer,thin_or_dangling_sheet_appearance=thin[j],exact_shared_connectivity='NOT_ASSESSED_FROM_IMAGES',
            topologically_disconnected_fragment='UNKNOWN_IMAGE_ONLY',notes=(raw,post)[j],metric_values_used_for_labels=False,source_full_mesh_shown=False,
            scope='Fixed process checkpoint observation; no checkpoint/data/seed selection or training-policy changes'))
    pairs.append(dict(request_id=page['request_id'],split=page['split'],part_description=part,RAW_to_POST=change,image_sha256=sha(path),reviewer='AI_PROXY',actual_image_viewed=True))
def write(name,text):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(text)
write('review.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
write('raw_post_comparison.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs))
write('review_schema.json',json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
md=['# NATIVE_S10000 过程图审（AI_PROXY）','',
 '已逐张实际查看8张固定process图页，各页一次；每页含RAW/POST、两个表面视角与C中心前/侧有限截面。保存16条阶段记录、8条处理前后对照，沿用既有图审字段。非盲AI_PROXY，不是人工评分；没有用数值metric决定标签，也没有读取完整来源款式作为生成答案。','',
 '本轮主体普遍可辨认，但输入采用与整体完整仍然不同。459f90完整蓝座板悬在高背凳的灰座面上方；20222蓝背框仍处于整块灰背板前方。122a窄板位于双层架体中间，其作为横向元素的解释存在歧义，不能仅按来源座面标签判失败。','',
 'POST使部分表面平顺，但9fd5腿架与20222组合请求的自由支撑明显变薄。459f90组合仍有多层座面与开口；截面线段消失或相交投影不自动等于拓扑断片、孔洞或非法穿透。','',
 '|请求（n0）|部位|RAW → POST实际观察|','|---|---|---|']
for p in pairs:md.append('|'+p['request_id'].split('__')[0][:6]+' / '+p['request_id'].split('__')[1]+'|'+p['part_description']+'|'+p['RAW_to_POST']+'|')
md += ['',
 '逐项详见`review.jsonl`和`raw_post_comparison.jsonl`。所有精确共享连接与三维拓扑断片状态继续为NOT_ASSESSED/UNKNOWN；可见用途和宏观完整不认证几何合法，闭合C也不因没有开放接口而被额外惩罚。','',
 '这是10000步普通Native过程观察，不是SEAM终点比较或收敛结论。未新增采样、模型调用、评分、数据审计或生产代码变更，也不据图改训练合同。有限投影及截面无法覆盖全部隐藏表面。','']
write('REVIEW.md','\n'.join(md))
write('receipt.json',json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',reviewer='AI_PROXY',reviewed_at_utc=now,images_viewed=8,image_view_calls_per_page=1,
    stage_rows=16,paired_rows=8,model_calls=0,new_samples=0,rescoring=0,data_audits=0,protected_gallery_definition_sha256=sha(GALLERY/'definition.json'),
    protected_manifest_sha256=sha(manifest_path),files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()}),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',images=8,stage_rows=16,model_calls=0)))
