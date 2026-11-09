"""Save actual image review of only the Native cells in eight joint-panel pages."""
import datetime
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
LABEL='NATIVE_S13000'
KEY='NATIVE_S13000__SEAM_S13000_process'
GALLERY=ROOT/'gallery'/KEY
PRIOR=ROOT/'visual_review/NATIVE_S10000_process'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest_path=ROOT/'gallery_manifests'/(KEY+'.json')
manifest=read(manifest_path)
completion=read(GALLERY/'completion.json')
assert len(manifest['pages'])==len(completion['images'])==8
assert completion['definition_sha256']==sha(GALLERY/'definition.json')
schema=read(PRIOR/'review_schema.json')
schema.update(checkpoint=LABEL,actual_view_count=8,prior_schema_sha256=sha(PRIOR/'review_schema.json'),
    scope='Eight fixed process requests; Native cells only in the joint Native/SEAM gallery;16 RAW/POST observations')
schema['limitations'][-1]='Eight process requests at13000 are not the full panel, a convergence conclusion, or a final arm comparison.'
schema['limitations'].append('Native observations are nonblind: SEAM cells were visible on the same pages but were not scored here.')
notes=[
 ('2d1e5eeb255e79dcab53bf7fa22f3636__BACK_SURFACE_1__n0','靠背板',
  ('COMPLETE_MACRO','COMPLETE_MACRO'),'UNCERTAIN','YES','YES',('UNCERTAIN','YES'),
  '灰座面、背部和弯曲裙板形成可辨认的凳体；蓝板位于靠背区域，但灰背层在其后方。侧截面可见蓝灰分层，底部是宽曲面结构，局部边缘较不规整。',
  '主要形状保持，灰背与蓝板仍有间距；右端出现更明显三角开口和细片边缘。前截面更平，不能由部分线段消失认定三维破洞或断片。',
  '主体保持，背部错层未消失，局部端面更薄。'),
 ('459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n0','座板加弯曲端支撑',
  ('PARTIAL','PARTIAL'),'UNCERTAIN','PARTIAL','YES',('YES','YES'),
  '蓝色宽座板及一端弯曲支撑上方，另有带三角开口的灰长薄板；另一端灰支撑较短。两种截面显示灰板高于蓝座板，部分灰线又贴近蓝板内部。',
  '上层灰板更平但条片和开口仍明显，蓝灰两层的高度差保持；远端灰支撑也较薄。不能仅凭条片外观确定它们是否拓扑独立。',
  '灰板略平整，多层座面和开口持续。'),
 ('459f90aa2162b1f1d46c340938e2ff1c__SEAT_SURFACE_1__n0','完整矩形座面',
  ('COMPLETE_MACRO','COMPLETE_MACRO'),'NO_VISIBLE_COHERENT_USE','PARTIAL','YES',('NO_OBVIOUS_IN_VIEWS','NO_OBVIOUS_IN_VIEWS'),
  '灰体形成带开放靠背、座框和端支撑的长凳；蓝板处在座面区域但高于现成灰座面。前、侧截面均显示独立的蓝水平层与下方灰层。',
  '宏观凳体保持，灰表面略平；蓝板仍高于灰座面，未见普通臂生成表面向其形成连续座面。',
  '主体完整但蓝灰座面高度差持续。'),
 ('9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n0','U形端支撑',
  ('COMPLETE_MACRO','PARTIAL'),'PLAUSIBLE','YES','UNCERTAIN',('YES','YES'),
  '灰座板从蓝U形端架向另一端延伸，整体相对位置支持端支撑用途；远端两条灰腿明显比蓝腿纤细。侧截面灰座板贴近蓝横梁下方，但共享连接仍不能从图像确定。',
  '灰座板更平，远端灰腿进一步变成近线状细支撑；蓝架内的小杂线减少，但没有足够图像证据说明接续更牢固。',
  '端架用途保持，POST显著削薄另一端自由腿。'),
 ('122a480cfcdd742650c626aa72455dae__LEG_SUPPORT_1__n0','单条实心支撑腿',
  ('COMPLETE_MACRO','COMPLETE_MACRO'),'PLAUSIBLE','YES','YES',('UNCERTAIN','YES'),
  '灰体有完整高背、座面及宽曲面侧支撑；蓝腿上端靠近座面下方，支撑位置合理，但大部分被灰侧板遮挡。侧截面有灰曲线穿过蓝腿投影，不能由投影直接判非法穿透。',
  '主要形状和蓝腿位置保持，右侧灰面边缘更细且出现齿状/缺口外观；蓝腿与宽灰侧板的多层关系仍在。',
  '宏观结构保持，蓝腿周围遮挡与多层关系持续。'),
 ('122a480cfcdd742650c626aa72455dae__SEAT_SURFACE_1__n0','单条完整座面板条',
  ('COMPLETE_MACRO','COMPLETE_MACRO'),'UNCERTAIN','NO','YES',('NO_OBVIOUS_IN_VIEWS','NO_OBVIOUS_IN_VIEWS'),
  '灰体为上下两层平板和四腿组成的架式凳体，蓝窄条位于两层之间，未成为任一灰水平面的一部分。其作为横向构件的解释存在歧义，不能只按来源座面角色判错。',
  '双层结构和中间蓝条保持，下层灰面更平；前截面部分线段消失并不直接意味着三维外壳缺失。蓝条用途仍不明确。',
  '双层架体保持，中间蓝条仍未呈现构造时的座面用途。'),
 ('20222f467d257022c8bbbd98dee48cb__BACK_SURFACE_1__n0','开放矩形背框',
  ('COMPLETE_MACRO','COMPLETE_MACRO'),'NO_VISIBLE_COHERENT_USE','PARTIAL','YES',('NO_OBVIOUS_IN_VIEWS','NO_OBVIOUS_IN_VIEWS'),
  '灰体为完整高背、厚座与四腿长凳，蓝矩形框位于整块灰背板前方。侧截面清楚显示两者前后位置不同，未见蓝框成为灰背连续组成部分。',
  '高背凳与蓝框的相对位置基本不变，框后仍有完整灰背板。部分截面线段变短不能代替三维删面判断。',
  '蓝框前后错层持续，宏观形状基本不变。'),
 ('20222f467d257022c8bbbd98dee48cb__COMBO_SEAT_SUPPORT__n0','后座板条加楔形端支撑',
  ('COMPLETE_MACRO','PARTIAL'),'UNCERTAIN','PARTIAL','UNCERTAIN',('UNCERTAIN','YES'),
  '蓝长条位于灰座面后沿附近，楔形蓝片处于一端支撑区，位置大致合理；旁边另有灰腿，侧截面显示蓝短条与灰座板端部并未明显合为一条连续轮廓。',
  '灰背和座面更平，但两端自由腿显著缩成细线，蓝楔形片保持。侧截面的局部间隙没有清楚消失，不能据此认证实际接续。',
  '表面较平，代价是自由腿明显变薄；局部接合仍不确定。'),
]
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
rows=[];pairs=[]
for page,n in zip(manifest['pages'],notes):
    rid,part,body,use,role,layer,thin,raw,post,change=n
    assert page['request_id']==rid
    assert [c['label'] for c in page['cells']][:2]==[LABEL+'_RAW',LABEL+'_POST']
    path=GALLERY/(page['page_id']+'.png');ir=read(path.with_suffix('.json'))
    assert sha(path)==ir['image_sha256'] and ir['definition_sha256']==sha(GALLERY/'definition.json')
    for j,stage in enumerate(('RAW','POST')):
        rows.append(dict(review_version=schema['review_version'],reviewer='AI_PROXY',actual_image_viewed=True,human_reviewed=False,nonblind=True,
            reviewed_at_utc=now,label=LABEL,global_step=13000,request_id=rid,condition_id=page['condition_id'],parent_id=page['parent_id'],
            split=page['split'],stage=stage,K=page['K'],N=page['N'],part_description=part,image_path=str(path),image_sha256=sha(path),
            image_receipt_sha256=sha(path.with_suffix('.json')),gallery_definition_sha256=sha(GALLERY/'definition.json'),
            views_used=['surface_-60_23','surface_120_23','finite_front_section_at_C_bbox_center','finite_side_section_at_C_bbox_center'],
            reviewed_cell_label=LABEL+'_'+stage,other_model_cells_visible=True,
            body_recognizable_as_bench=True,body_completeness=body[j],C_visual_coherent_use=use,construction_intended_role_visible=role,
            visible_offset_or_redundant_layer=layer,thin_or_dangling_sheet_appearance=thin[j],exact_shared_connectivity='NOT_ASSESSED_FROM_IMAGES',
            topologically_disconnected_fragment='UNKNOWN_IMAGE_ONLY',notes=(raw,post)[j],metric_values_used_for_labels=False,source_full_mesh_shown=False,
            scope='Fixed process checkpoint Native-only observation; no checkpoint/data/seed selection or training-policy changes'))
    pairs.append(dict(request_id=rid,split=page['split'],part_description=part,RAW_to_POST=change,image_sha256=sha(path),reviewer='AI_PROXY',actual_image_viewed=True))
def write(name,text):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(text)
write('review.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
write('raw_post_comparison.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs))
write('review_schema.json',json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
md=['# NATIVE_S13000 固定过程图审（AI_PROXY）','',
 '实际逐页查看8张联合图页，每页仅对左两列NATIVE_S13000_RAW/POST作本记录，含两个表面视角及固定前/侧截面。SEAM列在页面上可见，因此这是非盲审阅；本记录不作两臂胜负判定。保存16条阶段观察与8条前后对照，没有读取数值metric或完整来源网格。','',
 '宏观凳体多数可辨认，但C位置接近不等于自然采用：459f完整座板仍高于灰座面，20222背框仍位于整块灰背板前方。U形端架及单腿输入的位置较合理，真实连接仍未知。POST有时让表面更平，同时明显削薄自由腿或端部表面。','',
 '|请求|部位|RAW → POST|','|---|---|---|']
for p in pairs:md.append('|'+p['request_id'][:6]+'/'+p['request_id'].split('__')[1]+'|'+p['part_description']+'|'+p['RAW_to_POST']+'|')
md+=['','共享索引连接全部NOT_ASSESSED；拓扑断片全部UNKNOWN_IMAGE_ONLY。截面短线、空缺或投影交叉不直接认证断片、孔洞或非法穿透。此8请求过程面板不用于挑checkpoint、更改训练或宣布收敛。','']
write('REVIEW.md','\n'.join(md))
write('receipt.json',json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',reviewer='AI_PROXY',reviewed_at_utc=now,
    images_viewed=8,image_view_calls_per_page=1,reviewed_model=LABEL,source_gallery_key=KEY,reviewed_cell_indices=[0,1],
    stage_rows=16,paired_rows=8,model_calls=0,new_samples=0,rescoring=0,data_audits=0,
    protected_gallery_definition_sha256=sha(GALLERY/'definition.json'),protected_manifest_sha256=sha(manifest_path),
    files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',images=8,stage_rows=16,model_calls=0)))
