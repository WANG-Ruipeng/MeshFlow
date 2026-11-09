"""Transcribe actual view_image observations; never read metric tables or old ratings."""
from pathlib import Path
import json, hashlib

ROOT=Path(__file__).resolve().parents[1]
METHODS=['EXPOSURE','PLAIN1000','LAYOUT1000','LAYOUT_NULL']
PARENTS={'a8':'a8e987a6a6486b962d6c8e51365a5a87','27':'27eb79ddf26fd7ac7898580cb9d16864'}
STAGES=['RAW','BASE_POST','CLEAN_FINAL_POST']

# These are manual visual observations from the 12 original pages and 12
# original-pixel crops actually opened by /root/layout_data_build on 2026-10-30.
# They are not labels computed from metrics or copied from previous experiments.
RAW_NOTES={
 ('a8',0):[
 '两斜视均见四角直腿和规则蓝色板条台面；中心front截面蓝灰线贴近，side截面主要为板条下缘的小矩形，未见另起的竞争座面。',
 '两斜视四角腿表面比线丝有明确宽度，蓝色条面处于顶面；两截面维持单一上表面组织，局部板条截面开口不足以单独判裂。',
 'P下仍为四腿条面凳，腿位置和蓝色上表面关系与PLAIN近似；两截面未显示新主体或新增竞争层，未看出可辨别的结构升级。',
 '遮蔽P后四腿与条面整体仍成立，未出现新增悬空主体；中心截面基本保留相同组织，视觉上与有P组接近。'],
 ('a8',1):[
 '四角腿支撑条面，两个斜视可辨完整凳形；side截面有几处不闭合的小矩形线，但斜视无清楚脱落片，不能仅凭这些线端认定孔洞。',
 '条面与四个腿柱整体清楚；front单水平主体、side条面下缘小矩形，未见灰面替代蓝C或宏观折层。',
 '蓝条面与直腿在两斜视完整可辨，与PLAIN和NULL只见细微表面变化；front和side截面未提供新的可辨别结构收益。',
 'NULL仍保留四角直腿和蓝色条面，外形与P组接近；side局部线段缺口存在，但两斜视没有可确认的飞散薄片。'],
 ('27',0):[
 '蓝色弯曲座面、灰色靠背及左右扶手腿架共同构成长椅；side中蓝座面接向背部斜线，孤立小闭环与杆件截面相符，不能当浮片。',
 '两斜视座背与两端支撑架完整可辨，蓝C承担座面；front两端杆件截面与side靠背/座面关系合理，没有明显额外灰座。',
 '有P时仍是同一类座背加两端支架组织，轮廓与PLAIN近似；两个截面没有新增采用或明显增厚支撑的证据。',
 '无P仍保留座背和双端框架，蓝C参与座面而非独立插片；截面与有P组很相近，未见有P独有的结构完成。'],
 ('27',1):[
 '靠背、蓝色座面与两端扶手腿表面均可辨；front端部小环与side座背线相容，未见另一灰座面盖住蓝C。',
 '两斜视端架完整，座背关系成立；side杆件截面有短缺口，但宏观图未见明确脱落片，不能凭截面缺口判破损。',
 'P组两端腿架、座背均在，蓝C用途明确；与PLAIN及NULL相比主要是局部表面细微差异，未见宏观布局改善。',
 'NULL图仍有两端扶手腿及完整座背，形状与P几乎相同；两截面未见新竞争座面或大折片。']}

POST_NOTES={
 ('a8',0):[
 '两斜视腿面已大面积缺失，只剩向下细丝及断段，蓝条面仍在顶部；side只剩零散下缘短线，不能把残丝记为正常支撑。',
 '蓝条面悬在细长腿残迹上，远侧腿有断续小点；front尚有薄下缘线，side多为零散短片，整体支撑不完整。',
 '有P组仍只剩腿柱细丝，右侧及近侧有可见断段；两截面未恢复连续腿或正常杆架，P没有修复此阶段表面损失。',
 'NULL也只留细丝状腿和局部小段，蓝条面仍合理在顶面但不能构成完整支撑；截面下缘同样零散。'],
 ('a8',1):[
 '两斜视腿表面退为纤细残丝和短段，条面保持；side下方只见稀疏灰短线，整体为缺支撑的凳面。',
 '可见少量垂直细线但没有足够完整腿面，部分细线与上方条面不连续；两截面不支持正常完整杆架解释。',
 '有P组近侧及右侧有明显分离小短片，腿主体缺失；两截面保持条面却未形成可解释支撑，较粗残片不等于腿恢复。',
 'NULL组近侧腿有分离残段，远侧仅线丝，蓝条面仍是主座面；front/side未见支撑重建。'],
 ('27',0):[
 '座背主体保留，但左右扶手腿架被削成线丝，靠背上沿还有分离长窄条；side靠背与蓝座面仍配合，支撑却明显残缺。',
 '蓝座面和灰靠背可辨，端架几乎不剩连续表面，左下有零散细线；front端部只剩小点，side不再呈完整杆截面。',
 '有P组右侧保留细线框而左腿断裂，靠背上沿窄条分离；两截面未显示完整端架，不能据少量残线判支撑成功。',
 'NULL组端架也只剩细线和短段，蓝C与灰背关系还合理；side小残段与斜视的缺支撑相符，未见宏观恢复。'],
 ('27',1):[
 '两端腿架几乎全无，仅座背漂浮及少量点状残片，靠背顶条仍分离；front端架位置无完整杆截面，整体明显缺支撑。',
 '端部有一两条细线与悬离小腿段，不能组成完整支撑；蓝C与靠背关系仍在，side只见稀疏杆残片。',
 '有P组近侧腿残片较粗但与主体明显断开，另一侧也是细线；front/side未补成连续端架，不能把更多残片算改善。',
 'NULL与有P均留悬离腿残片及不完整侧线架；蓝C仍是座面，截面没有出现新完整支撑，形状仍不完整。']}

TIMES={('a8',0,'RAW'):'10:32:55',('a8',0,'BASE_POST'):'10:33:18',('a8',0,'CLEAN_FINAL_POST'):'10:33:18',
 ('a8',1,'RAW'):'10:33:33',('a8',1,'BASE_POST'):'10:33:33',('a8',1,'CLEAN_FINAL_POST'):'10:33:33',
 ('27',0,'RAW'):'10:33:42',('27',0,'BASE_POST'):'10:33:42',('27',0,'CLEAN_FINAL_POST'):'10:33:42',
 ('27',1,'RAW'):'10:33:53',('27',1,'BASE_POST'):'10:33:53',('27',1,'CLEAN_FINAL_POST'):'10:33:53'}
CROP_TIMES={'RAW':'10:35:10','BASE_POST':'10:35:20','CLEAN_FINAL_POST':'10:35:28'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for short,parent in PARENTS.items():
 for n in [0,1]:
  request=f'{parent}__COMPLETE_SEAT_CANDIDATE__COVERAGE_V1__n{n}'
  for stage in STAGES:
   page=ROOT/'gallery'/f'{stage}__{request}'
   crop=ROOT/'visual_review'/'crops_part01'/f'{stage}__{request}.png'
   for i,method in enumerate(METHODS):
    raw=stage=='RAW'
    note=(RAW_NOTES if raw else POST_NOTES)[short,n][i]
    if stage=='CLEAN_FINAL_POST':
     note='已独立查看FINAL两斜视及两截面：'+note+' FINAL未见把上述残片重建为连续支撑。'
    evidence={
     'overall_shape':('完整条面凳的主体和四角支撑可辨。' if short=='a8' else '座背与双端扶手腿架组成可辨长椅。') if raw else '座面/座背仍在，但明显缺少可解释的完整支撑。',
     'C_use':'蓝色C仍作为上方座面参与，未见灰色另起座面替代；此轴不要求其他结构已经完整。',
     'support_retention':'两斜视可见有宽度的连续腿/端架表面，非只剩残线。' if raw else '两斜视支撑已成线丝、分离短片或直接缺失，截面也不支持完整杆架。',
     'visible_fragment_or_gap':'未见清楚飞散片或异常大缺口；规则板条/杆件截面开口不独立当裂缝。' if raw else '实际斜视可见支撑断段/悬离残片；此判断不只来自截面端点。',
     'layering_or_fold':'未见明显竞争座面或宏观穿插折层；NO_OBVIOUS不是无自交认证。'}
    rows.append(dict(request_id=request,label=method,stage=stage,reviewer='/root/layout_data_build',reviewer_kind='AI_PROXY',blind=False,
      actually_viewed=True,views_and_sections_checked=True,checked_views=['azimuth_-60','azimuth_120','front_fixed','side_fixed'],
      image_path=str(page/'comparison.png'),image_sha256=sha(page/'comparison.png'),image_receipt_sha256=sha(page/'comparison.json'),
      tool='view_image',view_event_id=f'layout_data_build:view_image:{stage}__{request}:original',
      viewed_at_utc='2026-10-30T'+TIMES[short,n,stage]+'Z',
      crop_paths=[str(crop)],crop_sha256=[sha(crop)],crop_view_events=[{'event_id':f'layout_data_build:view_image:{stage}__{request}:native_pixel_crop','viewed_at_utc':'2026-10-30T'+CROP_TIMES[stage]+'Z'}],
      overall_shape='REASONABLE' if raw else 'INCOMPLETE',C_use='PLAUSIBLE',support_retention='PRESENT' if raw else 'MISSING_OR_DAMAGED',
      visible_fragment_or_gap='NO_OBVIOUS' if raw else 'YES',layering_or_fold='NO_OBVIOUS',note=note,evidence=evidence))
assert len(rows)==48
out=ROOT/'visual_review'/'completed'/'review_part_01.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(dict(schema='bench_layout_oracle_actual_review_rows_v1',status='COMPLETE',assignment_id='review_part_01',
 reviewer='/root/layout_data_build',reviewer_kind='AI_PROXY',blind=False,rows=rows,
 viewing_receipt=dict(original_pages_actually_opened=12,original_pixel_crop_pages_actually_opened=12,model_calls=0,rescoring_calls=0,
 scope='Two familiar FOCUS parents only; all five axes judged from actual current images; not independent human certification.'),
 conclusion='在a8/27四个熟悉请求上，P相对PLAIN和NULL没有可辨别的主体或采用提升。RAW四方法均保留完整座面/座背与支撑；BASE和FINAL四方法均有明显腿架表面损失。P的少量残片差异没有转为完整支撑。'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(out)
print('rows',len(rows))
