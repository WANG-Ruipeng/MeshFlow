import csv,json
from r2.tools.cref_layer_audit_v1 import common as c
rows=list(csv.DictReader((c.OUT/'gallery/AI_PROXY_review_template.csv').open(encoding='utf-8-sig')))
refs=list(csv.DictReader((c.OUT/'gallery/source_reference_images.csv').open(encoding='utf-8-sig')))
selected=[]
for r in rows:
 p=r['parent_id'][:6]
 if p not in ['610b16','608b46']:continue
 r.update(review_status='REVIEWED',reviewer='AI_PROXY_ROOT',blinded=False,prior_familiarity='PRIOR_REPORT_AND_LABELS_KNOWN',reviewed_image_sha256s=r['image_sha256s'],whole_body_complete='YES',evidence_sufficient='YES')
 ref=next(x for x in refs if x['parent_id']==r['parent_id'])
 r.update(source_reference_image_path=ref['image_path'],source_reference_image_sha256=ref['image_sha256'])
 if p=='610b16' and r['label']=='SMALL_C200':
  r.update(observed_layer_relation='NO_NEAR_PARALLEL',sidewall_support='YES',remaining_replacement_surface='YES',C_participates_in_seat='NO',visible_local_connection='NO',
   notes='完整生成的灰色座体明显低于蓝色C，前截面在C局部高度只见C；侧截面灰靠背穿过这一高度，低处独立座体在整体图中可见。座体自带厚度不等于C被使用。NO_NEAR_PARALLEL只说明紧邻C的候选为空，不能说明没有替代座面。')
 elif p=='610b16':
  r.update(observed_layer_relation='MIXED',sidewall_support='YES',remaining_replacement_surface='YES',C_participates_in_seat='UNCERTAIN',visible_local_connection='UNCERTAIN',
   notes='座体已升近C。FRONT25/50与SIDE50/75同时可见紧靠C下方的独立灰色上表面、较低底面及侧壁；因此厚度和额外上表面不能合并解释为单一正常底面。RAW橙面到POST变灰是距离阈值变化，独立上表面仍在。C局部是否真正嵌入结构不确定；原采用YES标签保留，本行是更严格的层结构观察。')
 elif r['label']=='SMALL_C200':
  r.update(observed_layer_relation='NORMAL_THICKNESS_SUPPORTED',sidewall_support='YES',remaining_replacement_surface='NO',C_participates_in_seat='YES',visible_local_connection='YES' if r['stage']=='POST' else 'UNCERTAIN',
   notes='蓝色完整座面在侧视近景中暴露，三组前剖面均从C两端向下延伸侧壁，底面位于更低处；未见另一层自由上表面覆盖C。厚座结构合理但不认证全网格无交，旧FF值单独保留。POST有效边只是一个局部接口。')
 else:
  r.update(observed_layer_relation='UNCERTAIN',sidewall_support='UNCERTAIN',remaining_replacement_surface='UNCERTAIN',C_participates_in_seat='UNCERTAIN',visible_local_connection='UNCERTAIN',evidence_sufficient='UNCERTAIN',
   notes='CREF变为包围式扶手椅。固定切面在C高度多见竖壁，橙色近平行片仅在后缘附近，不足以证明大面积冗余上层；全景中的座面与C关系被围板和遮挡影响。不能凭渲染中蓝色不显眼判定替代层，也不能凭低dup判定采用成功。保留不确定。')
 r['limitations']='非盲AI_PROXY，只有六个固定剖面和两视角；侧壁支持指形态证据，不认证C共边、流形、无交或水密；来源款式非唯一GT。'
 for p,h in zip(json.loads(r['image_paths']),json.loads(r['image_sha256s'])):assert c.sha(p)==h
 selected.append(r)
c.csvwrite(c.OUT/'reviews/reviewer_root.csv',selected)
c.write(c.OUT/'reviews/reviewer_root_receipt.json',dict(status='COMPLETE',actual_reviewed_main_images=4,actual_reviewed_source_images=2,cells=8,reviewer='AI_PROXY_ROOT',blinded=False,source_model_inference=0))
print(len(selected))