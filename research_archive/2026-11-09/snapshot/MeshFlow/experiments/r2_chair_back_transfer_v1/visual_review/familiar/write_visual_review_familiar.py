"""Serialize the agent's actual eight-page image observations; no metric reads."""
from pathlib import Path
import csv
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# Entered after viewing all eight original four-column gallery pages, including
# both whole-object views and both finite sections. No scorer table was read.
OBS = {
 '6c36a5b1_BACK_FULL_n0': dict(
    back='YES', complete='YES', reasonable='YES', fragments='NO', gaps='YES', overlays='NO', openings='NA_NO_INPUT_HOLE',
    common='蓝色斜板位于座面后上方，在朝座面的视角可作为靠背；灰色左右侧壁/扶手和底部形成完整箱式椅体。侧向截面中蓝C下缘与灰座面之间仍可见小间隔，不把用途YES解释成已精确接好。',
    raw='RAW侧壁与下部边线稍有零散错层；没有看到大块独立漂浮面。',
    base='D0后的灰色侧壁和下部轮廓更规整；蓝C和座面相对关系基本保留，截面小间隔仍在。',
    hinge='本页尺度下与BASE_POST接近，未见明确用途变化或新大裂口。',
    final='与BASE_POST/HINGE_OPP015的主体轮廓和C用途接近；不能由此图认证接口已闭合。',
    raw_to_base='SLIGHT_IMPROVEMENT', pair='SAME_AT_VIEW_SCALE'),
 '6c36a5b1_BACK_FULL_n1': dict(
    back='YES', complete='YES', reasonable='YES', fragments='UNCERTAIN', gaps='UNCERTAIN', overlays='NO', openings='NA_NO_INPUT_HOLE',
    common='可辨认座面、四向支撑和后背；蓝C承担内侧承靠表面，外围灰色背壳与其平行。截面展示蓝/灰背部双层位置，但图片不足以认定是正常厚度或非法重复层。',
    raw='靠背侧缘有细小尖片/短条，是否独立悬浮不确定；局部截面线段断开不能直接当三维裂口。',
    base='座面边框及背壳较RAW整齐，仍可见少量侧缘尖片；后背用途保持。',
    hinge='与BASE_POST近似，没有清楚的大幅结构变化。',
    final='仍是完整四腿椅外观；细小侧缘结构和双层背壳的合法性保持不确定。',
    raw_to_base='SLIGHT_IMPROVEMENT', pair='SAME_AT_VIEW_SCALE'),
 '8c9b2023_BACK_FULL_n0': dict(
    back='YES', complete='YES', reasonable='YES', fragments='YES', gaps='YES', overlays='NO', openings='NO',
    common='蓝色两侧竖条和横档构成可辨认的条栅靠背，灰座面和腿位于其下。横档之间的大开口在双视图中仍可见，没有观察到被整板封死。靠背下缘与座面的中央视图仍有间隔；不据此判断两侧是否已连接。',
    raw='腿间接近脚底高度有一小灰片可疑悬浮；横档旁存在错位灰边，前截面显示多条近邻边线。',
    base='主体和横档附近灰边稍规整，但小灰片候选仍可见，背下缘间隔保留。',
    hinge='本页尺度下条栅、座面和支撑与BASE_POST近似，开口继续保留。',
    final='没有看到明显新增封孔或主体破坏；小悬片候选没有得到可见消除。其三维独立性UNKNOWN。',
    raw_to_base='SLIGHT_IMPROVEMENT_WITH_REMAINING_COSTS', pair='SAME_AT_VIEW_SCALE'),
 '8c9b2023_BACK_FULL_n1': dict(
    back='YES', complete='YES', reasonable='YES', fragments='NO', gaps='YES', overlays='NO', openings='NO',
    common='蓝色条栅被使用为高背，灰色座面和多根腿形成完整椅子轮廓。横档之间仍有白色贯通视觉开口；灰色斜边局部靠近横档，但未见整体封遮。背下缘中央与座面仍分开，不能从中央切片推出两侧完全断连。',
    raw='座面边线及靠背后侧略碎；双视图中没有像n0那样清楚的脚间小片。',
    base='D0使部分边线收齐，条栅布局、座面高度和腿结构保持。',
    hinge='与BASE_POST相近，没有可确认的大幅视觉收益。',
    final='开口保持且主体没有明显破损；中央间隔并未在本页显示为完全填合。',
    raw_to_base='SLIGHT_IMPROVEMENT', pair='SAME_AT_VIEW_SCALE'),
 'b631b78c_BACK_FULL_n0': dict(
    back='YES', complete='YES', reasonable='YES', fragments='UNCERTAIN', gaps='UNCERTAIN', overlays='YES', openings='NA_NO_INPUT_HOLE',
    common='蓝色弯曲面顺着椅子后背竖起，灰色座面和细腿构成合理的完整外形。背部灰条与蓝C局部重叠可见；侧截面灰/蓝曲线存在厚度间隔，不能仅凭图断言非法双层。输入本身是连续曲面，不把灰条间白线当输入镂空。',
    raw='背部反向视角有较密碎细灰条，截面侧缘也较断续；可能是近表面薄片，三维浮离关系UNKNOWN。',
    base='灰色背面更连贯、座下轮廓更规整；前视蓝C上缘仍有横向灰条遮挡，局部可见线状缝隙。',
    hinge='与BASE_POST近似；不能从本页确认细缝或薄片已经消失。',
    final='曲面靠背采用和四腿外观保持；细条遮挡及背面厚度关系仍需几何证据，未见明显新增大损伤。',
    raw_to_base='SLIGHT_IMPROVEMENT_WITH_REMAINING_COSTS', pair='SAME_AT_VIEW_SCALE'),
 'b631b78c_BACK_FULL_n1': dict(
    back='YES', complete='YES', reasonable='UNCERTAIN', fragments='YES', gaps='UNCERTAIN', overlays='YES', openings='NA_NO_INPUT_HOLE',
    common='蓝C仍位于座面后方并承担靠背表面；主体成为厚实箱式/扶手式椅体。箱式款式本身不算失败，但蓝靠背上方和表面横跨多条细灰斜片、下缘灰片接入关系杂乱，整体是否合理因此记UNCERTAIN。不能用与来源四腿款式不同判失败。',
    raw='蓝C上方有多层细片，右上角有细尖片，前截面存在额外斜线；这些是可见杂片候选，不认证其独立连通性。',
    base='后侧大灰面变得规整，前侧部分细条减少；仍有灰斜带覆盖蓝C与右上细片，厚箱体并未因此获几何合法认证。',
    hinge='与BASE_POST接近，未观察到明确的细片清除或用途改变。',
    final='蓝色曲面靠背保持，但遮挡细片和不明确的下缘接入仍在；整体合理性继续UNCERTAIN。',
    raw_to_base='SLIGHT_IMPROVEMENT_WITH_REMAINING_COSTS', pair='SAME_AT_VIEW_SCALE'),
 'f4f1aba6_BACK_FULL_n0': dict(
    back='YES', complete='YES', reasonable='YES', fragments='NO', gaps='UNCERTAIN', overlays='NO', openings='NA_NO_INPUT_HOLE',
    common='完整蓝色矩形面直接成为高靠背内侧，灰色座面、短支撑和顶端装饰形成可辨认完整椅子；蓝面没有被另一块前侧灰背板全面替代。侧截面存在灰色背壳厚度，不能从图认证共边或无自交。',
    raw='整体已清楚；座下截面灰线较零散，背侧边有细小不齐。',
    base='座下截面形成更完整规整的轮廓，灰色边框更顺；靠背采用保持。',
    hinge='与BASE_POST的双视图和截面近似。',
    final='没有看到明显新增裂口、悬片或蓝C遮挡；不将图像整齐当水密认证。',
    raw_to_base='SLIGHT_IMPROVEMENT', pair='SAME_AT_VIEW_SCALE'),
 'f4f1aba6_BACK_FULL_n1': dict(
    back='YES', complete='YES', reasonable='UNCERTAIN', fragments='UNCERTAIN', gaps='UNCERTAIN', overlays='YES', openings='NA_NO_INPUT_HOLE',
    common='蓝矩形上半部分明确处于椅背位置，座面和腿也存在；但是灰座面接入C的位置较高，蓝C下端延伸到座下。它不是完全没有采用C，但不能忽略下端悬露和座背接入比例问题。',
    raw='座面左后侧有灰色折片跨到蓝C前方，前截面灰座高度位于蓝面下半段之内，背壳上下端边线较杂。',
    base='D0把灰座和背壳轮廓收齐，但蓝C伸到座下的关系保留；不能把收齐边线等同正确使用了整块C。',
    hinge='与BASE_POST近似，蓝色座下尾部没有明显改善。',
    final='仍能认作高背椅，但整块输入被自然采用的程度存疑，采用YES只指其可辨认靠背角色；整体合理性UNCERTAIN。',
    raw_to_base='SLIGHT_IMPROVEMENT_WITH_REMAINING_COSTS', pair='SAME_AT_VIEW_SCALE'),
}


def main():
    gallery = ROOT/'gallery/S2000_BACK_FULL'
    done = read(gallery/'completion.json')
    manifest = read(ROOT/'eval_manifest.json')
    requests = {q['request_id']:q for q in manifest['requests'] if q['split']=='FAMILIAR'}
    assert set(requests)==set(OBS) and len(done['images'])==16
    images={r['request_id']:r for r in done['images']}
    records=[];pairs=[]
    for rid in sorted(OBS):
        q,o=requests[rid],OBS[rid];im=images[rid]
        assert sha(im['path'])==im['image_sha256']
        assert sha(gallery/'definition.json')==done['definition_sha256']
        for stage,key in [('RAW','raw'),('BASE_POST','base'),('HINGE_OPP015','hinge'),('FINAL_POST','final')]:
            records.append(dict(request_id=rid,condition_id=q['condition_id'],parent_id=q['parent_id'],split=q['split'],noise_index=q['noise_index'],stage=stage,
                reviewer='AI_PROXY',blind=False,image_path=im['path'],image_sha256=im['image_sha256'],actual_viewed=True,
                back_role_adoption=o['back'],body_complete=o['complete'],overall_reasonable=o['reasonable'],visible_fragment_or_sliver_suspect=o['fragments'],
                visible_gap_or_split=o['gaps'],local_gray_overlay_on_C=o['overlays'],input_opening_filled_or_blocked=o['openings'],
                exact_topological_connection='UNKNOWN',intersection_legal='UNKNOWN',watertight='UNKNOWN',
                observations=o['common']+' '+o[key],section_seen=True,opposing_views_seen=True))
        pairs.append(dict(request_id=rid,RAW_to_BASE_visual_change=o['raw_to_base'],BASE_to_FINAL_visual_change=o['pair'],
                          BASE_to_FINAL_back_role='UNCHANGED',BASE_to_FINAL_obvious_new_body_damage='NO',
                          note='本页尺度下未见显著BASE→FINAL变化，不等同坐标未改、指标未改或零退化。'))
    result=dict(schema='chair_back_transfer_actual_visual_review_v1',status='COMPLETE_ACTUAL_VIEW',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        reviewer='AI_PROXY',blind=False,split='FAMILIAR',requests=8,stages_per_request=4,observations=records,paired_observations=pairs,
        evidence=dict(manifest_path=str(ROOT/'eval_manifest.json'),manifest_sha256=sha(ROOT/'eval_manifest.json'),
                      gallery_manifest_sha256=sha(ROOT/'gallery_manifests/S2000_BACK_FULL.json'),completion_sha256=sha(gallery/'completion.json'),definition_sha256=done['definition_sha256']),
        policy=dict(metrics_read_for_labels=False,source_style_match_required=False,
                    adoption='YES means a visually identifiable back role, not complete C usage, correct junction, uniqueness, or no occlusion.',
                    observation_scope='Two opposing surface views plus fixed front/side finite sections per stage; absence means not clearly visible, not certified absent.',
                    topology='Exact welding, connectedness, water-tightness and illegal face-pair classification are UNKNOWN from images.',
                    prior_exposure='These source parents were in TRAIN47; BACK_FULL may have been seen by EXP67. This is not an unseen-C guarantee.',
                    supplemental_input_images='Existing b631 and 8c source/C pages viewed to distinguish continuous panel from rung openings; source style was not used as target.'))
    with (OUT/'visual_review_familiar.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    with (OUT/'visual_review_familiar.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    with (OUT/'visual_review_familiar_pairs.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    print(json.dumps(dict(status=result['status'],requests=8,observations=len(records),pairs=len(pairs),review_json_sha256=sha(OUT/'visual_review_familiar.json'))))


if __name__=='__main__':main()
