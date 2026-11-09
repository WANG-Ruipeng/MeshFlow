"""Actual nonblind image observations, manually recorded after viewing56 PNGs.

No metric table is read; observations are not inferred from score changes.
"""
from collections import Counter
from pathlib import Path
from .evaluation import C,OUT,requests

# Each tuple: RAW->D0, RESTORED->POST, relative visual judgment, paired POST note.
NOTES={
'73b369b9_n0':(
 'D0将座下和靠背边的细碎锯齿整理为较直轮廓；座下长水平线与两端的间隔已在RAW可见。',
 '恢复C后座下长线仍分开；POST侧剖的靠背根部和前沿下方间隔保留，底座斜白缝仍可见。',
 'WORSE','相对S2000，SMOOTH底座在背视角出现更明显的斜向白缝，front座下两端也更分离；外部主体仍完整。CLEAN底座没有同样大的斜缝。'),
'73b369b9_n1':(
 'RAW已生成四腿及高侧板、斜三角面；D0减少边缘短碎线，但没有改变这些大面布局。',
 '恢复后蓝C下沿与灰色水平层分离；POST保留高侧板、斜面和座层附近的小间隔。',
 'MIXED','S2000的悬臂板式椅变成四腿且带大侧板的椅子；座面附近灰层在C上方，局部围挡明显。款式改变本身不判错，但没有清晰的全面质量提升。CLEAN也改了下部款式。'),
'73b369b9_n2':(
 'D0使多边形背板边缘和座下轮廓更整齐；RAW的主体构成已完整。',
 'POST的front座下框大致连续；side后根与下层之间仍有间隔，未观察到大块新缺损。',
 'IMPROVED','相对S2000及CLEAN，SMOOTH背板上部的长斜白开口不再可见，front下层小断点减少；side座下与前后根部仍有间隔。'),
'73b369b9_n3':(
 'D0整理RAW背板顶边和底座细裂线；座下双轮廓关系保持。',
 'POST侧剖前端有小分离斜线及较大断口，背板顶部小白开口仍在。',
 'WORSE','S2000背板顶部较完整；SMOOTH新增/扩大可见顶端白开口，side前端小分离线更显眼。总体椅形保留。'),
'3f741759_n0':(
 'D0减少背条的锯齿与细缝，side靠背竖向截线变得较连续；背条内侧薄斜条没有全部消失。',
 'POST保留front座下左端缺口以及右端未到C的间隔；侧剖底线更平整。',
 'MIXED','SMOOTH的side上方独立短横线在本平面中改为靠背竖截线，但front右上端比S2000更不连续；薄背条仍有扭斜。不能将截线变化说成旧悬片被修复。'),
'3f741759_n1':(
 'D0令座下底线及腿部轮廓变直，RAW细碎面线减少。',
 'POST的front左下角仍有可见断口，side下层保持大致连续。',
 'MIXED','SMOOTH补上S2000的front右上缺口，但出现更明显的左下缺口；外观主体近似，不能只选一个角说改善。'),
'3f741759_n2':(
 'RAW的side上方孤立短横截线、背部细斜条在D0后仍在；D0减少座下的细碎线。',
 '恢复及POST都保留上方短横线；front座下框更连续，背部斜薄条仍明显。',
 'IMPROVED','相对S2000，SMOOTH front座下左角缺口闭合；side上方短横线仍在，不能称碎片消失。CLEAN该例front/side下层缺口更大；SMOOTH只记局部可见改善。'),
'3f741759_n3':(
 'D0把RAW座框的小阶梯和靠背细裂线整理得较直，未改变完整椅形。',
 'POST front框和side下层较连贯，未观察到新的大块外壳缺失。',
 'IMPROVED','S2000 front右上缺口在SMOOTH中闭合，外观及side基本维持；这是局部可见改善，不认证水密。'),
'85b16941_n0':(
 'D0压平RAW座下杂乱短线；本来存在的近C水平灰层与更低两端短竖线保持。',
 'POST front与side都仍显示近C灰层和两端分开的下折线；完整背板与四腿保留。',
 'WORSE','SMOOTH改成更薄座框和较低背板，front/side由S2000较连续的厚框变成多段分开的轮廓，未见广泛接缝修复。款式薄厚本身不判错，分离线是局部负面观察。'),
'85b16941_n1':(
 'D0使RAW座框底线更连续，背板下方正常空档与四腿保持。',
 '恢复C后front左侧及右上端的小间隔仍在；POST没有填补这些间隔。',
 'WORSE','相对S2000，SMOOTH front左侧断口更明显并有右上端间隔；side前端也更分离。完整椅形仍合理。'),
'85b16941_n2':(
 'D0整理背板和座框内部细线，RAW的完整板式椅结构保持。',
 'POST seat框整体连续，front底线有细小不连续，未见明显大破损。',
 'UNCERTAIN','三模型外观接近，SMOOTH座框与背板略变形；在本尺度没有足够清楚的整体质量增量或严重退化。'),
'85b16941_n3':(
 'RAW的座下楔形白缺口已明显存在；D0整理边线但没有将其补齐。',
 'RESTORED/POST均保留座下楔形空隙，front左下长缺口与side斜白口可见。',
 'WORSE','相对S2000仅有较小右下断点，SMOOTH front左下开口更大；切开图可见该缺口来自RAW而非POST才出现。'),
'8c9b2023_n0':(
 'D0使RAW梯状背条更规整，并让side座框较闭合；front下层的多处间隔仍在。',
 'POST侧框保持较连续，但front下层与两侧分离，背条间仍有细斜片。',
 'MIXED','相对S2000，SMOOTH side右侧开口闭合；front却增加断点并失去两侧竖向延伸。外部大形未坏，但不能合并记成净视觉胜利。'),
'8c9b2023_n1':(
 'D0减少背条内部的白缝；RAW side的C上方小短线及左下大开口保留。',
 'POST front下层仍有右侧断点，side的短线与左侧开放状态保持。',
 'WORSE','S2000 front座下较连续，SMOOTH多了右侧下层断点；side仍未闭合且C上方小短线可见。不能仅凭小短线认证浮片。'),
'8c9b2023_n2':(
 'D0整理背条与座框细线；上方一条短横截线以及front左侧分段线从RAW保留。',
 'POST front左下多处缺口和side左侧开口均保留，背条内斜白缝没有全消。',
 'WORSE','相对S2000，SMOOTH front左边更碎、side左端由闭合变成开口；上方短线可能来自背条切面，不据此断言离体。'),
'8c9b2023_n3':(
 'D0改善RAW背条表面细裂线，但front座下两侧与下层的分离关系没有消失。',
 'POST仍有多个front下层小断点；side左侧开口保持，未见整体结构塌失。',
 'WORSE','SMOOTH front下层比S2000多一处可见分段；外观近似，属于局部裂线恶化而非整椅崩坏。'),
'2d08a64e_n0':(
 'D0整理RAW悬臂框架边缘，seat下方交叉斜截线和近双层横线仍在。',
 '恢复及POST继续保留seat下方交叉线；下部框架仍完整，未见直接把这些关系消除。',
 'MIXED','SMOOTH由较直立背板变成低斜背，整体可辨且完整；side座下斜线相交更明显，front双层线仍在。与CLEAN一样有款式变化，不能据外壳平整判断内部无交。'),
'2d08a64e_n1':(
 'D0使RAW箱式下部和背板边缘较直；front只有半段座下横线的结构已存在。',
 'POST保留front半段底线及side座下开口；前端C下方的尖斜边仍在。',
 'MIXED','相对S2000板式悬臂，SMOOTH变成短腿/条板式；front下层缺失范围较大，但大款式也不同。未能确定是更好整体完成。'),
'2d08a64e_n2':(
 'RAW已是宽侧板扶手箱式椅；D0减少细缝，四阶段大结构稳定。',
 'POST仍保留C前沿下伸的尖边和侧板；中央截面没有底层横线，不能因此把完整箱式外观判作主体缺失。',
 'UNCERTAIN','三模型分别形成框架、悬臂和箱式扶手椅。SMOOTH外壳较完整，但款式变化与缺口变化混在一起；不把更像某个来源款式作为成功标准。'),
'2d08a64e_n3':(
 'D0消减RAW箱式侧壁的细白缝；C前沿尖边与下部整块外壳从RAW延续。',
 'POST外壳更规整但前端尖边仍在，未发现额外的大面积主体丢失。',
 'UNCERTAIN','SMOOTH从S2000悬臂板式变成箱式扶手椅；外壳看起来完整，仍不能据此确认C周边接续或内部质量全面更好。'),
'610b1621_n0':(
 'D0使RAW箱式扶手与下部面更平整；大尺度座框关系未变。',
 'POST继续保留C下伸的斜侧面和箱体；未见大新增破洞。',
 'UNCERTAIN','SMOOTH与S2000都为宽箱式扶手椅，局部双线略少但整体差别有限；CLEAN改为高背细腿款式，不能作为唯一好坏依据。'),
'610b1621_n1':(
 'D0减少RAW座下重复细线，并整理框架椅边缘；C前沿超出灰框的关系从RAW已有。',
 'POST保留front/side座下水平层、C前沿悬伸和端部小间隔。',
 'MIXED','SMOOTH保留了框架与扶手，较CLEAN的窄小框架更能包围C，但相对S2000仍出现显著前沿悬伸和更多分段截线，不能称稳定改进。'),
'610b1621_n2':(
 'D0将RAW框架和seat下垂面整理得较直，但下方大垂片/横杆关系不消失。',
 'POST保留seat下方与C分离的水平线及局部下垂面，主体框架完整。',
 'IMPROVED','相对S2000小椅体配大C的明显比例不协调，SMOOTH形成更覆盖C宽度的扶手框架；仍有座下分离截线和下垂面，仅记局部外观采用改善。'),
'610b1621_n3':(
 'D0减少RAW座框的斜白裂线，靠背和四腿保持，C上方的小斜截线仍在。',
 'POST front座下上横线局部缺口仍可见，side座下几段小线与C前沿悬伸保持。',
 'UNCERTAIN','SMOOTH以窄板靠背替代S2000圆洞靠背且腿更细；大C悬伸仍明显，截面短线变化不足以证明全面改善。'),
'608b46b0_n0':(
 'D0减少RAW背板和座下细裂线，front薄座框更平整；背板孔和主体保留。',
 'POST座下长线更连续但前沿仍与C有局部间隔，腿部细裂缝没有全部消失。',
 'MIXED','SMOOTH front/side座下中段比S2000连续，但外观左前腿可见较明显斜白裂口；不能用截线改善抵消腿部代价。'),
'608b46b0_n1':(
 'D0明显减少RAW背板多条白裂线；侧剖靠背左侧小短线保留。',
 'POST front底线较连贯，side灰色底层也连续；背板边缘仍有细裂线。',
 'IMPROVED','相对S2000 front大断口及side中段缺口，SMOOTH座下连续度更好，腿部大白缝不明显；款式背板也改变，仍保留小短截线和局部间隙。'),
'608b46b0_n2':(
 'D0把RAW箱式扶手椅的底部和侧壁细裂线整理得较平；大结构保持。',
 'POST仍保留两条从C向下斜出的边，外壳整体没有新增明显缺失。',
 'UNCERTAIN','SMOOTH变为更低的箱式扶手椅，S2000较高侧板结构也完整；同平面的线型变化主要伴随款式改变，不能把少短线直接等价于少离体片。'),
'608b46b0_n3':(
 'RAW背板已出现大扇形/尖楔白裂口；D0没有补齐，seat附近线条被整理得较直。',
 'RESTORED/POST仍保留背板大裂口；front座下长线变连续，side上方折线与下层线继续存在。',
 'MIXED','SMOOTH在原front位置的座下线明显更连续，但靠背的大白裂口比S2000严重且RAW已出现；不能据局部旧短线不见就宣称整件更干净。CLEAN保留类似S2000的背板完整度。'),
}

def main():
    qs={q['request_id']:q for q in requests()};assert set(qs)==set(NOTES)
    sd=OUT/'sections/SMOOTH_PRIOR/definition.json';gd=OUT/'gallery/POST/definition.json'
    s=C.read(sd);g=C.read(gd);assert s['bounds']==g['bounds']
    assert [(x['name'],x['axis'],x['position_index']) for x in s['sections']]==[('front',2,1),('side',0,1)]
    rows=[]
    for rid,(raw_d0,post,judgment,comparison) in NOTES.items():
        q=qs[rid];p=OUT/'sections/SMOOTH_PRIOR'/(rid+'.png');pg=OUT/'gallery/POST'/(rid+'.png')
        for image in (p,pg):assert C.sha(image)==C.read(image.with_suffix('.json'))['image_sha256']
        rows.append(dict(request_id=rid,parent_id=q['parent_id'],split=q['split'],noise_index=q['noise_index'],
            label='SMOOTH_PRIOR',baseline='PLAIN_S2000',reviewer='AI_PROXY_NONBLIND',reviewer_agent='/root/edge_prior_training',actually_viewed=True,
            four_stage_image_path=str(p),four_stage_image_sha256=C.sha(p),paired_post_image_path=str(pg),paired_post_image_sha256=C.sha(pg),
            viewed_panels='All four saved stages: front/side finite sections and two halfspace cutaways; three-model POST: two exterior views and front/side sections',
            raw_to_d0_observation=raw_d0,
            d0_to_restored_observation='蓝色C按原输入恢复；可见free主体布局延续D0提议。D0蓝C有移动，不能将该阶段蓝线当作已满足C硬保持。',
            restored_to_post_observation=post,relative_visual_judgment=judgment,paired_post_observation=comparison,
            fragment_identity='NOT_CERTIFIED_FROM_SECTION_ALONE',cross_model_slot_identity='NOT_ASSUMED',
            uncertainty='Finite sections and cutaways miss off-plane connections; visible gaps/short segments do not certify disconnected components. Different styles are allowed. Labels concern displayed geometry only, not a numeric quality metric or manifold guarantee.'))
    dest=OUT/'visual';path=dest/'smooth_actual_review.csv';C.csvwrite(path,rows)
    C.write(dest/'smooth_actual_review_completion.json',dict(status='ACTUAL_REVIEW_COMPLETE',reviewer='AI_PROXY_NONBLIND',
        reviewed_four_stage_pages=28,reviewed_three_model_post_pages=28,reviewed_requests=28,metric_based_labels=False,
        same_section_geometry=dict(bounds_exact_equal=True,front_axis=2,side_axis=0,position_index=1,
            caveat='Identical planes and local limits; different render layout, raster resolution and line widths. Fixed-case crops use narrower declared windows.'),
        visual_judgment_counts=dict(Counter(r['relative_visual_judgment'] for r in rows)),
        files={str(path):C.sha(path),str(sd):C.sha(sd),str(gd):C.sha(gd),str(Path(__file__)):C.sha(__file__)},
        conclusion='Mixed local gains and costs, not broad removal of fragments.608n3 seat-section continuity improves but back tears worsen fromRAW;3fn2 upper short section remains. Most complete chair bodies remain recognizable; style changes are not reconstruction failures.',
        model_calls=0,scorer_calls=0))
    print('ACTUAL_REVIEW_COMPLETE:28 four-stage pages +28 paired POST pages',flush=True)

if __name__=='__main__':main()
