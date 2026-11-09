"""Manually entered observations after actually displaying all 16 frozen six-column pages.

This file only transcribes visual observations. It does not read metrics, meshes,
model outputs, or infer ratings from any numeric scorer.
"""
import json
from pathlib import Path

ROOT=Path('/mnt/e/MeshFlow/experiments/r2_chair_multipart_seam_v1')
KIND='S2000__MULTIPART_S1500_same_pipeline_roles_v2'

def observation(text, phases, role='YES', body='YES', fragments='UNCERTAIN',
                layering='YES', supports='NO', aperture='NO'):
    return dict(text=text,phases=phases,C_role_use=role,body_plausible_complete=body,
                visible_fragments_cracks=fragments,obvious_layering_or_cover=layering,
                support_thinning_or_loss=supports,aperture_fill_observed=aperture)

def case(time,old,new,change,positive,cost,old_process,new_process,note):
    return dict(time=time,old=old,new=new,change=change,positive=positive,cost=cost,
                old_process=old_process,new_process=new_process,note=note)

D={}
D['9d81bc69_BACK_FULL_n0']=case('2026-10-19 16:07:28 UTC',
 observation('蓝色拱形框明确位于椅背；中心大孔仍开放，四条支撑和座面完整。背框周边有灰橙条带。',
 ['RAW的框顶和腿边有轻微不齐。','BASE整理后轮廓稍平顺，中心仍开放。','FINAL与BASE宏观近似，未从截线端点判裂缝。'],layering='UNCERTAIN',aperture='NO'),
 observation('蓝色拱框仍担当靠背，灰色大板填入框内；四腿座面齐全。灰板边缘与蓝框存在深度差。',
 ['RAW中央灰板上缘还有白色斜缝。','BASE灰板填入更连续，框顶仍是蓝灰橙混合。','FINAL保持灰板和椅身，侧截面多一小段绿色修补线。'],aperture='YES'),
 'NO_CLEAR_CHANGE','两模型都清楚利用拱框作靠背，主体完整。','新模型把原来开放的中心填入灰板；输入未规定空洞必须保留，所以不按失败评分，也不把填满算必然改善。',
 ('BETTER','RAW到FINAL腿边及框顶略整齐，未解决或认证细小界面。'),
 ('MIXED','RAW到FINAL中央灰板上缘更闭合；填板是款式变化，图片不能证实精确连接。'),
 '初次16:02:12查看后于本条时间以3497x1485原尺寸复看。固定截面不是孔洞保持约束。')

D['9d81bc69_BACK_FULL_n1']=case('2026-10-19 16:07:28 UTC',
 observation('蓝拱框为背框，灰板填中心，背面有很大的橙色对角三角片；座面和四腿保留。',
 ['RAW橙色大斜片横跨中心填板。','BASE仍有同一明显橙色斜片。','FINAL未见宏观去除这张大斜片。'],aperture='YES'),
 observation('蓝拱框和中心灰板都明确用于背部；中央大橙斜片少于旧模型，但左侧橙条及灰板层仍在。',
 ['RAW灰板表面与上缘略不齐。','BASE灰板轮廓更整，蓝框仍可见。','FINAL与BASE宏观一致。'],aperture='YES'),
 'MIXED','新模型减少了横跨中心的大片橙面，椅身保持。','中心仍有灰色补面及层次，无法仅凭图确认厚度、穿插或共边。',
 ('NO_CLEAR_CHANGE','后处理没有明显消除中央大橙片，主体形状近似。'),
 ('BETTER','RAW灰板小幅平顺化，整体用途不变；不据此认证接续。'),
 '初次16:02:12查看后以原尺寸复看。两者都有填孔，不因源框镂空自动判错。')

D['abbe7279_BACK_FULL_n0']=case('2026-10-19 16:02:36 UTC',
 observation('蓝色高背板清楚构成椅背，另一视角为大块灰色背层；四腿及座面完整。',
 ['RAW腿边略有不齐。','BASE腿和座面边更直，灰背层保留。','FINAL与BASE宏观近似。']),
 observation('蓝背板保持靠背用途，灰背层和侧边条仍覆盖其另一侧。新模型的腿整体较细，但仍保留四条支撑。',
 ['RAW一条前腿有狭窄缝/薄条外观，不能确认拓扑断开。','BASE前腿较顺直，仍比旧模型纤细。','FINAL保持BASE的细支撑和背层。'],supports='YES'),
 'MIXED','新模型仍保有完整椅身和明确的靠背用途。','腿变细；灰色背层覆盖未消失，不能说整体质量全面提高。',
 ('BETTER','后处理减少腿边的小幅毛糙，背层关系未变。'),
 ('BETTER','RAW前腿狭缝/薄条的视觉不齐在BASE后减轻，但支撑变细仍保留。'),
 '细裂是否真实断开为UNKNOWN；不要求恢复来源款式。')

D['abbe7279_BACK_FULL_n1']=case('2026-10-19 16:02:36 UTC',
 observation('完整高背椅，蓝背板一侧清楚，另一侧灰色竖条/背层；座面及四腿都有。',
 ['RAW边缘局部不平顺。','BASE整理后支撑更规整，灰条结构仍在。','FINAL没有可见的大幅形状变化。']),
 observation('蓝背板仍是靠背，灰色横栏与两侧宽橙楔片并存；腿比旧模型明显纤细。',
 ['RAW背顶有灰色小凸起和不齐边。','BASE背顶有所整理，横栏和橙侧片仍明显。','FINAL与BASE近似，未消除横栏与蓝面之间的层次。'],supports='YES'),
 'MIXED','靠背使用和整体椅身仍成立。','灰条组织改变但腿变细，顶部局部薄片是否游离不能由这些图确认。',
 ('BETTER','后处理平顺部分边缘，完整性保留。'),
 ('MIXED','顶部小不齐有所减少，纤细支撑和宽橙侧片仍在。'),
 '未把横栏式完成与完整背板式完成做唯一GT款式排序。')

D['f199965d_BACK_FULL_n0']=case('2026-10-19 16:03:03 UTC',
 observation('扇形蓝背顶部及下部矩形输入均被放在背部；灰色竖向支柱/背条覆盖部分蓝面，椅身和四腿完整。',
 ['RAW背顶灰色边缘略毛糙。','BASE背顶更规整，覆盖条仍在。','FINAL整体与BASE相近。']),
 observation('蓝色扇形背仍清楚可辨，新灰色竖栏叠在蓝背周围/表面，四腿座面完整。',
 ['RAW背冠较不平整。','BASE背冠变整齐，但三根灰色竖栏仍覆盖蓝面。','FINAL保持上述结构。']),
 'NO_CLEAR_CHANGE','两模型都把输入用于靠背并生成完整椅子。','灰栏组织不同，未见稳定的整体质量胜出；覆盖是不是合理厚度不能从图定性为非法。',
 ('BETTER','RAW到FINAL背顶轮廓略平顺。'),
 ('BETTER','RAW背冠的局部不齐减少；灰栏覆盖没有消失。'),
 '截面中的灰蓝平行线只记录层次，不自动归为非法重复表面。')

D['f199965d_BACK_FULL_n1']=case('2026-10-19 16:03:03 UTC',
 observation('蓝背用于靠背，但一侧被大灰色背板遮住；下部蓝面附近有宽橙色斜面，四腿和座面均在。',
 ['RAW背面有明显灰板及橙色局部片。','BASE下方大橙三角仍显著。','FINAL保持背面灰板和下部橙片。']),
 observation('蓝背在双视图中更完整可见，灰色侧柱仍支撑背部；大片遮盖比旧模型减少，但腿相对更细。',
 ['RAW蓝背更清楚，背顶有小不齐。','BASE保持清楚蓝背和较细四腿。','FINAL冠顶有局部绿色补片，主体近似BASE。'],supports='YES'),
 'MIXED','新模型让蓝色背部用途更直观，减少旧模型的大块灰橙遮挡。','支撑变细；图片只能支持局部视觉改善，不能确认精确连接或无相交。',
 ('NO_CLEAR_CHANGE','后处理没有去除主要背层和橙片。'),
 ('BETTER','RAW背顶小不齐减少，明显蓝背用途保留。'),
 '局部正例：更清楚使用C，但不等于所有质量指标提高。')

D['fb8c3a83_BACK_FULL_n0']=case('2026-10-19 16:03:26 UTC',
 observation('弯曲灰色横栏背几乎遮住矩形蓝C；侧截面灰背与蓝板明显错开，无法确认C是实际背面还是附着层。完整椅身仍在。',
 ['RAW蓝C主要隐藏在弯灰背后。','BASE仍然大幅遮盖蓝C。','FINAL没有解决遮盖/错层的视觉不确定性。'],role='UNCERTAIN'),
 observation('蓝矩形背板在背面视图完整可见，靠背作用比旧模型更清楚；另一侧两条灰横栏仍明显偏置并超出蓝板上缘。',
 ['RAW四腿座面完整，橙侧边较明显。','BASE保持蓝背可见性及偏置横栏。','FINAL与BASE宏观相同，不能称完全融为一体。']),
 'MIXED','蓝C由被遮住转为明确可见的背板，支撑仍完整。','灰色偏置横栏仍在，改善不是无错层或连接认证。',
 ('NO_CLEAR_CHANGE','RAW到FINAL弯灰背和蓝C遮盖关系仍未解除。'),
 ('NO_CLEAR_CHANGE','后处理没有消除灰横栏偏置，主体与RAW相近。'),
 'C用途的局部改善从UNCERTAIN到YES；exact topology UNKNOWN。')

D['fb8c3a83_BACK_FULL_n1']=case('2026-10-19 16:03:26 UTC',
 observation('弯曲灰背覆盖矩形蓝C，蓝板部分突出；侧截面显示相互错开的灰蓝两层。椅身完整，但C实际用途仍不确定。',
 ['RAW有明显覆盖/错层。','BASE覆盖/错层仍在。','FINAL未见用途歧义得到消除。'],role='UNCERTAIN'),
 observation('新模型仍是弯灰背叠在蓝矩形板前后，蓝C上方突出；四腿座面完整，C是否真正承担背面仍不清楚。',
 ['RAW保留偏置层。','BASE仍是偏置灰蓝背层。','FINAL与BASE宏观一致。'],role='UNCERTAIN'),
 'NO_CLEAR_CHANGE','两模型都保有完整椅身。','最主要的灰背覆盖/蓝板突出没有解决，不能把原样保留C当作可靠采用。',
 ('NO_CLEAR_CHANGE','后处理未明显解除用途歧义。'),
 ('NO_CLEAR_CHANGE','后处理未明显解除用途歧义。'),
 '这一请求继续保留UNCERTAIN，不能因另一份噪声改善就转成YES。')

D['6c36a5b1_BACK_FULL_n0']=case('2026-10-19 16:03:56 UTC',
 observation('箱式扶手椅具有连续灰色侧墙与底座，倾斜蓝板明确为内侧靠背。',
 ['RAW侧墙外轮廓略有毛边。','BASE外墙更平顺，蓝背用途不变。','FINAL与BASE整体一致。'],layering='UNCERTAIN'),
 observation('新模型仍为箱式椅身、倾斜蓝靠背；背顶橙条较宽，背下方至座面间新增明显橙色内侧斜面。',
 ['RAW双视图及侧截面可见较长橙斜片。','BASE长橙斜片仍在。','FINAL也保留该内侧橙片。']),
 'MIXED','椅身和蓝C背部用途保持。','新增长橙内侧斜片；其是否游离或穿插不能由截线判定。',
 ('BETTER','RAW外边小幅整理，箱式完整性保持。'),
 ('NO_CLEAR_CHANGE','RAW到FINAL的内侧橙斜片仍存在。'),
 '箱式侧墙是可接受的开放生成款式；不能要求必须长四条独立腿。')

D['6c36a5b1_BACK_FULL_n1']=case('2026-10-19 16:03:56 UTC',
 observation('旧模型是四腿低椅，蓝色斜背用于靠背；灰后层与蓝面有明显偏置，橙边带环绕。',
 ['RAW腿和背边略不齐。','BASE整理支撑但背层偏置仍可见。','FINAL仍保留偏置背层。']),
 observation('新模型变为连续侧墙和底座的箱式扶手椅，蓝斜背夹在两侧之间；蓝背下方有向座面伸出的橙色三角片。',
 ['RAW完整箱式椅身，内侧橙三角明显。','BASE橙片仍在，外轮廓略顺直。','FINAL仍保留箱体和内侧橙三角。']),
 'MIXED','新蓝背与两侧墙的宏观组织更明确，椅身保持完整。','改成箱式并非自动更好；内侧橙片是新增风险，不能根据两噪声款式相似断言坍缩。',
 ('BETTER','细微外边整理，旧背层偏置未被解决。'),
 ('NO_CLEAR_CHANGE','整体款式和内侧橙三角基本不变。'),
 '没有独立四腿不是支撑丢失；新模型有完整箱式支撑。')

D['8c9b2023_BACK_FULL_n0']=case('2026-10-19 16:06:24 UTC',
 observation('梯形横栏蓝C明确是靠背，灰色栏和橙条与其并存；四条腿完整，椅脚之间/附近可见一小片与主体分离的灰三角。',
 ['RAW脚边小灰片可见。','BASE脚边小灰片仍可见。','FINAL脚边小灰片也未消失。'],fragments='YES',aperture='YES'),
 observation('蓝横栏背明确用于靠背，灰栏较厚；四腿完整。座面后缘有宽橙三角，脚边未见旧小灰片，但不认证所有微碎片不存在。',
 ['RAW座后缘有一排竖向狭缝/不齐条。','BASE座后缘比RAW连续，但橙色斜面保留。','FINAL加少量绿补片，宏观保留BASE外观。'],aperture='YES'),
 'MIXED','新图中旧脚边孤立小灰片不再可见，椅身和靠背用途保持。','新出现座背下方宽橙斜面，RAW后座边有狭缝状不齐；灰蓝栏层依然存在。',
 ('NO_CLEAR_CHANGE','脚边小灰片贯穿RAW/BASE/FINAL。'),
 ('MIXED','RAW座后边的竖条/狭缝视觉减轻，但内侧橙斜面保留。'),
 '栏间部分橙/灰填入只记录观察，不声明违反输入空洞要求。')

D['8c9b2023_BACK_FULL_n1']=case('2026-10-19 16:06:24 UTC',
 observation('蓝梯形横栏背、四腿和座面完整；栏间有橙色斜条和灰色背条。',
 ['RAW局部边缘略不齐。','BASE保留同样横栏结构。','FINAL有少量绿色局部片，主体无大变化。'],aperture='YES'),
 observation('新模型仍保有蓝梯形靠背和四腿，灰色横栏更厚并与蓝栏偏置，座背侧有橙条。',
 ['RAW后座裙边有细密不齐痕。','BASE裙边略整齐，灰蓝栏层保留。','FINAL有少量绿色修补，宏观接近BASE。'],aperture='YES'),
 'NO_CLEAR_CHANGE','背栏用途和主体都保持。','灰栏厚度和橙条位置改变，不能由这些图支持明确全面提升。',
 ('NO_CLEAR_CHANGE','后处理主要是局部整理。'),
 ('BETTER','RAW后座边的毛糙减轻，未解除栏层偏置。'),
 '切面中的单点线段是有限截面结果，不当成真实断边。')

D['b631b78c_BACK_FULL_n0']=case('2026-10-19 16:06:38 UTC',
 observation('曲面蓝C明显构成椅背，另一侧是一层灰色横向条面；多条橙片横跨蓝背，四条细腿和座面完整。',
 ['RAW橙横条与灰背层较杂。','BASE灰背面更平顺，橙横条仍在。','FINAL与BASE宏观相同。']),
 observation('新模型曲面蓝背更完整露出，原先跨蓝面的橙条减少到周边；背面灰层仍然存在，四条细腿完整。',
 ['RAW蓝背正面更干净，灰背面仍条纹状。','BASE灰背面更平整。','FINAL保持更清楚的蓝背与灰后层。']),
 'BETTER','两视图里新蓝背表面横穿的橙色片明显减少，靠背用途清楚，主体未见明显损失。','灰色后层与蓝面仍偏置，不将局部更清楚等同于无相交或精确接续。',
 ('MIXED','RAW灰后层变平顺，但橙条覆盖仍在。'),
 ('BETTER','RAW后层小幅平顺化，蓝面保持清楚。'),
 'BETTER限于这一展示中的局部视觉；细碎片状态不确定。')

D['b631b78c_BACK_FULL_n1']=case('2026-10-19 16:06:38 UTC',
 observation('箱式扶手椅中曲面蓝C作为内靠背，大橙三角横过蓝面；背冠还有灰色条片，外侧为大灰背墙。',
 ['RAW背顶灰条略毛糙，大橙片明显。','BASE下部橙片仍很宽。','FINAL大橙片和外灰墙均保留。']),
 observation('箱式椅身与蓝曲背都保留；新蓝背正面的大片橙三角消失，背冠灰条/小凸起仍在，外灰背墙仍厚。',
 ['RAW顶部仍有几条灰色薄条。','BASE顶部薄条整理后仍局部可见。','FINAL保留清楚蓝背及顶部细条。']),
 'BETTER','新模型减少横穿蓝靠背的大橙片，完整箱式椅身保持。','背顶疑似薄片的游离性不确定，背后大灰墙仍在，不能据此判拓扑正确。',
 ('NO_CLEAR_CHANGE','后处理没有消除主要橙覆盖。'),
 ('MIXED','蓝背保持清楚，但背顶小灰条没有完全消失。'),
 '箱式支撑与四腿样式均允许，不按来源款式打分。')

D['f4f1aba6_BACK_FULL_n0']=case('2026-10-19 16:06:53 UTC',
 observation('两三角蓝矩形板完整用于高靠背；后方灰背墙与其平行，上方有拱冠，短腿和座面完整。',
 ['RAW背板/座面边略有细小不齐。','BASE更规整，灰蓝背层仍在。','FINAL与BASE宏观近似。']),
 observation('新模型仍是蓝矩形高背、灰后背墙、拱冠与短腿椅身；橙条更多在左侧，整体和旧模型近似。',
 ['RAW蓝板完整可见，冠顶略粗糙。','BASE冠顶/边缘略平顺。','FINAL与BASE近似。']),
 'NO_CLEAR_CHANGE','两模型都清楚把C用作背板，完整主体得到保留。','灰蓝平行背层仍在；橙边位置变化未构成明确整体质量提升。',
 ('BETTER','RAW小幅边缘不齐被整理，双层关系不变。'),
 ('BETTER','RAW小幅冠顶/边缘整理，整体结构不变。'),
 '不以平行背层直接认定非法；没有从有限截线端点推断裂缝。')

D['f4f1aba6_BACK_FULL_n1']=case('2026-10-19 16:06:53 UTC',
 observation('蓝C清楚位于高椅背，但座面横过蓝板下部较高位置，使一段蓝面延伸到座下；顶部有宽橙片，四条支撑在。',
 ['RAW座面与蓝板下缘关系明显错开。','BASE仍保留座下蓝板段及顶橙片。','FINAL上述关系基本不变。']),
 observation('新模型仍保留蓝高背与四腿，座面更接近蓝板下沿，顶部宽橙片减少，冠顶变为圆拱/厚帽形。',
 ['RAW座面较靠下，与蓝板用途更直观协调。','BASE座面更整齐，蓝板下缘仍有局部重合层。','FINAL与BASE宏观相同。']),
 'BETTER','新座面更接近蓝C下沿，旧座下蓝延伸和顶部大橙片减少，主体仍完整。','后灰层仍存在；冠顶款式改变不作质量得分，不保证已精确相接。',
 ('NO_CLEAR_CHANGE','主要高度关系及顶橙片未改变。'),
 ('BETTER','RAW座边略平顺化，较合理的蓝背/座面关系保留。'),
 '局部视觉改善与接续指标独立；最终shared-edge/intersection为UNKNOWN。')

review=json.loads((ROOT/'visual_review_templates/BACK_FULL.json').read_text())
images={x['request_id']:x for x in json.loads((ROOT/'gallery'/KIND/'completion.json').read_text())['images']}
assert set(D)=={r['request_id'] for r in review['requests']}
for r in review['requests']:
    e=D[r['request_id']]
    r.update(image_sha256=images[r['request_id']]['image_sha256'],actual_view_status='ACTUALLY_VIEWED',viewed_at_utc=e['time'])
    for model_i,key in enumerate(('old','new')):
        o=e[key]
        for phase_i in range(3):
            s=r['stages'][model_i*3+phase_i]
            for flag in review['flags']:s[flag]=o[flag]
            s['notes']=o['text']+' '+o['phases'][phase_i]+' 细小裂片不可全凭宏观图排除；精确共边/非法相交均UNKNOWN。'
    for c in r['old_new_comparisons']:
        c.update(change=e['change'],positive_observations=e['positive'],costs_or_uncertainties=e['cost'])
    for c,key in zip(r['within_model_processing'],('old_process','new_process')):
        c.update(RAW_to_FINAL_change=e[key][0],observations=e[key][1])
    r['request_notes']=e['note']+' 逐页实际检查六列、双视角和固定前/侧截面；非盲AI_PROXY，不是人类受试评价。'
review['status']='MANUALLY_ENTERED_ACTUAL_REVIEW'
review['entry_source']='This entry script contains observations transcribed after displaying each frozen page through view_image; it does not read any metric tables.'
review['actual_display_policy']='All 16 pages displayed at original 3497x1485; first 9d81 pair was initially resized then re-viewed at original detail at recorded final time.'
dest=ROOT/'visual_review_entries/BACK_FULL_entered.json'
assert not dest.exists(),'Never overwrite entered observations'
dest.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(dest)
