"""Manually supplied observations after viewing all 28 four-stage PNG pages."""
from .evaluation import C,OUT,requests

NOTES={
'73b369b9_n0':('front底线两端和side座下转角已有断点；板背/悬臂结构可辨。','D0使front底线和局部转角更整齐；C提议存在细碎角点差异。','恢复C后轮廓基准重合，但side座下内层右转角仍不连续。','front较整齐；side下层右侧水平线至竖线的缺口更明显，未普遍消裂。'),
'73b369b9_n1':('座板厚度在front截面很薄；side右前端有斜短线，靠背底部也有断点。','小散线减少，side右上方短斜线仍在。','恢复C后side右上短斜线和座下线仍分离显示。','形体保持；右前端短线及front薄层保留，剖切可见前侧细长片状结构。'),
'73b369b9_n2':('front底线有微小阶差；side座下内层转角和背板顶边有缺口。','底线与板面更整齐，背顶斜开口未消失。','C恢复后侧面层间关系基本沿用D0。','side下层右端至竖段有明显截面间隔；背顶开口仍可见，主体没有整体消失。'),
'73b369b9_n3':('整体悬臂形体可辨，front/side下层有细微阶差及断点。','零散细线减少，座下线条拉直。','恢复C主要重建蓝色轮廓基准，未使所有下层转角连续。','板背与底架保持；side内层右转角仍有局部间隔，不能从此截面认证全局共边。'),
'3f741759_n0':('front座下左侧大开口、side上方孤立短截线已存在；背部有斜细条。','side下框被整理成较连续轮廓，上方短线保留。','C恢复后上方短线及front左侧空隙保持。','背条与四腿形体保持；front左侧开口、side上方短线仍在，短线不能独自当作悬片。'),
'3f741759_n1':('front底线有微断点，side底部阶差；背板有细缝。','线条趋直、噪声减少，front右上方仍未接到蓝C。','恢复C没有明显新增大尺度缺损。','整体形体保持；front右侧至C的空隙未消除，side底线仍有细微断点。'),
'3f741759_n2':('front与side座下都有明显宽开口；剖切座下存在三角缺口/斜片。','整体表面变整齐，但宽开口仍保留。','恢复C后底部大间隔和斜片格局没有根本改变。','宽开口与座下斜三角结构仍在；本模型不继承旧模型slot21的语义身份。'),
'3f741759_n3':('front右端有额外短斜线，side下框较完整。','front短斜杂线减少，下框更规整。','恢复C后主要轮廓保持。','整体较规整，front右端到C仍有缺口；side下框保留小台阶，不等于全部接口已合法。'),
'85b16941_n0':('front底部偏右有宽间隔，side截面基本呈盒状；座下有细碎内部线。','side盒状边界变整齐，front底部宽缺口保持。','恢复C未消除front宽缺口。','座下灰面更规整，front右下缺口依然清楚；上方背板U形截线属于切面观察，未认证悬片。'),
'85b16941_n1':('front左竖边断点、side左上接近C处断点均已存在。','底线更整齐，但左侧两个断点仍可见。','C恢复后side左上间隔不消失。','板背/细腿保持；side剖切座下前角的小白缺口仍在，不能视为普遍修复。'),
'85b16941_n2':('座下轮廓大致完整，但side左上与蓝C间有空隙；背板细缝可见。','板面变整齐，side左上间隔保持。','恢复C使蓝轮廓规整，未合上左上截面缺口。','形体保持，front较连续；side左上缺口仍在，无从此图认证水密。'),
'85b16941_n3':('front底线右端有断点，side近似盒状。','小噪声减少，front右底间隔仍在。','C恢复不改变该间隔格局。','灰面更整齐，front右下开口保持，side盒状轮廓较连贯。'),
'8c9b2023_n0':('front底线有多处小断点，side左侧座下竖边缺失；座下斜条/裂隙已存在。','线条变整齐，但side左侧仍未闭合。','恢复C后层间位置相似，座下斜结构保留。','side左端出现连续竖边；front底线仍有断点，座下斜条及缝隙没有全部消除。'),
'8c9b2023_n1':('side左端只有短竖段，上方背横档截线有开口；座下和背横档有斜条。','小噪声减少，右前段整理；横档开口保持。','恢复C后侧面左端仍不完整。','side左端竖短段不再贯通上下，front较平整；背横档与座下斜条保留。'),
'8c9b2023_n2':('front底线断点、侧壁有小缺口；背横档有斜线。','side盒状轮廓更连贯。','恢复C后side盒状结构保持。','side较完整，但front底线断点、背横档和座下斜线仍可见。'),
'8c9b2023_n3':('front底部中央大缺口，side几乎没有底部横线；座下斜薄片已存在。','线条更平直但底部中央缺口未填回。','C恢复后上述缺口仍在。','主体与横档保持，front/side下层大缺口持续；不是POST新制造的全部缺失。'),
'2d08a64e_n0':('front下层多条水平线，side弯曲下壳有多处断点；悬臂结构可辨。','细碎裂线明显减少，弯曲下壳变平顺。','恢复C后仍有分层下壳与局部间隔。','front双层水平线和side内层右端间隔仍在，整体形体保持。'),
'2d08a64e_n1':('front下层近完整，side蓝C下方与腿框明显分开；有短细截线。','front下层截线减少，side短细截线仍在。','恢复C后前端斜片与下方框架仍相隔。','front下层仅右部出现，side仍有短点线与层间分离；不能把切面内缺线直接等同删掉整张面。'),
'2d08a64e_n2':('front下层双线、side向下折的长斜线已有，局部多个断点。','细碎线减少，宏观折线和低于C的下壳不变。','恢复C后没有使下壳整体贴合到C轮廓。','front仍双层，side下壳折线与间隔保留；剖切灰色大斜面持续存在。'),
'2d08a64e_n3':('front下层横线在本窗口缺失；side C下有较大空隙及斜片，悬臂形体可辨。','杂线减少，大空隙/斜片保留。','恢复C后前端下垂三角和后方折层关系未改变。','大间隔和薄斜片仍可见；上/下架分离在RAW已出现，不是统一归因于GEO。'),
'610b1621_n0':('大箱式/扶手形体可辨，C两侧有下降斜边；背板内层竖线穿过切面。','表面噪声减少，座下宏观层次保留。','恢复C后轮廓和箱体布局相似。','箱体外形保持，side内层竖线与前端下垂斜边仍在；此窗口不足认证接续。'),
'610b1621_n1':('C比下方座盒更宽，front/side均见与其分开的矩形下层；两侧有斜片。','矩形下层变规整，但与C的间隔仍在。','恢复C保持同样外伸与下层分离。','下层矩形与C仍分开显示，背板更干净；宏观错层未因后处理消失。'),
'610b1621_n2':('C两侧外伸，下面独立矩形条框；背竖条存在细碎裂线。','下框变整齐、背条噪声减少。','恢复C后下方框与蓝C仍有明显竖向间隔。','外伸C与下方窄框格局保持，局部斜片/间隔仍在。'),
'610b1621_n3':('带圆孔靠背；side C下有两处短L形截线，front内侧有短竖线。','大面更整齐，短L形截线并未消失。','恢复C后短L形及下框保持。','背部表面更规整，但side两个短L形与front短线仍可见，需完整网格身份判断是否真正悬片。'),
'608b46b0_n0':('front座下局部断点、side下层中间大间隙已存在；板背与四腿形体可辨。','噪声减少但side下层间隙持续。','恢复C后蓝C与下层仍相隔，后侧斜面保留。','front和side间隙仍可见；剖切座下三角白缝保持，未观察到普遍闭合。'),
'608b46b0_n1':('front左侧下层断口，side下层小断点；背板顶部有小洞。','side底线更连续，但front左端断口保留。','恢复C后front断口不消失。','底线整体更平直；front左侧缺口、背板洞仍在，整体四腿形体保持。'),
'608b46b0_n2':('箱式扶手外壳，front两侧内短线、side前后上翘轮廓存在。','板面更整齐，宏观箱体形状基本不变。','恢复C后内侧短线/斜边仍在。','箱体外形保持，座面两侧小斜片仍可见；本窗口不能判断这些片是否孤立。'),
'608b46b0_n3':('指定front位置座下多段短线和间隔已在RAW出现；side下层也有断口。','D0后front短线间隔仍在，座面下边更平直。','恢复C后指定位置没有闭合；不能将新输出槽号等同旧slot77/113。','指定front座下间隔、side下层中段间隔仍清楚；右腿细缝被整理，但并未消灭所有局部缺口。'),
}

def run():
    rows=[];assert set(NOTES)=={q['request_id'] for q in requests()}
    for q in requests():
        rid=q['request_id'];p=OUT/'sections/CLEAN_EDGE_COH'/(rid+'.png');receipt=C.read(p.with_suffix('.json'))
        assert C.sha(p)==receipt['image_sha256'];notes=NOTES[rid]
        rows.append(dict(label='CLEAN_EDGE_COH',request_id=rid,split=q['split'],evaluator='AI_PROXY',blind=False,
            review_status='ACTUALLY_VIEWED_FOUR_STAGE_PAGE',image_path=str(p),image_sha256=C.sha(p),
            RAW_observation=notes[0],D0_observation=notes[1],RESTORED_observation=notes[2],POST_observation=notes[3],
            views='front finite section; side finite section; front and side true halfspace cutaways',
            restriction='Image observations only. No scalar quality score used to manufacture labels; section short traces do not certify floating; cross-model slots not semantic pairs.',
            cross_model_comparison='PENDING_POST_COMPARISON_GALLERY'))
    p=OUT/'visual/CLEAN_EDGE_COH_four_stage_actual_review.csv';C.csvwrite(p,rows)
    r=dict(status='ACTUAL_FOUR_STAGE_REVIEW_COMPLETE',pages=28,stage_observations=112,
        files={str(p):C.sha(p)},source_code_sha256=C.sha(__file__),evaluator='AI_PROXY',blind=False,
        comparison_status='PENDING_POST_COMPARISON_GALLERY',model_calls=0)
    C.write(OUT/'visual/CLEAN_EDGE_COH_four_stage_review_completion.json',r);return r

if __name__=='__main__':print(run()['status'],flush=True)
