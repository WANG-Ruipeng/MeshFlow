"""Actual nonblind CLEAN vs S2000 observations from the 28 POST gallery pages."""
from .evaluation import C,OUT,requests

NOTES={
'73b369b9_n0':('CLEAN的front底线两端更接近闭合，原基线右端缺口缩小；side内层转角间隔仍在，外观大体保持。','LOCAL_SECTION_GAIN_WITH_RESIDUAL_GAPS'),
'73b369b9_n1':('CLEAN从悬臂底座变为侧板支撑形体，front座板层较薄；不是原对象重建任务，款式改变本身不算失败，但side端部间隔/小线仍在，未见统一质量优势。','STYLE_CHANGE_NO_CLEAR_QUALITY_GAIN'),
'73b369b9_n2':('背板顶部大斜裂口两模型都存在；CLEAN front底线略连续，side右下转角仍开口，外观近似。','MIXED_LOCAL'),
'73b369b9_n3':('CLEAN与基线形体相近，side内层右转角更接近连贯；front底线右端及side左后间隔仍在。','LOCAL_SECTION_GAIN_WITH_RESIDUAL_GAPS'),
'3f741759_n0':('CLEAN front右侧到蓝C新增明显竖向间隔，基线该侧基本连续；side近似、上方短线都在。背条斜细片未消除。','LOCAL_GAP_WORSE'),
'3f741759_n1':('整体背条/四腿与两截面高度相近，front右端缺口仍在；未见清晰整体改善。','SIMILAR_VISIBLE'),
'3f741759_n2':('CLEAN座下front和side出现显著宽缺口，原基线相应底线连续；靠背顶弧也可见额外白缝，外观主体仍保持。','CLEAR_LOCAL_SURFACE_WORSE'),
'3f741759_n3':('两模型外观和盒状side截面近似，front右端到C的缺口未解决。','SIMILAR_VISIBLE'),
'85b16941_n0':('CLEAN front底线右部有明显宽缺口，原基线连续；外观板背/细腿相近，side盒状轮廓保持。','LOCAL_GAP_WORSE'),
'85b16941_n1':('两模型整体相似；CLEAN side左上接近C处出现小间隔，原基线较连贯，front左侧原缺口仍在。','LOCAL_GAP_WORSE'),
'85b16941_n2':('CLEAN front左竖边和side左上增加可见间隔，靠背上沿出现小白口；主体比例仍相近。','LOCAL_GAP_WORSE'),
'85b16941_n3':('两模型均保留front底线右端缺口；CLEAN另有轻微底线断点，外观差异有限，无明确整体收益。','SIMILAR_WITH_SMALL_GAPS'),
'8c9b2023_n0':('CLEAN side右端闭合更好，但front左端和底线多处间隔比基线明显；背横档斜薄条仍在。','MIXED_LOCAL'),
'8c9b2023_n1':('外观和side缺口格局基本相同；CLEAN front底线略有新细缝，未见明显改善。','SIMILAR_WITH_SMALL_GAPS'),
'8c9b2023_n2':('CLEAN front底线比基线更多断点，side基本相似；背横档仍有不规则斜线。','LOCAL_GAP_WORSE'),
'8c9b2023_n3':('CLEAN front底部中央出现大段缺失，side底线几乎不再出现，而原基线仍有长底线；外观整体椅子还在，不能用外观保持抵消截面退化。','CLEAR_LOCAL_SURFACE_WORSE'),
'2d08a64e_n0':('CLEAN变为低靠背、弯曲下壳形体；原基线side小短线被不同的弯曲层替代，但新层间距离与断口仍明显，front左侧也有较大间隔。款式变化不单独判失败。','STYLE_CHANGE_MIXED_GEOMETRY'),
'2d08a64e_n1':('外部侧板支撑风格相近；CLEAN front下层左半段不再出现在同切面，原基线长底线连续；side C与下方框架间隔持续。','LOCAL_SECTION_LOSS'),
'2d08a64e_n2':('CLEAN由扶手框架变为弯折低背，外观较简洁；front新增双层下线，side下壳出现较大折角和断点，不能判成全面更好。','STYLE_CHANGE_MIXED_GEOMETRY'),
'2d08a64e_n3':('两模型悬臂式大体相同；CLEAN下底板可见较宽斜向开口，C下斜片/空隙仍在；截面未给出明确改善。','LOCAL_SURFACE_COST'),
'610b1621_n0':('CLEAN从低箱式扶手椅变为高靠背细腿形体；仍可辨椅子，C前缘外伸与内层线保留，无证据说明接合质量整体更好。','STYLE_CHANGE_NO_CLEAR_QUALITY_GAIN'),
'610b1621_n1':('CLEAN变为小背板细腿，蓝C相对生成主体明显更宽且前伸；front/side下方窄矩形与C分开，未见采用与接续同步改善。','STYLE_CHANGE_RESIDUAL_MISALIGNMENT'),
'610b1621_n2':('CLEAN换成竖条靠背四腿，仍存在蓝C大幅外伸与窄下框；相较基线小散线变规整，但整体层间错位保留。','MIXED_LOCAL_WITH_RESIDUAL_MISALIGNMENT'),
'610b1621_n3':('圆孔背板与四腿形体近似；两模型side短L形截线和C下间隔都保留，未见整体改善。','SIMILAR_VISIBLE'),
'608b46b0_n0':('CLEAN靠背底部在背面视角出现较大的白色三角缺口，基线此处更满；front/side下层间隔仍在。','LOCAL_SURFACE_COST'),
'608b46b0_n1':('CLEAN腿部原明显细长裂口减少，side下层中段更连续；front左侧缺口仍在，背顶洞/缝也保留。','LOCAL_SURFACE_GAIN_WITH_RESIDUAL_GAPS'),
'608b46b0_n2':('箱式外形与两侧内小斜片近似，front/side所见层次基本保持，未见显著变化。','SIMILAR_VISIBLE'),
'608b46b0_n3':('CLEAN腿部原明显白缝减少，但指定front座下与side中段间隔仍在；不是所有碎裂都被解决。','LOCAL_SURFACE_GAIN_WITH_RESIDUAL_GAPS'),
}

def run():
    rows=[];assert set(NOTES)=={q['request_id'] for q in requests()}
    for q in requests():
        rid=q['request_id'];p=OUT/'gallery/POST'/(rid+'.png');r=C.read(p.with_suffix('.json'))
        assert C.sha(p)==r['image_sha256'];note,tag=NOTES[rid]
        rows.append(dict(candidate='CLEAN_EDGE_COH',baseline='PLAIN_S2000',stage='POST',request_id=rid,split=q['split'],
            evaluator='AI_PROXY',blind=False,status='ACTUALLY_VIEWED',observation=note,descriptive_tag=tag,
            image_path=str(p),image_sha256=C.sha(p),views='Two saved exterior angles plus exact inherited front/side finite sections',
            limitation='Descriptive development observations, not a pre-registered scalar success rate or certification of topology. Different styles allowed; no unique-source ranking.',
            prior_note='Shared base Gaussian only. SMOOTH column uses transformed bulk initial state and is reviewed separately.'))
    p=OUT/'visual/CLEAN_EDGE_COH_vs_PLAIN_S2000_POST_actual_review.csv';C.csvwrite(p,rows)
    result=dict(status='ACTUAL_POST_COMPARISON_COMPLETE',pages=28,evaluator='AI_PROXY',blind=False,
        files={str(p):C.sha(p)},source_code_sha256=C.sha(__file__),
        conclusion='No broad visual quality upgrade established. Some local gaps/leg cracks improve, while 3f/n2,85/n0,8c/n3 and608/n0 show material local costs. Complete rows retained.',
        metric_labels_derived_from_images=False,model_calls=0)
    C.write(OUT/'visual/CLEAN_EDGE_COH_comparison_completion.json',result);return result

if __name__=='__main__':print(run()['status'],flush=True)
