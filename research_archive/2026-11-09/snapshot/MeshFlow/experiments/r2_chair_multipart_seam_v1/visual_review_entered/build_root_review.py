"""Transcribe root's actual displayed-image observations; no automatic rating."""
import json, hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
doc=json.loads((ROOT/'visual_review_templates/seat_part_02.json').read_text(encoding='utf-8-sig'))
times={'2d':'2026-10-19T16:01:33.711Z','608':'2026-10-19T16:02:04.889Z',
       '610':'2026-10-19T16:02:44.496Z','8c':'2026-10-19T16:03:25.363Z'}
# Six notes in displayed order: old RAW/BASE/FINAL, new RAW/BASE/FINAL.
notes={
'2d08a64e_n0':[
'蓝色C位于座面；板背、落地侧框可辨。背板有细碎纹理，侧截面有若干短线，不能据截线认证孤岛。',
'板背与两侧落地框仍在；表面较RAW整理，座面下灰层仍有明显垂直间距。',
'大轮廓基本同BASE；固定侧截面下层间距仍在，未见宏观救援。',
'C仍作座面，但新背板向下折，背板中段可见白色斜缝；两侧落地框明显更细。',
'背板斜缝和细落地框仍可见，D0未消除；正截面灰底层更低，精确连通性未知。',
'背板斜缝和极细侧框仍在，两视角未显示明显修复；座面用途没有改变。'],
'2d08a64e_n1':[
'C作为座面；背板和下部板式支撑可辨，连接部位及板面有细碎纹理。',
'板面较整齐，但背板/下板之间的边界与座面附近关系仍不清楚。',
'宏观板式椅轮廓近似BASE，固定截面未显示明显整体改变。',
'仍是板式支撑椅，背板与下板之间多出醒目的横向细缝；C仍位于座面。',
'横向缝仍在，部分板面纹理被整理；仅凭两视角不能确认支撑完整。',
'主体与BASE近似，横向缝仍在；没有把截面断点直接认作网格破洞。'],
'2d08a64e_n2':[
'带扶手框架、板背和蓝色座面形成可辨椅形；后部有重叠感较强的斜板区域。',
'扶手框架保留，板面更整齐；座面附近斜板关系仍较复杂。',
'完整轮廓与BASE接近，未见大块主体新增或缺失。',
'变为无扶手的细侧框椅，款式变化本身不扣分；背板下段白缝和极细框架值得保留。',
'细框与背板白缝仍在；侧截面橙色斜段向下伸展，不能从截线认证相交。',
'未明显修复细框和白缝；C仍作为座面，整体简化也伴随可见薄弱处。'],
'2d08a64e_n3':[
'板式悬臂椅轮廓可辨，C作座面；背板边缘和底板有细碎不齐。',
'板面较RAW平整，座面、靠背和底座仍可辨。',
'同BASE大轮廓，未见大面积消失；截面不覆盖全部底座。',
'同类板式椅，座面仍在正确位置；背板下部与底座有细碎纹理。',
'板面更整齐，底座仍在；固定截面保留孤立短线但不认证孤岛。',
'整体与BASE接近，未见相对旧模型明确的宏观优劣。'],
'608b46b0_n0':[
'四腿及圆顶板背可辨，C作为座面；橙色三角面延伸到靠背区，座面下截面有分段灰线。',
'四腿保留、轮廓较整齐，C下方灰层仍在。',
'与BASE接近，四腿没有明显新缺失；不能从表面图判共享边。',
'变为高侧板扶手椅，C位于座位；不是来源款式失败，靠背可见水平细白缝。',
'高侧板和支撑保留，靠背的横向细缝仍可见，是否设计分片不确定。',
'与BASE近似；相比旧模型款式改变，未据此判成更好或更坏。'],
'608b46b0_n1':[
'四腿圆背椅，蓝色面是座面，局部面线粗糙。',
'主要部件保留，细纹减少；下部灰层仍与C有间隔。',
'宏观与BASE相近，没有显示明显新接续。',
'高侧板、前裙板和背板围住C座面；背部有水平分段纹理。',
'板面更规则；C仍位于座位，细缝/分段是否非法不由图判。',
'整体同BASE，增加包围式板件是款式变化，不作GT款式比较。'],
'608b46b0_n2':[
'板式扶手椅，C作为座面，侧板/前裙与靠背均在。',
'板面更规整，宏观各部位保留。',
'局部绿色补片可见，侧截面增添短绿色段；共享边是否成立需用账本。',
'与旧模型是相近板式椅；C用途一致，靠背仍有细条纹。',
'板件保留且更平整，未见宏观支撑消失。',
'与新BASE近似；旧最终绿色补片与新结果不同，不能由颜色推断成功。'],
'608b46b0_n3':[
'四腿圆顶背板椅，C为座面，背板和腿部有小面纹理。',
'轮廓较整齐，四支撑仍可辨。',
'主要部件与BASE相近，未见显著新增修补。',
'四腿仍在，但腿面有更显眼的白色楔隙/破碎轮廓，背缘也更零碎。',
'腿部轮廓经整理，细窄楔隙是否完全消失在此尺度仍不确定。',
'宏观四腿保住，细部仍不能认证完整；未把分段截线直接视作悬片。'],
'610b1621_n0':[
'箱式扶手椅，C在座面，前橙色裙板和细脚可辨。',
'箱式主体保留，局部板面更规整。',
'宏观与BASE基本一致，固定截面位置相同。',
'同样箱式扶手椅，C保持座面用途；侧/前板仍在。',
'略整理表面，未见明显缺少主部件。',
'与BASE近似；新旧差异主要局部面分配，非宏观质量突破。'],
'610b1621_n1':[
'带扶手框架椅，C为座面，背板、侧框及支撑可辨。',
'框架保留，座面边缘局部小面更规整。',
'宏观同BASE，中心截面下部斜线不等于完整可见碎片。',
'同样框架椅，靠背区橙色三角区域更大；颜色不等于缺陷。',
'框架完整外观保留；侧截面短灰线仍有，但不能认证悬空。',
'宏观与BASE接近，未见比旧终态明确更合理的整体轮廓。'],
'610b1621_n2':[
'简单四腿板背椅，宽C作为座面，橙色面延伸至背板旁。',
'四腿保留，板面整齐；C下方仍有第二层灰线，厚度合法性未知。',
'大轮廓近似BASE，未把双截线自动判重复面。',
'仍为四腿板背椅，背部橙色覆盖减少，侧截面有向下橙色斜段。',
'背板与腿较整齐，C用途仍明确；下方灰层与斜段关系不能仅凭图判断。',
'主体同BASE，外观差异有限；第三点朝向的代价留待几何表判断。'],
'610b1621_n3':[
'四腿及带椭圆开孔的背板可辨，C作为座面；孔缘有细碎纹理。',
'开孔仍在，孔下沿有橙色细片/唇边可见，未完全填死。',
'椭圆孔与支撑保留；固定截面仍有小段，不能认证破洞。',
'同样带开孔的背板椅，孔缘和腿更零碎，C用途清楚。',
'孔缘整理且开孔保留，底缘橙色片仍可见。',
'主要形状与BASE一致，未把完整开孔的保留误称成全部细部无损。'],
'8c9b2023_n0':[
'梯形横档靠背、四腿与C座面形成完整可辨椅形，背缘细碎。',
'横档与四腿保留，面线更规则。',
'座面前沿绿色补片可见，宏观仍同BASE。',
'主体与旧模型相近，C为座面；座后沿橙色梳齿状小面明显增多。',
'主体整理后橙色小面仍在，是否独立碎片不由表面图认证。',
'四腿、横档保留；后沿梳齿状小面未明显消除。'],
'8c9b2023_n1':[
'横档椅，C作为座面，整体与n0相似；后沿少量橙色小面。',
'整体结构保留、板边更整齐。',
'宏观与BASE近似，局部方向/索引需用几何账本。',
'横档四腿仍在，后沿橙色梳齿状面比旧图多。',
'表面整理未消除这些后沿小面。',
'前沿绿色补片可见，侧截面绿色竖段新增；主体外观保持，不据绿色推断连接。'],
'8c9b2023_n2':[
'可辨横档椅，C为座面，横档近端有细碎线。',
'大结构保留，细碎面线减少。',
'大轮廓与BASE近似，未见明显新增缺失。',
'主体同类，后沿梳齿状橙色面更连续，C仍作座面。',
'这些小面仍在；正截面下层曲线有分段，不能据此认证裂缝。',
'宏观与BASE相同，局部橙色小面没有明显改善。'],
'8c9b2023_n3':[
'横档四腿椅，C用途明确，背缘/座下少量细碎面线。',
'轮廓整理，主要部件保留。',
'一侧座缘绿色补片可见，完整轮廓仍在。',
'四腿横档保持，但后沿梳齿状面增多；正截面底层比旧图更分段。',
'后沿小面与截面分段仍在，未由截面断点认定物理断裂。',
'一侧补片出现，后沿小面仍在；原有宏观椅形保留，微小裂缝状态未知。']}

for r in doc['requests']:
    rid=r['request_id'];prefix=rid.split('b')[0] if rid.startswith('608') else ('610' if rid.startswith('610') else '2d' if rid.startswith('2d') else '8c')
    prefix='608' if rid.startswith('608') else prefix
    p=Path(r['image_path'].replace('/mnt/e/MeshFlow',str(ROOT.parents[1])))
    # On WSL the canonical path already exists; Windows replacement is fallback.
    p=Path(r['image_path']) if Path(r['image_path']).exists() else p
    r.update(image_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),actual_view_status='ACTUALLY_VIEWED',
             viewed_at_utc=times[prefix],evaluator='AI_PROXY',nonblind=True)
    for i,s in enumerate(r['stages']):
        s.update(C_role_use='YES',body_plausible_complete='YES',visible_fragments_cracks='UNCERTAIN',
                 obvious_layering_or_cover='UNCERTAIN',support_thinning_or_loss='NO',aperture_fill_observed='NO',
                 topology_by_image='UNKNOWN',notes=notes[rid][i])
        if rid=='2d08a64e_n1':s['body_plausible_complete']='UNCERTAIN'
        if rid in ('2d08a64e_n0','2d08a64e_n2') and i>=3:
            s.update(body_plausible_complete='UNCERTAIN',visible_fragments_cracks='YES',support_thinning_or_loss='YES')
        if rid=='608b46b0_n3' and i==3:s['visible_fragments_cracks']='YES'
        if rid.startswith('610') and i in (1,2,4,5):s['visible_fragments_cracks']='UNCERTAIN'
        if rid.startswith('8c'):s['obvious_layering_or_cover']='NO'
    changes=['NO_CLEAR_CHANGE']*3
    if rid=='2d08a64e_n0':changes=['WORSE']*3
    if rid=='2d08a64e_n2':changes=['MIXED']*3
    if rid in ('608b46b0_n0','608b46b0_n1'):changes=['MIXED']*3
    if rid=='608b46b0_n3':changes=['WORSE','MIXED','MIXED']
    for i,c in enumerate(r['old_new_comparisons']):
        c.update(change=changes[i],positive_observations='C仍可辨为座面，整体椅形保留；不按来源款式打分。',
                 costs_or_uncertainties=notes[rid][i+3]+' 对照旧图：'+notes[rid][i])
    for i,c in enumerate(r['within_model_processing']):
        change='BETTER' if rid.startswith('8c') or rid in ('2d08a64e_n3','608b46b0_n3','610b1621_n2','610b1621_n3') else 'NO_CLEAR_CHANGE'
        c.update(RAW_to_FINAL_change=change,observations='RAW：'+notes[rid][3*i]+' 最终：'+notes[rid][3*i+2])
    r['request_notes']='实际查看六列、两个整体视角及相同位置正/侧截面。UNKNOWN表示图像不能认证精确连接/自交；UNCERTAIN保留细节分辨率、遮挡和截面歧义。'
doc['status']='ENTERED_ACTUAL_AI_PROXY_REVIEW'
out=ROOT/'visual_review_entered/seat_part_02.json'
assert not out.exists()
out.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(out)
