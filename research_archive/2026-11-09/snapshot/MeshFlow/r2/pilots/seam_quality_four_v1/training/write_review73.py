"""Record the agent's actual eight-page visual inspection; no inference/scoring.

These observations were authored only after all saved gallery pages had been
opened at original resolution on 2026-10-17. They are not metric-derived labels.
"""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
EVAL = ROOT / 'experiments/r2_seam_quality_four_v1/evaluation'
MODELS = ['SEAM200', 'LONG', 'ANTI_COVER', 'LOCAL_NOISE']
PARENT = '73b369b9808bc2d0f6ba522682be8978'

# Ordered by displayed model column. All entries refer to images actually seen.
OBS = {
 (0, 'RAW'): [
  ('矩形靠背、蓝色座面、前竖板和底板均可见；座面下方另有平行灰色薄层轮廓，靠背与座面交界有细亮线。', 'REFERENCE'),
  ('同样的板式椅主体与蓝色座面；下方灰层较平直，前缘和背座细缝仍可见。', '主体与 SEAM200 很接近；下方灰层轮廓更规整，没有明显的用途变化。'),
  ('蓝色座面处于椅身正常座位处；矩形背板与前竖板完整，背板三角线方向不同。', '主要是背板三角线与下方轮廓的小变化，未见明确的整体质量跃升。'),
  ('矩形靠背和前竖板、底板构成与其他列近似的椅形；座面下方灰层仍在。', '与 LONG/ANTI_COVER 相近；相对 SEAM200 的外观变化很小。'),
 ],
 (0, 'POST'): [
  ('蓝色座面仍在完整板式椅身中；背板上沿有细小突出，底板与竖板附近可见细白缝。', 'REFERENCE'),
  ('座面、背板和支撑板完整；座面前缘与下部折角仍有细亮缝，背板顶角略突出。', '主体形态保持，与 SEAM200 没有明显的整体优劣差；下部细缝仍在。'),
  ('板式椅主体完整，蓝色座面位置合理；背板表面线网改变，下部折角仍有亮缝。', '未看到 SEAM200 的背板顶角突出那么明显，但细缝并未普遍消失。'),
  ('座面用于椅身中部；前竖板和底板完整，底板前端存在小亮缺口。', '总体近似 SEAM200；底板前缘小缺口可见，不判为普遍改善。'),
 ],
 (1, 'RAW'): [
  ('矩形背板、蓝座面与支撑板齐全；座面下方剖面有短的额外灰线，表面边缘细碎。', 'REFERENCE'),
  ('完整矩形背板和支撑板；蓝座面下方灰色轮廓较平直，细小边缝仍存在。', '比 SEAM200 少见中部短灰线；整体形状和蓝 C 用途基本相同。'),
  ('蓝 C 对应座面，完整背板和底板可见；下层轮廓近似 LONG。', '与 SEAM200 比没有主体缺失；细节更规整但未见独特的采用收益。'),
  ('完整板式椅形，蓝座面和前竖板邻接处仍有细边缘线。', '与 LONG/ANTI_COVER 很接近；没有明显的整体款式或采用变化。'),
 ],
 (1, 'POST'): [
  ('蓝座面、背板和前支撑板齐全；座面下方剖面仍有一小段额外灰线，底部连接附近有亮缝。', 'REFERENCE'),
  ('板式主体完整，座面前缘出现较显眼的细白带；底板仍有局部亮缝。', '中部短灰线不明显，但前缘白带比 SEAM200 更清楚，不作单向改善判断。'),
  ('蓝座面仍被采用；背板和支撑板齐全，下部亮缝仍可见。', '相对 SEAM200 中部短灰线更少；没有消除所有细缝。'),
  ('矩形背板、蓝座面及底板完整；前缘薄白带和下部细缝仍见。', '整体近似；不能凭较整齐的边线认定连接或冲突指标改善。'),
 ],
 (2, 'RAW'): [
  ('多边形上靠背出现明显三角形开缝，另一视角左侧也见亮缝；蓝座面、下部支撑板和底板齐全。', 'REFERENCE'),
  ('多边形靠背在对应区域更连续，蓝座面接在弯折前缘处；下部主体保留。', 'SEAM200 的上靠背明显开缝在此列不明显；座面下层轮廓仍在。'),
  ('上靠背大面连续，蓝座面和弯折前支撑处外观连续；下部板式形态完整。', '上靠背开缝比 SEAM200 减少；视觉上与 LONG 很接近。'),
  ('多边形靠背及底板完整；蓝座面前缘与灰色圆折边连成椅身轮廓。', '上靠背开缝不再明显；未见相对 LONG/ANTI_COVER 的清楚差异。'),
 ],
 (2, 'POST'): [
  ('上靠背三角形亮缝在 POST 仍明显，侧视角也有开口；座面和下部椅身存在。', 'REFERENCE'),
  ('对应上靠背区域较连续；蓝座面前缘及前竖板构成完整轮廓，下部仍有细边缝。', '明显保留了上靠背开缝较少的视觉改善，不等于全部细缝被修复。'),
  ('靠背大面连续，蓝 C 参与座位，前支撑和底板齐全。', '相对 SEAM200 上靠背更完整；与 LONG 的可见外形近似。'),
  ('上靠背与前支撑轮廓完整，蓝座面和前弯折边外观连续。', '改善主要也是上靠背开缝减少；没有清楚的额外整体收益。'),
 ],
 (3, 'RAW'): [
  ('多边形靠背、蓝座面和下部板式支撑完整；上缘有小亮缺口，侧剖面的座面前端有额外斜短线。', 'REFERENCE'),
  ('多边形背板与蓝座面完整；前折边轮廓较简洁，上缘仍有很小缺口。', 'SEAM200 前缘额外斜短线不明显，整体外观变化较小。'),
  ('蓝座面用于完整椅身，背板上端有微小边缘缺口，下部轮廓规整。', '相对 SEAM200 前缘更规整；未见明显优于 LONG 的形状变化。'),
  ('完整板式椅形，蓝座面前缘与灰色弯折板外观连续；上靠背边缘仍有微小缺口。', '主体和 C 用途与 SEAM200 相同；细线较规整但变化有限。'),
 ],
 (3, 'POST'): [
  ('多边形背板与座面完整，上缘小缺口仍见；底板折角有细亮缝，前缘额外斜线减弱。', 'REFERENCE'),
  ('多边形靠背、蓝座面和前支撑齐全；上缘小缺口和下部细缝仍见。', '与 SEAM200 更接近；未见清楚的整体质量优势。'),
  ('座面、背板和底板保留；前缘外观连续，下部边缝仍见。', '相对 SEAM200 未见明显主体变化；仍不能判定细缝全部消失。'),
  ('蓝座面前折边与灰色支撑外观连续；上靠背和底部仍有很小的边缘缝。', '与其他长训臂外观相近；没有明显的独特视觉优势。'),
 ],
}


def main():
    schema=json.loads((EVAL/'review_schema.json').read_text(encoding='utf-8-sig'))
    rows=[]
    for (noise,stage),observations in OBS.items():
        request=f'73b369b9_n{noise}'
        path=EVAL/'gallery'/f'{request}_{stage}.png'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        for model,(observation,change) in zip(MODELS,observations):
            uncertain_layers=(model=='SEAM200' and (noise==1 or (noise==3 and stage=='RAW')))
            rows.append(dict(request_id=request,parent_id=PARENT,noise_index=noise,stage=stage,model=model,
                reviewer='AI_PROXY_SEAM_PORTABLE_CORE',review_kind='AI_PROXY',review_status='ACTUALLY_VIEWED',
                blinding='NONBLIND_MODEL_LABELS_VISIBLE',C_adoption='YES',whole_body_complete='YES',
                support_missing='NO',fragments_or_folded_layers='UNCERTAIN' if uncertain_layers else 'NO',
                visible_cracks='YES' if (noise==2 and model=='SEAM200') else 'UNCERTAIN',
                visible_join='YES' if noise in (2,3) else 'UNCERTAIN',
                visible_observations=observation,change_vs_SEAM200=change,
                uncertainty=('蓝 C 可遮住近共面的灰面；下层轮廓可能是正常厚度或重层，不能凭图区分。'
                '细亮缝及上缘微小缺口有尺度/开口解释歧义；YES 接通仅指可见前折边外观连续。'
                '没有显著缺腿不证明隐藏结构或承载安全，未认证索引共边、无交或水密。'),
                image_path=path.relative_to(ROOT).as_posix(),image_sha256=digest))
    assert len(rows)==32 and len({(r['request_id'],r['stage'],r['model']) for r in rows})==32
    destination=EVAL/'reviews';destination.mkdir(exist_ok=True)
    with (destination/'73.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=schema['csv_columns']);writer.writeheader();writer.writerows(rows)
    note='''# 73 父实际图审

评阅者：AI_PROXY_SEAM_PORTABLE_CORE。非盲 AI_PROXY，模型标签可见，不是人工评分。
已逐页实际打开 73b369b9 的 n0–n3、RAW/POST 共 8 张 PNG，逐列检查 SEAM200、LONG、ANTI_COVER、LOCAL_NOISE，共 32 行。图像 SHA256 随每行保存；没有用数值评分替代图审。

四臂在这四份噪声中都给出可辨认的完整板式椅形，蓝色 C 明显承担座面用途。没有出现明显的大范围主体或支撑缺失。各模型整体外观差异很小，不能从本父图像宣称 ANTI_COVER 或 LOCAL_NOISE 有稳定独有优势。

最清楚的可见变化在 n2：SEAM200 上靠背的大三角开缝在 RAW 和 POST 都能看到，而三条长训臂对应区域更连续。n1 的 SEAM200 座面下方剖面有短的额外灰线，新臂更少见；但这条线不足以单独认证碎片或非法重层。n3 的变化主要是前折边和顶部细线，整体接近。

n0/n1 的 C-to-free 边缘包含细白带或被遮挡，仅记 visible_join=UNCERTAIN；n2/n3 在渲染尺度上能看到前折边轮廓连续，记 YES，绝不等同于已验证合法共享索引。常见下层平行轮廓、底部细缝、顶部小缺口保留不确定，不把普通开放式椅身当成缺腿，也不把所有近邻面当成冲突。

蓝 C 会遮挡近平面灰面，图片无法认证无重叠、无相交、流形、水密或受力安全。NO 仅表示这些视角未见明显缺陷。当前图审只覆盖四条已见 TRAIN 条件中的一个父，不支持新父泛化结论。
'''
    with (destination/'73.md').open('x',encoding='utf-8') as f:f.write(note)
    print(json.dumps(dict(status='COMPLETE',pages=8,rows=32,actual_images_viewed=True,
        neural_model_calls=0,csv=str(destination/'73.csv')),ensure_ascii=False))

if __name__=='__main__':main()
