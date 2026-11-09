"""Record root's actual eight-page image review; decisions below are manual."""
from pathlib import Path
import hashlib,json

OUT=Path(__file__).resolve().parents[1]
notes={
 '1ea895':[
  ('SEAT_SURFACE','单条宽座板的完整上表面，不是全座面；两三角规整矩形。'),
  ('LEG_SUPPORT','一端A形支架的连贯侧表面，包含两条斜腿及横向连接，不认证完整实体腿。'),
  ('LEG_SUPPORT','另一端A形支架侧表面；与第一端位置不同，形态相似。')],
 '36583b':[
  ('SEAT_SURFACE','座面外缘的一条很窄的连续板条上表面；矩形但不代表完整座面。'),
  ('LEG_SUPPORT','近端立柱的完整矩形侧面；只给一侧皮，不是完整柱体。'),
  ('LEG_SUPPORT','远端立柱的矩形侧面；完整腿侧面。')],
 '381317':[
  ('SEAT_SURFACE','C-only为宽矩形平面，但两幅完整来源图均被灰座板遮挡；上表面身份不清，暂不作为新示范。'),
  ('LEG_SUPPORT','两条细腿加上部横梁构成连贯U形框架，位于长边一侧。'),
  ('LEG_SUPPORT','另一长边的完整U形细框架，形态规整。')],
 '50ef39':[
  ('SEAT_SURFACE','清楚覆盖完整宽矩形座板顶面，两三角输入。'),
  ('LEG_SUPPORT','长边侧梁及其两端腿的侧表面；一侧腿在该视角近乎侧视线状，不据此判几何退化。'),
  ('LEG_SUPPORT','另一长边的侧梁及两端腿侧表面，连贯但不等于闭合实体。')],
 '65aa81':[
  ('SEAT_SURFACE','一条窄而规整的座板上表面；源长椅有多条平行板。'),
  ('LEG_SUPPORT','端部完整T形支撑，包含立柱与座下横向承托板。'),
  ('LEG_SUPPORT','中部完整T形支撑，包含立柱与上部横向承托板。')],
 '6b894c':[
  ('SEAT_SURFACE','转角长椅连续L形座面表面，4三角表达，两翼都实际可见。'),
  ('LEG_SUPPORT','长翼座下长边框侧表面与两腿侧表面，属于一侧支撑框。'),
  ('LEG_SUPPORT','短翼座下支撑框侧表面；不是整把长椅底架。'),
  ('BACK_SURFACE','长翼靠背内侧宽面板，位于座面上方，完整局部靠背表面。')],
 '7b7b25':[
  ('SEAT_SURFACE','宽座板上表面含边缘分片，整体连续，不是随机散面。'),
  ('LEG_SUPPORT','弧形座下侧梁及两端腿的窄侧面；不是完整实体腿。'),
  ('LEG_SUPPORT','另一侧弧形座下梁及两端腿侧面。')],
 '891b8a':[
  ('SEAT_SURFACE','完整宽座面表面，边缘有分片，视觉连贯。'),
  ('LEG_SUPPORT','完整倾斜矩形腿柱，侧面和顶盖清楚，部件用途合理。'),
  ('LEG_SUPPORT','另一端倾斜完整腿柱；视觉合理，但已有几何QUARANTINE必须保留剔除。')],
}

def main():
    manifest=json.loads((OUT/'candidates/manifest.json').read_text('utf-8'))
    records=[]
    for prefix, entries in notes.items():
        rows=[r for r in manifest['rows'] if r['parent_id'].startswith(prefix)]
        assert len(rows)==len(entries),(prefix,len(rows))
        pid=rows[0]['parent_id'];im=OUT/'gallery/data/candidates'/(pid+'.png')
        digest=hashlib.sha256(im.read_bytes()).hexdigest()
        for row,(part,note) in zip(rows,entries):
            uncertain=prefix=='381317' and row['candidate_id'].endswith('SEAT_SURFACE_1')
            records.append(dict(candidate_id=row['candidate_id'],parent_id=pid,group='TRAIN',
                decision='UNCERTAIN' if uncertain else 'ACCEPT',actual_part=part,
                extent_description=note,human_plausible_input='UNCERTAIN' if uncertain else 'YES',
                notes=note,reviewer='AI_PROXY_ROOT_NONBLIND',actually_viewed=True,
                image_path=str(im),image_sha256=digest,source_unique_style_as_success=False))
    assert len(records)==25
    with (OUT/'preparation/visual_reviews_root.json').open('x',encoding='utf-8') as f:
        json.dump(dict(rows=records,actual_pages=8,review_date='2026-10-21'),f,ensure_ascii=False,indent=2)
    print('RECORDED 25 conditions from 8 actually viewed pages')

if __name__=='__main__':main()
