"""Serialize actual image observations of combination pages 42..61; no classifier."""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G = json.loads((ROOT / 'gallery/combinations/manifest.json').read_text('utf-8'))
M = json.loads((ROOT / 'combinations/manifest.json').read_text('utf-8'))
byid = {r['candidate_id']: r for r in M['rows']}
# Explicit observed combination scope per image row. All images were newly
# viewed after composition; acceptance is visual semantics, not topology PASS.
NOTES = {
 'afe3b299': [
  ('SEAT_SUPPORT','一条窄座板表面与一根支柱侧表面','组合给出板的长度及一处支撑高度。C-only可见两部分有间隔，作为多部位指定输入合理，不称已接合。'),
  ('PART_ASSEMBLY','一条窄座板表面与两根支柱侧表面','两个柱面对应两端真实支撑位置，整体关系清楚；不是完整柱或整座板。')],
 'b71a0cca': [
  ('SEAT_SUPPORT','完整座板上表面与一根三维台阶柱腿','一处立柱和顶板的关系在组合C-only中可辨，尚无全部腿。'),
  ('PART_ASSEMBLY','完整座板上表面与两根三维台阶柱腿','两柱位于座板两端的同一侧，剩余支撑和横梁仍留Y。')],
 'b9535bf9': [
  ('SEAT_SUPPORT','一条座板表面与一侧倒U形腿架','座板和腿架平行且沿原位置分开，结构意义清楚，不误称整个凳面。'),
  ('PART_ASSEMBLY','一条座板表面与前后两副倒U腿架','组合含四条腿的两架及单条座板，其他条板待补；有充分支撑关系信息。')],
 'bb07fac5': [
  ('SEAT_SUPPORT','整块座面与一侧两腿横梁表面','C-only可看出座面及一侧腿框，侧框和座面边缘关系一致。'),
  ('PART_ASSEMBLY','整块座面与前后两侧腿框表面','双侧框架使座面高度及腿分布清晰，但没有底部横撑。')],
 'bf504bde': [
  ('SEAT_SUPPORT','座面与一个L形腿框侧面','视觉部位关系清楚，C-only边角保留原缝/缺口；视觉采用不代替非流形审计。'),
  ('PART_ASSEMBLY','座面与两侧L形支撑表面','两个侧边腿框使左右支撑关系清晰，扶手端板仍灰色；需独立拓扑门禁，不认证无缺陷。')],
 'c2365f10': [
  ('SEAT_SUPPORT','长座板平顶面与一根横向圆木底座','圆柱支撑和顶板为明确两部位输入；座板弧形下表面留Y。'),
  ('PART_ASSEMBLY','长座板平顶面与两根横向圆木底座','两圆木的方向及位置对称合理，构成明确的座面和支撑指定。')],
 'c3ce87b9': [
  ('SEAT_SUPPORT','完整宽座面与一个支脚内侧矩形面','C-only显示竖直小面与座面断开，但整图明确是支脚内侧，属于多部位输入，不当随机悬片。')],
 'c3d8c664': [
  ('SEAT_BACK','完整座面与高靠背内表面','两个宽面构成清楚座背关系，局部C-only的反视角有座面后伸，不改变来源。')],
 'c50aa1c3': [
  ('SEAT_SUPPORT','完整座面与一个端部叉腿支撑侧面','座板与下方叉腿之间原有厚度间隙仍可见，不假称源ID已焊接。'),
  ('PART_ASSEMBLY','完整座面与两端叉腿支撑侧面','两个端面支撑对称排列，均有明确部位意义；只是侧面并非整副实体腿。')],
 'c97683ce': [
  ('SEAT_SUPPORT','完整座面与一侧圆角纵向框架','圆角框和座面在同一尺度下关系清楚，框下梁不是靠背。'),
  ('PART_ASSEMBLY','完整座面与两侧圆角纵向框架','两框架加座面约束结构完整程度更高，底层搁板和端部填面仍留Y。')],
 'cdb6ceae': [
  ('SEAT_SUPPORT','完整座面与一个块状支脚的侧面','板和竖面之间存在来源板厚间隔，组合位置清晰。'),
  ('PART_ASSEMBLY','完整座面与两端块状支脚侧面','两竖面分处两支脚内外侧而非对称同向选择，仍是可解释的真实部件输入。')],
 'cedbe77b': [
  ('PART_ASSEMBLY','靠背上沿窄盖面与后柱及L形底框的侧表面','组合不含真正座面；顶部蓝条和两立柱给出背框结构，横向低段为底支撑。')],
 'd6580bbf': [
  ('SEAT_SUPPORT','中央一段座面与一个块状端部支撑','顶面局部向支撑延伸，保留部分座面而非整板。'),
  ('PART_ASSEMBLY','中央一段座面与两块端部支撑','C-only一侧块与座面组合分离，原本缺的另一段座面在Y；作为未完成部件布局可解释。')],
 'd9de3b77': [
  ('SEAT_BACK','窄座面条与上方长靠背板','组合为清楚的L形长凳座背表面，未包含下方支撑。')],
 'df31fdbb': [
  ('SEAT_SUPPORT','完整座面与一根细长方柱腿','C-only可看出顶面和一根原位柱腿的关系，横撑未给出。'),
  ('PART_ASSEMBLY','完整座面与两根细长方柱腿','同一侧两柱在顶板两端，提供支撑间距；其余腿/侧板留Y。')],
 'e8403ea9': [
  ('SEAT_SUPPORT','完整座面与一侧倒U形细腿框','侧框上横梁略低于座面，原有分离部位关系合理且未焊接。'),
  ('PART_ASSEMBLY','完整座面与前后两侧细腿框表面','两侧细腿框各有局部窄面，C-only可识别完整支撑布局；圆柱靠背留Y。')],
 'eedc7453': [
  ('SEAT_SUPPORT','一条长座板表面与一根木腿','一视图投影似相接，反视图能看到空间间隔；实际为两部位原位置输入。'),
  ('PART_ASSEMBLY','一条长座板表面与两根木腿','两木腿给出高度和跨度，窄座板未覆盖所有上方结构；输入关系合理。')],
 'f3e13747': [
  ('SEAT_SUPPORT','前沿一条座板表面与一端三角斜撑腿架','腿架横梁在座板下相交投影，三角撑结构在局部清楚。'),
  ('PART_ASSEMBLY','前沿一条座板表面与两端三角斜撑腿架','两端立足位置清楚，背板与其支柱仍待补；不是整模型重建输入。')],
 'f4b6526b': [
  ('SEAT_SUPPORT','完整圆角座面与一根立柱的侧面','C-only柱侧面因视角较窄仍可定位，整图对应真实支柱。'),
  ('PART_ASSEMBLY','完整圆角座面与两根立柱的侧面','两竖片都是已明确的柱面而非漂浮碎片；不声称包含柱体全部几何。')],
 'fd6ae258': [
  ('SEAT_SUPPORT','完整座面与一根倾斜支脚宽侧面','宽斜面在座板一端下方，提供倾角与高度。'),
  ('SEAT_BACK','完整座面与上方靠背横板后表面','两平面空间关系清楚且有间隙，背板后侧不应称正面。'),
  ('PART_ASSEMBLY','座面、靠背横板与两根斜支腿的侧面','部件关系在整图和C-only一致，视觉合理但未认证共边流形或连接完整。')],
}

results = []
for page in G['pages'][42:62]:
    parent = page['parent_id']; prefix = parent[:8]
    obs = NOTES[prefix]
    assert len(obs) == len(page['rows'])
    image = ROOT / 'gallery/combinations' / (parent + '.png')
    h = hashlib.sha256(image.read_bytes()).hexdigest()
    for item, (part, extent, note) in zip(page['rows'], obs):
        cid = item['candidate_id']
        results.append(dict(candidate_id=cid, parent_id=parent, group=byid[cid]['group'],
            decision='ACCEPT', actual_part=part, extent_description=extent, human_plausible_input='YES',
            notes=note, reviewer='AI_PROXY_bench_parts_quality', reviewer_type='AI_PROXY', blinding='NONE',
            image_path=str(image), image_sha256=h))
assert len(results) == sum(len(p['rows']) for p in G['pages'][42:62])
target = ROOT / 'reviews/combinations_c.json'
assert not target.exists()
target.write_text(json.dumps(dict(schema='BENCH_MULTI_PART_INPUT_ACTUAL_REVIEW_V1',
    page_indices_inclusive=[42,61], pages_actually_viewed=20, rows=results,
    protocol_sha256=hashlib.sha256((ROOT / 'VISUAL_REVIEW_PROTOCOL.json').read_bytes()).hexdigest(),
    note='Every combination page was actually newly viewed; base acceptance was not copied. ACCEPT concerns plausible part identity/composition only; independent quality QUARANTINE remains authoritative for ACTIVE.'),
    ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(rows=len(results), path=str(target))))
