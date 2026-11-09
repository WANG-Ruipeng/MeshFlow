"""Serialize manually recorded observations after viewing all assigned PNGs.

No image classification is performed by this script. The literal judgments
below were entered after the reviewer actually viewed the four-panel sheets.
"""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / 'candidates/manifest.json').read_text('utf-8'))
G = json.loads((ROOT / 'gallery/candidates/manifest.json').read_text('utf-8'))
# key prefix: suffix, actual part, extent, observation, optional uncertainty
MANUAL = {
 'afe3b299': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','一条完整窄座板的上表面','蓝色长条位于条板座面的外缘，形状规则；不是整块座面。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','一根立柱支撑的一侧表面，带底部窄边','整图中蓝面位于右侧承重立柱；C局部是竖直板面而非完整实心腿。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','另一根立柱支撑的一侧表面，带底部窄边','整图中蓝面位于左侧立柱；两端位置关系明确。')],
 'b71a0cca': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','完整长矩形座面上表面','两视图均显示蓝面覆盖长凳上板，保留完整外轮廓。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','一根带凹槽台阶的三维支撑柱','C有多个相接侧面和顶面，位于座面下；不是平面切片。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','另一根带凹槽台阶的三维支撑柱','与另一端对应的支撑构件，局部几何完整可解释。')],
 'b9535bf9': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','一条完整长座板上表面','蓝面横贯条板座面中部，其他座板仍为灰色。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','纵向倒U形腿架，含两腿和上横梁','C-only显示连续三维腿架；横梁为座下结构。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','另一侧纵向倒U形腿架','前后互补位置清楚，不是随机悬片。')],
 'bb07fac5': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','整块承坐面，边缘带窄安装延伸','蓝面是座位中央完整承坐区域；端部窄延伸保留来源，没有改造成矩形。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','纵向支撑框侧表面，含两竖腿和横梁','C连续表达框架的一侧表面，不能称完整实体框。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','对侧纵向支撑框侧表面','对侧位置和框架角色可见；不是靠背。')],
 'bf504bde': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','座面上表面及端部窄耳片','两端扶手之间的承坐平面清楚，边界非纯矩形但可解释。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','一侧L形竖腿与短上横梁的表面','局部C由竖直支撑和座下横向段组成，未包含整个侧框。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','另一侧L形竖腿与短上横梁的表面','同样是部分支撑框表面，整体位置清晰。'),
  ('BACK_SURFACE_1','OTHER','座面端部扶手板的一侧表面','蓝面位于长凳短边端部的扶手，不在长边背部；保留为端板输入，不能计入靠背。')],
 'c2365f10': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','半圆木座板的完整平顶面','平面承坐上表面位于两圆木支撑之上，角色清楚。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','一根横向圆木底座支撑','C-only呈有端盖的三维圆柱，整图在座板下端部。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','另一根横向圆木底座支撑','第二端圆木支撑清楚，并非狭长随机碎片。')],
 'c3ce87b9': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','靠背与扶手之间的完整座面','正面视角显示清楚，C-only为规则宽矩形面。'),
  ('LEG_SUPPORT_1','UNKNOWN','矩形竖直面，整图位置被遮挡','两个整物视图中几乎看不到蓝面，无法仅凭局部矩形确定其属于哪根腿或内部表面；保留不确定。','UNCERTAIN'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','一根块状支脚的内侧矩形表面','反面视图能看到左下支脚内侧蓝矩形；只是支脚一侧。')],
 'c3d8c664': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','完整座面上表面','正面视角蓝面处于高靠背下，两端扶手之间。'),
  ('BACK_SURFACE_1','BACK_SURFACE','靠背朝向座位的整块矩形表面','正面视图显示蓝色完整靠背，反面被板厚遮挡合理。')],
 'c50aa1c3': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','完整矩形座板上表面','上表面及外轮廓完整明确。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','端部叉腿支撑的一侧表面','C由上宽板面与下部两分叉腿构成，凹口在来源中可见。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','另一端叉腿支撑的一侧表面','与第一端对称的承重端面，不称实心完整腿架。')],
 'c6ccf94d': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','顶层完整长座面','座储一体长凳顶板表面清楚。'),
  ('LEG_SUPPORT_1','SEAT_SUPPORT','座面、底板和三格支撑隔板组成的大范围主体','该C包含整个座板和储物格支撑，仅细脚留在Y；不能当单腿输入统计。')],
 'c97683ce': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','顶层长座面平面','矩形蓝上表面位于弯曲端部支架之间。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','带圆弧上角的纵向矩形框侧表面','一整片连续外框面，包含上下横梁及两立腿。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','对侧带圆弧上角的纵向框侧表面','两整物视图分别显示该框的一侧位置。')],
 'cdb6ceae': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','完整座面上表面','粗块状两腿之上的规则承坐面。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','一端块状支撑的外侧面','竖向矩形蓝面与整体粗支脚对应，不含支脚其余侧面。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','另一端块状支撑的内侧面','蓝面处于右端支脚内面；可作用户指定侧面输入。')],
 'cedbe77b': [
  ('SEAT_SURFACE_1','BACK_SURFACE','靠背上沿的窄水平盖面','蓝条明显处于靠背顶部，高于真正灰色座面；必须从SEAT更正为靠背上沿。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','从短前脚到底部侧梁再到后立柱的L形侧框表面','整体图可识别下部支撑兼靠背立柱；不是完整背板。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','后支脚向上延伸至靠背的立柱侧面','蓝色高立柱贯穿座下和靠背高度，保留准确支撑柱范围。')],
 'd6580bbf': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','中央一段矩形座面上表面','只选长座面中段，蓝面边界规则而非整座面。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','一个完整块状端部支撑','多个侧面及顶面构成三维端部支撑，位于长板一端。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','另一个完整块状端部支撑','对应另一端的三维块状脚，完整程度与前者相当。')],
 'd9de3b77': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','座面前部一条长矩形表面','蓝条位于靠背下方承坐高度，非顶部靠背横梁。'),
  ('LEG_SUPPORT_1','SEAT_SUPPORT','长座体及下方多个板式支撑的组合','蓝C包含座板及多个下垂支撑，不能称仅腿部；是较丰富的组合输入。'),
  ('BACK_SURFACE_1','BACK_SURFACE','靠背正面长条板表面','竖直蓝板位于座面上方长边，靠背角色清楚。')],
 'de008382': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','前侧一条宽座板的完整上表面','前方承坐板面蓝色，背框/扶手仍灰色。')],
 'df31fdbb': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','完整上层承坐板面','蓝顶面横贯物体，上缘后挡板不是本C。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','一根完整细长方柱腿','C-only为六面方柱，整图可见其伸至地面的部分。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','另一根完整细长方柱腿','第二根立柱腿位置清晰，源面保持原尺寸。')],
 'e7bb6ef5': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','半圆木座体的完整平顶面','两个横向小圆木托住的上方平面座板。'),
  ('LEG_SUPPORT_1','SEAT_SURFACE','完整半圆木座体表面，含上平面和弧形底面','候选实际把整根承坐半圆木选出，下面两小圆木支撑仍灰色；应从LEG改为座体表面，非单腿。')],
 'e8403ea9': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','完整长座板上表面','圆柱靠背下方的平面板，角色明确。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','一侧倒U形细腿框架','横梁与两竖腿形成连续细长框架，位于座板下方。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','对侧倒U形框架的窄侧表面','C-only竖段较薄且有锥形端部；与原细腿框对应，只是表面范围而非完整实体。')],
 'eedc7453': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','一条长座板上表面','两视图均能定位于条板承坐部分。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','一根完整矩形木腿','C为三维方柱，整图位于座板下方并直达地面。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','另一根完整矩形木腿','对应另一端木腿，无需假设靠背语义。')],
 'f3e13747': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','座面前沿一条完整座板上表面','前视蓝条与座板轮廓一致，背板仍灰色。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','端部三角撑腿框，含横梁和斜撑','三维C含可见三角空洞和斜撑，与整凳下部端架对应。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','另一端三角撑腿框','同类结构在另一端，完整框范围可辨。')],
 'f4b6526b': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','完整座面上表面及端部圆角细分','长条整体上表面蓝色，端部轮廓完整。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','一根支柱的窄竖直侧面','整图可见支撑柱侧面，C仅一个矩形面，不含底座。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','另一根支柱的窄竖直侧面','第二根柱侧面位置明确，非任意悬空片。')],
 'fd6ae258': [
  ('SEAT_SURFACE_1','SEAT_SURFACE','两端斜腿之间的完整座面板面','蓝色承坐板面下有叉形支撑，上方另有靠背横板。'),
  ('LEG_SUPPORT_1','LEG_SUPPORT','一根倾斜支腿的宽侧表面','整图右侧后方支腿蓝面明确，局部为倾斜矩形面。'),
  ('LEG_SUPPORT_2','LEG_SUPPORT','另一根倾斜支腿的窄侧表面','窄条位于左前支腿边侧，几何细长但有明确部件位置。'),
  ('BACK_SURFACE_1','BACK_SURFACE','靠背横板的后侧表面','反面视图蓝色完整上横板，正面被板体遮挡，明确是背部表面。')],
}

byid = {row['candidate_id']: row for row in M['rows']}
pages = {page['parent_id']: page for page in G['pages']}
results = []
for prefix, observations in MANUAL.items():
    parents = [pid for pid in pages if pid.startswith(prefix)]
    assert len(parents) == 1
    parent = parents[0]
    image_path = ROOT / 'gallery/candidates' / (parent + '.png')
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    for observation in observations:
        suffix, part, extent, notes = observation[:4]
        decision = observation[4] if len(observation) > 4 else 'ACCEPT'
        cid = parent + '__' + suffix
        assert cid in byid
        results.append(dict(candidate_id=cid, parent_id=parent, group=byid[cid]['group'],
            decision=decision, actual_part=part, extent_description=extent,
            human_plausible_input='UNCERTAIN' if decision == 'UNCERTAIN' else 'YES', notes=notes,
            reviewer='AI_PROXY_bench_parts_quality', reviewer_type='AI_PROXY', blinding='NONE',
            image_path=str(image_path), image_sha256=image_hash))
expected = {row['candidate_id'] for page in G['pages'][48:71] for row in page['rows']}
assert {r['candidate_id'] for r in results} == expected, (len(results), len(expected))
target = ROOT / 'reviews/base_c.json'
assert not target.exists()
target.write_text(json.dumps(dict(schema='BENCH_MULTI_PART_INPUT_ACTUAL_REVIEW_V1',
    page_indices_inclusive=[48,70], pages_actually_viewed=23, rows=results,
    protocol_sha256=hashlib.sha256((ROOT / 'VISUAL_REVIEW_PROTOCOL.json').read_bytes()).hexdigest(),
    note='All assigned PNGs were actually viewed using view_image before entering these literal observations. This script only serializes and attaches identifiers/hashes.'),
    ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(rows=len(results), output=str(target))))
