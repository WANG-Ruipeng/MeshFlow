"""Transcribe actual AI image reviews; no model/scorer calls and no metric input.

All 16 source pages were opened at original resolution before these records
were written. Labels are manual visual observations, not programmatic inference.
"""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/r2_seam_abc_v1/evaluation'
SCHEMA = json.loads((BASE / 'review_schema.json').read_text(encoding='utf-8'))
MODELS = ['LONG', 'A_COUPLE', 'B_POOL47', 'C_SHARED_TIP']
LABEL_FIELDS = ['C_adoption', 'whole_body_complete', 'support_missing',
                'fragments_or_folded_layers', 'visible_cracks', 'visible_join']
Y, N, U = 'YES', 'NO', 'UNCERTAIN'
# Each entry is: six schema labels; actual visible observations; same-page delta.
PAGES = {
('73', 0, 'RAW'): [
 ((Y,Y,N,N,Y,U), 'Blue C is the seat of a rectangular-back cantilever chair. A narrow gap is visible at the lower back and a white strip at the front seat rim.', 'REFERENCE'),
 ((Y,Y,N,N,Y,U), 'Rectangular back, front support and broad base remain present. The lower-back gap and narrow front rim separation remain visible.', 'Very close to LONG; no clear closure of the visible seat rim gaps.'),
 ((Y,Y,N,N,Y,Y), 'Blue seat and rectangular back appear continuous at the rear at this scale. A narrow white strip remains at the front seat/support boundary.', 'Rear seat-to-back junction appears more filled than LONG; front rim separation persists.'),
 ((Y,Y,N,N,Y,Y), 'Rounded faceted back and complete cantilever body. Blue C forms the seat; rear contact appears continuous, with a thin front rim separation.', 'Back style becomes rounded; rear junction looks more continuous than LONG, without a clear front-rim repair.'),
],
('73', 0, 'POST'): [
 ((Y,Y,N,N,Y,U), 'Rectangular-back cantilever body is present. The blue seat still has a visible lower-back gap and narrow white front strip.', 'REFERENCE'),
 ((Y,Y,N,N,Y,U), 'Complete rectangular-back body; a small triangular part of the lower back is more filled, but the seat/back and front-rim gaps remain.', 'Small local filling near the back; overall appearance and remaining gaps are close to LONG.'),
 ((Y,Y,N,N,Y,Y), 'Rear blue-seat/back junction looks continuous. A narrow white front seat strip remains; base and front support are present.', 'Rear junction is visually more continuous than LONG, while the front strip remains.'),
 ((Y,Y,N,N,Y,Y), 'Rounded back, broad base and front support remain complete. Blue C is the seat with apparent rear contact and a thin front rim gap.', 'Rounded style and rear contact differ from LONG; no obvious new detached fragment.'),
],
('73', 1, 'RAW'): [
 ((Y,Y,N,N,Y,Y), 'Blue C forms the seat and meets the rectangular back visually. A thin front seat strip and a small white notch at the lower base corner are visible.', 'REFERENCE'),
 ((Y,Y,N,N,Y,Y), 'Rectangular-back chair retains its support and base. Rear contact looks continuous; a narrow front seat gap remains visible.', 'Broadly similar to LONG; no clear new adoption or completeness benefit.'),
 ((N,Y,N,Y,U,N), 'A raised gray seat is visibly above blue C, which remains as a lower strip. The frame chair body is complete, but the two seat levels are distinct in the central cuts.', 'Changes to a frame chair and introduces a distinct generated seat above C; C is no longer the visible upper seating surface.'),
 ((Y,Y,N,Y,Y,Y), 'Blue C is the seat of a rounded-back cantilever chair. A long thin diagonal sliver crosses the base plate; a narrow front seat gap remains.', 'Rounded back and a visible extra base sliver differ from LONG; seat use remains plausible.'),
],
('73', 1, 'POST'): [
 ((Y,Y,N,N,Y,Y), 'Rectangular-back body remains complete. Rear seat contact appears continuous; the front white strip and small lower-base notch remain visible.', 'REFERENCE'),
 ((Y,Y,N,N,Y,Y), 'Blue seat, back and cantilever base are present. The narrow front seat strip is still visible.', 'Similar to LONG with no clear adoption change or large gap repair.'),
 ((N,Y,N,Y,U,N), 'The gray seating surface remains above blue C after processing. C is visible below the frame chair seat, with two distinct levels in the cuts.', 'The replacement-seat pattern relative to LONG persists; processing does not bring the upper gray seat onto C.'),
 ((Y,Y,N,N,Y,Y), 'Rounded-back chair remains complete. The conspicuous diagonal base sliver from RAW is absent; a thin front seat gap remains.', 'Base looks cleaner than its RAW view; relative to LONG it mainly changes back style and preserves blue-seat use.'),
],
('73', 2, 'RAW'): [
 ((Y,Y,N,N,N,Y), 'Blue C is the seat of a rounded-back cantilever chair. Back, front support and base look coherent; no clear large tear is visible.', 'REFERENCE'),
 ((Y,Y,N,N,N,Y), 'Rounded back and base remain coherent, with blue C visibly serving as the seat. No clear detached part or large tear is visible.', 'Very similar to LONG, with small changes in panel triangulation and outline.'),
 ((Y,Y,N,N,Y,Y), 'Blue seat and support are present, but a conspicuous white wedge interrupts the upper-left back panel.', 'A visible upper-back opening appears where LONG has a more continuous panel.'),
 ((Y,Y,N,N,Y,Y), 'Blue C is the seat. A narrow white slit is visible in the upper back and a thin separation appears at the front seat rim.', 'Adds an upper-back slit and a more apparent front rim separation relative to LONG.'),
],
('73', 2, 'POST'): [
 ((Y,Y,N,N,N,Y), 'Rounded-back cantilever chair remains coherent, with blue C at the seat. No clear large detached piece or torn panel is visible.', 'REFERENCE'),
 ((Y,Y,N,N,N,Y), 'Blue seat, rounded back, front support and base remain present. The panels appear coherent at this rendered scale.', 'Close to LONG; no clear difference in overall completeness.'),
 ((Y,Y,N,N,Y,Y), 'The upper-left white wedge in the back remains visible. The blue seat and supporting cantilever body remain present.', 'Upper-back tear remains more apparent than in LONG.'),
 ((Y,Y,N,N,Y,Y), 'The upper-back slit is less apparent than in RAW, but a narrow white front seat gap remains. Body and blue-seat use remain plausible.', 'Near LONG in overall shape, with a more visible front rim separation.'),
],
('73', 3, 'RAW'): [
 ((Y,Y,N,N,U,Y), 'Rounded-back chair with blue seat and complete cantilever body. Fine irregular lines near the upper back may be small gaps or panel edges.', 'REFERENCE'),
 ((Y,Y,N,N,U,Y), 'Blue seat and rounded back remain coherent. A tiny upper-back notch or slit is difficult to separate from tessellation at this scale.', 'Near LONG; no clear large change in body or seat use.'),
 ((Y,Y,N,U,U,Y), 'Blue seat and support remain present. A small upper-back pinstripe and doubled diagonal traces on the front support suggest a possible local extra layer.', 'More doubled front-support traces than LONG, but their layer status is visually ambiguous.'),
 ((Y,Y,N,N,U,Y), 'Rounded back and complete base remain. Blue C is the seat; a very thin line at the front rim may be a small gap or ordinary thickness.', 'Broadly similar to LONG with a somewhat more apparent front rim line.'),
],
('73', 3, 'POST'): [
 ((Y,Y,N,N,N,Y), 'Rounded back panel appears more continuous than in RAW. Blue seat and cantilever support/base are present, without an obvious large tear.', 'REFERENCE'),
 ((Y,Y,N,N,N,Y), 'Back panel and supporting body appear coherent. Blue C remains the seat, with no obvious large detached patch.', 'Similar to LONG in completeness and seat use.'),
 ((Y,Y,N,U,N,Y), 'The upper back looks closed at this scale. Doubled diagonal traces remain on the front support, with uncertain layer interpretation.', 'Possible extra front-support layer remains more apparent than in LONG; no obvious missing body.'),
 ((Y,Y,N,N,U,Y), 'Blue seat and rounded-back body are intact. A fine front rim line remains, too small to determine whether it is a crack.', 'No clear completeness change from LONG; front rim continuity remains less certain.'),
],
('2d', 0, 'RAW'): [
 ((Y,Y,N,U,Y,U), 'Blue C lies at the seating surface of a complete sled-frame chair. Gray surfaces curve well below C; the left underside has broken white slits and several close lower traces.', 'REFERENCE'),
 ((Y,Y,N,U,Y,U), 'The blue top seat, back and sled frame are present. Several lower gray traces and small side-underseat openings remain visible.', 'Similar to LONG; underside traces are arranged differently without a clear repair.'),
 ((Y,Y,N,U,U,U), 'A complete frame chair surrounds the blue seat. The back/underside takes a deeper angular bend, with a relatively continuous side panel but uncertain close lower layers.', 'Deeper angular seat/back shell than LONG; less visibly ragged side edge, but the seat-to-lower-shell relationship remains ambiguous.'),
 ((Y,Y,N,U,Y,U), 'Blue C forms the top seat of a complete frame chair. The lower gray shell has multiple nearby traces and a small open slit on the left side.', 'Close to LONG with slightly different underside curvature; no clear removal of the side separation.'),
],
('2d', 0, 'POST'): [
 ((Y,Y,N,U,U,U), 'Blue upper seat and full sled frame remain. The underside is cleaner than RAW but still lies well below C, with small side openings difficult to classify.', 'REFERENCE'),
 ((Y,Y,N,U,U,U), 'Blue seat, back and frame remain complete. Lower-shell traces are smoother; small left-side gaps remain ambiguous.', 'Similar to LONG, without a clear new seat-adoption or completeness change.'),
 ((Y,Y,N,U,U,U), 'The angular back/underside shell remains deeper than C and the frame is complete. No obvious large detached panel is visible.', 'Different deeper shell style than LONG; local continuity under blue C remains uncertain.'),
 ((Y,Y,N,U,U,U), 'Frame chair remains complete with blue upper seat. Small left-side shell openings and close lower traces remain after processing.', 'Broadly similar to LONG; no clear visual advantage in hidden seat layers.'),
],
('2d', 1, 'RAW'): [
 ((Y,Y,N,N,Y,Y), 'Blue C is the seat above two broad panel supports and an upright back. A white triangular gap is visible at the left seat-side junction.', 'REFERENCE'),
 ((Y,Y,N,N,U,Y), 'Blue seat and full panel supports are present. The left seat-side transition looks more filled; remaining narrow traces are ambiguous.', 'Left seat-side gap is less apparent than LONG; overall chair style remains similar.'),
 ((Y,Y,N,N,U,Y), 'Blue C forms the seat of a panel-supported chair. Back and supports are present; the left side appears mostly filled at this scale.', 'Less obvious left-side opening than LONG, with otherwise similar complete body.'),
 ((Y,Y,N,N,Y,U), 'Blue seat and rounded-underseat shell sit above a cantilever base. A large white triangular opening is visible at the left seat-side transition.', 'Changes to a cantilever base; the large side opening remains, and exact visible seat contact is less clear than LONG.'),
],
('2d', 1, 'POST'): [
 ((Y,Y,N,N,Y,Y), 'Panel-supported chair is complete with blue C as its seat. The left seat-side triangular opening is still visible.', 'REFERENCE'),
 ((Y,Y,N,N,U,Y), 'Back and two panel supports remain complete. The left seat-side region appears more filled than LONG; very small seams remain unresolved visually.', 'The visible side opening is reduced relative to LONG; no clear change in overall seat use.'),
 ((Y,Y,N,N,U,Y), 'Blue seat and panel-supported body remain complete. Only small side seams are apparent, without an obvious missing support.', 'Similar complete body to LONG with a less apparent left seat-side gap.'),
 ((Y,Y,N,N,Y,U), 'Cantilever body remains recognizable and complete. A large left-side triangular gap and a narrow seat/front-support separation persist.', 'Different support style and more visible side separation than LONG; processing does not close the large opening.'),
],
('2d', 2, 'RAW'): [
 ((Y,N,Y,Y,Y,U), 'Blue seat and upper back are recognizable, but the lower supports break into many horizontal strips, slivers and open holes.', 'REFERENCE'),
 ((Y,N,Y,Y,Y,U), 'Blue C is still the seat. The front and side supports are heavily split into overlapping-looking strips with large missing intervals.', 'Retains the fragmented lower-support failure of LONG; no clear whole-body recovery.'),
 ((Y,Y,N,U,U,Y), 'Blue C sits in a complete sled-frame chair. The lower support is not a stack of broken plates; small underside slivers or doubled traces remain ambiguous.', 'Large visible improvement over LONG in the coherence of the lower supporting structure, accompanied by a style change.'),
 ((Y,N,Y,Y,Y,U), 'Upper chair and blue seat remain present, but the lower panel supports have broad horizontal breaks and loose-looking strips.', 'Lower support remains visibly fragmented like LONG; no clear recovery.'),
],
('2d', 2, 'POST'): [
 ((Y,N,Y,Y,Y,U), 'Processing fills some strips, but lower supports still show multiple horizontal tears and irregular plate fragments beneath the blue seat.', 'REFERENCE'),
 ((Y,N,Y,Y,Y,U), 'Some lower panels become more filled, but side openings and torn strip-like supports remain obvious. Blue C is still the seat.', 'Locally more filled than LONG, without a clear recovery to a coherent lower support.'),
 ((Y,Y,N,N,U,Y), 'A complete slender sled frame and coherent chair shell remain. No obvious detached plate is visible; a fine underside seam remains uncertain.', 'Lower supporting frame is much more coherent than LONG; blue C remains at the seat.'),
 ((Y,N,Y,Y,Y,U), 'Horizontal openings and torn lower support strips remain after processing. The back and blue seat are recognizable but the supporting body is visibly damaged.', 'Same broad lower-support failure as LONG, with differently placed tears.'),
],
('2d', 3, 'RAW'): [
 ((Y,Y,N,N,Y,Y), 'Blue seat and complete cantilever body are present. A diagonal white slit crosses the lower back panel; a small front seat-side opening is also visible.', 'REFERENCE'),
 ((Y,Y,N,N,Y,Y), 'Rounded-back cantilever chair uses blue C as its seat. The diagonal back line is much narrower, while a small front/side seat opening remains.', 'Lower-back tear is less apparent than LONG; overall structure remains similar.'),
 ((Y,Y,N,N,Y,Y), 'Blue seat and complete body remain. A thin white diagonal opening is visible higher in the back, rather than in the same lower-back location as LONG.', 'Back defect moves upward; the figure does not establish a uniformly cleaner back than LONG.'),
 ((Y,Y,N,N,Y,Y), 'Blue C remains the seat of a complete cantilever body. A broad diagonal white slit crosses the lower back and a white triangular opening appears near its lower corner.', 'Lower-back split is larger and more obvious than LONG.'),
],
('2d', 3, 'POST'): [
 ((Y,Y,N,N,Y,Y), 'Lower-back diagonal slit is largely closed, but a narrow front seat/support separation remains. The blue-seat cantilever body is complete.', 'REFERENCE'),
 ((Y,Y,N,N,Y,Y), 'Back panel appears closed at this scale. Small front/side seat gaps remain in a complete cantilever body.', 'Similar to LONG after processing; no large lower-back tear remains visible.'),
 ((Y,Y,N,N,Y,Y), 'Back panel is largely filled. A conspicuous white strip remains at the front seat/support transition; the body is otherwise complete.', 'Back closure is comparable to LONG; front separation remains visually apparent.'),
 ((Y,Y,N,N,Y,Y), 'A broad diagonal white opening remains across the lower back after processing. Blue seat, front support and base are still present.', 'Persistent lower-back tear is clearly worse than LONG on this request.'),
],
}

IDENTITIES = {
 '73': ('73b369b9', '73b369b9808bc2d0f6ba522682be8978', 'TRAIN'),
 '2d': ('2d08a64e', '2d08a64e4a257e007135fc51795b4038', 'DEV'),
}
COMMON_UNCERTAINTY = (
 'Nonblind AI_PROXY. Blue C can occlude near-coplanar gray layers; neither '
 'absence of visible overlap nor apparent contact certifies legal shared '
 'topology, self-intersection, watertightness or mechanical stability. '
 'Central cuts show only selected sections. Ordinary cantilever/frame openings '
 'and different chair styles are not counted as missing supports.'
)
NOTES = {
 '73': '''# 73：实际 AI_PROXY 图审

本次逐页打开了四个噪声的 RAW/POST 八页原分辨率图，并检查每页 LONG/A/B/C 四列与两条中心剖面，共 32 条记录。没有读取数值评分决定视觉标签，没有复制历史评审。模型名称可见，因此是非盲 AI_PROXY，不是人类评分。

- n0：四组均是可辨识的完整椅体。B/C 的后侧蓝色座面与靠背在图上更连续；LONG/A 的后侧间隙仍可见。前侧细缝未普遍消失。
- n1：B 的灰色座面明显高于蓝色 C，POST 后仍保留两层高差；这是该页明确的采用代价。C 的 RAW 底板上有细长斜片，POST 图中不再明显。
- n2：B 的靠背上侧白色楔形裂口在 POST 仍可见。A 与 LONG 整体相近；C 的 RAW 小裂口在 POST 较不明显，但前缘仍有细缝。
- n3：几组主体接近。细小边线、可能的前支撑叠层和座缘间隙无法全部确定，保留 UNCERTAIN，没有强行给赢家。

蓝色 C 会遮住邻近共面灰面。`visible_join=YES` 仅表示至少一段 C 与灰面在此渲染尺度看起来连续，不表示整圈接合、索引共边合法或没有 FF/I。悬臂椅没有四条独立腿不自动记为支撑缺失。RAW/POST 分别记录，TRAIN 不并入 DEV。
''',
 '2d': '''# 2d：实际 AI_PROXY 图审

本次逐页打开了四个噪声的 RAW/POST 八页原分辨率图，并检查每页 LONG/A/B/C 四列与两条中心剖面，共 32 条记录。没有读取数值评分决定视觉标签，没有复制历史评审。模型名称可见，因此是非盲 AI_PROXY，不是人类评分。

- n0：四组均形成框架椅，蓝色 C 位于上部座面；下方灰壳与 C 存在较大高度/轮廓差异。厚壳与多层的区别及局部接合被遮挡，相关项保留 UNCERTAIN。B 的下壳更深、更折角，不能仅因款式不同判失败。
- n1：LONG/A/B 是板式支撑，C 为悬臂底座。A/B 左侧座缘比 LONG 更填满；C 的侧面三角缺口仍很明显。几组都没有明显整体主体缺失。
- n2：最明显的视觉差异。LONG/A/C 下部支撑碎成多条板片并有横向破口，POST 虽填补部分区域仍未恢复完整；B 则生成较完整的细框架支撑，蓝 C 仍位于座面。这是该请求的可见正例，不能外推到全部噪声或数值合法性。
- n3：C 的大斜裂口在 POST 下部靠背中仍明显存在；LONG/A/B 的相应靠背面较完整。各组前部座缘仍有细缝，未宣称整圈接合。

蓝 C 下方的灰面既可能是厚度/支撑，也可能是多层，单凭图不能全面分辨；本表没有把所有下壳都判作替代座面。`visible_join=YES` 仅是局部可见接触，不能认证索引、流形性或 FF/I。剖面只覆盖局部截线，RAW/POST 分别记录，DEV 与 TRAIN 分开保留。
''',
}

def main():
    completion = {'schema_version': SCHEMA['schema_version'],
                  'review_kind': 'AI_PROXY', 'actual_pages_viewed': 16,
                  'numeric_scores_used_for_labels': False,
                  'model_forward_calls': 0, 'scorer_calls': 0, 'parents': {}}
    for prefix, (short, full, split) in IDENTITIES.items():
        rows = []
        for noise in range(4):
            for stage in ['RAW', 'POST']:
                page = BASE / 'gallery' / f'{short}_n{noise}_{stage}.png'
                sha = hashlib.sha256(page.read_bytes()).hexdigest()
                entries = PAGES[prefix, noise, stage]
                assert len(entries) == 4
                for model, (labels, observations, delta) in zip(MODELS, entries):
                    assert len(labels) == 6 and set(labels) <= {Y,N,U}
                    row = dict(request_id=f'{short}_n{noise}', parent_id=full,
                               split=split, noise_index=noise, stage=stage, model=model,
                               reviewer='AI_PROXY_SEAM_PORTABLE_CORE', review_kind='AI_PROXY',
                               review_status='ACTUALLY_VIEWED', blinding=SCHEMA['blinding'],
                               visible_observations=observations, change_vs_LONG=delta,
                               uncertainty=COMMON_UNCERTAINTY,
                               image_path=page.relative_to(ROOT).as_posix(), image_sha256=sha)
                    row.update(zip(LABEL_FIELDS, labels))
                    assert set(row) == set(SCHEMA['csv_columns'])
                    rows.append(row)
        assert len(rows) == 32
        assert len({(x['request_id'], x['stage'], x['model']) for x in rows}) == 32
        out = BASE / 'reviews' / f'{prefix}.csv'
        assert not out.exists(), f'Preserve prior reviews: {out}'
        with out.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=SCHEMA['csv_columns'])
            writer.writeheader()
            writer.writerows(rows)
        note = BASE / 'reviews' / f'{prefix}.md'
        assert not note.exists(), f'Preserve prior review notes: {note}'
        note.write_text(NOTES[prefix], encoding='utf-8')
        completion['parents'][prefix] = {'rows': 32, 'pages': 8, 'split': split,
                                        'csv_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                                        'notes_sha256': hashlib.sha256(note.read_bytes()).hexdigest()}
    completion['status'] = 'COMPLETE'
    target = BASE / 'reviews' / '73_2d_review_completion.json'
    assert not target.exists()
    target.write_text(json.dumps(completion, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(completion, indent=2))

if __name__ == '__main__':
    main()
