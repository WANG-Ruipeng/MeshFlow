"""Manual nonblind observations written after opening all 28 HINGE PNGs.

This module reads image receipts and the frozen request manifest, not metrics.
"""
from collections import Counter
from .evaluation import C, OUT, requests

# request -> observation, 015 visual change, 025 visual change
NOTES = {
    '73b369b9_n0': ('两外观视角的底座斜白缝保持；front 座下两端间隔仍在。两个 HINGE 的 side 下层长线略变斜，但与前后端的间隔未消失。', 'MINOR_LOCAL_SHAPE_CHANGE', 'MINOR_LOCAL_SHAPE_CHANGE'),
    '73b369b9_n1': ('高侧板和斜三角面外观保持，front 的近 C 灰层保持；side 后根附近短线和前端分离仍在。025 的 front 左角线有轻微转折变化，不能确认质量变好。', 'UNCHANGED_VISIBLE', 'MINOR_LOCAL_SHAPE_CHANGE'),
    '73b369b9_n2': ('多边背板和外壳保持。两个 HINGE 的 side 座下长线右端与下行竖线看起来接合，原细小间隔不再清楚；后根与下层间隔仍在，front 基本不变。', 'LOCAL_SECTION_GAP_CLOSED', 'LOCAL_SECTION_GAP_CLOSED'),
    '73b369b9_n3': ('背板顶部白洞保持。两个 HINGE 的 side 座下长线右端与下行线更连续，底线斜率略变；C 前沿附近的分离短斜线和后根间隔仍在。', 'LOCAL_SECTION_GAP_CLOSED', 'LOCAL_SECTION_GAP_CLOSED'),
    '3f741759_n0': ('椅背细斜条与四腿外观保持；front 左下大间隔、右上间隔及 side 较完整底框在三列都存在，未见明确改善。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '3f741759_n1': ('外观和两截面近似不变；front 左下缺口仍在，side 底线台阶保持，没有明显悬片消除。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '3f741759_n2': ('side 上方短横截线在 POST、015、025 全部保留；背部细斜条仍可见。front 座框已较连续，HINGE 未带来明显额外修复。短线本身不认证三维离体。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '3f741759_n3': ('两个外观视角与 front/side 较连续座框保持，细小台阶形态没有明显变化；未见新的大裂口或明显视觉增量。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '85b16941_n0': ('薄座框和背板外观保持；front 中间水平线断点、两侧下折短线，以及 side 前后端间隔仍在。没有广泛接合改善。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '85b16941_n1': ('front 左侧和右上小间隔、side 前沿间隔均保持；板背和四腿外观近似不变。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '85b16941_n2': ('两个 HINGE 的 front 右下角细间隔看起来闭合，右上靠近 C 处小间隔也较不明显；side 与外观主体基本保持。这只是显示平面上的局部改善。', 'LOCAL_SECTION_GAP_CLOSED', 'LOCAL_SECTION_GAP_CLOSED'),
    '85b16941_n3': ('front 右侧竖线的小断点在两个 HINGE 中闭合，side 右下角也更连续；但 front 左下大缺口和座下楔形白缺口仍保留，未修复主要外壳缺损。', 'LOCAL_SECTION_GAP_CLOSED', 'LOCAL_SECTION_GAP_CLOSED'),
    '8c9b2023_n0': ('梯状靠背外观保持；front 下层两处间隔、side 上方矩形背条截线均保持，未见明确新的修复。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '8c9b2023_n1': ('front 下层右半多段线和 side 左侧开放端仍在；上方 U 形背条截线及近 C 短线保持，外观近似不变。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '8c9b2023_n2': ('015 与 POST 的 front 左侧多段线近似；025 将其中一处下层小断点连成较长水平段，但最左断段、side 左端开口和上方短线仍在。', 'UNCHANGED_VISIBLE', 'LOCAL_SECTION_GAP_CLOSED'),
    '8c9b2023_n3': ('015 的 front 下层多处小断点仍在；025 的右半其中一处断点不再明显。side 左端大开口及靠背根部外观白缝保持，没有整椅修复。', 'UNCHANGED_VISIBLE', 'LOCAL_SECTION_GAP_CLOSED'),
    '2d08a64e_n0': ('低斜背和悬臂框架保持；front 座下多层水平线与 side 交叉斜线在三列都在，未见明显改动。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '2d08a64e_n1': ('板背和下部箱架形态保持；front 仅左半座下横线、side 下层前端开口保持，没有明显新连线或大破坏。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '2d08a64e_n2': ('箱式扶手外壳和 front 两侧双线保持；side 的 C 下方开放截线形态一致，未见可确认的修复。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '2d08a64e_n3': ('箱式外壳、背板及两截面轮廓近似不变；本尺度没有明确视觉增量。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '610b1621_n0': ('宽箱式扶手与外壳保持，front/side 侧壁和 C 的相对层位基本一致；未见大片新缺损，也未见接续全面改善。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '610b1621_n1': ('框架椅与扶手保持；front 两端小开口及 side 前沿短斜线与下层间隔仍在，未见明显修复。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '610b1621_n2': ('框架椅外观保持；两个 HINGE 的 front 左下小闭合轮廓略变形，但远低于 C 的长截线和 side 多段截线保留，不能认定接续改善。', 'MINOR_LOCAL_SHAPE_CHANGE', 'MINOR_LOCAL_SHAPE_CHANGE'),
    '610b1621_n3': ('小背板、四腿及超出主体的蓝 C 形态保持；front 下层断点和 side 多条分离短线仍在，未见明显改善。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '608b46b0_n0': ('椅背和前腿白裂缝保持；front 座下右端间隔仍在，025 右端短折线略改变；side 长下层与前沿分离关系保持。', 'UNCHANGED_VISIBLE', 'MINOR_LOCAL_SHAPE_CHANGE'),
    '608b46b0_n1': ('较完整背板和腿部保持；front 已较连续的底框近似不变，side 前沿短斜线与下层仍分离，未见额外明显修复。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '608b46b0_n2': ('箱式外观保持；front 侧壁分离短线和 side 下方分段轮廓均保留，未见明确变化。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
    '608b46b0_n3': ('背板大楔形裂口与顶部多孔在三个版本均保留；front 座下框本来已连续，side 前沿分离线仍在。HINGE 没有修复主视觉缺损。', 'UNCHANGED_VISIBLE', 'UNCHANGED_VISIBLE'),
}


def main():
    qq = {q['request_id']: q for q in requests()}
    assert set(NOTES) == set(qq)
    rows = []
    for rid, (note, change015, change025) in NOTES.items():
        image = OUT / 'gallery/HINGE_SMOOTH_PRIOR' / (rid + '.png')
        receipt = C.read(image.with_suffix('.json'))
        sha = C.sha(image)
        assert sha == receipt['image_sha256']
        assert [x['label'] for x in receipt['cells']] == [
            'SMOOTH_PRIOR_POST', 'SMOOTH_PRIOR_HINGE015', 'SMOOTH_PRIOR_HINGE025']
        for variant, change in (('HINGE015', change015), ('HINGE025', change025)):
            rows.append(dict(
                request_id=rid, parent_id=qq[rid]['parent_id'], split=qq[rid]['split'],
                noise_index=qq[rid]['noise_index'], label='SMOOTH_PRIOR_' + variant,
                baseline='SMOOTH_PRIOR_POST', reviewer='AI_PROXY_NONBLIND',
                reviewer_agent='/root/edge_prior_training', actually_viewed=True,
                image_path=str(image), image_sha256=sha,
                viewed_panels='POST/HINGE015/HINGE025: two exterior views plus fixed central front/side finite sections',
                visible_change=change, observation=note,
                local_section_closure='VISIBLE' if change == 'LOCAL_SECTION_GAP_CLOSED' else 'NOT_CONFIRMED',
                broad_fragment_removal='NOT_OBSERVED',
                overall_shape='PRESERVED_IN_DISPLAYED_VIEWS',
                new_severe_visual_damage='NOT_OBSERVED_IN_THESE_VIEWS',
                uncertainty='Finite cuts may miss off-plane connections; subpixel gaps may be invisible. Images do not certify topology, absence of intersection, or floating-face identity.',
            ))
    csvpath = OUT / 'visual/smooth_hinge_actual_review.csv'
    C.csvwrite(csvpath, rows)
    C.write(OUT / 'visual/smooth_hinge_actual_review_completion.json', dict(
        status='ACTUAL_REVIEW_COMPLETE', reviewer='AI_PROXY_NONBLIND',
        viewed_pages=28, variant_reviews=56, reviewed_requests=28,
        no_metric_based_labels=True, model_calls=0, scorer_calls=0,
        visible_change_counts=dict(Counter(r['visible_change'] for r in rows)),
        files={str(csvpath): C.sha(csvpath)},
        observation='Most visible structures persist. Local section gaps close for73n2/n3,85n2/n3 at both thresholds and8cn2/n3 mainly025.3fn2 upper short section and608n3 back tears persist. No broad fragment removal or whole-DEV visual repair observed.',
        limitation='No severe new exterior damage was observed in these displayed views; this is not a claim of zero hidden geometric damage. Original metrics and topology audits remain separate.',
    ))
    print('ACTUAL_REVIEW_COMPLETE: 28 pages, 56 SMOOTH HINGE variants', flush=True)


if __name__ == '__main__':
    main()
