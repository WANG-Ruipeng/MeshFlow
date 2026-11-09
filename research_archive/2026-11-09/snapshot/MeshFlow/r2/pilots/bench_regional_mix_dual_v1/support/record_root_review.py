"""Transcribe root's actual 12-page image review; never infer ratings from scores."""
from r2.pilots.bench_regional_mix_dual_v1.common import OUT, read, sha, write, utc


def run():
    target = OUT / 'visual_review/root_review.json'
    if target.exists():
        raise RuntimeError('Review already recorded; do not overwrite actual observations')
    panel = read(OUT / 'panel/manifest.json')['requests']
    variants = ['ASSIST10', 'ASSIST01', 'ASSIST11', 'AUTO_COVER']
    pages, observations = [], []
    # Each category below is an explicit human-readable transcription of images
    # actually displayed with view_image in the root conversation, not a classifier.
    ratings = {
        'good': ('REASONABLE', 'PLAUSIBLE', 'PRESENT', 'NO_OBVIOUS', 'NO_OBVIOUS'),
        'damaged': ('INCOMPLETE', 'PLAUSIBLE', 'MISSING_OR_DAMAGED', 'YES', 'NO_OBVIOUS'),
        'support_restored': ('REASONABLE', 'PLAUSIBLE', 'PRESENT', 'YES', 'NO_OBVIOUS'),
    }
    for i in range(3):
        rid = panel[i]['request_id']
        evidence = []
        for variant in variants:
            jid = f'J{i:02}_{variant}'
            image = OUT / 'runs' / jid / 'gallery/CLEAN_FINAL_POST.png'
            receipt = read(image.with_suffix('.json'))
            assert sha(image) == receipt['image_sha256']
            evidence.append(jid)
            bound = ('2026-10-28T03:37:35+00:00' if i == 0 or (i == 1 and variant in variants[:2])
                     else '2026-10-28T03:41:40+00:00' if i == 1
                     else '2026-10-28T03:42:19+00:00')
            pages.append({
                'job_id': jid, 'image_path': str(image), 'image_sha256': sha(image),
                'image_receipt_sha256': sha(image.with_suffix('.json')),
                'view_event': 'Root conversation functions.exec -> tools.view_image; actual image displayed',
                'viewed_by_utc': bound, 'view_time_precision': 'upper bound from adjacent clock call, not exact view timestamp',
                'views': ['oblique_-60', 'oblique_120', 'fixed_front_section', 'fixed_side_section'],
            })
        if i < 2:
            notes = {
                'RAW': '蓝色板条座面周围有四条具可见宽度的灰腿；主体和支撑关系可解释。正常板条间隔不算裂片，图像不认证焊接或无交。',
                'M_D0': '四腿多数退成针线/残段；两视角都缺少原支撑表面。中部板条侧壁和底面在截面中残缺。蓝色C仍像座面。',
                'RAW_ALL': '四条主要腿及板条侧壁大体保留，整体接近RAW；宏观完整不代表数值冲突已经消失。',
                'ASSIST10': '一端两腿恢复为有宽度的表面，另一端仍为针线残余；中部板条侧壁仍保留D0的损失。',
                'ASSIST01': '恢复另一端两腿，未选的一端仍明显缺主要支撑；中部侧壁残缺没有一起恢复。',
                'ASSIST11': '两端四腿都恢复，蓝色C座面被使用；中部板条侧壁/底部仍残缺，故支撑正向与可见裂口同时记录。',
                'AUTO_COVER': ('自动少量换回没有恢复主要四腿，外观仍接近D0针线支撑。' if i == 0
                               else '实际查看的自动列与D0一致，四腿仍针线化；本请求选择为空且复用D0，但标签来自实际图审。'),
            }
        else:
            notes = {
                'RAW': '蓝色弧形座面与灰色背板、端部扶手及竖直支柱形成可解释Bench；两视角保留主要结构。截面孤立轮廓可能属于远端扶手，不据此认定漂浮面。',
                'M_D0': '灰背板仍在、C仍充当座面，但端部主要支柱变成针线，扶手只剩细碎边线；不能把针线当保住支撑。',
                'RAW_ALL': '保住RAW的端柱和扶手等主要结构，宏观完整；此标签不代表无自交或接缝合格。',
                'ASSIST10': '只恢复一端下部支柱，另一端仍残缺；上部扶手细碎部分未纳入选区，仍有明显缺损。',
                'ASSIST01': '相反一端下部支柱恢复，但另一端仍针线化；上部端框/扶手仍受D0损伤。',
                'ASSIST11': '两端下部支柱都恢复，座背和主要支撑合理；上部扶手部分仍残缺/细线，记录局部缺口而不认证完整表面修复。',
                'AUTO_COVER': '自动列主要恢复一端支柱；另一端仍针线/短残段，扶手也未完整恢复。主体有座背但主要支撑仍不全。',
            }
        for model in ['RAW', 'M_D0', 'RAW_ALL'] + variants:
            category = 'good' if model in ('RAW', 'RAW_ALL') else 'support_restored' if model == 'ASSIST11' else 'damaged'
            values = ratings[category]
            row = dict(zip(['overall_shape', 'C_use', 'support_retention', 'visible_fragment_or_gap', 'layering_or_fold'], values))
            row.update({
                'request_id': rid, 'panel_index': i, 'model': model,
                'stage': 'RAW' if model == 'RAW' else 'CLEAN_FINAL_POST',
                'three_axis_joint': category in ('good', 'support_restored'),
                'notes': notes[model], 'image_evidence': evidence if model in ('RAW', 'M_D0', 'RAW_ALL') else [f'J{i:02}_{model}'],
                'reviewer': 'AI_PROXY_root', 'review_kind': 'AI_PROXY_NONBLIND',
            })
            observations.append(row)
    assert len(pages) == 12 and len(observations) == 21
    write(target, {
        'status': 'ACTUAL_VIEW_COMPLETE', 'reviewer': 'AI_PROXY_root', 'review_kind': 'AI_PROXY_NONBLIND',
        'recorded_at_utc': utc(), 'actual_viewed_page_count': 12, 'unique_output_count': 21,
        'protocol_sha256': sha(OUT / 'visual_review/PROTOCOL.md'),
        'candidate_metrics_read_before_review': False,
        'limitations': 'Same historical development requests; nonblind AI proxy, not independent human certification. Support/shape labels do not certify topology, no intersection, watertightness or source reconstruction.',
        'pages': pages, 'observations': observations,
    })
    print(target)


if __name__ == '__main__':
    run()
