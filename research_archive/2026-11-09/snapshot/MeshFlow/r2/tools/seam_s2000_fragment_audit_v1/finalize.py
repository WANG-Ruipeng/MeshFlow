"""Assemble already completed CPU measurements and actual image-review records."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_seam_s2000_fragment_audit_v1'


def read_csv(name):
    with (OUT / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    audit = json.loads((OUT / 'independent_audit.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'PASS' and audit['source_hashes_unchanged']
    visual = read_csv('visual_train.csv') + read_csv('visual_dev.csv')
    assert len(visual) == 28 and len({r['request_id'] for r in visual}) == 28
    order = {r['request_id']: i for i, r in enumerate(protocol['requests'])}
    assert set(order) == {r['request_id'] for r in visual}
    visual.sort(key=lambda r: order[r['request_id']])
    for row in visual:
        assert row['actually_viewed'].lower() == 'true'
        assert row['reviewer'] == 'AI_PROXY_NONBLIND'
        assert sha(OUT / 'gallery' / (row['request_id'] + '.png')) == row['image_sha256']
    with (OUT / 'visual_review.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(visual[0]))
        writer.writeheader()
        writer.writerows(visual)
    vs = {}
    for split in ('TRAIN', 'DEV'):
        group = [r for r in visual if r['split'] == split]
        vs[split] = {key: dict(Counter(r[key] for r in group)) for key in (
            'post_sliver_visible', 'post_detached_piece_visible', 'post_crack_visible', 'first_visible_stage')}
        vs[split]['outputs'] = len(group)
    dump('visual_summary.json', vs)
    main_rows = read_csv('numeric/split_summary.csv')
    retained_rows = read_csv('numeric/retained_split_summary.csv')
    def lookup(rows, split, stage):
        return next(r for r in rows if r['split'] == split and r['stage'] == stage and r['role'] == 'free')
    def integer(r, field):
        return str(int(float(r[field])))
    def pct(r, field):
        return f'{100 * float(r[field]):.2f}%'
    shape_table = []
    stage_table = []
    sensitivity = []
    for split in ('TRAIN', 'DEV'):
        r = lookup(main_rows, split, 'POST')
        shape_table.append('| ' + ' | '.join((split, integer(r, 'face_count_total'),
            integer(r, 'small_angle_count_total'), pct(r, 'small_angle_fraction_macro_mean'),
            integer(r, 'tiny_1e5_count_total'), pct(r, 'tiny_1e5_fraction_macro_mean'))) + ' |')
        counts = [integer(lookup(retained_rows, split, stage), 'small_angle_count_total') for stage in protocol['stages']]
        stage_table.append('| ' + split + ' | ' + ' | '.join(counts) + ' |')
        sensitivity.append('| ' + split + ' | ' + ' | '.join(pct(r, f'tiny_{t}_fraction_macro_mean') for t in ('1e6','1e5','1e4')) + ' |')
    visual_table = []
    for split in ('TRAIN', 'DEV'):
        def counts(field):
            c = vs[split][field]
            return '/'.join(str(c.get(v, 0)) for v in ('YES', 'NO', 'UNCERTAIN'))
        visual_table.append('| ' + split + ' | ' + ' | '.join(counts(f) for f in (
            'post_sliver_visible', 'post_detached_piece_visible', 'post_crack_visible')) + ' |')
    old_scores = read_csv('numeric/per_output.csv')
    ff_text = []
    for split in ('TRAIN', 'DEV'):
        means = []
        for stage in ('RAW','POST'):
            values = [float(r['original_FF_illegal_affected_free_fraction']) for r in old_scores
                      if r['role'] == 'free' and r['split'] == split and r['stage'] == stage]
            assert len(values) == vs[split]['outputs']
            means.append(100 * statistics.mean(values))
        ff_text.append(f'{split} {means[0]:.2f}%→{means[1]:.2f}%')
    report = '''# S2000细长面 小面积面与可见裂口诊断

对PLAIN_S2000的28份既有输出，完成了四阶段逐面统计和双视角实际图审。**用户看到的“碎、裂”确实值得保留为质量问题，但不能统一解释成极小面积的漂浮三角形。当前更明确的是细长结构面与局部裂口并存；其中一部分在RAW已出现，部分在后处理后更加明显。**

统计对象为TRAIN四父16份、DEV三父12份；原RAW、单次官方D0提议、恢复C和seam锚点后的连续提议、原GEO及来源绕序导出后的POST全部只读。本轮零训练、零模型加载/前向、零新噪声/生成、零GEO重跑。新增的是CPU统计表和既有网格的渲染图片。

**小角面多，但大多数并不是极小面积面。**

| 最终POST free面 | 保留面总数 | 最小角小于5度的面数 | 小角面比例 | 极小面积面数 | 极小面积比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
''' + '\n'.join(shape_table) + '''

比例为逐输出比例的宏均值，面数为合计，因此不应直接用表内两个合计数相除来复算宏均值。小角阈值沿用已有定义。极小面积暂用预先冻结的描述性门槛A/L²<1e-5，L=9.208138264587333；不是已验证的“坏面”标准。

全组924张小角面与25张极小面积面，交集只有8张。细长三角可以很长、面积不小；微小三角也可以接近等边。以下阈值敏感性也说明，不能用一个任意面积门槛替代视觉质量。

| 极小面积比例宏均值 | A/L²<1e-6 | A/L²<1e-5 主阈值 | A/L²<1e-4 |
| --- | ---: | ---: | ---: |
''' + '\n'.join(sensitivity) + '''

实际看图时，红色小角面大量位于椅腿、背柱、框架和板边；它们中有正常薄结构，不能一律删除。可见缺口本身没有面，因此也不会自动变成一张被面积/角度阈值标出的“坏三角”。

**D0会增加一部分小角面，恢复锚点与GEO又缓解一部分。**

为排除删面改变分母，下面只跟踪最终保留的同一组free槽：TRAIN固定2289面、DEV固定1625面。这里是面数，不是比例。

| 固定最终保留面 | RAW | D0提议 | 恢复锚点 | POST |
| --- | ---: | ---: | ---: | ---: |
''' + '\n'.join(stage_table) + '''

DEV小角数经D0增加80，最终仍比RAW多46；TRAIN经D0增加64，最终比RAW少5。因此，后处理的确参与了细长面形状的改变，而“恢复C才造成全部问题”不符合这组数据。D0提议中的C也发生移动，该阶段仅用于诊断，不能当作符合保留C合同的完成结果。

这不是停用D0的证据。原完整流水线的FF受影响free面比例为''' + '；'.join(ff_text) + '''。这些是复用的原评分，包含D0、恢复、合并及删面的共同效果；本轮没有重算中间阶段FF/I，不能单独归功或归罪于D0。

**后处理删掉了1182张面，但不能把1182次删除等同于1182个破洞。**

| 删除原因 | TRAIN | DEV | 合计 |
| --- | ---: | ---: | ---: |
| 合并后同面重复顶点而退化 | 427 | 612 | 1039 |
| 与另一free面重复 | 28 | 115 | 143 |
| 合计 | 455 | 727 | 1182 |

其中1000/1182张在GEO前已是小角面。以最终退化的bulk面为例，TRAIN的426面合计面积从RAW的14.143降为D0的3.693；DEV的611面从25.880降为4.026，随后合并为零面积。D0已经把许多这类面压薄，GEO才完成坍缩和清理。它们可能包含冗余覆盖；重复面删前面积也不是唯一表面覆盖面积，不能累加成“丢失壳面积”。

逐面删除身份、四阶段面积和原因保存在[deletion.csv](numeric/deletion.csv)。GEO删前几何仅由已保存位移代数还原，并核对所有保留角点完全一致，没有重新执行焊接。

**共享边断开和完全漂浮要分开。**

在POST真实索引上，以“不含C、至多3面、面积至多为总free面积0.1%”作描述性小分量门槛：按共边连接，TRAIN与DEV各有34个，共68个；其面积比例宏均值仅为0.053%与0.111%。按共点连接，同门槛分量为0。

这说明上述微小共边分量会通过顶点并入更大的分量，不能全部称为完全漂浮的小碎片；也不能因此认为连接合格。共一个点不等于表面共边接续，更不等于合法流形。全体共点分量仍有TRAIN46个、DEV20个，故本结果也没有排除较大分离部件。RAW三角汤未被用于“每面一个漂浮片”的错误统计。

**28份图都已实际评阅，视觉问题主要体现为局部开口和不规整薄片。**

AI_PROXY_NONBLIND，固定两个历史视角和尺度；下表每格为YES/NO/UNCERTAIN数量。细长面可见不表示损坏；NO只表示这两个视角没有明确看到，不证明隐藏表面完整。

| 图审 | 可见细长面 | 明确独立漂浮片 | 疑似非设计裂口或缺口 |
| --- | --- | --- | --- |
''' + '\n'.join(visual_table) + '''

重点例子：

- [608 n3](gallery/608b46b0_n3.png)：靠背顶部白色小孔在RAW已可辨，D0后更清楚，POST仍在；不能归因于最后恢复C。
- [8c n2](gallery/8c9b2023_n2.png)：后围板的小白三角缝在POST才清晰出现；这是后处理相关的局部风险例子，但图审不能单独判定是哪一张删面造成。
- [85 n3](gallery/85b16941_n3.png)：靠背顶端缺口贯穿四阶段；红色标出的腿和背柱细长面与这个缺口不是同一问题。
- [610 n0](gallery/610b16fb_n0.png)：箱形主体整体连续，没有明确漂浮片或裂口；不能把所有边缘小面自动记为质量失败。

所有请求与不确定项均保存在[visual_review.csv](visual_review.csv)，没有只选坏例或好例代替全组；[完整图集](gallery/index.html)可按请求核对。D0和GEO也有让原有小开口变小或消失的例子，逐图备注保留了这些相反证据。

**当前判断与优先问题。**

可以支持“当前SEAM流水线存在需要关注的细长面和局部裂口，D0及GEO会改变这些局部形状”。还不能据此认定“官方D0与SEAM不适配是主要根因”，因为若干缺口在RAW就已存在，且本轮未比较跳过D0但执行相同GEO的输出，也未比较适配D0。

若下一步只选一个验证目标，应是**在保住现有接续和主体覆盖的前提下，减少后处理新增/放大的局部开口**。小角率、面积与真实缺口应分别保留，不能靠统一删掉小角面或小面积面获得更漂亮的数字。本轮到统计与图审结束，没有自动启动新实验。

**复算与执行记录。**

完整逐输出、逐面、固定保留面、删除和索引分量表位于[numeric](numeric/)，定义和168份源资产SHA在[protocol.json](protocol.json)。独立实现复核112个主阶段、28个GEO删前代数阶段、25898条逐面记录、1182个删除身份、2240条转移记录、394个分量及28份图审收据，结果[PASS](independent_audit.json)。56项旧面数/小角评分匹配，168份原资产hash不变。

CPU数值计算单进程单线程约3.16秒；CPU渲染在首张实际页面测量后使用2进程，每进程单线程，总墙钟7.34秒、最大进程RSS约81MiB；独立审计4.29秒。以上不含人工式逐图AI评阅与文档时间，也不将并发墙钟当成单线程基准。初始资源：RTX5080，16303MiB总显存、2551MiB占用、GPU利用率2%；24逻辑CPU、约61.3GiB可用RAM，GPU科学任务为0。首次WSL权限失败和一次命令引号失败均发生于计算前，已记录于preparation_events.json；Windows CIM读取受限后使用WSL实际内存/核心测量。

只读状态与恢复入口：

```powershell
wsl -d Ubuntu -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/r2/tools/seam_s2000_fragment_audit_v1/resume.py
```

脚本位置为`r2/tools/seam_s2000_fragment_audit_v1/`。状态入口不会启动模型或重跑计算。代码hash见source_hashes.json，最终交付hash见completion.json。
'''
    # The manifest, not a guessed abbreviated identifier, owns example links.
    for req in protocol['requests']:
        if req['request_id'].startswith('610') and req['noise_index'] == 0:
            report = report.replace('610b16fb_n0.png', req['request_id'] + '.png')
    (OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    dump('source_hashes.json', source_hashes)
    gallery = json.loads((OUT / 'gallery/completion.json').read_text(encoding='utf-8'))
    artifacts = ('REPORT.md', 'protocol.json', 'visual_review.csv', 'visual_summary.json',
                 'source_hashes.json', 'independent_audit.json', 'numeric/completion.json', 'gallery/completion.json')
    dump('completion.json', {
        'status': 'COMPLETE', 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'requests': 28, 'actual_visual_reviews': 28, 'primary_stages': 112,
        'source_assets_unchanged': True, 'independent_audit': 'PASS',
        'model_loads': 0, 'model_forwards': 0, 'new_generation': 0, 'new_noise': 0,
        'GPU_scientific_workers': 0, 'GEO_runs': 0, 'new_FF_I_scoring': 0,
        'render_pages': len(gallery['images']), 'render_workers': gallery['workers'],
        'commit': False, 'push': False,
        'render_receipt_note': 'gallery/completion.json is the preserved render-time receipt; final review completion is here and in visual_review.csv.',
        'artifact_hashes': {name: sha(OUT / name) for name in artifacts},
    })
    print(json.dumps({'status': 'COMPLETE', 'visual_summary': vs, 'report': str(OUT / 'REPORT.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
