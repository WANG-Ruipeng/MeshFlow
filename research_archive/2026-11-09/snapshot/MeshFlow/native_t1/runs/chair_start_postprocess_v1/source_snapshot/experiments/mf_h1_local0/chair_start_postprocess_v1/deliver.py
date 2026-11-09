"""Finalize the completed CPU experiment; never processes or scores a mesh."""
from __future__ import annotations
import csv
import difflib
import hashlib
import html
import json
import shutil
import sys
import zipfile
from pathlib import Path
from . import run as driver

ROOT, OUT, CODE = driver.ROOT, driver.OUT, driver.CODE


def reviews(cases):
    collected = {}
    for name in ('visual_review_0_7.json', 'visual_review_8_15.json',
                 'visual_review_16_23.json', 'visual_review_24_31.json'):
        source = OUT/'audit'/name
        value = driver.read(source)
        for row in value.get('rows', value.get('outputs', [])):
            rid = row['id']
            if rid in collected:
                raise ValueError('Duplicate visual review '+rid)
            expected = row.get('comparison_sha256', row.get('image_sha256', row.get('sha256')))
            picture = OUT/'visuals'/rid/'comparison.webp'
            if driver.digest(picture) != expected:
                raise ValueError('Reviewed picture changed '+rid)
            collected[rid] = dict(id=rid, parent=row['parent'], noise=row['noise'],
                panel=row['panel'], review='AI_VISUAL_REVIEW', human_usability='NOT_RUN',
                image=str(picture.relative_to(OUT)), image_sha256=expected,
                source_review=str(source.relative_to(OUT)), source_sha256=driver.digest(source),
                observations=row)
    if set(collected) != {c['id'] for c in cases} or len(collected) != 32:
        raise ValueError('Incomplete actual visual review')
    rows = [collected[c['id']] for c in cases]
    driver.write(OUT/'audit/visual_review_all32.json', dict(status='PASS', count=32,
        type='AI_VISUAL_REVIEW', human_usability='NOT_RUN',
        limits='Raster review is not a geometric certificate or a human usability study.', rows=rows))
    driver.csv_write(OUT/'audit/visual_review_all32.csv', rows)


def gallery(cases):
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>START — RAW / GEO_C / GT</title>',
        '<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1200px;margin:auto;padding:24px;background:#edf1f5;color:#192431}section{background:white;padding:18px;margin:24px 0;border-radius:8px}img{max-width:100%;height:auto}a{color:#145ca3}h2{overflow-wrap:anywhere;font-size:19px}summary{cursor:pointer}nav{display:flex;gap:14px;flex-wrap:wrap}</style>',
        '<h1>START 后处理对照：全部 32 例</h1>',
        '<p>8 父 × 4 份既有噪声；零训练、零模型前向、零新生成。左 RAW，中 GEO_C，右 GT；蓝色为 C，橙色为自由面。每父所有噪声使用相同相机。学习式去噪资产未核验，未执行。</p>',
        '<p>全 32 页已实际进行 AI 图像复核；人工可用性评估未执行。GEO_C 经过固定几何处理，渲染器没有额外平滑或修改网格。图像不能证明水密、流形或没有相交。</p>',
        '<nav><a href="START_POSTPROCESS_REPORT.md">完整报告</a><a href="analysis/paired_per_output.csv">逐输出配对</a><a href="analysis/paired_parents_ALL.csv">逐父配对</a><a href="audit/visual_review_all32.json">逐例视觉复核</a></nav>',
        '<section><h2>数值配对图</h2><img src="figures/paired_metrics.png" alt="32 paired metrics"></section>']
    for c in cases:
        rid = c['id']
        path = 'visuals/'+rid
        parts.extend([f'<section id="{rid}"><h2>{html.escape(c["panel"])} · {html.escape(c["parent"])} · n{c["noise"]}</h2>',
            f'<img loading="lazy" src="{path}/comparison.webp" alt="{rid}: RAW / GEO_C / GT">',
            f'<p><a href="meshes/RAW/{rid}.ply">RAW PLY</a> · <a href="meshes/GEO_C/{rid}.ply">GEO_C PLY</a> · <a href="{path}/comparison.webp">打开原图</a></p>',
            f'<details><summary>边界与位移辅助图</summary><img loading="lazy" src="{path}/boundary_displacement.webp" alt="Boundary and displacement diagnostics"></details></section>'])
    parts.append('</html>')
    (OUT/'OPEN_GALLERY.html').write_text('\n'.join(parts), encoding='utf-8')


def sources():
    from . import scoring
    info = scoring.sources()
    paths = set(CODE.glob('*.py'))
    # Snapshot actual locally imported dependencies, including dynamically loaded scoring files.
    for module in tuple(sys.modules.values()):
        p = getattr(module, '__file__', None)
        if p:
            p = Path(p).resolve()
            if p.suffix == '.py' and p.is_relative_to(ROOT) and p.is_file():
                paths.add(p)
    def add_named(value):
        if isinstance(value, dict):
            for key, val in value.items():
                if key.endswith('.py'):
                    p = ROOT/key
                    if p.is_file():
                        if driver.digest(p) != val:
                            raise ValueError('Scientific source identity differs '+key)
                        paths.add(p)
                add_named(val)
        elif isinstance(value, list):
            for val in value:
                add_named(val)
    add_named(info)
    records = []
    for p in sorted(paths):
        rel = p.relative_to(ROOT)
        dest = OUT/'source_snapshot'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        records.append(dict(source=str(rel), packaged=str(dest.relative_to(OUT)),
            sha256=driver.digest(p), bytes=p.stat().st_size,
            role='new_experiment' if p.parent == CODE else 'read_only_local_dependency'))
    driver.write(OUT/'source_file_hashes.json', records)
    driver.csv_write(OUT/'source_file_hashes.csv', records)
    driver.write(OUT/'scoring_source_identities.json', info)
    patch = []
    for p in sorted(CODE.glob('*.py')):
        rel = p.relative_to(ROOT).as_posix()
        patch.append(f'diff --git a/{rel} b/{rel}\nnew file mode 100644\n')
        patch.extend(difflib.unified_diff([], p.read_text(encoding='utf-8').splitlines(True),
            fromfile='/dev/null', tofile='b/'+rel))
    (OUT/'code.diff').write_text(''.join(patch), encoding='utf-8')
    return records


def selected_files():
    selected = set()
    for name in ('analysis', 'audit', 'cpu_tests', 'geometry', 'inputs', 'outputs',
                 'references', 'resources', 'status', 'registration', 'engineering',
                 'failures', 'ledger', 'figures', 'source_snapshot'):
        selected.update(p for p in (OUT/name).rglob('*') if p.is_file())
    selected.update((OUT/'meshes').rglob('*.ply'))
    selected.update((OUT/'visuals').rglob('*.webp'))
    selected.update((OUT/'visuals').glob('*.json'))
    for name in ('protocol.json', 'cases.json', 'protected_before.json', 'DECISION.md',
                 'START_POSTPROCESS_REPORT.md', 'OPEN_GALLERY.html', 'code.diff',
                 'source_file_hashes.json', 'source_file_hashes.csv',
                 'scoring_source_identities.json', 'README_REVIEW.md',
                 'evaluation/per_output.json', 'evaluation/per_output.csv'):
        selected.add(OUT/name)
    return sorted(p for p in selected if p.is_file() and p.suffix not in ('.pyc', '.partial')
                  and '__pycache__' not in p.parts)


def main():
    cases = driver.read(OUT/'cases.json')
    for audit in ('independent_geometry.json', 'independent_scoring.json'):
        if driver.read(OUT/'audit'/audit)['status'] != 'PASS':
            raise ValueError('Independent audit not passed '+audit)
    reviews(cases)
    from .report import report
    report(OUT, driver.read(OUT/'analysis/results.json'), driver.read(OUT/'analysis/paired_per_output.json'))
    gallery(cases)
    source_records = sources()
    driver.resources('after_completion')
    protected = driver.verify_protected()
    scores = driver.read(OUT/'evaluation/per_output.json')
    assert len(scores) == 64 and all(r['status'] == 'PASS' for r in scores)
    (OUT/'README_REVIEW.md').write_text('''# START 后处理复核包

先读 START_POSTPROCESS_REPORT.md；浏览全部 32 例请打开 OPEN_GALLERY.html。

本包包含 64 行原始/处理后评分、32 行逐输出配对、逐父与逐噪声表、全部 64 份 PLY、32 份处理后 NPZ/来源映射、32 份 RAW 终点数组、8 组 C/GT 条件和冻结查询、32 对照图与32辅助图、实际评分依赖源码、code.diff、哈希、执行账本和独立审计。RAW 是既有生成的终点副本，没有重新采样或替代源数据。包内不含模型权重或完整生成轨迹。

分析是已经观察过的 8 父 × 4 噪声上的描述性比较。C 原样保留；学习式去噪 NOT_RUN_ASSET_UNAVAILABLE。数值收益不能等同于人工可用性或水密证明。

为满足 <10 MB，完整逐点评分向量和无损 PNG/渲染缓存留在原本地 run；省略清单及其哈希见 OMITTED_LOCAL_ARTIFACTS.json。它们没有被删除。逐例 CSV 和实际评分代码完整包含。

运行路径记录保留实际绝对路径，source_snapshot 是原样源码快照而非便携安装包。原工作区的可恢复总入口为：
`python -B -m experiments.mf_h1_local0.chair_start_postprocess_v1.run all`
完整性复核：`...run verify`；只刷新交付：`...deliver`。
上述入口只在既有本地条件/RAW/冻结评分资产存在时使用，不下载权重、不训练、不重生成；已完成的科学阶段复用其缓存。

PACKAGE_CONTENTS.json 列出其自身以外所有 ZIP 项的字节数和 SHA256；可先验证各项再查看。外部 DELIVERY_RECEIPT.json 记录压缩包哈希和大小。原模型、原数据、原 RAW、原评分器/报告和正式默认 profile 均未修改，没有提交或推送 Git。
''', encoding='utf-8')
    driver.event('DELIVERY_MATERIALIZED', cases=32, scoring_rows=64, visual_reviews=32,
        protected_files=protected['checked'], model_calls=0, new_generator_samples=0,
        training_updates=0, source_snapshots=len(source_records))
    driver.write(OUT/'status/complete.json', dict(status='COMPLETE', cases=32, scoring_rows=64,
        AI_visual_reviews=32, protected=protected, model_calls=0, training_updates=0,
        new_generator_samples=0, denoiser='NOT_RUN_ASSET_UNAVAILABLE',
        human_usability='NOT_RUN', default_model_changed=False, git_commit=False, git_push=False))
    files = selected_files()
    omitted = [dict(path=str(p.relative_to(OUT)), bytes=p.stat().st_size, sha256=driver.digest(p))
        for folder in ('evaluation', 'visuals') for p in sorted((OUT/folder).rglob('*'))
        if p.is_file() and p not in files]
    driver.write(OUT/'OMITTED_LOCAL_ARTIFACTS.json', dict(reason='Compact review package; originals retained locally.', files=omitted))
    files.append(OUT/'OMITTED_LOCAL_ARTIFACTS.json')
    entries = [dict(path=p.relative_to(OUT).as_posix(), bytes=p.stat().st_size,
                    sha256=driver.digest(p)) for p in sorted(files)]
    driver.write(OUT/'PACKAGE_CONTENTS.json', dict(schema='sha256_file_manifest_v1',
        excludes_self=True, count=len(entries), files=entries))
    files.append(OUT/'PACKAGE_CONTENTS.json')
    delivery = OUT/'delivery'
    delivery.mkdir(exist_ok=True)
    target = delivery/'START_POSTPROCESS_REVIEW.zip'
    pending = target.with_suffix('.zip.partial')
    with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(files):
            z.write(p, p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(pending) as z:
        if z.testzip() is not None:
            raise ValueError('Archive CRC check failed')
        for entry in entries:
            data = z.read(entry['path'])
            if hashlib.sha256(data).hexdigest() != entry['sha256'] or len(data) != entry['bytes']:
                raise ValueError('Archive hash verification failed '+entry['path'])
        names = set(z.namelist())
        assert len([n for n in names if n.startswith('meshes/') and n.endswith('.ply')]) == 64
        assert len([n for n in names if n.endswith('/comparison.webp')]) == 32
        assert len([n for n in names if n.endswith('/boundary_displacement.webp')]) == 32
        assert not any(n.endswith(('.pt', '.pth', '.safetensors')) for n in names)
        import re
        for link in re.findall(r'(?:src|href)="([^"]+)"', z.read('OPEN_GALLERY.html').decode()):
            if link not in names:
                raise ValueError('Gallery target absent '+link)
    if pending.stat().st_size >= 10_000_000:
        raise ValueError('Review archive exceeds decimal 10 MB budget')
    pending.replace(target)
    receipt = dict(status='PASS', archive=str(target), bytes=target.stat().st_size,
        decimal_MB=target.stat().st_size/1e6, sha256=driver.digest(target),
        files=len(files), all_member_hashes_verified=True, gallery_links_verified=True,
        protected_files=protected['checked'], source_files=len(source_records),
        outputs=32, paired_meshes=64, comparison_pages=32, AI_reviews=32)
    driver.write(delivery/'DELIVERY_RECEIPT.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
