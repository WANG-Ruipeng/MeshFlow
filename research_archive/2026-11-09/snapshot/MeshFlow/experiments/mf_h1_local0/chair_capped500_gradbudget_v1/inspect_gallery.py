"""Read-only, pixel-exact inspection of final galleries; no mesh/metric imports.

Print one JSON object containing metadata and png_base64. Consumers should send
the decoded PNG to their image viewer, never print the base64 to a user.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'native_t1/runs/chair_capped500_gradbudget_v1'
GROUPS = {
    'history': ('Input C', 'START', 'HIST_TEACHER500', 'HIST_CAPPED500', 'GT C+free'),
    'primary': ('START', 'T500', 'L500', 'G500', 'GT C+free'),
    'stability': ('START', 'T1000', 'L1000', 'G1000', 'GT C+free'),
}
VIEWS = ('front_3q', 'back_3q', 'front', 'side', 'wire')


def resolve(value):
    value = str(value).replace('\\', '/')
    if os.name != 'nt' and len(value) > 2 and value[1:3] == ':/':
        value = '/mnt/' + value[0].lower() + value[2:]
    elif os.name == 'nt' and value.startswith('/mnt/') and len(value) > 7:
        value = value[5].upper() + ':' + value[6:]
    return Path(value)


def digest(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def _font(size, bold=False):
    choices = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans' + ('-Bold' if bold else '') + '.ttf',
        'C:/Windows/Fonts/' + ('arialbd.ttf' if bold else 'arial.ttf'),
    ]
    for path in choices:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _sheet(images, columns, rows, title, subtitle):
    size, left, gap, top = 320, 110, 6, 106
    sheet = Image.new('RGB', (left + len(columns) * (size + gap) + 8,
                              top + len(rows) * (size + gap) + 12), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), title, font=_font(21, True), fill='#202833')
    draw.text((10, 39), subtitle, font=_font(14), fill='#465464')
    boxes = []
    for j, label in enumerate(columns):
        draw.text((left + j * (size + gap) + 3, 76), label,
                  font=_font(15, True), fill='#202833')
    for i, row in enumerate(rows):
        draw.text((6, top + i * (size + gap) + 147), row,
                  font=_font(14, True), fill='#344354')
        for j, image in enumerate(images[i]):
            if image.size != (size, size):
                raise ValueError('Expected original 320px panel; resizing is forbidden')
            x, y = left + j * (size + gap), top + i * (size + gap)
            sheet.paste(image.convert('RGB'), (x, y))
            # Verify every copied pixel, including panel border/background.
            if sheet.crop((x, y, x + size, y + size)).tobytes() != image.convert('RGB').tobytes():
                raise AssertionError('Panel pixels changed while assembling inspection view')
            boxes.append({'row': i, 'column': j, 'xyxy': [x, y, x + size, y + size]})
    return sheet, boxes


def inspect(out, panel, uid, candidate, group, overlay=False):
    out = resolve(out)
    if panel not in ('OLD', 'FRESH') or group not in GROUPS:
        raise ValueError('Unknown panel/group')
    if candidate not in ((0, 1) if panel == 'OLD' else (2, 3)):
        raise ValueError('Candidate is outside the registered panel')
    latest_path = out / 'visuals' / ('latest_' + panel + '.json')
    latest = read_json(latest_path)
    if latest.get('status') != 'COMPLETE' or latest.get('camera_status') != 'FINAL_COMPLETE':
        raise ValueError('Refusing provisional or incomplete gallery')
    manifest_path = resolve(latest['manifest'])
    if digest(manifest_path) != latest['sha256']:
        raise ValueError('Gallery manifest hash changed')
    manifest = read_json(manifest_path)
    if manifest.get('status') != 'COMPLETE' or manifest.get('camera_status') != 'FINAL_COMPLETE':
        raise ValueError('Manifest does not declare a complete final gallery')
    labels = manifest['labels']
    if len(labels) != 9 or len(set(labels)) != 9 or manifest.get('panel') != panel:
        raise ValueError('Expected all nine labels of the requested panel')
    camera_path = resolve(manifest['camera_manifest'])
    camera_sha = manifest['identity']['camera_sha256']
    if digest(camera_path) != camera_sha:
        raise ValueError('Shared camera manifest changed')
    cameras = read_json(camera_path)
    if cameras.get('status') != 'FINAL_COMPLETE' or cameras.get('available_outputs') != 288:
        raise ValueError('Shared camera does not include every registered raw output')
    requests = read_json(out / 'manifests' / (panel + '.json'))['rows']
    siblings = sorted((q for q in requests if q['uid'] == uid), key=lambda q: q['candidate'])
    targets = [q for q in siblings if q['candidate'] == candidate]
    if len(siblings) != 2 or len(targets) != 1:
        raise ValueError('Parent/candidate is not a unique registered request')
    request = targets[0]
    noise_index = [q['candidate'] for q in siblings].index(candidate)
    name = uid + ('_feedback.png' if overlay else '.png')
    pages = [p for p in manifest['parent_pages'] if resolve(p['path']).name == name]
    if len(pages) != 1:
        raise ValueError('Requested final parent page is missing or ambiguous')
    page = pages[0]
    if digest(page['path']) != page['sha256']:
        raise ValueError('Original parent page changed')
    columns = [x for x in GROUPS[group] if x in labels] if overlay else list(GROUPS[group])
    full_columns = labels if overlay else ['Input C'] + labels + ['GT C+free']
    row_names = ('S', 'I', 'E') if overlay else VIEWS
    first_row = noise_index * len(row_names)
    bybox = {(int(b['row']), int(b['column'])): b for b in page['panel_boxes']}
    render_meta = {str(resolve(r['path'])): r for r in manifest['render_receipts']}
    overlay_meta = {str(resolve(r['path'])): r for r in manifest['overlay_receipts']}
    images, sources = [], []
    for i in range(len(row_names)):
        row = []
        for label in columns:
            box = bybox[(first_row + i, full_columns.index(label))]
            path = resolve(box['path'])
            sha = digest(path)
            if overlay:
                receipt = overlay_meta[str(path)]
                expected = receipt['sha256']
                metadata_sha = None
            else:
                meta_path = path.with_suffix('.json')
                record = render_meta[str(meta_path)]
                metadata_sha = digest(meta_path)
                if metadata_sha != record['sha256']:
                    raise ValueError('Panel render metadata changed')
                receipt = read_json(meta_path)
                expected = receipt['png_sha256']
            if sha != expected:
                raise ValueError('Original panel pixels changed')
            with Image.open(path) as image:
                row.append(image.convert('RGB').copy())
            sources.append({'label': label, 'view': row_names[i], 'path': str(path),
                            'sha256': sha, 'metadata_sha256': metadata_sha,
                            'source_page_row': first_row + i, 'source_page_column': full_columns.index(label)})
        images.append(row)
    subtitle = ('Saved S/I/E sample overlays; x-ray points, no new scoring.' if overlay else
                'Original 320px panels; four views + wire; exact pixel copy, no resize or mesh crop.')
    sheet, boxes = _sheet(images, columns, row_names,
                         panel + ' | ' + uid + ' | n' + str(candidate) + ' | ' + group, subtitle)
    buffer = io.BytesIO()
    sheet.save(buffer, format='PNG', compress_level=6)
    png = buffer.getvalue()
    return {
        'metadata': {'schema': 'final_gallery_pixel_inspection_v1', 'panel': panel, 'uid': uid,
                     'request_id': request['request_id'], 'candidate': candidate, 'group': group,
                     'overlay': overlay, 'columns': columns, 'rows': list(row_names),
                     'gallery_manifest': str(manifest_path), 'gallery_manifest_sha256': latest['sha256'],
                     'camera_manifest': str(camera_path), 'camera_sha256': camera_sha,
                     'evidence_page': page['path'], 'evidence_page_sha256': page['sha256'],
                     'sources': sources, 'output_boxes': boxes, 'size': list(sheet.size),
                     'png_bytes': len(png), 'png_sha256': hashlib.sha256(png).hexdigest(),
                     'original_panel_pixels_unchanged': True, 'resizing': False,
                     'model_calls': 0, 'scorer_calls': 0, 'render_calls': 0, 'files_written': 0},
        'png_base64': base64.b64encode(png).decode('ascii'),
    }


def self_test():
    images = [[Image.new('RGB', (320, 320), (i * 31, j * 37, 117)) for j in range(5)] for i in range(5)]
    image, boxes = _sheet(images, list(GROUPS['history']), VIEWS, 'CPU pixel-copy test', 'Synthetic pixels only')
    assert image.size == (1748, 1748) and len(boxes) == 25
    return {'status': 'PASS', 'pixel_exact_panels': 25, 'model_calls': 0,
            'scorer_calls': 0, 'render_calls': 0, 'files_written': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--panel', choices=('OLD', 'FRESH'))
    parser.add_argument('--uid')
    parser.add_argument('--candidate', type=int)
    parser.add_argument('--group', choices=tuple(GROUPS))
    parser.add_argument('--overlay', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    else:
        if any(value is None for value in (args.panel, args.uid, args.candidate, args.group)):
            parser.error('--panel --uid --candidate --group are required')
        result = inspect(args.out, args.panel, args.uid, args.candidate, args.group, args.overlay)
    print(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
