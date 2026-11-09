"""Merge completed actual PNG-review shards; no image inference or model call."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, '/mnt/e/MeshFlow/r2/pilots/anchor_aware_denoiser_v1')
from common import OUT, checked, read, sha, csvwrite, write, textfile

G = OUT / 'gallery'
S = G / 'review_shards'
SHARDS = [
    ('anchor_data', 'anchor_data_images.json', 'anchor_data_notes.md', 36, 20),
    ('proposal', 'proposal_images.json', 'proposal_notes.md', 36, 12),
    ('wrapper', 'wrapper_images.json', 'wrapper_notes.md', 72, 24),
]

def csvrows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

manifest = read(G / 'gallery_manifest.json')
assert manifest['status'] == 'COMPLETE' and manifest['images'] == 56
index = csvrows(OUT / 'evaluation/post_index.csv')
pending = csvrows(G / 'review_pending.csv')
fields = list(pending[0])
key = lambda r: (r['denoiser'], r['generator'], r['request_id'])
expected = {(r['denoiser'], r['label'], r['request_id']): r for r in index}
assert len(expected) == len(index) == 144
pngs = {str(Path(r['png']).resolve()): r['png_sha256'] for r in manifest['receipts']}
assert len(pngs) == 56
combined, seen_images, sources, notes = {}, {}, {}, []
for name, receiptname, notesname, nr, ni in SHARDS:
    cp, ip, np = S / (name + '.csv'), S / receiptname, S / notesname
    rows, receipt, note = csvrows(cp), read(ip), np.read_text(encoding='utf-8-sig')
    assert len(rows) == nr and receipt['status'] == 'COMPLETE'
    assert receipt['reviewed_POST_outputs'] == nr and receipt['reviewed_images'] == ni
    assert len(receipt['images']) == ni and note.strip()
    for row in rows:
        k = key(row)
        assert k in expected and k not in combined, ('unexpected/duplicate review', k)
        assert row['parent_id'] == expected[k]['parent_id']
        assert row['region_type'] == expected[k]['region_type'] and row['tier'] == expected[k]['tier']
        assert row['review_kind'] == 'AI_PROXY' and row['review_status'] == 'COMPLETE'
        assert row['blind'].lower() in ('false', '0', 'no')
        for field in ('C_use', 'boundary_continuation', 'overall_quality', 'thin_structure', 'observations'):
            assert row[field].strip().upper() not in ('', 'PENDING', 'NOT_REVIEWED', 'TODO')
        assert set(row) == set(fields)
        combined[k] = row
    for item in receipt['images']:
        p = str(Path(item['path']).resolve())
        assert item['actually_viewed'] is True
        assert p not in seen_images and p in pngs
        assert item['sha256'] == pngs[p] == sha(p)
        seen_images[p] = dict(item, path=p, source_shard=str(ip))
    sources.update({str(p): sha(p) for p in (cp, ip, np)})
    notes.append(note.rstrip())
assert set(combined) == set(expected) and len(combined) == 144
assert set(seen_images) == set(pngs) and len(seen_images) == 56
ordered_rows = [combined[key(row)] for row in pending]
ordered_images = [seen_images[str(Path(r['png']).resolve())] for r in manifest['receipts']]
assert Counter(r['generator'] for r in ordered_rows) == {
    'R2_FROZEN':36, 'R2_CONT_OLD':36, 'R2_CONT_USER':36, 'R2_USER_MIX':36}

csvwrite(G / 'AI_PROXY_per_output_review.csv', ordered_rows)
write(G / 'AI_PROXY_review_receipt.json', dict(
    schema='anchor_aware_denoiser_actual_AI_PROXY_review_v1', status='COMPLETE',
    review_kind='AI_PROXY', blind=False, reviewed_POST_outputs=144, reviewed_images=56,
    reviewed_whole_object_images=48, reviewed_local_images=8,
    images=ordered_images, source_shard_hashes=sources,
    gallery_manifest_sha256=sha(G / 'gallery_manifest.json'),
    post_index_sha256=sha(OUT / 'evaluation/post_index.csv'),
    merge_script_sha256=sha(__file__),
    per_output_csv_sha256=sha(G / 'AI_PROXY_per_output_review.csv'),
    review_scope='All 144 POST outputs; 48 whole-object four-stage sheets and 8 predeclared local USER_MIX sheets.',
    no_image_selected_by_score=True, no_GT_ranking=True,
    no_model_calls=True, no_scores_computed=True,
    limitations=['Non-blind AI proxy, not a human study.',
        'These images establish observed appearance only; no vertex-ID or manifold certification.',
        'Gray/blue proximity and C preservation do not establish joining.',
        'Free-text review labels are not a preregistered quantitative success-rate metric.']))
intro = '''# Actual AI_PROXY visual review

All **144 POST outputs** were actually reviewed in the fixed **48 whole-object original PNG sheets**, plus all **8 predeclared USER_MIX local original PNG sheets**. RAW, D0_OFFICIAL, D_CONT_NO_ROLE and D_ANCHOR are displayed together. Three agents inspected non-overlapping image assignments and saved their original per-output CSVs, notes and image-hash receipts. The merged receipt preserves their provenance. This is non-blind AI_PROXY review, not a human rating or topological proof.

The per-output CSV separates C use, boundary continuation, whole-shape quality and thin structure. Uncertain connection is retained as uncertain. No request was selected by its score, and no completion is ranked by resemblance to the unique source mesh. Small numerical differences can exist even where whole-object appearance is similar.

The detailed independent observations follow verbatim. Their common visual finding is that the three POST variants generally retain very similar large-scale shapes and C relationships; no repeated, unmistakable anchor-specific boundary adoption is established by these sheets. Persistent placement, layering and thin-support problems are recorded individually rather than hidden by a single aggregate label.

There are material **shared adaptation improvements**, so the outputs are not declared identical: in R2_FROZEN 33ec n1, both adapted denoisers retain more thin-leg structure than D0; in R2_CONT_USER bdd n1, both remove a thin forward projection at mid-back. The C offsets persist in both examples. These improvements support a distinction between ordinary denoiser adaptation and the incremental effect of the anchor-role embedding. USER_MIX 33ec n0 still shows marked leg thinning after all three POST variants.

'''
textfile(G / 'AI_PROXY_REVIEW.md', intro + '\n\n---\n\n'.join(notes) + '\n')
print(json.dumps(dict(status='COMPLETE', outputs=144, actually_viewed_original_PNGs=56,
    csv_sha256=sha(G / 'AI_PROXY_per_output_review.csv'),
    receipt_sha256=sha(G / 'AI_PROXY_review_receipt.json'),
    notes_sha256=sha(G / 'AI_PROXY_REVIEW.md'))))
