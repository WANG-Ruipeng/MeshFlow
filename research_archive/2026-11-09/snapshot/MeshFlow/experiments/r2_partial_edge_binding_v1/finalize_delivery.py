"""Package already-computed CPU results and the root reviewer's actual image notes.

No decoder, metric, neural framework, or renderer is imported here.
"""
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


rows = read_csv(OUT / 'per_output.csv')
lookup = {(r['label'], r['short_id']): r for r in rows}
gallery = json.loads((OUT / 'gallery/manifest.json').read_text())['images']
selection = json.loads((OUT / 'gallery/selection.json').read_text())['selection']
reasons = {(r['label'], r['short_id']): r['reasons'] for r in selection}
full_size = {('B_020', '73b369b9_n0'), ('B_020', '73b369b9_n1'),
             ('A_035', '2d08a64e_n1')}
reviews = []
for index, image in enumerate(gallery):
    key = image['label'], image['short_id']
    r = lookup[key]
    sheet = OUT / f'gallery/contact_sheets/sheet_{index // 6:02d}.jpg'
    # These observations were entered AFTER the root reviewer displayed all six
    # contact sheets and the three listed full-resolution paired image files.
    if image['short_id'].startswith('73b369b9_n3'):
        use = 'UNCERTAIN'
        note = ('Blue C is partly obscured by multiple gray strips. The existing '
                'layered seat appearance persists; no new overall silhouette '
                'break is visible. This is not promoted to a use success.')
    elif image['short_id'].startswith('73b369b9'):
        use = 'YES'
        note = ('The blue surface still reads as the seat between the front '
                'support and back. The cantilever silhouette is retained. '
                'Small rim changes do not visually establish a continuous seam.')
    elif image['short_id'].startswith('85b16941'):
        use = 'YES'
        note = ('The blue seat, back and slender supports retain their roles '
                'and overall silhouette. Local rim changes are small at the '
                'reviewed scale; this does not certify absence of cracks.')
    elif image['short_id'] == '2d08a64e_n1':
        use = 'YES'
        note = ('Blue C remains the seat inside the surrounding frame; the '
                'chair body remains present. The pre-existing bent side/seat '
                'profile and back gaps are not repaired by this operation.')
    else:
        use = 'UNCERTAIN'
        note = ('The pre-existing unusual seat/side profile is retained. No '
                'clear newly missing major component is visible. Its use '
                'ambiguity is not resolved by a nearby shared edge.')
    if key == ('A_035', '2d08a64e_n1'):
        note += (' Full-size side view shows a local change/protrusion at the '
                 'front seat edge. Use is retained, but local damage is '
                 'UNCERTAIN visually; the independent ledger reports torn edges.')
    reviews.append(dict(
        label=key[0], short_id=key[1], request_id=r['request_id'], split=r['split'],
        reviewer='AI_PROXY', blinded=False,
        review_scope='CONTACT_SHEET_PLUS_FULL_SIZE' if key in full_size else 'CONTACT_SHEET',
        contact_sheet=str(sheet.relative_to(OUT)), contact_sheet_sha256=sha(sheet),
        full_image=f'gallery/{key[0]}/{key[1]}.jpg', full_image_sha256=image['sha256'],
        original_RAW_adoption=r['raw_adoption'], POST_C_use=use,
        newly_missing_major_body_component='NO_OBVIOUS_AT_REVIEWED_SCALE',
        local_damage_clearance='NOT_CERTIFIED', complete_seam='NOT_CERTIFIED',
        screening_joint=r['screening_joint'], baseline_joint=r['baseline_joint'],
        selection_reasons=';'.join(reasons[key]), observation=note))
write_csv('actual_image_review.csv', reviews)
new_reviews = [r for r in reviews
               if str(r['screening_joint']).lower() in ('true', '1')
               and str(r['baseline_joint']).lower() not in ('true', '1')]
assert len(new_reviews) == 12 and all(r['POST_C_use'] == 'YES' for r in new_reviews)

ledger_fields = ['label', 'short_id', 'parent_id', 'noise', 'split', 'raw_adoption',
                 'baseline_joint', 'screening_joint', 'accepted', 'candidates',
                 'CF_retained_legal_edges', 'CF_retained_baseline_length',
                 'CF_retained_current_length', 'CF_lost_legal_edges',
                 'CF_lost_baseline_length', 'CF_new_legal_edges',
                 'CF_new_current_length', 'FF_lost_legal_edges',
                 'FF_lost_baseline_length', 'affected_deleted_source_faces',
                 'affected_deleted_source_area', 'FF_new_illegal_pairs',
                 'FF_illegal_pairs_removed', 'CF_new_illegal_pairs']
write_csv('paired_connectivity_ledger.csv', [{k: r[k] for k in ledger_fields} for r in rows])
edge_changes, pair_changes = [], []
for r in rows:
    cell = OUT / 'cells' / r['label'] / r['short_id']
    identity = {k: r[k] for k in ('label', 'short_id', 'split')}
    for e in read_csv(cell / 'baseline_edge_fates.csv'):
        if e['fate'] != 'RETAINED_LEGAL':
            edge_changes.append(dict(identity, **e))
    for e in read_csv(cell / 'new_edges.csv'):
        edge_changes.append(dict(identity, fate='NEW_LEGAL', **e))
    for e in read_csv(cell / 'illegal_pair_fates.csv'):
        if e['fate'] != 'PERSISTENT_ILLEGAL':
            pair_changes.append(dict(identity, **e))
write_csv('edge_changes.csv', edge_changes)
write_csv('illegal_pair_changes.csv', pair_changes)
print(json.dumps(dict(image_reviews=len(reviews), new_screening_reviews=len(new_reviews),
                      paired_ledgers=len(rows), edge_changes=len(edge_changes),
                      illegal_pair_changes=len(pair_changes))))
