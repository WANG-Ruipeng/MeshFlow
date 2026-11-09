"""Store explicit human/AI visual observations; never infer labels from metrics.

Input is a JSON list on stdin. Each item has request_id, stage, display_label,
the eight original judgment fields, and notes. Call only after viewing images.
"""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, sys

OUT=Path('/mnt/e/MeshFlow/experiments/r2_user_data_scaling_32_67_v1')
FIELDS=('C_adoption','C_floating','C_replaced_by_other_surface','C_wrong_layer',
        'whole_body_complete','support_missing','fragments_or_folded_layers','visible_join')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--chunk',type=int,required=True)
    args=ap.parse_args();chunk=str(args.chunk)
    work=OUT/'review_work';gallery=OUT/'evaluation/gallery'
    assigned=json.loads((work/'assignments.json').read_text())['chunks'][chunk]
    allowed={x['request_id'] for x in assigned}
    template_path=gallery/'AI_PROXY_review_template.csv'
    if not template_path.exists():template_path=work/'early_review_template.csv'
    template=readcsv(template_path)
    key=lambda r:(r['request_id'],r['stage'],r['display_label'])
    table={key(r):r for r in template if r['request_id'] in allowed}
    target=work/f'chunk_{chunk}.csv'
    previous={key(r):r for r in readcsv(target)} if target.exists() else {}
    incoming=json.loads(sys.stdin.buffer.read().decode('utf-8-sig'))
    if not isinstance(incoming,list) or not incoming:raise ValueError('Nonempty explicit observations required')
    for note in incoming:
        k=key(note)
        if k not in table:raise ValueError(f'Unassigned identity {k}')
        if any(note.get(f) not in ('YES','NO','UNCERTAIN') for f in FIELDS):
            raise ValueError('Each actually observed dimension must explicitly say YES/NO/UNCERTAIN')
        if len(note.get('notes','').strip())<8:raise ValueError('Concrete visual notes required')
        row=dict(table[k]);row.update({f:note[f] for f in (*FIELDS,'notes')})
        page=gallery/note['stage']/(note['request_id']+'.jpg')
        receipt=json.loads(page.with_suffix('.json').read_text())
        digest=sha(page)
        if digest!=receipt['image_sha256']:raise ValueError('Viewed image identity changed')
        row.update(review_status='REVIEWED_AI_PROXY',reviewer='AI_PROXY',blind='False',reviewed_image_sha256=digest)
        if k in previous and previous[k]!=row:raise ValueError(f'Existing review cannot silently change: {k}')
        previous[k]=row
    rows=[previous[key(r)] for r in template if key(r) in previous]
    tmp=target.with_suffix('.csv.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(template[0]));w.writeheader();w.writerows(rows)
    tmp.replace(target)
    pages={(r['request_id'],r['stage']) for r in rows}
    rp=work/f'chunk_{chunk}_receipt.json'
    first=json.loads(rp.read_text()).get('first_recorded_utc') if rp.exists() else None
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    status={'status':'COMPLETE' if len(rows)==len(table) else 'IN_PROGRESS',
            'chunk':args.chunk,'review_kind':'AI_PROXY','blind':False,
            'numeric_ranking_seen_before_review':False,'rows_reviewed':len(rows),
            'images_actually_viewed':len(pages),'expected_rows':len(table),
            'review_file':str(target),'review_file_sha256':sha(target),
            'first_recorded_utc':first or now,'updated_utc':now}
    rp.write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status),flush=True)
if __name__=='__main__':main()
