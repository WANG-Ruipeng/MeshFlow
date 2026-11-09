"""Merge actual human-visible AI_PROXY records; never infer labels from metrics."""
import argparse, collections, csv, json
from pathlib import Path
from .protocol import ROOT, OUT, WAVE1, WAVE2
from r2.pilots.seam_abc_v1 import generate as G

FIELDS=('body_complete','C_adopted_as_seat','obvious_fragments_or_folded_sheets','visible_cracks')

def load(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def csvwrite(path, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def resolve(path):
    value=path.replace('\\','/')
    if value.lower().startswith('e:/meshflow/'):
        return ROOT/value[len('E:/MeshFlow/'):]
    return Path(value)

def run(scope):
    names=['wave1_73_3f.csv','wave1_85_8c.csv','wave1_DEV_root.csv',
           'wave1_bound_73_3f.csv','wave1_bound_85n1.csv','wave1_bound_DEV_root.csv']
    if scope=='final':
        names += ['wave2_73_3f.csv','wave2_85_8c.csv','wave2_DEV_root.csv',
                  'wave2_bound_73_3f.csv','wave2_bound_85n1.csv','wave2_bound_DEV_root.csv']
    registered=G.read(OUT/'protocol.json')['evaluation']['requests']
    identity={q['request_id']:q for q in registered}; merged={}; refs={}; duplicates=[]
    for name in names:
        path=OUT/'visual_review_parts'/name; refs[str(path)]=G.sha(path)
        for index,row in enumerate(load(path),2):
            if None in row:raise ValueError(f'Extra CSV fields: {name}:{index}')
            if row['actually_viewed'].lower() not in ('yes','true','1'):raise ValueError('Unviewed record')
            if row['request_id'] not in identity:raise ValueError('Unknown request')
            for f in FIELDS:
                if row[f] not in ('YES','NO','UNCERTAIN'):raise ValueError(f'Invalid visual label: {name}:{index}:{f}')
            image=resolve(row['page_path'])
            if G.sha(image)!=row['page_sha256']:raise ValueError('Image identity changed')
            key=(row['label'],row['request_id'],row['stage'])
            if key in merged:
                duplicates.append(dict(label=key[0],request_id=key[1],stage=key[2],source=name,
                    handling='Supplementary actual re-review retained in original file; first canonical record kept'))
                continue
            row.update(reviewer_identity=row['reviewer'],reviewer='AI_PROXY_NONBLIND',actually_viewed='True',
                       page_path=str(image),source_review_file=str(path),source_review_row=index,
                       split=identity[row['request_id']]['split'],parent_id=identity[row['request_id']]['parent_id'])
            merged[key]=row
    active=(*WAVE1,*WAVE2) if scope=='final' else WAVE1
    expected={(a,q['request_id'],'POST') for a in active for q in registered}
    selected={k for k in merged if k[0] in active and k[2]=='POST'}
    if selected!=expected:raise ValueError(f'POST coverage differs: {len(selected)} vs {len(expected)}')
    out=OUT/('visual_review.csv' if scope=='final' else 'visual_review_wave1.csv')
    rows=list(merged.values());csvwrite(out,rows)
    summary=[]
    for label in dict.fromkeys(r['label'] for r in rows):
        for stage in ('POST','BOUND015'):
            for split in ('TRAIN','DEV','ALL'):
                rr=[r for r in rows if r['label']==label and r['stage']==stage and (split=='ALL' or r['split']==split)]
                if not rr:continue
                for field in FIELDS:
                    c=collections.Counter(r[field] for r in rr)
                    summary.append(dict(label=label,stage=stage,split=split,field=field,n=len(rr),
                        **{x:c[x] for x in ('YES','NO','UNCERTAIN')},
                        scope='All new POST; preregistered critical BOUND subset, not whole BOUND population'))
    csvwrite(OUT/('visual_summary.csv' if scope=='final' else 'visual_summary_wave1.csv'),summary)
    result=dict(status='ACTUAL_AI_PROXY_RECORDS_COMPLETE',scope=scope,rows=len(rows),new_POST=len(selected),
        parts=refs,canonical_sha256=G.sha(out),supplementary_duplicate_reviews=duplicates,
        reviewer='AI_PROXY_NONBLIND',metric_inferred_labels=False,
        limitation='Finite views cannot certify topology, all surface defects, human preferences or reconstruction identity.')
    G.write(OUT/('visual_completion.json' if scope=='final' else 'visual_wave1_completion.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('parts','supplementary_duplicate_reviews')}));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope',choices=['wave1','final']);run(p.parse_args().scope)
