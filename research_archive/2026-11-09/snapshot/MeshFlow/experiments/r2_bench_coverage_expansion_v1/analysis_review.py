"""Collect actual image reviews without inventing missing ratings."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write_csv(p, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

def main():
    review_dir = OUT/'visual_review'
    sources = sorted(review_dir.glob('review_*.json'))
    rubric = sha(OUT/'REVIEW_RUBRIC.md')
    rows = []
    for p in sources:
        doc = read(p)
        if doc['rubric_sha256'] != rubric:
            raise ValueError('Rubric mismatch: '+str(p))
        rows.extend(doc['rows'])
    keys = [(r['request_id'],r['label'],r['stage']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate review cell')
    if len(rows) != 288:
        return dict(status='WAIT', reviewed=len(rows), expected=288)
    rows.sort(key=lambda r:(r['request_id'],r['stage'],r['label']))
    enum = {'C_use':{'YES','NO','UNCERTAIN'},
            'whole_shape':{'COMPLETE','PARTIAL','UNCERTAIN'},
            'layered_or_conflicting':{'YES','NO','UNCERTAIN'},
            'visible_fragment':{'YES','NO','UNCERTAIN'},
            'pair_preference':{'COVERAGE','A','TIE','UNCERTAIN'}}
    for r in rows:
        for k,choices in enum.items():
            if r[k] not in choices:
                raise ValueError((r['request_id'],k,r[k]))
        if 'AI_PROXY' not in r['reviewer']:
            raise ValueError('Reviewer identity missing')
    pairs = []
    by = defaultdict(dict)
    for r in rows:
        by[(r['request_id'],r['stage'])][r['label']] = r
    for (rid,stage),g in sorted(by.items()):
        a,b=g['A_S1500'],g['COVERAGE_S1500']
        if a['pair_preference'] != b['pair_preference']:
            raise ValueError('Conflicting paired preference')
        p = dict(request_id=rid,stage=stage,split=b['split'],parent_id=b['parent_id'],
                 condition_id=b['condition_id'],part_type=b['part_type'],
                 pair_preference=b['pair_preference'],
                 C_use_transition=a['C_use']+'->'+b['C_use'],
                 shape_transition=a['whole_shape']+'->'+b['whole_shape'])
        for k in ('C_use','whole_shape','layered_or_conflicting','visible_fragment','note'):
            p[k+'_A']=a[k]
            p[k+'_COVERAGE']=b[k]
        pairs.append(p)
    groups = defaultdict(list)
    for r in rows:
        groups[(r['split'],r['stage'],r['label'])].append(r)
    summary=[]
    for (split,stage,label),rs in sorted(groups.items()):
        item=dict(split=split,stage=stage,label=label,n=len(rs))
        for k in ('C_use','whole_shape','layered_or_conflicting','visible_fragment'):
            for value in sorted(enum[k]):
                item[k+'__'+value]=sum(r[k]==value for r in rs)
        summary.append(item)
    win_groups=defaultdict(list)
    for r in pairs:
        win_groups[(r['split'],r['stage'])].append(r)
    preferences=[dict(split=k[0],stage=k[1],n=len(rs),
        **{v:sum(r['pair_preference']==v for r in rs) for v in sorted(enum['pair_preference'])})
        for k,rs in sorted(win_groups.items())]
    write_csv(review_dir/'per_cell.csv',rows)
    write_csv(review_dir/'paired_requests.csv',pairs)
    write_csv(review_dir/'summary.csv',summary)
    write_csv(review_dir/'preferences.csv',preferences)
    result=dict(status='COMPLETE',reviewed_cells=288,reviewed_pages=144,
        reviewer_kind='AI_PROXY_NONBLIND',rubric_sha256=rubric,
        sources={str(p):sha(p) for p in sources},summary=summary,preferences=preferences,
        caveat='Visual judgments are not human ratings or certified topology; objective scores remain separate.')
    (review_dir/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return {k:v for k,v in result.items() if k not in ('sources','summary')}

if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False,indent=2))
