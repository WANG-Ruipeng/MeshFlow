"""Join actually completed AI observations and explicit root verdicts, no grading heuristic."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'native_t1/runs/chair_capped500_gradbudget_v1'
LABELS={'START','HIST_TEACHER500','HIST_CAPPED500','T500','T1000','L500','L1000','G500','G1000'}
PAIRS=(('HIST_CAPPED500','HIST_TEACHER500'),('L500','T500'),('G500','T500'),('L1000','T1000'),('G1000','T1000'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def run(out=OUT):
    out=Path(out);folder=out/'visuals/reviews'
    decisions_path=folder/'root_comparison_decisions.json'
    decisions=read(decisions_path)
    assert decisions['reviewer_type']=='AI_VISUAL_REVIEW'
    judgments={(r['panel'],r['candidate_label'],r['matched_label']):r for r in decisions['comparisons']}
    expected={(panel,c,m) for panel in ('OLD','FRESH') for c,m in PAIRS}
    assert set(judgments)==expected and len(decisions['comparisons'])==len(expected)
    all_rows=[];result=[];hashes={str(decisions_path):sha(decisions_path)}
    for panel in ('OLD','FRESH'):
        requests=read(out/'manifests'/(panel+'.json'))['rows']
        ridmap={r['request_id']:r for r in requests}
        assert len(requests)==len(ridmap)==16
        latest=read(out/'visuals'/('latest_'+panel+'.json'))
        assert latest['status']=='COMPLETE' and latest['camera_status']=='FINAL_COMPLETE'
        gallery_path=Path(latest['manifest']);gallery=read(gallery_path)
        assert sha(gallery_path)==latest['sha256'] and gallery['camera_status']=='FINAL_COMPLETE'
        assert gallery['all_vertices_inside'] and gallery['all_geometry_unchanged']
        hashes[str(gallery_path)]=sha(gallery_path)
        indexed={};contributors=[]
        for author in ('root','scoring','diagnostics','coupling'):
            path=folder/f'partial_{author}_{panel}.json';part=read(path)
            assert part['reviewer_type']=='AI_VISUAL_REVIEW' and part['scope_status']=='PARTIAL'
            assert len(part['request_ids'])==len(set(part['request_ids']))==4
            assert set(part['request_ids'])<=set(ridmap)
            assert len(part['outputs'])==36
            keys={(r['request_id'],r['label']) for r in part['outputs']}
            assert keys=={(rid,label) for rid in part['request_ids'] for label in LABELS}
            for row in part['outputs']:
                key=(row['request_id'],row['label']);assert key not in indexed
                assert row['uid']==ridmap[row['request_id']]['uid']
                for field in ('missing_body','duplicate_overlap','long_thin_sheet','chair_plausibility',
                              'certainty','vs_START','vs_matched_T','evidence_pages','view_scope'):
                    assert field in row,(key,field)
                assert row['evidence_pages'] and {'front_3q','back_3q','front','side','wire'}<=set(row['view_scope'])
                indexed[key]=row
                all_rows.append(dict(panel=panel,reviewer=author,**row))
            hashes[str(path)]=sha(path);contributors.append(dict(author=author,path=str(path),sha256=sha(path)))
        assert len(indexed)==144
        save(folder/('complete_panel_'+panel+'.json'),dict(reviewer_type='AI_VISUAL_REVIEW',scope_status='COMPLETE',
            panel=panel,request_ids=list(ridmap),outputs=list(indexed.values()),contributors=contributors,
            gallery_manifest=str(gallery_path),gallery_sha256=sha(gallery_path),human_usable=None,user_study='NOT_RUN'))
        for candidate,matched in PAIRS:
            judgment=judgments[(panel,candidate,matched)]
            assert judgment['status'] in ('PASS','FAIL','INCOMPLETE') and judgment['rationale']
            observations=[dict(request_id=rid,uid=ridmap[rid]['uid'],candidate=indexed[(rid,candidate)],
                matched=indexed[(rid,matched)],START=indexed[(rid,'START')]) for rid in ridmap]
            value=dict(judgment,reviewer_type='AI_VISUAL_REVIEW',scope_status='COMPLETE',request_ids=list(ridmap),
                observations=observations,contributors=contributors,gallery_manifest=str(gallery_path),
                gallery_sha256=sha(gallery_path),decision_source_sha256=sha(decisions_path),
                human_usable=None,user_study='NOT_RUN',model_calls=0,scoring_calls=0,
                rule='Explicit evidence-based root verdict after complete AI image review; no automatic style or leg-count veto.')
            path=folder/f'{candidate}_vs_{matched}__{panel}.json';save(path,value)
            result.append(dict(panel=panel,candidate=candidate,matched=matched,status=judgment['status'],path=str(path),sha256=sha(path)))
    csv_path=folder/'all_output_observations.csv'
    fields=list(dict.fromkeys(k for r in all_rows for k in r))
    with csv_path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in all_rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    receipt=dict(status='PASS',scope_status='COMPLETE',reviewer_type='AI_VISUAL_REVIEW',
        label_outputs_reviewed=len(all_rows),expected_label_outputs=288,comparisons=result,source_hashes=hashes,
        observations_csv=str(csv_path),observations_csv_sha256=sha(csv_path),human_usable=None,user_study='NOT_RUN',model_calls=0,scoring_calls=0)
    assert len(all_rows)==288
    save(folder/'completion.json',receipt)
    return receipt


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False))
