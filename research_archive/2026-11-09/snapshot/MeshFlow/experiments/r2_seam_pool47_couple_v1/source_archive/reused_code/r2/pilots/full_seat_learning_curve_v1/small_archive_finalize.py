"""Finalize the prepared small archive without rereading OT or changing results."""
import csv, difflib, hashlib, io, json, zipfile
from pathlib import Path
from . import common as C
from . import artifact_archive as A

def sha(b): return hashlib.sha256(b).hexdigest()
def jb(v): return (json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode()
def rows(b): return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
def cb(rs):
    f=io.StringIO(newline=''); w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    return f.getvalue().encode('utf-8-sig')

def main():
    prepared=C.OUT/'archive/prepared_entries.zip'
    ready=A.readiness()
    assert ready['status']=='READY_TO_ARCHIVE', ready['missing']
    with zipfile.ZipFile(prepared) as z:
        assert z.testzip() is None
        data={n:z.read(n) for n in z.namelist()}
    manifest=json.loads(data.pop('MANIFEST.json'))
    assert set(data)==set(manifest['file_hashes'])
    assert all(sha(b)==manifest['file_hashes'][n] for n,b in data.items())
    assert data['evidence/REPORT.md']==(C.OUT/'REPORT.md').read_bytes()
    omit=(
        'evidence/training/FULL_SEAT/training_log.jsonl',
        'evidence/audits/interim_execution_review.json',
        'evidence/audits/final/phases/contracts_and_prefix.json',
    )
    omitted=[]
    for name in omit:
        payload=data.pop(name)
        original=C.OUT/name.removeprefix('evidence/')
        assert sha(payload)==C.sha(original)
        omitted.append(dict(archive_path_omitted=name,local_path=str(original),bytes=len(payload),sha256=sha(payload),reason='REDUNDANT_RAW_LOG_OR_PRIOR_AUDIT_INPUT_HASH_LEDGER; all per-update/per-output CSV and actual audit code retained'))
    inventory=rows(data['artifact_inventory.csv'])
    for r in inventory:
        if r.get('archive_path') in omit:
            r.update(archive_path='',in_zip='False',reason='LOCAL_ONLY_REDUNDANT_TRACE_SMALL_PACKAGE')
    record=dict(status='SMALL_PACKAGE_SCOPE_RECORDED',prepared_zip_sha256=C.sha(prepared),local_only=omitted,
        scientific_results_changed=False,all_CSV_retained=True,all_scoring_sources_retained=True,all_48_previews_retained=True,
        model_calls=0,new_scores=0,training_updates=0)
    data['LOCAL_ONLY_REDUNDANT_TRACES.json']=jb(record)
    script=Path(__file__).resolve(); scriptname='source/repository/'+script.relative_to(C.ROOT).as_posix()
    data[scriptname]=script.read_bytes()
    sources=rows(data['source_hashes.csv'])
    assert str(script) not in {r['path'] for r in sources}
    sources.append(dict(path=str(script),sha256=sha(data[scriptname]),bytes=len(data[scriptname]),archive_path=scriptname,registered_sha256=''))
    data['source_hashes.csv']=cb(sources)
    inv={k:'' for k in inventory[0]};inv.update(path=str(script),exists='True',bytes=str(len(data[scriptname])),sha256=sha(data[scriptname]),checksum_method='ACTUAL_FINALIZATION_SHA256',archive_path=scriptname,in_zip='True',reason='INCLUDED_FINAL_ARCHIVE_SOURCE')
    inventory.append(inv);data['artifact_inventory.csv']=cb(inventory)
    codefiles=sorted(p for p in C.CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in A.SOURCE_SUFFIXES)
    data['code.diff'],git=A._code_diff(codefiles)
    data['git_state.json']=jb(git)
    note='\n\nSmall-package scope: LOCAL_ONLY_REDUNDANT_TRACES.json names three redundant raw-log / prior-audit ledgers kept on disk with exact SHA256. Every CSV, scoring source, actual review row and all 48 preview images remain included. The complete per_update.csv provides all 6000 updates. Original full traces remain local. This package was finalized from the prepared entries with `python -B -m r2.pilots.full_seat_learning_curve_v1.small_archive_finalize`; no OT rescan or scientific rerun was made.\n'
    data['RESUME_AND_CONTENTS.md']+=note.encode()
    manifest.update(status='COMPLETE',git=git,small_package_scope_record='LOCAL_ONLY_REDUNDANT_TRACES.json',
        archive_finalizer_source=scriptname,archive_finalizer_source_sha256=sha(data[scriptname]),
        no_scientific_result_omission=True,file_hashes={n:sha(b) for n,b in sorted(data.items())})
    data['MANIFEST.json']=jb(manifest)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_LZMA) as z:
        for name,payload in sorted(data.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0))
            z.writestr(info,payload,compress_type=zipfile.ZIP_LZMA)
    payload=buffer.getvalue()
    assert len(payload)<10_000_000, len(payload)
    assert A.readiness()==ready
    out=C.OUT/'archive'/('snapshot_'+sha(payload)[:16])
    zpath=out/'R2_FULL_SEAT_LEARNING_CURVE_V1_REVIEW.zip'
    C._write_bytes(zpath,payload)
    for name in ('artifact_inventory.csv','source_hashes.csv','code.diff','LOCAL_ONLY_REDUNDANT_TRACES.json'):
        C._write_bytes(out/name,data[name])
    C.write(out/'manifest.json',manifest)
    result=dict(status='COMPLETE',version=A.VERSION,zip_path=str(zpath),zip_sha256=sha(payload),zip_bytes=len(payload),max_bytes=10_000_000,
        entries=len(data),actual_AI_PROXY_main_cells=192,manifest_path=str(out/'manifest.json'),model_calls=0,training_updates=0,new_reviews=0,
        scope_note='Three redundant raw-log/prior-audit ledgers local-only with exact hashes; every CSV, score source and 48 preview images retained.')
    C.write(out/'completion.json',result)
    C.write(C.OUT/'archive/latest.json',dict(path=str(out/'completion.json'),sha256=C.sha(out/'completion.json'),**result),replace=True)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()