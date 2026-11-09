"""Read-only saved C-only layout versus source-completion diagnostic.

Exploratory; source owners are one legal reference completion, not the unique
generative answer. No head/model forward, threshold search, geometry scan or
modification of the formal evaluation schema.
"""
from __future__ import annotations
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'experiments/r2_bench_control_transfer_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Read-only diagnostic refuses overwrite: ' + str(path))
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def csvwrite(path, rows):
    if Path(path).exists():
        raise FileExistsError(str(path))
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def boundary(faces):
    book = collections.defaultdict(list)
    for fi, face in enumerate(faces):
        for j in range(3):
            a, b = int(face[j]), int(face[(j+1)%3])
            book[tuple(sorted((a,b)))].append((fi,a,b))
    return np.asarray([owners[0] for edge, owners in sorted(book.items()) if len(owners)==1], dtype=np.int64)


def main():
    registered = read(RUN/'data/evaluation_requests.json')
    by_c = {q['condition_id']:q for q in registered}
    assert len(by_c)==12 and len(registered)==24
    layout_root = RUN/'evaluation/SEAM/layouts'
    missing = [cid for cid in by_c if not (layout_root/cid/'receipt.json').exists()]
    if missing:
        raise RuntimeError('Wait for every registered layout receipt: '+repr(missing))
    rows=[]; edges=[]; inputs={}
    for cid, q in sorted(by_c.items()):
        folder=layout_root/cid; lp=folder/'layout.npz'; rp=folder/'receipt.json'; cp=Path(q['case_path'])
        r=read(rp)
        assert r['status']=='COMPLETE' and r['condition_id']==cid
        assert r['occupancy_forward']==1 and r['threshold']==0 and r['source_labels_read'] is False
        with np.load(lp,allow_pickle=False) as f: a={k:f[k] for k in f.files}
        with np.load(cp,allow_pickle=False) as f: c={k:f[k] for k in f.files}
        for key in ('C_vertices','C_faces_original','C_faces_model'):
            assert np.array_equal(a[key],c[key]), (cid,key)
        assert np.array_equal(a['features'],c['edge_features']), cid
        assert np.array_equal(a['edge_records'],boundary(c['C_faces_original'])), cid
        p=a['occupied']; z=a['occupancy_logits']; y=c['occupied_labels']
        alias=c['source_alias_labels']; supported=c['source_supported_labels']
        assert p.dtype==np.bool_ and y.dtype==np.bool_ and p.shape==y.shape==z.shape
        assert np.isfinite(z).all() and np.array_equal(p,z>=0)
        assert np.array_equal(y,supported & ~alias)
        assert r['edges']==len(p) and r['active']==int(p.sum())
        tp=int((p & y).sum()); fp=int((p & ~y).sum()); fn=int((~p & y).sum()); tn=int((~p & ~y).sum())
        row=dict(condition_id=cid,parent_id=q['parent_id'],split=q['split'],official_split=q['official_split'],
            official_pretraining_exposure=q['official_pretraining_exposure'],N=q['N'],K=q['K'],
            boundary_edges=len(p),source_owner_count=int(y.sum()),source_alias_count=int(alias.sum()),
            source_null_unsupported_count=int((~supported).sum()),predicted_active_count=int(p.sum()),
            TP=tp,FP=fp,FN=fn,TN=tn,false_positive_alias=int((p & alias).sum()),
            false_positive_null_unsupported=int((p & ~supported).sum()),
            source_owner_recall=tp/int(y.sum()) if y.sum() else None,
            source_owner_precision=tp/int(p.sum()) if p.sum() else None,
            edge_agreement_fraction=float(np.mean(p==y)),exact_source_layout_match=bool(np.array_equal(p,y)),
            all_edges_inactive=bool(not p.any()),min_logit=float(z.min()),max_logit=float(z.max()),
            threshold=0.0,interpretation='EXPLORATORY_SOURCE_COMPLETION_AGREEMENT_NOT_GENERATIVE_CORRECTNESS')
        rows.append(row)
        for ei,(pred,target,logit) in enumerate(zip(p,y,z)):
            fi,u,v=map(int,a['edge_records'][ei]); vids=c['C_source_vertex_ids']
            role='OWNER' if target else 'ALIAS' if alias[ei] else 'NULL_UNSUPPORTED'
            edges.append(dict(condition_id=cid,parent_id=q['parent_id'],split=q['split'],edge_id=ei,
                C_face_index=fi,C_vertex_u=u,C_vertex_v=v,source_vertex_u=int(vids[u]),source_vertex_v=int(vids[v]),
                saved_logit=float(logit),saved_threshold=0.0,predicted_active=bool(pred),source_role=role,
                source_owner_face_id=int(c['edge_owner_face'][ei]),source_agreement=bool(pred==target),
                confusion='TP' if pred and target else 'FP' if pred else 'FN' if target else 'TN'))
        for path in (lp,rp,cp):inputs[str(path)]=sha(path)
    summaries=[]
    for split in ('TRAIN','DEV','ALL','DEV_OFFICIAL_TEST','DEV_PRETRAIN_TRAIN'):
        rr=[r for r in rows if split=='ALL' or r['split']==split or
            split=='DEV_OFFICIAL_TEST' and r['split']=='DEV' and r['official_split']=='test' or
            split=='DEV_PRETRAIN_TRAIN' and r['split']=='DEV' and r['official_split']=='train']
        sums={key:sum(r[key] for r in rr) for key in ('boundary_edges','source_owner_count','source_alias_count',
            'source_null_unsupported_count','predicted_active_count','TP','FP','FN','TN','false_positive_alias',
            'false_positive_null_unsupported')}
        summaries.append(dict(split=split,conditions=len(rr),**sums,
            source_owner_recall=sums['TP']/sums['source_owner_count'] if sums['source_owner_count'] else None,
            source_owner_precision=sums['TP']/sums['predicted_active_count'] if sums['predicted_active_count'] else None,
            exact_source_layout_count=sum(r['exact_source_layout_match'] for r in rr),
            all_inactive_conditions=sum(r['all_edges_inactive'] for r in rr)))
    dest=RUN/'exploratory/layout_source_agreement';dest.mkdir(parents=True,exist_ok=True)
    for name,data in (('per_condition.csv',rows),('per_edge.csv',edges),('summary.csv',summaries)):
        csvwrite(dest/name,data)
    write(dest/'all_inactive.json',[r for r in rows if r['all_edges_inactive']])
    for path,digest in inputs.items():assert sha(path)==digest, path
    assert 'torch' not in sys.modules
    result=dict(status='COMPLETE_EXPLORATORY',conditions=len(rows),edges=len(edges),summaries=summaries,
        inputs=inputs,source_sha256=sha(__file__),
        scope='Saved C-only head decisions versus owner/alias/null in one source completion; not unique generative truth. No source label influenced these frozen layout decisions.',
        logits_and_threshold_unchanged=True,formal_evaluation_schema_changed=False,
        head_forwards=0,generator_forwards=0,denoiser_forwards=0,geometry_scans=0,
        all_inactive_conditions=[r['condition_id'] for r in rows if r['all_edges_inactive']],
        files={name:sha(dest/name) for name in ('per_condition.csv','per_edge.csv','summary.csv','all_inactive.json')})
    write(dest/'receipt.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','files')}))


if __name__=='__main__':main()
