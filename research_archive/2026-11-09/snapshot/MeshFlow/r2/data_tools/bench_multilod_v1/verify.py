"""Read-only verification of admitted cases, all 30 source files and sealed artifacts."""
from . import common as c


def main():
    receipt=c.read(c.OUT/'completion.json');m=c.read(c.OUT/'data/manifest.json')
    if c.sha(c.SOURCE)!=receipt['source_manifest_sha256']:raise RuntimeError('Source manifest changed')
    for row in receipt['files']:
        if c.sha(row['path'])!=row['sha256']:raise RuntimeError('Artifact changed: '+row['path'])
    for row in c.read(c.OUT/'source_hashes.json')['rows']:
        if c.sha(row['path'])!=row['sha256'] or c.sha(row['snapshot'])!=row['sha256']:raise RuntimeError('Code changed: '+row['path'])
    count=0
    for p in m['parents']:
        original=next(v for v in p['variants'] if v['kind']=='NATURAL');a=c.arrays(original['case_path'])
        for v in p['variants']:
            if c.sha(v['case_path'])!=v['case_sha256']:raise RuntimeError('Case changed')
            b=c.arrays(v['case_path'])
            if int(b['N'])!=v['N'] or int(b['K'])!=v['K']:raise RuntimeError('N/K mismatch')
            for key in ('C','C_vertices','C_faces_original','C_faces_model','C_source_vertex_ids','C_source_corner_permutations'):
                if a[key].shape!=b[key].shape or a[key].dtype!=b[key].dtype or a[key].tobytes()!=b[key].tobytes():raise RuntimeError('C mismatch')
            count+=1
    result=dict(status='PASS',source_parents=len(m['parents']),cases=count,artifact_files=len(receipt['files']),
        model_calls=0,optimizer_updates=0)
    c.write(c.OUT/'verification_receipt.json',result);print(c.json.dumps(result))


if __name__=='__main__':main()
