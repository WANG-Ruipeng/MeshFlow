"""P/raw-normal/local-factor interventions; 256 states evaluated in two domains."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import argparse,itertools,numpy as np
from common import ROOT,dump,sha
from single_state_common import arrays,geometry,target_reference

COMBINATIONS=[''.join(x) for x in itertools.product('PO',repeat=3)]
def main(nfe):
    data=arrays(ROOT/'data.npz');target=data['states'][0];sample=arrays(ROOT/f'sampling_{nfe}'/'endpoints.npz')
    reference=target_reference();out=ROOT/f'interventions_{nfe}';out.mkdir(exist_ok=False);rows=[]
    combinations=COMBINATIONS if nfe==64 else ['PPP']
    for i,(pred,perm) in enumerate(zip(sample['states'],sample['target_permutations'])):
        matching_available=np.all(perm>=0) and len(np.unique(perm))==256
        aligned=target[perm] if matching_available else None
        for combination in combinations:
            if matching_available:
                state=np.concatenate([pred[:,a:b] if c=='P' else aligned[:,a:b] for c,(a,b) in zip(combination,[(0,3),(3,6),(6,22)])],axis=1)
            else:state=np.full((256,22),np.nan)
            for domain in ['fixed','enclosing']:
                metadata=dict(sample_id=i,combination=combination,nfe=nfe,kind='free_generation' if combination=='PPP' else 'oracle_field_intervention',
                    generated=combination=='PPP',oracle_assisted=combination!='PPP',
                    matching_available=bool(matching_available),field_definition='extractor P=v / raw n / centered F; world Q decoded at selected P',
                    target_permutation=perm.tolist(),checkpoint_sha256=sha(ROOT/'training/direct/final.pt'))
                row=geometry(state,domain,out/f'{i:03d}_{combination}_{domain}',reference,406000+i,metadata,perm if matching_available else None)
                rows.append(row)
            print('MIX',nfe,i,combination,[r['valid'] for r in rows[-2:]],flush=True)
        dump(out/'LEDGER_PARTIAL.json',rows)
    dump(out/'LEDGER.json',rows)
    summary=dict(nfe=nfe,physical_configuration_count=32*len(combinations),extraction_count=len(rows),
        by_domain={d:{c:dict(valid=sum(r['valid'] for r in rows if r['domain']==d and r['combination']==c),denominator=32,
            evaluated=sum(r['status']=='EXTRACTED' for r in rows if r['domain']==d and r['combination']==c)) for c in combinations} for d in ['fixed','enclosing']},
        interpretation='Only PPP is free generation. OOO is repeated identity under permutations, not 32 independent objects. Domains are never merged by best result.')
    dump(out/'SUMMARY.json',summary)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--nfe',type=int,default=64);main(ap.parse_args().nfe)