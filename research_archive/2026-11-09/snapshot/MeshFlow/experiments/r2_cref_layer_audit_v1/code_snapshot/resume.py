"""Verify completed stages and report the exact next unfinished stage; never replay automatically."""
import argparse
from . import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.parse_args()
    stages=[('registration.json','prepare'),('science_freeze.json','freeze'),('geometry_completion.json','geometry'),('gallery/manifest.json','render'),('aggregation_completion.json','aggregate'),('independent_audit/audit.json','audit'),('completion.json','seal/report')]
    for filename,stage in stages:
        if not (c.OUT/filename).exists():
            print('NEXT_UNFINISHED_STAGE:',stage,'; no implicit scientific replay.');return
    for path,h in c.read(c.OUT/'protected_hashes.json').items():assert c.sha(path)==h,path
    n=0
    for receipt in sorted((c.OUT/'receipts').glob('*.json')):
        for path,h in c.read(receipt)['files'].items():assert c.sha(path)==h,path
        n+=1
    done=c.read(c.OUT/'completion.json')
    for path,h in done['deliverable_hashes'].items():assert c.sha(path)==h,path
    assert n==50
    print('COMPLETE_VERIFIED; 50 cached-geometry receipts, protected sources and sealed deliverables unchanged. No models, geometry replay, scoring, training or image rerender executed.')
if __name__=='__main__':main()