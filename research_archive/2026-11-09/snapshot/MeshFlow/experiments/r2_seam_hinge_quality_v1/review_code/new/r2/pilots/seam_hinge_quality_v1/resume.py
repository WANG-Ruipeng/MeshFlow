"""Explicit recoverable stage runner. Completed experiment is status-only."""
import argparse,subprocess,sys,json
from .protocol import ROOT,OUT,ARMS
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',nargs='?',default='status')
    a,extra=p.parse_known_args()
    if a.stage=='status':
        rows={}
        for f in [OUT/'protocol.json',OUT/'completion.json',OUT/'training_registration.json',
            *[OUT/'training'/arm/'status.json' for arm in ARMS],
            *[OUT/'evaluation'/arm/(s+'_complete.json') for arm in ARMS for s in ('raw','post')]]:
            v=json.loads(f.read_text()) if f.exists() else {'status':'NOT_COMPLETE'}
            rows[str(f.relative_to(OUT))]={k:v[k] for k in ('status','local_step','global_step','outputs') if k in v}
        print(json.dumps(rows,indent=2));return
    allowed=('protocol','training_prepare','training','training_summary','dispatch','generation',
             'scoring','binding','aggregation','gallery','section_gallery','case_sections',
             'hinge','hinge_collect','topology','audit','evaluation_audit','review_index')
    if a.stage not in allowed:raise SystemExit('Unknown explicitstage')
    if (OUT/'completion.json').exists():raise SystemExit('Taskclosed; noimplicitrepeat')
    raise SystemExit(subprocess.call([sys.executable,'-B','-m','r2.pilots.seam_hinge_quality_v1.'+a.stage,*extra],cwd=ROOT))
if __name__=='__main__':main()
