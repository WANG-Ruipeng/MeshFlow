"""Record actual independent report reviews and final resource state."""
from __future__ import annotations
import re
from .common import *

def run():
    report=OUT/'REPORT.md'
    text=report.read_text(encoding='utf-8')
    assert '\ufffd' not in text
    links=re.findall(r'\]\(([^)]+)\)',text)
    missing=[p for p in links if not p.startswith(('http:','https:','#')) and not (OUT/p).exists()]
    assert not missing,missing
    review=dict(status='PASS',report_sha256=sha(report),local_markdown_links_checked=len(links),
        independent_reviews=[
            dict(reviewer='/root/backend_scope_audit',result='PASS',
                 scope='Read-only comparison of report numbers, denominator, 4 parent alerts, visual transitions and 30 links against completed evidence.'),
            dict(reviewer='/root/backend_runtime',result='PASS',
                 scope='Read-only comparison of report against exact training budget, actual 180-cell visual review and complete-state no-replay runtime.')],
        accepted_wording_fixes=['Explicit denoiser units for relative edge formula and numeric thresholds.',
                              '433.42 seconds is recorded update cumulative time including synchronization and ledger IO, not pure GPU time; training phase530.28 and runtime total577.77 are distinct.'],
        root_actual_additional_spotcheck=['27eb/n1 six columns: candidate still lacks healthy end supports; not repaired by higher retained face count.',
                                         '4cd/n0 six columns: candidate and SHAPE macro structure visually similar.'],
        scientific_recomputations=0,model_calls=0)
    write(OUT/'audits/report_review.json',review)
    r=resource_snapshot('final_closeout')
    print(json.dumps(dict(status='PASS',report_sha256=review['report_sha256'],gpu=r['gpu']['stdout'],local_links=len(links))))

if __name__=='__main__':run()
