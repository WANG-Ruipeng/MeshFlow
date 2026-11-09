"""Presentation-only evidence preface; preserve automatic report and exact diff."""
from pathlib import Path
import difflib
import hashlib
import json
import re

OUT = Path(__file__).resolve().parents[1]
REPORT = OUT / 'CHAIR_MOMENT_CHANNEL_V2_REPORT.md'
DEST = OUT / 'presentation/editorial_review'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert (OUT / 'RUN_COMPLETE.json').is_file()
    assert not DEST.exists(), 'Preserve prior editorial work; no silent replay.'
    scientific = [p for p in (OUT / 'paired_results').iterdir() if p.suffix in ('.json','.csv')]
    before = {p.name:digest(p) for p in scientific}
    original = REPORT.read_text(encoding='utf-8')
    header, body = original.split('\n',1)
    assert header == '# Chair Moment Channel Learning V2'
    summary_path = OUT / 'presentation/EXECUTIVE_REVIEW.md'
    summary = summary_path.read_text(encoding='utf-8').replace('\u4eba\u5de5\u8bc1\u636e\u590d\u6838', '\u72ec\u7acb\u8bc1\u636e\u590d\u6838')
    summary_path.write_text(summary,encoding='utf-8')
    # Resolve presentation-relative links at the report root.
    summary = summary.replace('(../DELIVERY_INDEX.md)', '(DELIVERY_INDEX.md)')
    summary = summary.replace('(../independent_audit/', '(independent_audit/')
    summary = summary.replace('(figures/', '(presentation/figures/')
    # Preserve the machine-generated rule recommendation in its archived report.
    # The reviewed priority in the preface is the final recommendation.
    body = re.sub(r'^7\. .*$', '7. \u6700\u7ec8\u4f18\u5148\u7ea7\u4ee5\u672c\u62a5\u544a\u9996\u90e8\u201c\u552f\u4e00\u4f18\u5148\u4e0b\u4e00\u6b65\u201d\u4e3a\u51c6\uff1a\u6682\u7559\u7b80\u5355 HYBRID/START\uff0c\u4e0d\u6269\u5927 MOMENT \u8bad\u7ec3\uff1b\u5efa\u8bae\u540e\u7eed\u4ec5\u7528\u5df2\u6709 raw \u5b9a\u4f4d\u4e24\u81c2\u5171\u540c\u9000\u5316\u7684\u5c40\u90e8\u51e0\u4f55\u539f\u56e0\u3002\u672c\u6b21\u4e0d\u81ea\u52a8\u6267\u884c\u540e\u7eed\u4efb\u52a1\u3002', body, count=1, flags=re.M)
    final = header+'\n\n'+summary+'\n\n---\n'+body
    DEST.mkdir(parents=True)
    (DEST/'automatic_report_after_timing_fix.md').write_bytes(REPORT.read_bytes())
    REPORT.write_text(final,encoding='utf-8')
    (DEST/'code.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),final.splitlines(True),fromfile='automatic_report_after_timing_fix.md',tofile=REPORT.name)),encoding='utf-8')
    after = {p.name:digest(p) for p in scientific}
    assert before==after
    receipt={'status':'PASS','scientific_tables_unchanged':True,'scientific_table_count':len(before),'scientific_table_hashes':after,'model_calls':0,'scoring_calls':0,'original_report_sha256':digest(DEST/'automatic_report_after_timing_fix.md'),'report_sha256':digest(REPORT),'preface_sha256':digest(summary_path),'code_sha256':digest(Path(__file__)),'scope':'Final evidence interpretation and links only; frozen package, scientific tables and raw untouched.'}
    (DEST/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('REPORT_EDITORIAL_REVIEW PASS')

if __name__=='__main__':
    main()
