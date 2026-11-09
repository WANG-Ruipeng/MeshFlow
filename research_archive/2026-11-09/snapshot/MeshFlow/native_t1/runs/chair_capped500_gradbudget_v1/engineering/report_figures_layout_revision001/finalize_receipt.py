"""Retain approved one-line figure layout revision and actual visual QA record."""
from pathlib import Path
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]
ROOT = OUT.parents[2]
SOURCE = ROOT / 'experiments/mf_h1_local0/chair_capped500_gradbudget_v1/report_figures.py'
FIGS = OUT / 'report_figures'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

before = HERE / 'before'
initial = read(before / 'manifest.json')
final = read(FIGS / 'manifest.json')
assert initial['sources'] == final['sources'], 'Scientific input identity changed'
assert sha(before / 'displayed_values.csv') == sha(FIGS / 'displayed_values.csv')
assert sha(before / 'primary500_control_and_shape.png') == sha(FIGS / 'primary500_control_and_shape.png')
old = (before / 'report_figures.py').read_text(encoding='utf-8')
new = SOURCE.read_text(encoding='utf-8')
expected = old.replace('            if row == 0:',
    "            ax.set_ylabel(Y_LABELS[metric].replace('  ', '\\n'), fontsize=9)\n            if row == 0:")
assert new == expected, 'Revision exceeds approved one-line label wrapping'
diff = ''.join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
    fromfile='before/report_figures.py', tofile='report_figures.py'))
(HERE / 'source.diff').write_text(diff, encoding='utf-8')
split = read(FIGS / 'qa/split_receipt.json')
qa = dict(status='PASS', reviewer_type='AI_VISUAL_REVIEW', human_review='NOT_RUN',
    scope='Both final numerical PNG figures; all original image pixels inspected.',
    primary=dict(path=str(FIGS / 'primary500_control_and_shape.png'),
        sha256=sha(FIGS / 'primary500_control_and_shape.png'), actually_viewed=True,
        method='Full PNG viewed. Final PNG hash is exactly identical to the viewed first export.'),
    stability=dict(path=str(FIGS / 'stability500_to1000.png'),
        sha256=sha(FIGS / 'stability500_to1000.png'), actually_viewed=True,
        method='Final PNG inspected in three exact-pixel non-resized contiguous vertical pieces; full-image tool payload had exceeded display limit.',
        original_size=split['original_size'], all_source_pixels_included_once=True,
        actually_viewed_splits=split['splits']),
    checks=dict(titles_and_labels='PASS', metric_direction_and_units='PASS',
        OLD_and_FRESH_separate='PASS', eight_parent_legend='PASS',
        both_registered_noises_and_parent_means_present='PASS',
        full_tail_y_limits='PASS', adjacent_vertical_axis_labels_do_not_overlap='PASS',
        no_caption_or_legend_clipping='PASS', saved_numerical_bounds_not_claimed_as_confidence_intervals='PASS'),
    notes=['Values at exactly zero sit on the zero baseline; all data values are retained.',
        'This is figure layout and representation QA, not model-quality approval.',
        'The initial long vertical y-axis labels overlapped between rows. Only stability ylabel line wrapping was added.'],
    final_manifest_sha256=sha(FIGS / 'manifest.json'),
    displayed_values_sha256=sha(FIGS / 'displayed_values.csv'),
    script_sha256=sha(SOURCE), sources=final['sources'],
    model_calls=0, scoring_calls=0, metric_recomputation=False)
write(FIGS / 'qa/visual_qa.json', qa)
receipt = dict(status='PASS', reason='Actual first-export image QA found adjacent long vertical labels overlapping in stability figure.',
    change='One added set_ylabel call wraps the existing stability label text at its double-space separator. No data, ranges, units, means, or gate definitions changed.',
    original_source_sha256=sha(before / 'report_figures.py'), final_source_sha256=sha(SOURCE),
    diff_sha256=sha(HERE / 'source.diff'), original_manifest_sha256=sha(before / 'manifest.json'),
    final_manifest_sha256=sha(FIGS / 'manifest.json'), scientific_sources_identical=True,
    displayed_values_identical=True, primary_png_identical=True,
    initial_exports_retained=str(before), final_qa_path=str(FIGS / 'qa/visual_qa.json'),
    final_qa_sha256=sha(FIGS / 'qa/visual_qa.json'),
    engineering_attempt_note='First text-patch attempt expected CRLF and stopped before editing the source; second used a unique line. This produced no scientific calls.',
    model_calls=0, scoring_calls=0, metric_recomputation=False)
write(HERE / 'receipt.json', receipt)
print(json.dumps({k: receipt[k] for k in ('status','final_source_sha256','final_manifest_sha256','final_qa_sha256')}))
