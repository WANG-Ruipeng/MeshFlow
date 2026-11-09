"""One-time explicit adaptation of the prior audited CPU report implementation."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'r2/pilots/bench_natural_n_recipe_v1'
DST=ROOT/'r2/pilots/bench_high_joint_coverage_v1'

def change(text, old, new):
    if old not in text:raise ValueError('Missing source anchor: '+old)
    return text.replace(old,new)

def main():
    text=(SRC/'report_results.py').read_text(encoding='utf-8-sig')
    replacements=[
      ("GROUPS=('OLD','FOCUS','DEV','HOLDOUT','HIGH_TRAIN','HIGH_HOLDOUT')","GROUPS=('OLD','FOCUS','DEV','HOLDOUT','HIGH_TRAIN','HIGH_HOLDOUT','NEW_WIDE_TRAIN')"),
      ("CONTRASTS=(('P0','A'),('P0','B'),('A','B'))","CONTRASTS=(('B','W'),)"),
      ('bench_natural_n_recipe_report_v1','bench_high_joint_coverage_report_v1'),
      ('9*len(E.requests())','len(E.LABELS)*len(E.STAGES)*len(E.requests())'),
      ('ncols=3','ncols=len(E.LABELS)'),
      ('P0 prior B; A CONTROL_OLD; B NATURAL_HIGH. Same frozen noise, frame and sections.',
       'B narrow HIGH; W WIDE_HIGH. Same frozen noise, frame and sections.'),
      ('Bench natural N: P0 / CONTROL_OLD / NATURAL_HIGH','Bench natural coverage: B narrow / W wide'),
      ("E.budget()['new_score_rows'],90)","E.budget()['new_score_rows'],E.budget()['reused_score_rows'])"),
      ('# Bench natural-N recipe: complete numerical results','# Bench wider HIGH pool: complete numerical results'),
      ('P0 is frozen exposure B: old30 outputs are reused; high-N outputs are newly generated. A/B each add1500 updates. Contrasts: A-P0, B-P0 and B-A.',
       'B is the completed narrow HIGH reference (38 cached +8 new outputs). W is the one new1500-step wide HIGH arm from the identical P0. Contrast: W-B.'),
      ('OLD/FOCUS/DEV/HOLDOUT/HIGH_TRAIN/HIGH_HOLDOUT remain separate','OLD/FOCUS/DEV/HOLDOUT/HIGH_TRAIN/HIGH_HOLDOUT/NEW_WIDE_TRAIN remain separate'),
      ('| metric | P0 | A | B | unit |','| metric | B narrow | W wide | unit |'),
      ('| --- | ---: | ---: | ---: | --- |','| --- | ---: | ---: | --- |'),
      ("by.get(('P0',old['stage'],old['request_id']))","by.get(('B',old['stage'],old['request_id']))"),
      ('historical_P0_label_drift','historical_B_label_drift'),
      ('Do not copy historic P0 observations','Do not copy historic B observations'),
    ]
    for a,b in replacements:text=change(text,a,b)
    review=(SRC/'review_record.py').read_text(encoding='utf-8-sig')
    review=review.replace('bench_natural_n_recipe_v1','bench_high_joint_coverage_v1').replace('Explicit P0/A/B ratings required','Explicit B/W ratings required')
    review=review.replace('"P0":{<all five axes>,"note":"concrete visible evidence"},\n                 "A":{<all five axes>,"note":"concrete visible evidence"},\n                 "B":{<all five axes>,"note":"concrete visible evidence"}',
                          '"B":{<all five axes>,"note":"concrete visible evidence"},\n                 "W":{<all five axes>,"note":"concrete visible evidence"}')
    for name,value in (('report_results.py',text),('review_record.py',review)):
        path=DST/name
        if path.exists():
            if path.read_text(encoding='utf-8')!=value:raise ValueError('Existing derived source differs '+name)
        else:path.write_text(value,encoding='utf-8')
        compile(value,str(path),'exec')
    print(json.dumps(dict(status='PREPARED_NOT_RUN',files=['report_results.py','review_record.py'],model_calls=0)))

if __name__=='__main__':main()
