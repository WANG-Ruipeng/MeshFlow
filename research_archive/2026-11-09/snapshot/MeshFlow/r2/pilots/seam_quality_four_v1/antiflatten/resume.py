"""Read-only status/verification after this fixed CPU subexperiment completes."""
from . import run as A
import json


def main():
    path=A.OUT/'FINAL_STATUS.json'
    if not path.exists():
        stages={name:(A.OUT/name).exists() for name in ('source_gate.json','geometry_completion.json','score_completion.json','completion.json','gallery/review_completion.json')}
        print(json.dumps(dict(status='INCOMPLETE_NO_AUTOMATIC_DISPATCH',stages=stages)));return
    state=A.C.read(path)
    for key in ('code_hashes','protected_hashes','output_hashes'):
        for p,h in state[key].items():
            if A.C.sha(p)!=h:raise ValueError('Completed identity changed: '+p)
    print(json.dumps(dict(status='COMPLETE_VERIFIED',POST=16,new_model_calls=0,new_quality_scans=0,
                         report=str(A.OUT/'REPORT.md'))))


if __name__=='__main__':main()
