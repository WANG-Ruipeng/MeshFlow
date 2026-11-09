"""Copy the approved V2 rubric as metadata; never copy output ratings."""
from __future__ import annotations
import datetime
import json
from pathlib import Path
from . import evaluation as E


def main():
    source=E.ROOT/'experiments/r2_bench_group_weight_match_v1/visual_review/v2_definition.json'
    target=E.OUT/'visual_review/v2_definition.json'
    if target.exists():
        result=E.read(target)
        assert result['inherited_definition_sha256']==E.sha(source)
        assert result['main_comparison']==['B','S']
        return result
    result=E.read(source)
    result.pop('B',None)
    result.pop('before_new_M_sampling',None)
    result.update(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  before_new_S_sampling=True,main_comparison=['B','S'],
                  labels_description={'B':'Frozen narrow natural-N Native endpoint',
                                      'S':'Original owner-only SEAM1500 continuation'},
                  inherited_definition_path=str(source),inherited_definition_sha256=E.sha(source),
                  anchor_scope='Previously viewed interpretation anchors only; not observations of this experiment',
                  current_experiment_ratings_copied=0)
    E.immutable(target,result)
    return result


if __name__=='__main__':
    result=main();print(json.dumps({k:result[k] for k in ('status','main_comparison','current_experiment_ratings_copied')}))
