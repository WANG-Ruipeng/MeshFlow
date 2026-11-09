"""Metadata-only supplement; no scoring, models, or scientific definitions."""
from r2.pilots.target_cleanup_c200_v1 import common as C
from pathlib import Path
import json

r=C.load_registration()
exposure=C.csvread(C.OUT/'inputs/evaluation_exposure.csv')
assert len(exposure)==6
assert all(x['continued_in_this_pilot']=='False' for x in exposure)
dest=C.OUT/'evaluation/tables'
endpoints=[dict(label='FULL_S6000',role='FROZEN_PRE_CONTINUATION_REFERENCE',global_step=12000,
    local_step_in_this_pilot=0,historical_local_step=6000,new_updates_this_pilot=0)]
for label in C.ARMS:
    endpoints.append(dict(label=label,role='FIXED_200_UPDATE_CONTINUATION_ENDPOINT',global_step=12200,
        local_step_in_this_pilot=200,historical_local_step=None,new_updates_this_pilot=200))
conditions=[dict(x,pilot_exposure='NOT_IN_THE_16_PARENT_CONTINUATION_POOL',
    interpretation='historical_split is inherited EXP67/FULL history, not current continuation membership') for x in exposure]
C.csvwrite(dest/'endpoint_metadata.csv',endpoints)
C.csvwrite(dest/'panel_exposure_metadata.csv',conditions)
C.write(dest/'metadata_corrections.json',dict(schema='target_cleanup_metadata_supplement_v1',
    registration_sha256=C.sha(C.OUT/'registration.json'),code_sha256=C.sha(__file__),
    corrections=['Inherited per-arm tables use LEARNING_CURVE as a historical adapter role string; both new arms are fixed 200-update endpoints.',
        'FULL_S6000 has blank inherited local_step because it is a cache reference; this pilot step0 corresponds to historical FULL step6000/global12000.',
        'TRAIN and DEV are historical panel labels. All six evaluated parents were excluded from the 16-parent continuation pool.'],
    changes_to_metrics=0,changes_to_frozen_code=0,model_calls=0))
print(json.dumps(dict(status='COMPLETE',endpoint_rows=3,condition_rows=6,model_calls=0)))
