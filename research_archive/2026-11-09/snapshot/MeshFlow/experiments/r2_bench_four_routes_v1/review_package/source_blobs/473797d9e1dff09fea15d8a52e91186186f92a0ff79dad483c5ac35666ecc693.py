"""Same B parent allocation; explicit independent SEAM stream, no global RNG."""
from r2.pilots.bench_natural_n_recipe_v1 import stream as legacy
ARMS=('SEAM1500',)
DEFAULT_SEED=2026102703
SCHEMA='BENCH_FOUR_ROUTES_ORIGINAL_SEAM_STREAM_V1'

def make_schedule(train,arm='SEAM1500',updates=1500,seed=DEFAULT_SEED):
    if arm not in ARMS or updates!=1500:raise ValueError('Only1500 SEAM updates authorized')
    s=legacy.make_schedule(train,'NATURAL_HIGH',updates,seed)
    s.update(schema=SCHEMA,arm=arm,seed=seed,underlying_stream_schema=legacy.SCHEMA)
    for entry in s['steps']:entry['stage']='SEAM'
    return s

make=make_schedule

def validate(schedule,train,updates=1500,arm='SEAM1500'):
    if schedule!=make_schedule(train,arm,updates,schedule['seed']):raise ValueError('Frozen stream changed')
    expected=legacy.make_schedule(train,'NATURAL_HIGH',updates,schedule['seed'])
    audit=legacy.validate(expected,train,updates,'NATURAL_HIGH')
    return dict(audit,arm=arm,stage='SEAM',same_parent_policy_as_B=True,new_registered_random_stream=True)
