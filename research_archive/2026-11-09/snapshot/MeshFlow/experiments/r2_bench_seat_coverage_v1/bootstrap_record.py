from pathlib import Path
from r2.pilots.bench_adaptation_strength_v1.common import write_new, resources

out = Path(__file__).resolve().parent
if not (out / 'resources_initial.json').exists():
    write_new(out / 'resources_initial.json', resources())
if not (out / 'budget.json').exists():
    write_new(out / 'budget.json', dict(updates=1500, batch=8, microbatch=1,
        source_global_step=18000, endpoint_global_step=19500, training_arms=1,
        source='Authorized experiment; recommended finite1500 budget, optional user steering before freeze',
        automatic_seam_continuation=False))
write_new(out / 'engineering/bootstrap_shell_quoting.json', dict(
    event='First inline python resource capture command rejected by shell quoting',
    python_executed=False, model_calls=0, updates=0,
    correction='Use this path-based script; no scientific replay'))
print('Initial resources and finite budget recorded')
