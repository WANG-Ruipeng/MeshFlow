"""Extend the registered CPU stream without changing any original OT arithmetic."""
from pathlib import Path
from collections import Counter
from experiments.mf_h1_local0.control_mvp48.train_stream import FrozenStream
from .common import read_json, digest


class BudgetStream(FrozenStream):
    def __init__(self, source, out, arm):
        source, out = Path(source), Path(out)
        binding = read_json(source/'registration/train_data_binding.json')
        super().__init__(binding, source/'data/train_plan.json', source/'data')
        self.probe = arm == 'T1_fit2_probe'
        self.offset, self.end = (0, 1000) if self.probe else (1000, 5000)
        self.plan_path = out/('fit_plan.json' if self.probe else 'train_plan.json')
        plan = read_json(self.plan_path)
        self.records = plan['records']
        self.plan_sha256 = digest(self.plan_path)
        if len(self.records) != (self.end-self.offset)*8:
            raise ValueError('Fixed extension plan budget differs')
        self.batch_index = self.offset
        self.seed = plan.get('seed', 2026100302 if self.probe else 2026100301)
        allowed = set(binding['train_object_ids'])
        pairs = Counter()
        for r in self.records:
            cid = (r['uid'], r['patch_id'])
            if r['uid'] not in allowed or cid not in self.cases:
                raise ValueError('Extension includes an unregistered training condition')
            c = self.cases[cid]
            if (r['N'], r['K']) != (c['N'], c['K']):
                raise ValueError('Extension N/K identity differs')
            for field in ('gaussian_seed', 't_seed', 'permutation_seed'):
                if not isinstance(r[field], int):
                    raise ValueError('Each input must have a frozen integer seed')
            pairs[cid] += 1
        if self.probe:
            expected = {(uid, 'p0') for uid in sorted(allowed)[:2]}
            if set(pairs) != expected:
                raise ValueError('Fit2 must use the two registered UID-first training p0 cases')
            for i in range(0, len(self.records), 8):
                if sorted(Counter(r['uid'] for r in self.records[i:i+8]).values()) != [4, 4]:
                    raise ValueError('Each fit2 batch must have four examples of each object')
        elif len(pairs) != 128 or max(pairs.values())-min(pairs.values()) > 8:
            raise ValueError('Main extension must retain balanced 32 x 4 conditions')

    def state_dict(self):
        return dict(seed=self.seed, batch_index=self.batch_index,
                    sample_index=8*(self.batch_index-self.offset),
                    RNG_mode='stateless_per_sample_frozen_seeds',
                    plan_sha256=self.plan_sha256, offset=self.offset, end=self.end)

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ('seed','RNG_mode','plan_sha256','offset','end'):
            if state.get(key) != expected[key]:
                raise ValueError('Frozen extension stream identity differs: '+key)
        b = state['batch_index']
        if type(b) is not int or not self.offset <= b <= self.end or state['sample_index'] != 8*(b-self.offset):
            raise ValueError('Invalid extension cursor')
        self.batch_index = b

    def next_effective_batch(self):
        if not self.offset <= self.batch_index < self.end:
            raise RuntimeError('Fixed extension plan exhausted')
        i = 8*(self.batch_index-self.offset)
        samples = [self._sample(r) for r in self.records[i:i+8]]
        self.batch_index += 1
        return samples
