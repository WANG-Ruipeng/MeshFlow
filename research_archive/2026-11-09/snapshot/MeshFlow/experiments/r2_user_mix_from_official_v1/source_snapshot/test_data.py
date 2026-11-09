"""CPU contract tests only: no model, Gaussian training array, OT or GPU job."""
from collections import Counter
import copy
import unittest

import data


def fixture():
    parents = {f'p{i:02d}': {'N': 200} for i in range(25)}
    old, user = {}, {}
    for uid in parents:
        for ratio in (.2, .4):
            for variant in range(2):
                key = f'old_{uid}_{ratio}_{variant}'
                old[key] = dict(uid=uid, N=200, K=int(200*ratio), ratio=ratio)
        for variant in range(3):
            user[f'user_{uid}_{variant}'] = dict(uid=uid, N=200, K=1+variant*10,
                region_type='seat' if variant < 2 else 'back')
    return dict(parents=parents, tasks={data.prior.ARMS[0]: old, data.prior.ARMS[1]: user},
        condition_source_task_keys={'OLD': data.prior.ARMS[0], 'USER': data.prior.ARMS[1]},
        source_sha256='fixture', implementation_sources={})


class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registration = fixture()
        cls.plan = data.make_plan(cls.registration)

    def test_every_parent_source_augmentation_and_old_ratio_balanced(self):
        data.validate_plan(self.plan, self.registration)
        rows = self.plan['records'][data.ARM]
        self.assertEqual(Counter(r['condition_source'] for r in rows), {'USER': 36000, 'OLD': 12000})
        self.assertEqual(rows[0]['global_step'], 1)
        self.assertEqual(rows[-1]['global_step'], 6000)

    def test_schedule_reproducible_and_does_not_consume_global_rng(self):
        before = data.np.random.get_state()
        again = data.make_plan(self.registration)
        after = data.np.random.get_state()
        self.assertEqual(self.plan, again)
        self.assertEqual(before[0], after[0])
        self.assertTrue(data.np.array_equal(before[1], after[1]))
        self.assertEqual(before[2:], after[2:])
        self.assertNotEqual(data.domain_seed('gaussian', 0), data.prior.domain_seed('gaussian', 0))

    def test_loss_coordinate_fraction_not_misreported_as_slot_fraction(self):
        exposure = self.plan['actual_exposures'][data.ARM]
        self.assertEqual(exposure['by_source']['USER']['sample_fraction'], .75)
        self.assertGreater(exposure['by_source']['USER']['free_coordinate_fraction'], .75)
        self.assertEqual(sum(v['free_coordinate_count'] for v in exposure['by_source'].values()), exposure['free_coordinate_count'])

    def test_invalid_resume_clock_or_identity_rejected(self):
        stream = data.Stream.__new__(data.Stream)
        stream.identities = {'fixture': True}
        stream.batch_index = 0
        state = stream.state_dict()
        stream.load_state_dict(state)
        for key, value in (('global_step', 6000), ('batch_index', -1), ('sample_index', 8), ('identities', {})):
            bad = copy.deepcopy(state)
            bad[key] = value
            with self.assertRaises(ValueError):
                stream.load_state_dict(bad)
        valid = dict(state, batch_index=6000, sample_index=48000, global_step=6000, conditional_total_step=6000)
        stream.load_state_dict(valid)
        self.assertEqual(stream.batch_index, 6000)
        with self.assertRaises(RuntimeError):
            stream.next_batch()

    def test_repeated_parent_or_wrong_source_is_rejected(self):
        corrupt = copy.deepcopy(self.plan)
        corrupt['records'][data.ARM][0]['uid'] = corrupt['records'][data.ARM][1]['uid']
        with self.assertRaises(ValueError):
            data.validate_plan(corrupt, self.registration)


if __name__ == '__main__':
    unittest.main()
