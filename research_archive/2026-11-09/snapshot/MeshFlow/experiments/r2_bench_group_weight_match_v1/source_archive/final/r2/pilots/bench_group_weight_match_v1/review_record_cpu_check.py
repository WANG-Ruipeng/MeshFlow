"""Schema-only test; intentionally no image opens or emitted ratings."""
from . import review_record as R


def main():
    protocol=R.read(R.OUT/'visual_review/protocol.json')
    try:R.validate_page({},protocol['axes'],protocol['labels'])
    except ValueError:pass
    else:raise AssertionError('Missing actual-view attestation was accepted')
    a=R.make_assignments()
    assert [x['condition_count'] for x in a['assignments'].values()]==[8,8,7]
    assert [x['pages'] for x in a['assignments'].values()]==[48,48,42]
    assert a['total_observations']==276
    print('PASS: missing-view record rejected; condition ownership and full coverage verified; zero images viewed or ratings written')


if __name__=='__main__':main()
