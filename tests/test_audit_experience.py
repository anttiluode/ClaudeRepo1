import importlib.util
import numpy as np
import pytest
import sieve as S
from gate_a2b import FRESH


def test_fresh_sources_do_not_alias_training():
    canonical = lambda s: 'chelsea' if s == 'cat' else s
    assert not set(map(canonical, FRESH)) & set(map(canonical, S.TRAIN_PHOTOS))


def test_control_audit_measures_final_image():
    assert hasattr(S, 'control_errors'), 'missing final-image matching audit'
    y, x = np.mgrid[:64, :64]
    img = .2 + .4*(x > y) + .2*((x-35)**2+(y-25)**2 < 100)
    err = S.control_errors(img, S.phase_scramble(img, np.random.default_rng(1)))
    assert err['spectrum_relative_l2'] < 1e-12
    err = S.control_errors(img, S.marginal_matched(img, np.random.default_rng(1)))
    assert err['spectrum_relative_l2'] > 1e-4
    assert max(err['band_histogram_relative_l2']) > 1e-3


def test_alias_guard_rejects_overlap():
    assert hasattr(S, 'validate_sources'), 'missing source-identity guard'
    with pytest.raises(ValueError, match='chelsea'):
        S.validate_sources(['chelsea'], ['cat'])
    S.validate_sources(['chelsea'], ['cell'])


def test_experience_protocol_input_and_null_controls():
    assert importlib.util.find_spec('experience_gate') is not None, 'missing experiment'
    import experience_gate as E
    cues, nuisances = E.draw_examples(8, 7)
    for q, n in zip(cues, nuisances):
        a, b = E.scene(q, q, n), E.scene(q, -q, n)
        visible = E.visible_mask()
        assert np.array_equal(a[visible], b[visible])
        # Candidate generation can only see the observed part of an image.
        changed = a.copy()
        changed[~visible] = 999
        assert np.array_equal(E.candidates(a), E.candidates(changed))
    assert np.mean(cues) == 0
    # Scores independent of experience cannot get both opposite targets right.
    pred = np.ones(8, dtype=int)
    assert E.metrics(pred, pred, cues)['paired_correct'] == 0
