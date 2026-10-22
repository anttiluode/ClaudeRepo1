import numpy as np
from three_piece_gate import (dataset, predict_relations, relation_features, fit_relation,
                              visible_patches, candidate_evidence, TEMPLATES, evidence,
                              ridge_predictions)


def test_balanced_worlds_share_all_pairs_but_not_triples():
    signs, patches = dataset(2, 42)
    assert patches.shape == (8, 3, 2, 16, 16)
    a, b = signs.T
    for e in (1, -1):
        c = e*a*b
        assert np.mean(c) == 0
        assert np.mean(a*c) == np.mean(b*c) == np.mean(a*b) == 0
        assert np.mean(a*b*c) == e
    # Every factorial group uses identical nuisance for every alternative.
    assert np.array_equal(patches[:4], np.repeat(patches[:1], 4, axis=0))


def test_higher_order_needed_in_balanced_symbolic_world():
    q = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    # Exact candidate evidence; no hidden target passed to the predictor.
    z = np.stack([np.column_stack([q, np.full(4, c)]) for c in (-1, 1)], axis=1)
    truth = np.prod(q, axis=1)
    for degree in (2, 3):
        f = relation_features(z, degree)
        for e in (1, -1):
            w = fit_relation(f, e*truth)
            pred = predict_relations(f, w)
            if degree == 2:
                np.testing.assert_allclose(w, 0, atol=1e-14)
                assert np.mean(pred == e*truth) == .5
            else:
                np.testing.assert_array_equal(pred, e*truth)


def test_hidden_pixels_cannot_change_candidate_evidence():
    q, patches = dataset(1, 7)
    before = candidate_evidence(visible_patches(q, patches))
    patches[:, 2] = 999
    after = candidate_evidence(visible_patches(q, patches))
    np.testing.assert_array_equal(before, after)
    np.testing.assert_allclose([evidence(p) for p in TEMPLATES], [-1, 1])


def test_ridge_control_distinguishes_linear_and_product_inputs():
    q = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    z = np.stack([np.column_stack([q, np.full(4, c)]) for c in (-1, 1)], axis=1)
    for e in (1, -1):
        target = e*np.prod(q, axis=1)
        assert np.mean(ridge_predictions(z, z, target) == target) == .5
        np.testing.assert_array_equal(ridge_predictions(z, z, target, True), target)
