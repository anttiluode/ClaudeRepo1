import numpy as np
import pytest
import sieve as S
import relations as R


@pytest.fixture(scope="module")
def img():
    return S.crops([S.photo("camera")], 2, seed=1)[0]


def test_sieve_is_lossless(img):
    b, c = S.sieve(img)
    assert np.abs(S.unsieve(b, c) - img).max() < 1e-12


def test_phase_scramble_keeps_every_second_order_relation(img):
    y = S.phase_scramble(img, np.random.default_rng(0))
    assert np.allclose(np.abs(np.fft.fft2(y)), np.abs(np.fft.fft2(img)))
    assert np.allclose(R.relation_vector(img, "linear"), R.relation_vector(y, "linear"), atol=1e-9)
    assert np.abs(R.relation_vector(img, "rectified") - R.relation_vector(y, "rectified")).max() > .05


def test_block_shuffle_keeps_pixels(img):
    y = S.block_shuffle(img, np.random.default_rng(0))
    assert np.allclose(np.sort(y.ravel()), np.sort(img.ravel())) and not np.allclose(y, img)


def test_hebb_update_is_running_mean(img):
    xs = S.crops([S.photo("coffee")], 6, seed=4)
    L = R.HebbLayer("rectified").fit(xs)
    assert np.allclose(L.W, np.mean([R.relation_vector(x) for x in xs], 0))


def test_auc():
    assert R.auc([2, 3], [0, 1]) == 1. and R.auc([0], [0]) == .5
