"""Perturbation helpers shared by the benchmark, the dark-frame cache and training (src/perturb.py)."""
import numpy as np

from src import perturb as PT
from src.features import BODY, LEFT, NUM_JOINTS, RIGHT


def _joints(t=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, (t, NUM_JOINTS, 3)).astype(np.float32)


def test_low_light_is_darker_deterministic_and_keeps_shape():
    img = np.random.default_rng(0).integers(60, 255, (48, 64, 3), dtype=np.uint8)
    a = PT.low_light_frame(img, np.random.default_rng(1))
    b = PT.low_light_frame(img, np.random.default_rng(1))
    assert a.shape == img.shape and a.dtype == np.uint8
    assert a.mean() < img.mean() * 0.6
    assert np.array_equal(a, b)
    assert not np.array_equal(a, PT.low_light_frame(img, np.random.default_rng(2)))


def test_drop_hands_removes_whole_hands_only():
    j = _joints()
    out = PT.drop_hands(j, np.random.default_rng(0), p=1.0)
    assert np.all(out[:, LEFT] == 0) and np.all(out[:, RIGHT] == 0)
    assert np.array_equal(out[:, BODY], j[:, BODY])
    assert np.array_equal(PT.drop_hands(j, np.random.default_rng(0), p=0.0), j)


def test_drop_hands_is_per_frame_whole_hand_and_near_p():
    j = _joints(t=4000)
    out = PT.drop_hands(j, np.random.default_rng(3), p=0.15)
    left_gone = np.all(out[:, LEFT] == 0, axis=(1, 2))
    partial = np.any(out[:, LEFT] == 0, axis=(1, 2)) & ~left_gone
    assert not partial.any(), "a hand is either fully kept or fully dropped in a frame"
    assert abs(left_gone.mean() - 0.15) < 0.03
    assert abs(np.all(out[:, RIGHT] == 0, axis=(1, 2)).mean() - 0.15) < 0.03


def test_occlude_joints_zeroes_all_coords_together():
    j = _joints(t=2000)
    out = PT.occlude_joints(j, np.random.default_rng(4), p=0.15)
    gone = np.all(out == 0, axis=-1)
    assert not (np.any(out == 0, axis=-1) & ~gone).any()
    assert abs(gone.mean() - 0.15) < 0.02
    assert np.array_equal(PT.occlude_joints(j, np.random.default_rng(4), p=0.0), j)


def test_tilt_rotates_about_image_centre_and_keeps_missing_zero():
    j = _joints(t=8)
    j[:, LEFT] = 0
    out = PT.tilt(j, np.random.default_rng(5), deg=15.0)
    assert np.all(out[:, LEFT] == 0)
    r0 = np.hypot(j[..., 0] - 0.5, j[..., 1] - 0.5)
    r1 = np.hypot(out[..., 0] - 0.5, out[..., 1] - 0.5)
    pres = np.any(j != 0, axis=-1)
    assert np.allclose(r0[pres], r1[pres], atol=1e-5)
    assert np.array_equal(out[..., 2], j[..., 2])
    assert np.array_equal(out, PT.tilt(j, np.random.default_rng(5), deg=15.0))


def test_noise_perturbs_present_joints_only_and_is_deterministic():
    j = np.zeros((4, 53, 3), np.float32)
    j[:, 5] = 0.5
    a = PT.noise(j, np.random.default_rng(1))
    b = PT.noise(j, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert np.all(a[:, 0] == 0) and not np.array_equal(a[:, 5], j[:, 5])
