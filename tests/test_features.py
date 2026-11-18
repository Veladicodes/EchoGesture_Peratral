"""Contract tests for src/features.py (CLAUDE.md 5.2). Run in both 7- and 6-channel modes."""
import numpy as np
import pytest

from src import features as F
from src.common import get_num_channels, seed_everything

T, V = F.T_FRAMES, F.NUM_JOINTS
CH7 = {"x": 0, "y": 1, "z": 2, "vx": 3, "vy": 4, "cos": 5, "sin": 6}
CH6 = {"x": 0, "y": 1, "vx": 2, "vy": 3, "cos": 4, "sin": 5}


def ch(name: str, nc: int) -> int:
    return (CH7 if nc == 7 else CH6)[name]


def make_joints(t: int, seed: int = 0, left: bool = True, right: bool = True, body: bool = True) -> np.ndarray:
    """Plausible skeleton (T,53,3) in image-normalised coordinates."""
    rng = np.random.default_rng(seed)
    j = np.zeros((t, V, 3), np.float32)
    base = rng.uniform(0.2, 0.8, size=(t, 1, 3)).astype(np.float32)

    def hand(centre):
        pts = rng.normal(0, 0.03, size=(t, 21, 3)).astype(np.float32)
        pts[:, 0] = 0
        pts[:, 9] = [0.0, -0.08, 0.0]  # middle MCP: guaranteed palm_len ~ 0.08
        return centre + pts

    if left:
        j[:, F.LEFT] = hand(base)
    if right:
        j[:, F.RIGHT] = hand(base + 0.1)
    if body:
        b = rng.normal(0, 0.05, size=(t, 11, 3)).astype(np.float32) + 0.5
        b[:, 5] = [0.40, 0.50, 0.0]  # L shoulder (joint 47)
        b[:, 6] = [0.60, 0.50, 0.0]  # R shoulder (joint 48)
        j[:, F.BODY] = b
    return j


@pytest.fixture(params=[6, 7])
def nc(request):
    return request.param


def test_config_default_channels_is_7_or_6():
    assert get_num_channels() in (6, 7)
    assert F.NUM_CHANNELS == get_num_channels()


@pytest.mark.parametrize("t", [1, 2, 17, 64, 200])
def test_shape_dtype_finite(nc, t):
    out = F.build_features(make_joints(t), num_channels=nc)
    assert out.shape == (nc, 64, 53)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_parents_constant_is_valid():
    assert len(F.PARENTS) == V
    assert all(0 <= p < V for p in F.PARENTS)
    assert F.PARENTS[0] == 0 and F.PARENTS[21] == 21  # hand roots point to themselves
    assert F.PARENTS[9] == 0 and F.PARENTS[21 + 9] == 21  # middle MCP hangs off its wrist


def test_all_zero_input_no_nan_and_neutral_bones(nc):
    out = F.build_features(np.zeros((30, V, 3), np.float32), num_channels=nc)
    assert np.isfinite(out).all()
    assert out.shape == (nc, 64, 53)
    # zero-bone guard: cos=1, sin=0 everywhere, everything else 0
    assert np.all(out[ch("cos", nc)] == 1.0)
    assert np.all(out[ch("sin", nc)] == 0.0)
    for name in ("x", "y", "vx", "vy"):
        assert np.all(out[ch(name, nc)] == 0.0)


def test_empty_clip_does_not_raise(nc):
    out = F.build_features(np.zeros((0, V, 3), np.float32), num_channels=nc)
    assert out.shape == (nc, 64, 53) and np.isfinite(out).all()


def test_nan_inf_input_is_sanitised(nc):
    j = make_joints(40)
    j[5, 3, 0] = np.nan
    j[6, 4, 1] = np.inf
    out = F.build_features(j, num_channels=nc)
    assert np.isfinite(out).all()


def test_missing_hands_zero_their_channels(nc):
    out = F.build_features(make_joints(64, left=False, right=False), num_channels=nc)
    hands = out[:, :, :42]
    for name in ("x", "y", "vx", "vy"):
        assert np.all(out[ch(name, nc)][:, :42] == 0.0)
    assert np.all(hands[ch("cos", nc)] == 1.0) and np.all(hands[ch("sin", nc)] == 0.0)
    assert np.abs(out[ch("x", nc)][:, 42:]).sum() > 0  # body still encoded


def test_tiny_palm_zeroes_hand(nc):
    j = make_joints(64)
    j[:, F.RIGHT] = j[:, 21:22, :]  # collapse right hand to one point => palm_len = 0
    out = F.build_features(j, num_channels=nc)
    assert np.isfinite(out).all()
    assert np.all(out[ch("x", nc)][:, 21:42] == 0.0)


def test_velocity_is_zero_when_hand_appears_or_disappears(nc):
    j = make_joints(64)
    j[:20, F.RIGHT] = 0.0   # absent frames 0..19
    j[40:, F.RIGHT] = 0.0   # absent frames 40..63  (present 20..39)
    out = F.build_features(j, num_channels=nc)
    vx, vy = out[ch("vx", nc)], out[ch("vy", nc)]
    for t in (0, 20, 40, 41):  # t=0, first present frame, first absent frame, still absent
        assert np.all(vx[t, 21:42] == 0.0), f"vx spike at t={t}"
        assert np.all(vy[t, 21:42] == 0.0), f"vy spike at t={t}"
    assert np.abs(vx[21:40, 21:42]).sum() > 0  # real motion while present is kept
    assert np.abs(vx).max() < 20  # no appearance spike


def test_resample_does_not_invent_phantom_hand():
    j = make_joints(2)
    j[1, F.RIGHT] = 0.0  # right hand present on frame 0 only
    r = F.resample_time(j, 64)
    assert r.shape == (64, V, 3)
    for t in range(64):
        h = r[t, F.RIGHT]
        assert np.all(h == 0.0) or np.allclose(h, j[0, F.RIGHT])


def test_resample_identity_when_length_matches():
    j = make_joints(64)
    assert np.array_equal(F.resample_time(j, 64), j)


def test_translation_and_scale_invariance(nc):
    j = make_joints(64, seed=3)
    present = np.any(j != 0, axis=-1, keepdims=True)
    shifted = np.where(present, j * 1.7 + np.array([0.3, -0.2, 0.05], np.float32), 0.0).astype(np.float32)
    a = F.build_features(j, num_channels=nc)
    b = F.build_features(shifted, num_channels=nc)
    for name in ("x", "y", "cos", "sin"):
        assert np.allclose(a[ch(name, nc)], b[ch(name, nc)], atol=1e-3), name


def test_six_channel_is_seven_without_z():
    j = make_joints(50, seed=5)
    a = F.build_features(j, num_channels=7)
    b = F.build_features(j, num_channels=6)
    assert np.array_equal(np.delete(a, 2, axis=0), b)


def test_deterministic_no_aug():
    j = make_joints(77, seed=9)
    assert np.array_equal(F.build_features(j), F.build_features(j))


def test_deterministic_with_seeded_aug_and_seed_sensitivity(nc):
    j = make_joints(64, seed=11)
    a = F.build_features(j, num_channels=nc, rng=np.random.default_rng(42))
    b = F.build_features(j, num_channels=nc, rng=np.random.default_rng(42))
    c = F.build_features(j, num_channels=nc, rng=np.random.default_rng(43))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.isfinite(a).all() and a.shape == (nc, 64, 53)


def test_augmentation_keeps_missing_hand_zero(nc):
    j = make_joints(64, left=False)
    for s in range(5):
        out = F.build_features(j, num_channels=nc, rng=np.random.default_rng(s))
        assert np.all(out[ch("x", nc)][:, :21] == 0.0)
        assert np.isfinite(out).all()


def test_standard_path_matches_the_documented_formula(nc):
    j = make_joints(64, seed=21)
    out = F.build_features(j, num_channels=nc)
    wrist, mcp = j[:, 0], j[:, 9]
    palm = np.linalg.norm(mcp - wrist, axis=-1)
    expect_x = (j[:, 5, 0] - wrist[:, 0]) / palm
    assert np.allclose(out[ch("x", nc)][:, 5], expect_x, atol=1e-5)


def test_lost_anchor_zeroes_hand_cleanly(nc):
    """Run-1 normalisation (restored 14 Nov after the centroid fallback cost ~19 pts clean accuracy): a lost
    wrist or middle-MCP zeroes that hand for the frame, with no NaN and no velocity spike."""
    for lost in (0, 9):
        j = make_joints(64, seed=22)
        j[:, lost] = 0.0
        out = F.build_features(j, num_channels=nc)
        assert np.isfinite(out).all()
        assert np.all(out[ch("x", nc)][:, :21] == 0) and np.all(out[ch("vx", nc)][:, :21] == 0)


def test_hand_with_fewer_than_three_visible_joints_is_zeroed(nc):
    j = make_joints(64, seed=24)
    j[:, 2:21] = 0.0  # only wrist + joint 1 visible
    out = F.build_features(j, num_channels=nc)
    assert np.all(out[ch("x", nc)][:, :21] == 0.0) and np.isfinite(out).all()


def test_missing_joints_become_zero_not_garbage(nc):
    j = make_joints(64, seed=25)
    j[:, [3, 7, 15], :] = 0.0
    out = F.build_features(j, num_channels=nc)
    for v in (3, 7, 15):
        for name in ("x", "y", "vx", "vy"):
            assert np.all(out[ch(name, nc)][:, v] == 0.0), (v, name)
        assert np.all(out[ch("cos", nc)][:, v] == 1.0) and np.all(out[ch("sin", nc)][:, v] == 0.0)


def test_raw_hand_and_joint_dropout_augmentations_keep_features_finite(nc):
    j = make_joints(64, seed=26)
    for s in range(20):
        out = F.build_features(j, num_channels=nc, rng=np.random.default_rng(s))
        assert np.isfinite(out).all() and out.shape == (nc, 64, 53)
        assert np.abs(out).max() < 60


def test_seed_everything_is_reproducible():
    seed_everything(42)
    a = np.random.rand(3)
    seed_everything(42)
    b = np.random.rand(3)
    assert np.array_equal(a, b)
