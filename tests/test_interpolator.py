"""Tests for src/interpolator.py (CLAUDE.md 5.1): k=3 interpolation, zero-hand guard, handedness, streaming.

MediaPipe itself is replaced by duck-typed fakes: the .task bundles are not in weights/ yet, so the
real-detector path is covered separately (test_missing_bundle_raises, and scripts/ once bundles exist).
"""
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from src import interpolator as I
from src.features import BODY, LEFT, NUM_JOINTS, RIGHT


def frame(seed=0, left=True, right=True, body=True):
    rng = np.random.default_rng(seed)
    p = np.zeros((NUM_JOINTS, 3), np.float32)
    if left:
        p[LEFT] = rng.uniform(0.1, 0.9, (21, 3))
    if right:
        p[RIGHT] = rng.uniform(0.1, 0.9, (21, 3))
    if body:
        p[BODY] = rng.uniform(0.1, 0.9, (11, 3))
    return p


def test_formula_k3_matches_spec():
    p0, p3 = frame(1), frame(2)
    mid = I.interpolate_between(p0, p3, k=3)
    assert len(mid) == 2
    assert np.allclose(mid[0], p0 + (1 / 3) * (p3 - p0), atol=1e-6)
    assert np.allclose(mid[1], p0 + (2 / 3) * (p3 - p0), atol=1e-6)


@pytest.mark.parametrize("zero_in", ["start", "end"])
def test_zero_hand_guard_no_phantom_limb(zero_in):
    a, b = frame(1), frame(2)
    if zero_in == "start":
        a[RIGHT] = 0.0
    else:
        b[RIGHT] = 0.0
    for m in I.interpolate_between(a, b, k=3):
        assert np.all(m[RIGHT] == 0.0), "right hand must stay zeroed"
        assert np.any(m[LEFT] != 0.0), "left hand (present in both) is still interpolated"
        assert np.any(m[BODY] != 0.0)


def test_both_zero_stays_zero():
    a, b = frame(1, left=False), frame(2, left=False)
    for m in I.interpolate_between(a, b, k=3):
        assert np.all(m[LEFT] == 0.0)


def test_streaming_interpolator_matches_batch_formula():
    kf = [frame(i) for i in range(3)]
    it = I.KeyframeInterpolator(k=3)
    out = []
    for p in kf:
        out.extend(it.push(p))
    assert len(out) == 1 + 3 + 3  # P0, then (t+1,t+2,t+3) per new keyframe
    assert np.array_equal(out[0], kf[0])
    assert np.array_equal(out[3], kf[1]) and np.array_equal(out[6], kf[2])
    assert np.allclose(out[1], I.interpolate_between(kf[0], kf[1], 3)[0])
    assert np.allclose(out[5], I.interpolate_between(kf[1], kf[2], 3)[1])


def test_hold_tail_repeats_last_keyframe():
    it = I.KeyframeInterpolator(k=3)
    last = frame(5)
    it.push(frame(4))
    it.push(last)
    held = it.hold(2)
    assert len(held) == 2 and all(np.array_equal(h, last) for h in held)


def test_ring_buffer_keeps_last_64():
    rb = I.RingBuffer(64)
    for i in range(100):
        rb.append(np.full((NUM_JOINTS, 3), i, np.float32))
    arr = rb.as_array()
    assert arr.shape == (64, NUM_JOINTS, 3)
    assert arr[0, 0, 0] == 36 and arr[-1, 0, 0] == 99


# ---------------------------- landmark -> joint assembly ---------------------------- #
def lm(v):
    return NS(x=v, y=v + 0.001, z=v + 0.002)


def hand_obj(base, label, score=0.9):
    return [lm(base + i * 0.001) for i in range(21)], [NS(category_name=label, score=score)]


def hand_result(*hands):
    return NS(hand_landmarks=[h[0] for h in hands], handedness=[h[1] for h in hands])


def pose_result():
    return NS(pose_landmarks=[[lm(i / 100.0) for i in range(33)]])


def test_handedness_convention_is_documented_and_mapped():
    assert I.MIRRORED_INPUT is True  # verified empirically on real clips (scripts/verify_handedness.py)
    hr = hand_result(hand_obj(0.30, "Left"))
    f = I.assemble_frame(hr, pose_result())
    assert np.all(f[RIGHT] == 0.0)
    assert np.isclose(f[0, 0], 0.30) and np.isclose(f[20, 0], 0.30 + 20 * 0.001)


def test_handedness_swaps_when_not_mirrored(monkeypatch):
    monkeypatch.setattr(I, "MIRRORED_INPUT", False)
    f = I.assemble_frame(hand_result(hand_obj(0.30, "Left")), pose_result())
    assert np.isclose(f[21, 0], 0.30) and np.all(f[LEFT] == 0.0)


def test_pose_joint_order_matches_spec():
    f = I.assemble_frame(hand_result(), pose_result())
    expected = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]
    for k, mp_idx in enumerate(expected):
        assert np.isclose(f[42 + k, 0], mp_idx / 100.0), (k, mp_idx)
        assert np.isclose(f[42 + k, 2], mp_idx / 100.0 + 0.002)


def test_duplicate_label_keeps_higher_score():
    hr = hand_result(hand_obj(0.20, "Left", 0.6), hand_obj(0.70, "Left", 0.95))
    f = I.assemble_frame(hr, pose_result())
    assert np.isclose(f[0, 0], 0.70)


def test_no_detections_gives_all_zero_frame():
    f = I.assemble_frame(NS(hand_landmarks=[], handedness=[]), NS(pose_landmarks=[]))
    assert f.shape == (NUM_JOINTS, 3) and np.all(f == 0.0)


# ------------------------------ extractor / offline ------------------------------ #
def test_missing_bundle_raises_clear_error_without_downloading(tmp_path):
    ex = I.LandmarkExtractor(weights_dir=tmp_path)
    with pytest.raises(FileNotFoundError) as e:
        ex.load()
    assert "hand_landmarker.task" in str(e.value) and str(tmp_path) in str(e.value)


class FakeLandmarker:
    def __init__(self, fn):
        self.fn, self.calls, self.ts = fn, 0, []

    def detect_for_video(self, image, ts):
        self.calls += 1
        self.ts.append(ts)
        return self.fn(self.calls)


def _write_video(path, n=20, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, size)
    assert w.isOpened()
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), (i * 10) % 255, np.uint8))
    w.release()


def test_extract_runs_landmarks_every_3rd_frame_and_returns_all_frames(tmp_path):
    vid = tmp_path / "clip.avi"
    _write_video(vid, n=20)
    hand = FakeLandmarker(lambda n: hand_result(hand_obj(0.1 + 0.01 * n, "Right")))
    pose = FakeLandmarker(lambda n: pose_result())
    ex = I.LandmarkExtractor(hand_landmarker=hand, pose_landmarker=pose, frame_skip=3)
    joints = ex.extract(str(vid))
    assert joints.shape == (20, NUM_JOINTS, 3)
    assert hand.calls == pose.calls == 7  # frames 0,3,6,9,12,15,18
    assert all(b > a for a, b in zip(hand.ts, hand.ts[1:])), "timestamps must increase"
    # frames 19 has no future keyframe -> held at the last keyframe (frame 18)
    assert np.array_equal(joints[19], joints[18])
    # frame 1 lies 1/3 of the way between keyframes 0 and 3 (Right label -> LEFT slot on un-mirrored input)
    assert np.allclose(joints[1], joints[0] + (joints[3] - joints[0]) / 3, atol=1e-5)


def test_extract_on_video_without_hands_never_raises(tmp_path):
    vid = tmp_path / "blank.avi"
    _write_video(vid, n=9)
    ex = I.LandmarkExtractor(
        hand_landmarker=FakeLandmarker(lambda n: NS(hand_landmarks=[], handedness=[])),
        pose_landmarker=FakeLandmarker(lambda n: NS(pose_landmarks=[])),
    )
    j = ex.extract(str(vid))
    assert j.shape == (9, NUM_JOINTS, 3) and np.all(j == 0.0)


def test_extract_bad_path_raises_clear_error(tmp_path):
    ex = I.LandmarkExtractor(
        hand_landmarker=FakeLandmarker(lambda n: None), pose_landmarker=FakeLandmarker(lambda n: None)
    )
    with pytest.raises(FileNotFoundError):
        ex.extract(str(tmp_path / "nope.mp4"))


def test_preprocess_resizes_big_frames_and_clahe_only_when_dark(monkeypatch):
    from src import interpolator as I
    from src.interpolator import preprocess_frame

    monkeypatch.setattr(I, "MP_MAX_SIDE", 224)   # opt-in steps, off by default (measured to hurt accuracy)
    monkeypatch.setattr(I, "CLAHE_LUMA", 85.0)

    bright = np.full((480, 640, 3), 160, np.uint8)
    out = preprocess_frame(bright)
    assert max(out.shape[:2]) == 224 and out.shape[:2] == (168, 224)
    assert np.array_equal(out, np.full_like(out, 160))        # bright: resize only, no contrast change
    small = np.full((200, 200, 3), 160, np.uint8)
    assert preprocess_frame(small).shape == small.shape        # never upscaled
    rng = np.random.default_rng(0)
    dark = (rng.random((240, 320, 3)) * 50).astype(np.uint8)    # mean luma ~25 < 85
    enh = preprocess_frame(dark)
    assert enh.mean() > dark.mean() and enh.dtype == np.uint8   # CLAHE lifted it


def test_preprocess_is_a_noop_by_default(monkeypatch):
    from src import interpolator as I

    monkeypatch.setattr(I, "MP_MAX_SIDE", 0)
    monkeypatch.setattr(I, "CLAHE_LUMA", 0.0)
    f = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    assert np.array_equal(I.preprocess_frame(f), f)
