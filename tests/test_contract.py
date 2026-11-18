"""predict() contract (CLAUDE.md 4.1) + streaming-latency semantics (CLAUDE.md 3.1).

Uses a random-weight INT8 student exported into tmp_path and a fake landmark extractor, so it runs
with no trained model. A real-MediaPipe integration test runs only if the .task bundles are present.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import models as M
from src import predict as P
from src import quantize as Q
from src.common import load_vocab, repo_path, seed_everything
from src.features import LEFT, NUM_JOINTS, build_features

KEYS = {"clip_id", "prediction", "top5", "confidence", "latency_ms"}
VOCAB = load_vocab()


def _hands_frame(seed=0):
    rng = np.random.default_rng(seed)
    f = np.zeros((NUM_JOINTS, 3), np.float32)
    h = rng.normal(0, 0.03, (21, 3)).astype(np.float32)
    h[0] = 0
    h[9] = [0, -0.08, 0]
    f[LEFT] = 0.5 + h
    f[47], f[48] = [0.4, 0.5, 0], [0.6, 0.5, 0]
    return f


class FakeExtractor:
    """Canned landmarks. detect_frame is called per keyframe (video) or per frame (frame dir)."""

    def __init__(self, hands=True):
        self.hands, self.calls, self.image_mode = hands, 0, False

    def load(self):
        pass

    def reset_tracking(self, shape):
        pass

    def detect_frame(self, bgr, step_ms=33):
        self.calls += 1
        return _hands_frame(self.calls) if self.hands else np.zeros((NUM_JOINTS, 3), np.float32)


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory):
    seed_everything(42)
    d = tmp_path_factory.mktemp("w")
    fused = Q.fuse_student(M.StudentSTGCN(len(VOCAB), 7).eval())
    fp32 = Q.export_onnx_fp32(fused, d / "student_fp32.onnx", 7)
    return Q.quantize_onnx_int8(fp32, d / "student_int8.onnx", Q.synthetic_calibration(32, 1, 7))


def _write_clip(path: Path, n=20, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, size)
    assert w.isOpened()
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), (i * 9) % 255, np.uint8))
    w.release()
    return str(path)


@pytest.fixture
def clip(tmp_path):
    return _write_clip(tmp_path / "test_0412.avi")


@pytest.fixture
def pipe(onnx_path):
    ex = FakeExtractor()
    P.configure(onnx_path=onnx_path, extractor=ex, image_extractor=FakeExtractor())
    yield ex
    P.reset()


def test_exact_schema_and_types(pipe, clip):
    r = P.predict(clip)
    assert set(r) == KEYS
    assert isinstance(r["clip_id"], str) and isinstance(r["prediction"], str)
    assert isinstance(r["top5"], list) and len(r["top5"]) == 5 and all(isinstance(g, str) for g in r["top5"])
    assert type(r["confidence"]) is float and type(r["latency_ms"]) is float
    assert 0.0 <= r["confidence"] <= 1.0 and r["latency_ms"] > 0 and math.isfinite(r["latency_ms"])
    json.dumps(r)


def test_invariants(pipe, clip):
    r = P.predict(clip)
    assert r["prediction"] == r["top5"][0] and len(set(r["top5"])) == 5
    assert all(g == g.upper() and g in VOCAB for g in r["top5"])
    assert r["clip_id"] == "test_0412"


def test_clip_id_is_stem_without_extension(pipe, tmp_path):
    assert P.predict(_write_clip(tmp_path / "a.b_01.avi"))["clip_id"] == "a.b_01"


def test_latency_ms_is_last_streaming_step_and_clip_stats_are_logged(pipe, clip):
    r = P.predict(clip)
    s = P.LAST_STATS
    assert r["latency_ms"] == pytest.approx(s["step_ms"][-1])
    assert s["n_frames"] == 20 and len(s["step_ms"]) == 20
    assert s["total_clip_ms"] > r["latency_ms"] and s["fps"] > 0
    assert s["fps"] == pytest.approx(20 / (s["total_clip_ms"] / 1000.0))


def test_landmarks_run_on_every_third_frame_only(pipe, clip):
    P.predict(clip)
    assert pipe.calls == math.ceil(20 / 3)  # frames 0,3,6,...,18
    kinds = P.LAST_STATS["is_key"]
    assert [i for i, k in enumerate(kinds) if k] == [0, 3, 6, 9, 12, 15, 18]


def test_confidence_is_softmax_of_prediction(pipe, clip, onnx_path):
    import onnxruntime as ort

    r = P.predict(clip)
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    z = s.run(None, {"features": build_features(P.LAST_STATS["window"])[None]})[0][0].astype(np.float64)
    p = np.exp(z - z.max())
    p /= p.sum()
    assert r["prediction"] == VOCAB[int(z.argmax())]
    assert abs(r["confidence"] - float(p.max())) < 1e-5
    assert r["top5"] == [VOCAB[i] for i in np.argsort(-z, kind="stable")[:5]]  # ties -> lowest index


def test_frames_directory_is_accepted_as_a_clip(onnx_path, tmp_path):
    d = tmp_path / "00618"
    d.mkdir()
    for i in range(16):
        cv2.imwrite(str(d / f"accident_{i}.jpg"), np.full((48, 64, 3), i * 10, np.uint8))
    ex = FakeExtractor()
    P.configure(onnx_path=onnx_path, extractor=FakeExtractor(), image_extractor=ex)
    try:
        r = P.predict(str(d))
    finally:
        P.reset()
    assert set(r) == KEYS and r["clip_id"] == "00618"
    assert ex.calls == 16  # every archive frame is a keyframe (k=1), nothing skipped


def test_no_hands_never_raises_and_is_low_confidence(onnx_path, clip):
    P.configure(onnx_path=onnx_path, extractor=FakeExtractor(hands=False))
    try:
        r = P.predict(clip)
    finally:
        P.reset()
    assert set(r) == KEYS and r["prediction"] in VOCAB and r["confidence"] <= P.NO_HANDS_MAX_CONF


def test_unreadable_video_raises_clear_error(pipe, tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="cannot open"):
        P.predict(str(bad))


def test_model_loaded_once_not_per_call(pipe, clip):
    P.predict(clip)
    first = P.LOAD_COUNT["session"]
    for _ in range(3):
        P.predict(clip)
    assert P.LOAD_COUNT["session"] == first


def test_deterministic_across_calls(pipe, clip):
    a = P.predict(clip)
    pipe.calls = 0
    b = P.predict(clip)
    for k in ("clip_id", "prediction", "top5", "confidence"):
        assert a[k] == b[k]


def test_vocab_size_mismatch_is_a_clear_error(onnx_path, clip):
    P.configure(onnx_path=onnx_path, extractor=FakeExtractor(), vocab=VOCAB[:99])
    try:
        with pytest.raises(ValueError, match="vocab"):
            P.predict(clip)
    finally:
        P.reset()


def test_missing_model_raises_without_download(tmp_path, clip):
    P.configure(onnx_path=tmp_path / "nope.onnx", extractor=FakeExtractor())
    try:
        with pytest.raises(FileNotFoundError, match="nope.onnx"):
            P.predict(clip)
    finally:
        P.reset()


def test_missing_video_raises_file_not_found(pipe, tmp_path):
    with pytest.raises(FileNotFoundError):
        P.predict(str(tmp_path / "missing.mp4"))


BUNDLES = [repo_path("weights/hand_landmarker.task"), repo_path("weights/pose_landmarker_lite.task")]


@pytest.mark.skipif(not all(b.is_file() for b in BUNDLES), reason="MediaPipe .task bundles not in weights/")
def test_real_mediapipe_on_blank_video_returns_valid_contract(onnx_path, tmp_path):
    vid = _write_clip(tmp_path / "blank_0001.avi", n=30, size=(160, 120))
    P.configure(onnx_path=onnx_path)  # real LandmarkExtractor from ./weights
    try:
        r, r2 = P.predict(vid), P.predict(vid)
    finally:
        P.reset()
    assert set(r) == KEYS and r["clip_id"] == "blank_0001" and r["prediction"] in VOCAB
    assert r["confidence"] <= P.NO_HANDS_MAX_CONF
    assert (r["prediction"], r["top5"]) == (r2["prediction"], r2["top5"])
