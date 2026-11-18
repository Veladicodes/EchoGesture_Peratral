"""Official entry point: predict(video_path) -> dict  (project spec 4.1).

    {"clip_id", "prediction", "top5", "confidence", "latency_ms"}   exactly these keys.

The pipeline is a causal STREAMING one (the same code the live HUD uses), one step per source frame:

    keyframe step (every k-th frame, k=3): MediaPipe landmarks -> linear interpolation emits the
                                           skipped frames into a queue
    every step:                            pop one frame into the 64-frame ring -> features (C,64,53)
                                           -> INT8 ONNX student -> softmax / top-5

`latency_ms` is the wall-clock time of the FINAL streaming step (landmark/interpolation + feature
update + INT8 inference + post-processing; frame decode excluded because a camera hands frames over).
Full-clip batch throughput is logged separately in LAST_STATS: total_clip_ms (decode included) and FPS.

A directory of image frames (the WLASL-100 archive layout, 16 sparse frames/clip) is accepted as a clip:
every frame is then a keyframe (k=1, stateless IMAGE-mode landmarks), exactly like training extraction.

Model session, landmarkers and vocab are module-level lazy singletons (loaded once, not per call).
Everything is local; a missing file raises a clear error and nothing is ever downloaded.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .common import load_config, load_vocab, repo_path
from .features import build_features
from .interpolator import WARMUP_STEPS, KeyframeInterpolator, LandmarkExtractor, RingBuffer

# With no hand visible in any frame the model has nothing to read: report a low-confidence guess.
NO_HANDS_MAX_CONF = 0.2

LOAD_COUNT = {"session": 0, "extractor": 0, "vocab": 0}  # observable by tests: loads happen once
LAST_STATS: dict[str, Any] = {}  # per-clip timing of the most recent predict() call

_STATE: dict[str, Any] = {"session": None, "extractor": None, "image_extractor": None, "vocab": None,
                          "onnx_path": None}


def configure(onnx_path: str | Path | None = None, extractor: Any = None, vocab: list[str] | None = None,
              image_extractor: Any = None) -> None:
    """Override singletons (tests / alternative weights). Clears any loaded state."""
    reset()
    _STATE["onnx_path"] = Path(onnx_path) if onnx_path else None
    _STATE["extractor"] = extractor
    _STATE["image_extractor"] = image_extractor
    _STATE["vocab"] = vocab


def reset() -> None:
    for k in _STATE:
        _STATE[k] = None
    LAST_STATS.clear()


def _vocab() -> list[str]:
    if _STATE["vocab"] is None:
        _STATE["vocab"] = load_vocab()
        LOAD_COUNT["vocab"] += 1
    return _STATE["vocab"]


def _session():
    if _STATE["session"] is None:
        import onnxruntime as ort

        cfg = load_config()
        path = _STATE["onnx_path"] or repo_path(cfg["paths"]["student_onnx"])
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"model not found: {path}. Run `python scripts/train_pipeline.py` or `python -m src.quantize` "
                "(or place student_int8.onnx in weights/); predict() never downloads anything."
            )
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(cfg.get("ort_threads", 1))  # explicit => reproducible latency
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        shape = sess.get_inputs()[0].shape
        dummy = np.zeros([d if isinstance(d, int) else 1 for d in shape], np.float32)
        for _ in range(WARMUP_STEPS):  # first inferences are slow (allocation, kernel init): pay that at load time
            sess.run(None, {"features": dummy})
        _STATE["session"] = sess
        LOAD_COUNT["session"] += 1
    return _STATE["session"]


def _extractor(image_mode: bool = False):
    key = "image_extractor" if image_mode else "extractor"
    if _STATE[key] is None:
        _STATE[key] = LandmarkExtractor(image_mode=image_mode)
        _STATE[key].load()  # includes the 20-step landmarker warm-up
        LOAD_COUNT["extractor"] += 1
    return _STATE[key]


def _postprocess(logits: np.ndarray, vocab: list[str], hands_seen: bool) -> tuple[str, list[str], float]:
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    if z.shape[0] != len(vocab):
        raise ValueError(
            f"model outputs {z.shape[0]} classes but vocab has {len(vocab)} entries; "
            "re-export the model or fix data/vocab.json"
        )
    p = np.exp(z - z.max())
    p /= p.sum()
    order = np.argsort(-z, kind="stable")[:5]  # ties (common in INT8) -> lowest class index
    top5 = [vocab[int(i)] for i in order]
    conf = float(p[int(order[0])])
    if not hands_seen:
        conf = min(conf, NO_HANDS_MAX_CONF)
    return top5[0], top5, conf


def _predict_window(window: np.ndarray, hands_seen: bool) -> tuple[str, list[str], float]:
    """features -> INT8 model -> post-processing for a (T,53,3) window of joints."""
    feats = build_features(window)[None]  # (1, NUM_CHANNELS, 64, 53)
    logits = _session().run(None, {"features": feats})[0][0]
    return _postprocess(logits, _vocab(), hands_seen)


def predict_from_joints(joints: np.ndarray, clip_id: str, t0: float | None = None) -> dict:
    """Everything after landmark extraction on a whole joint array (used by benchmarks/robustness)."""
    t0 = time.perf_counter() if t0 is None else t0
    j = np.asarray(joints)
    hands_seen = bool(len(j) and np.any(j[:, :42] != 0.0))
    pred, top5, conf = _predict_window(j, hands_seen)
    return {"clip_id": clip_id, "prediction": pred, "top5": top5, "confidence": conf,
            "latency_ms": float((time.perf_counter() - t0) * 1000.0)}


class StreamingPredictor:
    """Causal per-frame pipeline shared by predict(), the benchmark and the HUD."""

    def __init__(self, extractor: Any, k: int = 3, step_ms: int = 33, window: int = 64):
        self.ex, self.k, self.step_ms, self.window = extractor, k, step_ms, window
        self.reset()

    def reset(self) -> None:
        self.ring = RingBuffer(self.window)
        self.interp = KeyframeInterpolator(self.k)
        self.pending: deque[np.ndarray] = deque()
        self.idx = 0
        self.hands_seen = False
        self.log: list[tuple[bool, float]] = []  # (is_keyframe, step_ms)

    def step(self, bgr: np.ndarray, infer: bool = True):
        """Process one source frame. Returns ((prediction, top5, conf) | None, step_ms)."""
        t0 = time.perf_counter()
        is_key = self.idx % self.k == 0
        if is_key:
            f = self.ex.detect_frame(bgr, self.step_ms)
            self.hands_seen = self.hands_seen or bool(np.any(f[:42] != 0.0))
            self.pending.extend(self.interp.push(f))
        if self.pending:
            self.ring.append(self.pending.popleft())
        out = _predict_window(self.ring.as_array(), self.hands_seen) if infer else None
        ms = (time.perf_counter() - t0) * 1000.0
        self.log.append((is_key, ms))
        self.idx += 1
        return out, ms


def _iter_frames(path: Path) -> tuple[Iterator[np.ndarray], bool, float]:
    """(frame iterator, is_frame_dir, fps)."""
    import cv2

    if path.is_dir():
        files = sorted(path.glob("*.jpg"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]) if "_" in p.stem else 0)

        def gen_dir():
            for f in files:
                img = cv2.imread(str(f))
                if img is None:
                    raise ValueError(f"cannot read frame: {f}")
                yield img

        return gen_dir(), True, 30.0
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    def gen_video():
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    return
                yield img
        finally:
            cap.release()

    return gen_video(), False, (fps if fps and fps > 1.0 else 30.0)


def predict(video_path: str) -> dict:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")
    _vocab()
    _session()  # singletons are loaded before the clock starts (first-call cost is not per-step latency)
    frames, is_dir, fps = _iter_frames(path)
    k = 1 if is_dir else int(load_config()["frame_skip"])
    ex = _extractor(image_mode=is_dir)
    ex.load()
    sp = StreamingPredictor(ex, k=k, step_ms=int(round(1000.0 / fps)))

    t_clip = time.perf_counter()
    prev, out, last_ms, n = None, None, 0.0, 0
    for frame in frames:
        if prev is None and not is_dir:
            ex.reset_tracking(frame.shape)
        if prev is not None:
            sp.step(prev, infer=False)  # only the final step needs a model run for a one-shot prediction
        prev, n = frame, n + 1
    if prev is not None:
        out, last_ms = sp.step(prev, infer=True)
    else:
        t0 = time.perf_counter()
        out = _predict_window(np.zeros((0, 53, 3), np.float32), False)
        last_ms = (time.perf_counter() - t0) * 1000.0
    total_ms = (time.perf_counter() - t_clip) * 1000.0

    pred, top5, conf = out
    LAST_STATS.clear()
    LAST_STATS.update(
        clip_id=path.stem, n_frames=n, total_clip_ms=total_ms, fps=(n / (total_ms / 1000.0)) if n else 0.0,
        step_ms=[ms for _, ms in sp.log] if sp.log else [last_ms], is_key=[k_ for k_, _ in sp.log],
        window=sp.ring.as_array(),
    )
    return {"clip_id": path.stem, "prediction": pred, "top5": top5, "confidence": float(conf),
            "latency_ms": float(last_ms)}
