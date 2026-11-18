"""Air-gapped MediaPipe Tasks landmark extractor with k-frame skipping + linear interpolation (CLAUDE.md 5.1).

* HandLandmarker + PoseLandmarker, RunningMode.VIDEO, loaded ONLY from local ./weights/*.task bundles.
  A missing bundle raises FileNotFoundError; there is never a download fallback.
* Landmark inference runs on frames 0, k, 2k, ...; the skipped frames are linearly interpolated:
      P(t+j) = P(t) + j/k * (P(t+k) - P(t))            (k=3: 1/3 and 2/3)
* Zero-hand guard: a hand (or the body) that is all-zero in P(t) OR P(t+k) is NOT interpolated;
  it stays zeroed across the gap (no phantom limbs).
* Joint layout (53): 0-20 left hand, 21-41 right hand, 42-52 upper body (see features.py).

HANDEDNESS DECISION (documented once, here):
  MediaPipe's hand label assumes a mirrored (selfie) image. We first ASSUMED our un-mirrored
  exocentric frames needed the labels swapped; scripts/verify_handedness.py disproved that on
  120 real WLASL-100 val clips: when exactly one hand is visible, MediaPipe "Right" sits next to the
  pose model's right wrist (idx 16) on 621/630 frames (98.6%). So label and pose side already agree
  and MIRRORED_INPUT = True (label "Left" -> slots 0-20, "Right" -> slots 21-41). Keep it consistent
  between training and inference; re-run the script if the camera setup changes.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .common import load_config, repo_path
from .features import BODY, GROUPS, LEFT, NUM_JOINTS, RIGHT

MIRRORED_INPUT: bool = True  # VERIFIED on real clips (scripts/verify_handedness.py): label == pose side

# MediaPipe Pose index for joints 42..52: nose, L/R eye, L/R ear, L/R shoulder, L/R elbow, L/R wrist.
POSE_INDICES = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16)
HAND_TASK_NAME = "hand_landmarker.task"
POSE_TASK_NAME = "pose_landmarker_lite.task"


# --------------------------------------------------------------------------- #
# interpolation (pure numpy)
# --------------------------------------------------------------------------- #
def _present(frame: np.ndarray, group: slice) -> bool:
    return bool(np.any(frame[group] != 0.0))


def interpolate_between(p0: np.ndarray, p1: np.ndarray, k: int = 3) -> list[np.ndarray]:
    """The k-1 frames strictly between keyframes p0=P(t) and p1=P(t+k), each (53,3)."""
    out = [np.zeros_like(p0, dtype=np.float32) for _ in range(k - 1)]
    for g in GROUPS:
        if _present(p0, g) and _present(p1, g):  # zero-hand guard
            for j in range(1, k):
                out[j - 1][g] = p0[g] + (j / k) * (p1[g] - p0[g])
    return out


class KeyframeInterpolator:
    """Streaming form: push() each new keyframe, get back the frames it completes.

    First push -> [P0]. Every later push -> k-1 interpolated frames followed by the keyframe itself,
    so the output has exactly one frame per source frame.
    """

    def __init__(self, k: int = 3):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self._last: np.ndarray | None = None

    def push(self, keyframe: np.ndarray) -> list[np.ndarray]:
        kf = np.asarray(keyframe, dtype=np.float32)
        if self._last is None:
            self._last = kf
            return [kf]
        out = interpolate_between(self._last, kf, self.k) + [kf]
        self._last = kf
        return out

    def hold(self, n: int) -> list[np.ndarray]:
        """n copies of the last keyframe (trailing frames that have no future keyframe)."""
        if self._last is None or n <= 0:
            return []
        return [self._last.copy() for _ in range(n)]


class RingBuffer:
    """Fixed-size rolling window of (53,3) frames (default 64) for streaming/HUD use."""

    def __init__(self, size: int = 64):
        self.size = size
        self._d: deque[np.ndarray] = deque(maxlen=size)

    def append(self, frame: np.ndarray) -> None:
        self._d.append(np.asarray(frame, dtype=np.float32))

    def __len__(self) -> int:
        return len(self._d)

    def as_array(self) -> np.ndarray:
        if not self._d:
            return np.zeros((0, NUM_JOINTS, 3), np.float32)
        return np.stack(self._d, axis=0)


# --------------------------------------------------------------------------- #
# MediaPipe result -> (53,3)
# --------------------------------------------------------------------------- #
def _slot_for_label(label: str) -> slice:
    left_label = label.lower() == "left"
    anatomical_left = left_label if MIRRORED_INPUT else not left_label
    return LEFT if anatomical_left else RIGHT


def _hand_xyz(landmarks: Any) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)


def assemble_frame(hand_result: Any, pose_result: Any) -> np.ndarray:
    """Build one (53,3) frame from MediaPipe HandLandmarkerResult / PoseLandmarkerResult (duck-typed)."""
    frame = np.zeros((NUM_JOINTS, 3), np.float32)

    best: dict[tuple[int, int], float] = {}
    hands = getattr(hand_result, "hand_landmarks", None) or []
    handed = getattr(hand_result, "handedness", None) or []
    for lms, cats in zip(hands, handed):
        if not cats or len(lms) != 21:
            continue
        cat = cats[0]
        slot = _slot_for_label(cat.category_name)
        key = (slot.start, slot.stop)
        score = float(getattr(cat, "score", 0.0))
        if score >= best.get(key, -1.0):  # duplicate label: keep the more confident hand
            best[key] = score
            frame[slot] = _hand_xyz(lms)

    poses = getattr(pose_result, "pose_landmarks", None) or []
    if poses and len(poses[0]) >= 17:
        pl = poses[0]
        frame[BODY] = np.array([[pl[i].x, pl[i].y, pl[i].z] for i in POSE_INDICES], dtype=np.float32)
    return frame


# --------------------------------------------------------------------------- #
# extractor
# --------------------------------------------------------------------------- #
# Both are config.yaml tunables (mp_max_side, clahe_luma); 0 switches a step off. Defaults are decided by measurement
# (scripts/ablate_preproc.py), not by assumption.
MP_MAX_SIDE = int(load_config().get("mp_max_side", 224))      # frames larger than this are shrunk (aspect kept)
CLAHE_LUMA = float(load_config().get("clahe_luma", 85.0))     # CLAHE only below this mean luma (0 ms in normal light)
WARMUP_STEPS = 20


def preprocess_frame(bgr: np.ndarray) -> np.ndarray:
    """Downscale to <= 224 px (long side) and, only when dark (mean luma < 85), CLAHE on the LAB L channel.

    The detector still runs on every frame (hands-down transitions need it); this only changes its input."""
    import cv2

    h, w = bgr.shape[:2]
    m = max(h, w)
    if MP_MAX_SIDE and m > MP_MAX_SIDE:
        sc = MP_MAX_SIDE / m
        bgr = cv2.resize(bgr, (max(1, round(w * sc)), max(1, round(h * sc))), interpolation=cv2.INTER_AREA)
    if CLAHE_LUMA and float(bgr[..., 0].mean() * 0.114 + bgr[..., 1].mean() * 0.587 + bgr[..., 2].mean() * 0.299) < CLAHE_LUMA:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[..., 0])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


class LandmarkExtractor:
    """Loads the two .task bundles once, then streams a video through them with frame skipping.

    Landmarkers can be injected (tests / reuse). Frames are streamed with OpenCV; the video is
    never loaded into memory (only the (T,53,3) landmark array is kept).
    """

    def __init__(
        self,
        weights_dir: str | Path | None = None,
        frame_skip: int | None = None,
        hand_landmarker: Any = None,
        pose_landmarker: Any = None,
        image_mode: bool = False,
    ):
        cfg = load_config()
        self.image_mode = image_mode  # stateless RunningMode.IMAGE, for sparse frame sets (no tracking)
        self.weights_dir = Path(weights_dir) if weights_dir else repo_path(cfg["paths"]["weights"])
        self.frame_skip = int(frame_skip if frame_skip is not None else cfg["frame_skip"])
        self._hand = hand_landmarker
        self._pose = pose_landmarker
        self._ts_ms = 0          # monotonic across clips: VIDEO mode requires increasing timestamps
        self._used = False

    # ---- loading ----
    def load(self) -> None:
        if self._hand is not None and self._pose is not None:
            return
        hand_path = self.weights_dir / HAND_TASK_NAME
        pose_path = self.weights_dir / POSE_TASK_NAME
        for p in (hand_path, pose_path):
            if not p.is_file():
                raise FileNotFoundError(
                    f"MediaPipe bundle missing: {p}. Place {HAND_TASK_NAME} and {POSE_TASK_NAME} in "
                    f"'{self.weights_dir}' (this code never downloads; scripts/fetch_mediapipe_models.py "
                    "is the dev-time helper)."
                )
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        # delegate=CPU is the default; models are read from explicit local paths only.
        base = lambda p: mp_python.BaseOptions(model_asset_path=str(p))  # noqa: E731
        mode = vision.RunningMode.IMAGE if self.image_mode else vision.RunningMode.VIDEO
        self._hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=base(hand_path), running_mode=mode, num_hands=2)
        )
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=base(pose_path), running_mode=mode)
        )
        self.warm_up()

    def warm_up(self, steps: int = WARMUP_STEPS) -> None:
        """Run dummy frames through both landmarkers so the first real frame never pays the cold-start cost."""
        rng = np.random.default_rng(42)
        side = MP_MAX_SIDE or 224
        dummy = rng.integers(0, 255, (side, side, 3), dtype=np.uint8)
        for _ in range(steps):
            if self.image_mode:
                img = self._to_mp_image(dummy)
                self._hand.detect(img)
                self._pose.detect(img)
            else:
                self._detect(dummy, 33)

    # ---- per-frame ----
    @staticmethod
    def _to_mp_image(bgr: np.ndarray) -> Any:
        import cv2
        import mediapipe as mp

        rgb = np.ascontiguousarray(cv2.cvtColor(preprocess_frame(bgr), cv2.COLOR_BGR2RGB))
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def _detect(self, bgr: np.ndarray, step_ms: int) -> np.ndarray:
        self._ts_ms += max(1, int(step_ms))
        img = self._to_mp_image(bgr)
        return assemble_frame(
            self._hand.detect_for_video(img, self._ts_ms), self._pose.detect_for_video(img, self._ts_ms)
        )

    def detect_frame(self, bgr: np.ndarray, step_ms: int = 33) -> np.ndarray:
        """One frame -> (53,3). IMAGE mode is stateless; VIDEO mode uses a monotonic timestamp."""
        self.load()
        if self.image_mode:
            img = self._to_mp_image(bgr)
            return assemble_frame(self._hand.detect(img), self._pose.detect(img))
        return self._detect(bgr, step_ms)

    def reset_tracking(self, shape: tuple[int, int, int]) -> None:
        """Drop tracker state left over from a previous clip (no-op in IMAGE mode / on first use)."""
        if not self.image_mode and self._used:
            self._reset_tracking(shape)
        self._used = True

    def _reset_tracking(self, shape: tuple[int, int, int]) -> None:
        """A blank frame makes the trackers drop their previous clip's state (cheap determinism aid)."""
        self._detect(np.zeros(shape, np.uint8), 1)

    def extract_arrays(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        """Like extract_frames but for in-memory BGR frames (e.g. perturbed frames in the robustness test)."""
        self.load()
        out = [self.detect_frame(f, 33) for f in frames_bgr]
        return np.stack(out).astype(np.float32) if out else np.zeros((0, NUM_JOINTS, 3), np.float32)

    def extract_frames(self, frame_paths: list[str | Path]) -> np.ndarray:
        """Landmarks for an ordered set of already-sparse frames (e.g. the 16-frame WLASL archive).

        Every frame is a keyframe: no skipping and no interpolation. Use image_mode=True so each
        frame is detected independently (frames are far apart in time, tracking would mislead).
        """
        import cv2

        self.load()
        out = []
        for p in frame_paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise ValueError(f"cannot read frame: {p}")
            if self.image_mode:
                img = self._to_mp_image(bgr)
                out.append(assemble_frame(self._hand.detect(img), self._pose.detect(img)))
            else:
                out.append(self._detect(bgr, 33))
        return np.stack(out).astype(np.float32) if out else np.zeros((0, NUM_JOINTS, 3), np.float32)

    # ---- whole clip ----
    def extract(self, video_path: str) -> np.ndarray:
        """Stream video_path; return (T,53,3) float32 with one row per source frame."""
        import cv2

        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"video not found: {path}")
        self.load()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps > 1.0 else 30.0
            step_ms = int(round(1000.0 * self.frame_skip / fps))
            interp = KeyframeInterpolator(self.frame_skip)
            frames: list[np.ndarray] = []
            idx = 0
            while True:
                if not cap.grab():  # grab() skips the decode/colour work for non-key frames
                    break
                if idx % self.frame_skip == 0:
                    ok, bgr = cap.retrieve()
                    if not ok:
                        break
                    if self._used and idx == 0:
                        self._reset_tracking(bgr.shape)
                    self._used = True
                    frames.extend(interp.push(self._detect(bgr, step_ms)))
                idx += 1
            n_total = idx
        finally:
            cap.release()
        frames.extend(interp.hold(n_total - len(frames)))
        if not frames:
            return np.zeros((0, NUM_JOINTS, 3), np.float32)
        return np.stack(frames[:n_total], axis=0).astype(np.float32)
