"""Test-time / robustness perturbations (numpy only; seeded; no I/O).

Shared by scripts/benchmark.py (robustness metric), scripts/gcp_extract.py (dark-frame landmark cache used as
low-light training augmentation) and src/distill.py (perturbed validation set).

low_light_frame : gamma 2.2, x0.5 brightness, gaussian sensor noise (sigma 8) on an image BEFORE landmark extraction
drop_hands      : whole hand disappears for a frame (what MediaPipe really does when detection fails), p per hand per frame
occlude_joints  : single joints vanish (diagnostic only: MediaPipe does not drop one hand joint at a time)
tilt            : +-deg in-plane camera roll about the image centre (x,y only; z untouched; missing stays zero)
"""
from __future__ import annotations

import numpy as np

from .features import LEFT, RIGHT


def low_light_frame(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = (bgr.astype(np.float32) / 255.0) ** 2.2 * 0.5
    x = x * 255.0 + rng.normal(0.0, 8.0, x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def drop_hands(joints: np.ndarray, rng: np.random.Generator, p: float = 0.15) -> np.ndarray:
    """Zero the whole left / right hand (21 joints) independently on ~p of the frames."""
    out = np.array(joints, dtype=np.float32, copy=True)
    gone = rng.random((out.shape[0], 2)) < p
    out[gone[:, 0], LEFT] = 0.0
    out[gone[:, 1], RIGHT] = 0.0
    return out


def occlude_joints(joints: np.ndarray, rng: np.random.Generator, p: float = 0.15) -> np.ndarray:
    """Zero x, y, z of ~p of the (frame, joint) entries together."""
    keep = rng.random(joints.shape[:2]) >= p
    return (joints * keep[..., None]).astype(np.float32)


def noise(joints: np.ndarray, rng: np.random.Generator, sigma: float = 0.015) -> np.ndarray:
    """Gaussian coordinate noise on present joints only (missing joints stay exactly zero)."""
    j = np.asarray(joints, dtype=np.float32)
    present = np.any(j != 0, axis=-1, keepdims=True)
    return np.where(present, j + rng.normal(0.0, sigma, j.shape).astype(np.float32), 0.0).astype(np.float32)


def tilt(joints: np.ndarray, rng: np.random.Generator, deg: float = 15.0) -> np.ndarray:
    th = np.deg2rad(rng.uniform(-deg, deg))
    c, s = np.cos(th), np.sin(th)
    present = np.any(joints != 0, axis=-1)
    xy = joints[..., :2] - 0.5
    out = joints.astype(np.float32, copy=True)
    out[..., 0] = np.where(present, c * xy[..., 0] - s * xy[..., 1] + 0.5, 0.0)
    out[..., 1] = np.where(present, s * xy[..., 0] + c * xy[..., 1] + 0.5, 0.0)
    return out
