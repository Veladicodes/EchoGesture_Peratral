"""View-invariant 53-joint features (CLAUDE.md 5.2).

Input : raw joints (T, 53, 3)  [x, y, z], missing hand/body == all zeros.
Output: float32 tensor (NUM_CHANNELS, 64, 53), never NaN/Inf.

Joint layout: 0-20 left hand, 21-41 right hand, 42-52 upper body.
Channels (7): x_norm, y_norm, z_norm, vx, vy, cos_bone, sin_bone.
NUM_CHANNELS = 6 (z-ablation) drops z_norm (channel 2) only.

Pipeline order: resample -> [train aug: time-warp, rotate, jitter] -> normalise
-> [train aug: joint dropout] -> channels.  Augmenting before the channel
computation keeps velocity and bone angles consistent with the augmented pose.
"""
from __future__ import annotations

import numpy as np

from .common import get_num_channels

T_FRAMES = 64
NUM_JOINTS = 53
LEFT = slice(0, 21)
RIGHT = slice(21, 42)
BODY = slice(42, 53)
GROUPS = (LEFT, RIGHT, BODY)

NUM_CHANNELS: int = get_num_channels()  # from config.yaml, never hard-coded

PALM_EPS = 1e-5
BONE_EPS = 1e-8
MIN_VISIBLE_JOINTS = 3      # fewer visible joints than this and a hand is treated as missing
FALLBACK_MIN_SCALE = 5e-3   # centroid-fallback needs a hand at least this big (image-normalised units)

# ---- kinematic tree: single source of truth (also builds the GCN adjacency) ----
# MediaPipe hand skeleton rooted at the wrist (index 0); root's parent is itself.
_HAND_PARENTS = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
# Body (global joint ids): 42 nose, 43 L-eye, 44 R-eye, 45 L-ear, 46 R-ear,
# 47 L-shoulder, 48 R-shoulder, 49 L-elbow, 50 R-elbow, 51 L-wrist, 52 R-wrist.
# head -> shoulders -> elbows -> wrists; L-shoulder is the (single) body root.
_BODY_PARENTS = [47, 42, 42, 43, 44, 47, 47, 47, 48, 49, 50]
PARENTS: list[int] = (
    list(_HAND_PARENTS) + [p + 21 for p in _HAND_PARENTS] + list(_BODY_PARENTS)
)
assert len(PARENTS) == NUM_JOINTS and all(0 <= p < NUM_JOINTS for p in PARENTS)

_PARENT_IDX = np.asarray(PARENTS, dtype=np.int64)
_IS_ROOT = _PARENT_IDX == np.arange(NUM_JOINTS)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sanitize(joints: np.ndarray) -> np.ndarray:
    arr = np.asarray(joints, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (NUM_JOINTS, 3):
        raise ValueError(f"joints must have shape (T, {NUM_JOINTS}, 3), got {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)  # NaN/Inf == missing
    if arr.shape[0] == 0:
        arr = np.zeros((1, NUM_JOINTS, 3), np.float32)
    return arr


def group_presence(joints: np.ndarray) -> np.ndarray:
    """(T, 3) bool: left hand / right hand / body has any non-zero coordinate."""
    return np.stack([np.any(joints[:, g] != 0.0, axis=(1, 2)) for g in GROUPS], axis=1)


def resample_time(joints: np.ndarray, t_out: int = T_FRAMES) -> np.ndarray:
    """Uniform temporal resampling to t_out frames with a zero-hand guard.

    Linear interpolation is used for a group only when it is present in BOTH
    neighbouring frames; otherwise the present neighbour is copied and an absent
    group stays zero, so no phantom limbs are created.
    """
    j = _sanitize(joints)
    t_in = j.shape[0]
    if t_in == t_out:
        return j.copy()
    pos = np.linspace(0.0, t_in - 1, t_out) if t_in > 1 else np.zeros(t_out)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, t_in - 1)
    w = (pos - i0).astype(np.float32)
    pres = group_presence(j)
    out = np.zeros((t_out, NUM_JOINTS, 3), np.float32)
    for gi, g in enumerate(GROUPS):
        a0, a1 = j[i0][:, g], j[i1][:, g]
        p0, p1 = pres[i0, gi][:, None, None], pres[i1, gi][:, None, None]
        ww = w[:, None, None]
        lerp = a0 * (1.0 - ww) + a1 * ww
        out[:, g] = np.where(p0 & p1, lerp, np.where(p0, a0, np.where(p1, a1, 0.0)))
    return out


def normalize_joints(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-group translation/scale normalisation on x, y, z together.

    Hands : subtract wrist, divide by ||middle_MCP - wrist||; palm_len < 1e-5 -> zeroed.
    Body  : subtract shoulder midpoint, divide by shoulder width (+1e-5).
    Returns (norm (T,53,3), valid (T,53) bool).
    """
    j = joints.astype(np.float32, copy=False)
    t = j.shape[0]
    out = np.zeros_like(j)
    valid = np.zeros((t, NUM_JOINTS), dtype=bool)

    for g in (LEFT, RIGHT):
        h = j[:, g]  # (T,21,3)
        vis = np.any(h != 0.0, axis=-1)  # (T,21) a joint with all-zero coordinates is missing
        wrist, mcp = h[:, 0], h[:, 9]
        palm = np.linalg.norm(mcp - wrist, axis=-1)
        std_ok = vis[:, 0] & vis[:, 9] & (palm >= PALM_EPS)  # documented wrist / middle-MCP normalisation
        ok = std_ok
        scaled = (h - wrist[:, None, :]) / np.where(ok, palm, 1.0)[:, None, None]
        keep = ok[:, None] & vis  # missing joints stay exactly zero
        out[:, g] = np.where(keep[..., None], scaled, 0.0)
        valid[:, g] = keep

    b = j[:, BODY]  # (T,11,3); L-shoulder = 5, R-shoulder = 6
    vis_b = np.any(b != 0.0, axis=-1)
    ls, rs = b[:, 5], b[:, 6]
    width = np.linalg.norm(ls - rs, axis=-1)
    ok = vis_b[:, 5] & vis_b[:, 6] & (width >= PALM_EPS)
    mid = 0.5 * (ls + rs)
    scaled = (b - mid[:, None, :]) / (width + PALM_EPS)[:, None, None]
    keep = ok[:, None] & vis_b
    out[:, BODY] = np.where(keep[..., None], scaled, 0.0)
    valid[:, BODY] = keep
    return out, valid


def _channels(norm: np.ndarray, valid: np.ndarray, num_channels: int) -> np.ndarray:
    x, y, z = norm[..., 0], norm[..., 1], norm[..., 2]

    vx = np.zeros_like(x)
    vy = np.zeros_like(y)
    both = valid[1:] & valid[:-1]  # zeroed hand => velocity 0, never a spike
    vx[1:] = np.where(both, x[1:] - x[:-1], 0.0)
    vy[1:] = np.where(both, y[1:] - y[:-1], 0.0)

    dx = x - x[:, _PARENT_IDX]
    dy = y - y[:, _PARENT_IDX]
    length = np.hypot(dx, dy)
    ok = valid & valid[:, _PARENT_IDX] & (length >= BONE_EPS) & (~_IS_ROOT)[None, :]
    safe = np.where(ok, length, 1.0)
    cos_b = np.where(ok, dx / safe, 1.0)
    sin_b = np.where(ok, dy / safe, 0.0)

    stack = np.stack([x, y, z, vx, vy, cos_b, sin_b], axis=0).astype(np.float32)
    if num_channels == 6:
        stack = np.delete(stack, 2, axis=0)  # drop z_norm only
    elif num_channels != 7:
        raise ValueError(f"num_channels must be 6 or 7, got {num_channels}")
    return np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# training augmentations (train split only)
# --------------------------------------------------------------------------- #
def time_warp(joints64: np.ndarray, speed: float, rng: np.random.Generator) -> np.ndarray:
    """0.8x-1.2x temporal speed change on a 64-frame window (crop or edge-pad back to 64)."""
    n = max(2, int(round(joints64.shape[0] * speed)))
    w = resample_time(joints64, n)
    if n >= T_FRAMES:
        s = int(rng.integers(0, n - T_FRAMES + 1))
        return w[s : s + T_FRAMES]
    left = int(rng.integers(0, T_FRAMES - n + 1))
    return np.pad(w, ((left, T_FRAMES - n - left), (0, 0), (0, 0)), mode="edge")


def augment_geometry(
    joints64: np.ndarray,
    rng: np.random.Generator,
    rot_deg: float = 15.0,
    jitter: float = 0.015,
    speed: tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """Time-warp, +-rot_deg in-plane (x,y) rotation, Gaussian jitter. Missing groups stay zero."""
    j = time_warp(joints64, float(rng.uniform(*speed)), rng)
    pres = group_presence(j)  # (T,3)
    joint_pres = np.concatenate(
        [np.repeat(pres[:, 0:1], 21, 1), np.repeat(pres[:, 1:2], 21, 1), np.repeat(pres[:, 2:3], 11, 1)],
        axis=1,
    )[..., None]
    th = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    c, s = np.cos(th), np.sin(th)
    out = j.copy()
    out[..., 0] = c * j[..., 0] - s * j[..., 1]
    out[..., 1] = s * j[..., 0] + c * j[..., 1]  # z unchanged
    out = out + rng.normal(0.0, jitter, size=out.shape).astype(np.float32)
    out = np.where(joint_pres, out, 0.0).astype(np.float32)
    return out


def _raw_dropout(j: np.ndarray, rng: np.random.Generator, p_hand: float = 0.10, p_joint: float = 0.05) -> np.ndarray:
    """Raw-landmark failures seen at inference: a whole hand lost for a frame (p_hand per hand per frame) and an
    occasional single lost joint (p_joint). The anchor-fallback in normalize_joints handles the latter."""
    out = j.copy()
    gone = rng.random((out.shape[0], 2)) < p_hand
    out[gone[:, 0], LEFT] = 0.0
    out[gone[:, 1], RIGHT] = 0.0
    out *= (rng.random(out.shape[:2]) >= p_joint)[..., None]
    return out.astype(np.float32)


def apply_joint_dropout(
    norm: np.ndarray, valid: np.ndarray, rng: np.random.Generator, p: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Zero x, y, z of ~p of (frame, joint) entries together and mark them invalid."""
    keep = rng.random(valid.shape) >= p
    return norm * keep[..., None], valid & keep


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def build_features(
    joints: np.ndarray,
    num_channels: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """(T,53,3) raw joints -> (num_channels, 64, 53) float32. Pass rng for train augmentation."""
    nc = NUM_CHANNELS if num_channels is None else num_channels
    j = resample_time(joints, T_FRAMES)
    if rng is not None:
        j = augment_geometry(j, rng)
    norm, valid = normalize_joints(j)
    if rng is not None:
        norm, valid = apply_joint_dropout(norm, valid, rng)
    return _channels(norm, valid, nc)
