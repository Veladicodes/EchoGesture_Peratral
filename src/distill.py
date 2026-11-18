"""MAG-KD (correctness-masked knowledge distillation) + the no-distill fallback (project spec 5.3).

    mask      = teacher_logits.argmax(1) == y_true             # distil only where the teacher is right
    masked_kl = (KL(softmax(t/tau) || softmax(s/tau)) * mask).sum() / mask.sum().clamp(min=1)
    loss      = (1 - lam) * CE + lam * tau^2 * masked_kl        # tau = 4.0, lam = 0.6

Fallback (--no_distill): label-smoothed CE (0.1) with temporal Mixup / CutMix on the keypoint tensors.
Also here: the keypoint Dataset, evaluate(), a generic train loop, and the promotion / cutoff rules.
No network, no downloads.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import perturb as PT
from .features import build_features

TAU = 4.0
LAMBDA = 0.6
LABEL_SMOOTHING = 0.1


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def mag_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    y: torch.Tensor,
    tau: float = TAU,
    lam: float = LAMBDA,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    teacher_logits = teacher_logits.detach()
    ce = F.cross_entropy(student_logits, y, label_smoothing=label_smoothing)
    mask = (teacher_logits.argmax(dim=1) == y).float()
    kl = F.kl_div(
        F.log_softmax(student_logits / tau, dim=-1), F.softmax(teacher_logits / tau, dim=-1), reduction="none"
    ).sum(dim=-1)
    masked_kl = (kl * mask).sum() / mask.sum().clamp(min=1.0)
    return (1.0 - lam) * ce + lam * (tau**2) * masked_kl


def no_distill_loss(student_logits: torch.Tensor, y: torch.Tensor, smoothing: float = LABEL_SMOOTHING) -> torch.Tensor:
    return F.cross_entropy(student_logits, y, label_smoothing=smoothing)


def mix_loss(logits: torch.Tensor, ya: torch.Tensor, yb: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * no_distill_loss(logits, ya) + (1.0 - lam) * no_distill_loss(logits, yb)


def _partner(n: int, rng: np.random.Generator) -> torch.Tensor:
    """Permutation with no fixed point (cyclic shift) so every sample mixes with a different one."""
    shift = int(rng.integers(1, n)) if n > 1 else 0
    return torch.from_numpy((np.arange(n) + shift) % n)


def temporal_mixup(x: torch.Tensor, y: torch.Tensor, rng: np.random.Generator, alpha: float = 0.4):
    """Convex mix of two keypoint sequences. Returns (x_mix, y_a, y_b, lam)."""
    lam = float(rng.beta(alpha, alpha))
    p = _partner(x.shape[0], rng)
    return lam * x + (1.0 - lam) * x[p], y, y[p], lam


def temporal_cutmix(x: torch.Tensor, y: torch.Tensor, rng: np.random.Generator):
    """Replace a random time segment of each sequence with the partner's. lam = kept-frame fraction."""
    n, _, t, _ = x.shape
    cut = int(np.clip(round((1.0 - float(rng.beta(1.0, 1.0))) * t), 1, t - 1))
    start = int(rng.integers(0, t - cut + 1))
    p = _partner(n, rng)
    xm = x.clone()
    xm[:, :, start : start + cut, :] = x[p][:, :, start : start + cut, :]
    return xm, y, y[p], 1.0 - cut / t


# --------------------------------------------------------------------------- #
# rules
# --------------------------------------------------------------------------- #
def choose_student(distilled_top1: float, baseline_top1: float, min_gain: float) -> str:
    """Promote the distilled student only if it beats the --no_distill baseline by >= min_gain."""
    return "distilled" if (distilled_top1 - baseline_top1) >= min_gain - 1e-9 else "baseline"


def run_score(run: dict) -> float:
    """Seed-selection score: mean of clean val top-1 and perturbed ('robust') val top-1."""
    return 0.5 * (run["clean"] + run["robust"])


def pick_best_run(runs: list[dict]) -> dict:
    return max(runs, key=run_score)


def cutoff_passed(cutoff_iso: str, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    return now >= datetime.fromisoformat(cutoff_iso)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class KeypointDataset(Dataset):
    """Cached (T,53,3) .npy clips -> (C,64,53) tensors. Train split is augmented deterministically:
    the rng depends only on (seed, epoch, index), so runs are reproducible for any worker count."""

    def __init__(self, rows: list[dict], split: str, cache_dir: str | Path, num_channels: int,
                 train: bool = False, seed: int = 42, dark_dir: str | Path | None = None, p_dark: float = 0.0,
                 perturb: bool = False, gentle: bool = False):
        """dark_dir: parallel landmark cache extracted from low-light frames (same layout as cache_dir).
        Train: each sample uses the dark landmarks with probability p_dark (genuine low-light augmentation).
        Eval + perturb=True: a fixed, seeded 'hard' validation set = dark landmarks (if given) + tilt + 15% hand drops."""
        self.nc, self.train, self.seed, self.epoch, self.p_dark = num_channels, train, seed, 0, p_dark
        self.gentle = gentle  # light fine-tune mode: optional dark landmarks + mild tilt/hand-drop, nothing else
        self.joints = [np.load(Path(cache_dir) / split / f"{r['clip_id']}.npy") for r in rows]
        self.dark = None
        if dark_dir is not None:
            self.dark = [np.load(Path(dark_dir) / split / f"{r['clip_id']}.npy") for r in rows]
        self.labels = [int(r["label"]) for r in rows]
        if train:
            self._eval = None
        elif perturb:
            src = self.dark if self.dark is not None else self.joints
            self._eval = []
            for i, j in enumerate(src):
                rng = np.random.default_rng([seed, 99, i])
                self._eval.append(build_features(PT.drop_hands(PT.tilt(j, rng), rng), num_channels))
        else:
            self._eval = [build_features(j, num_channels) for j in self.joints]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        if self._eval is not None:
            f = self._eval[i]
        else:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            use_dark = self.dark is not None and rng.random() < self.p_dark
            if self.gentle:
                j = (self.dark if use_dark else self.joints)[i]
                if rng.random() < 0.5:
                    j = PT.tilt(j, rng, deg=8.0)
                if rng.random() < 0.5:
                    j = PT.drop_hands(j, rng, 0.05)
                return torch.from_numpy(build_features(j, self.nc)), torch.tensor(self.labels[i], dtype=torch.long)
            f = build_features((self.dark if use_dark else self.joints)[i], self.nc, rng=rng)
        return torch.from_numpy(f), torch.tensor(self.labels[i], dtype=torch.long)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> dict:
    model.eval()
    c1 = c5 = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        k = min(5, logits.shape[1])
        top = logits.topk(k, dim=1).indices
        c1 += int((top[:, 0] == y).sum())
        c5 += int((top == y[:, None]).any(dim=1).sum())
        n += int(y.numel())
    return {"top1": c1 / max(n, 1), "top5": c5 / max(n, 1), "n": n}


# --------------------------------------------------------------------------- #
# generic training loop
# --------------------------------------------------------------------------- #
def train_model(
    model: nn.Module,
    train_ds: KeypointDataset,
    val_ds: KeypointDataset,
    *,
    epochs: int,
    lr: float,
    mode: str,                      # "ce" (teacher) | "nodistill" (baseline student) | "kd" (MAG-KD student)
    teacher: nn.Module | None = None,
    batch_size: int = 32,
    weight_decay: float = 1e-2,
    mix_prob: float = 0.5,
    device: str = "cpu",
    seed: int = 42,
    log: Callable[[str], None] = print,
    tag: str = "",
) -> tuple[dict, list[dict], dict]:
    """Returns (best_state_dict on cpu, history, best_val_metrics). Best = highest val top-1."""
    assert mode in ("ce", "nodistill", "kd")
    if mode == "kd":
        assert teacher is not None
        teacher.to(device).eval()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * 0.02)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    mix_rng = np.random.default_rng(seed)
    best, best_state, hist = {"top1": -1.0}, None, []
    t0 = time.time()
    for ep in range(epochs):
        train_ds.set_epoch(ep)
        g = torch.Generator().manual_seed(seed + ep)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=len(train_ds) > batch_size)
        model.train()
        tot, nb = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            if mode == "kd":
                with torch.no_grad():
                    t_logits = teacher(x)
                loss = mag_kd_loss(model(x), t_logits, y)
            elif mode == "nodistill" and mix_rng.random() < mix_prob:
                fn = temporal_mixup if mix_rng.random() < 0.5 else temporal_cutmix
                xm, ya, yb, lam = fn(x, y, mix_rng)
                loss = mix_loss(model(xm), ya, yb, lam)
            else:
                loss = no_distill_loss(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot, nb = tot + float(loss), nb + 1
        sched.step()
        m = evaluate(model, val_loader, device)
        hist.append({"epoch": ep + 1, "loss": tot / max(nb, 1), "val_top1": m["top1"], "val_top5": m["top5"]})
        if m["top1"] > best["top1"]:
            best = m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(f"[{tag or mode}] ep {ep + 1}/{epochs} loss={tot / max(nb, 1):.4f} val_top1={m['top1']:.4f} "
            f"val_top5={m['top5']:.4f} best={best['top1']:.4f} t={time.time() - t0:.0f}s")
    return best_state, hist, best
