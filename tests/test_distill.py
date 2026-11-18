"""MAG-KD loss, no-distill fallback, temporal Mixup/CutMix, dataset, promotion rule (project spec 5.3)."""
from datetime import datetime, timezone

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src import distill as D
from src.features import NUM_JOINTS, T_FRAMES


def _kl_per_sample(s, t, tau):
    return F.kl_div(F.log_softmax(s / tau, -1), F.softmax(t / tau, -1), reduction="none").sum(-1)


def test_mag_kd_masks_samples_where_teacher_is_wrong():
    torch.manual_seed(0)
    s, t = torch.randn(4, 10), torch.randn(4, 10)
    y = t.argmax(1).clone()
    y[2] = (y[2] + 1) % 10  # teacher wrong on samples 2 and 3
    y[3] = (y[3] + 2) % 10
    tau, lam = 4.0, 0.6
    loss = D.mag_kd_loss(s, t, y, tau=tau, lam=lam)
    kl = _kl_per_sample(s, t, tau)
    expect = (1 - lam) * F.cross_entropy(s, y) + lam * tau**2 * (kl[:2].sum() / 2)
    assert torch.allclose(loss, expect, atol=1e-5)


def test_mag_kd_all_teacher_wrong_is_pure_ce_part():
    s, t = torch.randn(3, 5), torch.randn(3, 5)
    y = (t.argmax(1) + 1) % 5
    loss = D.mag_kd_loss(s, t, y, tau=4.0, lam=0.6)
    assert torch.allclose(loss, 0.4 * F.cross_entropy(s, y), atol=1e-6)
    assert torch.isfinite(loss)


def test_mag_kd_lambda_zero_is_ce_and_defaults_match_spec():
    s, t = torch.randn(6, 7), torch.randn(6, 7)
    y = torch.randint(0, 7, (6,))
    assert torch.allclose(D.mag_kd_loss(s, t, y, lam=0.0), F.cross_entropy(s, y), atol=1e-6)
    assert D.TAU == 4.0 and D.LAMBDA == 0.6 and D.LABEL_SMOOTHING == 0.1


def test_mag_kd_gradient_reaches_student_not_teacher():
    s = torch.randn(4, 6, requires_grad=True)
    t = torch.randn(4, 6, requires_grad=True)
    D.mag_kd_loss(s, t, t.argmax(1), 4.0, 0.6).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert t.grad is None or torch.all(t.grad == 0)  # teacher is detached


def test_no_distill_uses_label_smoothing_01():
    s = torch.randn(5, 9)
    y = torch.randint(0, 9, (5,))
    assert torch.allclose(D.no_distill_loss(s, y), F.cross_entropy(s, y, label_smoothing=0.1))


def test_temporal_mixup_properties():
    x = torch.randn(8, 7, T_FRAMES, NUM_JOINTS)
    y = torch.arange(8)
    xm, ya, yb, lam = D.temporal_mixup(x, y, np.random.default_rng(0), alpha=0.4)
    assert xm.shape == x.shape and 0.0 <= lam <= 1.0
    assert torch.equal(ya, y) and sorted(yb.tolist()) == list(range(8))
    assert torch.equal(xm, D.temporal_mixup(x, y, np.random.default_rng(0), alpha=0.4)[0])


def test_temporal_cutmix_replaces_a_time_segment():
    x = torch.randn(6, 7, T_FRAMES, NUM_JOINTS)
    y = torch.arange(6)
    xm, ya, yb, lam = D.temporal_cutmix(x, y, np.random.default_rng(1))
    assert xm.shape == x.shape and 0.0 < lam < 1.0
    kept = (xm == x).all(dim=3).all(dim=1)  # (N,T): frames untouched
    # lam is the fraction of original frames kept
    assert abs(kept[0].float().mean().item() - lam) <= 1 / T_FRAMES + 1e-6


def test_mix_loss_interpolates_two_targets():
    s = torch.randn(4, 6)
    ya, yb = torch.randint(0, 6, (4,)), torch.randint(0, 6, (4,))
    exp = 0.3 * D.no_distill_loss(s, ya) + 0.7 * D.no_distill_loss(s, yb)
    assert torch.allclose(D.mix_loss(s, ya, yb, 0.3), exp, atol=1e-6)


def test_choose_student_promotion_rule():
    assert D.choose_student(0.50, 0.47, 0.02) == "distilled"
    assert D.choose_student(0.49, 0.47, 0.02) == "distilled"      # exactly +2 pts counts
    assert D.choose_student(0.489, 0.47, 0.02) == "baseline"
    assert D.choose_student(0.40, 0.47, 0.02) == "baseline"


def test_cutoff_passed():
    iso = "2026-11-16T23:59:00+05:30"
    assert not D.cutoff_passed(iso, now=datetime(2026, 11, 14, tzinfo=timezone.utc))
    assert D.cutoff_passed(iso, now=datetime(2026, 11, 17, tzinfo=timezone.utc))


@pytest.fixture
def fake_cache(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "train").mkdir()
    rows = []
    for i in range(6):
        j = np.zeros((16, NUM_JOINTS, 3), np.float32)
        h = rng.normal(0, 0.03, (16, 21, 3)).astype(np.float32)
        h[:, 0] = 0
        h[:, 9] = [0, -0.08, 0]
        j[:, :21] = 0.5 + h
        j[:, 47] = [0.4, 0.5, 0]
        j[:, 48] = [0.6, 0.5, 0]
        np.save(tmp_path / "train" / f"c{i}.npy", j)
        rows.append({"clip_id": f"c{i}", "gloss": "A", "label": str(i % 3)})
    return tmp_path, rows


def test_dataset_shapes_labels_and_epoch_determinism(fake_cache):
    cache, rows = fake_cache
    ds = D.KeypointDataset(rows, "train", cache, num_channels=7, train=True)
    x, y = ds[2]
    assert x.shape == (7, 64, 53) and x.dtype == torch.float32 and int(y) == 2 % 3
    assert torch.equal(ds[2][0], ds[2][0])            # same epoch + sample -> identical
    ds.set_epoch(1)
    assert not torch.equal(x, ds[2][0])               # new epoch -> new augmentation
    ev = D.KeypointDataset(rows, "train", cache, num_channels=6, train=False)
    assert ev[0][0].shape == (6, 64, 53) and torch.equal(ev[0][0], ev[0][0])


@pytest.fixture
def dark_cache(fake_cache, tmp_path):
    """A 'dark' landmark cache where the left hand is never detected (what low light does to MediaPipe)."""
    cache, rows = fake_cache
    (tmp_path / "dk" / "train").mkdir(parents=True)
    for r in rows:
        j = np.load(cache / "train" / f"{r['clip_id']}.npy").copy()
        j[:, :21] = 0.0
        np.save(tmp_path / "dk" / "train" / f"{r['clip_id']}.npy", j)
    return cache, rows, tmp_path / "dk"


def test_dark_cache_is_mixed_into_training_with_probability_p(dark_cache):
    cache, rows, dk = dark_cache
    always = D.KeypointDataset(rows, "train", cache, 7, train=True, dark_dir=dk, p_dark=1.0)
    never = D.KeypointDataset(rows, "train", cache, 7, train=True, dark_dir=dk, p_dark=0.0)
    for i in range(len(rows)):
        assert np.all(always[i][0][0, :, :21].numpy() == 0.0), "p_dark=1 must use the dark landmarks"
    assert np.abs(never[0][0][0, :, :21].numpy()).sum() > 0
    mixed = D.KeypointDataset(rows, "train", cache, 7, train=True, dark_dir=dk, p_dark=0.5)
    hits = 0
    for ep in range(40):
        mixed.set_epoch(ep)
        hits += int(np.all(mixed[0][0][0, :, :21].numpy() == 0.0))
    assert 8 <= hits <= 32  # roughly half of the epochs (some clean draws also lose the hand to raw hand-drop aug)


def test_perturbed_eval_dataset_is_deterministic_and_differs(dark_cache):
    cache, rows, dk = dark_cache
    clean = D.KeypointDataset(rows, "train", cache, 7, train=False)
    p1 = D.KeypointDataset(rows, "train", cache, 7, train=False, perturb=True)
    p2 = D.KeypointDataset(rows, "train", cache, 7, train=False, perturb=True)
    assert torch.equal(p1[1][0], p2[1][0])
    assert not torch.equal(clean[1][0], p1[1][0])
    pd = D.KeypointDataset(rows, "train", cache, 7, train=False, perturb=True, dark_dir=dk)
    assert np.all(pd[0][0][0, :, :21].numpy() == 0.0)  # dark landmarks are used when a dark cache is given


def test_pick_best_seed_uses_clean_plus_robust_score():
    runs = [{"seed": 1, "clean": 0.50, "robust": 0.30}, {"seed": 2, "clean": 0.48, "robust": 0.40},
            {"seed": 3, "clean": 0.52, "robust": 0.20}]
    assert D.pick_best_run(runs)["seed"] == 2          # 0.44 beats 0.40 and 0.36
    assert D.run_score({"clean": 0.5, "robust": 0.3}) == pytest.approx(0.4)


def test_evaluate_reports_top1_top5():
    class M(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.0, 5.0, 1.0, 0.5, 0.2, 0.1]] * x.shape[0])

    ds = torch.utils.data.TensorDataset(torch.zeros(4, 1), torch.tensor([1, 1, 2, 0]))
    r = D.evaluate(M(), torch.utils.data.DataLoader(ds, batch_size=2))
    assert r["top1"] == 0.5 and r["top5"] == 0.75 and r["n"] == 4
