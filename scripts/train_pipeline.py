"""Teacher -> MAG-KD / no-distill students -> promotion rule -> QAT -> INT8 ONNX + TFLite export.

    python scripts/train_pipeline.py                      # full run (GPU if available)
    python scripts/train_pipeline.py --quick              # tiny smoke run
    python scripts/train_pipeline.py --num_channels 6     # skip the ablation, force 6 channels

Stages (all logged to outputs/train_log.txt, summary to outputs/training_summary.json):
  1. ablation   : baseline student (--no_distill) with 7 channels (x,y,z,..) vs 6 (no z); keep z only if it does not hurt
  2. teacher    : DualStreamTeacherGCN, label-smoothed CE                      -> weights/teacher_best.pt
  3. students   : baseline (--no_distill: LS 0.1 + temporal Mixup/CutMix) and MAG-KD distilled (tau=4, lam=0.6)
  4. promotion  : distilled is promoted only if val top-1 beats the baseline by >= --teacher_min_gain (default 0.02);
                  if the training cutoff has passed, distillation is skipped and the baseline is promoted
  5. QAT        : fuse Conv+BN(+ReLU), 15 epochs fake-quant fine-tune at lr 1e-4
  6. export     : weights/student_int8.onnx (+ .tflite), INT8 val accuracy vs FP32 (gate: drop <= 1 pt)
Never fabricates numbers: every metric is computed from the data on disk and written to the summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import distill as D  # noqa: E402
from src import quantize as Q  # noqa: E402
from src.common import CONFIG_PATH, load_config, load_vocab, seed_everything  # noqa: E402
from src.models import DualStreamTeacherGCN, StudentSTGCN, count_params  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def set_config_channels(nc: int) -> None:
    txt = CONFIG_PATH.read_text(encoding="utf-8")
    CONFIG_PATH.write_text(re.sub(r"(?m)^num_channels:\s*\d+", f"num_channels: {nc}", txt), encoding="utf-8")
    load_config.cache_clear()


def train_student(nc, ncls, tr, va, *, mode, teacher, epochs, lr, device, log, tag, seed=42, mix_prob=0.5):
    seed_everything(seed)
    m = StudentSTGCN(ncls, nc)
    state, hist, best = D.train_model(m, tr, va, epochs=epochs, lr=lr, mode=mode, teacher=teacher, device=device,
                                      log=log, tag=tag, seed=seed, mix_prob=mix_prob)
    m.load_state_dict(state)
    return m.cpu().eval(), hist, best


def onnx_val_metrics(onnx_path: Path, va: D.KeypointDataset) -> dict:
    x = np.stack([va[i][0].numpy() for i in range(len(va))])
    logits = Q._run_onnx(onnx_path, x)
    y = np.array(va.labels)
    top5 = np.argsort(-logits, axis=1, kind="stable")[:, :5]
    return {"top1": float((top5[:, 0] == y).mean()), "top5": float((top5 == y[:, None]).any(1).mean()), "n": int(len(y))}


def main(argv: list[str] | None = None) -> dict:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="weights")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num_channels", type=int, choices=[6, 7], default=None, help="skip the ablation and force this")
    ap.add_argument("--ablation-epochs", type=int, default=40)
    ap.add_argument("--teacher-epochs", type=int, default=80)
    ap.add_argument("--student-epochs", type=int, default=120)
    ap.add_argument("--qat-epochs", type=int, default=int(cfg["qat_epochs"]))
    ap.add_argument("--student-lr", type=float, default=2e-3)
    ap.add_argument("--teacher-lr", type=float, default=1e-3)
    ap.add_argument("--teacher_min_gain", type=float, default=float(cfg["teacher_min_gain"]))
    ap.add_argument("--calib_samples", type=int, default=int(cfg["calib_samples"]))
    ap.add_argument("--cutoff", default=cfg["training_cutoff"])
    ap.add_argument("--no_distill", action="store_true", help="skip the teacher/distillation path entirely")
    ap.add_argument("--skip-tflite", action="store_true")
    ap.add_argument("--no-update-config", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43], help="student seeds; best clean+robust val wins")
    ap.add_argument("--mix-prob", type=float, default=0.6, help="temporal Mixup/CutMix probability (--no_distill student)")
    ap.add_argument("--dark-dir", default="data/cache_dark", help="low-light landmark cache (training aug + robust val)")
    ap.add_argument("--p-dark", type=float, default=0.3, help="probability a training sample uses the dark landmarks")
    ap.add_argument("--gentle-qat", action="store_true", help="QAT data = clean + low-light landmarks + mild tilt/hand-drop only (run-1 recipe + dark)")
    ap.add_argument("--quick", action="store_true", help="1-2 epoch smoke run")
    ap.add_argument("--export-only", action="store_true", help="skip training; redo QAT + export from weights/student_best.pt")
    ap.add_argument("--subset", type=int, default=0, help="debug: use only ~N train and ~N/3 val clips (evenly spaced)")
    ap.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = all cores)")
    a = ap.parse_args(argv)
    import os
    torch.set_num_threads(a.threads or os.cpu_count() or 1)
    if a.quick:
        a.ablation_epochs = a.teacher_epochs = a.student_epochs = 2
        a.qat_epochs = 1

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    data, cache = Path(a.data_dir), Path(a.cache)
    data = data if data.is_absolute() else ROOT / data
    cache = cache if cache.is_absolute() else ROOT / cache
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    outputs = Path(a.outputs) if Path(a.outputs).is_absolute() else ROOT / a.outputs
    out.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    log = Tee(outputs / "train_log.txt")
    t_start = time.time()
    seed_everything(42)

    vocab = load_vocab()
    ncls = len(vocab)
    tr_rows, va_rows = read_rows(data / "train_manifest.csv"), read_rows(data / "val_manifest.csv")
    if a.subset:
        tr_rows = tr_rows[:: max(1, len(tr_rows) // a.subset)]
        va_rows = va_rows[:: max(1, len(va_rows) // max(1, a.subset // 3))]
    log(f"=== run start | device={device} classes={ncls} train={len(tr_rows)} val={len(va_rows)} args={vars(a)}")
    summary: dict = {"device": device, "num_classes": ncls, "train_clips": len(tr_rows), "val_clips": len(va_rows),
                     "args": vars(a), "vocab_proxy": True}

    dark = Path(a.dark_dir) if Path(a.dark_dir).is_absolute() else ROOT / a.dark_dir
    dark = dark if (dark / "train").is_dir() and (dark / "val").is_dir() else None
    if dark is None:
        log("WARNING: no dark-frame cache found; training without low-light augmentation and robust val without dark frames")

    def datasets(nc):
        """train (augmented, mixes dark landmarks), clean val, and a fixed perturbed 'robust' val
        (dark landmarks + tilt + 15% whole-hand drops)."""
        return (D.KeypointDataset(tr_rows, "train", cache, nc, train=True, dark_dir=dark, p_dark=a.p_dark if dark else 0.0),
                D.KeypointDataset(va_rows, "val", cache, nc, train=False),
                D.KeypointDataset(va_rows, "val", cache, nc, train=False, perturb=True, dark_dir=dark))

    def run_seeds(mode, teacher, tag, tr, va, vr, nc):
        runs = []
        for sd in a.seeds:
            m, _, _ = train_student(nc, ncls, tr, va, mode=mode, teacher=teacher, epochs=a.student_epochs,
                                    lr=a.student_lr, device=device, log=log, tag=f"{tag}-seed{sd}", seed=sd,
                                    mix_prob=a.mix_prob)
            r = {"seed": sd, "clean": D.evaluate(m, DataLoader(va, batch_size=128))["top1"],
                 "robust": D.evaluate(m, DataLoader(vr, batch_size=128))["top1"], "model": m}
            log(f"{tag} seed {sd}: clean val top1={r['clean']:.4f} robust val top1={r['robust']:.4f} score={D.run_score(r):.4f}")
            runs.append(r)
        best = D.pick_best_run(runs)
        log(f"{tag}: best seed={best['seed']} (clean={best['clean']:.4f}, robust={best['robust']:.4f})")
        return best, [{k: v for k, v in r.items() if k != "model"} for r in runs]

    if a.export_only:  # redo QAT + export from the saved promoted student, no retraining
        prev = json.loads((outputs / "training_summary.json").read_text(encoding="utf-8"))
        summary = {**prev, "args": vars(a), "export_only_rerun": True}
        nc = int(prev["num_channels"])
        tr, va, vr = datasets(nc)
        chosen = StudentSTGCN(ncls, nc)
        chosen.load_state_dict(torch.load(out / "student_best.pt", map_location="cpu"))
        log("export-only: loaded weights/student_best.pt, skipping ablation/teacher/students")
    else:
        # ---- 1. ablation: 7-ch (with z) vs 6-ch (without) ---------------------------------------------
        if a.num_channels:
            nc = a.num_channels
            summary["ablation"] = {"skipped": f"--num_channels {nc} forced"}
        else:
            abl = {}
            for c in (7, 6):
                tr, va, vr = datasets(c)
                _, _, best = train_student(c, ncls, tr, va, mode="nodistill", teacher=None, epochs=a.ablation_epochs,
                                           lr=a.student_lr, device=device, log=log, tag=f"ablation-{c}ch",
                                           mix_prob=a.mix_prob)
                abl[f"{c}ch"] = {"val_top1": best["top1"], "val_top5": best["top5"]}
                log(f"ablation {c}ch: val top1={best['top1']:.4f} top5={best['top5']:.4f}")
            nc = 7 if abl["7ch"]["val_top1"] >= abl["6ch"]["val_top1"] else 6  # keep z only if it does not hurt
            abl["chosen"] = nc
            summary["ablation"] = abl
            log(f"ablation result -> num_channels={nc}")
        summary["num_channels"] = nc
        tr, va, vr = datasets(nc)

        # ---- 2/3. teacher + students -----------------------------------------------------------------
        cutoff_hit = D.cutoff_passed(a.cutoff)
        use_distill = not (a.no_distill or cutoff_hit)
        if not use_distill:
            log(f"FALLBACK FIRED: distillation skipped (--no_distill={a.no_distill}, cutoff passed={cutoff_hit})")
        summary["distillation_skipped"] = not use_distill

        teacher = None
        if use_distill:
            seed_everything(42)
            teacher = DualStreamTeacherGCN(ncls, nc)
            summary["teacher_params"] = count_params(teacher)
            state, hist, best = D.train_model(teacher, tr, va, epochs=a.teacher_epochs, lr=a.teacher_lr, mode="ce",
                                              device=device, log=log, tag="teacher")
            teacher.load_state_dict(state)
            torch.save(state, out / "teacher_best.pt")
            teacher = teacher.to(device).eval()
            summary["teacher"] = {"val_top1": best["top1"], "val_top5": best["top5"], "params": summary["teacher_params"]}
            log(f"teacher best val: top1={best['top1']:.4f} top5={best['top5']:.4f}")

        base_run, base_runs = run_seeds("nodistill", None, "student-baseline", tr, va, vr, nc)
        summary["student_baseline"] = {"val_top1": base_run["clean"], "robust_val_top1": base_run["robust"],
                                       "seed": base_run["seed"], "all_seeds": base_runs}
        chosen, chosen_name = base_run["model"], "baseline"
        if use_distill:
            dist_run, dist_runs = run_seeds("kd", teacher, "student-magkd", tr, va, vr, nc)
            summary["student_distilled"] = {"val_top1": dist_run["clean"], "robust_val_top1": dist_run["robust"],
                                            "seed": dist_run["seed"], "all_seeds": dist_runs}
            chosen_name = D.choose_student(dist_run["clean"], base_run["clean"], a.teacher_min_gain)
            chosen = dist_run["model"] if chosen_name == "distilled" else base_run["model"]
            log(f"promotion: distilled={dist_run['clean']:.4f} baseline={base_run['clean']:.4f} "
                f"gain={dist_run['clean'] - base_run['clean']:+.4f} (need >= {a.teacher_min_gain}) -> {chosen_name}")
        summary["promoted_student"] = chosen_name
        summary["student_params"] = count_params(chosen)
        torch.save(chosen.state_dict(), out / "student_best.pt")
        fp32_val = D.evaluate(chosen, DataLoader(va, batch_size=128))
        summary["student_fp32_val"] = fp32_val

    # ---- 5. QAT ----------------------------------------------------------------------------------
    fused = Q.fuse_student(chosen.cpu().eval())

    def ev(m):
        return (D.evaluate(m, DataLoader(va, batch_size=128))["top1"], D.evaluate(m, DataLoader(vr, batch_size=128))["top1"])

    pre_c, pre_r = ev(fused)
    seed_everything(42)  # QAT starts from a fixed RNG state so a re-export reproduces
    qat_ds = D.KeypointDataset(tr_rows, "train", cache, nc, train=True, dark_dir=dark, p_dark=a.p_dark if dark else 0.0,
                               gentle=a.gentle_qat)
    qat_loader = DataLoader(qat_ds, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(42))
    log(f"QAT (WITH online augmentation: joint/hand dropout, tilt, jitter, time-warp, low-light landmarks): "
        f"{a.qat_epochs} epochs, lr={cfg['qat_lr']} | before: clean={pre_c:.4f} robust={pre_r:.4f}")
    qat_model = Q.qat_finetune(fused, qat_loader, epochs=a.qat_epochs, lr=float(cfg["qat_lr"]), epoch_hook=qat_ds.set_epoch)
    torch.save(qat_model.state_dict(), out / "student_qat.pt")
    post_c, post_r = ev(qat_model)
    use_qat = D.run_score({"clean": post_c, "robust": post_r}) >= D.run_score({"clean": pre_c, "robust": pre_r}) - 0.005
    export_model = qat_model if use_qat else fused
    summary["qat"] = {"epochs": a.qat_epochs, "augmented": True, "val_top1_before": pre_c, "val_top1_after": post_c,
                      "robust_val_top1_before": pre_r, "robust_val_top1_after": post_r,
                      "weights_used_for_export": "qat" if use_qat else "pre-qat"}
    log(f"QAT: clean {pre_c:.4f} -> {post_c:.4f}, robust {pre_r:.4f} -> {post_r:.4f}; "
        f"exporting {'QAT' if use_qat else 'pre-QAT'} weights")

    # ---- 6. export + INT8 accuracy ---------------------------------------------------------------
    calib = np.stack([tr_eval[0].numpy() for tr_eval in
                      D.KeypointDataset(tr_rows[: a.calib_samples], "train", cache, nc, train=False)]).astype(np.float32)
    summary["calibration"] = {"source": "real training clips (unaugmented)", "samples": int(len(calib))}
    fp32_onnx = Q.export_onnx_fp32(export_model, out / "student_fp32.onnx", nc)
    int8_path = Q.quantize_onnx_int8(fp32_onnx, out / "student_int8.onnx", calib)
    int8_val = onnx_val_metrics(int8_path, va)
    fp32_final = D.evaluate(export_model, DataLoader(va, batch_size=128))
    drop = fp32_final["top1"] - int8_val["top1"]
    summary["export"] = {"onnx_int8_mb": int8_path.stat().st_size / 1e6, "fp32_val_top1": fp32_final["top1"],
                         "int8_onnx_val_top1": int8_val["top1"], "int8_onnx_val_top5": int8_val["top5"],
                         "int8_top1_drop_pts": 100.0 * drop, "gate_drop_le_1pt": bool(drop <= 0.01 + 1e-9)}
    val_x = np.stack([va[i][0].numpy() for i in range(len(va))])
    summary["export"]["onnx_vs_fp32"] = Q.compare_onnx(fp32_onnx, int8_path, val_x[:64])
    summary["export"]["int8_onnx_robust_val_top1"] = onnx_val_metrics(int8_path, vr)["top1"]
    if not a.skip_tflite:
        try:
            tfl = Q.export_tflite_int8(export_model, out / "student_int8.tflite", calib, nc, test_x=val_x[:64],
                                   val_x=val_x, val_y=np.array(va.labels))
            summary["export"]["tflite"] = tfl
        except Exception as e:  # reported, never hidden (CLAUDE.md 2.5)
            summary["export"]["tflite"] = f"NOT EXPORTED: {type(e).__name__}: {e}"
    log(f"export: INT8 ONNX {summary['export']['onnx_int8_mb']:.4f} MB, val top1 FP32={fp32_final['top1']:.4f} "
        f"INT8={int8_val['top1']:.4f} (drop {100 * drop:+.2f} pts)")

    if not a.no_update_config and cfg["num_channels"] != nc:
        set_config_channels(nc)
        log(f"config.yaml num_channels -> {nc}")
    summary["elapsed_s"] = time.time() - t_start
    (outputs / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"=== run done in {summary['elapsed_s']:.0f}s")
    return summary


if __name__ == "__main__":
    main()
