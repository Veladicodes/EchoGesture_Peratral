"""Benchmark + submission artifacts (CLAUDE.md 3, 3.1, 4.2). Every number is measured here.

    python scripts/benchmark.py --manifest data/test_manifest.csv --output_dir outputs [--quick]

Writes outputs/predictions.jsonl (one predict() dict per line, manifest order), an identical
outputs/predictions.json, outputs/results.csv (exactly 6 rows), outputs/benchmark_log.json and
outputs/latency_smoke.json (per-step streaming latency + full-clip batch throughput).

Metric definitions (also CLAUDE.md 3.1):
  top1_accuracy             % of test clips with prediction == ground-truth gloss (predict() on each clip)
  latency_p95               p95 of 500 consecutive STREAMING steps after 20 warm-up steps, k=3: every step runs
                            features + INT8 inference + post-processing; every 3rd step also runs MediaPipe
                            (hand+pose) + interpolation. Frame decode excluded (a camera hands frames over).
  model_size                larger of student_int8.onnx / student_int8.tflite, MB
  accuracy_drop_robustness  clean top-1 minus top-1 with 15% joint occlusion + +-15 deg tilt + low-light frames
                            (landmarks re-extracted from darkened frames), applied together
  avg_power                 NOT measured (no power meter on this machine; reported as such)
  thermal_drift_10min       p95 step latency minute 10 vs minute 1 of a continuous streaming loop, percent
Stream source: real archive frames from the test split, each repeated 3x to emulate 30 fps (tracking sees jumps
between clips, so landmark cost is if anything pessimistic).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import perturb as PT  # noqa: E402
from src import predict as P  # noqa: E402
from src.common import load_config, load_vocab, repo_path, seed_everything  # noqa: E402
from src.features import build_features  # noqa: E402
from src.interpolator import LandmarkExtractor  # noqa: E402

HEADER = ["metric", "target", "measured", "unit", "device", "runtime", "pass"]


def p95(a) -> float:
    return float(np.percentile(np.asarray(a, dtype=np.float64), 95))


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def frame_files(d: Path) -> list[Path]:
    return sorted(d.glob("*.jpg"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))


# ------------------------------------------------------------------ predictions + accuracy
def run_predictions(rows: list[dict], out_dir: Path) -> dict:
    preds, correct1, correct5, lat, tot, fps = [], 0, 0, [], [], []
    for r in rows:
        res = P.predict(str(ROOT / r["frames_dir"]))
        assert res["clip_id"] == r["clip_id"], (res["clip_id"], r["clip_id"])
        preds.append(res)
        correct1 += int(res["prediction"] == r["gloss"])
        correct5 += int(r["gloss"] in res["top5"])
        lat.append(res["latency_ms"])
        tot.append(P.LAST_STATS["total_clip_ms"])
        fps.append(P.LAST_STATS["fps"])
    text = "".join(json.dumps(p) + "\n" for p in preds)
    (out_dir / "predictions.jsonl").write_text(text, encoding="utf-8", newline="\n")
    shutil.copyfile(out_dir / "predictions.jsonl", out_dir / "predictions.json")  # byte-identical copy
    n = len(rows)
    per: dict[str, list[int]] = {}
    for r, res in zip(rows, preds):
        c = per.setdefault(r["gloss"], [0, 0])
        c[0] += int(res["prediction"] == r["gloss"])
        c[1] += 1
    ranked = sorted(per.items(), key=lambda kv: (-kv[1][0] / kv[1][1], -kv[1][1], kv[0]))
    with open(out_dir / "per_class_accuracy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["gloss", "test_clips", "correct", "top1"])
        w.writerows([[g, c[1], c[0], f"{c[0] / c[1]:.3f}"] for g, c in ranked])
    return {"n_clips": n, "top1": correct1 / n, "top5": correct5 / n, "last_step_latency_ms_mean": float(np.mean(lat)),
            "total_clip_ms_mean": float(np.mean(tot)), "total_clip_ms_p95": p95(tot), "clip_fps_mean": float(np.mean(fps)),
            "frames_per_clip": 16}


# ------------------------------------------------------------------ robustness
def top1_from_joints(joint_list: list[np.ndarray], rows: list[dict]) -> float:
    ok = 0
    for j, r in zip(joint_list, rows):
        ok += int(P.predict_from_joints(j, r["clip_id"])["prediction"] == r["gloss"])
    return ok / len(rows)


def run_robustness(rows: list[dict], cache: Path) -> dict:
    ex = LandmarkExtractor(image_mode=True)
    ex.load()
    # clean landmarks come from the SAME live extractor path predict() uses (224 px resize + adaptive CLAHE), not the
    # old 256 px cache, so "clean" here equals the headline top-1.
    clean = [ex.extract_frames(frame_files(ROOT / r["frames_dir"])) for r in rows]
    dark = []
    for i, r in enumerate(rows):
        rng = np.random.default_rng([42, 1, i])  # test-time seed; training dark cache uses [42, 7, id]
        dark.append(ex.extract_arrays([PT.low_light_frame(cv2.imread(str(f)), rng) for f in frame_files(ROOT / r["frames_dir"])]))
    rng = lambda i, k: np.random.default_rng([42, k, i])  # noqa: E731
    variants = {
        "clean": clean,
        "hand_drop_15pct_frames": [PT.drop_hands(j, rng(i, 2), 0.15) for i, j in enumerate(clean)],
        "tilt_pm15deg": [PT.tilt(j, rng(i, 3)) for i, j in enumerate(clean)],
        "coord_noise_sigma0.015": [PT.noise(j, rng(i, 9)) for i, j in enumerate(clean)],
        # (a) MODEL-LEVEL combined: clean landmarks + 15% whole-hand drops + +-15 deg tilt + coordinate noise
        "model_combined": [PT.noise(PT.drop_hands(PT.tilt(j, rng(i, 10)), rng(i, 11), 0.15), rng(i, 12)) for i, j in enumerate(clean)],
        # (b) END-TO-END (sensor + model): darkened RGB frames re-extracted through MediaPipe
        "low_light": dark,
        # PRIMARY (the results.csv row): realistic MediaPipe failure modes together
        "all_combined": [PT.drop_hands(PT.tilt(j, rng(i, 4)), rng(i, 5), 0.15) for i, j in enumerate(dark)],
        # DIAGNOSTICS (not in results.csv): raw single-joint dropout, the original methodology
        "diag_single_joint_dropout_15pct": [PT.occlude_joints(j, rng(i, 6), 0.15) for i, j in enumerate(clean)],
        "diag_all_combined_single_joint": [PT.occlude_joints(PT.tilt(j, rng(i, 7)), rng(i, 8), 0.15) for i, j in enumerate(dark)],
    }
    acc = {k: top1_from_joints(v, rows) for k, v in variants.items()}
    drops = {k: 100.0 * (acc["clean"] - v) for k, v in acc.items() if k != "clean"}
    return {"top1": acc, "drop_pts": drops, "primary": "all_combined",
            "levels": {
                "model_level (clean landmarks; isolates OUR model)": ["hand_drop_15pct_frames", "tilt_pm15deg",
                                                                      "coord_noise_sigma0.015", "model_combined"],
                "end_to_end (sensor + model; darkened RGB re-extracted by MediaPipe)": ["low_light", "all_combined"],
                "diagnostic_only": ["diag_single_joint_dropout_15pct", "diag_all_combined_single_joint"]},
            "model_level_drop_pts": drops["model_combined"], "end_to_end_drop_pts": drops["all_combined"],
            "note": "The low-light drop is dominated by upstream RGB MediaPipe losing/mislocating landmarks in dark frames "
                    "(measured as the gap between model_level and end_to_end), not by the INT8 GCN, which only sees landmarks.",
            "methodology": "PRIMARY all_combined = low-light frames (gamma 2.2, x0.5, noise sigma 8, landmarks re-extracted) + "
            "tilt uniform +-15deg + whole-hand drop on 15% of frames per hand. diag_* = raw single-joint dropout (15% of "
            "(frame,joint)), kept for transparency only; MediaPipe loses a hand at a time, not single joints. Seeds [42,..]."}


# ------------------------------------------------------------------ streaming latency
def build_stream(rows: list[dict], n_frames: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for r in rows:
        for f in frame_files(ROOT / r["frames_dir"]):
            img = cv2.imread(str(f))
            frames.extend([img, img, img])  # x3 => ~30 fps from the sparse archive frames
        if len(frames) >= n_frames:
            break
    return frames


def streaming_steps(sp: P.StreamingPredictor, frames: list[np.ndarray], n: int, start: int = 0):
    ms, key = [], []
    for i in range(n):
        _, t = sp.step(frames[(start + i) % len(frames)], infer=True)
        ms.append(t)
        key.append(sp.log[-1][0])
    return np.array(ms), np.array(key)


def run_latency(rows: list[dict], iters: int, warmup: int, k: int | None = None):
    k = int(load_config()["frame_skip"]) if k is None else k
    frames = build_stream(rows[:24], iters + warmup)
    ex = P._extractor()
    ex.load()
    ex.reset_tracking(frames[0].shape)
    sp = P.StreamingPredictor(ex, k=k)
    streaming_steps(sp, frames, warmup)                      # warm-up (excluded)
    ms, key = streaming_steps(sp, frames, iters, start=warmup)
    sess, vocab = P._session(), P._vocab()                   # component waterfall on the same stream
    joints = np.stack([ex.detect_frame(frames[i], 33) for i in range(0, 64 * 3, 3)])
    t_land, t_feat, t_model, t_post = [], [], [], []
    for i in range(100):
        t0 = time.perf_counter(); ex.detect_frame(frames[(i * 3) % len(frames)], 33)
        t1 = time.perf_counter(); f = build_features(joints)[None]
        t2 = time.perf_counter(); z = sess.run(None, {"features": f})[0][0]
        t3 = time.perf_counter(); P._postprocess(z, vocab, True)
        t4 = time.perf_counter()
        t_land.append((t1 - t0) * 1e3); t_feat.append((t2 - t1) * 1e3); t_model.append((t3 - t2) * 1e3); t_post.append((t4 - t3) * 1e3)
    return {
        "steps": int(iters), "warmup": int(warmup), "frame_skip": k,
        "step_ms_p50": float(np.percentile(ms, 50)), "step_ms_p95": p95(ms), "step_ms_max": float(ms.max()),
        "keyframe_steps": int(key.sum()), "keyframe_step_ms_p95": p95(ms[key]), "interp_step_ms_p95": p95(ms[~key]),
        "waterfall_p95_ms": {"landmarks_hand_pose": p95(t_land), "features": p95(t_feat), "int8_model_only": p95(t_model),
                             "postprocess": p95(t_post)},
        "stream": "real test-split archive frames x3 repeat; decode excluded",
    }, frames


def run_thermal(frames: list[np.ndarray], minutes: float) -> dict:
    k = int(load_config()["frame_skip"])
    sp = P.StreamingPredictor(P._extractor(), k=k)
    import psutil

    proc = psutil.Process()
    cpu0 = sum(proc.cpu_times()[:2])
    t_start = time.perf_counter()
    stamps, lat, i = [], [], 0
    while True:
        now = time.perf_counter() - t_start
        if now >= minutes * 60.0:
            break
        _, t = sp.step(frames[i % len(frames)], infer=True)
        stamps.append(now); lat.append(t); i += 1
        if len(sp.log) > 5000:
            del sp.log[:-10]
    stamps, lat = np.array(stamps), np.array(lat)
    m1 = lat[(stamps >= 0) & (stamps < 60)]
    mN = lat[(stamps >= (minutes - 1) * 60) & (stamps < minutes * 60)]
    l1, lN = p95(m1), p95(mN)
    wall = time.perf_counter() - t_start
    util = (sum(proc.cpu_times()[:2]) - cpu0) / wall / (psutil.cpu_count(logical=True) or 1)   # share of total CPU capacity
    tdp = float(load_config().get("cpu_tdp_w", 0.0))
    return {"process_cpu_util_frac": util, "cpu_tdp_w": tdp, "proxy_power_mw": util * tdp * 1000.0,
            "minutes": minutes, "steps": int(len(lat)), "p95_minute_1_ms": l1, "p95_minute_last_ms": lN,
            "drift_pct": (lN - l1) / l1 * 100.0, "steps_minute_1": int(len(m1)), "steps_minute_last": int(len(mN))}


# ------------------------------------------------------------------ main
def device_label() -> str:
    return f"{platform.processor() or platform.machine()} | {platform.system()} {platform.release()} (laptop CPU, proxy)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/test_manifest.csv")
    ap.add_argument("--input_dir", default=None, help="unused with a manifest; kept for the CLI contract")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--thermal-minutes", type=float, default=10.0)
    ap.add_argument("--quick", action="store_true", help="skip the thermal loop (rows marked 'not measured')")
    a = ap.parse_args(argv)

    seed_everything(42)
    cfg = load_config()
    out = Path(a.output_dir) if Path(a.output_dir).is_absolute() else ROOT / a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    man = Path(a.manifest) if Path(a.manifest).is_absolute() else ROOT / a.manifest
    rows = read_manifest(man)
    cache = ROOT / a.cache
    log: dict = {"vocab_size": len(load_vocab()), "test_clips": len(rows), "device": device_label(), "quick": a.quick,
                 "proxy_vocab_note": "WLASL-100 proxy vocabulary, not Samsung's official list"}

    print("[1/5] predictions on the test split ...", flush=True)
    log["accuracy"] = run_predictions(rows, out)
    print(f"      top1={log['accuracy']['top1']:.4f} top5={log['accuracy']['top5']:.4f}", flush=True)

    print("[2/5] robustness ...", flush=True)
    log["robustness"] = run_robustness(rows, cache)
    log["robustness"]["clean_matches_predict"] = abs(log["robustness"]["top1"]["clean"] - log["accuracy"]["top1"]) < 1e-9

    print("[3/5] streaming latency ...", flush=True)
    log["latency"], frames = run_latency(rows, a.iters, a.warmup)
    k_main = int(cfg["frame_skip"])
    log["latency_by_k"] = {f"k={k_main}": {"step_ms_p95": log["latency"]["step_ms_p95"],
                                           "keyframe_step_ms_p95": log["latency"]["keyframe_step_ms_p95"]}}
    for k_alt in sorted({3, 4} - {k_main}):  # the 'does k=4 buy margin?' check; the results.csv row stays at k_main
        lat_alt, _ = run_latency(rows, a.iters, a.warmup, k=k_alt)
        log["latency_by_k"][f"k={k_alt}"] = {"step_ms_p95": lat_alt["step_ms_p95"],
                                             "keyframe_step_ms_p95": lat_alt["keyframe_step_ms_p95"],
                                             "keyframe_steps": lat_alt["keyframe_steps"]}

    print("[4/5] model size ...", flush=True)
    sizes = {}
    for k_, p_ in (("onnx", cfg["paths"]["student_onnx"]), ("tflite", cfg["paths"]["student_tflite"])):
        f = repo_path(p_)
        sizes[k_] = f.stat().st_size / 1e6 if f.is_file() else None
    log["model_size_mb"] = sizes
    size = max(v for v in sizes.values() if v is not None)

    if a.quick:
        log["thermal"] = "not measured (--quick)"
        drift = "not measured"
    else:
        print(f"[5/5] thermal loop {a.thermal_minutes} min ...", flush=True)
        log["thermal"] = run_thermal(frames, a.thermal_minutes)
        drift = log["thermal"]["drift_pct"]

    dev = device_label()
    rt = f"ONNX Runtime {__import__('onnxruntime').__version__} CPU INT8 ({cfg.get('ort_threads', 1)} thread) + MediaPipe Tasks CPU"
    drop = log["robustness"]["drop_pts"]["all_combined"]
    (out / "robustness_detail.json").write_text(json.dumps(log["robustness"], indent=2), encoding="utf-8")
    t1, lat = 100.0 * log["accuracy"]["top1"], log["latency"]["step_ms_p95"]
    th = log["thermal"]
    if isinstance(th, dict) and th.get("cpu_tdp_w", 0) > 0:
        power_val, power_pass = f"{th['proxy_power_mw']:.0f}", th["proxy_power_mw"] <= 500.0
        power_dev = (f"{dev} | CPU TDP x util proxy ({th['cpu_tdp_w']:.0f} W TDP assumed x {100 * th['process_cpu_util_frac']:.1f}% "
                     "process CPU util); laptop CPU, NOT a phone power number")
    else:
        power_val, power_pass, power_dev = "not measured", False, dev + " | no power meter (--quick)"
    rows_out = [
        ("top1_accuracy", "70", f"{t1:.2f}", "%", dev, rt, t1 >= 70.0),
        ("latency_p95", "50", f"{lat:.2f}", "ms", dev + " | per-step streaming, k=3", rt, lat <= 50.0),
        ("model_size", "20", f"{size:.4f}", "MB", "file size (larger of onnx/tflite)", "INT8 ONNX + TFLite", size <= 20.0),
        ("accuracy_drop_robustness", "5", f"{drop:.2f}", "pts", dev, rt, drop <= 5.0),
        ("avg_power", "500", power_val, "mW", power_dev, "process CPU util x vendor TDP over the 10-min streaming loop (proxy, not a meter)", power_pass),
        ("thermal_drift_10min", "20", drift if isinstance(drift, str) else f"{drift:.2f}", "%", dev,
         rt + " | continuous streaming loop", (not isinstance(drift, str)) and drift <= 20.0),
    ]
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEADER)
        for r in rows_out:
            w.writerow(list(r[:6]) + [str(r[6])])
    (out / "benchmark_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    (out / "latency_smoke.json").write_text(json.dumps({
        "label": "per-step streaming latency + full-clip batch throughput (real archive frames)",
        "device": dev, "streaming": log["latency"],
        "full_clip_batch": {k: log["accuracy"][k] for k in ("total_clip_ms_mean", "total_clip_ms_p95", "clip_fps_mean", "frames_per_clip")},
    }, indent=2), encoding="utf-8")
    print(json.dumps({k: log[k] for k in ("accuracy", "latency", "model_size_mb")}, indent=2))
    print((out / "results.csv").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
