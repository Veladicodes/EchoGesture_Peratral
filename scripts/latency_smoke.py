"""Latency smoke test on the random-weight export (project spec Build Order step 6).

Not the final benchmark (that is scripts/benchmark.py: 500 iterations, robustness, thermal).
Every number is measured here, on this machine, and labelled with what it covers:

  model_only_onnx   : onnxruntime INT8 session.run on a (1,C,64,53) tensor
  model_only_tflite : TFLite interpreter (XNNPACK) on the INT8 tflite, if tensorflow is importable
  post_landmark     : build_features -> model -> postprocess  (predict_from_joints)
  full_predict      : predict(video) incl. decode + MediaPipe landmarks, on a BLANK synthetic clip
                      (no hands => proxy only; hand-present frames cost more, not measured)
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import predict as P  # noqa: E402
from src.common import load_config, repo_path, seed_everything  # noqa: E402
from src.features import NUM_CHANNELS  # noqa: E402
from src.quantize import _synthetic_joints, synthetic_calibration  # noqa: E402


def stats(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {"n": len(a), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}


def timed(fn, n: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000.0)
    return stats(out)


def write_blank_clip(path: Path, frames: int, size=(640, 480)) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, size)
    blank = np.zeros((size[1], size[0], 3), np.uint8)
    for _ in range(frames):
        w.write(blank)
    w.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--frames", type=int, default=64, help="frames in the synthetic clip")
    args = ap.parse_args()

    seed_everything(42)
    cfg = load_config()
    result: dict = {
        "label": "random-weight smoke test (NOT a final benchmark)",
        "device": f"{platform.processor() or platform.machine()} | {platform.system()} {platform.release()} (proxy: laptop CPU)",
        "python": platform.python_version(), "ort_threads": cfg.get("ort_threads", 1),
        "frame_skip": cfg["frame_skip"], "num_channels": NUM_CHANNELS,
        "iters": args.iters, "warmup": args.warmup,
    }

    import onnxruntime as ort

    onnx_path = repo_path(cfg["paths"]["student_onnx"])
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = int(cfg.get("ort_threads", 1)), 1
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    x = synthetic_calibration(1, 3, NUM_CHANNELS)
    result["model_only_onnx"] = timed(lambda: sess.run(None, {"features": x}), args.iters, args.warmup)

    tfl = repo_path(cfg["paths"]["student_tflite"])
    try:
        import tensorflow as tf

        it = tf.lite.Interpreter(model_path=str(tfl), num_threads=1)
        it.allocate_tensors()
        i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
        xt = np.transpose(x, (0, 2, 3, 1)).copy()

        def run_tfl():
            it.set_tensor(i, xt)
            it.invoke()
            it.get_tensor(o)

        result["model_only_tflite_x86"] = timed(run_tfl, args.iters, args.warmup)
    except Exception as e:  # report, never hide
        result["model_only_tflite_x86"] = f"not measured: {type(e).__name__}: {e}"

    rng = np.random.default_rng(0)
    joints = _synthetic_joints(rng)
    P.reset()
    result["post_landmark"] = timed(lambda: P.predict_from_joints(joints, "x"), args.iters, args.warmup)

    bundles = [repo_path(cfg["paths"]["hand_task"]), repo_path(cfg["paths"]["pose_task"])]
    if all(b.is_file() for b in bundles):
        with tempfile.TemporaryDirectory() as d:
            clip = Path(d) / "blank_clip.avi"
            write_blank_clip(clip, args.frames)
            P.reset()
            P.predict(str(clip))  # loads singletons (excluded from timing)
            result["full_predict"] = timed(lambda: P.predict(str(clip)), args.iters, args.warmup)
            result["full_predict"]["clip"] = f"{args.frames} blank frames 640x480 @30fps, no hands (proxy)"
            result["full_predict"]["landmark_inferences_per_clip"] = -(-args.frames // int(cfg["frame_skip"]))
    else:
        result["full_predict"] = "not measured: MediaPipe .task bundles missing (run scripts/fetch_mediapipe_models.py)"

    out = repo_path(cfg["paths"]["outputs"]) / "latency_smoke.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
