"""Re-quantize the EXISTING FP32 student ONNX (no retraining, weights untouched) and check the 1-pt validation gate.

    python scripts/requant_onnx.py                       # percentile 99.999, 600 real train calibration clips
    python scripts/requant_onnx.py --method entropy --exclude /fc/Conv --out weights/try.onnx

Sweep result that chose the default (validation split, FP32 = 59.94 %, see outputs/requant_onnx.json for the shipped one):
percentile 99.999 / 200 clips = -1.48 pts (old), keep /fc/Conv FP32 = -2.08, Entropy = -59, percentile 99.999 / 600 clips = -0.30.
The sweep is tuned on the validation split, so a pass here is within about +-1 pt (3 clips) of noise; the test split is the
unbiased number and is reported by scripts/benchmark.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import distill as D  # noqa: E402
from src import quantize as Q  # noqa: E402
from src.common import get_num_channels, load_config, seed_everything  # noqa: E402


def rows(split: str) -> list[dict]:
    with open(ROOT / "data" / f"{split}_manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def val_top1(onnx_path: Path, x: np.ndarray, y: np.ndarray) -> float:
    import onnxruntime as ort

    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    pred = [int(s.run(None, {"features": x[i : i + 1]})[0][0].argmax()) for i in range(len(x))]
    return float((np.array(pred) == y).mean())


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default="weights/student_fp32.onnx")
    ap.add_argument("--out", default=cfg["paths"]["student_onnx"])
    ap.add_argument("--method", default="percentile", choices=["percentile", "entropy", "minmax"])
    ap.add_argument("--percentile", type=float, default=99.999)
    ap.add_argument("--calib", type=int, default=int(cfg.get("onnx_calib_samples", 600)))
    ap.add_argument("--exclude", nargs="*", default=[], help="ONNX node names kept in FP32 (e.g. /fc/Conv)")
    ap.add_argument("--no-write-summary", action="store_true")
    a = ap.parse_args()

    seed_everything(42)
    nc = get_num_channels()
    tr, va = rows("train"), rows("val")
    cal_ds = D.KeypointDataset(tr[: a.calib], "train", ROOT / "data/cache", nc, train=False)
    calib = np.stack([cal_ds[i][0].numpy() for i in range(len(cal_ds))]).astype(np.float32)
    v_ds = D.KeypointDataset(va, "val", ROOT / "data/cache", nc, train=False)
    vx = np.stack([v_ds[i][0].numpy() for i in range(len(v_ds))]).astype(np.float32)
    vy = np.array([int(v_ds[i][1]) for i in range(len(v_ds))])

    fp32 = ROOT / a.fp32
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    Q.quantize_onnx_int8(fp32, out, calib, method=a.method, percentile=a.percentile, nodes_to_exclude=a.exclude)
    fp, q = val_top1(fp32, vx, vy), val_top1(out, vx, vy)
    drop = 100.0 * (fp - q)
    report = {"method": a.method, "percentile": a.percentile, "calib_clips": int(len(calib)), "excluded_nodes": a.exclude,
              "val_top1_fp32": fp, "val_top1_int8_onnx": q, "drop_pts": drop, "gate_drop_le_1pt": bool(drop <= 1.0),
              "size_mb": out.stat().st_size / 1e6, "size_lt_0_5_mb": bool(out.stat().st_size / 1e6 < 0.5),
              "note": "hyper-parameters were chosen on the validation split; treat the pass as +-1 pt noise"}
    (ROOT / "outputs" / "requant_onnx.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summ = ROOT / "outputs" / "training_summary.json"
    if not a.no_write_summary and summ.is_file():
        s = json.loads(summ.read_text(encoding="utf-8"))
        e = s.setdefault("export", {})
        e.update(int8_onnx_val_top1=q, fp32_val_top1=fp, int8_top1_drop_pts=drop, gate_drop_le_1pt=bool(drop <= 1.0),
                 onnx_int8_mb=out.stat().st_size / 1e6, onnx_requant=report)
        summ.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
