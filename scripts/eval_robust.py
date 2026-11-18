"""Score ANY INT8 ONNX on the test split: clean + the realistic robustness variants (same as benchmark.py).

    python scripts/eval_robust.py --onnx weights/prev_run1/student_int8.onnx --out outputs/run1_realistic_robustness.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import benchmark as B  # noqa: E402
from src import predict as P  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--manifest", default="data/test_manifest.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P.configure(onnx_path=ROOT / a.onnx if not Path(a.onnx).is_absolute() else a.onnx)
    rows = B.read_manifest(ROOT / a.manifest)
    res = B.run_robustness(rows, ROOT / "data/cache")
    res["onnx"] = a.onnx
    Path(ROOT / a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    t = res["top1"]
    print(f"{a.onnx}: clean={t['clean']:.4f} low_light={t['low_light']:.4f} combined={t['all_combined']:.4f} "
          f"drop_combined={res['drop_pts']['all_combined']:.2f} pts diag_joint_drop={res['drop_pts']['diag_single_joint_dropout_15pct']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
