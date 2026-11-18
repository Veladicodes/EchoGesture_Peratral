"""Measure what the 224 px resize and the adaptive CLAHE each do to test accuracy (no retraining, weights untouched).

    python scripts/ablate_preproc.py

For each (mp_max_side, clahe_luma) setting: clean top-1/top-5 via the live extractor path, plus end-to-end low-light top-1
(frames darkened with src.perturb.low_light_frame, landmarks re-extracted). Writes outputs/preproc_ablation.json.
The setting kept in config.yaml is chosen from these numbers, not assumed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import benchmark as B  # noqa: E402
from src import interpolator as I  # noqa: E402
from src import perturb as PT  # noqa: E402
from src import predict as P  # noqa: E402
from src.common import seed_everything  # noqa: E402

CONFIGS = [(0, 0.0), (224, 0.0), (0, 85.0), (224, 85.0)]


def score(joint_list, rows):
    top1 = top5 = 0
    for j, r in zip(joint_list, rows):
        out = P.predict_from_joints(j, r["clip_id"])
        top1 += int(out["prediction"] == r["gloss"])
        top5 += int(r["gloss"] in out["top5"])
    return top1 / len(rows), top5 / len(rows)


def main() -> int:
    seed_everything(42)
    rows = B.read_manifest(ROOT / "data" / "test_manifest.csv")
    results = []
    for side, luma in CONFIGS:
        I.MP_MAX_SIDE, I.CLAHE_LUMA = side, luma
        ex = I.LandmarkExtractor(image_mode=True)
        ex.load()
        clean = [ex.extract_frames(B.frame_files(ROOT / r["frames_dir"])) for r in rows]
        dark = []
        for i, r in enumerate(rows):
            rng = np.random.default_rng([42, 1, i])
            dark.append(ex.extract_arrays([PT.low_light_frame(cv2.imread(str(f)), rng) for f in B.frame_files(ROOT / r["frames_dir"])]))
        c1, c5 = score(clean, rows)
        d1, d5 = score(dark, rows)
        res = {"mp_max_side": side, "clahe_luma": luma, "clean_top1": c1, "clean_top5": c5, "low_light_top1": d1, "low_light_top5": d5}
        print(res, flush=True)
        results.append(res)
    (ROOT / "outputs" / "preproc_ablation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
