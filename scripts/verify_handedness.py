"""Empirical handedness check on real archive clips (CLAUDE.md 5.1 warning).

For every frame with exactly ONE detected hand, compare the hand wrist (hand landmark 0) with the two
pose wrists (pose idx 15 = joint 51 "L-wrist", idx 16 = joint 52 "R-wrist"). The pose wrist that is
closer tells which anatomical side the hand really is, independent of MediaPipe's handedness label.
Then it reports, for each MediaPipe label, how often it sits next to pose-15 vs pose-16, and
whether src.interpolator.MIRRORED_INPUT puts those hands into slots 0-20 (L) / 21-41 (R) consistently.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import interpolator as I  # noqa: E402
from src.features import LEFT, RIGHT  # noqa: E402


def main(n_clips: int = 120) -> int:
    ex = I.LandmarkExtractor(image_mode=True)
    ex.load()
    rows = list(csv.DictReader(open(ROOT / "data" / "val_manifest.csv", encoding="utf-8")))
    rng = np.random.default_rng(42)
    rows = [rows[i] for i in rng.permutation(len(rows))[:n_clips]]

    tally: Counter = Counter()  # (mp_label, nearer_pose_wrist) -> frames
    slot_ok = slot_bad = 0
    for r in rows:
        import cv2

        for p in sorted((ROOT / r["frames_dir"]).glob("*.jpg"), key=lambda q: int(q.stem.rsplit("_", 1)[1])):
            img = ex._to_mp_image(cv2.imread(str(p)))
            hres, pres = ex._hand.detect(img), ex._pose.detect(img)
            if len(hres.hand_landmarks) != 1 or not pres.pose_landmarks:
                continue
            w = hres.hand_landmarks[0][0]
            pl = pres.pose_landmarks[0]
            d15 = np.hypot(w.x - pl[15].x, w.y - pl[15].y)
            d16 = np.hypot(w.x - pl[16].x, w.y - pl[16].y)
            label = hres.handedness[0][0].category_name
            near = "pose15(L-wrist)" if d15 < d16 else "pose16(R-wrist)"
            tally[(label, near)] += 1
            frame = I.assemble_frame(hres, pres)
            in_left = bool(np.any(frame[LEFT] != 0))
            slot_ok += int(in_left == (d15 < d16))
            slot_bad += int(in_left != (d15 < d16))
    print("clips sampled:", len(rows))
    for k, v in sorted(tally.items()):
        print(f"  MediaPipe label={k[0]:5s} nearest {k[1]}: {v} frames")
    tot = slot_ok + slot_bad
    print(f"MIRRORED_INPUT={I.MIRRORED_INPUT}: slot agrees with nearest pose wrist on {slot_ok}/{tot} frames "
          f"({100.0 * slot_ok / max(tot, 1):.1f}%)")
    print("RESULT:", "CONSISTENT" if tot and slot_ok / tot > 0.8 else "SWAPPED or inconclusive -> flip MIRRORED_INPUT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
