"""Multi-process MediaPipe Tasks keypoint extraction -> (T, 53, 3) float32 .npy per clip.

    python scripts/gcp_extract.py --splits train val test --workers 16
    python scripts/gcp_extract.py --splits train val --limit-per-class 3      # small subset

Input: data/{split}_manifest.csv (from scripts/prepare_data.py). A row either has a `frames_dir` of
already-sparse frames (WLASL-100 archive, 16 frames/clip: every frame is a keyframe, IMAGE mode) or a
`video_path` (decoded with the k=3 frame-skip extractor + interpolation).
Output: data/cache/{split}/{clip_id}.npy and data/cache/{split}_index.csv.
Deterministic: rows are processed and logged in manifest order (imap, not imap_unordered); the
extractor is stateless per clip, so the same inputs give the same arrays on any worker count.
Runs on Windows and Linux (GCP); never touches the network.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_EX = None  # per-worker extractor


def read_manifest(split: str) -> list[dict]:
    with open(ROOT / "data" / f"{split}_manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_rows(rows: list[dict], limit_per_class: int | None) -> list[dict]:
    """Manifest order preserved; keep the first `limit_per_class` clips of each gloss."""
    if not limit_per_class:
        return rows
    seen: dict[str, int] = defaultdict(int)
    out = []
    for r in rows:
        if seen[r["gloss"]] < limit_per_class:
            seen[r["gloss"]] += 1
            out.append(r)
    return out


def _frame_paths(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))


def extract_clip(extractor, row: dict, dark: bool = False) -> np.ndarray:
    """dark=True: landmarks from LOW-LIGHT frames (src.perturb.low_light_frame), used as training augmentation."""
    if row.get("frames_dir"):
        paths = _frame_paths(ROOT / row["frames_dir"])
        if dark:
            import cv2

            from src.perturb import low_light_frame

            rng = np.random.default_rng([42, 7, int(row["clip_id"])])
            return extractor.extract_arrays([low_light_frame(cv2.imread(str(p)), rng) for p in paths])
        return extractor.extract_frames(paths)
    return extractor.extract(str(ROOT / row["video_path"]))


def _init_worker(image_mode: bool) -> None:
    global _EX
    from src.interpolator import LandmarkExtractor

    _EX = LandmarkExtractor(image_mode=image_mode)
    _EX.load()


def _work(args: tuple[str, dict, Path, bool, bool]):
    split, row, out_dir, force, dark = args
    path = out_dir / split / f"{row['clip_id']}.npy"
    if path.exists() and not force:
        return row["clip_id"], int(np.load(path, mmap_mode="r").shape[0]), 0
    arr = extract_clip(_EX, row, dark).astype(np.float32)
    np.save(path, arr)
    hands = int(np.any(arr[:, :42] != 0))
    return row["clip_id"], int(arr.shape[0]), hands


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dark", action="store_true", help="extract from low-light frames (use with --out data/cache_dark)")
    a = ap.parse_args(argv)

    out_dir = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    first = read_manifest(a.splits[0])[0]
    image_mode = bool(first.get("frames_dir"))
    ctx = mp.get_context("spawn")  # same behaviour on Windows and Linux
    t_all = time.time()
    for split in a.splits:
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        rows = select_rows(read_manifest(split), a.limit_per_class)
        print(f"[{split}] {len(rows)} clips, workers={a.workers}, image_mode={image_mode}", flush=True)
        t0, no_hand, index = time.time(), 0, []
        with ctx.Pool(a.workers, initializer=_init_worker, initargs=(image_mode,)) as pool:
            tasks = [(split, r, out_dir, a.force, a.dark) for r in rows]
            for i, (cid, t, h) in enumerate(pool.imap(_work, tasks, chunksize=4), 1):
                index.append((cid, t))
                no_hand += int(h == 0 and t > 0)
                if i % a.log_every == 0 or i == len(rows):
                    print(f"[{split}] {i}/{len(rows)} last={cid} elapsed={time.time() - t0:.0f}s", flush=True)
        with open(out_dir / f"{split}_index.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "n_frames"])
            w.writerows(index)
        print(f"[{split}] done; clips with no hand in any frame (fresh runs only): {no_hand}", flush=True)
    print(f"total {time.time() - t_all:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
