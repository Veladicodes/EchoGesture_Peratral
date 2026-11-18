"""Build data/vocab.json + data/{train,val,test}_manifest.csv from the unpacked WLASL-100 frame archive.

Layout on disk: data/raw/preprocessing/<split>/frames/<gloss>/<clip_id>/<gloss>_<n>.jpg  (16 frames/clip)
Classes are sorted alphabetically (deterministic). Manifest columns: clip_id,gloss,label,split,frames_dir,n_frames
(frames_dir is relative to the repo root, forward slashes).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "preprocessing"
SPLITS = ("train", "val", "test")


def main() -> int:
    rows = {s: [] for s in SPLITS}
    for s in SPLITS:
        base = RAW / s / "frames"
        for gdir in sorted(base.iterdir()):
            for cdir in sorted(gdir.iterdir()):
                n = len(list(cdir.glob("*.jpg")))
                rows[s].append((cdir.name, gdir.name.upper(), s, cdir.relative_to(ROOT).as_posix(), n))
    glosses = sorted({r[1] for s in SPLITS for r in rows[s]})
    idx = {g: i for i, g in enumerate(glosses)}
    ids = [r[0] for s in SPLITS for r in rows[s]]
    assert len(ids) == len(set(ids)), "clip ids are not globally unique"

    for s in SPLITS:
        with open(ROOT / "data" / f"{s}_manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "gloss", "label", "split", "frames_dir", "n_frames"])
            for cid, g, sp, d, n in sorted(rows[s]):
                w.writerow([cid, g, idx[g], sp, d, n])

    counts = {s: dict(sorted(Counter(r[1] for r in rows[s]).items())) for s in SPLITS}
    vocab = {
        "_proxy": True,
        "_note": "PROXY VOCABULARY - NOT SAMSUNG'S. WLASL-100 (pre-extracted frames, 16 per clip) from archive (1).zip; "
                 "official Track 02 starter kit not received. Class index = position in 'glosses' (alphabetical).",
        "source": "WLASL-100 frame archive (proxy)",
        "num_classes": len(glosses),
        "clip_counts_total": {s: len(rows[s]) for s in SPLITS},
        "glosses": glosses,
        "clip_counts": counts,
    }
    (ROOT / "data" / "vocab.json").write_text(json.dumps(vocab, indent=1), encoding="utf-8")
    print(f"classes={len(glosses)} clips={ {s: len(rows[s]) for s in SPLITS} }")
    print("frames/clip:", Counter(r[4] for s in SPLITS for r in rows[s]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
