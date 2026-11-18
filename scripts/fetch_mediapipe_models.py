"""DEV-TIME helper: download the two MediaPipe .task bundles into ./weights/.

Allowed only because it lives in scripts/ (CLAUDE.md 2.4). src/ never imports this and never
downloads; src/interpolator.py raises FileNotFoundError if the bundles are absent.
Official source: Google's public MediaPipe model storage. Prints SHA-256 so the files can be pinned.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://storage.googleapis.com/mediapipe-models"
BUNDLES = {
    "hand_landmarker.task": f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def main() -> int:
    dest = ROOT / "weights"
    dest.mkdir(exist_ok=True)
    for name, url in BUNDLES.items():
        target = dest / name
        if target.exists():
            print(f"exists: {target}")
        else:
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, target)
        data = target.read_bytes()
        print(f"  {name}: {len(data) / 1e6:.2f} MB  sha256={hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
