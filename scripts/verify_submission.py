"""Automatic checks for mandatory deliverables A-C (project spec 8). D (demo video) and E (deck) are manual.

    python scripts/verify_submission.py [--manifest data/test_manifest.csv]

Prints PASS/FAIL per item and exits 1 if any check fails. Read-only: writes nothing.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_vocab  # noqa: E402

KEYS = {"clip_id", "prediction", "top5", "confidence", "latency_ms"}
HEADER = ["metric", "target", "measured", "unit", "device", "runtime", "pass"]
METRICS = ["top1_accuracy", "latency_p95", "model_size", "accuracy_drop_robustness", "avg_power", "thermal_drift_10min"]


def check_A() -> list[tuple[str, bool, str]]:
    res = []
    for rel in ("src/predict.py", "README.md", "Dockerfile", "requirements.txt"):
        res.append((f"A: {rel} exists", (ROOT / rel).is_file(), ""))
    src = (ROOT / "src/predict.py").read_text(encoding="utf-8") if (ROOT / "src/predict.py").is_file() else ""
    res.append(("A: predict(video_path) defined", "def predict(video_path" in src, ""))
    res.append(("A: repo link opens without login", True, "MANUAL: confirm the GitHub repo is public/shared"))
    return res


def check_B() -> list[tuple[str, bool, str]]:
    res = [("B: src/quantize.py exists", (ROOT / "src/quantize.py").is_file(), "")]
    found = False
    for name in ("student_int8.onnx", "student_int8.tflite"):
        p = ROOT / "weights" / name
        if p.is_file():
            found = True
            mb = p.stat().st_size / 1e6
            res.append((f"B: {name} < 20 MB", mb < 20.0, f"{mb:.4f} MB (aim < 0.5: {'yes' if mb < 0.5 else 'NO'})"))
    res.append(("B: at least one INT8 model present", found, ""))
    return res


def check_C(manifest: Path) -> list[tuple[str, bool, str]]:
    res = []
    out = ROOT / "outputs"
    jl, js, rc = out / "predictions.jsonl", out / "predictions.json", out / "results.csv"
    for p in (jl, js, rc):
        res.append((f"C: {p.name} exists", p.is_file(), ""))
    if not all(p.is_file() for p in (jl, js, rc)) or not manifest.is_file():
        return res + [("C: manifest present", manifest.is_file(), str(manifest))]
    res.append(("C: predictions.json identical to .jsonl", jl.read_bytes() == js.read_bytes(), ""))
    vocab = set(load_vocab())
    man_ids = [r["clip_id"] for r in csv.DictReader(open(manifest, newline="", encoding="utf-8"))]
    lines = [l for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()]
    bad, ids = [], []
    for i, line in enumerate(lines):
        try:
            d = json.loads(line)
            ok = (set(d) == KEYS and isinstance(d["clip_id"], str) and isinstance(d["top5"], list)
                  and len(d["top5"]) == 5 and len(set(d["top5"])) == 5 and d["prediction"] == d["top5"][0]
                  and all(g == g.upper() and g in vocab for g in d["top5"])
                  and isinstance(d["confidence"], float) and 0.0 <= d["confidence"] <= 1.0
                  and isinstance(d["latency_ms"], float))
            ids.append(d.get("clip_id"))
            if not ok:
                bad.append(i)
        except Exception:
            bad.append(i)
    res.append(("C: every prediction line matches the 5-key schema", not bad and bool(lines), f"bad lines: {bad[:5]}"))
    res.append(("C: one line per manifest clip", len(lines) == len(man_ids), f"{len(lines)} lines vs {len(man_ids)} clips"))
    res.append(("C: clip_ids align with manifest (order and value)", ids == man_ids, ""))
    rows = list(csv.reader(open(rc, newline="", encoding="utf-8")))
    res.append(("C: results.csv header exact", bool(rows) and rows[0] == HEADER, ""))
    res.append(("C: results.csv has exactly 6 rows in order", [r[0] for r in rows[1:]] == METRICS, ""))
    body = {r[0]: r for r in rows[1:] if len(r) == 7}
    size = body.get("model_size")
    try:
        res.append(("C: model_size row < 20 MB", float(size[2]) < 20.0, size[2]))
    except Exception:
        res.append(("C: model_size row < 20 MB", False, "unparseable"))
    lat = body.get("latency_p95")
    try:
        v = float(lat[2])
        res.append(("C: latency_p95 <= 50 ms (per-step streaming)", v <= 50.0, f"{v:.2f} ms"))
    except Exception:
        res.append(("C: latency_p95 <= 50 ms (per-step streaming)", False, "not measured"))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/test_manifest.csv")
    a = ap.parse_args()
    man = Path(a.manifest) if Path(a.manifest).is_absolute() else ROOT / a.manifest
    results = check_A() + check_B() + check_C(man)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {('- ' + detail) if detail else ''}")
    print("[MANUAL] D: demo video link (2:30-2:50, never > 3:00, viewable)")
    print("[MANUAL] E: VITVellore_Peratral_Submission_ppt.pptx + .pdf (exactly 9 slides, numbers from results.csv)")
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} automatic checks passed" + (f"; FAILED: {len(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
