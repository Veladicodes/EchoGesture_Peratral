"""Real-phone ARM64 model-only benchmark (CLAUDE.md 5.4 item 6 / 17). Dev tool: never imported by src/.

    python scripts/arm_benchmark.py --adb "C:/Users/Adith/Downloads/platform-tools-latest-windows/platform-tools/adb.exe" \
        --benchmark-model path/to/android_aarch64_benchmark_model

What it does: checks `adb devices`, reads the device's OWN identity (never assumed), pushes the INT8 TFLite and the
arm64-v8a benchmark_model, runs `benchmark_model --use_xnnpack=true --num_runs=500 --warmup_runs=20` N times, and writes
everything (raw output included) to outputs/arm_benchmark.txt. MODEL-ONLY latency: no MediaPipe, no video decode, so it
never replaces the end-to-end latency_p95 row.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/data/local/tmp"
PROPS = ["ro.product.manufacturer", "ro.product.model", "ro.soc.manufacturer", "ro.soc.model", "ro.board.platform",
         "ro.hardware", "ro.product.cpu.abi", "ro.build.version.release"]


def sh(adb: str, *args: str, check: bool = True) -> str:
    r = subprocess.run([adb, *args], capture_output=True, text=True, timeout=600)
    if check and r.returncode != 0:
        raise SystemExit(f"[arm] adb {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return (r.stdout + r.stderr).strip()


def parse(out: str) -> dict:
    """Pull the headline numbers from benchmark_model output (format differs slightly by TFLite version)."""
    d: dict = {}
    tail = out[out.rfind("Running benchmark") :] if "Running benchmark" in out else out
    m = re.search(r"count=(\d+)\s+first=(\d+)\s+curr=(\d+)\s+min=(\d+)\s+max=(\d+)\s+avg=([\d.]+)\s+std=(\d+)"
                  r"(?:\s+p5=(\d+)\s+median=(\d+)\s+p95=(\d+))?", tail)
    if m:
        d.update(count=int(m[1]), min_us=int(m[4]), max_us=int(m[5]), avg_us=float(m[6]), std_us=int(m[7]))
        if m[10]:
            d.update(p5_us=int(m[8]), median_us=int(m[9]), p95_us=int(m[10]))
    m = re.search(r"Inference \(avg\):\s*([\d.]+)", out)
    if m and "avg_us" not in d:
        d["avg_us"] = float(m.group(1))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adb", default="adb")
    ap.add_argument("--benchmark-model", required=True, help="Android arm64-v8a benchmark_model binary (official TFLite build)")
    ap.add_argument("--model", default=str(ROOT / "weights" / "student_int8.tflite"))
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--num-runs", type=int, default=500)
    ap.add_argument("--warmup-runs", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0, help="optional --num_threads (0 = tool default, like the canonical command)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "arm_benchmark.txt"))
    a = ap.parse_args()

    if not Path(a.model).is_file() or not Path(a.benchmark_model).is_file():
        raise SystemExit(f"[arm] missing file: model={a.model} benchmark_model={a.benchmark_model}")
    devs = [ln for ln in sh(a.adb, "devices").splitlines()[1:] if ln.strip()]
    if not [ln for ln in devs if ln.split()[-1] == "device"]:
        raise SystemExit("[arm] no authorised device. Enable USB debugging, plug in, accept the RSA prompt on the phone. "
                         f"adb devices said: {devs or 'nothing'}")

    props = {p: sh(a.adb, "shell", "getprop", p) for p in PROPS}
    battery = sh(a.adb, "shell", "dumpsys", "battery", check=False)
    batt = re.search(r"level:\s*(\d+)", battery)
    temp = re.search(r"temperature:\s*(\d+)", battery)
    sh(a.adb, "push", a.benchmark_model, f"{REMOTE}/benchmark_model")
    sh(a.adb, "push", a.model, f"{REMOTE}/student_int8.tflite")
    sh(a.adb, "shell", "chmod", "+x", f"{REMOTE}/benchmark_model")

    lines = [f"EchoGesture ARM64 model-only benchmark | {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
             "SCOPE: model-only latency (INT8 TFLite + XNNPACK). Excludes MediaPipe landmarks and video decode. "
             "Does NOT replace the laptop end-to-end latency_p95.",
             f"model: {Path(a.model).name} ({Path(a.model).stat().st_size / 1e6:.4f} MB)",
             "device identity (read from the device, not assumed):"]
    lines += [f"  {k} = {v or '<empty>'}" for k, v in props.items()]
    lines.append(f"battery level = {batt.group(1) + '%' if batt else 'unknown'} | "
                 f"battery temp = {int(temp.group(1)) / 10:.1f} C" if temp else "battery temp = unknown")
    summaries = []
    for i in range(1, a.runs + 1):
        cmd = ["shell", f"{REMOTE}/benchmark_model", f"--graph={REMOTE}/student_int8.tflite", "--use_xnnpack=true",
               f"--num_runs={a.num_runs}", f"--warmup_runs={a.warmup_runs}"]
        if a.threads:
            cmd.append(f"--num_threads={a.threads}")
        print(f"[arm] run {i}/{a.runs} ...", flush=True)
        raw = sh(a.adb, *cmd)
        p = parse(raw)
        summaries.append(p)
        lines += ["", f"===== run {i} | benchmark_model {' '.join(cmd[2:])} =====", raw, f"parsed: {p}"]
    avgs = [s["avg_us"] / 1000.0 for s in summaries if "avg_us" in s]
    if avgs:
        lines += ["", f"SUMMARY: model-only avg latency per run (ms) = {[round(x, 3) for x in avgs]} | "
                      f"spread = {max(avgs) - min(avgs):.3f} ms | mean = {sum(avgs) / len(avgs):.3f} ms"]
    else:
        lines += ["", "SUMMARY: could not parse avg latency; read the raw output above (nothing was estimated)."]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]), f"\n[arm] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
