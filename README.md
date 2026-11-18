# EchoGesture (Team Peratral, VIT Vellore) - Samsung XR Hackathon 2026, Track 02

Fully on-device, air-gapped, real-time sign language recognition: video -> MediaPipe landmarks (every 3rd frame)
-> interpolation -> 53 joints -> `(C, 64, 53)` tensor -> tiny INT8 student GCN -> gloss + confidence.

> **HONESTY NOTE (proxy vocabulary).** Results are on a **WLASL-100 proxy vocabulary** (`len(vocab)` = 100, see
> `data/vocab.json`, `"_proxy": true`) because the official Track 02 starter kit was not received in time. The pipeline reads
> `data/vocab.json` at runtime and ingests the official file without code changes. Any test clip whose true word is outside
> this proxy list is wrong by construction; proxy accuracy does not predict accuracy on the official test set.
> The dataset is the WLASL-100 *frame archive* (16 sparse frames per clip, 256x256 JPG), not full-rate video.

Every number lives in a file under `outputs/` written by a script; nothing here is typed in by hand:

| What | File |
|---|---|
| 6-row metric table | `outputs/results.csv` |
| Per-clip predictions (JSON lines) + identical copy | `outputs/predictions.jsonl`, `outputs/predictions.json` |
| Accuracy (top-1/top-5), robustness variants, latency waterfall, thermal loop | `outputs/benchmark_log.json` |
| Per-step streaming latency + full-clip batch throughput (`total_clip_ms`, `FPS`) | `outputs/latency_smoke.json` |
| Training: ablation (6 vs 7 channels), teacher, baseline vs distilled student, QAT, INT8 export | `outputs/training_summary.json`, `outputs/train_log.txt` |


## Final measured results (test split, 258 clips, 100-class proxy vocabulary)
Source: `outputs/results.csv`, `outputs/benchmark_log.json`. Laptop = AMD Ryzen 7 5800H, Windows (**proxy**), ONNX Runtime INT8, 1 thread.

| Metric | Target | Measured | Pass |
|---|---|---|---|
| top1_accuracy (top-5 = 84.9%) | >= 70 % | **61.63 %** | No (honest shortfall: 100-class proxy, ~14 train clips/class) |
| latency_p95 (per-step streaming, k=3) | <= 50 ms | **40.94 ms** (landmarks 43.0 on keyframes, features 0.8, INT8 model 2.5, post 0.1) | Yes (also under our <= 42 ms aim) |
| model_size | <= 20 MB | **0.129 MB** (ONNX 0.128, TFLite 0.129) | Yes |
| accuracy_drop_robustness (end-to-end: low-light frames + tilt +-15 deg + 15% whole-hand drops) | <= 5 pts | **25.97 pts** | No |
| avg_power (**proxy**: process CPU util x assumed 45 W TDP) | <= 500 mW | 2897 mW (6.4 % util) | No |
| thermal_drift_10min | <= 20 % | -0.83 % | Yes |

**Robustness, split (`outputs/robustness_detail.json`).**
(a) *Model-level* (clean landmarks, isolates our GCN): whole-hand drop 15 % = 3.1 pts, tilt +-15 deg = 2.3 pts, coordinate noise (sigma 0.015) = 15.1 pts, all three combined = **17.05 pts**.
(b) *End-to-end sensor + model* (darkened RGB frames re-extracted through MediaPipe): low light alone = 19.0 pts, low light + tilt + hand drops = **25.97 pts** (the results.csv row).
Low-light frames mostly hurt upstream MediaPipe, not the INT8 GCN, but the GCN is also not robust to jitter: it loses 15 pts to sigma-0.015 coordinate noise, so a model-only fix is still needed. Diagnostic single-joint dropout = 48.8 pts (not the results row).

**Power is a proxy, not a meter.** `avg_power` = process CPU utilisation (share of all logical CPUs over the 10-minute streaming loop) x 45 W, the vendor-listed default TDP of this laptop CPU (an assumed constant in `config.yaml`, `cpu_tdp_w`). It is a laptop number, not a phone number, and it fails the 500 mW target.

**Quantization.** TFLite INT8 drop vs FP32 on validation = 0.3 pts. ONNX INT8: the original 200-clip calibration lost 1.48 pts (missed the 1-pt gate); re-quantizing the same FP32 weights with 600 real calibration clips (percentile 99.999) loses **0.30 pts** (`scripts/requant_onnx.py`, `outputs/requant_onnx.json`). Settings were chosen on the validation split, so the pass is within about +-1 pt of noise. Keeping `/fc/Conv` in FP32 (2.08 pts), excluding the SE-gate nodes (1.19 pts) and Entropy calibration (59 pts) did not pass.

**Cold start and preprocessing (all measured, `outputs/preproc_ablation.json`).** A 20-step warm-up of the ONNX session and both MediaPipe landmarkers runs at load time; it removed the minute-1 latency spike (thermal drift moved from -16.6 % to -0.8 %). Two optional input steps exist but are **OFF by default** because they hurt: resizing frames to 224 px cost 4.3 pts clean top-1, and adaptive LAB-CLAHE (mean luma < 85) cut end-to-end low-light top-1 from 42.6 % to 26.7 %. Switches: `mp_max_side`, `clahe_luma` in `config.yaml`.

**Model-selection history (Slide 8 ablation).**
| Run | What changed | Test top-1 | Realistic robustness drop |
|---|---|---|---|
| 1 | wrist/MCP normalization, augmented training, unaugmented QAT | 59.69 % | 25.58 pts (re-scored on the new test) |
| 2 | centroid-fallback normalization + heavy Mixup/CutMix + augmented QAT | 40.70 % | 11.63 pts (lower only because it started lower) |
| final | run 1 normalization restored, run 1 student + gentle low-light QAT fine-tune, ONNX re-quantized | **61.63 %** | 25.97 pts |
Run 2 was discarded (`weights/run2/`, `outputs/run2/`). `--frame_skip 4` gave 46.17 vs 46.48 ms in the earlier run, so k=3 is kept.

**Compute used.** Training (ablation, Teacher, students) ran on the team laptop's RTX 3060 (CUDA); extraction, QAT, export and all benchmarks ran on the laptop CPU. A GCP VM was started early and deleted without producing any result (GPU quota was 0); no EchoGesture VM exists in the shared team GCP project (two unrelated VMs there belong to others and were not touched).

### Real-phone ARM64 (model-only, NOT end to end)
`outputs/arm_benchmark.txt`: Motorola Edge 60 Pro, SoC reported by the device = Mediatek **MT6897**, arm64-v8a, Android 17,
`student_int8.tflite` + XNNPACK, 500 runs after 20 warm-ups, two runs: avg **3.097 ms** and **2.785 ms** (p95 3.147 / 3.142 ms).
Excludes MediaPipe and decode, so it does not replace the laptop end-to-end latency above.
Reproduce: `python scripts/arm_benchmark.py --adb <adb> --benchmark-model <android_aarch64_benchmark_model>`.

## Live HUD (`src/hud_demo.py`)
```bash
python src/hud_demo.py                       # webcam 0 (mirrored selfie view), q quits
python src/hud_demo.py --source clip.mp4     # video file or frame directory; --no-tts to mute
```
64-frame window, inference every 5 frames, 2-of-3 vote with confidence >= 0.60 to confirm (< 0.40 "Uncertain...", 0.40-0.60 dimmed),
hands-down reset (hand speed < 0.005 and wrists below shoulders + 0.35 torso for 5 frames) with a 0.3 s fade, offline `pyttsx3`
on a daemon thread. Latency/FPS/top-5 are live; HUD render cost is logged separately (7.9 ms mean on the idle dev laptop).
The network counter is the **system-wide** NIC byte delta from `psutil` (other apps add bytes): record the demo with Wi-Fi off.
Torso height uses 1.8 x shoulder width because hips are not part of the 53 stored joints. Font: bundled DejaVu (`assets/fonts/`, licence included).

## Latency definitions (project spec 3.1)
* `latency_p95` = p95 of **500 consecutive streaming steps** after 20 warm-up steps, frame skip **k=3** (1/3 of steps run
  MediaPipe hand+pose + interpolation, all steps run features + INT8 inference + post-processing; decode excluded).
* `predict()` returns `latency_ms` = wall-clock of the **final streaming step**. Whole-clip timing (`total_clip_ms`, `FPS`,
  decode included) is logged separately in `outputs/latency_smoke.json` and is never the `latency_p95`.
* Laptop numbers are a **proxy** (Windows x86 CPU). The real-phone ARM64 `benchmark_model` number is model-only and is
  reported separately above (`outputs/arm_benchmark.txt`).
* `avg_power` is **not measured** (no power meter); the row says so.

## Setup and run (Windows or Linux)
```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/fetch_mediapipe_models.py             # dev-time only: puts the two .task bundles in weights/ (src/ never downloads)
python -m pytest tests -q                            # all unit tests
python scripts/benchmark.py --manifest data/test_manifest.csv --output_dir outputs   # add --quick to skip the 10-min thermal loop
python scripts/verify_submission.py
```
Retraining from the frame archive: `python scripts/prepare_data.py` (vocab + manifests), `python scripts/gcp_extract.py`
(keypoint cache), `python scripts/train_pipeline.py` (ablation -> teacher -> MAG-KD / `--no_distill` -> QAT -> INT8 export).
GCP: `python scripts/gcp_orchestrate.py` provisions a VM, trains, pulls artifacts and **always deletes the VM** (billing guard).

```python
from src.predict import predict
predict("path/to/clip.mp4")   # or a directory of frames
# {"clip_id": "...", "prediction": "BOOK", "top5": [...], "confidence": 0.61, "latency_ms": 12.3}
```

## Engineering defaults (ours, not Samsung rules)
`teacher_min_gain=0.02`, `calib_samples=200`, `hud_warn_ms=50`, `frame_skip=3`, QAT 15 epochs at lr 1e-4, MAG-KD `tau=4`, `lambda=0.6`,
label smoothing 0.1. Robustness perturbations (15% joint occlusion, +-15 deg tilt, low-light gamma/noise) are our own methodology.

## Air-gap
`src/` imports no network modules (enforced by `tests/test_airgap.py`); all weights load from `./weights/`. The Docker image has
**not been built or run** on the development machine (no Docker installed); the Wireshark / airplane-mode proof is still to do.

## Air-gapped container check (measured)

```bash
docker build -t echogesture:airgapped .            # needs network once, to download wheels
docker run --rm --network none echogesture:airgapped pytest tests -q
```

Result on the build machine (Docker Desktop, Linux container, CPU-only PyTorch, image about 5.3 GB): **169 passed** with `--network none`
(exit code 0). Full output: `outputs/docker_verify.txt`. The container runs the unit and contract tests; the 258-clip benchmark needs the
frame archive, which is not shipped in the image.
