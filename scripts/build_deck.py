"""Build the 9-slide submission deck from the measured files in outputs/ (no number is typed by hand).

    python scripts/build_deck.py            # writes VITVellore_Peratral_Submission_ppt.pptx and .pdf in the repo root

Dev tool only: never imported by src/. The official template (Submission_Final_forGithub.pptx) was not available in the
workspace, so this builds a clean 16:9 deck with the same 9-slide order. If the template arrives, copy the text into it.
"""
from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
NAME = "VITVellore_Peratral_Submission_ppt"
REPO_URL = "https://github.com/Veladicodes/EchoGesture_Peratral"

BG, PANEL, INK, MUTED = RGBColor(0x0D, 0x11, 0x17), RGBColor(0x16, 0x1B, 0x22), RGBColor(0xE6, 0xED, 0xF3), RGBColor(0x8B, 0x94, 0x9E)
ACCENT, WARN, BAD = RGBColor(0x00, 0xFF, 0xAA), RGBColor(0xF2, 0xB7, 0x3D), RGBColor(0xFF, 0x55, 0x55)
CODE_BG = RGBColor(0x0A, 0x0E, 0x13)


# ---------------------------------------------------------------- measured numbers
def load() -> dict:
    rows = {r["metric"]: r for r in csv.DictReader(open(OUT / "results.csv", newline="", encoding="utf-8"))}
    bl = json.loads((OUT / "benchmark_log.json").read_text(encoding="utf-8"))
    ts = json.loads((OUT / "training_summary.json").read_text(encoding="utf-8"))
    arm = (OUT / "arm_benchmark.txt").read_text(encoding="utf-8")
    parsed = [ast.literal_eval(m) for m in re.findall(r"^parsed: (\{.*\})$", arm, re.M)]
    ident = dict(re.findall(r"^\s+(ro\.[\w.]+) = (.+)$", arm, re.M))
    minmax = re.search(r"val top1 FP32=([\d.]+) INT8=([\d.]+)", (OUT / "train_log.txt").read_text(encoding="utf-8"))

    def stdout_top1(name: str) -> float:
        return float(re.search(r"^top1_accuracy,70,([\d.]+),", (OUT / name).read_text(encoding="utf-8"), re.M).group(1))

    docker = OUT / "docker_verify.txt"
    return {"rows": rows, "bl": bl, "ts": ts, "arm": parsed, "ident": ident,
            "minmax": (float(minmax.group(1)), float(minmax.group(2))),
            "run1": stdout_top1("benchmark_stdout.txt"), "run2": stdout_top1("benchmark_stdout2.txt"),
            "docker": docker.read_text(encoding="utf-8", errors="replace") if docker.is_file() else None}


# ---------------------------------------------------------------- drawing helpers
def txt(slide, x, y, w, h, text, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, line in enumerate(text if isinstance(text, list) else [text]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, "Calibri"
        p.space_after = Pt(6)
    return tb


def box(slide, x, y, w, h, fill=PANEL, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    s.shadow.inherit = False
    return s


def footer(s, n):
    txt(s, 0.6, 7.05, 9, 0.3, "EchoGesture  ·  Team Peratral, VIT Vellore  ·  Samsung XR Hackathon 2026  ·  Track 02", 9, MUTED)
    txt(s, 11.9, 7.05, 0.85, 0.3, "%02d / 09" % n, 9, MUTED, False, PP_ALIGN.RIGHT)


def chip(s, x, y, w, big, small, color=None):
    box(s, x, y, w, 1.15, PANEL, RGBColor(0x2A, 0x33, 0x3D))
    txt(s, x + 0.2, y + 0.08, w - 0.4, 0.6, big, 26, color or ACCENT, True)
    txt(s, x + 0.2, y + 0.68, w - 0.4, 0.45, small, 11, MUTED)


def new_slide(prs, title, kicker):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    num = len(prs.slides._sldIdLst)
    box(s, 0.6, 0.44, 0.13, 0.13, ACCENT, shape=MSO_SHAPE.OVAL)
    txt(s, 0.82, 0.33, 11, 0.35, "%02d   %s" % (num, kicker.upper()), 12, ACCENT, True)
    txt(s, 0.6, 0.65, 12.1, 0.9, title, 28, INK, True)
    footer(s, num)
    return s


def bullets(slide, x, y, w, items, size=17, gap=0.62):
    for i, it in enumerate(items):
        box(slide, x, y + i * gap + 0.12, 0.09, 0.09, ACCENT, shape=MSO_SHAPE.OVAL)
        txt(slide, x + 0.25, y + i * gap, w - 0.25, gap, it, size)


def table(slide, x, y, w, col_w, header, body, size=12, row_h=0.42, colors=None):
    t = slide.shapes.add_table(len(body) + 1, len(header), Inches(x), Inches(y), Inches(w), Inches(row_h * (len(body) + 1))).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate([header] + body):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = RGBColor(0x1F, 0x29, 0x33) if i == 0 else PANEL
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.03)
            tf = c.text_frame
            tf.word_wrap = True
            r = tf.paragraphs[0].add_run()
            r.text = str(val)
            r.font.size, r.font.name = Pt(size), "Calibri"
            r.font.bold = i == 0
            r.font.color.rgb = ACCENT if i == 0 else (colors[i - 1][j] if colors and colors[i - 1][j] else INK)
    return t


# ---------------------------------------------------------------- deck
def build(d: dict) -> Path:
    R, bl, ts = d["rows"], d["bl"], d["ts"]
    m = lambda k: float(R[k]["measured"])  # noqa: E731
    top1, top5 = m("top1_accuracy"), 100 * bl["accuracy"]["top5"]
    lat, size_mb, rob, pw, th = m("latency_p95"), m("model_size"), m("accuracy_drop_robustness"), m("avg_power"), m("thermal_drift_10min")
    drops, wf = bl["robustness"]["drop_pts"], bl["latency"]["waterfall_p95_ms"]
    params = ts["student_params"]
    n_vocab, n_test = bl["vocab_size"], bl["test_clips"]
    ok = lambda k: R[k]["pass"] == "True"  # noqa: E731
    arm1, arm2 = d["arm"][0], d["arm"][1]
    ident = d["ident"]

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # 1 Cover
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    phone_lo = min(arm1["avg_us"], arm2["avg_us"]) / 1000
    phone_hi = max(arm1["avg_us"], arm2["avg_us"]) / 1000
    box(s, 0.6, 0.72, 0.16, 0.16, ACCENT, shape=MSO_SHAPE.OVAL)
    txt(s, 0.85, 0.6, 11.5, 0.4, "TRACK 02: ECHOGESTURE  ·  SIGN LANGUAGE PREDICTION ON EDGE DEVICES", 14, ACCENT, True)
    txt(s, 0.6, 1.35, 12.1, 0.9, "EchoGesture", 60, INK, True)
    txt(s, 0.6, 2.4, 11.8, 1.3, "Air-gapped, on-device sign language recognition via correctness-masked distillation", 28, INK)
    cw = 2.85
    for i, (big, small) in enumerate([("%.3f MB" % size_mb, "INT8 model (%.0fx under the 20 MB limit)" % (20 / size_mb)),
                                      ("%.2f ms" % lat, "streaming p95, laptop CPU (<= 50 ms)"),
                                      ("%.1f-%.1f ms" % (phone_lo, phone_hi), "model-only, Motorola Edge 60 Pro"),
                                      ("0 sockets", "no network code in src/ (tested)")]):
        chip(s, 0.6 + i * (cw + 0.233), 4.15, cw, big, small)
    txt(s, 0.6, 5.85, 12, 0.5, "Team Peratral  ·  VITVellore (Vellore Institute of Technology, Vellore)", 20, INK, True)
    txt(s, 0.6, 6.35, 12, 0.5, "Adithya A (B.Tech IT, VIT Vellore)  ·  Tharun Sankar  ·  Samsung XR Hackathon 2026", 15, MUTED)

    # 2 Problem
    s = new_slide(prs, "Conversation needs speed and privacy, and cloud signing gives neither", "The problem")
    box(s, 0.6, 1.8, 5.9, 4.6)
    txt(s, 0.9, 2.0, 5.3, 0.5, "Cadence gap", 22, ACCENT, True)
    txt(s, 0.9, 2.7, 5.3, 3.5, ["Natural dialogue needs the caption to keep up with the signer: end-to-end latency of 50 ms or less.",
                                 "A cloud round trip adds network delay on top of inference, and breaks completely with no signal."], 17)
    box(s, 6.8, 1.8, 5.9, 4.6)
    txt(s, 7.1, 2.0, 5.3, 0.5, "Bystander privacy", 22, ACCENT, True)
    txt(s, 7.1, 2.7, 5.3, 3.5, ["An exocentric (third-person) camera captures everyone in frame, not just the signer.",
                                 "Streaming that raw video to a server exposes people who never agreed to be recorded.",
                                 "Skeleton landmarks computed on the device mean no image ever has to leave it."], 17)

    # 3 Solution
    s = new_slide(prs, "A 100%% air-gapped pipeline that fits in %.3f MB" % size_mb, "The solution")
    flow = ["MediaPipe Tasks\nk=3 keyframes", "53-joint 3D\nnormalizer", "%.1fK-param INT8\nStudentSTGCN" % (params / 1000), "XR HUD +\noffline pyttsx3 TTS"]
    for i, f in enumerate(flow):
        x = 0.6 + i * 3.1
        box(s, x, 1.9, 2.7, 1.3, PANEL, ACCENT)
        txt(s, x, 1.9, 2.7, 1.3, f.split("\n"), 15, INK, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
        if i < 3:
            txt(s, x + 2.68, 2.3, 0.45, 0.5, ">", 24, ACCENT, True, PP_ALIGN.CENTER)
    bullets(s, 0.6, 3.7, 12, [
        "Teacher (4.76M params, trained locally on an RTX 3060 Laptop) distilled into the student with MAG-KD correctness masking.",
        "Student is export-safe (1x1 and depthwise convs, static adjacency): INT8 ONNX %.3f MB, INT8 TFLite %.3f MB." % (bl["model_size_mb"]["onnx"], bl["model_size_mb"]["tflite"]),
        "No network code in src/: enforced by tests/test_airgap.py, plus a live system byte counter on the HUD.",
        "Honest scope: %d-class WLASL proxy vocabulary (Samsung's official kit was not received)." % n_vocab,
    ], 16, 0.7)

    # 4 Architecture
    s = new_slide(prs, "Architecture: from raw frames to a spoken gloss, all on-device", "Architecture")
    blocks = [("Video / frames", "any camera"), ("k=3 landmarks", "MediaPipe hand+pose"), ("Interp + guard", "zero-hand guard"),
              ("(7, 64, 53) tensor", "view-invariant"), ("INT8 Student", "%.3f MB" % size_mb),
              ("Window + reset", "64 fr, 2/3 vote, hands-down"), ("HUD + TTS", "offline pyttsx3")]
    bw, gap = 1.62, 0.2
    for i, (a, b) in enumerate(blocks):
        x = 0.55 + i * (bw + gap)
        box(s, x, 2.3, bw, 1.7, PANEL, ACCENT if i == 4 else MUTED)
        txt(s, x, 2.45, bw, 0.8, a, 14, INK, True, PP_ALIGN.CENTER)
        txt(s, x, 3.2, bw, 0.7, b, 11, MUTED, False, PP_ALIGN.CENTER)
        if i < len(blocks) - 1:
            txt(s, x + bw - 0.05, 2.9, gap + 0.1, 0.4, ">", 18, ACCENT, True, PP_ALIGN.CENTER)
    bullets(s, 0.6, 4.5, 12, [
        "Frames 0, 3, 6... go through MediaPipe; the two skipped frames are linearly interpolated, so only 1 in 3 steps pays for landmarks.",
        "Hands-down reset: hand speed < 0.005 and wrists below shoulder + 0.35 x torso for 5 frames flushes the buffer and fades the caption.",
        "Confirm a gloss only when 2 of 3 predictions agree and confidence is at least 0.60; weaker output is shown dimmed and not spoken.",
    ], 15, 0.62)

    # 5 Track final output
    s = new_slide(prs, "Track final output: measured on %d held-out test clips (%d-class proxy)" % (n_test, n_vocab), "Track final output")
    pm = lambda k: "PASS" if ok(k) else "MISS"  # noqa: E731
    body = [["top1_accuracy", ">= 70 %", "%.2f %%  (top-5 %.1f %%)" % (top1, top5), pm("top1_accuracy")],
            ["latency_p95 (per streaming step)", "<= 50 ms", "%.2f ms" % lat, pm("latency_p95")],
            ["model_size", "<= 20 MB", "%.3f MB" % size_mb, pm("model_size")],
            ["accuracy_drop_robustness (end-to-end)", "<= 5 pts", "%.2f pts (model-only %.2f)" % (rob, drops["model_combined"]), pm("accuracy_drop_robustness")],
            ["avg_power (laptop CPU proxy)", "<= 500 mW", "%d mW (45 W TDP x %.1f %% util)" % (pw, 100 * bl["thermal"]["process_cpu_util_frac"]), pm("avg_power")],
            ["thermal_drift_10min", "<= 20 %", "%.2f %%" % th, pm("thermal_drift_10min")]]
    cols = [[None, None, None, ACCENT if b[3] == "PASS" else BAD] for b in body]
    table(s, 0.6, 1.7, 7.4, [3.0, 1.1, 2.6, 0.7], ["metric", "target", "measured", "pass"], body, 12, 0.5, cols)
    hl = box(s, 0.6, 5.32, 7.4, 0.72, RGBColor(0x0B, 0x2A, 0x22), ACCENT)
    txt(s, 0.75, 5.36, 7.1, 0.66, "%.3f MB INT8 (%.0fx under limit)  ·  %.2f ms streaming p95  ·  %.1f-%.1f ms on Motorola Edge 60 Pro (%s)  ·  %.1f %% Top-5 (%d classes)" %
        (size_mb, 20 / size_mb, lat, min(arm1["avg_us"], arm2["avg_us"]) / 1000, max(arm1["avg_us"], arm2["avg_us"]) / 1000, ident["ro.soc.model"], top5, n_vocab),
        13, ACCENT, True, PP_ALIGN.LEFT, MSO_ANCHOR.MIDDLE)
    txt(s, 0.6, 6.12, 7.4, 0.9, ["Device: AMD Ryzen laptop CPU (proxy), ONNX Runtime INT8 + MediaPipe CPU. Power is a CPU-utilisation x TDP estimate, not a meter reading.",
                                 "Top-1 is below the 70 %% target: ~14 train clips per class on a sparse 16-frame archive. Vocabulary = %d (proxy, not Samsung's list)." % n_vocab], 10, MUTED)
    txt(s, 8.4, 1.7, 4.5, 0.4, "Latency waterfall, keyframe step p95 (ms)", 14, ACCENT, True)
    parts = [("MediaPipe landmarks", wf["landmarks_hand_pose"]), ("Features", wf["features"]), ("INT8 student", wf["int8_model_only"]), ("Post-process", wf["postprocess"])]
    scale = 4.2 / 50.0
    for i, (n, v) in enumerate(parts):
        yy = 2.25 + i * 0.62
        txt(s, 8.4, yy, 4.5, 0.3, "%s  %.2f ms" % (n, v), 12)
        box(s, 8.4, yy + 0.3, max(v * scale, 0.04), 0.18, ACCENT, shape=MSO_SHAPE.RECTANGLE)
    box(s, 8.4 + 50 * scale - 0.02, 2.2, 0.03, 2.55, BAD, shape=MSO_SHAPE.RECTANGLE)
    txt(s, 8.4 + 50 * scale - 1.1, 4.75, 1.4, 0.3, "50 ms limit", 11, BAD, True, PP_ALIGN.CENTER)
    txt(s, 8.4, 5.2, 4.5, 1.2, ["Keyframe step p95 %.1f ms; interpolated step p95 %.1f ms; streaming p95 over all steps %.2f ms (k=3, 500 steps, 20 warm-up)." %
                                (bl["latency"]["keyframe_step_ms_p95"], bl["latency"]["interp_step_ms_p95"], lat)], 11, MUTED)

    # 6 Differentiation
    s = new_slide(prs, "What makes EchoGesture different (measured validation gains)", "Differentiation")
    dz = 100 * (ts["ablation"]["7ch"]["val_top1"] - ts["ablation"]["6ch"]["val_top1"])
    dk = 100 * (ts["student_distilled"]["val_top1"] - ts["student_baseline"]["val_top1"])
    mm_fp, mm_i8 = d["minmax"]
    cards = [("MAG-KD correctness masking", "Distil only where the teacher is right. Distilled student %.1f %% vs baseline %.1f %% val top-1: %+.2f pts." %
              (100 * ts["student_distilled"]["val_top1"], 100 * ts["student_baseline"]["val_top1"], dk)),
             ("3D pose-invariant geometry", "Wrist/MCP and shoulder normalisation with z kept. 7 channels %.1f %% vs 6 channels %.1f %% val top-1: %+.1f pts (40-epoch ablation)." %
              (100 * ts["ablation"]["7ch"]["val_top1"], 100 * ts["ablation"]["6ch"]["val_top1"], dz)),
             ("Calibrated INT8 that survives", "MinMax calibration collapsed INT8 to %.1f %% val (FP32 %.1f %%). Percentile 99.999 with 600 clips: %.2f %% INT8 vs %.2f %% FP32 val, a %.2f pt drop." %
              (100 * mm_i8, 100 * mm_fp, 100 * ts["export"]["int8_onnx_val_top1"], 100 * ts["export"]["fp32_val_top1"], ts["export"]["int8_top1_drop_pts"])),
             ("Dual-condition word boundary", "Low hand speed AND wrists below chest level for 5 frames triggers a smooth 0.3 s fade-out reset, so signs do not bleed into each other.")]
    for i, (h, b) in enumerate(cards):
        x, y = 0.6 + (i % 2) * 6.2, 1.75 + (i // 2) * 2.45
        box(s, x, y, 5.95, 2.3)
        txt(s, x + 0.25, y + 0.15, 5.5, 0.5, h, 19, ACCENT, True)
        txt(s, x + 0.25, y + 0.75, 5.5, 1.5, b, 14)
    txt(s, 0.6, 6.6, 12, 0.4, "Trade-off we accept: a small skeleton model gives up raw accuracy versus a cloud video model, in exchange for privacy, speed and size.", 12, MUTED)

    # 7 Demo & local setup
    s = new_slide(prs, "Demo and local setup: three commands, no network", "Demo and local setup")
    shot = OUT / "hud_live.png"
    if shot.is_file():
        s.shapes.add_picture(str(shot), Inches(0.6), Inches(1.75), height=Inches(3.0))
    txt(s, 0.6, 4.78, 4.8, 0.55, "Live XR HUD on a held-out test stream (k=3, Top-5 led by the true gloss NO). Network counter is system-wide; our src/ opens no sockets.", 10, MUTED)
    code = ["pip install -r requirements.txt", "pytest tests -q", "python scripts/verify_submission.py"]
    box(s, 5.6, 1.75, 7.1, 1.25, CODE_BG, MUTED)
    txt(s, 5.8, 1.82, 6.8, 1.2, ["$ " + c for c in code], 13, ACCENT)
    txt(s, 5.6, 3.1, 7.1, 0.3, "predict(video_path) returns exactly 5 keys:", 13, INK, True)
    box(s, 5.6, 3.45, 7.1, 1.55, CODE_BG, MUTED)
    txt(s, 5.8, 3.5, 6.8, 1.5, ['{"clip_id": "test_0412",', ' "prediction": "<UPPERCASE_GLOSS>",', ' "top5": ["W1","W2","W3","W4","W5"],', ' "confidence": 0.85, "latency_ms": 24.1}'], 12, INK)
    dk_txt = ["Docker run (network disabled): see outputs/docker_verify.txt"] if not d["docker"] else [ln.strip() for ln in d["docker"].strip().splitlines() if re.search(r"\d+ passed", ln)][-1:]
    txt(s, 0.6, 5.4, 12, 0.3, "Air-gapped container check", 13, ACCENT, True)
    box(s, 0.6, 5.75, 12.1, 1.0, CODE_BG, MUTED)
    txt(s, 0.8, 5.85, 11.8, 0.9,["docker run --rm --network none echogesture:airgapped pytest tests -q"] + dk_txt, 11, INK)

    # 8 Brownie
    s = new_slide(prs, "Brownie: real phone silicon, ablation history and robustness breakdown", "Brownie slides")
    txt(s, 0.6, 1.65, 6.1, 0.4, "Motorola Edge 60 Pro, model-only (500 runs, XNNPACK)", 14, ACCENT, True)
    table(s, 0.6, 2.1, 6.1, [2.1, 2.0, 2.0], ["run", "avg (ms)", "p95 (ms)"],
          [["Run 1", "%.3f" % (arm1["avg_us"] / 1000), "%.3f" % (arm1["p95_us"] / 1000)], ["Run 2", "%.3f" % (arm2["avg_us"] / 1000), "%.3f" % (arm2["p95_us"] / 1000)]], 12, 0.4)
    txt(s, 0.6, 3.4, 6.1, 1.0, ["%s | %s %s | %s | Android %s" % (ident["ro.product.model"].title(), ident["ro.soc.manufacturer"],
                               ident["ro.soc.model"], ident["ro.product.cpu.abi"], ident["ro.build.version.release"]),
                              "Excludes MediaPipe and video decode; it does not replace the laptop end-to-end latency."], 11, MUTED)
    txt(s, 0.6, 4.45, 6.1, 0.4, "Engineering ablation (test top-1)", 14, ACCENT, True)
    table(s, 0.6, 4.9, 6.1, [3.9, 2.2], ["run", "top-1"],
          [["Run 1: wrist/MCP norm, plain QAT", "%.2f %%" % d["run1"]], ["Run 2: centroid fallback + heavy Mixup (discarded)", "%.2f %%" % d["run2"]],
           ["Final: run-1 recipe + gentle QAT + 600-clip INT8", "%.2f %%" % top1]], 11, 0.42)
    txt(s, 7.0, 1.65, 5.8, 0.4, "Robustness drop vs clean (pts, lower is better)", 14, ACCENT, True)
    names = [("whole-hand drop, 15 % frames", "hand_drop_15pct_frames"), ("tilt +-15 deg", "tilt_pm15deg"), ("coordinate noise sigma 0.015", "coord_noise_sigma0.015"),
             ("model-level combined", "model_combined"), ("low light (frames re-extracted)", "low_light"), ("end-to-end combined (results.csv)", "all_combined")]
    for i, (n, k) in enumerate(names):
        yy = 2.15 + i * 0.62
        txt(s, 7.0, yy, 5.8, 0.3, "%s  %.2f" % (n, drops[k]), 12)
        box(s, 7.0, yy + 0.3, max(drops[k] * 0.17, 0.04), 0.17, BAD if k == "all_combined" else ACCENT, shape=MSO_SHAPE.RECTANGLE)
    txt(s, 7.0, 5.95, 5.8, 1.2, ["Low light costs most, and mostly because MediaPipe loses landmarks in dark frames. Coordinate noise is the model's own weak spot."], 11, MUTED)

    # 9 Thank you
    s = new_slide(prs, "", "Team Peratral | VITVellore")
    txt(s, 0.6, 0.85, 12, 1.1, "Thank you", 54, INK, True)
    txt(s, 0.6, 2.0, 12, 0.5, "Code, weights and results", 16, MUTED)
    txt(s, 0.6, 2.45, 12, 0.6, REPO_URL, 26, ACCENT, True)
    txt(s, 0.6, 3.5, 12, 0.5, "Demo video (2:30-2:50)", 16, MUTED)
    txt(s, 0.6, 3.95, 12, 0.6, "<PASTE_DEMO_VIDEO_URL_HERE>", 22, WARN, True)
    txt(s, 0.6, 5.0, 12, 0.5, "Contact", 16, MUTED)
    txt(s, 0.6, 5.45, 12, 1.0, ["Team Peratral: Adithya A, Tharun Sankar  ·  VITVellore (Vellore Institute of Technology, Vellore)",
                                 "Adithya A  ·  adithya.am05@gmail.com  ·  +91 94870 38353  ·  VIT Vellore",
                                 "Tharun Sankar  ·  tharunsankar.my2023@vitstudent.ac.in  ·  +91 63833 98396  ·  VIT Vellore"], 18)

    pptx = ROOT / (NAME + ".pptx")
    prs.save(pptx)
    return pptx


def to_pdf(pptx: Path) -> Path:
    lo = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")
    subprocess.run([str(lo) if lo.is_file() else "soffice", "--headless", "--convert-to", "pdf", "--outdir", str(pptx.parent), str(pptx)],
                   check=True, capture_output=True)
    return pptx.with_suffix(".pdf")


def main() -> int:
    p = build(load())
    print("wrote", p)
    print("wrote", to_pdf(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
