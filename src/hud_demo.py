"""Live XR HUD (CLAUDE.md 5.5, 13, 14): sliding window, 2/3 vote, hands-down reset, offline TTS, live telemetry.

    python src/hud_demo.py                          # webcam 0 (mirrored selfie view = the model's expected handedness)
    python src/hud_demo.py --source clip.mp4        # a video file or a frame directory (not mirrored)
    python src/hud_demo.py --source <frames dir> --headless --loop --max-frames 60 --save-frame outputs/hud.png

Everything shown is computed from the running loop: latency, FPS, network delta, top-5. Nothing is typed in.
No network access: Pillow draws locally, pyttsx3 uses the OS speech engine, psutil only READS the local NIC byte
counters (system-wide, not per process, so the label says "system network delta").
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WINDOW = 64
STRIDE = 5                 # inference every 5 frames (5.5)
MIN_WINDOW = 16            # do not vote on a nearly empty buffer
CONF_UNCERTAIN = 0.40
CONF_CONFIRM = 0.60
VOTES = 3
VOTES_NEEDED = 2
VEL_THRESH = 0.005         # mean hand-joint speed per frame (normalised image units)
BELOW_FRAC = 0.35          # wrist must be this many torso heights below the shoulders
TORSO_PER_SHOULDER = 1.8   # hips are not stored in the 53 joints => torso_h = 1.8 x shoulder width (5.5 fallback)
RESET_FRAMES = 5
FADE_S = 0.3
BUDGET_MS, WARN_MS = 50.0, 42.0

COL_PILL = (0x0D, 0x11, 0x17)
COL_OK = (0x00, 0xFF, 0xAA)
COL_RED = (0xFF, 0x55, 0x55)
COL_AMBER = (0xFF, 0xB8, 0x4D)
COL_DIM = (0x9A, 0xA4, 0xB0)
COL_TEXT = (0xE6, 0xED, 0xF3)
PILL_ALPHA = 0.75

LEFT_WRIST, RIGHT_WRIST, L_SH, R_SH = 0, 21, 47, 48   # 42 + 5, 42 + 6
POSE_LW, POSE_RW = 51, 52                              # 42 + 9, 42 + 10


# --------------------------------------------------------------------------- #
# pure logic (unit-tested without a camera)
# --------------------------------------------------------------------------- #
class HandsDownDetector:
    """Dual-condition word boundary: hands still AND below the chest for 5 consecutive frames."""

    def __init__(self, frames: int = RESET_FRAMES):
        self.need, self.run, self.prev = frames, 0, None

    def reset(self) -> None:
        self.run, self.prev = 0, None

    @staticmethod
    def mean_hand_velocity(prev: np.ndarray, cur: np.ndarray) -> float:
        a, b = prev[:42], cur[:42]
        both = np.any(a != 0, axis=-1) & np.any(b != 0, axis=-1)
        if not both.any():
            return 0.0                                   # nothing tracked => nothing moving
        return float(np.linalg.norm(b[both, :2] - a[both, :2], axis=-1).mean())

    @staticmethod
    def hands_low(f: np.ndarray) -> bool:
        ls, rs = f[L_SH], f[R_SH]
        if not (np.any(ls != 0) and np.any(rs != 0)):
            return False                                 # cannot judge without both shoulders
        torso = TORSO_PER_SHOULDER * float(np.linalg.norm(ls[:2] - rs[:2]))
        sh_y = 0.5 * (ls[1] + rs[1])
        wrists = []
        for hand_idx, pose_idx in ((LEFT_WRIST, POSE_LW), (RIGHT_WRIST, POSE_RW)):
            if np.any(f[hand_idx] != 0):
                wrists.append(f[hand_idx, 1])
            elif np.any(f[pose_idx] != 0):
                wrists.append(f[pose_idx, 1])
        if not wrists:
            return True                                  # hands out of view with the body visible == hands down
        return all(y > sh_y + BELOW_FRAC * torso for y in wrists)

    def update(self, frame: np.ndarray) -> bool:
        vel = 0.0 if self.prev is None else self.mean_hand_velocity(self.prev, frame)
        self.prev = frame.copy()
        self.run = self.run + 1 if (vel < VEL_THRESH and self.hands_low(frame)) else 0
        if self.run >= self.need:
            self.run = 0
            return True
        return False


class Voter:
    """Last 3 predictions; confirmed iff >= 2 agree AND the winning confidence >= 0.60."""

    def __init__(self):
        self.q: deque[str] = deque(maxlen=VOTES)

    def clear(self) -> None:
        self.q.clear()

    def push(self, gloss: str, conf: float) -> str:
        """-> state in {'uncertain','tentative','confirmed'}."""
        self.q.append(gloss)
        if conf < CONF_UNCERTAIN:
            return "uncertain"
        if list(self.q).count(gloss) >= VOTES_NEEDED and conf >= CONF_CONFIRM:
            return "confirmed"
        return "tentative"


class Speaker:
    """Non-blocking offline TTS: daemon thread + Queue(maxsize=1); a full queue drops the stale item."""

    def __init__(self, enabled: bool = True, engine_factory: Callable | None = None):
        self.q: queue.Queue[str] = queue.Queue(maxsize=1)
        self.last_spoken: str | None = None
        self.ok = enabled
        self._factory = engine_factory
        self.spoken: list[str] = []
        if enabled:
            threading.Thread(target=self._run, daemon=True, name="tts").start()

    def _run(self) -> None:
        try:
            if self._factory is not None:
                engine = self._factory()
            else:
                import pyttsx3

                engine = pyttsx3.init()  # created inside the worker thread (SAPI5 COM is thread-affine)
        except Exception as e:  # noqa: BLE001
            print(f"[hud] TTS unavailable ({e!r}); continuing silently", file=sys.stderr)
            self.ok = False
            return
        while True:
            word = self.q.get()
            try:
                engine.say(word.replace("-", " ").replace("_", " ").lower())
                engine.runAndWait()
                self.spoken.append(word)
            except Exception as e:  # noqa: BLE001
                print(f"[hud] TTS error ({e!r}); disabling speech", file=sys.stderr)
                self.ok = False
                return

    def say(self, gloss: str) -> bool:
        """Speak only when it differs from the last spoken gloss. Never blocks."""
        if not self.ok or gloss == self.last_spoken:
            return False
        self.last_spoken = gloss
        try:
            self.q.put_nowait(gloss)
        except queue.Full:
            try:
                self.q.get_nowait()          # drop the stale item, keep the newest
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(gloss)
            except queue.Full:
                pass
        return True

    def reset(self) -> None:
        self.last_spoken = None


class NetCounter:
    """psutil system-wide NIC byte delta since launch (read-only, local), refreshed once per second."""

    def __init__(self, reader: Callable | None = None):
        self._read = reader or self._psutil_read
        self.base = self._safe()
        self.sent = self.recv = 0
        self._t = 0.0

    @staticmethod
    def _psutil_read():
        import psutil

        c = psutil.net_io_counters()
        return c.bytes_sent, c.bytes_recv

    def _safe(self):
        try:
            return self._read()
        except Exception:  # noqa: BLE001
            return None

    def poll(self, now: float | None = None) -> tuple[int, int]:
        now = time.monotonic() if now is None else now
        if now - self._t >= 1.0 and self.base is not None:
            self._t = now
            cur = self._safe()
            if cur is not None:
                self.sent, self.recv = max(0, cur[0] - self.base[0]), max(0, cur[1] - self.base[1])
        return self.sent, self.recv


class Fade:
    """Frame-time based caption fade: start() -> alpha runs 1 -> 0 over FADE_S seconds."""

    def __init__(self):
        self.alpha, self.active = 1.0, False

    def start(self) -> None:
        self.alpha, self.active = 1.0, True

    def tick(self, dt: float) -> float:
        if self.active:
            self.alpha = max(0.0, self.alpha - dt / FADE_S)
            if self.alpha <= 0.0:
                self.active = False
        return self.alpha


def latency_color(ms: float):
    return COL_OK if ms <= WARN_MS else (COL_AMBER if ms <= BUDGET_MS else COL_RED)


def fmt_bytes(n: int) -> str:
    return f"{n} B" if n < 1024 else f"{n / 1024:.1f} KB" if n < 1024**2 else f"{n / 1024**2:.1f} MB"


# --------------------------------------------------------------------------- #
# rendering (Pillow RGBA layer alpha-blended onto the OpenCV frame)
# --------------------------------------------------------------------------- #
class HudRenderer:
    def __init__(self, font_dir: Path | None = None):
        from PIL import ImageFont

        fd = font_dir or ROOT / "assets" / "fonts"
        reg, bold = fd / "DejaVuSans.ttf", fd / "DejaVuSans-Bold.ttf"
        for f in (reg, bold):
            if not f.is_file():
                raise FileNotFoundError(f"bundled font missing: {f} (fonts are never downloaded)")
        self._bold = bold
        self._big: dict[int, object] = {}
        self.f_big = ImageFont.truetype(str(bold), 54)
        self.f_mid = ImageFont.truetype(str(bold), 20)
        self.f_sm = ImageFont.truetype(str(reg), 15)
        self.f_xs = ImageFont.truetype(str(reg), 13)
        self._pill_cache: dict[tuple[int, int], object] = {}
        self.render_ms: deque[float] = deque(maxlen=300)

    def _fit_big(self, d, text: str, max_w: int):
        """Largest bold size (54 -> 24) whose rendered text fits max_w; fonts cached."""
        from PIL import ImageFont

        for size in range(54, 23, -4):
            f = self._big.get(size) or self._big.setdefault(size, ImageFont.truetype(str(self._bold), size))
            bb = d.textbbox((0, 0), text, font=f)
            if bb[2] - bb[0] <= max_w:
                return f
        return f

    def _pill(self, w: int, h: int):
        from PIL import Image, ImageDraw

        if (w, h) not in self._pill_cache:  # static layer rendered once
            im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(im).rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2 - 4,
                                                 fill=(*COL_PILL, int(255 * PILL_ALPHA)))
            self._pill_cache[(w, h)] = im
        return self._pill_cache[(w, h)].copy()

    def render(self, frame: np.ndarray, s: dict) -> np.ndarray:
        """frame BGR uint8 -> new BGR frame with the HUD. s = state dict built by the loop."""
        from PIL import Image, ImageDraw, ImageFilter

        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        boxes: list[tuple[int, int, int, int]] = []   # regions actually drawn (x0, y0, x1, y1)

        # ---- centre pill: gloss + confidence bar (fades + blurs on reset) -------------------------
        pw, ph = min(int(w * 0.62), 560), 128
        px, py = (w - pw) // 2, int(h * 0.70) - ph // 2
        status, a = s.get("status", "searching"), float(s.get("caption_alpha", 1.0))
        pill = self._pill(pw, ph)
        pd = ImageDraw.Draw(pill)
        if status == "searching":
            self._center(pd, "Searching for hands...", self.f_mid, pw, 50, COL_DIM)
        elif status == "uncertain":
            self._center(pd, "Uncertain...", self.f_mid, pw, 50, COL_DIM)
        else:
            colour = COL_OK if status == "confirmed" else COL_DIM          # tentative = dimmed grey
            gtxt = str(s.get("gloss", "")).upper()
            self._center(pd, gtxt, self._fit_big(pd, gtxt, pw - 60), pw, 12, colour)
            conf = float(np.clip(s.get("conf", 0.0), 0.0, 1.0))
            bx0, bx1, by = 40, pw - 40, ph - 26
            pd.rounded_rectangle((bx0, by, bx1, by + 8), radius=4, fill=(60, 68, 78, 255))
            pd.rounded_rectangle((bx0, by, bx0 + max(8, int((bx1 - bx0) * conf)), by + 8), radius=4, fill=(*colour, 255))
            pd.text((bx1 - 40, by - 18), f"{conf * 100:.0f}%", font=self.f_xs, fill=(*COL_TEXT, 255))
        if s.get("predicting"):   # pulsing dot while the model is working but nothing is confirmed yet
            r = 4 + 2 * (0.5 + 0.5 * np.sin(s.get("t", 0.0) * 6.0))
            pd.ellipse((30 - r, 30 - r, 30 + r, 30 + r), fill=(*COL_OK, 220))
        if a < 1.0:               # hands-down reset: ~0.3 s blur + fade
            pill = pill.filter(ImageFilter.GaussianBlur(radius=6.0 * (1.0 - a)))
            pill.putalpha(pill.getchannel("A").point(lambda v, k=a: int(v * k)))
        if a > 0.0:
            layer.alpha_composite(pill, (px, py))
            boxes.append((px, py, px + pw, py + ph))

        # ---- top-right: E2E latency | FPS (live), 50 ms budget bar with a red limit marker ---------
        lat, fps = s.get("lat_ms"), s.get("fps")
        have = lat is not None and fps is not None
        txt = f"E2E Latency: {lat:.0f}ms | FPS: {fps:.0f}" if have else "E2E Latency: -- | FPS: --"
        col = latency_color(lat) if lat is not None else COL_DIM
        boxes.append(self._chip(d, txt, self.f_mid, w - 16, 14, col, right=True))
        bx0, bx1, by = w - 16 - 260, w - 16, 56
        d.rounded_rectangle((bx0, by, bx1, by + 8), radius=4, fill=(*COL_PILL, 190))
        scale = 260 / 70.0   # bar spans 0..70 ms
        if lat is not None:
            d.rounded_rectangle((bx0, by, bx0 + max(6, min(260, int(lat * scale))), by + 8), radius=4, fill=(*col, 255))
        mx = bx0 + int(BUDGET_MS * scale)
        d.rectangle((mx - 1, by - 5, mx + 1, by + 13), fill=(*COL_RED, 255))
        d.text((mx - 16, by + 14), "50ms", font=self.f_xs, fill=(*COL_RED, 255))
        boxes.append((bx0 - 4, by - 6, bx1 + 4, by + 32))
        k = s.get("k", 3)
        tele = (f"Camera: {fps:.0f} FPS | Landmark Rate: {fps / k:.0f} Hz | E2E: {lat:.0f} ms"
                if have else "Camera: -- FPS | Landmark Rate: -- Hz | E2E: -- ms")
        boxes.append(self._chip(d, tele, self.f_xs, w - 16, 90, COL_TEXT, right=True))

        # ---- bottom-left: air-gapped badge + network delta -----------------------------------------
        tx, rx = s.get("net", (0, 0))
        boxes.append(self._chip(d, f"AIR-GAPPED | system network delta: {fmt_bytes(tx)} TX / {fmt_bytes(rx)} RX",
                                self.f_sm, 16, h - 44, COL_TEXT, dot=COL_OK))

        # ---- bottom-right: live top-5 ---------------------------------------------------------------
        top5 = s.get("top5") or []
        boxes.append(self._chip(d, "Top-5: [" + ", ".join(top5) + "]" if top5 else "Top-5: [ ]", self.f_sm, w - 16,
                                h - 88, COL_TEXT, right=True))

        ov = np.asarray(layer)
        out = frame.copy()
        for x0, y0, x1, y1 in boxes:                  # integer alpha blend, only where the HUD drew something
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            if x1 <= x0 or y1 <= y0:
                continue
            reg = ov[y0:y1, x0:x1]
            al = reg[..., 3:4].astype(np.uint16)
            dst = out[y0:y1, x0:x1].astype(np.uint16)
            out[y0:y1, x0:x1] = ((dst * (255 - al) + reg[..., 2::-1].astype(np.uint16) * al) // 255).astype(np.uint8)
        self.render_ms.append((time.perf_counter() - t0) * 1000.0)
        return out

    @staticmethod
    def _center(d, text, font, width, y, colour):
        bb = d.textbbox((0, 0), text, font=font)
        d.text(((width - (bb[2] - bb[0])) // 2 - bb[0], y), text, font=font, fill=(*colour, 255))

    @staticmethod
    def _chip(d, text, font, x, y, colour, right=False, dot=None):
        bb = d.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad, dotw = 10, (18 if dot else 0)
        w = tw + 2 * pad + dotw
        x0 = x - w if right else x
        d.rounded_rectangle((x0, y, x0 + w, y + th + 2 * pad - 2), radius=12, fill=(*COL_PILL, int(255 * PILL_ALPHA)))
        if dot:
            cy = y + (th + 2 * pad - 2) // 2
            d.ellipse((x0 + pad - 2, cy - 5, x0 + pad + 8, cy + 5), fill=(*dot, 255))
        d.text((x0 + pad + dotw - bb[0], y + pad - 2 - bb[1]), text, font=font, fill=(*colour, 255))
        return x0, y, x0 + w + 1, y + th + 2 * pad


# --------------------------------------------------------------------------- #
# the live loop
# --------------------------------------------------------------------------- #
def open_source(source: str, loop: bool = False):
    """-> (frame iterator, mirrored). Camera index "0" is mirrored (selfie view = what the model expects)."""
    import cv2

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            raise SystemExit(f"[hud] cannot open camera {source}. Check that it is connected and not used by another "
                             "app, or pass --source <video file | frame directory>.")

        def cam():
            try:
                while True:
                    ok, f = cap.read()
                    if not ok:
                        return
                    yield f
            finally:
                cap.release()

        return cam(), True
    from src.predict import _iter_frames

    it, _, _ = _iter_frames(Path(source))
    if not loop:
        return it, False
    cached = list(it)

    def forever():
        while True:
            yield from cached

    return forever(), False


def fit_canvas(frame: np.ndarray, min_w: int = 640, min_h: int = 480) -> np.ndarray:
    """Scale + letterbox small sources (e.g. 256x256 archive frames) onto a >= 640x480 canvas so the HUD layout fits."""
    import cv2

    h, w = frame.shape[:2]
    if w >= min_w and h >= min_h:
        return frame
    sc = min(min_w / w, min_h / h) if (w < min_w or h < min_h) and max(w / min_w, h / min_h) < 1 else 1.0
    sc = max(sc, min_h / h) if h < min_h else sc
    img = cv2.resize(frame, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_LINEAR)
    H, W = img.shape[:2]
    top, left = max(0, (min_h - H) // 2), max(0, (min_w - W) // 2)
    return cv2.copyMakeBorder(img, top, max(0, min_h - H - top), left, max(0, min_w - W - left), cv2.BORDER_CONSTANT, value=(13, 17, 23))


def run(a: argparse.Namespace) -> int:
    import cv2

    from src import predict as P

    frames, mirrored = open_source(a.source, a.loop)
    image_mode = (not a.source.isdigit()) and Path(a.source).is_dir()
    sp = P.StreamingPredictor(P._extractor(image_mode=image_mode), k=1 if image_mode else a.frame_skip, window=WINDOW)
    P._vocab(), P._session()  # load once (ONNX session + landmarkers are warmed up with 20 dummy steps at load)
    hud, voter, down, fade = HudRenderer(), Voter(), HandsDownDetector(), Fade()
    speaker, net = Speaker(enabled=not a.no_tts), NetCounter()
    st = {"status": "searching", "k": sp.k, "top5": [], "gloss": "", "conf": 0.0, "predicting": False}
    step_ms: deque[float] = deque(maxlen=120)
    t_prev = start = time.perf_counter()
    fps_ema, n = None, 0
    print("[hud] headless run." if a.headless else "[hud] running. Press q to quit.", flush=True)
    for raw in frames:
        frame = cv2.flip(raw, 1) if mirrored else raw
        out, ms = sp.step(frame, infer=(sp.idx % STRIDE == 0) and len(sp.ring) >= MIN_WINDOW)
        step_ms.append(ms)
        now = time.perf_counter()
        dt, t_prev = now - t_prev, now
        fps_ema = 1.0 / max(dt, 1e-6) if fps_ema is None else 0.9 * fps_ema + 0.1 / max(dt, 1e-6)
        n += 1

        latest = sp.ring.as_array()[-1] if len(sp.ring) else np.zeros((53, 3), np.float32)
        hands_now = bool(np.any(latest[:42] != 0))
        if down.update(latest) and not fade.active and st["status"] in ("tentative", "confirmed", "uncertain"):
            sp.reset(); voter.clear(); speaker.reset(); down.reset()   # word boundary: flush buffer + votes + spoken
            fade.start(); st["predicting"] = False
        elif out is not None and not fade.active:
            pred, top5, conf = out
            st["top5"] = top5
            if hands_now:
                st.update(status=voter.push(pred, conf), gloss=pred, conf=conf)
                st["predicting"] = st["status"] != "confirmed"
                if st["status"] == "confirmed":
                    speaker.say(pred)
        a_now = fade.tick(dt)
        if not fade.active and a_now <= 0.0:       # fade finished: clear the caption
            st.update(status="searching", top5=[], gloss="", conf=0.0, predicting=False)
            fade.alpha = 1.0
        elif not hands_now and not fade.active:
            st.update(status="searching", predicting=False)
        if len(step_ms) >= 10:
            st["lat_ms"] = float(np.percentile(step_ms, 95)) if a.lat_stat == "p95" else float(np.mean(step_ms))
        st.update(fps=fps_ema, caption_alpha=fade.alpha if fade.active else 1.0, t=now - start, net=net.poll())
        shown = hud.render(fit_canvas(frame), st)
        if a.save_frame and n == a.max_frames:
            cv2.imwrite(a.save_frame, shown)
        if not a.headless:
            cv2.imshow("EchoGesture", shown)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        if a.max_frames and n >= a.max_frames:
            break
    if not a.headless:
        cv2.destroyAllWindows()
    r = np.array(hud.render_ms) if hud.render_ms else np.zeros(1)
    s95 = float(np.percentile(step_ms, 95)) if step_ms else float("nan")
    print(f"[hud] frames={n} | HUD render ms: mean={r.mean():.2f} p95={np.percentile(r, 95):.2f} | "
          f"streaming step ms p95={s95:.1f} | spoken={speaker.spoken}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EchoGesture live XR HUD (air-gapped)")
    ap.add_argument("--source", default="0", help="camera index, video file, or directory of frames")
    ap.add_argument("--frame_skip", type=int, default=3)
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--headless", action="store_true", help="no window (testing)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--save-frame", default=None, help="write HUD frame number --max-frames to this PNG")
    ap.add_argument("--loop", action="store_true", help="loop a file/frame-dir source (testing)")
    ap.add_argument("--lat-stat", choices=("p95", "mean"), default="p95", help="statistic in the E2E latency chip")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
