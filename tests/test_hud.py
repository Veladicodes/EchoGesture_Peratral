"""HUD logic (project spec 5.5 / 13): hands-down reset, 2/3 vote tiers, non-blocking TTS, net counter, rendering."""
import threading
import time
from pathlib import Path

import numpy as np

from src import hud_demo as H

LOW = 0.4 + 0.35 * 0.36 + 0.05   # clearly below the "shoulder + 0.35 torso" line for the pose() geometry


def pose(lw_y: float, rw_y: float = None, hands: bool = True) -> np.ndarray:
    """(53,3) frame: shoulders at y=0.4 (width 0.2 => torso_h 0.36), hand wrists at lw_y / rw_y."""
    f = np.zeros((53, 3), np.float32)
    f[H.L_SH] = (0.6, 0.4, 0.0)
    f[H.R_SH] = (0.4, 0.4, 0.0)
    if hands:
        rw_y = lw_y if rw_y is None else rw_y
        f[H.LEFT_WRIST] = (0.6, lw_y, 0.0)
        f[H.RIGHT_WRIST] = (0.4, rw_y, 0.0)
        f[1:21] = f[H.LEFT_WRIST] + 0.01
        f[22:42] = f[H.RIGHT_WRIST] + 0.01
    return f


def test_hands_down_needs_both_conditions_for_five_frames():
    d = H.HandsDownDetector()
    fired = [d.update(pose(LOW)) for _ in range(6)]
    assert fired == [False, False, False, False, True, False]   # 5th still frame fires, then the run restarts


def test_hands_up_and_still_does_not_reset():
    d = H.HandsDownDetector()
    assert not any(d.update(pose(0.45)) for _ in range(20))     # wrists above the line


def test_hands_low_but_moving_does_not_reset():
    d = H.HandsDownDetector()
    res = []
    for i in range(20):
        f = pose(LOW)
        f[:42, 0] += 0.02 * (i % 2)      # 0.02 per-frame jump >> 0.005 threshold
        res.append(d.update(f))
    assert not any(res)


def test_run_must_be_consecutive():
    d = H.HandsDownDetector()
    seq = [LOW] * 4 + [0.45] + [LOW] * 4
    assert not any(d.update(pose(y)) for y in seq)


def test_no_hands_visible_with_body_counts_as_down():
    d = H.HandsDownDetector()
    assert [d.update(pose(0, hands=False)) for _ in range(5)][-1] is True


def test_no_shoulders_never_resets():
    d = H.HandsDownDetector()
    assert not any(d.update(np.zeros((53, 3), np.float32)) for _ in range(20))


def test_voter_confidence_tiers_and_majority():
    v = H.Voter()
    assert v.push("A", 0.30) == "uncertain"
    assert v.push("A", 0.50) == "tentative"          # 0.40-0.60: shown dimmed, not confirmed
    assert v.push("A", 0.59) == "tentative"          # 3/3 agree but conf < 0.60
    assert v.push("A", 0.61) == "confirmed"
    v.clear()
    assert v.push("A", 0.9) == "tentative"           # 1/3 agree: not confirmed even with high confidence
    assert v.push("B", 0.9) == "tentative"
    assert v.push("B", 0.9) == "confirmed"           # 2 of last 3 agree


class FakeEngine:
    def __init__(self, gate: threading.Event | None = None):
        self.said, self.gate = [], gate

    def say(self, t):
        self.said.append(t)

    def runAndWait(self):
        if self.gate:
            self.gate.wait(2)


def _wait(cond, t=2.0):
    end = time.time() + t
    while time.time() < end and not cond():
        time.sleep(0.01)
    return cond()


def test_speaker_speaks_once_per_gloss_until_reset():
    eng = FakeEngine()
    s = H.Speaker(engine_factory=lambda: eng)
    assert s.say("HELLO") and not s.say("HELLO")
    assert _wait(lambda: eng.said == ["hello"])
    s.reset()
    assert s.say("HELLO")
    assert _wait(lambda: eng.said == ["hello", "hello"])


def test_speaker_never_blocks_and_drops_stale_items():
    gate = threading.Event()
    eng = FakeEngine(gate)
    s = H.Speaker(engine_factory=lambda: eng)
    t0 = time.perf_counter()
    for g in ("A", "B", "C", "D", "E"):
        s.say(g)                                    # engine is stuck speaking: must not block
    assert time.perf_counter() - t0 < 0.2
    assert s.q.maxsize == 1
    gate.set()
    assert _wait(lambda: "e" in eng.said)           # the newest survives; stale ones were dropped
    assert len(eng.said) < 5


def test_speaker_survives_tts_init_failure():
    def boom():
        raise RuntimeError("no speech engine")

    s = H.Speaker(engine_factory=boom)
    assert _wait(lambda: s.ok is False)
    assert s.say("HELLO") is False                   # silent, no exception


def test_net_counter_reports_delta_since_launch():
    vals = iter([(100, 200), (150, 260)])
    c = H.NetCounter(reader=lambda: next(vals))
    assert c.poll(now=10.0) == (50, 60)
    assert c.poll(now=10.5) == (50, 60)              # refreshed at most once per second


def test_fade_runs_in_about_0_3_s_frame_time_based():
    f = H.Fade()
    f.start()
    assert abs(f.tick(0.15) - 0.5) < 1e-6
    assert f.tick(0.2) == 0.0 and not f.active


def test_latency_colour_thresholds():
    assert H.latency_color(30) == H.COL_OK
    assert H.latency_color(46) == H.COL_AMBER
    assert H.latency_color(60) == H.COL_RED


def test_renderer_draws_all_states_without_crashing_and_keeps_shape():
    r = H.HudRenderer()
    frame = np.full((480, 640, 3), 90, np.uint8)
    base = {"k": 3, "fps": 28.0, "lat_ms": 31.0, "net": (0, 0), "top5": ["HELLO", "HELP", "HAND", "HOME", "HAPPY"]}
    for st in ("searching", "uncertain", "tentative", "confirmed"):
        out = r.render(frame, {**base, "status": st, "gloss": "hello", "conf": 0.8, "predicting": True, "t": 1.0})
        assert out.shape == frame.shape and out.dtype == np.uint8 and not np.array_equal(out, frame)
    faded = r.render(frame, {**base, "status": "confirmed", "gloss": "hello", "conf": 0.8, "caption_alpha": 0.4})
    assert faded.shape == frame.shape
    assert len(r.render_ms) == 5


def test_renderer_shows_placeholders_before_first_measurement():
    out = H.HudRenderer().render(np.zeros((360, 640, 3), np.uint8), {"status": "searching", "net": (0, 0)})
    assert out.shape == (360, 640, 3)


def test_bundled_fonts_exist_and_licence_present():
    fd = Path(H.__file__).resolve().parents[1] / "assets" / "fonts"
    assert (fd / "DejaVuSans.ttf").is_file() and (fd / "DejaVuSans-Bold.ttf").is_file()
    assert (fd / "LICENSE_DEJAVU").is_file()
