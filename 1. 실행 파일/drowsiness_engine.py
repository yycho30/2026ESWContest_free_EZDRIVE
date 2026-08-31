"""
EZdrive drowsiness detection engine (real-time inference on the Raspberry Pi).

Components
  CalibrationCollector : extracts personal references from a 10 s calibration
                         (3 s eyes open / 3 s eyes closed / 4 s blinking)
  FeatureBuffer        : builds the same rolling-window features used in training
  DrowsinessDetector   : RF inference + personal threshold + state (0/2/3)

Usage
  1) Calibration: call CalibrationCollector.feed() for 10 s, then finish() for a CalibrationRef
  2) Driving: call DrowsinessDetector.update() every frame -> (state, ir_reliable)
"""

import json
import time
from collections import deque
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import joblib

# ===== Calibration phases (seconds) =====
PHASE_OPEN_END = 3.0      # 0-3 s: look straight ahead, eyes open
PHASE_CLOSED_END = 6.0    # 3-6 s: eyes closed
PHASE_BLINK_END = 10.0    # 6-10 s: repeated blinks

CALIB_MIN_GAP = 30.0      # calibration fails if open_ref - closed_ref is smaller than this

# ===== Adaptive open_ref =====
ADAPT_GATE = 0.4          # frames above this (fixed-calibration scale) count as "eyes open"
ADAPT_WINDOW = 90.0       # seconds of candidates kept
ADAPT_PCTL = 40           # percentile of the candidates used for the update
ADAPT_ALPHA = 0.03        # exponential smoothing factor
ADAPT_LO, ADAPT_HI = 0.5, 2.0   # clamp, relative to the calibrated open_ref
ADAPT_MIN_N = 50          # candidates needed before adapting starts

# ===== Feature windows =====
ROLL_SEC = 3.0            # rolling window length (seconds)
CLOSED_THRESHOLD = 0.3    # ir_norm below this counts as "eyes closed"
YAWN_RECENT_SEC = 10.0    # window for "yawned recently"

# ===== Decision =====
PERSONAL_PCTL = 75        # percentile of the early-drive probability distribution used as the threshold
TH_LO, TH_HI = 0.20, 0.85
WARMUP_SEC = 60.0         # personal threshold is learned over this long; the default is used before that
DEFAULT_TH = 0.65   # used only before the personal threshold is learned (first WARMUP_SEC).
                    # 0.5 was too sensitive - it flagged normal driving as drowsy.
SLEEP_DURATION_SEC = 3.0  # eyes closed at least this long -> state 3 (asleep), otherwise state 2 (drowsy)

# ===== IR reliability =====
IR_UNRELIABLE_SEC = 5.0   # unreliable once the condition below holds this long
IR_STUCK_STD = 3.0        # rolling std below this
IR_STUCK_LEVEL = 0.5      # and the normalised value stuck below this


@dataclass
class CalibrationRef:
    open_ref: float
    closed_ref: float
    yaw_offset: float
    pitch_offset: float
    blink_range: float
    valid: bool = True

    def save(self, path="calibration.json"):
        json.dump(asdict(self), open(path, "w"), indent=2)

    @staticmethod
    def load(path="calibration.json"):
        return CalibrationRef(**json.load(open(path)))


class CalibrationCollector:
    """Collects the 10 s calibration and produces the personal references."""

    def __init__(self):
        self.t0 = None
        self.last_t = None
        self.samples = []   # (elapsed, ir, yaw, pitch)

    def feed(self, ir, yaw, pitch, now=None):
        """Call every frame. Returns the current guidance text (for display/voice)."""
        # now=0.0 is a valid value too, so check for None instead of using `or`
        now = time.time() if now is None else now
        if self.t0 is None:
            self.t0 = now
        self.last_t = now
        el = now - self.t0

        if ir is not None:
            self.samples.append((el, float(ir),
                                 yaw if yaw is not None else np.nan,
                                 pitch if pitch is not None else np.nan))

        if el < PHASE_OPEN_END:
            return "Look straight ahead"
        if el < PHASE_CLOSED_END:
            return "Close your eyes"
        if el < PHASE_BLINK_END:
            return "Blink (close and open) repeatedly"
        return "Done"

    @property
    def elapsed(self):
        """Time since the first feed, based on the last feed's timestamp
        (so this also behaves correctly when replaying a CSV)."""
        if self.t0 is None or self.last_t is None:
            return 0.0
        return self.last_t - self.t0

    def is_done(self):
        return self.elapsed >= PHASE_BLINK_END

    def finish(self):
        """Compute the references and run the quality check. valid=False on failure."""
        if not self.samples:
            return CalibrationRef(0, 0, 0, 0, 1, valid=False)

        arr = np.array([(e, ir, y, p) for e, ir, y, p in self.samples], dtype=float)
        el, ir, yaw, pitch = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        m_open = el < PHASE_OPEN_END
        m_closed = (el >= PHASE_OPEN_END) & (el < PHASE_CLOSED_END)
        m_blink = (el >= PHASE_CLOSED_END) & (el < PHASE_BLINK_END)

        if m_open.sum() < 5 or m_closed.sum() < 5:
            return CalibrationRef(0, 0, 0, 0, 1, valid=False)

        # Percentiles, not means: a blink inside the "eyes open" phase would
        # otherwise drag open_ref down.
        open_ref = float(np.percentile(ir[m_open], 75))
        closed_ref = float(np.percentile(ir[m_closed], 25))
        blink_range = (float(np.percentile(ir[m_blink], 90) - np.percentile(ir[m_blink], 10))
                       if m_blink.sum() >= 5 else max(open_ref - closed_ref, 1.0))

        yaw_off = float(np.nanmedian(yaw[m_open])) if not np.all(np.isnan(yaw[m_open])) else 0.0
        pitch_off = float(np.nanmedian(pitch[m_open])) if not np.all(np.isnan(pitch[m_open])) else 0.0

        valid = (open_ref - closed_ref) >= CALIB_MIN_GAP
        return CalibrationRef(open_ref, closed_ref, yaw_off, pitch_off,
                              max(blink_range, 1.0), valid)


class FeatureBuffer:
    """Generates the same features online that were used for training."""

    def __init__(self, ref: CalibrationRef):
        self.ref = ref
        self.o_cur = ref.open_ref          # adaptive open_ref
        self.buf = deque()                 # (t, ir, ir_norm)
        self.adapt_t, self.adapt_v = deque(), deque()
        self.yawn_t = deque()              # yawn timestamps
        self.prev_ir = None
        self.closed_run = 0.0              # continuous eye-closure duration
        self.prev_t = None
        self.ang_buf = deque()             # (t, ang_vel_mag)
        self.pitch_buf = deque()           # (t, pitch_norm)

    def _trim(self, dq, now, span):
        while dq and now - dq[0][0] > span:
            dq.popleft()

    def update(self, now, ir, yaw, pitch, yaw_vel, pitch_vel,
               is_yawning, mouth_open_duration):
        r = self.ref
        c = r.closed_ref

        # ---- adaptive open_ref: only frames judged "clearly eyes open" are candidates ----
        if ir is not None:
            if (ir - c) / max(r.open_ref - c, 1e-6) > ADAPT_GATE:
                self.adapt_t.append(now)
                self.adapt_v.append(float(ir))
            while self.adapt_t and now - self.adapt_t[0] > ADAPT_WINDOW:
                self.adapt_t.popleft()
                self.adapt_v.popleft()
            if len(self.adapt_v) >= ADAPT_MIN_N:
                cand = float(np.percentile(self.adapt_v, ADAPT_PCTL))
                self.o_cur = (1 - ADAPT_ALPHA) * self.o_cur + ADAPT_ALPHA * cand
                self.o_cur = float(np.clip(self.o_cur,
                                           r.open_ref * ADAPT_LO, r.open_ref * ADAPT_HI))

        ir_norm = ((ir - c) / max(self.o_cur - c, 1e-6)) if ir is not None else np.nan

        # ---- update buffers ----
        if ir is not None:
            self.buf.append((now, float(ir), ir_norm))
        self._trim(self.buf, now, ROLL_SEC)

        ang_mag = float(np.hypot(yaw_vel or 0.0, pitch_vel or 0.0))
        self.ang_buf.append((now, ang_mag))
        self._trim(self.ang_buf, now, ROLL_SEC)

        pitch_norm = (pitch - r.pitch_offset) if pitch is not None else 0.0
        yaw_norm = (yaw - r.yaw_offset) if yaw is not None else 0.0
        self.pitch_buf.append((now, pitch_norm))
        self._trim(self.pitch_buf, now, ROLL_SEC)

        if is_yawning:
            self.yawn_t.append(now)
        while self.yawn_t and now - self.yawn_t[0] > YAWN_RECENT_SEC:
            self.yawn_t.popleft()

        # ---- continuous eye-closure duration ----
        dt = (now - self.prev_t) if self.prev_t is not None else 0.0
        self.prev_t = now
        if not np.isnan(ir_norm) and ir_norm < CLOSED_THRESHOLD:
            self.closed_run += dt
        else:
            self.closed_run = 0.0

        # ---- rolling statistics ----
        irs = np.array([b[1] for b in self.buf], dtype=float)
        norms = np.array([b[2] for b in self.buf], dtype=float)
        if len(irs) >= 2:
            roll_std = float(np.std(irs, ddof=1))
            roll_range = float(irs.max() - irs.min())
        else:
            roll_std = 0.0
            roll_range = 0.0

        ir_diff_abs = abs(float(ir) - self.prev_ir) if (ir is not None and self.prev_ir is not None) else 0.0
        if ir is not None:
            self.prev_ir = float(ir)

        angs = np.array([a[1] for a in self.ang_buf], dtype=float)
        pit = np.array([p[1] for p in self.pitch_buf], dtype=float)

        feats = {
            "ir_norm": 0.0 if np.isnan(ir_norm) else ir_norm,
            "ir_roll_mean": float(np.nanmean(norms)) if len(norms) else 0.0,
            "ir_roll_min": float(np.nanmin(norms)) if len(norms) else 0.0,
            "perclos": float(np.nanmean(norms < CLOSED_THRESHOLD)) if len(norms) else 0.0,
            "closed_duration_s": self.closed_run,
            "ir_diff_abs": ir_diff_abs,
            "ir_roll_std": roll_std,
            "ir_roll_range": roll_range,
            "ir_activity_ratio": roll_range / max(r.blink_range, 1e-6),
            "yaw_norm": yaw_norm,
            "pitch_norm": pitch_norm,
            "pitch_roll_mean": float(np.mean(pit)) if len(pit) else 0.0,
            "yaw_angular_velocity_deg_s": float(yaw_vel or 0.0),
            "pitch_angular_velocity_deg_s": float(pitch_vel or 0.0),
            "ang_vel_mag": ang_mag,
            "ang_roll_std": float(np.std(angs, ddof=1)) if len(angs) >= 2 else 0.0,
            "is_yawning": int(bool(is_yawning)),
            "mouth_open_duration_s": float(mouth_open_duration or 0.0),
            "yawn_recent": 1 if self.yawn_t else 0,
        }
        return feats, roll_std, (0.0 if np.isnan(ir_norm) else ir_norm)


class DrowsinessDetector:
    """RF inference + personal threshold correction + state output."""

    def __init__(self, ref: CalibrationRef,
                 model_path="drowsiness_rf.pkl", meta_path="model_meta.json"):
        self.model = joblib.load(model_path)
        self.feature_order = json.load(open(meta_path))["features"]
        self.fb = FeatureBuffer(ref)

        self.start_t = None
        self.warmup_probs = []
        self.threshold = DEFAULT_TH
        self.threshold_fixed = False

        self.ir_bad_since = None
        self.ir_reliable = True

    def update(self, ir, yaw, pitch, yaw_vel, pitch_vel,
               is_yawning, mouth_open_duration, now=None):
        """Call every frame. Returns (state, ir_reliable, info)."""
        now = time.time() if now is None else now
        if self.start_t is None:
            self.start_t = now

        feats, roll_std, ir_norm = self.fb.update(
            now, ir, yaw, pitch, yaw_vel, pitch_vel, is_yawning, mouth_open_duration)

        # Same column names/order as training, so sklearn doesn't warn.
        x = pd.DataFrame([[feats[k] for k in self.feature_order]],
                         columns=self.feature_order)
        prob = float(self.model.predict_proba(x)[0, 1])

        # ---- personal threshold: treat the early drive as normal and derive
        # the threshold from that probability distribution ----
        elapsed = now - self.start_t
        if not self.threshold_fixed:
            if elapsed < WARMUP_SEC:
                self.warmup_probs.append(prob)
            else:
                if len(self.warmup_probs) >= 30:
                    th = np.percentile(self.warmup_probs, PERSONAL_PCTL)
                    self.threshold = float(np.clip(th, TH_LO, TH_HI))
                self.threshold_fixed = True

        danger = prob >= self.threshold

        # ---- IR reliability: a low value that barely moves suggests the
        # sensor has drifted off the eye ----
        stuck = (roll_std < IR_STUCK_STD) and (ir_norm < IR_STUCK_LEVEL)
        if stuck:
            if self.ir_bad_since is None:
                self.ir_bad_since = now
            # Real sleep looks the same, so only flag it as a fault when an
            # alertness signal (e.g. a yawn) is present at the same time.
            if (now - self.ir_bad_since) >= IR_UNRELIABLE_SEC and feats["yawn_recent"]:
                self.ir_reliable = False
        else:
            self.ir_bad_since = None
            self.ir_reliable = True

        # ---- state mapping: if at risk, split 2/3 by eye-closure duration ----
        if not danger:
            state = 0
        elif feats["closed_duration_s"] >= SLEEP_DURATION_SEC:
            state = 3
        else:
            state = 2

        info = {"prob": prob, "threshold": self.threshold,
                "ir_norm": ir_norm, "closed_s": feats["closed_duration_s"],
                "perclos": feats["perclos"], "warmup": not self.threshold_fixed}
        return state, self.ir_reliable, info
