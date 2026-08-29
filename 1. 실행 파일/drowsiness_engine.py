"""
EZdrive 졸음 판단 엔진 (라즈베리파이 실시간 추론)

구성
  CalibrationCollector : 10초 캘리브레이션(3초 뜸 / 3초 감음 / 4초 깜빡임)에서 개인 기준값 추출
  FeatureBuffer        : 롤링 윈도우로 학습 때와 동일한 피처 생성
  DrowsinessDetector   : RF 추론 + 개인 임계값 보정 + 상태(0/2/3) 산출

사용 흐름
  1) 캘리브레이션: CalibrationCollector.feed()를 10초간 호출 -> finish()로 CalibrationRef 획득
  2) 주행 중: DrowsinessDetector.update()를 매 프레임 호출 -> (state, ir_reliable) 반환
"""

import json
import time
from collections import deque
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import joblib

# ===== 캘리브레이션 구간 (초) =====
PHASE_OPEN_END = 3.0      # 0~3초: 정면 응시(눈 뜬 상태)
PHASE_CLOSED_END = 6.0    # 3~6초: 눈 감고 유지
PHASE_BLINK_END = 10.0    # 6~10초: 감았다 뜨기 반복

CALIB_MIN_GAP = 30.0      # open_ref - closed_ref 가 이보다 작으면 캘리브레이션 실패

# ===== 적응형 open_ref =====
ADAPT_GATE = 0.4          # 캘리브 기준 정규화값이 이 값 초과면 '눈 뜬 상태'로 보고 갱신 후보에 포함
ADAPT_WINDOW = 90.0       # 갱신 후보 유지 시간(초)
ADAPT_PCTL = 40           # 후보값의 이 퍼센타일로 갱신
ADAPT_ALPHA = 0.03        # 지수평활 계수
ADAPT_LO, ADAPT_HI = 0.5, 2.0   # 캘리브 open_ref 대비 허용 배율
ADAPT_MIN_N = 50          # 후보가 이만큼 쌓여야 갱신 시작

# ===== 피처 윈도우 =====
ROLL_SEC = 3.0            # 롤링 윈도우 길이(초)
CLOSED_THRESHOLD = 0.3    # ir_norm 이 값 미만이면 '눈 감김'
YAWN_RECENT_SEC = 10.0    # 최근 하품 판정 구간

# ===== 판정 =====
PERSONAL_PCTL = 75        # 주행 초반 정상구간 확률의 이 퍼센타일을 임계값으로
PERSONAL_MARGIN = 0.0
TH_LO, TH_HI = 0.20, 0.85
WARMUP_SEC = 60.0         # 이 시간 동안 개인 임계값 수집(그 전에는 기본 임계값 사용)
DEFAULT_TH = 0.5
SLEEP_DURATION_SEC = 3.0  # 눈감김이 이 시간 이상이면 state 3(수면), 미만이면 2(졸음)

# ===== IR 신뢰도 =====
IR_UNRELIABLE_SEC = 5.0   # 아래 조건이 이 시간 이상 지속되면 신뢰 불가
IR_STUCK_STD = 3.0        # 롤링 표준편차가 이보다 작고
IR_STUCK_LEVEL = 0.5      # 정규화값이 이보다 낮은 상태로 계속 머무름


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
    """10초 캘리브레이션 데이터를 모아 개인 기준값을 만든다."""

    def __init__(self):
        self.t0 = None
        self.last_t = None
        self.samples = []   # (elapsed, ir, yaw, pitch)

    def feed(self, ir, yaw, pitch, now=None):
        """매 프레임 호출. 반환값은 현재 안내 문구(음성/화면 표시용)."""
        # now=0.0 도 유효한 값이므로 `or` 대신 None 검사를 쓴다
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
            return "정면을 보세요"
        if el < PHASE_CLOSED_END:
            return "눈을 감으세요"
        if el < PHASE_BLINK_END:
            return "눈을 감았다 뜨기를 반복하세요"
        return "완료"

    @property
    def elapsed(self):
        """마지막으로 feed된 시각 기준 경과시간 (CSV 재생 시에도 올바르게 동작)."""
        if self.t0 is None or self.last_t is None:
            return 0.0
        return self.last_t - self.t0

    def is_done(self):
        return self.elapsed >= PHASE_BLINK_END

    def finish(self):
        """기준값 산출 + 품질 검증. 실패 시 valid=False."""
        if not self.samples:
            return CalibrationRef(0, 0, 0, 0, 1, valid=False)

        arr = np.array([(e, ir, y, p) for e, ir, y, p in self.samples], dtype=float)
        el, ir, yaw, pitch = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        m_open = el < PHASE_OPEN_END
        m_closed = (el >= PHASE_OPEN_END) & (el < PHASE_CLOSED_END)
        m_blink = (el >= PHASE_CLOSED_END) & (el < PHASE_BLINK_END)

        if m_open.sum() < 5 or m_closed.sum() < 5:
            return CalibrationRef(0, 0, 0, 0, 1, valid=False)

        # 평균이 아닌 퍼센타일: 깜빡임으로 낮아진 값이 open_ref를 끌어내리지 않게
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
    """학습 때와 동일한 피처를 실시간으로 생성한다."""

    def __init__(self, ref: CalibrationRef):
        self.ref = ref
        self.o_cur = ref.open_ref          # 적응형 open_ref
        self.buf = deque()                 # (t, ir, ir_norm)
        self.adapt_t, self.adapt_v = deque(), deque()
        self.yawn_t = deque()              # 하품 발생 시각
        self.prev_ir = None
        self.closed_run = 0.0              # 연속 눈감김 지속시간
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

        # ---- 적응형 open_ref: '확실히 눈 뜬' 프레임만 갱신 후보로 ----
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

        # ---- 버퍼 갱신 ----
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

        # ---- 연속 눈감김 지속시간 ----
        dt = (now - self.prev_t) if self.prev_t is not None else 0.0
        self.prev_t = now
        if not np.isnan(ir_norm) and ir_norm < CLOSED_THRESHOLD:
            self.closed_run += dt
        else:
            self.closed_run = 0.0

        # ---- 롤링 통계 ----
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
    """RF 추론 + 개인 임계값 보정 + 상태 산출."""

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
        """매 프레임 호출. (state, ir_reliable, info) 반환."""
        now = time.time() if now is None else now
        if self.start_t is None:
            self.start_t = now

        feats, roll_std, ir_norm = self.fb.update(
            now, ir, yaw, pitch, yaw_vel, pitch_vel, is_yawning, mouth_open_duration)

        # 학습 때와 동일한 컬럼명/순서로 넘겨야 sklearn 경고가 뜨지 않는다
        x = pd.DataFrame([[feats[k] for k in self.feature_order]],
                         columns=self.feature_order)
        prob = float(self.model.predict_proba(x)[0, 1])

        # ---- 개인 임계값 보정: 주행 초반을 정상으로 보고 그 분포에서 임계값 산출 ----
        elapsed = now - self.start_t
        if not self.threshold_fixed:
            if elapsed < WARMUP_SEC:
                self.warmup_probs.append(prob)
            else:
                if len(self.warmup_probs) >= 30:
                    th = np.percentile(self.warmup_probs, PERSONAL_PCTL) + PERSONAL_MARGIN
                    self.threshold = float(np.clip(th, TH_LO, TH_HI))
                self.threshold_fixed = True

        danger = prob >= self.threshold

        # ---- IR 신뢰도: 값이 낮게 붙은 채 거의 움직이지 않으면 센서 이탈 의심 ----
        stuck = (roll_std < IR_STUCK_STD) and (ir_norm < IR_STUCK_LEVEL)
        if stuck:
            if self.ir_bad_since is None:
                self.ir_bad_since = now
            # 실제 수면도 같은 패턴이므로, 하품 등 각성 신호가 함께 있을 때만 이상으로 본다
            if (now - self.ir_bad_since) >= IR_UNRELIABLE_SEC and feats["yawn_recent"]:
                self.ir_reliable = False
        else:
            self.ir_bad_since = None
            self.ir_reliable = True

        # ---- 상태 매핑: 위험이면 눈감김 지속시간으로 2/3 구분 ----
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
