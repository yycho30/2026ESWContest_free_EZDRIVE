"""
EZdrive forward-attention (inattention) engine -- state 1.

Flow
  External camera (front-facing) YOLO tracking -> obstacle bearing + risk level
  Internal camera yaw -> did the driver look that way?
  If not confirmed within the reaction window -> inattention (state 1)

Risk levels
  LEVEL_OBSTACLE : ordinary obstacle. One look clears it.
  LEVEL_DANGER   : object got close enough to be dangerous.
                   Promotion to this level requires a fresh look.

Left/right convention
  The external camera faces forward, the internal camera faces the driver,
  so the two look in opposite directions.
  Obstacle bearing: screen-right (= vehicle's right) is positive.
  Head angle: multiplied by HEAD_YAW_SIGN so that turning right is positive.
              The sign of solvePnP yaw flips with camera mirroring and the
              chosen coordinate frame, so it must be fixed by measurement
              once (see calibrate_yaw_sign).
"""

import time
from dataclasses import dataclass

import numpy as np

# ===== External camera =====
CAMERA_HFOV_DEG = 102.0      # Raspberry Pi Camera Module 3 Wide, horizontal FOV
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_YAW_OFFSET_DEG = 0.0  # correction if the external camera is not aimed straight ahead

# ===== Internal camera (head pose) =====
HEAD_YAW_SIGN = +1.0         # set by measurement: +1 if yaw grows when turning right, else -1
HEAD_YAW_OFFSET_DEG = 0.0    # driver's neutral yaw from calibration, if available

# ===== Decision parameters =====
# Measured: obstacles placed where a driver naturally turns to look were at
# -18.7deg / +22.3deg, and the head turned -24.6deg / +24.7deg in response.
PERIPHERAL_ANGLE_DEG = 17.0  # obstacles beyond this bearing must be checked by turning the head
REACTION_TIME_SEC = 2.0      # inattention if not confirmed within this window
CONFIRM_RATIO = 0.5          # turning this fraction of the obstacle bearing counts as a look
CONFIRM_MIN_DEG = 25.0        # but at least this many degrees (measured natural response ~25deg)

# ===== Risk levels, by bounding-box area ratio =====
# Large vehicles look big even far away, while a person only looks big up close,
# so the thresholds are per class.
# "person" values are measured (obstacle ~10m, danger ~5m); the rest are scaled
# from person as a rough placeholder until measured directly.
AREA_THRESHOLDS = {
    #  class            obstacle   danger
    "person":        (0.005,     0.025),   # measured: 10m=0.0062, 5m=0.028, 2m=0.1835
    "bicycle":       (0.008,     0.035),
    "motorcycle":    (0.008,     0.035),
    "car":           (0.020,     0.090),
    "bus":           (0.035,     0.150),
    "truck":         (0.035,     0.150),
}
DEFAULT_AREA_THRESHOLD = (0.015, 0.075)

LEVEL_NONE = 0
LEVEL_OBSTACLE = 1
LEVEL_DANGER = 2

# Drop a track this long after it was last seen
TRACK_EXPIRE_SEC = 3.0


def bbox_to_angle(x1, x2, frame_width=FRAME_WIDTH,
                  hfov=CAMERA_HFOV_DEG, offset=CAMERA_YAW_OFFSET_DEG):
    """
    Convert the horizontal centre of a bounding box into a bearing.
    Screen centre is 0 deg, right is positive.
    Uses a pinhole model so bearings stay accurate near the frame edges.
    """
    cx = (x1 + x2) / 2.0
    nx = (cx - frame_width / 2.0) / (frame_width / 2.0)   # normalised, -1 .. +1
    focal = 1.0 / np.tan(np.radians(hfov / 2.0))
    angle = np.degrees(np.arctan2(nx, focal))
    return float(angle + offset)


def bbox_area_ratio(x1, y1, x2, y2,
                    frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT):
    """Fraction of the frame covered by the bounding box."""
    w = max(x2 - x1, 0)
    h = max(y2 - y1, 0)
    return (w * h) / float(frame_width * frame_height)


def risk_level(class_name, area_ratio):
    """Map area ratio to a risk level for this class."""
    lo, hi = AREA_THRESHOLDS.get(class_name, DEFAULT_AREA_THRESHOLD)
    if area_ratio >= hi:
        return LEVEL_DANGER
    if area_ratio >= lo:
        return LEVEL_OBSTACLE
    return LEVEL_NONE


def normalize_head_yaw(yaw_deg):
    """Convert internal-camera yaw to the 'right is positive' convention."""
    if yaw_deg is None:
        return None
    return HEAD_YAW_SIGN * (yaw_deg - HEAD_YAW_OFFSET_DEG)


@dataclass
class TrackedObject:
    """State of one tracked obstacle."""
    track_id: int
    class_name: str
    angle: float = 0.0
    area_ratio: float = 0.0
    level: int = LEVEL_NONE
    last_seen: float = 0.0

    # Highest level the driver has already looked at (NONE = nothing confirmed yet)
    confirmed_level: int = LEVEL_NONE
    # When the current level started demanding a look
    pending_since: float = None
    # Level we already warned about, so one level warns only once
    warned_level: int = LEVEL_NONE

    def needs_check(self):
        """True if the current level still demands a look."""
        if self.level == LEVEL_NONE:
            return False
        if abs(self.angle) < PERIPHERAL_ANGLE_DEG:
            return False          # inside the forward view, no head turn needed
        return self.confirmed_level < self.level

    def confirm_threshold(self):
        """Signed head angle that counts as having looked."""
        need = max(abs(self.angle) * CONFIRM_RATIO, CONFIRM_MIN_DEG)
        return np.sign(self.angle) * need


class InattentionDetector:
    """
    Call update() every frame to get the forward-attention state.

    Usage:
        det = InattentionDetector()
        state, info = det.update(detections, head_yaw)
        # detections: [{'track_id': 1, 'class_name': 'car', 'bbox': (x1, y1, x2, y2)}, ...]
        # state: 0 (ok) or 1 (inattention)
    """

    def __init__(self):
        self.tracks = {}          # track_id -> TrackedObject
        self.last_warn_time = None

    def update(self, detections, head_yaw, now=None):
        now = time.time() if now is None else now
        head = normalize_head_yaw(head_yaw)

        # ---- 1. refresh tracks ----
        for d in detections:
            tid = d.get("track_id")
            if tid is None:
                continue          # without a track id we cannot manage levels

            x1, y1, x2, y2 = d["bbox"]
            angle = bbox_to_angle(x1, x2)
            area = bbox_area_ratio(x1, y1, x2, y2)
            level = risk_level(d["class_name"], area)

            obj = self.tracks.get(tid)
            if obj is None:
                obj = TrackedObject(track_id=tid, class_name=d["class_name"])
                self.tracks[tid] = obj

            prev_level = obj.level
            obj.angle = angle
            obj.area_ratio = area
            obj.level = level
            obj.last_seen = now

            # Promotion (obstacle -> danger) means the driver must look again
            if level > prev_level:
                obj.pending_since = None      # restarted below
                if level > obj.confirmed_level:
                    obj.warned_level = LEVEL_NONE   # new level, allow a warning again

            if obj.needs_check():
                if obj.pending_since is None:
                    obj.pending_since = now
            else:
                obj.pending_since = None

        # ---- 2. clear objects the driver looked at ----
        if head is not None:
            for obj in self.tracks.values():
                if not obj.needs_check():
                    continue
                th = obj.confirm_threshold()
                # obstacle on the right (+): head >= th; on the left (-): head <= th
                looked = (head >= th) if th > 0 else (head <= th)
                if looked:
                    obj.confirmed_level = obj.level
                    obj.pending_since = None

        # ---- 3. decide ----
        state = 0
        offenders = []
        for obj in self.tracks.values():
            if not obj.needs_check() or obj.pending_since is None:
                continue
            if (now - obj.pending_since) >= REACTION_TIME_SEC and obj.warned_level < obj.level:
                state = 1
                offenders.append(obj)

        if state == 1:
            for obj in offenders:
                obj.warned_level = obj.level      # do not warn twice for the same level
            self.last_warn_time = now

        # ---- 4. drop stale tracks ----
        for tid in [t for t, o in self.tracks.items()
                    if now - o.last_seen > TRACK_EXPIRE_SEC]:
            del self.tracks[tid]

        info = {
            "head_yaw_norm": head,
            "n_tracks": len(self.tracks),
            "pending": [
                {
                    "id": o.track_id, "class": o.class_name,
                    "angle": round(o.angle, 1),
                    "level": o.level,
                    "area": round(o.area_ratio, 4),
                    "wait_s": round(now - o.pending_since, 2) if o.pending_since else 0.0,
                }
                for o in self.tracks.values() if o.needs_check() and o.pending_since
            ],
            "offenders": [
                {"id": o.track_id, "class": o.class_name,
                 "angle": round(o.angle, 1), "level": o.level}
                for o in offenders
            ],
        }
        return state, info


def calibrate_yaw_sign(samples):
    """
    Helper for fixing HEAD_YAW_SIGN by measurement.

    samples: [(yaw_deg, direction), ...] where direction is the real head
             turn: +1 for right, -1 for left.
    Put the returned value into HEAD_YAW_SIGN.
    """
    if not samples:
        return +1.0
    score = sum(np.sign(y) * d for y, d in samples if y is not None)
    return +1.0 if score >= 0 else -1.0
