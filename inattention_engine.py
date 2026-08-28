"""
EZdrive Forward Inattention Detection Engine (State 1)

Operation
  External camera (Front) YOLO tracking -> Calculates left/right angle of the obstacle + risk level
  Internal camera yaw -> Checks if the driver looked in that direction
  If not checked within a certain time, classified as forward inattention (State 1)

Risk Levels
  LEVEL_OBSTACLE : Normal obstacle. Cleared after looking once.
  LEVEL_DANGER   : Object that became dangerous due to proximity. Must be re-checked if the level increases.

Left/Right Direction Convention
  External camera faces forward, internal camera faces the driver (opposite directions).
  Obstacle angle: Right side of the screen (=right side of the car) is positive (+).
  Head angle  : Multiplied by HEAD_YAW_SIGN so that "turning right is positive (+)".
                The sign of solvePnP yaw flips depending on camera mirroring/coordinate systems, 
                so it must be determined by an actual measurement (see calibrate_yaw_sign below).
"""

import time
from dataclasses import dataclass, field

import numpy as np

# ===== External Camera =====
CAMERA_HFOV_DEG = 102.0      # Raspberry Pi Camera Module 3 Wide horizontal FOV
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_YAW_OFFSET_DEG = 0.0  # Calibration if the external camera is skewed from the center

# ===== Internal Camera (Head Angle) =====
HEAD_YAW_SIGN = +1.0         # Determined by measurement: +1 if yaw increases when turning right, -1 if it decreases
HEAD_YAW_OFFSET_DEG = 0.0    # Frontal reference value obtained from calibration (if available)

# ===== Detection Parameters =====
PERIPHERAL_ANGLE_DEG = 17.0  # Obstacles appearing outside this angle require turning the head to check
REACTION_TIME_SEC = 2.0      # Inattention if not checked within this time after appearance
CONFIRM_RATIO = 0.5          # Acknowledged as checked if the head is turned by this ratio of the obstacle angle
CONFIRM_MIN_DEG = 20.0        # However, the head must be turned by at least this minimum angle

# ===== Risk Levels (Bounding Box Area Ratio) =====
# Large objects appear large even from afar, while people only appear large when close -> Different thresholds per class
AREA_THRESHOLDS = {
    # Class          Obstacle Level  Danger Level
    "person":        (0.010,         0.060),
    "bicycle":       (0.015,         0.080),
    "motorcycle":    (0.015,         0.080),
    "car":           (0.040,         0.180),
    "bus":           (0.070,         0.300),
    "truck":         (0.070,         0.300),
}
DEFAULT_AREA_THRESHOLD = (0.030, 0.150)

LEVEL_NONE = 0
LEVEL_OBSTACLE = 1
LEVEL_DANGER = 2

# Discard the record if this time passes after the object disappears from the screen
TRACK_EXPIRE_SEC = 3.0


def bbox_to_angle(x1, x2, frame_width=FRAME_WIDTH,
                  hfov=CAMERA_HFOV_DEG, offset=CAMERA_YAW_OFFSET_DEG):
    """
    Converts the left/right center of the bounding box to an angle.
    Screen center is 0 degrees, right is +.
    Uses a pinhole model to reduce distortion at the edges of the screen.
    """
    cx = (x1 + x2) / 2.0
    # Normalized coordinates based on the center (-1 to +1)
    nx = (cx - frame_width / 2.0) / (frame_width / 2.0)
    focal = 1.0 / np.tan(np.radians(hfov / 2.0))
    angle = np.degrees(np.arctan2(nx, focal))
    return float(angle + offset)


def bbox_area_ratio(x1, y1, x2, y2,
                    frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT):
    """The area ratio the bounding box occupies on the screen."""
    w = max(x2 - x1, 0)
    h = max(y2 - y1, 0)
    return (w * h) / float(frame_width * frame_height)


def risk_level(class_name, area_ratio):
    """Determines the risk level based on the area ratio."""
    lo, hi = AREA_THRESHOLDS.get(class_name, DEFAULT_AREA_THRESHOLD)
    if area_ratio >= hi:
        return LEVEL_DANGER
    if area_ratio >= lo:
        return LEVEL_OBSTACLE
    return LEVEL_NONE


def normalize_head_yaw(yaw_deg):
    """Converts internal camera yaw to the 'right is +' convention."""
    if yaw_deg is None:
        return None
    return HEAD_YAW_SIGN * (yaw_deg - HEAD_YAW_OFFSET_DEG)


@dataclass
class TrackedObject:
    """The state of a single tracked obstacle."""
    track_id: int
    class_name: str
    angle: float = 0.0
    area_ratio: float = 0.0
    level: int = LEVEL_NONE
    last_seen: float = 0.0

    # The driver has confirmed up to this level (LEVEL_NONE = nothing confirmed yet)
    confirmed_level: int = LEVEL_NONE
    # The time when the current level started requiring confirmation
    pending_since: float = None
    # Has an inattention warning already been issued for this level (prevents duplicate warnings for the same level)
    warned_level: int = LEVEL_NONE

    def needs_check(self):
        """Does the current level require confirmation?"""
        if self.level == LEVEL_NONE:
            return False
        if abs(self.angle) < PERIPHERAL_ANGLE_DEG:
            return False          # Within frontal vision - no need to turn the head
        return self.confirmed_level < self.level

    def confirm_threshold(self):
        """The head angle (including sign) acknowledged as confirmation."""
        need = max(abs(self.angle) * CONFIRM_RATIO, CONFIRM_MIN_DEG)
        return np.sign(self.angle) * need


class InattentionDetector:
    """
    Calling update() every frame returns whether there is forward inattention.

    Usage:
        det = InattentionDetector()
        state, info = det.update(detections, head_yaw)
        # detections: [{'track_id':1,'class_name':'car','bbox':(x1,y1,x2,y2)}, ...]
        # state: 0 (Normal) or 1 (Forward Inattention)
    """

    def __init__(self):
        self.tracks = {}          # track_id -> TrackedObject
        self.last_warn_time = None

    def update(self, detections, head_yaw, now=None):
        now = time.time() if now is None else now
        head = normalize_head_yaw(head_yaw)

        # ---- 1. Update Tracking Info ----
        seen_ids = set()
        for d in detections:
            tid = d.get("track_id")
            if tid is None:
                continue          # Skip detections without a track ID as level management is impossible
            seen_ids.add(tid)

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

            # Re-confirmation is required if the level increases (Obstacle -> Danger)
            if level > prev_level:
                obj.pending_since = None      # Start fresh below
                if level > obj.confirmed_level:
                    obj.warned_level = LEVEL_NONE   # Allow warning again for the new level

            # Record the time confirmation was requested
            if obj.needs_check():
                if obj.pending_since is None:
                    obj.pending_since = now
            else:
                obj.pending_since = None

        # ---- 2. Process Confirmation by Head Direction ----
        if head is not None:
            for obj in self.tracks.values():
                if not obj.needs_check():
                    continue
                th = obj.confirm_threshold()
                # If obstacle is right (+), head >= th; if left (-), head <= th
                looked = (head >= th) if th > 0 else (head <= th)
                if looked:
                    obj.confirmed_level = obj.level
                    obj.pending_since = None

        # ---- 3. Inattention Detection ----
        state = 0
        offenders = []
        for obj in self.tracks.values():
            if not obj.needs_check() or obj.pending_since is None:
                continue
            elapsed = now - obj.pending_since
            if elapsed >= REACTION_TIME_SEC and obj.warned_level < obj.level:
                state = 1
                offenders.append(obj)

        if state == 1:
            for obj in offenders:
                obj.warned_level = obj.level      # Do not warn again for the same level
            self.last_warn_time = now

        # ---- 4. Clean up Old Tracks ----
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
    Helper to determine HEAD_YAW_SIGN via actual measurement.

    samples: [(yaw_deg, direction), ...]
             direction is the actual turned head direction. Right +1, Left -1.
    Assign the return value to HEAD_YAW_SIGN.
    """
    if not samples:
        return +1.0
    score = sum(np.sign(y) * d for y, d in samples if y is not None)
    return +1.0 if score >= 0 else -1.0
