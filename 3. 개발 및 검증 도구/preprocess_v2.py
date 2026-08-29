"""
EZdrive training-data preprocessing.

Turns raw session CSVs into the feature table used to train the RandomForest.

  1. Reads each subject's calibration session (the first 10 s session) and
     extracts open_ref / closed_ref / head-angle offsets.
  2. Normalises every driving session against those references, with the
     adaptive open_ref that only learns from frames judged "eyes open".
  3. Builds the same 19 features the runtime engine computes.

Input   newdata/*.csv          raw logs from face_monitor.py
        session_labels.csv     file -> subject / state mapping
Output  features_v2.csv        training table

Run: python3 preprocess_v2.py
"""

import numpy as np
import pandas as pd

DATA_DIR = "newdata/"

# ===== Calibration phases (seconds) =====
CALIB_OPEN_END = 3.0        # 0-3 s: look straight ahead, eyes open
CALIB_CLOSED_END = 6.0      # 3-6 s: eyes closed
                            # 6-10 s: repeated blinks

# ===== Adaptive open_ref =====
ADAPT_GATE = 0.4            # frames above this (fixed-calibration scale) count as "eyes open"
ADAPT_WINDOW = 90.0         # seconds of candidates kept
ADAPT_PCTL = 40             # percentile of the candidates used for the update
ADAPT_ALPHA = 0.03          # exponential smoothing factor
ADAPT_LO, ADAPT_HI = 0.5, 2.0   # clamp, relative to the calibrated open_ref
ADAPT_MIN_N = 50            # candidates needed before adapting starts

# ===== Feature windows =====
ROLL_SEC = 3.0
CLOSED_THRESHOLD = 0.3      # ir_norm below this counts as "eyes closed"
YAWN_RECENT_SEC = 10.0
FPS = 9.5                   # approximate logging rate


def load_session(path):
    d = pd.read_csv(path)
    d["ir"] = d["ir_value"].str.extract(r"(\d+)").astype(float)
    d["is_yawning"] = d["is_yawning"].astype(str).str.strip().str.lower().eq("true")
    return d.sort_values("timestamp").reset_index(drop=True)


def extract_calibration(path):
    """Personal references from one calibration session, plus a quality check."""
    d = load_session(path)
    rel = d["timestamp"] - d["timestamp"].min()
    A = d[rel <= CALIB_OPEN_END]
    B = d[(rel > CALIB_OPEN_END) & (rel <= CALIB_CLOSED_END)]
    C = d[rel > CALIB_CLOSED_END]

    # Percentiles, not means: a blink inside phase A would drag open_ref down.
    open_ref = A["ir"].quantile(0.75)
    closed_ref = B["ir"].quantile(0.25)

    return {
        "open_ref": open_ref,
        "closed_ref": closed_ref,
        "yaw_offset": A["yaw_deg"].median(),
        "pitch_offset": A["pitch_deg"].median(),
        "blink_range": max(C["ir"].quantile(0.9) - C["ir"].quantile(0.1), 1.0),
        "valid": (open_ref - closed_ref) >= 30,
    }


def normalize_with_adaptive_open(sessions, ref):
    """
    Normalise a subject's sessions in time order.

    Only frames that the fixed calibration scale already judges "eyes open"
    feed the adaptive open_ref. A plain rolling percentile would be dragged
    down by long drowsy stretches, which is exactly the failure this avoids.
    """
    o_fix, c = ref["open_ref"], ref["closed_ref"]
    o_cur = o_fix
    hist_t, hist_v = [], []
    out = []

    for d in sessions:
        norm = []
        for t, ir in zip(d["timestamp"].values, d["ir"].values):
            norm.append((ir - c) / max(o_cur - c, 1e-6))

            if (ir - c) / (o_fix - c) > ADAPT_GATE:
                hist_t.append(t)
                hist_v.append(ir)
            while hist_t and t - hist_t[0] > ADAPT_WINDOW:
                hist_t.pop(0)
                hist_v.pop(0)

            if len(hist_v) >= ADAPT_MIN_N:
                cand = np.percentile(hist_v, ADAPT_PCTL)
                o_cur = (1 - ADAPT_ALPHA) * o_cur + ADAPT_ALPHA * cand
                o_cur = float(np.clip(o_cur, o_fix * ADAPT_LO, o_fix * ADAPT_HI))

        d = d.copy()
        d["ir_norm"] = norm
        out.append(d)
    return out


def add_features(d, ref):
    """Build the 19 features the runtime engine also computes."""
    w = max(int(ROLL_SEC * FPS), 5)
    d = d.copy()

    d["yaw_norm"] = d["yaw_deg"] - ref["yaw_offset"]
    d["pitch_norm"] = d["pitch_deg"] - ref["pitch_offset"]
    d["ang_vel_mag"] = np.sqrt(
        d["yaw_angular_velocity_deg_s"] ** 2 + d["pitch_angular_velocity_deg_s"] ** 2
    )

    # --- absolute-value features (strong while the sensor is aligned) ---
    d["ir_roll_mean"] = d["ir_norm"].rolling(w, min_periods=3).mean()
    d["ir_roll_min"] = d["ir_norm"].rolling(w, min_periods=3).min()
    d["perclos"] = (d["ir_norm"] < CLOSED_THRESHOLD).rolling(w, min_periods=3).mean()

    # --- change-based features (still valid if the sensor shifts) ---
    d["ir_diff_abs"] = d["ir"].diff().abs()
    d["ir_roll_std"] = d["ir"].rolling(w, min_periods=3).std()
    d["ir_roll_range"] = (
        d["ir"].rolling(w, min_periods=3).max() - d["ir"].rolling(w, min_periods=3).min()
    )
    d["ir_activity_ratio"] = d["ir_roll_range"] / max(ref["blink_range"], 1e-6)

    # --- continuous eye-closure duration ---
    closed = (d["ir_norm"] < CLOSED_THRESHOLD).values
    ts = d["timestamp"].values
    dur = np.zeros(len(d))
    run = 0.0
    for i in range(len(d)):
        run = run + (ts[i] - ts[i - 1]) if (closed[i] and i > 0) else 0.0
        dur[i] = run
    d["closed_duration_s"] = dur

    d["ang_roll_std"] = d["ang_vel_mag"].rolling(w, min_periods=3).std()
    d["pitch_roll_mean"] = d["pitch_norm"].rolling(w, min_periods=3).mean()
    d["yawn_recent"] = (
        d["is_yawning"].astype(int).rolling(int(YAWN_RECENT_SEC * FPS), min_periods=1).max()
    )
    return d


def main():
    lab = pd.read_csv("session_labels.csv").sort_values("start").reset_index(drop=True)

    all_rows = []
    report = []

    for model in sorted(lab["model"].unique()):
        rows = lab[lab["model"] == model].sort_values("start")
        calib_file = rows[rows["state"] == "CALIB"].iloc[0]["file"]
        ref = extract_calibration(DATA_DIR + calib_file)
        report.append({"model": model,
                       "open_ref": ref["open_ref"], "closed_ref": ref["closed_ref"],
                       "gap": ref["open_ref"] - ref["closed_ref"], "valid": ref["valid"]})

        if not ref["valid"]:
            print(f"[warn] subject {model}: calibration quality too low, skipped")
            continue

        drive = rows[rows["state"] != "CALIB"]
        sessions = [load_session(DATA_DIR + f) for f in drive["file"]]
        sessions = normalize_with_adaptive_open(sessions, ref)

        for d, (_, meta) in zip(sessions, drive.iterrows()):
            d = add_features(d, ref)
            d["model"] = model
            d["state"] = int(meta["state"])
            d["file"] = meta["file"]
            all_rows.append(d)

    A = pd.concat(all_rows, ignore_index=True)
    A.to_csv("features_v2.csv", index=False)

    print(pd.DataFrame(report).round(1).to_string(index=False))
    print(f"\n{len(A)} rows from {A['model'].nunique()} subjects -> features_v2.csv")
    print("\nmedian by state:")
    cols = ["ir_norm", "perclos", "closed_duration_s", "ir_roll_std", "ir_activity_ratio"]
    print(A.groupby("state")[cols].median().round(3))


if __name__ == "__main__":
    main()
