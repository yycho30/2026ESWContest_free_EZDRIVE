"""
EZdrive RandomForest training and validation.

  --loso    Leave-One-Subject-Out validation. Reports the honest
            "unseen driver" numbers, including the miss rate broken down
            by how long the eyes were actually closed.
  (default) Trains on all subjects and writes drowsiness_rf.pkl.

Two classes are used (normal vs at-risk). The runtime engine splits
at-risk into state 2 and 3 by eye-closure duration, because that split
is a duration threshold rather than something the model needs to learn.

Run: python3 train_final.py [--loso]
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

FEATURES = [
    # absolute-value based
    "ir_norm", "ir_roll_mean", "ir_roll_min", "perclos", "closed_duration_s",
    # change based, survives sensor misalignment
    "ir_diff_abs", "ir_roll_std", "ir_roll_range", "ir_activity_ratio",
    # head pose
    "yaw_norm", "pitch_norm", "pitch_roll_mean",
    "yaw_angular_velocity_deg_s", "pitch_angular_velocity_deg_s",
    "ang_vel_mag", "ang_roll_std",
    # yawning
    "is_yawning", "mouth_open_duration_s", "yawn_recent",
]

# yawn_count is deliberately excluded: it is a per-session running total whose
# range differs so much between subjects that the model used it as an identity
# hint rather than a drowsiness signal.

RF_PARAMS = dict(n_estimators=200, max_depth=12, min_samples_leaf=20,
                 class_weight="balanced", random_state=42, n_jobs=-1)

PERSONAL_PCTL = 75      # personal threshold percentile, matches the runtime engine


def prep(df):
    X = df[FEATURES].copy()
    X["is_yawning"] = X["is_yawning"].astype(int)
    return X.fillna(0)


def loso(A):
    """Leave-One-Subject-Out validation with the personal threshold applied."""
    rows = []
    frames = []

    for tr, te in LeaveOneGroupOut().split(A, A["binary"], groups=A["model"]):
        trd, ted = A.iloc[tr], A.iloc[te].copy()
        m = RandomForestClassifier(**RF_PARAMS)
        m.fit(prep(trd), trd["binary"])
        proba = m.predict_proba(prep(ted))[:, 1]

        # Personal threshold: take the first normal session as the warm-up
        # window the runtime engine would see, and exclude it from scoring.
        s0 = ted[ted["state"] == 0]["file"].unique()
        idx = np.where((ted["file"] == s0[0]).values)[0]
        warm = idx[:len(idx) // 2]
        th = float(np.clip(np.percentile(proba[warm], PERSONAL_PCTL), 0.20, 0.85))

        ted["proba"] = proba
        ted["pred"] = (proba >= th).astype(int)
        ted["scored"] = True
        ted.iloc[warm, ted.columns.get_loc("scored")] = False
        frames.append(ted)

        ev = ted[ted["scored"]]
        y, p = ev["binary"].values, ev["pred"].values
        rows.append({
            "subject": ted["model"].iloc[0],
            "threshold": round(th, 2),
            "acc": (p == y).mean() * 100,
            "FN": (p[y == 1] == 0).mean() * 100 if (y == 1).any() else np.nan,
            "FP": (p[y == 0] == 1).mean() * 100 if (y == 0).any() else np.nan,
        })

    R = pd.DataFrame(rows).round(1)
    print(R.to_string(index=False))
    print(f"\nmean: acc {R['acc'].mean():.1f}%  "
          f"miss {R['FN'].mean():.1f}%  false alarm {R['FP'].mean():.1f}%")

    # The number that actually matters: misses by real eye-closure duration.
    F = pd.concat(frames, ignore_index=True)
    F = F[F["scored"] & (F["binary"] == 1)].copy()
    F["bin"] = pd.cut(F["closed_duration_s"], [0, 0.5, 1, 2, 3, 5, 1e9], right=False)
    print("\nmiss rate by eye-closure duration:")
    print(F.groupby("bin", observed=True).apply(
        lambda g: pd.Series({"n": len(g), "miss_%": (g["pred"] == 0).mean() * 100})
    ).round(1))

    print("\nmiss rate by labelled state:")
    print(F.groupby("state").apply(
        lambda g: pd.Series({"n": len(g), "miss_%": (g["pred"] == 0).mean() * 100})
    ).round(1))


def train_full(A):
    m = RandomForestClassifier(**RF_PARAMS)
    m.fit(prep(A), A["binary"])
    joblib.dump(m, "drowsiness_rf.pkl")

    json.dump({"features": FEATURES,
               "n_train": int(len(A)),
               "n_subjects": int(A["model"].nunique()),
               "classes": {"0": "normal", "1": "at_risk"}},
              open("model_meta.json", "w"), indent=2)

    fi = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("top features:")
    print(fi.head(8).round(4).to_string())
    print(f"\nsaved drowsiness_rf.pkl  ({len(A)} rows, {A['model'].nunique()} subjects)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loso", action="store_true",
                    help="run Leave-One-Subject-Out validation instead of training")
    args = ap.parse_args()

    A = pd.read_csv("features_v2.csv")
    A["binary"] = (A["state"] != 0).astype(int)

    if args.loso:
        loso(A)
    else:
        train_full(A)


if __name__ == "__main__":
    main()
