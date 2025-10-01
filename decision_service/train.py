from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, brier_score_loss
from .models_store import save_model
from .db import engine, init_metrics_schema
from sqlalchemy import text
import time


FEATURES_EP = [
    "down",
    "ydstogo",
    "yardline_100",
    "qtr",
    "time_remaining",
    "score_diff",
    "offense_timeouts",
    "defense_timeouts",
    "home",
]

FEATURES_WP = FEATURES_EP + [
    "possession",
    "team_strength_off",
    "team_strength_def",
]


def _simulate_fake_history(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame()
    df["down"] = rng.integers(1, 5, size=n)
    df["ydstogo"] = rng.integers(1, 20, size=n)
    df["yardline_100"] = rng.integers(1, 99, size=n)
    df["qtr"] = rng.integers(1, 5, size=n)
    df["time_remaining"] = rng.integers(0, 3600, size=n)
    df["score_diff"] = rng.integers(-28, 28, size=n)
    df["offense_timeouts"] = rng.integers(0, 4, size=n)
    df["defense_timeouts"] = rng.integers(0, 4, size=n)
    df["home"] = rng.integers(0, 2, size=n)
    df["possession"] = 1  # offense perspective
    df["team_strength_off"] = rng.normal(0, 1, size=n)
    df["team_strength_def"] = rng.normal(0, 1, size=n)

    # Expected points ground truth (synthetic but monotonic-ish):
    # closer to endzone (higher yardline_100) → higher EP; more time and better score diff → higher EP
    base_ep = (
        0.08 * df["yardline_100"]
        + 0.02 * (3600 - df["time_remaining"]) / 60.0
        + 0.05 * df["score_diff"]
        - 0.1 * np.maximum(df["ydstogo"] - 3, 0)
        - 0.15 * (df["down"] - 1)
        + 0.05 * df["home"]
    )
    noise_ep = rng.normal(0, 0.7, size=n)
    df["ep_next_score"] = base_ep + noise_ep

    # Win probability proxy: logistic of EP and time/score context
    lin_wp = (
        0.15 * df["ep_next_score"]
        + 0.002 * df["time_remaining"]
        + 0.06 * df["score_diff"]
        + 0.05 * df["home"]
        + 0.03 * df["team_strength_off"]
        - 0.03 * df["team_strength_def"]
    )
    df["wp"] = 1 / (1 + np.exp(-lin_wp))

    # Label for calibration examples (binary outcome win)
    df["win"] = (rng.random(n) < df["wp"]).astype(int)
    return df


def train_models(fake_team: str = "Demo Team", seed: int = 42, csv_path: str | None = None) -> Dict[str, float]:
    start = time.time()
    if csv_path:
        raw = pd.read_csv(csv_path)
        # Minimal mapping if user provides nflfastR-like schema
        colmap = {
            "down": "down",
            "ydstogo": "ydstogo",
            "yardline_100": "yardline_100",
            "qtr": "qtr",
            "time_remaining": "half_seconds_remaining",
            "score_diff": "score_differential",
            "offense_timeouts": "posteam_timeouts_remaining",
            "defense_timeouts": "defteam_timeouts_remaining",
            "home": "home",
        }
        df = pd.DataFrame()
        for out_col, src_col in colmap.items():
            src = src_col
            if src not in raw.columns and out_col in raw.columns:
                src = out_col
            if src not in raw.columns:
                # fallback defaults
                if out_col in ("offense_timeouts", "defense_timeouts"):
                    df[out_col] = 3
                elif out_col == "home":
                    df[out_col] = 1
                else:
                    df[out_col] = 0
                continue
            df[out_col] = raw[src]
        df["possession"] = 1
        df["team_strength_off"] = 0.0
        df["team_strength_def"] = 0.0
        # crude EP/WP targets if not available
        if "ep_next_score" not in raw.columns:
            base_ep = (
                0.08 * df["yardline_100"]
                + 0.02 * (3600 - df["time_remaining"]) / 60.0
                + 0.05 * df["score_diff"]
                - 0.1 * np.maximum(df["ydstogo"] - 3, 0)
                - 0.15 * (df["down"] - 1)
                + 0.05 * df["home"]
            )
            df["ep_next_score"] = base_ep
        if "win" not in raw.columns:
            lin_wp = (
                0.15 * df["ep_next_score"]
                + 0.002 * df["time_remaining"]
                + 0.06 * df["score_diff"]
                + 0.05 * df["home"]
            )
            prob = 1 / (1 + np.exp(-lin_wp))
            df["win"] = (np.random.random(len(df)) < prob).astype(int)
    else:
        df = _simulate_fake_history(seed=seed)

    # EP model
    X_ep = df[FEATURES_EP]
    y_ep = df["ep_next_score"]
    X_train, X_val, y_train, y_val = train_test_split(X_ep, y_ep, test_size=0.2, random_state=seed)
    ep_model = GradientBoostingRegressor(random_state=seed)
    ep_model.fit(X_train, y_train)
    ep_mae = mean_absolute_error(y_val, ep_model.predict(X_val))

    # WP base model + calibration
    X_wp = df[FEATURES_WP]
    y_win = df["win"]
    X_train2, X_val2, y_train2, y_val2 = train_test_split(X_wp, y_win, test_size=0.2, random_state=seed)
    wp_base = LogisticRegression(max_iter=200)
    wp_base.fit(X_train2, y_train2)
    # Platt scaling
    calibrator = CalibratedClassifierCV(wp_base, method="sigmoid", cv="prefit")
    calibrator.fit(X_train2, y_train2)
    preds = calibrator.predict_proba(X_val2)[:, 1]
    wp_brier = brier_score_loss(y_val2, preds)

    version = time.strftime("%Y-%m")
    save_model("ep_model", version, ep_model, None, {"mae": float(ep_mae)})
    save_model("wp_model", version, wp_base, calibrator, {"brier": float(wp_brier)})

    # Log model record
    init_metrics_schema()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO models(name, version, metrics_json)
                VALUES (:n, :v, :m)
                ON CONFLICT(name, version) DO UPDATE SET metrics_json=excluded.metrics_json
                """
            ),
            {"n": "ep_model", "v": version, "m": pd.Series({"mae": float(ep_mae)}).to_json()},
        )
        conn.execute(
            text(
                """
                INSERT INTO models(name, version, metrics_json)
                VALUES (:n, :v, :m)
                ON CONFLICT(name, version) DO UPDATE SET metrics_json=excluded.metrics_json
                """
            ),
            {"n": "wp_model", "v": version, "m": pd.Series({"brier": float(wp_brier)}).to_json()},
        )

    return {"ep_mae": float(ep_mae), "wp_brier": float(wp_brier), "version": version, "train_seconds": round(time.time() - start, 3)}


