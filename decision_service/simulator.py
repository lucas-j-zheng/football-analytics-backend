from __future__ import annotations

from typing import Dict
import numpy as np


def _predict_ep_with_possession(ep_model, base: Dict, yardline_offense_view: float, possession: int) -> float:
    """Predict EP for the offense given a yardline expressed in offense view.
    Note: EP model does NOT take possession flag; we convert yardline to the possessor's perspective and use the 9 EP features.
    """
    yardline_for_model = yardline_offense_view if possession == 1 else 100.0 - yardline_offense_view
    x = [
        base["down"],
        base["ydstogo"],
        yardline_for_model,
        base["qtr"],
        base["time_remaining"],
        base["score_diff"],
        base["offense_timeouts"],
        base["defense_timeouts"],
        1 if base["home"] else 0,
    ]
    return float(ep_model.predict([x])[0])


def simulate_actions_to_wp(ep_model, wp_calibrated, features: Dict) -> Dict[str, Dict[str, float]]:
    """
    Crude deterministic simulator mapping: for each action (GO, PUNT, FG),
    approximate next EP and convert to WP via calibrated classifier on context.
    """
    base = features.copy()
    yardline = float(base["yardline_100"])  # distance to opponent goal
    togo = float(base["ydstogo"]) if base["ydstogo"] else 1.0

    # Simple outcome approximations
    # GO: convert with prob p_conv; if convert, new set around +min(6, togo+1) yards, else turnover at same spot
    p_conv = 0.65 if togo <= 2 else 0.5 if togo <= 4 else 0.3
    go_success_yardline_off = min(99.0, yardline + min(6.0, togo + 1.0))
    # Fail -> opponent ball at same spot
    go_fail_yardline_off = yardline
    ep_go_success = _predict_ep_with_possession(ep_model, base, go_success_yardline_off, possession=1)
    ep_go_fail = _predict_ep_with_possession(ep_model, base, go_fail_yardline_off, possession=0)
    ep_go = p_conv * ep_go_success + (1 - p_conv) * ep_go_fail

    # PUNT: typical net around 38 yards; if near midfield, pin deep
    punt_net = 38.0
    y_after_punt_off = max(1.0, yardline - punt_net)  # offense view distance to opponent goal
    ep_punt = _predict_ep_with_possession(ep_model, base, y_after_punt_off, possession=0)

    # FG: make prob scales by distance (yardline_100 ~ field position)
    dist_fg = 118 - yardline  # rough yards to posts
    p_make = 0.95 - 0.01 * max(0, dist_fg - 25)
    p_make = float(np.clip(p_make, 0.05, 0.98))
    # On make: +3 points, kickoff -> opponent at 25 (offense view yardline=75), possession=0
    ep_after_kick = _predict_ep_with_possession(ep_model, base, 75.0, possession=0)
    ep_if_make = 3.0 + ep_after_kick
    # On miss: opponent ball at approximate spot of kick (7 yards deeper from LOS)
    y_off_miss = max(1.0, yardline - 7.0)
    ep_if_miss = _predict_ep_with_possession(ep_model, base, y_off_miss, possession=0)
    ep_fg = p_make * ep_if_make + (1 - p_make) * ep_if_miss

    # Convert EP deltas to WP via calibrated model using the same context features
    wp_go = _ep_to_wp(wp_calibrated, base, ep_go, possession=1)
    wp_punt = _ep_to_wp(wp_calibrated, base, ep_punt, possession=0)
    # For FG, possession after play is defense
    wp_fg = _ep_to_wp(wp_calibrated, base, ep_fg, possession=0)

    return {
        "GO": {"wp": wp_go, "ep": ep_go},
        "PUNT": {"wp": wp_punt, "ep": ep_punt},
        "FG": {"wp": wp_fg, "ep": ep_fg},
    }


def _ep_to_wp(wp_calibrated, base: Dict, ep_value: float, possession: int) -> float:
    x = [
        base["down"],
        base["ydstogo"],
        base["yardline_100"],
        base["qtr"],
        base["time_remaining"],
        base["score_diff"],
        base["offense_timeouts"],
        base["defense_timeouts"],
        1 if base["home"] else 0,
        possession,
        base.get("team_strength_off", 0.0),
        base.get("team_strength_def", 0.0),
    ]
    # Lightly blend EP influence without overriding score context
    x_mod = x.copy()
    x_mod[5] = float(x_mod[5]) + 0.2 * ep_value
    prob = float(wp_calibrated.predict_proba([x_mod])[0, 1])
    # Sanity clamp using score and possession to avoid absurd values
    score = float(base["score_diff"])
    time = float(base["time_remaining"])
    bias = 1.0 / (1.0 + np.exp(0.18 * (-(score))))  # lower if trailing
    pos_adj = 0.06 if possession == 1 else -0.06
    blended = 0.7 * prob + 0.3 * (0.5 * bias + pos_adj + 0.5)
    return float(np.clip(blended, 0.01, 0.99))


