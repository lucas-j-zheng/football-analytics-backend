import numpy as np
import pandas as pd
from pathlib import Path


def generate(n: int = 50000, seed: int = 42) -> pd.DataFrame:
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
    df["weather_temp"] = rng.normal(55, 15, size=n).round(1)
    df["weather_wind"] = rng.normal(7, 5, size=n).round(1)
    df["weather_rain"] = rng.integers(0, 2, size=n)
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate()
    out = out_dir / "fake_team_history.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


