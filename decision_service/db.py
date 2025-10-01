from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os


def get_engine():
    # Reuse existing DATABASE_URL if present; else default to local sqlite file
    database_url = os.getenv("DECISION_DATABASE_URL") or os.getenv("DATABASE_URL") or "sqlite:///football_analytics.db"
    return create_engine(database_url, pool_pre_ping=True)


engine = get_engine()
SessionLocal = sessionmaker(bind=engine)


def init_metrics_schema():
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS requests (
                  trace_id TEXT PRIMARY KEY,
                  params_json TEXT,
                  latency_ms REAL,
                  rec TEXT,
                  delta_wp REAL,
                  ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS models (
                  name TEXT,
                  version TEXT,
                  metrics_json TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  PRIMARY KEY (name, version)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS backtests (
                  team TEXT,
                  season INTEGER,
                  wp_gain_per_game REAL,
                  epa_gain_per_game REAL,
                  details_json TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  PRIMARY KEY (team, season)
                )
                """
            )
        )


