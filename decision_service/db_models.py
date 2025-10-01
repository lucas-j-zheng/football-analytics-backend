from dataclasses import dataclass
from typing import Optional


@dataclass
class RequestLog:
    trace_id: str
    params_json: str
    latency_ms: float
    rec: str
    delta_wp: float
    ts: Optional[str] = None


@dataclass
class ModelMeta:
    name: str
    version: str
    metrics_json: str
    created_at: Optional[str] = None


@dataclass
class BacktestRow:
    team: str
    season: int
    wp_gain_per_game: float
    epa_gain_per_game: float
    details_json: str
    created_at: Optional[str] = None


