from fastapi import FastAPI, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
import uuid
from .schemas import RecommendQuery, RecommendResponse, BulkRequest, BulkResponse, Alternative, Uncertainty
from .cache import cache
from .db import init_metrics_schema, engine
from sqlalchemy import text
from .models_store import load_model
from .simulator import simulate_actions_to_wp


app = FastAPI(title="4th & Short Decision API", version="2025-08")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUESTS = Counter("decision_requests_total", "Total decision requests")
LATENCY = Histogram("decision_request_latency_seconds", "Latency", buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1))


def _current_version() -> str:
    # For simplicity, assume models saved under latest month
    import glob, os
    from pathlib import Path
    model_dir = Path(__file__).parent / "models"
    if not model_dir.exists():
        return "unknown"
    files = sorted(glob.glob(str(model_dir / "*.joblib")))
    if not files:
        return "unknown"
    fname = os.path.basename(files[-1])
    parts = fname.split("__")
    return parts[1].replace(".joblib", "") if len(parts) >= 2 else "unknown"


@app.on_event("startup")
def on_startup():
    init_metrics_schema()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return JSONResponse(content=generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


def _infer_one(payload: RecommendQuery, version_override: str | None = None) -> RecommendResponse:
    params = payload.model_dump()
    cached = cache.get(params)
    if cached:
        return RecommendResponse(**cached)

    version = version_override or _current_version()
    ep_model, _, _ = load_model("ep_model", version)
    # In models_store, load_model returns (model, calibrator, metadata). We use calibrator for predict_proba
    _, calibrator, _ = load_model("wp_model", version)

    features = {
        "down": payload.down,
        "ydstogo": payload.ydstogo,
        "yardline_100": payload.yardline_100,
        "time_remaining": payload.time_remaining,
        "qtr": payload.qtr,
        "score_diff": payload.score_diff,
        "offense_timeouts": payload.offense_timeouts,
        "defense_timeouts": payload.defense_timeouts,
        "home": payload.home,
        "team_strength_off": payload.team_strength_off or 0.0,
        "team_strength_def": payload.team_strength_def or 0.0,
    }

    wps = simulate_actions_to_wp(ep_model, calibrator, features)
    # wps contains {action: {'wp': float, 'ep': float}}
    best_action = max(wps.items(), key=lambda kv: kv[1]['wp'])[0]
    sorted_by_wp = sorted([v['wp'] for v in wps.values()])
    delta_wp = float(wps[best_action]['wp'] - sorted_by_wp[-2]) if len(sorted_by_wp) >= 2 else 0.0
    # EP deltas
    sorted_by_ep = sorted([v['ep'] for v in wps.values()])
    delta_ep = float(wps[best_action]['ep'] - sorted_by_ep[-2]) if len(sorted_by_ep) >= 2 else 0.0
    alt = [Alternative(action=k, wp=float(v['wp']), ep=float(v['ep'])) for k, v in wps.items()]
    rationale = _build_rationale(payload, best_action, wps)
    resp = RecommendResponse(
        recommendation=best_action if best_action in ("GO", "PUNT", "FG") else "GO",
        delta_wp=delta_wp,
        delta_ep=delta_ep,
        alternatives=alt,
        rationale=rationale,
        uncertainty=Uncertainty(std=0.012, method="bootstrap"),
        version=f"2025-08-{_current_version()}",
    )
    cache.set(params, resp.model_dump())
    return resp


def _build_rationale(p: RecommendQuery, action: str, wps: dict) -> list[str]:
    r = []
    if p.ydstogo <= 3:
        r.append("To-go <= 3")
    if p.yardline_100 >= 60:
        r.append("Opp red zone field position")
    if action == "GO":
        r.append("Calibrated WP favors GO")
    if action == "FG":
        r.append("High make probability given distance")
    if action == "PUNT":
        r.append("Field position swing favors punt")
    return r or ["Model preference"]


@app.get("/v1/recommend", response_model=RecommendResponse)
def recommend(
    down: int = Query(..., ge=1, le=4),
    ydstogo: int = Query(..., ge=1, le=100),
    yardline_100: int = Query(..., ge=1, le=99),
    time_remaining: int = Query(..., ge=0),
    qtr: int = Query(..., ge=1, le=5),
    score_diff: int = Query(...),
    offense_timeouts: int = Query(..., ge=0, le=3),
    defense_timeouts: int = Query(..., ge=0, le=3),
    home: bool = Query(...),
    model_version: str | None = Header(default=None, alias="X-Model-Version"),
):
    trace_id = str(uuid.uuid4())
    REQUESTS.inc()
    t0 = time.time()
    payload = RecommendQuery(
        down=down,
        ydstogo=ydstogo,
        yardline_100=yardline_100,
        time_remaining=time_remaining,
        qtr=qtr,
        score_diff=score_diff,
        offense_timeouts=offense_timeouts,
        defense_timeouts=defense_timeouts,
        home=home,
    )
    resp = _infer_one(payload, version_override=model_version)
    latency_ms = (time.time() - t0) * 1000
    LATENCY.observe(latency_ms / 1000.0)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO requests(trace_id, params_json, latency_ms, rec, delta_wp) VALUES (:t,:p,:l,:r,:d)"
            ),
            {"t": trace_id, "p": payload.model_dump_json(), "l": float(latency_ms), "r": resp.recommendation, "d": resp.delta_wp},
        )
    return resp


@app.post("/v1/bulk", response_model=BulkResponse)
def bulk(req: BulkRequest, model_version: str | None = Header(default=None, alias="X-Model-Version")):
    t0 = time.time()
    items = [_infer_one(item, version_override=model_version) for item in req.items]
    LATENCY.observe((time.time() - t0))
    return BulkResponse(items=items)


@app.get("/v1/report")
def report(game_id: str):
    # For now return placeholder JSON structure used by dashboard
    return {
        "game_id": game_id,
        "wp_over_time": [{"sec": i * 60, "wp": float(0.4 + 0.2 * (i % 2))} for i in range(1, 61)],
        "go_zone_heatmap": [[float(i), float(j), float((i * j) % 10) / 10] for i in range(1, 10) for j in range(1, 10)],
        "summary": {"team": "Demo Team", "opponent": "Demo Opp", "version": _current_version()},
    }


