from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class RecommendQuery(BaseModel):
    down: int = Field(ge=1, le=4)
    ydstogo: int = Field(ge=1, le=100)
    yardline_100: int = Field(ge=1, le=99, description="Distance to opponent goal line (1-99)")
    time_remaining: int = Field(ge=0, description="Seconds remaining in half or game context")
    qtr: int = Field(ge=1, le=5, description="1-4, 5=OT")
    score_diff: int = Field(description="Offense score minus defense score")
    offense_timeouts: int = Field(ge=0, le=3)
    defense_timeouts: int = Field(ge=0, le=3)
    home: bool
    weather_temp: Optional[float] = None
    weather_wind: Optional[float] = None
    weather_rain: Optional[bool] = None
    possession: Optional[Literal["offense", "defense"]] = "offense"
    team_strength_off: Optional[float] = None
    team_strength_def: Optional[float] = None


class Alternative(BaseModel):
    action: Literal["GO", "PUNT", "FG", "KNEEL", "QB_SNEAK"]
    wp: float
    ep: float


class Uncertainty(BaseModel):
    std: float
    method: Literal["bootstrap", "delta"]


class RecommendResponse(BaseModel):
    recommendation: Literal["GO", "PUNT", "FG"]
    delta_wp: float
    delta_ep: float
    alternatives: List[Alternative]
    rationale: List[str]
    uncertainty: Uncertainty
    version: str


class BulkRequest(BaseModel):
    items: List[RecommendQuery]


class BulkResponseItem(RecommendResponse):
    pass


class BulkResponse(BaseModel):
    items: List[BulkResponseItem]


