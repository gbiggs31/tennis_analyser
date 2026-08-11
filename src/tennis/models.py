"""Data model.

Time is always seconds (float) from the start of the source video — never frame
indices. The pipeline mixes a 120fps original with a 30fps proxy, and frame
numbers across those two timebases are a reliable source of off-by-N bugs.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Player(StrEnum):
    NEAR = "near"
    FAR = "far"
    UNKNOWN = "unknown"


class Stroke(StrEnum):
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    SERVE = "serve"
    VOLLEY = "volley"
    OVERHEAD = "overhead"
    SLICE_BACKHAND = "slice_backhand"
    UNKNOWN = "unknown"


class OnsetFeatures(BaseModel):
    """Acoustic descriptors measured over a short window at the onset.

    These separate racket strikes from court bounces: a strike has a faster
    attack and more high-frequency energy than a ball landing on the surface.
    """

    peak_db: float
    spectral_centroid_hz: float
    attack_time_ms: float
    hf_lf_ratio: float
    onset_strength: float


class Hit(BaseModel):
    """A candidate ball-contact event detected in the audio track."""

    index: int
    t_audio: float = Field(description="Onset time as heard at the microphone")
    t_contact: float = Field(
        description="Onset time corrected for sound propagation delay. Equals "
        "t_audio until a player has been attributed."
    )
    features: OnsetFeatures
    player: Player = Player.UNKNOWN
    player_confidence: float = 0.0
    is_bounce: bool = False
    bounce_confidence: float = 0.0

    # Loud enough to have happened on *our* court rather than an adjacent one.
    # Set in stage 1 from an automatic level split.
    #
    # This deliberately says nothing about which player, or even whether it was
    # a racket strike: review of real clips confirmed the loud population mixes
    # near-side shots, far-side shots, ball bounces and net cords. Distance to
    # the microphone separates our court from the courts beside it, and nothing
    # finer. Player attribution is stage 2; event type is settled at labelling.
    is_own_court: bool = False


class Shot(BaseModel):
    index_in_point: int
    t_contact: float
    t_start: float
    t_end: float
    player: Player
    is_serve: bool = False
    is_second_serve: bool = False
    stroke: Stroke = Stroke.UNKNOWN
    clip_path: str | None = None
    sheet_path: str | None = None
    detect_confidence: float = 0.0
    verified: bool = False

    @property
    def slug(self) -> str:
        return f"s{self.index_in_point:02d}"


class Point(BaseModel):
    index: int
    t_start: float
    t_end: float
    server: Player = Player.UNKNOWN
    had_fault: bool = False
    shots: list[Shot] = Field(default_factory=list)

    @property
    def slug(self) -> str:
        return f"p{self.index:02d}"


class ProbeInfo(BaseModel):
    path: str
    size_bytes: int = 0
    duration_s: float
    fps: float
    fps_nominal: float | None = None
    fps_average: float | None = None
    width: int
    height: int
    video_codec: str | None = None
    n_frames: int | None = None
    rotation: int = 0
    has_audio: bool = False
    audio: dict | None = None

    # Whether the container's rotation flag was honoured when building the proxy
    # and clips. Recorded so later stages interpret pixel coordinates correctly.
    autorotate_applied: bool = False
    proxy_width: int | None = None
    proxy_height: int | None = None
    proxy_fps: float | None = None


class HitsFile(BaseModel):
    """Stage 1 artifact."""

    session_id: str
    audio_duration_s: float
    threshold_k: float
    own_court_threshold_db: float | None = None
    hits: list[Hit]

    @property
    def own_court(self) -> list[Hit]:
        return [h for h in self.hits if h.is_own_court]


class PointsFile(BaseModel):
    """Stage 2 artifact."""

    session_id: str
    points: list[Point]
