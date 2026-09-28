from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _env_or_yaml(env_name: str, config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = os.getenv(env_name)
    if value is not None:
        return value
    return config.get(section, {}).get(key, default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    camera_source: str
    camera_uri: str
    frame_width: int
    frame_height: int
    target_fps: int
    detection_confidence: float
    tracker_max_age: int
    target_lost_seconds: float
    distance_estimation: bool
    face_width_meters: float
    camera_focal_px: float
    drone_mode: str
    root: Path

    @classmethod
    def load(cls) -> "Settings":
        _load_dotenv(ROOT / ".env")
        cfg = _yaml_config(ROOT / "config" / "config.yaml")
        return cls(
            host=str(_env_or_yaml("APP_HOST", cfg, "app", "host", "127.0.0.1")),
            port=int(_env_or_yaml("APP_PORT", cfg, "app", "port", 8000)),
            camera_source=str(_env_or_yaml("CAMERA_SOURCE", cfg, "camera", "source", "webcam")).lower(),
            camera_uri=str(_env_or_yaml("CAMERA_URI", cfg, "camera", "uri", "0")),
            frame_width=int(_env_or_yaml("FRAME_WIDTH", cfg, "camera", "width", 1280)),
            frame_height=int(_env_or_yaml("FRAME_HEIGHT", cfg, "camera", "height", 720)),
            target_fps=max(1, int(_env_or_yaml("TARGET_FPS", cfg, "camera", "target_fps", 25))),
            detection_confidence=float(_env_or_yaml("DETECTION_CONFIDENCE", cfg, "detector", "confidence", 0.55)),
            tracker_max_age=int(_env_or_yaml("TRACKER_MAX_AGE", cfg, "tracker", "max_age", 20)),
            target_lost_seconds=float(_env_or_yaml("TARGET_LOST_SECONDS", cfg, "tracking", "lost_seconds", 2.0)),
            distance_estimation=_bool(_env_or_yaml("DISTANCE_ESTIMATION", cfg, "analytics", "distance_estimation", True)),
            face_width_meters=float(_env_or_yaml("FACE_WIDTH_METERS", cfg, "analytics", "face_width_meters", 0.16)),
            camera_focal_px=float(_env_or_yaml("CAMERA_FOCAL_PX", cfg, "analytics", "focal_length_px", 900)),
            # Physical-drone modes are deliberately coerced to simulation in this release.
            drone_mode="simulation",
            root=ROOT,
        )


settings = Settings.load()
