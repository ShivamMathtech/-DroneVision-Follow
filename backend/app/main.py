from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.pipeline import VisionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("dronevision")
service = VisionService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("dronevision_started mode=SIMULATION")
    yield
    service.stop()
    logger.info("dronevision_stopped")


app = FastAPI(
    title="DroneVision-Follow",
    description="Local-first, observation-mode face tracking and UAV vision telemetry.",
    version="0.1.0",
    lifespan=lifespan,
)


class CameraStart(BaseModel):
    source: str | None = None
    uri: str | None = None


class TargetSelect(BaseModel):
    track_id: int | None = Field(default=None, ge=1)
    x: float | None = Field(default=None, ge=0, le=1)
    y: float | None = Field(default=None, ge=0, le=1)


class ControllerRequest(BaseModel):
    enabled: bool = True


def _session_dir(session_id: str) -> Path:
    if not session_id.startswith("session_") or "/" in session_id or "\\" in session_id:
        raise HTTPException(status_code=400, detail="Invalid session id.")
    directory = settings.root / "sessions" / session_id
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Session not found.")
    return directory


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    path = settings.root / "frontend" / "index.html"
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "SIMULATION", "version": app.version}


@app.get("/api/system/status")
async def system_status():
    return service.status()


@app.get("/api/camera/status")
async def camera_status():
    return {
        "connected": service.status()["camera_connected"],
        "source": service.source_kind,
        "uri": service.source_uri,
    }


@app.post("/api/camera/start")
async def camera_start(request: CameraStart):
    try:
        return service.start(request.source, request.uri)
    except Exception as exc:
        logger.warning("camera_start_failed error=%s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/camera/stop")
async def camera_stop():
    return service.stop()


@app.get("/api/tracking/status")
async def tracking_status():
    status = service.status()
    return {"enabled": status["tracking_enabled"], "target": status["target"]}


@app.post("/api/tracking/start")
async def tracking_start():
    if not service.running.is_set():
        raise HTTPException(status_code=409, detail="Start a camera source first.")
    service.tracking_enabled = True
    service._event("TRACKING_STARTED", "info", "Face detection and tracking enabled.")
    return {"enabled": True}


@app.post("/api/tracking/stop")
async def tracking_stop():
    service.tracking_enabled = False
    service.release_target()
    service._event("TRACKING_STOPPED", "info", "Face detection and tracking paused.")
    return {"enabled": False}


@app.post("/api/target/select")
async def target_select(request: TargetSelect):
    if request.track_id is not None:
        result = service.select_track(request.track_id)
    elif request.x is not None and request.y is not None:
        result = service.select_target(request.x, request.y)
    else:
        raise HTTPException(status_code=422, detail="Supply a track_id or normalized x and y coordinates.")
    if not result.get("selected"):
        raise HTTPException(status_code=404, detail="No active face track was found at that selection.")
    return result


@app.post("/api/target/release")
async def target_release():
    return service.release_target()


@app.get("/api/target/current")
async def target_current():
    telemetry = service.latest()
    target_id = service.target.target_id
    target = next((track for track in telemetry.get("tracks", []) if track["track_id"] == target_id), None)
    return {**service.target.status(), "target": target, "telemetry": telemetry}


@app.get("/api/telemetry/latest")
async def telemetry_latest():
    return service.latest()


@app.get("/api/telemetry/history")
async def telemetry_history(limit: int = Query(default=500, ge=1, le=5000)):
    return {"items": service.history(limit), "count": min(limit, len(service.telemetry_history))}


@app.get("/api/telemetry/export")
async def telemetry_export(format: Literal["csv", "json"] = "csv"):
    items = service.history(5000)
    if format == "json":
        content = json.dumps(items, indent=2)
        return StreamingResponse(io.BytesIO(content.encode()), media_type="application/json",
                                 headers={"Content-Disposition": "attachment; filename=telemetry.json"})
    columns = [
        "timestamp", "frame_id", "fps", "inference_latency_ms", "face_count", "target_id",
        "target_confidence", "target_center_x", "target_center_y", "normalized_x", "normalized_y",
        "error_x", "error_y", "velocity_x", "velocity_y", "speed", "acceleration", "direction",
        "estimated_distance", "tracking_status",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(items)
    return StreamingResponse(io.BytesIO(output.getvalue().encode()), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=telemetry.csv"})


@app.get("/api/events")
async def get_events(since: float = 0.0, limit: int = Query(default=200, ge=1, le=2000)):
    return {"items": service.events_since(since)[-limit:]}


@app.get("/api/analytics")
async def analytics():
    items = service.history(5000)
    latencies = [item["inference_latency_ms"] for item in items if isinstance(item.get("inference_latency_ms"), (int, float))]
    confidences = [item["target_confidence"] for item in items if isinstance(item.get("target_confidence"), (int, float))]
    return {
        "samples": len(items),
        "average_fps": sum(item.get("fps", 0) for item in items) / len(items) if items else None,
        "average_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "p95_latency_ms": sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None,
        "average_confidence": sum(confidences) / len(confidences) if confidences else None,
        "events_total": len(service.events),
        "dropped_frames": service.dropped_frames,
    }


@app.post("/api/recording/start")
async def recording_start():
    if not service.running.is_set():
        raise HTTPException(status_code=409, detail="Start the camera before recording.")
    return service.start_recording()


@app.post("/api/recording/stop")
async def recording_stop():
    return service.stop_recording()


@app.get("/api/sessions")
async def sessions():
    root = settings.root / "sessions"
    result = []
    if root.exists():
        for directory in sorted(root.glob("session_*"), reverse=True):
            summary_path = directory / "session_summary.json"
            metadata_path = directory / "metadata.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            result.append({"session_id": directory.name, **metadata, "summary": summary})
    return {"items": result}


@app.get("/api/sessions/{session_id}")
async def session_detail(session_id: str):
    directory = _session_dir(session_id)
    result = {"session_id": session_id}
    for name in ("metadata.json", "session_summary.json"):
        path = directory / name
        if path.exists():
            result[name.removesuffix(".json")] = json.loads(path.read_text())
    result["files"] = [path.name for path in directory.iterdir() if path.is_file()]
    return result


@app.get("/api/sessions/{session_id}/files/{filename}")
async def session_file(session_id: str, filename: str):
    directory = _session_dir(session_id)
    allowed = {"metadata.json", "session_summary.json", "performance_report.json", "events.json",
               "telemetry.csv", "tracking.csv", "events.csv", "trajectory.csv", "raw_video.mp4", "annotated_video.mp4"}
    if filename not in allowed:
        raise HTTPException(status_code=400, detail="File is not an allowed session export.")
    path = directory / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Session export is not available.")
    return FileResponse(path, filename=filename)


@app.post("/api/drone/connect")
async def drone_connect():
    service.drone.connect()
    return service.drone.telemetry()


@app.post("/api/drone/disconnect")
async def drone_disconnect():
    service.drone.disconnect()
    service.controller_enabled = False
    return service.drone.telemetry()


@app.post("/api/controller/enable")
async def controller_enable(request: ControllerRequest | None = None):
    enabled = request.enabled if request else True
    if enabled and service.target.target_id is None:
        raise HTTPException(status_code=409, detail="Lock a target before enabling simulated desired-command output.")
    return service.set_controller(enabled)


@app.post("/api/controller/disable")
async def controller_disable():
    return service.set_controller(False)


@app.post("/api/emergency-stop")
async def emergency_stop():
    return service.emergency_stop()


@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket):
    await websocket.accept()
    last_sequence = -1
    try:
        while True:
            with service.lock:
                sequence = service.sequence
                frame = service.latest_jpeg
            if frame is not None and sequence != last_sequence:
                await websocket.send_bytes(frame)
                last_sequence = sequence
            await asyncio.sleep(0.025)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await websocket.accept()
    last_sequence = -1
    try:
        while True:
            with service.lock:
                sequence = service.sequence
                telemetry = service.latest_telemetry
            if telemetry and sequence != last_sequence:
                await websocket.send_json(telemetry)
                last_sequence = sequence
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await websocket.accept()
    timestamp = 0.0
    try:
        while True:
            events = service.events_since(timestamp)
            for event in events:
                await websocket.send_json(event)
                timestamp = max(timestamp, event["timestamp"])
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/system")
async def ws_system(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(service.status())
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
