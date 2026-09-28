# DroneVision-Follow

**Intelligent Face Detection, Tracking & UAV Visual Monitoring Platform**

Prepared by Shivam Singh, Founder of MathTech.

This package is the runnable first implementation of the supplied master prompt. It provides a local monitoring dashboard, camera-source abstraction, face detection, temporary multi-object track IDs, manual target locking, target-loss state, live telemetry, and research-session recording. The default mode is simulation/observation. Physical drone commands are not implemented.

## Start locally

Requirements: Python 3.10 or newer. Install a camera driver for your operating system before using a webcam.

### Linux / macOS

~~~sh
cd dronevision-follow
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
cp .env.example .env
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
~~~

### Windows PowerShell

~~~powershell
cd dronevision-follow
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
~~~

Open http://127.0.0.1:8000. API documentation is available at http://127.0.0.1:8000/docs.

The default camera is device index 0. Choose a different source in the dashboard:

| Source | Example URI |
| --- | --- |
| Webcam | 0 or 1 |
| Video file | /path/to/clip.mp4 |
| RTSP | rtsp://host:port/path |
| HTTP/MJPEG | http://host:port/video |
| GStreamer | a pipeline string supported by the local OpenCV build |

The source is selected at runtime, so replacing a webcam with an OpenCV-compatible stream does not require application code changes. A drone camera SDK or ROS topic needs a new adapter and is not included yet.

## First run

1. Select a camera source and press **Start camera**.
2. Click an on-screen face to lock that temporary track ID. The lock will not jump to a different track.
3. Watch normalized position, image-space motion, confidence, FPS, and processing latency update in the dashboard.
4. Press **Start recording** to save a session under the sessions directory. Stop recording to write a session summary.
5. Export live telemetry as CSV or JSON, or retrieve a finished session from the session API.

Use video recorded with consent for repeatable testing. The dashboard is intended for authorized, consent-based research and operator-supervised observation.

## Modules

~~~text
backend/app/
  camera/sources.py          camera adapter interface and webcam/file/stream adapters
  detection/face.py          detector interface and OpenCV Haar implementation
  tracking/iou_tracker.py    temporary multi-face track IDs
  tracking/target_manager.py operator target lock and loss state machine
  analytics.py               image-space motion, center error, trajectory, range estimate
  control/                   bounded PID output and safety gate
  drone/interface.py         DroneInterface plus MockDrone only
  recording/recorder.py      CSV/JSON/MP4 session outputs
  pipeline.py                producer/consumer capture and inference service
  main.py                    FastAPI REST and WebSocket endpoints
frontend/index.html           responsive live dashboard
config/config.yaml            defaults; environment variables take priority
backend/tests/test_core.py    dependency-light core unit tests
~~~

## Configuration

Copy .env.example to .env. Environment variables override config/config.yaml.

~~~env
CAMERA_SOURCE=webcam
CAMERA_URI=0
FRAME_WIDTH=1280
FRAME_HEIGHT=720
TARGET_FPS=25
DETECTION_CONFIDENCE=0.55
TRACKER_MAX_AGE=20
TARGET_LOST_SECONDS=2.0
DISTANCE_ESTIMATION=true
DRONE_MODE=simulation
~~~

DRONE_MODE is always coerced to simulation in this release. The approximate range display uses distance ≈ focal_length_px × assumed_face_width_m / detected_face_width_px; until calibrated for a specific camera and subject, it is only a rough estimate. Haar cascade detections do not provide a calibrated probability; the displayed score is a detector adapter score, not a biometric identity confidence.

## API and WebSockets

FastAPI serves the dashboard at / and interactive API docs at /docs.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | /health | Service health |
| GET | /api/system/status | Camera, tracker, target, drone and safety state |
| POST | /api/camera/start | Start a source; body may contain source and uri |
| POST | /api/camera/stop | Stop and release resources |
| POST | /api/tracking/start, /api/tracking/stop | Enable or pause analysis |
| POST | /api/target/select | Select by track_id or normalized click x, y |
| POST | /api/target/release | Release target lock |
| GET | /api/telemetry/latest, /api/telemetry/history | Latest or in-memory telemetry |
| GET | /api/telemetry/export?format=csv | Download live CSV or JSON |
| GET | /api/events, /api/analytics | Event history and live-window metrics |
| POST | /api/recording/start, /api/recording/stop | Record session files |
| GET | /api/sessions, /api/sessions/{id} | List or inspect saved sessions |
| POST | /api/drone/connect, /api/drone/disconnect | Connect/disconnect mock simulation |
| POST | /api/controller/enable, /api/controller/disable | Enable bounded desired output to mock sink |
| POST | /api/emergency-stop | Latch all simulated command output to zero |

WebSockets: /ws/video (JPEG binary frames), /ws/telemetry (JSON samples), /ws/events (JSON events), /ws/system (system status).

## Session outputs

Each recording creates sessions/session_<UTC timestamp>/ containing:

- raw_video.mp4 and annotated_video.mp4
- telemetry.csv, tracking.csv, trajectory.csv, and events.csv
- events.json, metadata.json, session_summary.json, and performance_report.json

The video is encoded with OpenCV's MP4V writer. Some operating-system OpenCV builds may not include a compatible MP4 encoder; in that case CSV/JSON data remains available and the runtime may emit an OpenCV warning.

## Tests

~~~sh
PYTHONPATH=backend python -m unittest discover -s backend/tests -v
~~~

The current tests cover ID association/expiry, persistent target lock, reacquisition state, motion and range calculations, and safety clamping/emergency stop.

## Scope and next phases

The master prompt describes a much larger research platform. This ZIP deliberately implements the first usable baseline rather than claiming every item in that roadmap is finished. In this release:

- Face detector: OpenCV Haar cascade, CPU-only. Neural detectors and CUDA/TensorRT are not included.
- Tracker: geometry-only temporary IDs. Appearance embeddings and person re-identification are not included; after a long disappearance, a new face may receive a new ID.
- Motion: image-space pixels per second. Optical flow, IMU fusion, camera-motion compensation, and metric world coordinates are not included.
- Storage: portable session folders and CSV/JSON, not PostgreSQL/TimescaleDB.
- Drone: mock simulation only. There is no physical arming, takeoff, landing, MAVLink, or motor command path.
- Security: intended for local single-operator use on 127.0.0.1; there is no account authentication or multi-user authorization.

See docs/ARCHITECTURE.md for the boundaries and staged extension points. Do not expose the local dashboard publicly without adding authentication, authorization, TLS, and an independently reviewed control interface.

## Troubleshooting

- **Camera will not open:** close other apps using the camera, try index 1, or verify the URI in OpenCV.
- **Black video in a container:** devices are not automatically forwarded into containers. Run natively or explicitly pass the host camera device.
- **No face boxes:** use a frontal, well-lit face and ensure the face is large enough in the frame; the baseline cascade is sensitive to profile views and low resolution.
- **Network stream freezes:** try a lower resolution or FPS and check stream connectivity. The processing queue is bounded and will skip stale frames.
- **MP4 files are empty:** check that your OpenCV build includes MP4V encoding support.
