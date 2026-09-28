# Architecture

## Runtime data flow

~~~mermaid
flowchart LR
  A["Webcam / file / stream"] --> B["Bounded capture queue"]
  B --> C["Preprocess and resize"]
  C --> D["Face detector"]
  D --> E["IoU tracker"]
  E --> F["Target lock manager"]
  F --> G["Motion and range estimate"]
  G --> H["Safety layer"]
  H --> I["Mock drone sink"]
  G --> J["Annotated video and telemetry"]
  J --> K["WebSocket dashboard"]
  J --> L["Session recorder"]
~~~

Capture and processing run on separate threads. The two-frame capture queue discards the oldest frame under load so the display favors freshness over a growing latency backlog. The detector, tracker, camera adapter, mock drone, and recorder have separate modules so a future detector or camera can replace one adapter without rewriting the dashboard.

## Safety boundary

The controller is disabled by default. When enabled by an operator, it computes bounded desired yaw and pitch values from normalized image-center error. The safety layer suppresses output when no selected target is currently visible, when manual override is active, or when emergency stop is latched. Output terminates at the MockDrone simulation sink, which records virtual values only. This release has no physical arming, motor, MAVLink, or flight-control transport.

## Current implementation

| Area | Included now | Extension point |
| --- | --- | --- |
| Camera | USB/webcam, recorded video, OpenCV-compatible stream URI | ROS image topics, drone SDK |
| Detection | OpenCV Haar face cascade | YOLO face, RetinaFace, SCRFD, MediaPipe |
| Tracking | Greedy IoU plus center-distance association and temporary IDs | ByteTrack, DeepSORT, BoT-SORT, appearance re-identification |
| Target state | Click/track-ID selection, retained lock, lost/search/reacquired states | Re-identification embeddings and occlusion scoring |
| Analytics | Image-space velocity, direction, acceleration, center error, uncalibrated monocular range | Camera/IMU motion compensation, calibrated metric motion |
| Dashboard | WebSocket video, telemetry, event stream, plots, session exports | Authenticated multi-user web application |
| Storage | Session directories with CSV, JSON, MP4 | PostgreSQL / TimescaleDB |
| Drone | Mock simulation telemetry and safety gate | Simulator-specific adapter or separately reviewed MAVLink integration |
