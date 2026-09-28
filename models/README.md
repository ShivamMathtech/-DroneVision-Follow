# Model adapters

The current application uses OpenCV's installed Haar cascade and does not need a separately downloaded model file.

To add a neural detector, implement the FaceDetector interface in backend/app/detection/face.py. Keep model weights in this directory or provide a configurable model path; do not commit large model weights or secrets to source control. Detection output should use the FaceDetection schema.
