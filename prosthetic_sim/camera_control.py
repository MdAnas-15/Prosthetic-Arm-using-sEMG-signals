"""Webcam hand-gesture controller for the PyBullet prosthetic arm.

Requires opencv-python and mediapipe. Camera gestures are intentionally discrete:
hold a pose briefly to register one movement, then return to REST before the next.
"""
from __future__ import annotations
import csv
import time
from pathlib import Path

def _gesture_from_landmarks(hand):
    # MediaPipe hand landmark indices: wrist, finger MCP/PIP/DIP/tip.
    pts = hand.landmark
    fingers = []
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        fingers.append(pts[tip].y < pts[pip].y)
    thumb_open = abs(pts[4].x - pts[2].x) > 0.045
    count = sum(fingers)
    if count >= 4 and thumb_open:
        return "OPEN_HAND"
    if count == 0 and not thumb_open:
        return "CLOSE_HAND"
    if fingers == [True, False, False, False]:
        return "WRIST_FLEXION"
    if fingers == [True, True, False, False]:
        return "WRIST_EXTENSION"
    if thumb_open and count == 0:
        return "WRIST_SUPINATION"
    if fingers == [True, True, True, False]:
        return "WRIST_PRONATION"
    # A deliberate lateral tilt of the palm selects radial/ulnar deviation.
    palm_dx = pts[9].x - pts[0].x
    palm_dy = pts[9].y - pts[0].y
    if abs(palm_dx) > abs(palm_dy) * 1.15:
        return "RADIAL_DEVIATION" if palm_dx > 0 else "ULNAR_DEVIATION"
    return "REST"

def run_camera_demo(camera_index=0, hold_seconds=0.8, stable_seconds=0.55):
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("Camera mode needs OpenCV and MediaPipe. Run: pip install opencv-python mediapipe") from exc

    from .mapper import GestureMapper
    from .simulator import PyBulletProstheticSimulator
    from .logger import SimulationLogger

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open webcam {camera_index}. Check Windows camera permissions or try --camera 1.")
    hands_api = mp.solutions.hands
    mapper = GestureMapper()
    last_candidate, candidate_since = "REST", time.monotonic()
    last_emitted = None
    print("Camera demo started. Show a hand pose; hold it briefly. Press Q in the camera window to stop.")
    try:
        with hands_api.Hands(max_num_hands=1, min_detection_confidence=0.65,
                             min_tracking_confidence=0.60) as hands, \
             PyBulletProstheticSimulator(gui=True) as sim, SimulationLogger() as log:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Webcam stopped returning frames.")
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                detected = "REST"
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    hands_api_module = mp.solutions.drawing_utils
                    hands_api_module.draw_landmarks(frame, hand, hands_api.HAND_CONNECTIONS)
                    detected = _gesture_from_landmarks(hand)
                now = time.monotonic()
                if detected != last_candidate:
                    last_candidate, candidate_since = detected, now
                stable = now - candidate_since >= stable_seconds
                if stable and detected != "REST" and detected != last_emitted:
                    command = mapper.apply(detected, 1.0)
                    sim.apply(command)
                    sim.step(max(1, int(240 * hold_seconds)), realtime=False)
                    log.write(command)
                    last_emitted = detected
                    print(f"LABEL: {detected} | camera gesture accepted")
                if detected == "REST" and stable:
                    last_emitted = None
                    command = mapper.apply("REST", 1.0)
                    sim.apply(command)
                    sim.step(12, realtime=False)
                cv2.putText(frame, f"Detected: {detected}", (18, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 140), 2)
                cv2.putText(frame, "Hold pose to register | Q = quit", (18, 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1)
                cv2.imshow("Prosthetic Arm - Camera Control", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
