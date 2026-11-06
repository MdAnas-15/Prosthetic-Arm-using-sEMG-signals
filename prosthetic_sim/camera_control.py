"""Webcam hand-gesture controller for the PyBullet prosthetic arm."""
from __future__ import annotations
import time

def _gesture_from_landmarks(hand):
    """Map a MediaPipe normalized hand to one of the project's nine gestures."""
    pts = hand.landmark
    # Compare fingertip to PIP and MCP so slight bent fingers are not misread.
    fingers = []
    for tip, pip, mcp in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
        fingers.append(pts[tip].y < pts[pip].y and pts[pip].y < pts[mcp].y + 0.02)
    # Thumb extension is measured relative to its own MCP and palm width.
    palm_width = max(0.04, abs(pts[5].x - pts[17].x))
    thumb_open = abs(pts[4].x - pts[2].x) / palm_width > 0.55
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
    palm_dx = pts[9].x - pts[0].x
    palm_dy = pts[9].y - pts[0].y
    if abs(palm_dx) > abs(palm_dy) * 1.15:
        return "RADIAL_DEVIATION" if palm_dx > 0 else "ULNAR_DEVIATION"
    return "REST"

def run_camera_demo(camera_index=0, stable_seconds=0.55):
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("Camera mode requires OpenCV and MediaPipe. Install requirements with the project Python 3.11 environment.") from exc

    from .mapper import GestureMapper
    from .simulator import PyBulletProstheticSimulator
    from .logger import SimulationLogger

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open webcam {camera_index}. Check camera permissions or try --camera 1.")
    hands_api = mp.solutions.hands
    mapper = GestureMapper()
    last_candidate, candidate_since = "REST", time.monotonic()
    last_emitted = None
    print("Camera demo: hold a gesture for about half a second. Return to no hand/REST between gestures. Press Q to quit.")
    try:
        with hands_api.Hands(max_num_hands=1, min_detection_confidence=0.65,
                             min_tracking_confidence=0.60) as hands, \
             PyBulletProstheticSimulator(gui=True) as sim, SimulationLogger() as log:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Webcam stopped returning frames.")
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                detected = "REST"
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand, hands_api.HAND_CONNECTIONS)
                    detected = _gesture_from_landmarks(hand)

                now = time.monotonic()
                if detected != last_candidate:
                    last_candidate, candidate_since = detected, now
                stable = now - candidate_since >= stable_seconds

                if stable and detected != last_emitted:
                    command = mapper.apply(detected, 1.0)
                    sim.apply(command)
                    log.write(command)
                    last_emitted = detected
                    print(f"LABEL: {detected} | camera gesture accepted")

                # Advance one physics frame per camera frame; do not freeze webcam
                # processing for the whole movement dwell period.
                sim.step(1, realtime=False)
                cv2.putText(frame, f"Detected: {detected}", (18, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 140), 2)
                cv2.putText(frame, "Hold pose to register | Return to REST between poses | Q = quit",
                            (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1)
                cv2.imshow("Prosthetic Arm - Camera Control", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
