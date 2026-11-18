"""Webcam hand-pose controller for the PyBullet prosthetic arm."""
from __future__ import annotations

import time


def _gesture_from_landmarks(hand):
    """Classify a hand pose using wrist-relative geometry from MediaPipe landmarks."""
    pts = hand.landmark

    def distance(a, b):
        dx = float(pts[a].x) - float(pts[b].x)
        dy = float(pts[a].y) - float(pts[b].y)
        return (dx * dx + dy * dy) ** 0.5

    # Landmark indices: wrist=0; PIP joints=6,10,14,18; fingertips=8,12,16,20.
    # Wrist-relative distances are less sensitive to the hand's position in the frame.
    fingers = [
        distance(tip, 0) > distance(pip, 0) * 1.12
        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
    ]
    count = sum(fingers)

    palm_width = max(1e-4, distance(5, 17))
    thumb_open = distance(4, 5) > palm_width * 0.65

    # A mostly horizontal palm with the fingers extended is mapped to deviation.
    # This is a 2-D webcam pose mapping, not direct measurement of wrist torque.
    palm_dx = float(pts[9].x) - float(pts[0].x)
    palm_dy = float(pts[9].y) - float(pts[0].y)
    if count >= 3 and abs(palm_dx) > abs(palm_dy) * 1.25:
        return "RADIAL_DEVIATION" if palm_dx > 0 else "ULNAR_DEVIATION"

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
    if count == 3:
        return "WRIST_PRONATION"
    return "REST"


def run_camera_demo(camera_index=0, stable_seconds=0.55, movement_gap_seconds=1.5):
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "Camera mode requires OpenCV and MediaPipe. Install requirements "
            "with the project Python 3.11 environment."
        ) from exc

    from .mapper import GestureMapper
    from .simulator import PyBulletProstheticSimulator
    from .logger import SimulationLogger

    cap = cv2.VideoCapture(
        camera_index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
    )
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open webcam {camera_index}. Check camera permissions or try --camera 1."
        )

    hands_api = mp.solutions.hands
    mapper = GestureMapper()
    last_candidate = "REST"
    candidate_since = time.monotonic()
    last_emitted = None
    next_command_at = 0.0
    last_sim_time = time.monotonic()
    gap = max(1.5, float(movement_gap_seconds))
    stable_seconds = max(0.1, float(stable_seconds))
    print(
        f"Camera demo: hold a pose for {stable_seconds:.2f}s. "
        f"At least {gap:.1f}s between accepted movements. Press Q to quit."
    )

    try:
        with hands_api.Hands(
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.70,
        ) as hands, PyBulletProstheticSimulator(gui=True) as sim, SimulationLogger() as log:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Webcam stopped returning frames.")
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                detected = "REST"
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand, hands_api.HAND_CONNECTIONS
                    )
                    detected = _gesture_from_landmarks(hand)

                now = time.monotonic()
                if detected != last_candidate:
                    last_candidate, candidate_since = detected, now
                stable = now - candidate_since >= stable_seconds

                if stable and detected == "REST" and last_emitted != "REST":
                    command = mapper.reset()
                    command.gesture = "REST"
                    sim.apply(command)
                    log.write(command)
                    last_emitted = "REST"
                    print("LABEL: REST | arm returned to neutral")

                elif (
                    stable
                    and detected != "REST"
                    and detected != last_emitted
                    and now >= next_command_at
                ):
                    # Each camera pose is an absolute target, not a cumulative delta.
                    mapper.reset()
                    command = mapper.apply(detected, 1.0)
                    sim.apply(command)
                    log.write(command)
                    last_emitted = detected
                    next_command_at = now + gap
                    print(
                        f"LABEL: {detected} | accepted | next movement in at least {gap:.1f}s"
                    )

                # Advance the physics clock by elapsed wall time. One physics step
                # per webcam frame made the simulation run far below real-time.
                elapsed = max(0.0, now - last_sim_time)
                physics_steps = max(1, min(60, int(round(elapsed * 240.0))))
                sim.step(physics_steps, realtime=False)
                last_sim_time = now

                cv2.putText(
                    frame, f"Detected: {detected}", (18, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 140), 2
                )
                remaining = max(0.0, next_command_at - now)
                cv2.putText(
                    frame,
                    f"Hold {stable_seconds:.1f}s | Movement gap {remaining:.1f}s | Q = quit",
                    (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1
                )
                cv2.imshow("Prosthetic Arm - Camera Control", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
