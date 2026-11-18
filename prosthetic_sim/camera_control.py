"""Webcam hand-pose controller for the PyBullet prosthetic arm."""
from __future__ import annotations

import math
import time


def _point_angle(a, b, c):
    """Return the angle ABC in degrees using 2-D or 3-D MediaPipe landmarks."""
    def vector(point, origin):
        return (
            float(point.x) - float(origin.x),
            float(point.y) - float(origin.y),
            float(getattr(point, "z", 0.0)) - float(getattr(origin, "z", 0.0)),
        )

    ba = vector(a, b)
    bc = vector(c, b)
    norm_ba = math.sqrt(sum(v * v for v in ba))
    norm_bc = math.sqrt(sum(v * v for v in bc))
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 180.0
    cosine = sum(x * y for x, y in zip(ba, bc)) / (norm_ba * norm_bc)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _finger_flexions_from_landmarks(hand):
    """Estimate independent 0..1 curl values for the thumb and four fingers.

    PIP/DIP angles are used for the four fingers. The thumb uses its MCP/IP
    angles. World landmarks from MediaPipe are preferred by the camera loop,
    so these values are less dependent on camera angle and hand position.
    """
    pts = hand.landmark

    def curl(joint_a, joint, joint_b, max_flex):
        flexion = 180.0 - _point_angle(pts[joint_a], pts[joint], pts[joint_b])
        return max(0.0, min(1.0, flexion / max_flex))

    values = {}
    for name, mcp, pip, dip, tip in (
        ("index", 5, 6, 7, 8),
        ("middle", 9, 10, 11, 12),
        ("ring", 13, 14, 15, 16),
        ("little", 17, 18, 19, 20),
    ):
        pip_curl = curl(mcp, pip, dip, 110.0)
        dip_curl = curl(pip, dip, tip, 80.0)
        values[name] = max(0.0, min(1.0, 0.65 * pip_curl + 0.35 * dip_curl))

    thumb_mcp_curl = curl(1, 2, 3, 75.0)
    thumb_ip_curl = curl(2, 3, 4, 85.0)
    values["thumb"] = max(
        0.0, min(1.0, 0.55 * thumb_mcp_curl + 0.45 * thumb_ip_curl)
    )
    return values


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
    smoothed_fingers = {
        "thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "little": 0.0
    }
    gap = max(1.5, float(movement_gap_seconds))
    stable_seconds = max(0.1, float(stable_seconds))
    print(
        f"Camera demo: hold a pose for {stable_seconds:.2f}s. "
        f"At least {gap:.1f}s between accepted wrist movements. "
        "Individual fingers follow the tracked hand continuously. Press Q to quit."
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
                finger_targets = None
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand, hands_api.HAND_CONNECTIONS
                    )
                    detected = _gesture_from_landmarks(hand)

                    # Use MediaPipe's metric 3-D landmarks where available; fall
                    # back to the image landmarks on versions without world output.
                    if getattr(result, "multi_hand_world_landmarks", None):
                        finger_hand = result.multi_hand_world_landmarks[0]
                    else:
                        finger_hand = hand
                    finger_targets = _finger_flexions_from_landmarks(finger_hand)

                now = time.monotonic()
                elapsed = max(1e-3, now - last_sim_time)
                alpha = 1.0 - math.exp(-elapsed / 0.10)

                # When the hand leaves the frame, fade fingers back to neutral,
                # matching the existing REST behaviour.
                if finger_targets is None:
                    finger_targets = {
                        "thumb": 0.0, "index": 0.0, "middle": 0.0,
                        "ring": 0.0, "little": 0.0
                    }

                # Smooth each finger independently and continuously. Unlike the
                # old OPEN_HAND/CLOSE_HAND mapping, this does not set every finger
                # to the same value or wait for a whole-hand gesture transition.
                for name, target in finger_targets.items():
                    smoothed_fingers[name] += alpha * (
                        float(target) - smoothed_fingers[name]
                    )

                if detected != last_candidate:
                    last_candidate, candidate_since = detected, now
                stable = now - candidate_since >= stable_seconds

                if stable and detected == "REST" and last_emitted != "REST":
                    command = mapper.reset()
                    command.gesture = "REST"
                    log.write(command)
                    last_emitted = "REST"
                    print("LABEL: REST | wrist returned to neutral")

                elif (
                    stable
                    and detected not in ("REST", "OPEN_HAND", "CLOSE_HAND")
                    and detected != last_emitted
                    and now >= next_command_at
                ):
                    # Keep the existing wrist gesture mappings unchanged.
                    mapper.reset()
                    command = mapper.apply(detected, 1.0)
                    log.write(command)
                    last_emitted = detected
                    next_command_at = now + gap
                    print(
                        f"LABEL: {detected} | accepted | next wrist movement in at least {gap:.1f}s"
                    )

                # Apply the current independently tracked finger values on every
                # frame. Wrist targets remain those supplied by the existing mapper.
                command = mapper.state
                for name, value in smoothed_fingers.items():
                    setattr(command, name, value)
                sim.apply(command)

                # Advance physics according to elapsed wall time to keep motion smooth.
                physics_steps = max(1, min(60, int(round(elapsed * 240.0))))
                sim.step(physics_steps, realtime=False)
                last_sim_time = now

                cv2.putText(
                    frame, f"Detected: {detected}", (18, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 140), 2
                )
                cv2.putText(
                    frame,
                    "Finger curl  T:{:.0f} I:{:.0f} M:{:.0f} R:{:.0f} L:{:.0f}".format(
                        *(smoothed_fingers[k] * 100 for k in
                          ("thumb", "index", "middle", "ring", "little"))
                    ),
                    (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1
                )
                remaining = max(0.0, next_command_at - now)
                cv2.putText(
                    frame,
                    f"Wrist hold {stable_seconds:.1f}s | Gap {remaining:.1f}s | Q = quit",
                    (18, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1
                )
                cv2.imshow("Prosthetic Arm - Camera Control", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
