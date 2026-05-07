import os
import time
import urllib.request
from collections import deque
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import mediapipe as mp

from camera_util import open_webcam
import numpy as np
import requests
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# MediaPipe solutions 패키지가 없는 환경에서도 동작하도록 손 연결선을 상수로 정의한다.
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

STATUS_ENDPOINT = "http://127.0.0.1:8000/status"
SEND_INTERVAL_SEC = 0.3
BACKEND_TIMEOUT_SEC = 0.08
SEQUENCE_LENGTH = 20
SMOOTHING_LENGTH = 7
ROTATION_COMMAND_HOLD_SEC = 0.7
WRIST_SMOOTHING_LENGTH = 5
SWIPE_WRIST_BUFFER_SIZE = 12
MOTION_THRESHOLD = 0.10
LINEARITY_THRESHOLD = 0.55
SWIPE_COOLDOWN_SEC = 0.35
CAM_WIDTH = 480
CAM_HEIGHT = 360
SWIPE_SERIES_LENGTH = 30
PROCESS_EVERY_N_FRAMES = 3
SHOW_LANDMARKS_DEFAULT = True
SHOW_DEBUG_DEFAULT = False

_http_session = requests.Session()


def ensure_hand_landmarker_model() -> Path:
    """HandLandmarker 모델 파일이 없으면 다운로드한다."""
    model_dir = Path("C:/mp_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "hand_landmarker.task"

    if not model_path.exists():
        model_url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        urllib.request.urlretrieve(model_url, str(model_path))

    return model_path


def detect_temporal_gesture(sequence_buffer: deque[np.ndarray]) -> str:
    if len(sequence_buffer) < 8:
        return "unknown"
    sequence = np.array(sequence_buffer, dtype=np.float32)
    tip_indices = [8, 12, 16, 20]
    pip_indices = [6, 10, 14, 18]
    extended = sequence[:, tip_indices, 1] < sequence[:, pip_indices, 1]
    extended_ratio = float(np.mean(extended))
    if extended_ratio >= 0.75:
        return "palm"
    if extended_ratio <= 0.30:
        return "fist"
    return "unknown"


def classify_static_gesture(landmarks: np.ndarray) -> str:
    """
    현재 프레임만 보고 palm/fist를 빠르게 분류한다.
    시계열 평균보다 반응이 빠르며, prediction_history에서 한 번 더 안정화한다.
    """
    if landmarks.shape[0] < 21:
        return "unknown"

    tip_indices = [8, 12, 16, 20]
    pip_indices = [6, 10, 14, 18]
    extended_count = int(np.sum(landmarks[tip_indices, 1] < landmarks[pip_indices, 1] - 0.01))
    curled_count = int(np.sum(landmarks[tip_indices, 1] > landmarks[pip_indices, 1] + 0.005))

    if extended_count >= 3:
        return "palm"
    if curled_count >= 3:
        return "fist"
    return "unknown"


def draw_hand_landmarks(frame, hand_landmarks):
    """21개 랜드마크 점과 연결선을 화면에 그린다."""
    h, w = frame.shape[:2]
    points = []

    for lm in hand_landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))
        cv2.circle(frame, (px, py), 3, (0, 255, 255), -1, cv2.LINE_AA)

    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], (255, 0, 0), 2, cv2.LINE_AA)


def get_smoothed_gesture(prediction_history: deque[str]) -> tuple[str, int]:
    if not prediction_history:
        return "unknown", 0
    history_list = list(prediction_history)
    labels = ["palm", "fist", "unknown"]
    count_map = {label: history_list.count(label) for label in labels}
    smoothed_gesture = max(count_map, key=count_map.get)
    return smoothed_gesture, count_map[smoothed_gesture]


def smooth_wrist_x(wrist_x_history: deque[float], wrist_x: float) -> float:
    """손목 x 좌표 흔들림을 이동평균으로 줄인다."""
    if not np.isfinite(wrist_x):
        return 0.5
    wrist_x_history.append(wrist_x)
    return float(np.mean(wrist_x_history))


def detect_swipe_command(
    wrist_x_history: deque[float],
    stable_gesture: str,
    gesture: str,
) -> tuple[str, float, float]:
    if len(wrist_x_history) < 2:
        return "NONE", 0.0, 0.0
    xs = list(wrist_x_history)
    first_x, last_x = xs[0], xs[-1]
    delta_x = float(last_x - first_x)
    motion_span = float(max(xs) - min(xs))
    if motion_span < MOTION_THRESHOLD:
        return "NONE", delta_x, motion_span
    linearity = abs(delta_x) / motion_span if motion_span > 1e-9 else 0.0
    if linearity < LINEARITY_THRESHOLD:
        return "NONE", delta_x, motion_span
    if stable_gesture != "palm" and gesture != "palm":
        return "NONE", delta_x, motion_span
    if delta_x > 0:
        return "RIGHT", delta_x, motion_span
    if delta_x < 0:
        return "LEFT", delta_x, motion_span
    return "NONE", delta_x, motion_span


def gesture_to_command(stable_gesture: str, swipe_command: str = "NONE", allow_palm_stop: bool = True) -> str:
    """안정 제스처를 제어 명령으로 변환한다."""
    if swipe_command != "NONE":
        return swipe_command
    if stable_gesture == "palm" and allow_palm_stop:
        return "STOP"
    if stable_gesture == "fist":
        return "RESUME"
    return "NONE"


def draw_main_ui(frame, mode: str, command: str, robot_status: str):
    """시연용 메인 UI(Mode/Command/Robot)만 크게 표시한다."""
    mode_color = (0, 0, 255) if mode == "OVERRIDE" else (0, 255, 0)
    command_color = (255, 255, 255)
    if command == "STOP":
        command_color = (0, 0, 255)
    elif command == "RESUME":
        command_color = (0, 255, 0)
    elif command in ("LEFT", "RIGHT"):
        command_color = (0, 255, 255)

    cv2.putText(frame, f"Mode: {mode}", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, mode_color, 3, cv2.LINE_AA)
    cv2.putText(
        frame, f"Command: {command}", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.1, command_color, 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"Robot: {robot_status}", (12, 126), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA
    )


def draw_debug_ui(
    frame,
    gesture: str,
    stable_gesture: str,
    smoothing_count: int,
    prediction_history: list[str],
    hold_seconds: float,
    rotate_command: str,
    wrist_x: float,
    delta_x: float,
    motion_span: float,
    moving_now: bool,
    stop_blocked: bool,
    swipe_series_text: str,
    delta_series_text: str,
):
    """디버그 ON일 때만 보이는 작은 상태 텍스트."""
    y = 160
    debug_lines = [
        f"Gesture: {gesture}",
        f"Stable Gesture: {stable_gesture}",
        f"Rotate Command: {rotate_command}",
        f"Wrist X: {wrist_x:.3f}",
        f"delta_x: {delta_x:.3f}",
        f"motion_span: {motion_span:.3f}",
        f"Moving Now: {moving_now}",
        f"Stop Blocked: {stop_blocked}",
        f"Swipe Series: {swipe_series_text}",
        f"Delta Series: {delta_series_text}",
        f"Frame Count: {smoothing_count}/5",
        f"History Size: {len(prediction_history)}",
        f"Hold: {hold_seconds:.2f}s",
        "Keys: [L] landmarks  [D] debug  [Q] quit",
    ]
    for line in debug_lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        y += 24


def post_status_to_server(payload: dict) -> bool:
    try:
        response = _http_session.post(STATUS_ENDPOINT, json=payload, timeout=BACKEND_TIMEOUT_SEC)
        return response.status_code == 200
    except Exception:
        return False


def open_camera_capture() -> cv2.VideoCapture:
    while True:
        try:
            return open_webcam()
        except RuntimeError:
            time.sleep(0.3)


def main():
    # OpenCV 내부 최적화 활성화
    cv2.setUseOptimized(True)
    cap = open_camera_capture()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    # 지연 누적을 줄이기 위해 카메라 버퍼를 최소화한다.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    model_path = ensure_hand_landmarker_model()
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,  # 성능을 위해 한 손만 처리
        min_hand_detection_confidence=0.55,
        min_tracking_confidence=0.55,
        min_hand_presence_confidence=0.55,
    )

    gesture = "unknown"  # 현재 프레임 원시 제스처
    stable_gesture = "unknown"
    smoothing_count = 0
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prediction_history = deque(maxlen=SMOOTHING_LENGTH)
    wrist_smooth_buf = deque(maxlen=WRIST_SMOOTHING_LENGTH)
    wrist_x_history = deque(maxlen=SWIPE_WRIST_BUFFER_SIZE)
    swipe_series = deque(maxlen=SWIPE_SERIES_LENGTH)
    delta_series = deque(maxlen=SWIPE_SERIES_LENGTH)
    last_swipe_confirm_time = 0.0
    last_turn_command = "NONE"
    last_turn_time = 0.0

    # 테스트용 모드/명령 상태
    mode = "AUTO"
    command = "NONE"
    show_landmarks = SHOW_LANDMARKS_DEFAULT  # l 키로 토글 가능
    show_debug = SHOW_DEBUG_DEFAULT  # d 키로 토글 가능
    robot_status = "Moving"
    pending_payload = None
    last_sent_time = 0.0
    frame_index = 0
    last_result = None

    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)  # 셀피 뷰
            display_frame = frame

            frame_index += 1
            # 추론 부하를 줄이기 위해 N프레임마다 MediaPipe 추론을 수행한다.
            if last_result is None or frame_index % PROCESS_EVERY_N_FRAMES == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(time.monotonic() * 1000)
                result = detector.detect_for_video(mp_image, timestamp_ms)
                last_result = result
            else:
                result = last_result

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]  # 첫 번째 손만 사용
                # 'l' 토글이 켜진 경우에만 랜드마크를 그린다.
                if show_landmarks:
                    draw_hand_landmarks(display_frame, hand_landmarks)
                landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
                sequence_buffer.append(landmark_array)
                gesture = classify_static_gesture(landmark_array)
            else:
                sequence_buffer.clear()
                wrist_smooth_buf.clear()
                wrist_x_history.clear()
                gesture = "unknown"

            prediction_history.append(gesture)
            stable_gesture, smoothing_count = get_smoothed_gesture(prediction_history)
            hold_seconds = 0.0
            now = time.monotonic()
            swipe_rotate = "NONE"
            rotate_command = "NONE"
            rotation_delta = 0.0
            wrist_x = 0.5
            swipe_delta_x = 0.0
            swipe_motion_span = 0.0
            if result.hand_landmarks:
                raw_wrist_x = float(result.hand_landmarks[0][0].x)
                wrist_x = smooth_wrist_x(wrist_smooth_buf, raw_wrist_x)
                wrist_x_history.append(wrist_x)
                raw_cmd, swipe_delta_x, swipe_motion_span = detect_swipe_command(
                    wrist_x_history, stable_gesture, gesture
                )
                if raw_cmd in ("LEFT", "RIGHT") and (now - last_swipe_confirm_time) >= SWIPE_COOLDOWN_SEC:
                    swipe_rotate = raw_cmd
                    last_swipe_confirm_time = now
                    wrist_x_history.clear()
            else:
                wrist_smooth_buf.clear()
                last_turn_command = "NONE"
                last_turn_time = 0.0

            if swipe_rotate in ("LEFT", "RIGHT"):
                last_turn_command = swipe_rotate
                last_turn_time = now
            if (now - last_turn_time) < ROTATION_COMMAND_HOLD_SEC and last_turn_command in ("LEFT", "RIGHT"):
                rotate_command = last_turn_command
            else:
                rotate_command = swipe_rotate if swipe_rotate in ("LEFT", "RIGHT") else "NONE"

            # LEFT/RIGHT 시계열 생성(최근 프레임 이력)
            if swipe_rotate == "LEFT":
                swipe_series.append("L")
                rotation_delta = -1.0
            elif swipe_rotate == "RIGHT":
                swipe_series.append("R")
                rotation_delta = 1.0
            else:
                swipe_series.append(".")
            delta_series.append(rotation_delta)
            swipe_series_text = "".join(swipe_series)
            delta_series_text = " ".join(f"{v:+.0f}" for v in list(delta_series)[-10:])

            # 안정 제스처를 명령으로 매핑
            moving_now = rotate_command in {"LEFT", "RIGHT"}
            rotation_hold_active = (
                (now - last_turn_time) < ROTATION_COMMAND_HOLD_SEC and last_turn_command in ("LEFT", "RIGHT")
            )
            recently_swiped = (now - last_swipe_confirm_time) < SWIPE_COOLDOWN_SEC
            stop_blocked = stable_gesture == "palm" and (moving_now or recently_swiped or rotation_hold_active)
            allow_palm_stop = not stop_blocked
            command = gesture_to_command(stable_gesture, rotate_command, allow_palm_stop)
            if rotation_hold_active:
                command = last_turn_command
            if command == "STOP":
                mode = "OVERRIDE"
                robot_status = "Stopped"
            elif command == "RESUME":
                mode = "OVERRIDE"
                robot_status = "Moving"
            elif command in {"LEFT", "RIGHT"}:
                mode = "OVERRIDE"
                robot_status = "Moving"
            else:
                mode = "AUTO"
                robot_status = "Moving"

            draw_main_ui(display_frame, mode, command, robot_status)
            if show_debug:
                draw_debug_ui(
                    display_frame,
                    gesture,
                    stable_gesture,
                    smoothing_count,
                    list(prediction_history),
                    hold_seconds,
                    rotate_command,
                    wrist_x,
                    swipe_delta_x,
                    swipe_motion_span,
                    moving_now,
                    stop_blocked,
                    swipe_series_text,
                    delta_series_text,
                )

            # 상태가 바뀌면 백엔드 전송 대기열(pending)에 등록한다.
            current_payload = {
                "command": command,
                "gesture": gesture,
                "stable_gesture": stable_gesture,
                "mode": mode,
                "robot_status": robot_status,
                "source": "gesture_client",
                "swipe_series": swipe_series_text,
                "swipe_delta_series": delta_series_text,
            }
            pending_payload = dict(current_payload)
            if swipe_rotate in ("LEFT", "RIGHT"):
                post_status_to_server(current_payload)
                last_sent_time = now
            elif (now - last_sent_time) >= SEND_INTERVAL_SEC:
                post_status_to_server(pending_payload)
                last_sent_time = now

            cv2.imshow("Gesture Test (palm/fist)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("l"):
                show_landmarks = not show_landmarks
            elif key == ord("d"):
                show_debug = not show_debug
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
