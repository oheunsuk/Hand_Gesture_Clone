import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

STATUS_ENDPOINT = "http://localhost:8000/status"
SEND_INTERVAL_SEC = 0.2
SWIPE_COOLDOWN_SEC = 1.0
GESTURE_CLASSES = ["palm", "fist", "thumb"]
SEQUENCE_LENGTH = 20
SMOOTHING_LENGTH = 5
MOTION_WINDOW = 20
MOTION_THRESHOLD = 0.15


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
        print("HandLandmarker 모델 다운로드 중...")
        urllib.request.urlretrieve(model_url, str(model_path))
        print(f"다운로드 완료: {model_path}")

    return model_path


def decode_prediction(prediction: np.ndarray) -> str:
    class_index = int(np.argmax(prediction, axis=1)[0])
    return GESTURE_CLASSES[class_index]


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
    labels = GESTURE_CLASSES + ["unknown"]
    count_map = {label: history_list.count(label) for label in labels}
    smoothed_gesture = max(count_map, key=count_map.get)
    return smoothed_gesture, count_map[smoothed_gesture]


def detect_swipe_command(wrist_x_history: deque[float], now: float, last_swipe_time: float) -> tuple[str, float]:
    if len(wrist_x_history) < MOTION_WINDOW:
        return "NONE", last_swipe_time
    if (now - last_swipe_time) < SWIPE_COOLDOWN_SEC:
        return "NONE", last_swipe_time

    delta_x = wrist_x_history[-1] - wrist_x_history[0]
    if delta_x >= MOTION_THRESHOLD:
        wrist_x_history.clear()
        return "RIGHT", now
    if delta_x <= -MOTION_THRESHOLD:
        wrist_x_history.clear()
        return "LEFT", now
    return "NONE", last_swipe_time


def gesture_to_command(stable_gesture: str, swipe_command: str = "NONE") -> str:
    if swipe_command in ("LEFT", "RIGHT"):
        return swipe_command
    if stable_gesture == "palm":
        return "STOP"
    if stable_gesture == "fist":
        return "RESUME"
    return "NONE"


def draw_overlay_ui(frame, mode: str, command: str):
    """스트리밍/미리보기 프레임 좌측 상단에 Mode/Command 오버레이를 그린다."""
    mode_color = (0, 0, 255) if mode == "OVERRIDE" else (0, 255, 0)
    command_color = (255, 255, 255)
    if command == "STOP":
        command_color = (0, 0, 255)
    elif command == "RESUME":
        command_color = (0, 255, 0)

    cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2, cv2.LINE_AA)
    cv2.putText(
        frame, f"Command: {command}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, command_color, 2, cv2.LINE_AA
    )


def draw_debug_ui(
    frame,
    gesture: str,
    stable_gesture: str,
    smoothing_count: int,
    prediction_history: list[str],
    hold_seconds: float,
    swipe_command: str,
    wrist_delta_x: float,
):
    """디버그 ON일 때만 보이는 작은 상태 텍스트."""
    y = 160
    debug_lines = [
        f"Gesture: {gesture}",
        f"Stable Gesture: {stable_gesture}",
        f"Swipe: {swipe_command}",
        f"Wrist delta_x(20): {wrist_delta_x:.3f}",
        f"Frame Count: {smoothing_count}/5",
        f"History Size: {len(prediction_history)}",
        f"Hold: {hold_seconds:.2f}s",
    ]
    for line in debug_lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        y += 24


def send_status_to_backend(payload: dict) -> bool:
    """
    FastAPI 백엔드로 상태를 전송한다.
    서버가 꺼져 있거나 오류가 나도 예외를 삼켜 앱이 계속 동작하게 한다.
    """
    try:
        requests.post(STATUS_ENDPOINT, json=payload, timeout=0.5)
        return True
    except requests.RequestException:
        return False


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    model = load_model("gesture_lstm.h5")
    model_path = ensure_hand_landmarker_model()
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,  # 성능을 위해 한 손만 처리
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        min_hand_presence_confidence=0.7,
    )

    gesture = "unknown"  # 현재 프레임 원시 제스처
    stable_gesture = "unknown"
    wrist_delta_x = 0.0
    smoothing_count = 0

    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prediction_history = deque(maxlen=SMOOTHING_LENGTH)
    # 최근 20프레임 손목 x 좌표 버퍼 (swipe 판단용)
    wrist_x_history = deque(maxlen=MOTION_WINDOW)

    # 테스트용 모드/명령 상태
    mode = "AUTO"
    command = "NONE"
    show_landmarks = False  # 시연용 기본값: 랜드마크 OFF
    show_overlay = True  # 시연용 기본값: 오버레이 ON (필요 시 False로 비활성화 가능)
    show_debug = False  # 시연용 기본값: 디버그 OFF
    robot_status = "Moving"
    last_sent_payload = None
    pending_payload = None
    last_sent_time = 0.0
    last_swipe_time = 0.0

    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽지 못했습니다.")
                break

            frame = cv2.flip(frame, 1)  # 셀피 뷰
            display_frame = frame.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.monotonic() * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]  # 첫 번째 손만 사용
                # 'l' 토글이 켜진 경우에만 랜드마크를 그린다.
                if show_landmarks:
                    draw_hand_landmarks(display_frame, hand_landmarks)
                landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
                sequence_buffer.append(landmark_array)
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    input_tensor = np.array(sequence_buffer, dtype=np.float32)
                    input_tensor = input_tensor.reshape(1, SEQUENCE_LENGTH, 63)
                    prediction = model.predict(input_tensor, verbose=0)
                    raw_gesture = decode_prediction(prediction)
                else:
                    raw_gesture = "unknown"

            else:
                sequence_buffer.clear()
                raw_gesture = "unknown"
                wrist_x_history.clear()
                swipe_command = "NONE"
                wrist_delta_x = 0.0

            prediction_history.append(raw_gesture)
            stable_gesture, smoothing_count = get_smoothed_gesture(prediction_history)
            hold_seconds = 0.0
            now = time.monotonic()
            swipe_command = "NONE"

            # 손을 편 상태에서만 손목 x 이동 시계열로 좌/우 회전을 인식한다.
            if result.hand_landmarks and stable_gesture == "palm":
                wrist_x_history.append(result.hand_landmarks[0][0].x)
            else:
                wrist_x_history.clear()

            if len(wrist_x_history) == MOTION_WINDOW:
                wrist_delta_x = wrist_x_history[-1] - wrist_x_history[0]
            else:
                wrist_delta_x = 0.0

            swipe_command, last_swipe_time = detect_swipe_command(wrist_x_history, now, last_swipe_time)

            # gesture 표시는 분류 결과를 우선 사용하고, swipe가 발생하면 명령 라벨을 표시한다.
            if raw_gesture in ("palm", "fist"):
                gesture = raw_gesture
            elif swipe_command in ("LEFT", "RIGHT"):
                gesture = swipe_command.lower()
            else:
                gesture = "unknown"

            command = gesture_to_command(stable_gesture, swipe_command)
            if command == "STOP":
                mode = "OVERRIDE"
                robot_status = "Stopped"
            elif command == "RESUME":
                mode = "OVERRIDE"
                robot_status = "Moving"
            elif command == "LEFT":
                mode = "OVERRIDE"
            elif command == "RIGHT":
                mode = "OVERRIDE"
            else:
                mode = "AUTO"

            # 랜드마크 표시와 독립적으로 overlay를 켜고 끌 수 있다.
            if show_overlay:
                draw_overlay_ui(display_frame, mode, command)
            if show_debug:
                draw_debug_ui(
                    display_frame,
                    gesture,
                    stable_gesture,
                    smoothing_count,
                    list(prediction_history),
                    hold_seconds,
                    swipe_command,
                    wrist_delta_x,
                )

            # 상태가 바뀌면 백엔드 전송 대기열(pending)에 등록한다.
            current_payload = {
                "mode": mode,
                "gesture": gesture,
                "stable_gesture": stable_gesture,
                "command": command,
                "robot_status": robot_status,
                "source": "gesture",
            }
            if current_payload != last_sent_payload:
                pending_payload = current_payload

            # 너무 자주 보내지 않도록 0.2초 간격 제한을 적용한다.
            if pending_payload is not None and (now - last_sent_time) >= SEND_INTERVAL_SEC:
                success = send_status_to_backend(pending_payload)
                last_sent_time = now
                if success:
                    last_sent_payload = pending_payload
                    pending_payload = None

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
