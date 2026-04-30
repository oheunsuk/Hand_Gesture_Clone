import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import requests
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

STATUS_ENDPOINT = "http://localhost:8000/status"
SEND_INTERVAL_SEC = 0.2
SWIPE_COOLDOWN_SEC = 1.0


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


def detect_raw_gesture(hand_landmarks) -> str:
    """
    랜드마크 기반 원시 제스처를 판별한다.
    - palm: 손가락 끝(8,12,16,20) 중 4개 이상이 PIP 관절(6,10,14,18)보다 위(y 작음)
    - fist: 손가락 끝이 대부분 접힌 상태(위 조건을 거의 만족하지 않음)
    - unknown: 그 외
    """
    tip_pip_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    extended_count = sum(1 for tip, pip in tip_pip_pairs if hand_landmarks[tip].y < hand_landmarks[pip].y)

    if extended_count >= 4:
        return "palm"
    if extended_count <= 1:
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


def get_stable_gesture(gesture_history: list[str]) -> tuple[str, int]:
    """
    최근 프레임 버퍼에서 안정 제스처 후보를 계산한다.
    - 최근 10프레임 중 동일 제스처 8프레임 이상
    반환: (gesture, count)
    """
    if len(gesture_history) < 10:
        return "unknown", 0

    count_map = {
        "palm": gesture_history.count("palm"),
        "fist": gesture_history.count("fist"),
        "unknown": gesture_history.count("unknown"),
    }
    dominant_gesture = max(count_map, key=count_map.get)
    dominant_count = count_map[dominant_gesture]
    if dominant_count >= 8:
        return dominant_gesture, dominant_count
    return "unknown", dominant_count


def detect_swipe_gesture(wrist_x_history: deque) -> str:
    """
    손목 x 좌표의 시계열 변화량으로 swipe를 판단한다.
    - 최근 20프레임 기준 delta_x = 마지막 x - 처음 x
    - delta_x < -0.15: swipe_left
    - delta_x >  0.15: swipe_right
    - 그 외: unknown
    """
    if len(wrist_x_history) < 20:
        return "unknown"

    delta_x = wrist_x_history[-1] - wrist_x_history[0]
    if delta_x < -0.15:
        return "swipe_left"
    if delta_x > 0.15:
        return "swipe_right"
    return "unknown"


def gesture_to_command(stable_gesture: str, swipe_gesture: str) -> str:
    """
    제스처를 제어 명령으로 변환한다.
    우선순위: palm/fist(안정화) > swipe
    """
    if stable_gesture == "palm":
        return "STOP"
    if stable_gesture == "fist":
        return "RESUME"
    if swipe_gesture == "swipe_left":
        return "LEFT"
    if swipe_gesture == "swipe_right":
        return "RIGHT"
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
    dominant_count: int,
    gesture_history: list[str],
    hold_seconds: float,
    swipe_gesture: str,
    wrist_delta_x: float,
):
    """디버그 ON일 때만 보이는 작은 상태 텍스트."""
    y = 160
    debug_lines = [
        f"Gesture: {gesture}",
        f"Stable Gesture: {stable_gesture}",
        f"Swipe: {swipe_gesture}",
        f"Wrist delta_x(20): {wrist_delta_x:.3f}",
        f"Frame Count: {dominant_count}/10",
        f"History Size: {len(gesture_history)}",
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
    stable_candidate = "unknown"  # 프레임 일관성 조건을 통과한 후보
    stable_gesture = "unknown"  # 일관성 + 유지시간을 둘 다 통과한 최종 제스처
    swipe_gesture = "unknown"
    wrist_delta_x = 0.0

    # 최근 10프레임 제스처 결과를 저장하는 버퍼
    gesture_history = []
    # 최근 20프레임 손목 x 좌표 버퍼 (swipe 판단용)
    wrist_x_history = deque(maxlen=20)

    # 안정 후보(stable_candidate)가 연속 유지된 시간을 측정
    candidate_start_time = time.monotonic()
    last_candidate = "unknown"

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
                raw_gesture = detect_raw_gesture(hand_landmarks)

                # 손목(landmark 0) x 좌표를 최근 20프레임 버퍼에 누적한다.
                wrist_x_history.append(hand_landmarks[0].x)
            else:
                raw_gesture = "unknown"
                wrist_x_history.clear()
                swipe_gesture = "unknown"
                wrist_delta_x = 0.0

            # 프레임 버퍼(최근 10개) 업데이트 (palm/fist 안정화용)
            gesture_history.append(raw_gesture)
            if len(gesture_history) > 10:
                gesture_history.pop(0)

            # 1) 프레임 일관성 조건 판단
            stable_candidate, dominant_count = get_stable_gesture(gesture_history)

            # 2) 유지 시간 조건 판단(같은 candidate가 0.5초 이상)
            now = time.monotonic()
            if stable_candidate != last_candidate:
                last_candidate = stable_candidate
                candidate_start_time = now
            hold_seconds = now - candidate_start_time

            if stable_candidate != "unknown" and hold_seconds >= 0.5:
                stable_gesture = stable_candidate
            else:
                stable_gesture = "unknown"

            # 손목 x 이동량 기반 swipe 판단 + 1초 쿨다운
            swipe_now = detect_swipe_gesture(wrist_x_history)
            if len(wrist_x_history) == 20:
                wrist_delta_x = wrist_x_history[-1] - wrist_x_history[0]
            else:
                wrist_delta_x = 0.0

            now = time.monotonic()
            if swipe_now in ("swipe_left", "swipe_right") and (now - last_swipe_time) >= SWIPE_COOLDOWN_SEC:
                swipe_gesture = swipe_now
                last_swipe_time = now
                wrist_x_history.clear()
            else:
                swipe_gesture = "unknown"

            # gesture 표시는 palm/fist 우선, 없으면 swipe 반영
            if raw_gesture in ("palm", "fist"):
                gesture = raw_gesture
            elif swipe_gesture != "unknown":
                gesture = swipe_gesture
            else:
                gesture = "unknown"

            # stable_gesture도 palm/fist 우선, 없으면 swipe 반영 (웹 UI 반영용)
            if stable_gesture == "unknown" and swipe_gesture != "unknown":
                stable_gesture = swipe_gesture

            # 명령 우선순위: palm/fist(안정화) > swipe
            command = gesture_to_command(stable_gesture, swipe_gesture)
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
                    dominant_count,
                    gesture_history,
                    hold_seconds,
                    swipe_gesture,
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
