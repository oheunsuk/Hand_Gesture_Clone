import re
import time
import urllib.request
import json
from pathlib import Path

import cv2
import mediapipe as mp

from camera_util import open_webcam
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

CLASSES = ["palm", "thumb", "fist"]
IMAGE_ROOT_DIR = Path("YOLO") / "data"
LANDMARK_ROOT_DIR = Path("YOLO") / "landmarks"


def get_start_index(class_dir: Path, class_name: str) -> int:
    """기존 파일 번호를 확인해 다음 저장 인덱스를 반환한다."""
    pattern = re.compile(rf"^{re.escape(class_name)}_(\d+)\.jpg$", re.IGNORECASE)
    max_index = 0
    for image_path in class_dir.glob("*.jpg"):
        match = pattern.match(image_path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def save_frame(frame, class_dir: Path, class_name: str, image_index: int) -> bool:
    """원본 프레임을 640x480으로 저장한다 (오버레이/랜드마크 미포함)."""
    resized = cv2.resize(frame, (640, 480))
    file_path = class_dir / f"{class_name}_{image_index:03d}.jpg"
    return cv2.imwrite(str(file_path), resized)


def build_landmark_payload(class_name: str, image_name: str, hand_landmarks, width: int, height: int) -> dict:
    """손 랜드마크를 normalized/pixel 좌표 형태의 JSON 데이터로 만든다."""
    landmarks = []
    for idx, lm in enumerate(hand_landmarks):
        landmarks.append(
            {
                "id": idx,
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "px": int(lm.x * width),
                "py": int(lm.y * height),
            }
        )

    return {
        "class_name": class_name,
        "image_name": image_name,
        "landmarks": landmarks,
    }


def save_landmarks_json(landmark_dir: Path, class_name: str, image_index: int, hand_landmarks) -> bool:
    """이미지 파일명과 동일한 기준으로 랜드마크 JSON을 저장한다."""
    image_name = f"{class_name}_{image_index:03d}.jpg"
    json_path = landmark_dir / f"{class_name}_{image_index:03d}.json"
    payload = build_landmark_payload(class_name, image_name, hand_landmarks, width=640, height=480)

    try:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False


def save_sample(raw_frame, state: dict, class_name: str, hand_landmarks) -> bool:
    """
    이미지와 랜드마크 JSON을 같은 번호로 함께 저장한다.
    저장 성공 시에만 카운트 증가를 위해 True를 반환한다.
    """
    image_index = state["next_index"]
    image_ok = save_frame(raw_frame, state["dir"], class_name, image_index)
    if not image_ok:
        return False

    json_ok = save_landmarks_json(state["landmark_dir"], class_name, image_index, hand_landmarks)
    return json_ok


def draw_overlay(
    display_frame,
    class_name: str,
    saved_count: int,
    auto_mode: bool,
    hand_detected: bool,
    warning_text: str,
):
    """좌측 상단 상태 정보를 화면에 그린다."""
    hand_text = "Detected" if hand_detected else "Not Detected"
    auto_text = "ON" if auto_mode else "OFF"
    lines = [
        f"Class: {class_name}",
        f"Count: {saved_count}",
        f"Auto Save: {auto_text}",
        f"Hand: {hand_text}",
        "Class: 1=palm, 2=thumb, 3=fist",
        "Controls: SPACE=save, A=auto, Q=quit",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            display_frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30

    if warning_text:
        cv2.putText(
            display_frame,
            warning_text,
            (10, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def ensure_hand_landmarker_model() -> Path:
    """HandLandmarker 모델 파일을 확인하고 없으면 다운로드한다."""
    model_dir = Path("C:/mp_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "hand_landmarker.task"

    if not model_path.exists():
        model_url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        print("MediaPipe HandLandmarker 모델 다운로드 중...")
        urllib.request.urlretrieve(model_url, str(model_path))
        print(f"다운로드 완료: {model_path}")

    return model_path


def draw_hand_landmarks(display_frame, hand_landmarks):
    """감지된 손 랜드마크(21개)와 연결선을 화면에 그린다."""
    h, w = display_frame.shape[:2]
    points = []

    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
        cv2.circle(display_frame, (x, y), 3, (0, 255, 255), -1, cv2.LINE_AA)

    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(display_frame, points[start_idx], points[end_idx], (255, 0, 0), 2, cv2.LINE_AA)


def main():
    # 클래스별 저장 상태 초기화 (아까처럼 1/2/3 전환 방식)
    class_state = {}
    for class_name in CLASSES:
        image_dir = IMAGE_ROOT_DIR / class_name
        landmark_dir = LANDMARK_ROOT_DIR / class_name
        image_dir.mkdir(parents=True, exist_ok=True)
        landmark_dir.mkdir(parents=True, exist_ok=True)
        next_index = get_start_index(image_dir, class_name)
        class_state[class_name] = {
            "dir": image_dir,
            "landmark_dir": landmark_dir,
            "next_index": next_index,
            "saved_count": next_index - 1,
        }
    current_class = CLASSES[0]

    try:
        cap = open_webcam()
    except RuntimeError as e:
        print(e)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    model_path = ensure_hand_landmarker_model()
    landmarker_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        min_hand_presence_confidence=0.7,
    )

    auto_mode = False
    auto_interval = 0.3
    last_auto_save_time = 0.0
    warning_text = ""
    warning_until = 0.0

    print("\n[조작법]")
    print(" - 1    : palm")
    print(" - 2    : thumb")
    print(" - 3    : fist")
    print(" - Space: 수동 저장(손 감지 시만)")
    print(" - a    : 자동 저장 ON/OFF")
    print(" - q    : 종료\n")

    with vision.HandLandmarker.create_from_options(landmarker_options) as hand_detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽지 못했습니다. 프로그램을 종료합니다.")
                break

            # 미러 화면(사용자 친화)
            frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()  # 저장용 원본(오버레이/랜드마크 없는 프레임)
            display_frame = frame.copy()  # 화면 표시용 프레임

            # MediaPipe Tasks HandLandmarker 입력
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.monotonic() * 1000)
            results = hand_detector.detect_for_video(mp_image, timestamp_ms)
            hand_detected = len(results.hand_landmarks) > 0
            current_hand_landmarks = results.hand_landmarks[0] if hand_detected else None

            # 손이 감지되면 랜드마크 + 연결선 표시
            if hand_detected:
                for hand_landmarks in results.hand_landmarks:
                    draw_hand_landmarks(display_frame, hand_landmarks)

            # 자동 저장: 손 감지된 프레임만 저장
            now = time.monotonic()
            state = class_state[current_class]
            if auto_mode and hand_detected and (now - last_auto_save_time) >= auto_interval:
                if save_sample(raw_frame, state, current_class, current_hand_landmarks):
                    state["saved_count"] += 1
                    state["next_index"] += 1
                last_auto_save_time = now
            elif auto_mode and not hand_detected and (now - last_auto_save_time) >= auto_interval:
                # 손이 없으면 저장하지 않되, 과도한 경고 깜빡임 방지를 위해 간격 기준만 갱신
                last_auto_save_time = now

            # 경고 문구 타이머 처리
            if warning_text and now > warning_until:
                warning_text = ""

            draw_overlay(
                display_frame,
                current_class,
                state["saved_count"],
                auto_mode,
                hand_detected,
                warning_text,
            )
            cv2.imshow("Gesture Data Collector (MediaPipe Hands)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                auto_mode = not auto_mode
                last_auto_save_time = 0.0
            elif key == ord(" "):
                if hand_detected:
                    if save_sample(raw_frame, state, current_class, current_hand_landmarks):
                        state["saved_count"] += 1
                        state["next_index"] += 1
                else:
                    warning_text = "No hand detected - frame not saved"
                    warning_until = now + 1.2
            elif key == ord("1"):
                current_class = "palm"
            elif key == ord("2"):
                current_class = "thumb"
            elif key == ord("3"):
                current_class = "fist"

    cap.release()
    cv2.destroyAllWindows()
    print("\n저장 완료(클래스별):")
    for class_name in CLASSES:
        state = class_state[class_name]
        print(f"- {class_name}: {state['saved_count']}장 (images: {state['dir']}, landmarks: {state['landmark_dir']})")


if __name__ == "__main__":
    main()

# 실행 방법
# 1) 가상환경에서 필수 패키지 설치
#    .\venv\Scripts\python.exe -m pip install opencv-python mediapipe
# 2) 스크립트 실행
#    .\venv\Scripts\python.exe YOLO\collect_gesture_data.py
# 3) 첫 실행 시 HandLandmarker 모델이 자동 다운로드된다.
