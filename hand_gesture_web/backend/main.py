import time
import urllib.request
from collections import deque
from pathlib import Path
from threading import Lock, Thread
from typing import Literal

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from pydantic import BaseModel
from tensorflow.keras.models import load_model


# 실행 방법:
# 1) backend 폴더로 이동
# 2) uvicorn main:app --reload --host 0.0.0.0 --port 8000

ModeType = Literal["AUTO", "OVERRIDE"]
CommandType = Literal["NONE", "STOP", "RESUME", "LEFT", "RIGHT"]
SourceType = Literal["gesture", "web"]


class StatusState(BaseModel):
    mode: ModeType = "AUTO"
    gesture: str = "unknown"
    stable_gesture: str = "unknown"
    command: CommandType = "NONE"
    robot_status: str = "Moving"
    source: SourceType = "gesture"


class StatusUpdateRequest(BaseModel):
    mode: ModeType
    gesture: str
    stable_gesture: str
    command: CommandType
    robot_status: str
    source: SourceType = "gesture"


class CommandRequest(BaseModel):
    command: Literal["STOP", "RESUME", "LEFT", "RIGHT"]


app = FastAPI(title="Hand Gesture Robot Backend")
BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"

# 프론트엔드(localhost 포함)에서 API 호출 가능하도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state_lock = Lock()
current_status = StatusState()
camera_lock = Lock()
camera_capture = None
frame_lock = Lock()
latest_frame_jpg: bytes | None = None
camera_worker_running = False
camera_worker_thread: Thread | None = None
landmark_enabled = False
show_overlay = True
GESTURE_CLASSES = ["palm", "fist", "thumb"]
SEQUENCE_LENGTH = 20
SMOOTHING_LENGTH = 5
MOTION_WINDOW = 20
MOTION_THRESHOLD = 0.15
SWIPE_COOLDOWN_SEC = 0.7
gesture_model = load_model("gesture_lstm.h5")


def ensure_hand_landmarker_model() -> Path:
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


def decode_prediction(prediction: np.ndarray) -> str:
    class_index = int(np.argmax(prediction, axis=1)[0])
    return GESTURE_CLASSES[class_index]


def get_smoothed_gesture(prediction_history: deque[str]) -> tuple[str, int]:
    if not prediction_history:
        return "unknown", 0
    history_list = list(prediction_history)
    labels = GESTURE_CLASSES + ["unknown"]
    count_map = {label: history_list.count(label) for label in labels}
    smoothed_gesture = max(count_map, key=count_map.get)
    return smoothed_gesture, count_map[smoothed_gesture]


def gesture_to_command(stable_gesture: str, swipe_command: CommandType = "NONE") -> CommandType:
    if swipe_command in {"LEFT", "RIGHT"}:
        return swipe_command
    if stable_gesture == "palm":
        return "STOP"
    if stable_gesture == "fist":
        return "RESUME"
    return "NONE"


def detect_swipe_command(wrist_x_buffer: deque[float], now: float, last_swipe_time: float) -> tuple[CommandType, float]:
    if len(wrist_x_buffer) < MOTION_WINDOW:
        return "NONE", last_swipe_time

    if (now - last_swipe_time) < SWIPE_COOLDOWN_SEC:
        return "NONE", last_swipe_time

    delta_x = wrist_x_buffer[-1] - wrist_x_buffer[0]
    if delta_x >= MOTION_THRESHOLD:
        wrist_x_buffer.clear()
        return "RIGHT", now
    if delta_x <= -MOTION_THRESHOLD:
        wrist_x_buffer.clear()
        return "LEFT", now
    return "NONE", last_swipe_time


def draw_hand_landmarks(frame, hand_landmarks) -> None:
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


def get_camera_capture():
    global camera_capture
    with camera_lock:
        if camera_capture is None or not camera_capture.isOpened():
            camera_capture = cv2.VideoCapture(0)
            if camera_capture.isOpened():
                camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return camera_capture


def make_placeholder_frame(message: str, robot_status: str = "Stopped") -> bytes:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Gesture Backend Stream", (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, message, (28, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Robot: {robot_status}",
        (28, 430),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return b""
    return encoded.tobytes()


def camera_worker_loop() -> None:
    global latest_frame_jpg, current_status
    model_path = ensure_hand_landmarker_model()
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        min_hand_presence_confidence=0.7,
    )
    sequence_buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    prediction_history: deque[str] = deque(maxlen=SMOOTHING_LENGTH)
    wrist_x_buffer: deque[float] = deque(maxlen=MOTION_WINDOW)
    last_swipe_time = 0.0

    with vision.HandLandmarker.create_from_options(options) as detector:
        while camera_worker_running:
            cap = get_camera_capture()
            if cap is None or not cap.isOpened():
                with state_lock:
                    robot_status = current_status.robot_status
                jpg = make_placeholder_frame("Camera Not Available", robot_status)
                with frame_lock:
                    latest_frame_jpg = jpg
                time.sleep(0.2)
                continue

            ok, frame = cap.read()
            if not ok:
                with state_lock:
                    robot_status = current_status.robot_status
                jpg = make_placeholder_frame("Frame Read Failed", robot_status)
                with frame_lock:
                    latest_frame_jpg = jpg
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.monotonic() * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
                sequence_buffer.append(landmark_array)

                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    input_tensor = np.array(sequence_buffer, dtype=np.float32)
                    input_tensor = input_tensor.reshape(1, SEQUENCE_LENGTH, 63)
                    prediction = gesture_model.predict(input_tensor, verbose=0)
                    gesture = decode_prediction(prediction)
                else:
                    gesture = "unknown"

                # 랜드마크 토글이 켜진 경우에만 오버레이를 그린다.
                if landmark_enabled:
                    draw_hand_landmarks(frame, hand_landmarks)
            else:
                sequence_buffer.clear()
                wrist_x_buffer.clear()
                gesture = "unknown"

            prediction_history.append(gesture)
            stable_gesture, _ = get_smoothed_gesture(prediction_history)
            now = time.monotonic()
            swipe_command: CommandType = "NONE"

            # 손을 편 상태(palm)에서만 좌우 흔들기 시계열 인식을 활성화한다.
            if result.hand_landmarks and stable_gesture == "palm":
                wrist_x_buffer.append(result.hand_landmarks[0][0].x)
                swipe_command, last_swipe_time = detect_swipe_command(wrist_x_buffer, now, last_swipe_time)
            else:
                wrist_x_buffer.clear()

            auto_command = gesture_to_command(stable_gesture, swipe_command)
            auto_robot_status = "Stopped" if auto_command == "STOP" else "Moving"

            with state_lock:
                status = current_status
                mode_is_web_override = status.mode == "OVERRIDE" and status.source == "web"
                if mode_is_web_override:
                    current_status = status.model_copy(
                        update={
                            "gesture": gesture,
                            "stable_gesture": stable_gesture,
                        }
                    )
                else:
                    current_status = status.model_copy(
                        update={
                            "mode": "AUTO",
                            "gesture": gesture,
                            "stable_gesture": stable_gesture,
                            "command": auto_command,
                            "robot_status": auto_robot_status,
                            "source": "gesture",
                        }
                    )

            with state_lock:
                mode_text = current_status.mode
                command_text = current_status.command

            # 스트리밍 프레임 좌측 상단에 현재 제어 상태 오버레이를 표시한다.
            if show_overlay:
                mode_color = (0, 0, 255) if mode_text == "OVERRIDE" else (0, 255, 0)
                command_color = (255, 255, 255)
                if command_text == "STOP":
                    command_color = (0, 0, 255)
                elif command_text == "RESUME":
                    command_color = (0, 255, 0)

                cv2.putText(
                    frame,
                    f"Mode: {mode_text}",
                    (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    mode_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Command: {command_text}",
                    (10, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    command_color,
                    2,
                    cv2.LINE_AA,
                )

            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with frame_lock:
                    latest_frame_jpg = encoded.tobytes()


def mjpeg_frame_generator():
    while True:
        with frame_lock:
            jpg = latest_frame_jpg
        if not jpg:
            with state_lock:
                robot_status = current_status.robot_status
            jpg = make_placeholder_frame("Waiting for camera...", robot_status)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.03)


@app.get("/status", response_model=StatusState)
def get_status() -> StatusState:
    with state_lock:
        return current_status.model_copy(deep=True)


class LandmarkToggleRequest(BaseModel):
    enabled: bool


@app.get("/landmark")
def get_landmark_state() -> dict:
    return {"enabled": landmark_enabled}


@app.post("/landmark")
def set_landmark_state(payload: LandmarkToggleRequest) -> dict:
    global landmark_enabled
    landmark_enabled = payload.enabled
    return {"enabled": landmark_enabled}


@app.post("/status", response_model=StatusState)
def update_status(payload: StatusUpdateRequest) -> StatusState:
    # 외부 업데이트도 호환 유지하되, 통합 백엔드에서는 기본적으로 내부 인식 루프가 상태를 관리한다.
    with state_lock:
        updated = StatusState(**payload.model_dump())
        global current_status
        current_status = updated
        return current_status.model_copy(deep=True)


@app.post("/command", response_model=StatusState)
def send_command(payload: CommandRequest) -> StatusState:
    command = payload.command

    if command == "STOP":
        robot_status = "Stopped"
    elif command == "RESUME":
        robot_status = "Moving"
    elif command in {"LEFT", "RIGHT"}:
        robot_status = "Moving"
    else:
        raise HTTPException(status_code=400, detail="Unsupported command")

    with state_lock:
        global current_status
        current_status = current_status.model_copy(
            update={
                "mode": "OVERRIDE",
                "command": command,
                "robot_status": robot_status,
                "source": "web",
            }
        )
        return current_status.model_copy(deep=True)


@app.get("/stream/laptop.mjpg")
def stream_laptop_camera():
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.on_event("startup")
def startup_camera_worker() -> None:
    global camera_worker_running, camera_worker_thread
    camera_worker_running = True
    camera_worker_thread = Thread(target=camera_worker_loop, daemon=True)
    camera_worker_thread.start()


@app.get("/")
def serve_frontend_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


# /style.css, /app.js 등 정적 파일 서빙
app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend-static")


@app.on_event("shutdown")
def shutdown_camera() -> None:
    global camera_capture, camera_worker_running, camera_worker_thread
    camera_worker_running = False
    if camera_worker_thread is not None:
        camera_worker_thread.join(timeout=1.0)
    camera_worker_thread = None
    with camera_lock:
        if camera_capture is not None and camera_capture.isOpened():
            camera_capture.release()
        camera_capture = None

