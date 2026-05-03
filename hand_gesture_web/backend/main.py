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
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from pydantic import BaseModel


# 실행 방법:
# 1) backend 폴더로 이동
# 2) uvicorn main:app --reload --host 0.0.0.0 --port 8000

ModeType = Literal["AUTO", "OVERRIDE"]
CommandType = Literal["NONE", "STOP", "RESUME", "LEFT", "RIGHT"]
SourceType = Literal["gesture", "web", "gesture_client"]


class StatusState(BaseModel):
    mode: ModeType = "AUTO"
    gesture: str = "unknown"
    stable_gesture: str = "unknown"
    command: CommandType = "NONE"
    robot_status: str = "Moving"
    source: SourceType = "gesture"
    swipe_series: str = ""
    swipe_delta_series: str = ""


class StatusUpdateRequest(BaseModel):
    mode: ModeType
    gesture: str
    stable_gesture: str
    command: CommandType
    robot_status: str
    source: SourceType = "gesture"
    swipe_series: str = ""
    swipe_delta_series: str = ""


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
last_status_post: StatusState | None = None
camera_lock = Lock()
camera_capture = None
frame_lock = Lock()
latest_frame_jpg: bytes | None = None
camera_worker_running = False
camera_worker_thread: Thread | None = None
landmark_enabled = False
show_overlay = True
SMOOTHING_LENGTH = 5
WRIST_SMOOTHING_LENGTH = 5
SWIPE_WRIST_BUFFER_SIZE = 12
MOTION_THRESHOLD = 0.10
LINEARITY_THRESHOLD = 0.55
SWIPE_COOLDOWN_SEC = 0.35
ROTATION_COMMAND_HOLD_SEC = 0.7
SWIPE_SERIES_LENGTH = 30


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


def get_smoothed_gesture(prediction_history: deque[str]) -> tuple[str, int]:
    if not prediction_history:
        return "unknown", 0
    history_list = list(prediction_history)
    labels = ["palm", "fist", "unknown"]
    count_map = {label: history_list.count(label) for label in labels}
    smoothed_gesture = max(count_map, key=count_map.get)
    return smoothed_gesture, count_map[smoothed_gesture]


def classify_static_gesture(landmarks: np.ndarray) -> str:
    """현재 프레임만 보고 palm/fist를 빠르게 분류한다."""
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


def gesture_to_command(
    stable_gesture: str, swipe_command: CommandType = "NONE", allow_palm_stop: bool = True
) -> CommandType:
    if swipe_command in {"LEFT", "RIGHT"}:
        return swipe_command
    if stable_gesture == "palm" and allow_palm_stop:
        return "STOP"
    if stable_gesture == "fist":
        return "RESUME"
    return "NONE"


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
    prediction_history: deque[str] = deque(maxlen=SMOOTHING_LENGTH)
    wrist_smooth_buf: deque[float] = deque(maxlen=WRIST_SMOOTHING_LENGTH)
    wrist_x_history: deque[float] = deque(maxlen=SWIPE_WRIST_BUFFER_SIZE)
    swipe_series_chars: deque[str] = deque(maxlen=SWIPE_SERIES_LENGTH)
    delta_series_vals: deque[float] = deque(maxlen=SWIPE_SERIES_LENGTH)
    last_swipe_confirm_time = 0.0
    last_turn_command: CommandType = "NONE"
    last_turn_time = 0.0

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
                gesture = classify_static_gesture(landmark_array)

                # 랜드마크 토글이 켜진 경우에만 오버레이를 그린다.
                if landmark_enabled:
                    draw_hand_landmarks(frame, hand_landmarks)
            else:
                wrist_smooth_buf.clear()
                wrist_x_history.clear()
                gesture = "unknown"

            prediction_history.append(gesture)
            stable_gesture, _ = get_smoothed_gesture(prediction_history)
            now = time.monotonic()
            swipe_rotate: CommandType = "NONE"
            swipe_command: CommandType = "NONE"
            rotation_delta = 0.0

            if result.hand_landmarks:
                raw_wrist_x = float(result.hand_landmarks[0][0].x)
                smoothed_x = smooth_wrist_x(wrist_smooth_buf, raw_wrist_x)
                wrist_x_history.append(smoothed_x)
                raw_cmd, _, _ = detect_swipe_command(wrist_x_history, stable_gesture, gesture)
                if raw_cmd in ("LEFT", "RIGHT") and (now - last_swipe_confirm_time) >= SWIPE_COOLDOWN_SEC:
                    swipe_rotate = "LEFT" if raw_cmd == "LEFT" else "RIGHT"
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
                swipe_command = last_turn_command
            else:
                swipe_command = swipe_rotate if swipe_rotate in ("LEFT", "RIGHT") else "NONE"

            if swipe_rotate == "LEFT":
                swipe_series_chars.append("L")
                rotation_delta = -1.0
            elif swipe_rotate == "RIGHT":
                swipe_series_chars.append("R")
                rotation_delta = 1.0
            else:
                swipe_series_chars.append(".")
            delta_series_vals.append(rotation_delta)
            swipe_series_text = "".join(swipe_series_chars)
            delta_series_text = " ".join(f"{v:+.0f}" for v in list(delta_series_vals)[-10:])

            moving_now = swipe_command in {"LEFT", "RIGHT"}
            rotation_hold_active = (
                (now - last_turn_time) < ROTATION_COMMAND_HOLD_SEC and last_turn_command in ("LEFT", "RIGHT")
            )
            recently_swiped = (now - last_swipe_confirm_time) < SWIPE_COOLDOWN_SEC
            stop_blocked = stable_gesture == "palm" and (moving_now or recently_swiped or rotation_hold_active)
            allow_palm_stop = not stop_blocked
            auto_command = gesture_to_command(stable_gesture, swipe_command, allow_palm_stop)
            if rotation_hold_active:
                auto_command = last_turn_command

            if auto_command != "NONE":
                auto_mode: ModeType = "OVERRIDE"
                auto_robot_status = "Stopped" if auto_command == "STOP" else "Moving"
            else:
                auto_mode = "AUTO"
                auto_command = "NONE"
                auto_robot_status = "Moving"

            with state_lock:
                current_status = current_status.model_copy(
                    update={
                        "mode": auto_mode,
                        "gesture": gesture,
                        "stable_gesture": stable_gesture,
                        "command": auto_command,
                        "robot_status": auto_robot_status,
                        "source": "gesture",
                        "swipe_series": swipe_series_text,
                        "swipe_delta_series": delta_series_text,
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
        out = current_status.model_copy(deep=True)
        if last_status_post is not None:
            out = out.model_copy(
                update={
                    "swipe_series": last_status_post.swipe_series or "",
                    "swipe_delta_series": last_status_post.swipe_delta_series or "",
                }
            )
        return out


@app.get("/get_status", response_model=StatusState)
def get_status_from_post() -> StatusState:
    with state_lock:
        if last_status_post is None:
            return current_status.model_copy(deep=True)
        return last_status_post.model_copy(deep=True)


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
    global current_status, last_status_post
    with state_lock:
        updated = StatusState(**payload.model_dump())
        current_status = updated
        last_status_post = updated.model_copy(deep=True)
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


@app.get("/style.css")
def serve_style_css() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")


@app.get("/app.js")
def serve_app_js() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


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

