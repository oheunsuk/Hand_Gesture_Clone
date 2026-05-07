import cv2
import mediapipe as mp
import sys
import traceback
import time
import urllib.request
from pathlib import Path
from importlib.metadata import version as pkg_version
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

HAND_GESTURE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HAND_GESTURE_ROOT))
from camera_util import open_webcam

# 1. 경로 설정 (현재 파일 기준)
YOLO_DIR = Path(__file__).resolve().parent
BASE_PATH = YOLO_DIR / "datasets" / "train"
IMG_DIR = BASE_PATH / "images"
LBL_DIR = BASE_PATH / "labels"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

# 2. 클래스 설정
# 0: 주먹(fist), 1: 보자기(palm)
CLASS_MAP = {0: "fist", 1: "palm"}
CLASS_DISPLAY_MAP = {0: "FIST(주먹)", 1: "PALM(보자기)"}
CLASS_COLOR_MAP = {0: (0, 255, 255), 1: (255, 100, 0)}
LEGACY_PREFIX_MAP = {"muk": "fist", "bba": "palm", "plam": "palm"}
current_class_id = 0
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    arg_class_id = int(sys.argv[1])
    if arg_class_id in CLASS_MAP:
        current_class_id = arg_class_id
current_class_name = CLASS_MAP[current_class_id]


def _extract_sequential_index(file_stem: str, expected_prefix: str) -> int | None:
    prefix = f"{expected_prefix}_"
    if not file_stem.startswith(prefix):
        return None
    index_part = file_stem[len(prefix):]
    if not index_part.isdigit():
        return None
    # 과거 타임스탬프 파일명(긴 숫자)을 순번으로 간주하지 않는다.
    if len(index_part) > 6:
        return None
    return int(index_part)


def get_next_index_for_class(class_name: str) -> int:
    used_indices = set()

    for label_file in LBL_DIR.glob(f"{class_name}_*.txt"):
        index = _extract_sequential_index(label_file.stem, class_name)
        if index is not None:
            used_indices.add(index)

    next_index = 1
    while next_index in used_indices:
        next_index += 1
    return next_index


def migrate_legacy_dataset_names() -> None:
    legacy_label_files = []
    for label_file in LBL_DIR.glob("*.txt"):
        stem = label_file.stem
        if "_" not in stem:
            continue
        prefix = stem.split("_", 1)[0]
        if prefix in LEGACY_PREFIX_MAP:
            legacy_label_files.append(label_file)

    if not legacy_label_files:
        return

    legacy_label_files.sort(key=lambda p: (p.stat().st_mtime, p.name))
    print(f"[MIGRATE] 레거시 라벨 {len(legacy_label_files)}개를 fist/palm으로 변환합니다.")

    for label_file in legacy_label_files:
        old_stem = label_file.stem
        old_prefix = old_stem.split("_", 1)[0]
        target_prefix = LEGACY_PREFIX_MAP[old_prefix]
        next_index = get_next_index_for_class(target_prefix)
        new_stem = f"{target_prefix}_{next_index}"

        new_label_path = LBL_DIR / f"{new_stem}.txt"
        label_file.rename(new_label_path)

        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            old_image_path = IMG_DIR / f"{old_stem}{ext}"
            if old_image_path.exists():
                new_image_path = IMG_DIR / f"{new_stem}{ext}"
                old_image_path.rename(new_image_path)
                break

        print(f"[MIGRATE] {old_stem} -> {new_stem}")

# 3. MediaPipe 및 드로잉 도구 초기화
print(f"[INFO] Python version: {sys.version.split()[0]}")
print(f"[INFO] mediapipe version: {getattr(mp, '__version__', 'unknown')}")
try:
    print(f"[INFO] mediapipe package version(metadata): {pkg_version('mediapipe')}")
except Exception:
    print("[WARN] mediapipe metadata version 조회 실패")

mp_root = Path(mp.__file__).resolve().parent
hand_landmark_dir = mp_root / "modules" / "hand_landmark"
print(f"[INFO] mediapipe root: {mp_root}")
print(f"[INFO] hand_landmark dir exists: {hand_landmark_dir.exists()} ({hand_landmark_dir})")
print(
    "[INFO] hand graph exists: "
    f"{(hand_landmark_dir / 'hand_landmark_tracking_cpu.binarypb').exists()}"
)

def ensure_hand_landmarker_model() -> Path:
    model_dir = Path("C:/mp_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "hand_landmarker.task"
    if not model_path.exists():
        model_url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        print("[INFO] HandLandmarker 모델 다운로드 중...")
        urllib.request.urlretrieve(model_url, str(model_path))
    return model_path


hand_detector = None
fallback_mode = False
try:
    model_path = ensure_hand_landmarker_model()
    hand_detector_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        min_hand_presence_confidence=0.8,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_detector_options)
    print("[OK] MediaPipe HandLandmarker initialized")
except Exception as e:
    fallback_mode = True
    print(f"[ERROR] MediaPipe HandLandmarker 초기화 실패: {e}")
    print(f"[ERROR] 예외 타입: {type(e).__name__}")
    print(f"[ERROR] mediapipe root path: {mp_root}")
    print(f"[ERROR] hand_landmark dir path: {hand_landmark_dir}")
    print(
        "[ERROR] hand graph path: "
        f"{hand_landmark_dir / 'hand_landmark_tracking_cpu.binarypb'}"
    )
    print("[ACTION] 아래 명령으로 mediapipe 재설치를 시도하세요:")
    print("[ACTION] python -m pip uninstall -y mediapipe")
    print("[ACTION] python -m pip install --no-cache-dir --force-reinstall mediapipe==0.10.9")
    print(traceback.format_exc())
    print("[경고] 폴백 모드로 동작합니다. 화면 중앙 박스 기준으로 라벨 저장합니다.")

cap = open_webcam()
migrate_legacy_dataset_names()

print(f"=== [{current_class_name.upper()}] 수집 및 테스트 모드 ===")
print("- 's' 키: 이미지 + YOLO 라벨 자동 저장")
print("- '1' 키: 주먹(fist)로 전환")
print("- '2' 키: 보자기(palm)로 전환")
print("- 'q' 키: 프로그램 종료")

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # 거울 모드
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.monotonic() * 1000)
    results = hand_detector.detect_for_video(mp_image, timestamp_ms) if hand_detector is not None else None
    
    yolo_label = ""
    display_frame = frame.copy()

    if results and results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # [데이터 1] YOLO용 자동 박스 계산
            x_coords = [lm.x for lm in hand_landmarks]
            y_coords = [lm.y for lm in hand_landmarks]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # 중심은 유지하고, 폭/높이 기준 padding을 양쪽으로 확장한다.
            bw_raw = x_max - x_min
            bh_raw = y_max - y_min
            padding_ratio = 0.10
            x_min -= bw_raw * padding_ratio
            x_max += bw_raw * padding_ratio
            y_min -= bh_raw * padding_ratio
            y_max += bh_raw * padding_ratio

            # YOLO 정규화 좌표 범위(0~1)로 제한한다.
            x_min = max(0.0, min(1.0, x_min))
            x_max = max(0.0, min(1.0, x_max))
            y_min = max(0.0, min(1.0, y_min))
            y_max = max(0.0, min(1.0, y_max))

            # 중심좌표 및 너비/높이 (YOLO 포맷)
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            bw, bh = (x_max - x_min), (y_max - y_min)
            yolo_label = f"{current_class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            
            # [데이터 2] 팀원이 쓸 손목(Wrist) 좌표 시각화
            wrist = hand_landmarks[0]
            cv2.circle(display_frame, (int(wrist.x*w), int(wrist.y*h)), 10, (255, 0, 0), -1)
            
            # 시각화 가이드 (녹색 박스)
            cv2.rectangle(display_frame, (int(x_min*w), int(y_min*h)), 
                         (int(x_max*w), int(y_max*h)), (0, 255, 0), 2)
            cv2.putText(display_frame, f"WRIST READY", (int(wrist.x*w)-20, int(wrist.y*h)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # MediaPipe 초기화가 실패한 경우, 중앙 고정 박스로 수집을 계속 가능하게 한다.
    if fallback_mode:
        box_w = 0.45
        box_h = 0.55
        cx, cy = 0.5, 0.5
        x_min = int((cx - box_w / 2) * w)
        y_min = int((cy - box_h / 2) * h)
        x_max = int((cx + box_w / 2) * w)
        y_max = int((cy + box_h / 2) * h)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w - 1, x_max)
        y_max = min(h - 1, y_max)
        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 165, 255), 2)
        cv2.putText(
            display_frame,
            "MEDIAPIPE FAILED - CENTER BOX MODE",
            (x_min, max(20, y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )
        yolo_label = f"{current_class_id} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}"

    cv2.putText(
        display_frame,
        f"CLASS: {CLASS_DISPLAY_MAP[current_class_id]} (1:fist 2:palm)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        CLASS_COLOR_MAP[current_class_id],
        2
    )

    cv2.imshow('Final Collector & Wrist Tester', display_frame)
    
    key = cv2.waitKey(1)
    # 저장 로직
    if key == ord('1'):
        current_class_id = 0
        current_class_name = CLASS_MAP[current_class_id]
        print(f"[CLASS] 현재 클래스: {CLASS_DISPLAY_MAP[current_class_id]}")
    elif key == ord('2'):
        current_class_id = 1
        current_class_name = CLASS_MAP[current_class_id]
        print(f"[CLASS] 현재 클래스: {CLASS_DISPLAY_MAP[current_class_id]}")
    elif key == ord('s') and yolo_label != "":
        next_index = get_next_index_for_class(current_class_name)
        file_name = f"{current_class_name}_{next_index}"
        
        # 이미지 저장
        image_saved = cv2.imwrite(str(IMG_DIR / f"{file_name}.jpg"), frame)
        # 라벨 저장
        with open(LBL_DIR / f"{file_name}.txt", 'w', encoding="utf-8") as f:
            f.write(yolo_label)
            
        count += 1
        image_status = "성공" if image_saved else "실패"
        print(
            f"[{count}] {file_name} 저장 완료 "
            f"(이미지:{image_status}, 클래스:{CLASS_DISPLAY_MAP[current_class_id]})"
        )
        
    elif key == ord('q'):
        break

cap.release()
if hand_detector is not None:
    hand_detector.close()
cv2.destroyAllWindows()