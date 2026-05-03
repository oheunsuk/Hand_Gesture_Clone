import cv2
import mediapipe as mp
import os
import sys
import time
import traceback
from pathlib import Path

# 1. 경로 설정 (현재 파일 기준)
YOLO_DIR = Path(__file__).resolve().parent
BASE_PATH = YOLO_DIR / "datasets" / "train"
IMG_DIR = BASE_PATH / "images"
LBL_DIR = BASE_PATH / "labels"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

# 2. 클래스 설정 (수집 시 이 부분을 0 또는 1로 바꿔서 실행하세요)
# 0: 주먹(muk), 1: 보자기(bba)
CLASS_ID = 0  
CLASS_NAME = "muk" if CLASS_ID == 0 else "bba"

# 3. MediaPipe 및 드로잉 도구 초기화
print(f"[INFO] Python version: {sys.version.split()[0]}")
print(f"[INFO] mediapipe version: {getattr(mp, '__version__', 'unknown')}")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = None
fallback_mode = False
try:
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )
except Exception as e:
    fallback_mode = True
    print(f"[ERROR] MediaPipe Hands 초기화 실패: {e}")
    print(f"[ERROR] 예외 타입: {type(e).__name__}")
    print(traceback.format_exc())
    print("[경고] 폴백 모드로 동작합니다. 화면 중앙 박스 기준으로 라벨 저장합니다.")

cap = cv2.VideoCapture(0)

print(f"=== [{CLASS_NAME.upper()}] 수집 및 테스트 모드 ===")
print("- 's' 키: 이미지 + YOLO 라벨 자동 저장")
print("- 'q' 키: 프로그램 종료")

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # 거울 모드
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame) if hands is not None else None
    
    yolo_label = ""
    display_frame = frame.copy()

    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # [데이터 1] YOLO용 자동 박스 계산
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # 중심은 유지하고, 폭/높이 기준 padding을 양쪽으로 확장한다.
            bw_raw = x_max - x_min
            bh_raw = y_max - y_min
            padding_ratio = 0.09
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
            yolo_label = f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            
            # [데이터 2] 팀원이 쓸 손목(Wrist) 좌표 시각화
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
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
        yolo_label = f"{CLASS_ID} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}"

    cv2.imshow('Final Collector & Wrist Tester', display_frame)
    
    key = cv2.waitKey(1)
    # 저장 로직
    if key == ord('s') and yolo_label != "":
        timestamp = int(time.time() * 100)
        file_name = f"{CLASS_NAME}_{timestamp}"
        
        # 이미지 저장
        cv2.imwrite(str(IMG_DIR / f"{file_name}.jpg"), frame)
        # 라벨 저장
        with open(LBL_DIR / f"{file_name}.txt", 'w', encoding="utf-8") as f:
            f.write(yolo_label)
            
        count += 1
        print(f"[{count}] {file_name} 저장 완료 (Wrist 포함)")
        
    elif key == ord('q'):
        break

cap.release()
if hands is not None:
    hands.close()
cv2.destroyAllWindows()