import cv2
import mediapipe as mp
import os
import time

# 1. 경로 설정 (강남대 프로젝트 구조)
BASE_PATH = r'C:\LGH_kangnam_univ\2026-1\dont_touch\datasets\train'
IMG_DIR = os.path.join(BASE_PATH, 'images')
LBL_DIR = os.path.join(BASE_PATH, 'labels')
os.makedirs(IMG_DIR, exist_ok=True); os.makedirs(LBL_DIR, exist_ok=True)

# 2. 클래스 설정 (수집 시 이 부분을 0 또는 1로 바꿔서 실행하세요)
# 0: 주먹(muk), 1: 보자기(bba)
CLASS_ID = 0  
CLASS_NAME = "muk" if CLASS_ID == 0 else "bba"

# 3. MediaPipe 및 드로잉 도구 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

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
    results = hands.process(rgb_frame)
    
    yolo_label = ""
    display_frame = frame.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # [데이터 1] YOLO용 자동 박스 계산
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # 중심좌표 및 너비/높이 (YOLO 포맷)
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            bw, bh = (x_max - x_min) * 1.15, (y_max - y_min) * 1.15 # 여유 계수 1.15
            yolo_label = f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            
            # [데이터 2] 팀원이 쓸 손목(Wrist) 좌표 시각화
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            cv2.circle(display_frame, (int(wrist.x*w), int(wrist.y*h)), 10, (255, 0, 0), -1)
            
            # 시각화 가이드 (녹색 박스)
            cv2.rectangle(display_frame, (int(x_min*w), int(y_min*h)), 
                         (int(x_max*w), int(y_max*h)), (0, 255, 0), 2)
            cv2.putText(display_frame, f"WRIST READY", (int(wrist.x*w)-20, int(wrist.y*h)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Final Collector & Wrist Tester', display_frame)
    
    key = cv2.waitKey(1)
    # 저장 로직
    if key == ord('s') and yolo_label != "":
        timestamp = int(time.time() * 100)
        file_name = f"{CLASS_NAME}_{timestamp}"
        
        # 이미지 저장
        cv2.imwrite(os.path.join(IMG_DIR, f"{file_name}.jpg"), frame)
        # 라벨 저장
        with open(os.path.join(LBL_DIR, f"{file_name}.txt"), 'w') as f:
            f.write(yolo_label)
            
        count += 1
        print(f"[{count}] {file_name} 저장 완료 (Wrist 포함)")
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()