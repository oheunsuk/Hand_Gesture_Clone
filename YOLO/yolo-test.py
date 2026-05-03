from ultralytics import YOLO
import cv2
import time
from pathlib import Path

# 1. 모델 로드
yolo_dir = Path(__file__).resolve().parent
run_dir = yolo_dir / "runs"
best_candidates = list(run_dir.glob("serbot_test*/weights/best.pt"))

if not best_candidates:
    # 이전 고정 경로도 호환
    legacy_best = run_dir / "serbot_test" / "weights" / "best.pt"
    if legacy_best.exists():
        best_pt = legacy_best
    else:
        raise FileNotFoundError(f"best.pt 파일을 찾을 수 없습니다: {legacy_best}")
else:
    # 가장 최근 학습 결과(best.pt 수정시각 기준) 사용
    best_pt = max(best_candidates, key=lambda p: p.stat().st_mtime)

print(f"사용 모델: {best_pt}")
model = YOLO(str(best_pt))

cap = cv2.VideoCapture(0)

# 마지막 판단 시간을 저장할 변수
last_inference_time = 0
inference_interval = 0.5  # 0.5초 단위

# 최근 결과를 저장할 변수 (0.5초 동안 화면에 유지하기 위함)
annotated_frame = None

print("테스트 시작 (q를 누르면 종료)...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    current_time = time.time()

    # 마지막 판단으로부터 0.5초가 지났는지 확인
    if current_time - last_inference_time >= inference_interval:
        # AI 판단 수행
        results = model(frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        
        # 마지막 판단 시간 업데이트
        last_inference_time = current_time
        
        # 터미널에도 결과 출력 (판단 시점 확인용)
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                label = model.names[int(box.cls[0])]
                print(f"[{time.strftime('%H:%M:%S')}] 감지됨: {label}")

    # 화면 표시 (판단된 프레임이 있으면 그것을, 없으면 원본을 표시)
    display_frame = annotated_frame if annotated_frame is not None else frame
    cv2.imshow("YOLOv8 0.5s Interval Test", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()