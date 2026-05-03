from ultralytics import YOLO
import cv2
import time

# 1. 모델 로드
model = YOLO(r"C:\LGH_kangnam_univ\2026-1\dont_touch\runs\serbot_test\weights\best.pt")

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