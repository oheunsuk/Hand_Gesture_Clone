from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # GPU 사용 가능 여부 확인 (MX450 체크)
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"사용 장치: {device}")

    # 1. 모델 로드 (가장 가벼운 Nano 모델)
    model = YOLO('yolov8n.pt') 

    # 2. 학습 시작
    model.train(
    data=r'C:\LGH_kangnam_univ\2026-1\dont_touch\data.yaml', 
    epochs=50, 
    imgsz=416,           # 속도와 정확도의 타협점
    device=device,
    batch=8,             # 16은 MX450에서 튕길 확률이 높으니 8로 하세요
    amp=True,            # 메모리 절약
    project='C:/LGH_kangnam_univ/2026-1/dont_touch/runs',
    name='serbot_test'
)