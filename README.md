# 손 제스처 기반 로봇 제어 시스템 (FastAPI + Web UI)

손 제스처 인식 결과를 실시간으로 확인하고, 필요 시 버튼으로 로봇을 수동 제어할 수 있는 웹 대시보드입니다.

## 주요 기능
- **실시간 상태 모니터링**
  - 현재 모드(`AUTO` / `OVERRIDE`)
  - 인식된 제스처(`palm`, `thumb`, `left`, `right`)
  - confidence 값(예: `0.93` -> `93%`)
  - 실행 명령(`STOP`, `RESUME`, `LEFT`, `RIGHT`)
  - 로봇 상태(`Moving`, `Stopped`)
- **fallback 수동 제어**
  - 버튼: `STOP`, `RESUME`, `LEFT`, `RIGHT`
  - 클릭 즉시 백엔드로 명령 전송
- **자동 갱신**
  - 프론트엔드가 1초마다 상태를 재조회
- **에러 처리**
  - API 호출 실패 시 화면 에러 영역에 메시지 표시
- **카메라 영역 UI**
  - `로봇 카메라`, `노트북 카메라` 2개 슬롯(placeholder) 제공

## API 사용법

### 1) 상태 조회
- `GET /status`

응답 예시:
```json
{
  "mode": "AUTO",
  "gesture": "palm",
  "confidence": 0.93,
  "command": "STOP",
  "robot_status": "moving"
}
```

### 2) 명령 전송
- `POST /command`
- `Content-Type: application/json`

요청 예시:
```json
{
  "command": "STOP"
}
```

## 실행 방법 (Windows / PowerShell)
1. 프로젝트 폴더 이동
   - `cd "c:\Users\오은석\Desktop\학술제"`

2. 가상환경 생성 및 활성화
   - `python -m venv venv`
   - `.\venv\Scripts\Activate.ps1`

3. 의존성 설치
   - `pip install -r backend/requirements.txt`

4. 서버 실행
   - `uvicorn backend.main:app --reload --port 8000`

5. 웹 UI 접속
   - 브라우저: `http://localhost:8000`

## 프로젝트 구조
- `backend/main.py` : FastAPI 서버, 상태/명령 로직, 정적 프론트 서빙
- `backend/requirements.txt` : Python 패키지 목록
- `frontend/index.html` : 대시보드 마크업
- `frontend/styles.css` : UI 스타일
- `frontend/app.js` : 1초 폴링, 버튼 명령 전송, 에러 처리

## 참고
- 현재 `gesture_recognition_loop()`은 데모용 랜덤 데이터 생성 로직입니다.
- 실제 제스처 인식 모델/로봇 제어 모듈이 있다면 `backend/main.py` 상태 업데이트 부분에 연결하면 됩니다.
