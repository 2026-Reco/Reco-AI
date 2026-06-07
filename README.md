# Reco Server

분리수거 **재질 분석** API + **웹 카메라 UI** (목업 스타일 도넛 차트).

## 실행

```bash
cd Reco_Server
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

브라우저에서 **http://localhost:8000** 접속 → 카메라 시작 → 물체 비추기.

## 기능

- 웹캠 실시간 분석 (브라우저)
- 검출 박스 + 꼭짓점·선 + 재질 라벨 (캔 → **금속**)
- 인식 확정 시 화면 캡처 + **재질 분석** 도넛 차트 (`output/` 저장)
- **Gemini Vision AI** (API 키 설정 시)
  - 재활용 쓰레기 **오염도** (깨끗함 / 경미 / 심각)
  - **재활용 가능 여부** 및 근거
  - 사진 맞춤 **분리배출 방법** (단계별 안내)
- API: `POST /api/v1/materials/analyze?use_gemini=true`, Swagger `/docs`

## Gemini 설정

```bash
cp .env.example .env
# .env 에 GEMINI_API_KEY=your_key 입력
```

[Google AI Studio](https://aistudio.google.com/apikey)에서 API 키를 발급한 뒤 서버를 재시작하세요.
웹 UI 오른쪽 **AI 분석** 패널에 오염도·재활용·분리배출 안내가 표시됩니다.

## 구조

```
app/           # FastAPI + 분류/검출
static/        # 웹 UI (index.html, css, js)
output/        # 확정 시 캡처·차트 PNG
scripts/       # train_trashnet.py, smoke_test.py
models/        # TrashNet 학습 가중치 (선택)
```

## 분류 엔진

1. **CLIP** — 12종 쓰레기 텍스트 매칭 (페트병, 알루미늄 캔, 유리병, 종이 등)
2. **비전 휴리스틱** — 은색 캔·유색 페트 등 형태/색 보정
3. **TrashNet** (선택) — `scripts/train_trashnet.py` 실행 후 `models/trashnet_resnet18.pth` 생성

```bash
pip install -r requirements.txt
PYTHONPATH=. python scripts/train_trashnet.py   # 약 10~20분, 최초 1회 데이터 다운로드
```

화면에 **「알루미늄 캔」「페트병」「유리병」** 처럼 종류가 바로 표시됩니다.
