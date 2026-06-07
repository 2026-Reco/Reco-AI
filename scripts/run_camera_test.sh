#!/usr/bin/env bash
# API 서버 + 웹캠 자동 감지 테스트 실행
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="$(pwd)"
PY="${ROOT}/.venv/bin/python"
UVICORN="${ROOT}/.venv/bin/uvicorn"

if [[ ! -x "$PY" ]]; then
  echo "가상환경이 없습니다. python3 -m venv .venv && pip install -r requirements.txt"
  exit 1
fi

# GUI 표시용 opencv (headless는 imshow 불가)
if ! "$PY" -c "import cv2; assert hasattr(cv2,'imshow')" 2>/dev/null; then
  "$PY" -m pip install -q opencv-python
fi
"$PY" -m pip install -q httpx 2>/dev/null || true

export PYTHONPATH="$ROOT"

echo ">> API 서버 시작 (http://127.0.0.1:8000)"
"$UVICORN" app.main:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2
"$PY" scripts/camera_auto_detect.py --api http://127.0.0.1:8000 "$@"
