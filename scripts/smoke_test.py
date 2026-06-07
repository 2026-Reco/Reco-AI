#!/usr/bin/env python3
"""API 스모크 테스트. 프로젝트 루트에서: PYTHONPATH=. python scripts/smoke_test.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def main() -> int:
    client = TestClient(app)

    health = client.get("/api/v1/health")
    assert health.status_code == 200, health.text
    health_body = health.json()
    assert health_body["status"] == "ok"
    assert "gemini" in health_body
    print("OK  GET /api/v1/health", "gemini configured:", health_body["gemini"]["configured"])

    img = np.zeros((320, 240, 3), dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (200, 280), (0, 180, 255), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    res = client.post(
        "/api/v1/materials/analyze",
        files={"image": ("test.jpg", buf.tobytes(), "image/jpeg")},
        headers={"X-Session-Id": "smoke-test-session"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert "summary" in body and len(body["summary"]) == 4
    assert "detail" in body and len(body["detail"]) == 5
    assert body["session_id"] == "smoke-test-session"
    assert "detections" in body
    assert "ai_enabled" in body
    assert "disposal_steps" in body
    total = sum(item["percent"] for item in body["summary"])
    assert 99.0 <= total <= 101.0, total
    print("OK  POST /api/v1/materials/analyze")
    print("    primary:", body["primary_material"], "confidence:", body["confidence"])
    print("    summary:", body["summary"])

    deleted = client.delete("/api/v1/materials/sessions/smoke-test-session")
    assert deleted.status_code == 200
    print("OK  DELETE /api/v1/materials/sessions/{id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
