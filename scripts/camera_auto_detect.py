#!/usr/bin/env python3
"""웹캠 자동 캡처 + 검출 박스·재질 라벨 오버레이."""

import argparse
import base64
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 재질별 색 (BGR)
MATERIAL_BGR = {
    "플라스틱": (92, 184, 92),
    "유리": (70, 120, 50),
    "종이": (80, 160, 220),
    "금속": (180, 160, 60),
    "기타": (190, 190, 190),
}
DEFAULT_BGR = (100, 220, 100)

_FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/malgun.ttf",
]


def _load_font(size: int):
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def wait_for_server(base_url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/api/v1/health", timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"서버가 응답하지 않습니다: {base_url}")


def analyze_frame(
    client: httpx.Client,
    base_url: str,
    frame: np.ndarray,
    session_id: str,
) -> dict:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("JPEG 인코딩 실패")

    res = client.post(
        f"{base_url}/api/v1/materials/analyze",
        files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
        headers={"X-Session-Id": session_id},
        timeout=60.0,
    )
    res.raise_for_status()
    return res.json()


def motion_score(current_gray: np.ndarray, prev_gray: Optional[np.ndarray]) -> float:
    if prev_gray is None:
        return 0.0
    return float(np.mean(cv2.absdiff(current_gray, prev_gray)))


def _label_text(det: Dict) -> str:
    ko = det.get("object_name_ko") or det.get("object_name", "물체")
    mat = det.get("material", "?")
    conf = det.get("confidence", 0) * 100
    return f"{ko}  →  {mat}  {conf:.0f}%"


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict],
    status: str,
) -> np.ndarray:
    """꼭짓점·연결선·재질 라벨을 물체 위에 표시."""
    out = frame.copy()
    h, w = out.shape[:2]

    for det in detections:
        bbox = det.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        material = det.get("material", "기타")
        color = MATERIAL_BGR.get(material, DEFAULT_BGR)

        corners = det.get("corners")
        if corners and len(corners) == 4:
            pts = [(int(p[0]), int(p[1])) for p in corners]
        else:
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        # 박스 변 (선)
        for i in range(4):
            cv2.line(out, pts[i], pts[(i + 1) % 4], color, 2, cv2.LINE_AA)

        # 꼭짓점 (점)
        for p in pts:
            cv2.circle(out, p, 6, color, -1, cv2.LINE_AA)
            cv2.circle(out, p, 6, (255, 255, 255), 1, cv2.LINE_AA)

        # 박스 중심 → 라벨 위치로 안내선
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        label_y = max(28, y1 - 12)
        cv2.line(out, (cx, cy), (cx, label_y), color, 1, cv2.LINE_AA)

    # 한글 라벨 (PIL)
    if detections:
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = _load_font(18)
        font_sm = _load_font(14)

        for det in detections:
            bbox = det.get("bbox") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            material = det.get("material", "기타")
            rgb = tuple(reversed(MATERIAL_BGR.get(material, DEFAULT_BGR)))

            text = _label_text(det)
            cx = (x1 + x2) // 2
            label_y = max(4, y1 - 36)

            bbox_text = draw.textbbox((0, 0), text, font=font)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
            tx = int(cx - tw / 2)
            tx = max(4, min(tx, w - tw - 4))

            pad = 4
            draw.rectangle(
                [tx - pad, label_y - pad, tx + tw + pad, label_y + th + pad],
                fill=(30, 30, 30),
            )
            draw.rectangle(
                [tx - pad, label_y - pad, tx + tw + pad, label_y + th + pad],
                outline=rgb,
                width=2,
            )
            draw.text((tx, label_y), text, font=font, fill=rgb)

            # 박스 하단에 재질만 작게
            sub = material
            draw.text((x1, y2 + 4), sub, font=font_sm, fill=rgb)

        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # 상단 상태바
    cv2.rectangle(out, (0, 0), (w, 40), (35, 35, 35), -1)
    pil_bar = Image.fromarray(cv2.cvtColor(out[0:40], cv2.COLOR_BGR2RGB))
    draw_bar = ImageDraw.Draw(pil_bar)
    draw_bar.text((10, 10), status, font=_load_font(16), fill=(240, 240, 240))
    out[0:40] = cv2.cvtColor(np.array(pil_bar), cv2.COLOR_RGB2BGR)

    if not detections:
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text(
            (w // 2 - 140, h // 2 - 10),
            "쓰레기를 화면 중앙에 비춰 주세요",
            font=_load_font(20),
            fill=(200, 200, 200),
        )
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    return out


def decode_chart_b64(b64: str) -> Optional[np.ndarray]:
    if not b64:
        return None
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def show_chart_window(chart_img: np.ndarray) -> None:
    cv2.imshow("재질 분석 — 도넛 차트", chart_img)


def print_result(body: dict) -> None:
    print("\n--- 재질 분석 ---")
    for det in body.get("detections", []):
        print(
            f"  [{det.get('object_name_ko')}] → {det['material']} "
            f"({det['confidence']*100:.0f}%)  bbox={det.get('bbox')}"
        )
    if not body.get("detections"):
        print("  (검출된 물체 없음)")
    print(f"  전체 주요: {body['primary_material']} ({body['confidence']*100:.1f}%)")
    if body.get("locked"):
        print(f"  ✓ 인식 확정 — 캡처: {body.get('capture_path')}")
        print(f"  ✓ 도넛 차트: {body.get('chart_path')}")
    print("-----------------\n")


def run(
    base_url: str,
    camera_index: int,
    motion_threshold: float,
    min_interval: float,
    max_interval: float,
) -> int:
    wait_for_server(base_url)
    print(f"서버 연결됨: {base_url}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"카메라({camera_index})를 열 수 없습니다.", file=sys.stderr)
        return 1

    session_id = f"camera-{uuid.uuid4().hex[:8]}"
    prev_gray: Optional[np.ndarray] = None
    last_analyze = 0.0
    last_body: Optional[Dict] = None
    last_chart: Optional[np.ndarray] = None
    saved_once = False
    status = "감지 대기 중…"

    print("자동 감지 시작 (q 또는 ESC 종료)")
    print("물체에 초록 박스·점·선과 재질 라벨이 표시됩니다.")
    print(f"session_id: {session_id}")

    with httpx.Client(timeout=90.0) as client:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임 읽기 실패", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            score = motion_score(gray, prev_gray)
            prev_gray = gray
            now = time.time()
            elapsed = now - last_analyze

            should_analyze = False
            if elapsed >= max_interval:
                should_analyze = True
                trigger = "주기"
            elif elapsed >= min_interval and score >= motion_threshold:
                should_analyze = True
                trigger = "움직임"

            if should_analyze:
                status = f"분석 중… ({trigger})"
                try:
                    last_body = analyze_frame(client, base_url, frame, session_id)
                    n = len(last_body.get("detections", []))
                    if last_body.get("locked"):
                        status = f"인식 확정! {last_body['primary_material']}"
                        chart = decode_chart_b64(
                            last_body.get("chart_image_base64") or ""
                        )
                        if chart is not None:
                            last_chart = chart
                            show_chart_window(chart)
                        if not saved_once:
                            print_result(last_body)
                            saved_once = True
                    else:
                        status = f"감지됨 {n}개 — 안정되면 자동 캡처"
                        if n > 0:
                            print_result(last_body)
                except httpx.HTTPError as exc:
                    status = f"API 오류: {exc}"
                    print(status, file=sys.stderr)
                last_analyze = now

            dets = (last_body or {}).get("detections", [])
            display = draw_detections(frame, dets, status)
            if last_body and last_body.get("locked"):
                cv2.putText(
                    display,
                    "CAPTURED",
                    (display.shape[1] - 120, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (80, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("Reco 재질 감지 — 박스·라벨 (q 종료)", display)
            if last_chart is not None:
                show_chart_window(last_chart)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="웹캠 자동 캡처 + 재질 분석")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--motion", type=float, default=10.0)
    parser.add_argument("--min-interval", type=float, default=0.8)
    parser.add_argument("--max-interval", type=float, default=2.5)
    args = parser.parse_args()
    return run(
        base_url=args.api.rstrip("/"),
        camera_index=args.camera,
        motion_threshold=args.motion,
        min_interval=args.min_interval,
        max_interval=args.max_interval,
    )


if __name__ == "__main__":
    sys.exit(main())
