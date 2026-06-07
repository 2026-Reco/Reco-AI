from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.config import get_settings

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    import cv2
    import numpy as np

    from app.api.deps import get_analysis_service

    settings = get_settings()
    service = get_analysis_service(settings)
    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", blank)
    if ok:
        service.analyze(encoded.tobytes())
        from app.services.clip_classifier import predict_clip
        from app.services.object_detector import ObjectDetector

        ObjectDetector().detect(blank)
        predict_clip(blank)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="분리수거 재질 분석 API — 프론트에서 이미지(카메라 프레임)를 전송하면 재질 비율을 반환합니다.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def web_index():
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
