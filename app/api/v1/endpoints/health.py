from fastapi import APIRouter, Depends

from app.api.deps import get_gemini_service
from app.core.config import Settings, get_settings
from app.services.gemini_vision import GeminiVisionService

router = APIRouter()


@router.get("/health")
def health_check(
    settings: Settings = Depends(get_settings),
    gemini: GeminiVisionService = Depends(get_gemini_service),
) -> dict:
    return {
        "status": "ok",
        "gemini": {
            "enabled": settings.gemini_enabled,
            "configured": gemini.enabled,
            "model": settings.gemini_model,
        },
    }
