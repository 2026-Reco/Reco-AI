from typing import Annotated, Optional

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.analysis_service import AnalysisService
from app.services.gemini_vision import GeminiVisionService

_analysis_service: Optional[AnalysisService] = None
_gemini_service: Optional[GeminiVisionService] = None


def get_analysis_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AnalysisService:
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AnalysisService(settings)
    return _analysis_service


def get_gemini_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> GeminiVisionService:
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiVisionService(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )
    return _gemini_service


AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]
GeminiServiceDep = Annotated[GeminiVisionService, Depends(get_gemini_service)]
