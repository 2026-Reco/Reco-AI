from typing import Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile

from app.api.deps import AnalysisServiceDep, GeminiServiceDep, get_settings
from app.core.config import Settings
from app.schemas.material import MaterialAnalyzeResponse, MessageResponse, ReanalyzeRequest

router = APIRouter(prefix="/materials", tags=["materials"])


@router.post("/analyze", response_model=MaterialAnalyzeResponse)
async def analyze_material(
    service: AnalysisServiceDep,
    gemini: GeminiServiceDep,
    settings: Settings = Depends(get_settings),
    image: UploadFile = File(..., description="카메라/업로드 이미지 (JPEG, PNG)"),
    x_session_id: Optional[str] = Header(
        default=None,
        description="연속 촬영 시 동일 세션 ID를 사용하면 비율이 평활화됩니다.",
    ),
    use_gemini: bool = True,
) -> MaterialAnalyzeResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    try:
        run_gemini = use_gemini and settings.gemini_enabled and gemini.enabled
        return service.analyze(
            data,
            session_id=x_session_id,
            gemini=gemini if run_gemini else None,
            mime_type=image.content_type or "image/jpeg",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reanalyze")
async def reanalyze_material(
    request: ReanalyzeRequest,
    gemini: GeminiServiceDep,
):
    return gemini.reanalyze(
        request.previous_result,
        request.additional_answers,
        request.question_type,
    )


@router.delete("/sessions/{session_id}", response_model=MessageResponse)
def reset_session(
    session_id: str,
    service: AnalysisServiceDep,
) -> MessageResponse:
    if service.reset_session(session_id):
        return MessageResponse(message="세션 분석 기록이 초기화되었습니다.")
    raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
