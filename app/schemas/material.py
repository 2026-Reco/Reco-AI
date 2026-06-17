from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.gemini_analysis import (
    ContaminationInfo,
    MaterialComponent,
    RecyclableInfo,
)


class MaterialRatio(BaseModel):
    label: str = Field(..., examples=["플라스틱"])
    percent: float = Field(..., ge=0, le=100, examples=[85.2])


class ObjectDetection(BaseModel):
    """화면에 그릴 검출 결과 — 박스 꼭짓점 + 재질 라벨."""

    object_name: str = Field(..., description="COCO 객체 이름 (예: bottle)")
    object_name_ko: str = Field(..., description="한글 객체 이름 (예: 페트병)")
    waste_type_ko: str = Field(default="", description="구체적 쓰레기 종류 (예: 알루미늄 캔)")
    material: str = Field(..., description="인식 재질 (예: 플라스틱)")
    confidence: float = Field(..., ge=0, le=1)
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2] 픽셀 좌표")
    corners: List[List[int]] = Field(
        ..., description="꼭짓점 4개 [[x,y], ...] 시계방향"
    )


class MaterialAnalyzeResponse(BaseModel):
    """프론트 도넛 차트용 요약(3종) + 상세(5종) 재질 비율."""

    summary: List[MaterialRatio] = Field(
        description="플라스틱·유리·기타 (차트 표시용)"
    )
    detail: List[MaterialRatio] = Field(
        description="플라스틱·유리·종이·금속·기타"
    )
    item_name: str = Field(default="알 수 없는 품목", description="이미지 속 실제 품목명")
    primary_material: str = Field(..., examples=["플라스틱"])
    material: str = Field(..., examples=["플라스틱"], description="대표 주 재질")
    waste_type_ko: str = Field(default="미확인", description="주요 쓰레기 종류 (예: 알루미늄 캔)")
    materialProbabilities: List[MaterialRatio] = Field(
        default_factory=list,
        description="대표 재질과 일치하도록 보정된 재질 확률",
    )
    confidence: float = Field(..., ge=0, le=1, description="주요 재질 신뢰도(0~1)")
    session_id: Optional[str] = Field(
        default=None, description="연속 프레임 평활화에 사용한 세션 ID"
    )
    detections: List[ObjectDetection] = Field(
        default_factory=list,
        description="검출된 쓰레기 영역 (박스·재질 라벨)",
    )
    locked: bool = Field(
        default=False,
        description="안정 인식 확정 여부 (연속 동일 재질)",
    )
    capture_path: Optional[str] = Field(
        default=None, description="확정 시 저장된 캡처 이미지 경로"
    )
    chart_path: Optional[str] = Field(
        default=None, description="확정 시 저장된 도넛 차트 PNG 경로"
    )
    chart_image_base64: Optional[str] = Field(
        default=None, description="도넛 차트 PNG (base64)"
    )
    # Gemini Vision AI
    ai_enabled: bool = Field(default=False, description="AI 분석 결과 표시 여부")
    ai_source: Optional[str] = Field(
        default=None, description="gemini | local"
    )
    ai_summary: Optional[str] = Field(default=None, description="AI 한 줄 요약")
    materials: List[MaterialComponent] = Field(
        default_factory=list,
        max_length=3,
        description="Gemini가 추정한 복합 재질 구성",
    )
    contamination: Optional[ContaminationInfo] = Field(
        default=None, description="오염도 분석"
    )
    recyclable: Optional[RecyclableInfo] = Field(
        default=None, description="재활용 가능 여부"
    )
    disposal_steps: List[str] = Field(
        default_factory=list, description="맞춤형 분리배출 방법"
    )
    warnings: List[str] = Field(default_factory=list, description="주의사항")
    ai_error: Optional[str] = Field(
        default=None, description="Gemini 호출 실패 시 오류 메시지"
    )


class MessageResponse(BaseModel):
    message: str


class ReanalyzeRequest(BaseModel):
    previous_result: dict
    additional_answers: list = Field(default_factory=list)
    question_type: str = "general_reanalysis"
