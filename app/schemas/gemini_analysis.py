from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ContaminationInfo(BaseModel):
    """재활용 쓰레기 오염도."""

    level: str = Field(
        ...,
        description="clean(깨끗) | low(경미 오염) | high(심각 오염)",
    )
    score: float = Field(
        ...,
        ge=0,
        le=100,
        description="오염 점수 (0=깨끗, 100=심하게 오염)",
    )
    detail: str = Field(..., description="오염 상태 설명")


class RecyclableInfo(BaseModel):
    """재활용 가능 여부."""

    possible: bool = Field(..., description="재활용 가능 여부")
    label: str = Field(
        ...,
        description="재활용 가능 | 재활용 불가 | 조건부 가능",
    )
    reason: str = Field(..., description="판단 근거")


class GeminiAnalysisResult(BaseModel):
    waste_type_ko: str
    material: str
    contamination: ContaminationInfo
    recyclable: RecyclableInfo
    disposal_steps: List[str] = Field(
        default_factory=list, description="사진 맞춤 분리배출 방법"
    )
    warnings: List[str] = Field(default_factory=list)
    summary: str = Field(default="", description="한 줄 요약")
