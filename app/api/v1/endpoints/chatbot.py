import logging
import re
from typing import List, Literal, Optional

import google.generativeai as genai
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings
from app.services.local_chatbot import local_reply

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])


BIUM_SYSTEM_PROMPT = """당신은 분리수거·재활용 도우미 챗봇 \"비움이\"입니다.

역할:
- 대한민국 분리배출 기준에 맞춰 핵심만 짧게 안내합니다.
- 플라스틱, 유리, 종이, 금속, 비닐, 일반쓰레기, 음식물 등 품목별 배출 방법.
- 헷갈리는 품목(예: 영수증, 치킨 박스, 깨진 유리)도 간결히 알려줍니다.
- 분리배출 전 세척 방법, 오염 제거 방법, 음식물·기름·국물 제거 방법도 안내합니다.

응답 형식 (반드시 지키세요):
- 한국어 존댓말, 이모지 금지.
- 전체 길이는 한국어 기준 200자 이내.
- 2~4문장 또는 핵심 단계 최대 3개. 군더더기·인사말·재질 설명 등 부가 설명 금지.
- **마크다운 문법 금지**: `**굵게**`, `*기울임*`, `_밑줄_`, 백틱, # 제목 등 어떤 마크다운 기호도 쓰지 말고 일반 텍스트만 사용합니다.
- 단계가 필요하면 한 단계를 한 줄에 적습니다. 줄마다 \"1. <행동>: <간단 설명>\" 형식이며, 단계 사이에는 반드시 줄바꿈(\\n)을 넣습니다.
- 도입 문장(한 문장)을 먼저 쓰고 줄바꿈 후 단계를 나열합니다. 한 줄에 여러 단계를 이어붙이지 않습니다.

주제 제한 (매우 중요):
- 분리수거·재활용·쓰레기 배출과 무관한 질문(날씨, 건강, 코딩, 일상잡담, 인물 등)에는 답하지 말고 정확히 다음 한 줄만 출력하세요:
[OFF_TOPIC]
- 위 토큰 외 어떤 텍스트도 같이 출력하지 마세요.

추측이 어려운 품목은 \"정확한 안내가 어려워요. 거주 지역 분리수거 안내를 확인해 주세요.\"라고 한 줄로 답합니다."""


_FALLBACK_MODELS = (
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
)


MAX_REPLY_CHARS = 500

OFF_TOPIC_REPLY = (
    "저는 분리수거를 도와드리는 비움이예요. 그 질문은 답변할 수 없어요.\n"
    "분리배출·재활용에 관한 질문을 해주세요.\n"
    "예시)\n"
    "- 페트병은 어떻게 버려야 하나요?\n"
    "- 영수증도 종이류로 분리수거 되나요?\n"
    "- 치킨 박스 분리배출 방법 알려주세요."
)


_TRASH_KEYWORDS: tuple = (
    "분리", "재활용", "쓰레기", "배출", "버려", "버리", "처리", "수거", "종량제",
    "페트", "플라스틱", "비닐", "봉투", "봉지", "랩",
    "종이", "박스", "신문", "노트", "우유팩", "종이팩", "감열", "코팅", "영수증",
    "캔", "알루미늄", "통조림", "부탄",
    "유리", "병", "소주", "맥주", "와인", "깨진",
    "음식", "잔반", "음식물",
    "스티로폼", "형광등", "전구", "건전지", "배터리", "마스크",
    "옷", "의류", "신발", "컵", "빨대", "라벨", "헹굼", "압축",
    "비움","세척","씻","헹구","닦","오염","기름","국물","제거","청소",
    "묻은","더러움",
)

_GREETING_KEYWORDS: tuple = (
    "안녕", "반가", "처음", "고마", "감사", "도와", "도움",
    "hi", "hello",
)


def _last_user_text(history: List[dict]) -> str:
    for item in reversed(history):
        if item.get("role") == "user":
            return (item.get("text") or "").strip()
    return ""


def _is_trash_related(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if any(k in lowered for k in _GREETING_KEYWORDS):
        return True
    return any(k in lowered for k in _TRASH_KEYWORDS)


def _truncate(text: str, limit: int = MAX_REPLY_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    for sep in ("\n", ". ", ".", "!", "?", " "):
        idx = cut.rfind(sep)
        if idx >= int(limit * 0.6):
            return cut[: idx + 1].rstrip() + "…"
    return cut + "…"


_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_BOLD2_RE = re.compile(r"__(.+?)__", re.DOTALL)
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)([^*\n]+?)\*(?!\*)")
_MD_ITALIC2_RE = re.compile(r"(?<!_)_(?!_)([^_\n]+?)_(?!_)")
_MD_INLINE_CODE_RE = re.compile(r"`+([^`\n]+?)`+")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_NUMBER_STEP_RE = re.compile(r"(?<!\n)\s*(?=\d+\.\s)")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_SPACE_AROUND_NL_RE = re.compile(r"[ \t]*\n[ \t]*")


def _strip_markdown(text: str) -> str:
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_BOLD2_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _MD_ITALIC2_RE.sub(r"\1", text)
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_BULLET_RE.sub("- ", text)
    return text


def _format_reply(text: str) -> str:
    if not text:
        return ""
    cleaned = _strip_markdown(text)
    cleaned = _NUMBER_STEP_RE.sub("\n", cleaned)
    cleaned = _SPACE_AROUND_NL_RE.sub("\n", cleaned)
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    lines = [ln.rstrip() for ln in cleaned.split("\n")]
    lines = [ln for ln in lines if ln != ""] or [""]
    return "\n".join(lines).strip()


class ChatMessage(BaseModel):
    role: Optional[Literal["user", "model"]] = "user"
    text: str


class ChatRequest(BaseModel):
    history: List[ChatMessage]


class ChatResponse(BaseModel):
    reply: str
    source: Literal["gemini", "local"] = "gemini"
    notice: Optional[str] = None


def _normalize_history(items: List[ChatMessage]) -> List[dict]:
    out = []
    for m in items:
        role = m.role if m.role in ("user", "model") else "user"
        out.append({"role": role, "text": m.text})
    return out


def _build_contents(history: List[dict]) -> List[dict]:
    return [{"role": m["role"], "parts": [{"text": m["text"]}]} for m in history]


def _try_gemini(history: List[dict], primary_model: str) -> str:
    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)

    candidates = [primary_model] + [m for m in _FALLBACK_MODELS if m != primary_model]
    contents = _build_contents(history)

    last_exc: Optional[Exception] = None
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(
                model_name,
                system_instruction=BIUM_SYSTEM_PROMPT,
            )
            response = model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    top_p=0.8,
                    max_output_tokens=1024,
                ),
            )

            text = ""
            for part in response.candidates[0].content.parts:
                text += part.text

            text = text.strip()
            if text:
                return text
            raise RuntimeError("empty response")
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "404" in msg or "not found" in msg:
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("사용 가능한 Gemini 모델이 없습니다.")


def _notice_for_error(msg: str) -> str:
    lowered = msg.lower()
    if "403" in msg or "denied" in lowered or "permission" in lowered:
        return "Gemini 접근 권한이 차단되어 로컬 분리수거 안내로 응답 중이에요."
    if "429" in msg or "quota" in lowered:
        return "Gemini 사용량 한도에 도달해 로컬 분리수거 안내로 응답 중이에요."
    if "api key" in lowered or "expired" in lowered or "invalid_argument" in lowered:
        return "Gemini 키 문제로 로컬 분리수거 안내로 응답 중이에요."
    return "Gemini 일시 오류로 로컬 분리수거 안내로 응답 중이에요."


@router.post("/message", response_model=ChatResponse)
async def chatbot_message(req: ChatRequest) -> ChatResponse:
    settings = get_settings()
    history = _normalize_history(req.history)
    gemini_history = history[-4:]

    if not history:
        return ChatResponse(
            reply="안녕하세요! 분리수거 도우미 비움이예요. 어떤 품목이 궁금하신가요?",
            source="local",
            notice=None,
        )

    last_user = _last_user_text(history)
    if last_user and not _is_trash_related(last_user):
        return ChatResponse(reply=OFF_TOPIC_REPLY, source="local", notice=None)

    if not settings.gemini_enabled or not settings.gemini_api_key:
        return ChatResponse(
            reply=_truncate(_format_reply(local_reply(history))),
            source="local",
            notice="Gemini가 비활성화되어 로컬 분리수거 안내로 응답 중이에요.",
        )

    try:
        reply = _try_gemini(gemini_history, settings.gemini_model)

        if "[OFF_TOPIC]" in reply or reply.strip() == "OFF_TOPIC":
            return ChatResponse(reply=OFF_TOPIC_REPLY, source="gemini", notice=None)

        return ChatResponse(
            reply=_truncate(_format_reply(reply)),
            source="gemini",
            notice=None,
        )

    except Exception as exc:
        msg = str(exc)
        logger.warning("Gemini 호출 실패, 로컬 폴백으로 전환: %s", msg)

        return ChatResponse(
            reply=_truncate(_format_reply(local_reply(history))),
            source="local",
            notice=_notice_for_error(msg),
        )
