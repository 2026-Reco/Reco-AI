import logging

from fastapi import APIRouter
from pydantic import BaseModel
from geopy.geocoders import Nominatim

router = APIRouter()
logger = logging.getLogger(__name__)

geolocator = Nominatim(user_agent="reco-user-report", timeout=10)

class GeocodeRequest(BaseModel):
    address: str
    district: str | None = None
    placeType: str | None = None

@router.post("/geocode")
def geocode(request: GeocodeRequest):
    query = request.address.strip()
    logger.info(
        "[geocode] request address=%s district=%s placeType=%s",
        request.address,
        request.district,
        request.placeType,
    )

    if request.district and request.district not in query:
        query = f"서울특별시 {request.district} {query}"

    logger.info("[geocode] query=%s", query)

    try:
        location = geolocator.geocode(query)
    except Exception:
        logger.exception("[geocode] geolocator failed query=%s", query)
        return {
            "success": False,
            "latitude": None,
            "longitude": None,
            "roadAddress": None,
            "message": "좌표 변환 중 오류가 발생했습니다."
        }

    if not location:
        logger.warning("[geocode] location not found query=%s", query)
        return {
            "success": False,
            "latitude": None,
            "longitude": None,
            "roadAddress": None,
            "message": "주소를 찾을 수 없습니다."
        }

    logger.info(
        "[geocode] success lat=%s lng=%s roadAddress=%s",
        location.latitude,
        location.longitude,
        location.address,
    )

    return {
        "success": True,
        "latitude": location.latitude,
        "longitude": location.longitude,
        "roadAddress": location.address,
        "message": "좌표 변환 성공"
    }
