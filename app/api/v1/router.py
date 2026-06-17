from fastapi import APIRouter

from app.api.v1.endpoints import health, materials, chatbot
from app.api.v1.endpoints import geocode

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(materials.router)
api_router.include_router(chatbot.router)
api_router.include_router(geocode.router, tags=["geocode"])
