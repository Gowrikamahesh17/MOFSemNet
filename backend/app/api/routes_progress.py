from fastapi import APIRouter

from app.models.schemas import ProgressData
from app.services.content_loader import load_progress

router = APIRouter(prefix="/api/progress", tags=["progress"])


@router.get("", response_model=ProgressData)
def get_progress() -> ProgressData:
    return load_progress()
