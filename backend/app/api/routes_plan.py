from fastapi import APIRouter

from app.models.schemas import PlanData
from app.services.content_loader import load_plan

router = APIRouter(prefix="/api/plan", tags=["plan"])


@router.get("", response_model=PlanData)
def get_plan() -> PlanData:
    return load_plan()
