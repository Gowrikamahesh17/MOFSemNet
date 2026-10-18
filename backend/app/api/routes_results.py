from fastapi import APIRouter, HTTPException

from app.models.schemas import StepResult
from app.services.content_loader import ContentError, load_step_result

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("/{step_id}", response_model=StepResult)
def get_step_result(step_id: str) -> StepResult:
    """Real, precomputed results for Steps 1-4 (content/results/*.yaml) — never invented,
    never run live (graph construction and sparsification take 20-90s, not demo-safe)."""
    try:
        return load_step_result(step_id)
    except ContentError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
