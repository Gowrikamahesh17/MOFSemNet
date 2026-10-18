from fastapi import APIRouter

from app.services.content_loader import reload_all

router = APIRouter(prefix="/api/content", tags=["content"])


@router.post("/reload")
def reload_content() -> dict:
    """Re-reads content/*.yaml from disk without restarting the server — for editing a
    section, model, or diagram node and seeing it live. See docs/ADDING_CONTENT.md."""
    reload_all()
    return {"status": "reloaded"}
