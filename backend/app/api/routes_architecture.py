from fastapi import APIRouter

from app.models.schemas import ArchitectureData
from app.services.content_loader import load_architecture

router = APIRouter(prefix="/api/architecture", tags=["architecture"])


@router.get("", response_model=ArchitectureData)
def get_architecture() -> ArchitectureData:
    """Nodes and edges for the interactive explorer (/architecture). The frontend lays
    this out with dagre, so node order and any coordinates are the frontend's concern, not
    this response's."""
    return load_architecture()
