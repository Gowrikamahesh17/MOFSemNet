from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.schemas import EmbeddingDemoResponse, MetalOption
from app.services.embedding_service import (
    InvalidMetalError,
    InvalidSMILESError,
    embed_and_find_neighbors,
    list_metals,
)

router = APIRouter(prefix="/api/demo", tags=["demo"])


class EmbedRequest(BaseModel):
    smiles: str
    metal: str | None = None


@router.post("/embed", response_model=EmbeddingDemoResponse)
def embed(request: EmbedRequest) -> EmbeddingDemoResponse:
    """Embeds a user-submitted SMILES string live with the real ChemBERTa model, then finds
    its nearest neighbours among the real cached embeddings for all 14,296 MOFs. The only
    endpoint in this app that actually runs a model — Steps 3/4 are too slow (20-90s) to be
    demo-safe live, so those are served as precomputed results instead (see routes_results)."""
    try:
        return embed_and_find_neighbors(request.smiles, metal=request.metal)
    except (InvalidSMILESError, InvalidMetalError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/metals", response_model=list[MetalOption])
def metals() -> list[MetalOption]:
    """Every metal in the real dataset (53) with its MOF count, for the demo's metal filter."""
    return list_metals()
