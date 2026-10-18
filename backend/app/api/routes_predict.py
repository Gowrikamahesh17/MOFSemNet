from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.schemas import EmbeddingDemoResponse
from app.services.embedding_service import InvalidSMILESError, embed_and_find_neighbors

router = APIRouter(prefix="/api/demo", tags=["demo"])


class EmbedRequest(BaseModel):
    smiles: str


@router.post("/embed", response_model=EmbeddingDemoResponse)
def embed(request: EmbedRequest) -> EmbeddingDemoResponse:
    """Embeds a user-submitted SMILES string live with the real ChemBERTa model, then finds
    its nearest neighbours among the real cached embeddings for all 14,296 MOFs. The only
    endpoint in this app that actually runs a model — Steps 3/4 are too slow (20-90s) to be
    demo-safe live, so those are served as precomputed results instead (see routes_results)."""
    try:
        return embed_and_find_neighbors(request.smiles)
    except InvalidSMILESError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
