from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.models.schemas import GraphComparison
from app.services.content_loader import GRAPH_COMPARISON_MODELS, ContentError, load_graph_comparison
from src.common.structure_svg import structure_svg

router = APIRouter(prefix="/api/graphs", tags=["graphs"])


@router.get("/comparison", response_model=GraphComparison)
def get_graph_comparison(model: Literal["chemberta", "molt5"] = "chemberta") -> GraphComparison:
    """Real linker neighbourhoods from the cached fingerprint and semantic graphs, plus summary
    statistics on how their edges differ. Precomputed — see scripts/export_graph_comparison.py."""
    try:
        return load_graph_comparison(model)
    except ContentError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/structure")
def get_structure(smiles: str = Query(..., max_length=2000)) -> Response:
    """A drawn structure for one linker shown in the comparison. Only linkers that appear in
    the comparison data are drawn, so this isn't a general-purpose rendering endpoint."""
    known: set[str] = set()
    for model in GRAPH_COMPARISON_MODELS:
        try:
            data = load_graph_comparison(model)
        except ContentError:
            continue
        known |= {node.smiles for hood in data.neighbourhoods for node in hood.nodes}
    if smiles not in known:
        raise HTTPException(status_code=404, detail="Linker not part of the graph comparison")
    svg = structure_svg(smiles)
    if not svg:
        raise HTTPException(status_code=404, detail="Linker could not be drawn")
    return Response(content=svg, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})
