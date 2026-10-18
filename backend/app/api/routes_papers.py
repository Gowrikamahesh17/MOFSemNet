from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings
from app.models.schemas import PapersData
from app.services.content_loader import load_papers

router = APIRouter(prefix="/api/papers", tags=["papers"])


@router.get("", response_model=PapersData)
def get_papers() -> PapersData:
    """All literature clusters and papers, for /studied (narrative) and /references
    (bibliography) — both pages render the same data differently."""
    return load_papers()


@router.get("/{paper_id}/pdf")
def get_paper_pdf(paper_id: str):
    """Serves the locally-held PDF for a paper, when one exists. Papers without a local
    PDF (has_local_pdf=false) link out to their URL instead — see content/papers.yaml's
    comment on why those two papers aren't bundled."""
    papers = {paper.id: paper for paper in load_papers().papers}
    paper = papers.get(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Unknown paper id '{paper_id}'")
    if not paper.has_local_pdf or not paper.local_pdf:
        raise HTTPException(status_code=404, detail=f"No local PDF for '{paper_id}' — see its url field instead")

    pdf_path = settings.papers_dir / paper.local_pdf
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF file missing on disk: {paper.local_pdf}")
    return FileResponse(pdf_path, media_type="application/pdf", filename=paper.local_pdf)
