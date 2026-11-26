import dataclasses
import sys
from functools import lru_cache
from pathlib import Path

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


_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


@lru_cache(maxsize=1)
def _predictor():
    from src.demo.predictor import MOFPredictor   # loads the saved models once (outputs/models/mofsemnet_models)
    return MOFPredictor()


class PredictRequest(BaseModel):
    smiles: str
    metal: str
    lcd: float | None = None
    lfs: float | None = None
    graph_type: str = "semantic_chemberta"
    k: int = 3
    pruning: str = "none"


@router.post("/predict")
def predict(request: PredictRequest) -> dict:
    """Predicts the PLD of a linker + metal (optionally with LCD/LFS) with all saved tree models, the no-edge GNNs and the
    GNNs of the chosen graph / k / pruning. A new MOF is added to the graph as one extra node. Returns the leaderboard
    (best test R2 first) with 90% conformal intervals and the most similar known MOFs."""
    mode = "with_pore" if request.lcd is not None and request.lfs is not None else "chemistry_only"
    try:
        r = _predictor().predict(request.smiles, request.metal, mode, request.lcd, request.lfs,
                                 request.graph_type, request.k, request.pruning)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"saved models not found: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return dataclasses.asdict(r)


_LINKER_NAMES = {"OC(=O)c1ccc(cc1)C(=O)O": "terephthalic acid (BDC)", "OC=O": "formic acid", "OC(=O)C(=O)O": "oxalic acid",
                 "OC(=O)c1cc(cc(c1)C(=O)O)C(=O)O": "trimesic acid (BTC)"}


@router.get("/linker")
def linker(smiles: str) -> dict:
    """Structure drawing (RDKit SVG) and basic facts for a linker SMILES: whether it is in the data (matched by canonical
    SMILES), in how many MOFs, its formula and molar mass. Used live next to the Live Demo's SMILES box."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    from src.common.structure_svg import structure_svg

    mol = Chem.MolFromSmiles(smiles) if smiles.strip() else None
    if mol is None or mol.GetNumAtoms() == 0:
        return {"valid": False}
    p = _predictor()
    dataset = p.canonical.get(Chem.MolToSmiles(mol))
    n = int((p.mofs["linker_smiles"] == dataset).sum()) if dataset else 0
    names = {Chem.MolToSmiles(Chem.MolFromSmiles(k)): v for k, v in _LINKER_NAMES.items()}
    sub = p.mofs[p.mofs["linker_smiles"] == dataset] if dataset else None
    usage = {} if sub is None or sub.empty else {
        "metals": [[m, int(c)] for m, c in sub["metal"].value_counts().head(4).items()], "n_metals": int(sub["metal"].nunique()),
        "pld_min": round(float(sub["pld"].min()), 2), "pld_max": round(float(sub["pld"].max()), 2), "pld_mean": round(float(sub["pld"].mean()), 2)}
    return {**usage, "valid": True, "svg": structure_svg(smiles, 260, 150), "known": bool(dataset), "n_mofs": n,
            "name": names.get(Chem.MolToSmiles(mol), ""), "formula": rdMolDescriptors.CalcMolFormula(mol),
            "mol_weight": round(Descriptors.MolWt(mol), 1), "heavy_atoms": mol.GetNumHeavyAtoms()}
