"""Loads and validates content/*.yaml at startup. A malformed content file is a real
failure boundary (someone hand-editing YAML every month will typo a field eventually), so
this raises a specific, actionable error rather than letting a KeyError surface later deep
in a request handler.

Loaded once and cached: these files change when someone edits them and restarts the app,
not per-request.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

from app.core.config import settings
from app.models.schemas import (
    ArchitectureData,
    GraphComparison,
    PapersData,
    PlanData,
    ProgressData,
    StepResult,
)


class ContentError(RuntimeError):
    """A content/*.yaml file is missing, malformed, or internally inconsistent (e.g. an
    edge or a reference pointing at a node/paper id that doesn't exist)."""


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise ContentError(f"Content file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        try:
            return yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ContentError(f"{path} is not valid YAML: {exc}") from exc


@lru_cache(maxsize=1)
def load_architecture() -> ArchitectureData:
    raw = _load_yaml(settings.content_dir / "architecture.yaml")
    try:
        data = ArchitectureData.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed content error below
        raise ContentError(f"architecture.yaml failed validation: {exc}") from exc

    node_ids = {node.id for node in data.nodes}
    for edge in data.edges:
        if edge.from_ not in node_ids:
            raise ContentError(f"architecture.yaml: edge references unknown node '{edge.from_}'")
        if edge.to not in node_ids:
            raise ContentError(f"architecture.yaml: edge references unknown node '{edge.to}'")

    paper_ids = {paper.id for paper in load_papers().papers}
    for node in data.nodes:
        for ref in node.references:
            if ref not in paper_ids:
                raise ContentError(
                    f"architecture.yaml: node '{node.id}' references unknown paper '{ref}' "
                    "(check content/papers.yaml)"
                )

    return data


@lru_cache(maxsize=1)
def load_papers() -> PapersData:
    raw = _load_yaml(settings.content_dir / "papers.yaml")
    try:
        data = PapersData.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed content error below
        raise ContentError(f"papers.yaml failed validation: {exc}") from exc

    cluster_ids = {cluster.id for cluster in data.clusters}
    for paper in data.papers:
        if paper.cluster not in cluster_ids:
            raise ContentError(
                f"papers.yaml: paper '{paper.id}' has unknown cluster '{paper.cluster}'"
            )

    return data


@lru_cache(maxsize=1)
def load_plan() -> PlanData:
    raw = _load_yaml(settings.content_dir / "plan.yaml")
    try:
        return PlanData.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed content error below
        raise ContentError(f"plan.yaml failed validation: {exc}") from exc


@lru_cache(maxsize=1)
def load_progress() -> ProgressData:
    raw = _load_yaml(settings.content_dir / "progress.yaml")
    try:
        return ProgressData.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed content error below
        raise ContentError(f"progress.yaml failed validation: {exc}") from exc


_VALID_STEP_IDS = {"step1", "step2", "step3", "step4"}


@lru_cache(maxsize=None)
def load_step_result(step_id: str) -> StepResult:
    if step_id not in _VALID_STEP_IDS:
        raise ContentError(f"Unknown step id '{step_id}', expected one of {sorted(_VALID_STEP_IDS)}")
    raw = _load_yaml(settings.content_dir / "results" / f"{step_id}.yaml")
    try:
        return StepResult.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed content error below
        raise ContentError(f"results/{step_id}.yaml failed validation: {exc}") from exc


GRAPH_COMPARISON_MODELS = ("chemberta", "molt5")


@lru_cache(maxsize=None)
def load_graph_comparison(model: str = "chemberta") -> GraphComparison:
    """Precomputed by scripts/export_graph_comparison.py from the cached Step 3 graphs."""
    if model not in GRAPH_COMPARISON_MODELS:
        raise ContentError(f"Unknown model '{model}', expected one of {list(GRAPH_COMPARISON_MODELS)}")
    path = settings.content_dir / f"graph_comparison_{model}.json"
    if not path.exists():
        raise ContentError(
            f"Content file not found: {path} "
            f"(run scripts/export_graph_comparison.py --model {model} to create it)"
        )
    try:
        with open(path, encoding="utf-8") as fh:
            return GraphComparison.model_validate(json.load(fh))
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed content error below
        raise ContentError(f"graph_comparison_{model}.json failed validation: {exc}") from exc


def reload_all() -> None:
    """Clears the cache so the next request re-reads content/*.yaml from disk. Used by the
    dev-only /api/content/reload endpoint — editing a YAML file and refreshing the browser
    shouldn't require restarting the server."""
    load_architecture.cache_clear()
    load_papers.cache_clear()
    load_plan.cache_clear()
    load_progress.cache_clear()
    load_step_result.cache_clear()
    load_graph_comparison.cache_clear()
