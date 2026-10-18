"""Pydantic response models. These mirror content/architecture.yaml and content/papers.yaml
exactly — a validation error here means the content file has a typo or a missing field,
not a code bug, which is the point: editing a YAML file should fail loudly and specifically
if it's wrong, not silently break the frontend.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

NodeStatus = Literal["done", "next", "planned"]


class ArchitectureNode(BaseModel):
    id: str
    label: str
    group: str
    status: NodeStatus
    summary: str
    description: str
    why_chosen: str | None = None
    studied: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    metrics: dict[str, str] = Field(default_factory=dict)


class ArchitectureEdge(BaseModel):
    from_: str = Field(alias="from")
    to: str
    animated: bool = False

    model_config = {"populate_by_name": True}


class ArchitectureData(BaseModel):
    nodes: list[ArchitectureNode]
    edges: list[ArchitectureEdge]


class PaperCluster(BaseModel):
    id: str
    label: str
    narrative: str


class Paper(BaseModel):
    id: str
    citation_key: str
    cluster: str
    title: str
    authors: str
    year: int
    venue: str
    url: str
    summary: str
    has_local_pdf: bool = False
    local_pdf: str | None = None


class PapersData(BaseModel):
    clusters: list[PaperCluster]
    papers: list[Paper]


class EmbeddingNeighbor(BaseModel):
    refcode: str
    linker_smiles: str
    metal: str
    pld_value: float
    cosine_similarity: float


class EmbeddingDemoResponse(BaseModel):
    smiles: str
    model: str
    embedding_dim: int
    embedding_norm: float
    neighbors: list[EmbeddingNeighbor]


class ResearchQuestion(BaseModel):
    id: str
    text: str


class PlanData(BaseModel):
    context: str
    thesis_statement: str
    research_questions: list[ResearchQuestion]
    guardrails: list[str]
    negative_result_framing: str


class WorkPackage(BaseModel):
    id: str
    label: str
    window: str
    status: NodeStatus


class Milestone(BaseModel):
    date: str
    label: str


class ProgressData(BaseModel):
    work_packages: list[WorkPackage]
    milestones: list[Milestone]
    ahead_of_schedule_note: str


class ResultStat(BaseModel):
    label: str
    value: str
    emphasis: bool = False


class StepResult(BaseModel):
    title: str
    status: NodeStatus
    summary: str
    stats: list[ResultStat] = Field(default_factory=list)
    finding: str | None = None
    headline: str | None = None
    headline_label: str | None = None
    retention_shown: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
