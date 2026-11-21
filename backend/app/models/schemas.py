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


class MetalOption(BaseModel):
    metal: str
    mofs: int


class EmbeddingDemoResponse(BaseModel):
    smiles: str
    metal_filter: str | None = None
    model: str
    embedding_dim: int
    embedding_norm: float
    neighbors: list[EmbeddingNeighbor]


class ResearchQuestion(BaseModel):
    id: str
    text: str
    answer: str | None = None


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


class ResultFigure(BaseModel):
    src: str
    caption: str
    alt: str


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
    figures: list[ResultFigure] = Field(default_factory=list)
    gallery_url: str | None = None
    gallery_title: str | None = None


class GraphNode(BaseModel):
    smiles: str
    mofs: int
    mean_pld: float
    x: float
    y: float


class GraphNeighbourhood(BaseModel):
    seed: int
    nodes: list[GraphNode]
    # [node_index, node_index, number of MOF-level edges that collapse into this link]
    fingerprint: list[tuple[int, int, int]]
    semantic: list[tuple[int, int, int]]
    shared_edges: int


class GraphComparisonStats(BaseModel):
    mofs: int
    distinct_linkers: int
    mof_edges: dict[str, int]
    mof_jaccard: float
    same_linker_edge_share: dict[str, float]
    linker_pair_edges: dict[str, int]
    linker_pair_jaccard: float
    components: dict[str, int]
    mean_abs_pld_gap_between_neighbours: dict[str, float]
    mean_abs_pld_gap_random_pair: float


class GraphComparison(BaseModel):
    model: str
    k: int
    stats: GraphComparisonStats
    neighbourhoods: list[GraphNeighbourhood]
