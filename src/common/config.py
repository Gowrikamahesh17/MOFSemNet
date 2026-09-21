"""Single central config for the new work (Steps 2-6: language embeddings, graph construction,
sparsification, model comparison, explainability). src/baseline/config.py (Step 1, ported as-is)
is left untouched and keeps its own dataset/task switches — this module is deliberately scoped to
everything downstream of the baseline reproduction.

Every run-time-tunable value that Steps 2-6 need lives here so it's never hardcoded or repeated
across scripts: dataset path, seeds, pruning levels, which language model to embed with, which
device to run on. A script changes behavior by constructing a different RuntimeConfig, never by
editing a magic number inline.

Scope guardrails from the thesis expose (kept explicit here so they're visible at the one place
that could violate them):
  - One dataset: the large, 14,296-MOF MOFCSD set. Never the small 2,000-MOF one from Step 1.
  - One prediction target: continuous PLD (pld_value). Never used as an input feature anywhere
    downstream (src/language, src/graphs, src/evaluation must not read pld_value except as the
    training/eval label).
  - One pretrained language model *per run*, chosen via `language_model` — both ChemBERTa and
    MolT5 are wired in (so their results can be compared against each other), but a single run
    embeds with exactly one of them; nothing here runs a run with the other model implicitly.
  - At most two GNN architectures (GCN, GraphSAGE) and two retention levels (0.3, 0.5).

Note on retention vs. threshold: this config (and src.sparsification_baselines) speak in terms
of *retention* — the fraction of nodes kept, matching the thesis exposé's "prune at 30% and 50%
retention" — while the ported src/baseline/black_hole_sparsification.py speaks in terms of a
removal `threshold` (fraction of nodes/edges dropped). src.evaluation.run_comparison is the one
place that converts between them (threshold = 1 - retention) when calling the ported Black Hole
function, so nothing else in the new modules needs to remember the sign flip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

LARGE_DATASET_PATH = "data/raw/MOFCSD.csv"
PLD_TARGET_COLUMN = "pld_value"

GRAPH_TYPES = ("fingerprint", "semantic", "hybrid")
SPARSIFICATION_METHODS = ("black_hole", "random", "k_center")
RETENTION_LEVELS = (0.3, 0.5)  # fraction of nodes KEPT after sparsification
GNN_ARCHITECTURES = ("gcn", "graphsage")
NON_GRAPH_BASELINES = ("random_forest", "mlp")

LANGUAGE_MODEL_CHOICES = ("chemberta", "molt5")
# Hugging Face Hub ids. ChemBERTa is a RoBERTa encoder; MolT5 is a T5 encoder-decoder (only the
# encoder is used for embeddings) — see src/language/chemberta.py and src/language/molt5.py.
LANGUAGE_MODEL_HF_IDS = {
    "chemberta": "DeepChem/ChemBERTa-zinc-base-v1",
    "molt5": "laituan245/molt5-base",
}

DEFAULT_SEEDS = (0, 1, 2, 3, 4)  # >=5 seeds per configuration, per Step 5's reporting requirement


@dataclass
class RuntimeConfig:
    # --- Data (Step 1 continuity) ---
    dataset_path: str = LARGE_DATASET_PATH
    target_column: str = PLD_TARGET_COLUMN

    # --- Reproducible splits: fixed once, test set never touched until final evaluation ---
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15
    split_random_state: int = 42

    # --- Step 5: seeds run per configuration ---
    seeds: tuple[int, ...] = DEFAULT_SEEDS

    # --- Step 2: language representation ---
    language_model: str = "chemberta"

    # --- Step 3: graph types to build ---
    graph_types: tuple[str, ...] = GRAPH_TYPES

    # --- Step 4: sparsification ---
    sparsification_methods: tuple[str, ...] = SPARSIFICATION_METHODS
    retention_levels: tuple[float, ...] = RETENTION_LEVELS
    # Black Hole gravity weights — reused default from src/baseline/config.py (paper's
    # equal-weighting configuration); independently overridable here for the new graph types.
    gravity_degree_weight: float = 0.33
    gravity_betweenness_weight: float = 0.33
    gravity_edge_weight_sum_weight: float = 0.33

    # --- Step 5: models to compare ---
    non_graph_baselines: tuple[str, ...] = NON_GRAPH_BASELINES
    gnn_architectures: tuple[str, ...] = GNN_ARCHITECTURES

    # --- Device (resolved once via src.common.runtime.get_device) ---
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"

    # --- Cache / output locations ---
    embeddings_cache_dir: str = "data/embeddings"
    graphs_cache_dir: str = "data/processed/graphs"
    results_dir: str = "outputs/results"
    figures_dir: str = "outputs/figures"
    models_dir: str = "outputs/models"

    extra: dict = field(default_factory=dict)  # escape hatch for one-off experiment overrides

    def __post_init__(self) -> None:
        if self.language_model not in LANGUAGE_MODEL_CHOICES:
            raise ValueError(f"language_model must be one of {LANGUAGE_MODEL_CHOICES}, got {self.language_model!r}")
        for graph_type in self.graph_types:
            if graph_type not in GRAPH_TYPES:
                raise ValueError(f"Unknown graph_type {graph_type!r}, expected one of {GRAPH_TYPES}")
        for method in self.sparsification_methods:
            if method not in SPARSIFICATION_METHODS:
                raise ValueError(f"Unknown sparsification method {method!r}, expected one of {SPARSIFICATION_METHODS}")
        for level in self.retention_levels:
            if not 0.0 < level < 1.0:
                raise ValueError(f"retention_levels entries must be in (0, 1), got {level}")
        for arch in self.gnn_architectures:
            if arch not in GNN_ARCHITECTURES:
                raise ValueError(f"Unknown GNN architecture {arch!r}, expected one of {GNN_ARCHITECTURES}")
        if len(self.seeds) < 5:
            raise ValueError(f"Step 5 requires at least 5 seeds per configuration, got {len(self.seeds)}")
        split_total = self.train_frac + self.val_frac + self.test_frac
        if abs(split_total - 1.0) > 1e-6:
            raise ValueError(f"train_frac + val_frac + test_frac must sum to 1.0, got {split_total}")

    @property
    def language_model_hf_id(self) -> str:
        return LANGUAGE_MODEL_HF_IDS[self.language_model]

    @property
    def gravity_weights_normalized(self) -> tuple[float, float, float]:
        raw = (self.gravity_degree_weight, self.gravity_betweenness_weight, self.gravity_edge_weight_sum_weight)
        total = sum(raw)
        if total == 0:
            return (1 / 3, 1 / 3, 1 / 3)
        return tuple(w / total for w in raw)


DEFAULT_CONFIG = RuntimeConfig()
