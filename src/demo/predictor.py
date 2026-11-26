"""Predict the PLD of a new (or known) MOF with the saved models — the engine behind the live demo.

Uses the models trained and saved by MOFSemNet_server_save_models.ipynb (unpacked into outputs/models/mofsemnet_models/).

How a prediction is made
  1. The linker SMILES is checked with RDKit and turned into the same inputs the models were trained on: a 1,024-bit Morgan
     fingerprint (radius 2), the metal as one-hot flags (one of the 53 metals in the data), and, in "with_pore" mode, the two
     pore-geometry values (largest cavity diameter, largest free sphere) supplied by the user.
  2. Tree models (ExtraTrees, Random Forest, XGBoost, LightGBM, CatBoost) predict directly from that row.
  3. GNNs with no edges predict from the row alone.
  4. GNNs with a graph: the new MOF becomes one extra node. It is joined to its k most similar existing MOFs by the same rule
     the graph was built with (Tanimoto on fingerprints, cosine on ChemBERTa / MolT5 embeddings, or their 50/50 blend), and,
     for a Black Hole pruned graph, only to MOFs that survived pruning. One forward pass gives its prediction. A linker that
     is not in the data is embedded live with the same language model and pooling as the cached embeddings.
  5. Every prediction comes with a 90% interval: prediction +/- the split-conformal half-width measured on the validation MOFs.

Only the new node's own k edges are added (existing MOFs do not re-pick it as a neighbour), which is the usual way to place an
unseen node into a fixed graph.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODELS_DIR = os.path.join(ROOT, "outputs", "models", "mofsemnet_models")
DEFAULT_EMB_DIR = os.path.join(ROOT, "data", "embeddings")
INVALID_SMILES, INVALID_REPLACEMENT = "F[Si](F)(F)(F)(F)F", "c1ccccc1"   # same replacement as in training
GRAPH_TYPES = ("fingerprint", "semantic_chemberta", "hybrid_chemberta", "semantic_molt5", "hybrid_molt5")
PRUNINGS = ("none", "black_hole_30", "black_hole_50")
MODES = ("with_pore", "chemistry_only")


def _build_net(kind: str, d_in: int, p: dict):
    """Same architecture as the notebook's Net class (names match, so the saved state_dict loads)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            h, L, self.drop = p["hidden"], p["layers"], p["dropout"]
            if kind == "gcn":
                self.convs = nn.ModuleList([GCNConv(d_in if i == 0 else h, 1 if i == L - 1 else h) for i in range(L)])
            elif kind == "graphsage":
                self.convs = nn.ModuleList([SAGEConv(d_in if i == 0 else h, 1 if i == L - 1 else h) for i in range(L)])
            elif kind == "residual_gcn":
                self.lin_in, self.lin_out = nn.Linear(d_in, h), nn.Linear(h, 1)
                self.blocks = nn.ModuleList([GCNConv(h, h) for _ in range(L)])
                self.gates = nn.Parameter(torch.zeros(L))
            else:
                raise ValueError(kind)

        def forward(self, x, ei, ew=None):
            if self.kind in ("gcn", "graphsage"):
                for conv in self.convs[:-1]:
                    x = F.dropout(F.relu(conv(x, ei, ew) if self.kind == "gcn" else conv(x, ei)), p=self.drop, training=self.training)
                return self.convs[-1](x, ei, ew) if self.kind == "gcn" else self.convs[-1](x, ei)
            h = F.dropout(F.relu(self.lin_in(x)), p=self.drop, training=self.training)
            for gate, conv in zip(self.gates, self.blocks):
                h = h + gate * F.relu(conv(h, ei, ew))
            return self.lin_out(h)

    return Net()


@dataclass
class Prediction:
    model_id: str
    family: str
    model: str
    graph: str
    k: int
    pruning: str
    pld: float
    low90: float
    high90: float
    test_r2: float
    test_mae: float


@dataclass
class PredictionResult:
    smiles: str
    metal: str
    mode: str
    known_linker: bool
    mofs_with_this_linker: int
    known_mofs_same_linker_and_metal: list = field(default_factory=list)   # [(refcode, true PLD)]
    predictions: list = field(default_factory=list)                         # list[Prediction], best test R2 first
    nearest_mofs: list = field(default_factory=list)                        # [{refcode, linker_smiles, metal, pld, similarity}]
    notes: list = field(default_factory=list)


class MOFPredictor:
    def __init__(self, models_dir: str = DEFAULT_MODELS_DIR, embeddings_dir: str = DEFAULT_EMB_DIR):
        from rdkit import RDLogger
        from rdkit.Chem import rdFingerprintGenerator

        RDLogger.DisableLog("rdApp.*")
        if not os.path.exists(os.path.join(models_dir, "manifest.json")):
            raise FileNotFoundError(f"No saved models at {models_dir}. Unzip mofsemnet_models.zip into outputs/models/.")
        self.dir, self.emb_dir = models_dir, embeddings_dir
        self.manifest = json.load(open(os.path.join(models_dir, "manifest.json")))
        self.layout = json.load(open(os.path.join(models_dir, "features", "feature_layout.json")))
        self.metals = self.layout["metals_in_one_hot_order"]
        self.results = pd.read_csv(os.path.join(models_dir, "results", "model_results.csv"))
        self.mofs = pd.read_csv(os.path.join(models_dir, "features", "mofs.csv"))
        self.scalers = {m: json.load(open(os.path.join(models_dir, "features", f"scaler_{m}.json")))["input_scaler"] for m in MODES}
        self.hybrid_beta = float(self.manifest["config"].get("hybrid_beta", 0.5))
        self._gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.layout["morgan"]["radius"], fpSize=self.layout["morgan"]["bits"])
        # fingerprints of the known MOFs (computed once per distinct linker)
        cache = {s: self._fingerprint(s) for s in self.mofs["linker_smiles"].unique()}
        # canonical form -> the dataset's own spelling, so a known linker written differently is still recognised
        from rdkit import Chem
        self.canonical = {}
        for s_ in self.mofs["linker_smiles"].unique():
            m_ = Chem.MolFromSmiles(s_)
            if m_ is not None:
                self.canonical.setdefault(Chem.MolToSmiles(m_), s_)
        self.fp = np.stack([cache[s] for s in self.mofs["linker_smiles"]]).astype(np.float32)
        self.fp_sum = self.fp.sum(1)
        metal_index = {m: i for i, m in enumerate(self.metals)}
        onehot = np.zeros((len(self.mofs), len(self.metals)), np.float32)
        onehot[np.arange(len(self.mofs)), self.mofs["metal"].map(metal_index).values] = 1.0
        raw = pd.read_csv(os.path.join(ROOT, "data", "raw", "MOFCSD.csv"), index_col=0) if os.path.exists(os.path.join(ROOT, "data", "raw", "MOFCSD.csv")) else None
        if raw is None:
            raise FileNotFoundError("data/raw/MOFCSD.csv is needed for the known MOFs' pore columns")
        raw = raw.loc[self.mofs["refcode"]]
        pore = raw[["Largest Cavity Diameter", "Largest Free Sphere"]].values.astype(np.float32)
        self.features = {"with_pore": np.hstack([self.fp, pore, onehot]), "chemistry_only": np.hstack([self.fp, onehot])}
        self.scaled = {m: self._scale(self.features[m], m) for m in MODES}
        self._emb, self._emb_lookup, self._embedders, self._trees, self._gnns, self._graphs = {}, {}, {}, {}, {}, {}

    # ------------------------------------------------------------------ inputs
    def _fingerprint(self, smiles: str) -> np.ndarray:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return np.zeros(self.layout["morgan"]["bits"], np.float32)
        return self._gen.GetFingerprintAsNumPy(mol).astype(np.float32)

    def _scale(self, X: np.ndarray, mode: str) -> np.ndarray:
        s = self.scalers[mode]
        out = X.astype(np.float32).copy()
        cols = np.array(s["columns"], dtype=int)
        if len(cols):
            out[:, cols] = (out[:, cols] - np.array(s["mean"], np.float32)) / np.array(s["std"], np.float32)
        return out

    def feature_row(self, smiles: str, metal: str, mode: str, lcd: float | None = None, lfs: float | None = None) -> np.ndarray:
        fp = self._fingerprint(smiles)
        onehot = np.zeros(len(self.metals), np.float32)
        onehot[self.metals.index(metal)] = 1.0
        parts = [fp] + ([np.array([lcd, lfs], np.float32)] if mode == "with_pore" else []) + [onehot]
        return np.concatenate(parts)[None, :]

    def _embeddings(self, lm: str) -> np.ndarray:
        if lm not in self._emb:
            z = np.load(os.path.join(self.emb_dir, f"{lm}_embeddings.npz"), allow_pickle=True)
            lookup = {}
            for s, v in zip(z["smiles"], z["embeddings"]):
                lookup.setdefault(str(s), v)
            self._emb_lookup[lm] = lookup
            E = np.stack([lookup[s] for s in self.mofs["linker_smiles"]]).astype(np.float32)
            self._emb[lm] = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        return self._emb[lm]

    def embed(self, smiles: str, lm: str) -> tuple[np.ndarray, bool]:
        """(unit-length embedding, computed_live). Known linkers use the cached vector; new ones are embedded live."""
        self._embeddings(lm)
        if smiles in self._emb_lookup[lm]:
            v, live = np.asarray(self._emb_lookup[lm][smiles], np.float32), False
        else:
            if lm not in self._embedders:
                from src.common.config import RuntimeConfig
                from src.language.embeddings import get_embedder
                self._embedders[lm] = get_embedder(RuntimeConfig(language_model=lm))
            v, live = self._embedders[lm].embed([smiles])[0].astype(np.float32), True
        return v / max(float(np.linalg.norm(v)), 1e-12), live

    def similarity(self, smiles: str, graph_type: str) -> tuple[np.ndarray, bool]:
        """Similarity of the new linker to every known MOF, by the rule `graph_type` was built with."""
        fp = self._fingerprint(smiles)
        with np.errstate(all="ignore"):   # macOS Accelerate can raise spurious float32 matmul warnings
            inter = self.fp @ fp
        tan = inter / np.maximum(self.fp_sum + fp.sum() - inter, 1.0)
        if graph_type == "fingerprint":
            return tan, False
        kind, lm = graph_type.split("_", 1)
        v, live = self.embed(smiles, lm)
        with np.errstate(all="ignore"):
            cos = self._embeddings(lm) @ v
        return (cos if kind == "semantic" else self.hybrid_beta * cos + (1 - self.hybrid_beta) * tan), live

    # ------------------------------------------------------------------ models
    def _tree(self, row):
        f = os.path.join(self.dir, row.file)
        if f not in self._trees:
            if f.endswith(".joblib"):
                import joblib
                m = joblib.load(f)
                self._trees[f] = m.predict
            elif f.endswith(".json"):
                import xgboost as xgb
                m = xgb.XGBRegressor(); m.load_model(f)
                self._trees[f] = m.predict
            elif f.endswith(".txt"):
                import lightgbm as lgb
                self._trees[f] = lgb.Booster(model_file=f).predict
            else:
                from catboost import CatBoostRegressor
                m = CatBoostRegressor(); m.load_model(f)
                self._trees[f] = m.predict
        return self._trees[f]

    def _gnn(self, row):
        import torch
        f = os.path.join(self.dir, row.file)
        if f not in self._gnns:
            ck = torch.load(f, map_location="cpu")
            net = _build_net(ck["model"], ck["d_in"], ck["settings"])
            net.load_state_dict(ck["state_dict"])
            net.eval()
            self._gnns[f] = (net, ck)
        return self._gnns[f]

    def _graph(self, graph_file):
        if graph_file not in self._graphs:
            z = np.load(os.path.join(self.dir, graph_file))
            self._graphs[graph_file] = (z["edge_index"], z["edge_weight"], z["kept_nodes"])
        return self._graphs[graph_file]

    # ------------------------------------------------------------------ main entry point
    def predict(self, smiles: str, metal: str, mode: str = "chemistry_only", lcd: float | None = None, lfs: float | None = None,
                graph_type: str = "semantic_chemberta", k: int = 3, pruning: str = "none", all_graphs: bool = False,
                n_nearest: int = 5) -> PredictionResult:
        """Predict with every tree model and no-edge GNN of `mode`, plus the GNNs for (`graph_type`, `k`, `pruning`)
        — or every GNN of the mode when all_graphs=True."""
        import torch
        from rdkit import Chem

        smiles = (smiles or "").strip()
        if smiles == INVALID_SMILES:
            smiles = INVALID_REPLACEMENT
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Not a valid SMILES: {smiles!r}")
        typed = smiles
        smiles = self.canonical.get(Chem.MolToSmiles(mol), smiles)
        if metal not in self.metals:
            raise ValueError(f"Unknown metal {metal!r}. The models know these 53: {', '.join(self.metals)}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        notes = []
        if mode == "with_pore":
            if lcd is None or lfs is None or lcd < 0 or lfs < 0:
                raise ValueError("with_pore mode needs the largest cavity diameter (lcd) and largest free sphere (lfs), in Angstrom")
            if lfs > lcd:
                notes.append("The largest free sphere is normally not larger than the largest cavity diameter; please check the values.")
        if not all_graphs and (graph_type not in GRAPH_TYPES or pruning not in PRUNINGS or k not in (3, 5, 10)):
            raise ValueError(f"graph_type in {GRAPH_TYPES}, k in (3, 5, 10), pruning in {PRUNINGS}")

        raw_row = self.feature_row(smiles, metal, mode, lcd, lfs)
        scaled_row = self._scale(raw_row, mode)
        same = self.mofs[self.mofs["linker_smiles"] == smiles]
        result = PredictionResult(smiles=smiles, metal=metal, mode=mode, known_linker=len(same) > 0, mofs_with_this_linker=int(len(same)),
                                  known_mofs_same_linker_and_metal=[(r.refcode, round(float(r.pld), 2)) for r in same[same["metal"] == metal].itertuples()][:10],
                                  notes=notes)
        if smiles != typed:
            notes.append(f"This linker is in the data written as {smiles}; that form is used (same molecule as you typed).")
        if mode == "chemistry_only":
            notes.append("Linker + metal only: typical accuracy is R2 about 0.6; supplying the pore geometry raises it to about 0.9.")

        res = self.results[self.results["mode"] == mode]
        sel = (res.family == "tree") | ((res.family == "gnn") & (res.graph == "none"))
        if all_graphs:
            sel |= (res.family == "gnn")
        else:
            sel |= (res.family == "gnn") & (res.graph == graph_type) & (res.k == k) & (res.pruning == pruning)
        sims, live_used = {}, False
        X_all = torch.tensor(self.scaled[mode])
        for row in res[sel].itertuples():
            if row.family == "tree":
                pld = float(self._tree(row)(raw_row)[0])
            else:
                net, ck = self._gnn(row)
                if row.graph == "none":
                    with torch.no_grad():
                        out = net(torch.tensor(scaled_row), torch.zeros((2, 0), dtype=torch.long), None)
                    pld = float(out.squeeze()) * ck["target_scaler"]["std"] + ck["target_scaler"]["mean"]
                else:
                    if row.graph not in sims:
                        sims[row.graph], live = self.similarity(smiles, row.graph)
                        live_used |= live
                    ei, ew, kept = self._graph(ck["graph_file"])
                    s = np.where(kept, sims[row.graph], -np.inf)
                    nb = np.argpartition(-s, row.k - 1)[: row.k]
                    new = len(self.mofs)
                    add_ei = np.array([np.concatenate([np.full(row.k, new), nb]), np.concatenate([nb, np.full(row.k, new)])])
                    add_ew = np.concatenate([s[nb], s[nb]]).astype(np.float32)
                    x = torch.cat([X_all, torch.tensor(scaled_row)])
                    with torch.no_grad():
                        out = net(x, torch.tensor(np.concatenate([ei, add_ei], 1), dtype=torch.long),
                                  torch.tensor(np.concatenate([ew, add_ew])) if ew.size else None)
                    pld = float(out[new].squeeze()) * ck["target_scaler"]["std"] + ck["target_scaler"]["mean"]
            half = float(row.interval_halfwidth_90)
            result.predictions.append(Prediction(row.model_id, row.family, row.model, row.graph, int(row.k), row.pruning, round(pld, 3),
                                                 round(max(0.0, pld - half), 3), round(pld + half, 3), round(float(row.test_r2), 4),
                                                 round(float(row.test_mae), 3)))
        result.predictions.sort(key=lambda p: -p.test_r2)
        if live_used:
            notes.append("This linker is not in the data: its language-model embedding was computed live.")
        nn_graph = graph_type if graph_type in sims else "fingerprint"
        s = sims.get(nn_graph)
        if s is None:
            s, _ = self.similarity(smiles, "fingerprint")
        for i in np.argsort(-s)[:n_nearest]:
            r = self.mofs.iloc[i]
            result.nearest_mofs.append({"refcode": r.refcode, "linker_smiles": r.linker_smiles, "metal": r.metal, "pld": round(float(r.pld), 2),
                                        "similarity": round(float(s[i]), 3), "by": nn_graph})
        return result
