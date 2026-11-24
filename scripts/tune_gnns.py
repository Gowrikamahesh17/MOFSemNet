#!/usr/bin/env python
"""Tune the GNNs the same way the tree models were tuned, so the graph-vs-no-graph comparison is fair (original data and target).

Models: GCN, GraphSAGE (PyG SAGEConv), residual GCN, and four more standard architectures (GAT, GIN, APPNP, SGC).
Graphs: no edges, the CS2 graph (k=3), and the CS2 graph pruned by Black Hole to 30% retention.
Training: the "v2" setup that fixed the under-trained GNNs (z-scored target and continuous inputs; statistics from non-test /
training MOFs), Adam, early stopping on the validation split.

Protocol
  Stage A  random search of architecture / optimiser settings (hidden size, depth, dropout, learning rate, weight decay, model-specific
           options), scored on the VALIDATION split only (random-test-set splits, seeds 0-1). Settings are chosen on "no edges" for the
           no-edge rows and on the CS2 full graph for the graph rows (the Black Hole graph reuses the CS2 choice). Test sets are never used.
  Stage B  chosen settings and the old default settings (hidden 64, 2 layers, dropout 0.3, lr 0.005, weight decay 1e-3) are trained on the
           training split and tested on the random 15% test set and the 664-MOF Case Study 2 hub set, seeds 0-4.

Usage:  PYTHONPATH=. python scripts/tune_gnns.py [--quick]
Writes  outputs/results/gnn_tuning_validation_search.csv, outputs/results/gnn_tuning_per_seed.csv
"""

import argparse
import hashlib
import itertools
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.nn import APPNP, GATConv, GCNConv, GINConv, SAGEConv, SGConv

from src.baseline.data_ingestion import load_dataset
from src.baseline.feature_engineering import build_features
from src.baseline.gnn_training import graph_to_edge_arrays, make_splits
from src.common.config import RuntimeConfig
from src.evaluation.run_comparison import compute_global_fixed_test_nodes, restore_test_nodes

SCRIPT = "scripts/tune_gnns.py"
DEFAULT = {"hidden": 64, "layers": 2, "dropout": 0.3, "lr": 0.005, "weight_decay": 1e-3}
COMMON = {"hidden": (64, 128, 256), "layers": (2, 3), "dropout": (0.1, 0.3, 0.5), "lr": (0.002, 0.005, 0.01), "weight_decay": (1e-4, 1e-3, 1e-2)}
SPACES = {
    "gcn": COMMON, "graphsage": COMMON, "gin": COMMON,
    "residual_gcn": {**COMMON, "layers": (1, 2)},            # number of gated GCN blocks
    "gat": {**COMMON, "heads": (2, 4)},
    "appnp": {"hidden": (64, 128, 256), "dropout": (0.1, 0.3, 0.5), "lr": (0.002, 0.005, 0.01), "weight_decay": (1e-4, 1e-3, 1e-2), "K": (5, 10), "alpha": (0.1, 0.2)},
    "sgc": {"K": (1, 2, 3), "dropout": (0.0, 0.1, 0.3), "lr": (0.005, 0.01, 0.02), "weight_decay": (1e-4, 1e-3, 1e-2)},
}
BASE_MODELS = ["gcn", "graphsage", "residual_gcn"]
EXTRA_MODELS = ["gat", "gin", "appnp", "sgc"]


class Net(nn.Module):
    def __init__(self, kind, d_in, p):
        super().__init__()
        self.kind, self.p = kind, p
        h, L, drop = p.get("hidden", 64), p.get("layers", 2), p.get("dropout", 0.3)
        self.drop = drop
        if kind == "gcn":
            self.convs = nn.ModuleList([GCNConv(d_in if i == 0 else h, 1 if i == L - 1 else h) for i in range(L)])
        elif kind == "graphsage":
            self.convs = nn.ModuleList([SAGEConv(d_in if i == 0 else h, 1 if i == L - 1 else h) for i in range(L)])
        elif kind == "gin":
            self.convs = nn.ModuleList([GINConv(nn.Sequential(nn.Linear(d_in if i == 0 else h, h), nn.ReLU(), nn.Linear(h, h))) for i in range(L - 1)])
            self.out = nn.Linear(h, 1)
        elif kind == "gat":
            heads = p["heads"]
            self.convs = nn.ModuleList([GATConv(d_in if i == 0 else h, h // heads, heads=heads, dropout=drop / 2) for i in range(L - 1)]
                                       + [GATConv(h, 1, heads=1, concat=False, dropout=drop / 2)])
        elif kind == "residual_gcn":
            self.lin_in, self.lin_out = nn.Linear(d_in, h), nn.Linear(h, 1)
            self.blocks = nn.ModuleList([GCNConv(h, h) for _ in range(L)])
            self.gates = nn.Parameter(torch.zeros(L))
        elif kind == "appnp":
            self.mlp = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 1))
            self.prop = APPNP(K=p["K"], alpha=p["alpha"])
        elif kind == "sgc":
            self.sgc = SGConv(d_in, 1, K=p["K"])
        else:
            raise ValueError(kind)

    def forward(self, x, ei, ew):
        k, drop = self.kind, self.drop
        if k in ("gcn", "graphsage", "gat"):
            for conv in self.convs[:-1]:
                x = F.dropout(F.relu(conv(x, ei, ew) if k == "gcn" else conv(x, ei)), p=drop, training=self.training)
            return self.convs[-1](x, ei, ew) if k == "gcn" else self.convs[-1](x, ei)
        if k == "gin":
            for conv in self.convs:
                x = F.dropout(F.relu(conv(x, ei)), p=drop, training=self.training)
            return self.out(x)
        if k == "residual_gcn":
            h = F.dropout(F.relu(self.lin_in(x)), p=drop, training=self.training)
            for gate, conv in zip(self.gates, self.blocks):
                h = h + gate * F.relu(conv(h, ei, ew))
            return self.lin_out(h)
        if k == "appnp":
            return self.prop(self.mlp(x), ei, ew)
        return self.sgc(F.dropout(x, p=drop, training=self.training), ei, ew)


def standardise(X, non_test):
    nb = ~((X == 0) | (X == 1)).all(axis=0)
    mu, sd = X[non_test][:, nb].mean(0), X[non_test][:, nb].std(0)
    out = X.copy()
    out[:, nb] = (out[:, nb] - mu) / np.where(sd == 0, 1.0, sd)
    return out.astype(np.float32)


def train_eval(kind, p, Xs, y, graph, tr, va, te, seed, epochs=250, patience=25):
    """Trains on `tr` (early stopping on `va`), returns (validation R2, test predictions or None, epochs run)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    ei_np, ew_np = graph
    ei = torch.tensor(ei_np, dtype=torch.long)
    ew = torch.tensor(ew_np, dtype=torch.float) if ew_np.size else None
    x = torch.tensor(Xs)
    ym, ys = float(y[tr].mean()), float(y[tr].std())
    yt = torch.tensor((y - ym) / ys, dtype=torch.float)
    trt, vat = torch.tensor(tr), torch.tensor(va)
    model = Net(kind, Xs.shape[1], p)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    best, state, wait, ran = float("inf"), None, 0, 0
    for ran in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        F.mse_loss(model(x, ei, ew).squeeze(-1)[trt], yt[trt]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(x, ei, ew).squeeze(-1)[vat], yt[vat]).item()
        if vl < best:
            best, wait, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        full = model(x, ei, ew).squeeze(-1).numpy() * ys + ym
    return float(r2_score(y[va], full[va])), (None if te is None else full[te]), ran


def sample(space, n, seed=0):
    combos = [dict(zip(space, v)) for v in itertools.product(*space.values())]
    rng = np.random.default_rng(seed)
    return [combos[i] for i in rng.choice(len(combos), size=min(n, len(combos)), replace=False)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 config, 1 seed, 1 test set")
    ap.add_argument("--n-configs", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    config = RuntimeConfig(language_model="chemberta")
    df = load_dataset("large")
    y = df["pld_value"].values.astype(np.float32)
    X, _ = build_features(df, feature_scheme="fingerprint")
    N = len(X)
    sha = hashlib.sha256(open(config.dataset_path, "rb").read()).hexdigest()[:16]
    tests = {"random15": set(compute_global_fixed_test_nodes(N, config)),
             "cs2_hubs (664, repo code)": set(pd.read_csv(os.path.join(config.results_dir, "cs2_hub_test_nodes.csv")).node_index.tolist())}
    if args.quick:
        tests = dict(list(tests.items())[:1])
    sel_seeds, test_seeds = ([0], [0]) if args.quick else ([0, 1], list(range(5)))
    n_cfg = 1 if args.quick else args.n_configs
    t0 = time.time()

    from src.baseline.black_hole_sparsification import apply_black_hole_sparsification
    from src.baseline.graph_construction import build_similarity_graphs
    cs2 = build_similarity_graphs(df, "large")["knn_3"]
    weights = (config.gravity_degree_weight, config.gravity_betweenness_weight, config.gravity_edge_weight_sum_weight)
    bh = apply_black_hole_sparsification(cs2, df, weights, threshold=0.7)["graph"]
    print(f"[{time.time() - t0:.0f}s] graphs built: CS2 {cs2.number_of_edges()} edges, Black Hole 30% {bh.number_of_edges()} edges", flush=True)
    empty = (np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32))
    cs2_arr = graph_to_edge_arrays(cs2, N)
    standardised = {name: standardise(X, np.array([i for i in range(N) if i not in test])) for name, test in tests.items()}

    # ------------------------------------------------ Stage A
    sel_test = tests["random15"]
    Xs_sel = standardised["random15"]
    chosen, rows = {}, []
    combos = [("none", m) for m in BASE_MODELS] + [("cs2", m) for m in BASE_MODELS + EXTRA_MODELS]
    if args.quick:
        combos = combos[:2]
    for gname, kind in combos:
        graph = empty if gname == "none" else cs2_arr
        best = None
        for p in sample(SPACES[kind], n_cfg):
            sc = []
            for seed in sel_seeds:
                tr, va, _ = make_splits(N, sel_test, seed=seed)
                sc.append(train_eval(kind, p, Xs_sel, y, graph, tr, va, None, seed)[0])
            rows.append({"graph": gname, "model": kind, "settings": str(p), "val_r2_mean": float(np.mean(sc)), "val_r2_std": float(np.std(sc)),
                         "script": SCRIPT, "data_sha256_16": sha})
            print(f"[{time.time() - t0:5.0f}s] validation {gname:4s} {kind:12s} {p}: R2 {np.mean(sc):.4f}", flush=True)
            if best is None or rows[-1]["val_r2_mean"] > best[0]:
                best = (rows[-1]["val_r2_mean"], p)
        chosen[(gname, kind)] = best[1]
        print(f"[{time.time() - t0:5.0f}s] CHOSEN {gname} {kind}: {best[1]} (validation R2 {best[0]:.4f})", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(config.results_dir, "gnn_tuning_validation_search.csv"), index=False)

    # ------------------------------------------------ Stage B
    out = []
    for tname, test in tests.items():
        te = np.array(sorted(test))
        Xs = standardised[tname]
        bh_graph = graph_to_edge_arrays(restore_test_nodes(bh, cs2, set(test)), N)
        graphs = {"none": empty, "cs2": cs2_arr, "cs2_blackhole30": bh_graph}
        for seed in test_seeds:
            tr, va, _ = make_splits(N, test, seed=seed)
            for (gname, kind), p in chosen.items():
                targets = [("cs2", cs2_arr), ("cs2_blackhole30", bh_graph)] if gname == "cs2" else [("none", empty)]
                if kind in EXTRA_MODELS:
                    targets = [("cs2", cs2_arr)]
                for gl, graph in targets:
                    variants = [("tuned", p)] + ([("default", {**DEFAULT, "layers": 1} if kind == "residual_gcn" else DEFAULT)] if kind in BASE_MODELS else [])
                    for vname, vp in variants:
                        v, pt, ran = train_eval(kind, vp, Xs, y, graph, tr, va, te, seed)
                        out.append({"test_set": tname, "graph": gl, "model": kind, "setting": vname, "settings": str(vp), "seed": seed, "val_r2": v,
                                    "r2": float(r2_score(y[te], pt)), "mae": float(mean_absolute_error(y[te], pt)),
                                    "rmse": float(np.sqrt(mean_squared_error(y[te], pt))), "epochs": ran, "script": SCRIPT, "data_sha256_16": sha})
                        print(f"[{time.time() - t0:5.0f}s] {tname} seed {seed} {gl:15s} {kind:12s} {vname:7s}: test R2 {out[-1]['r2']:.4f}", flush=True)
            pd.DataFrame(out).to_csv(os.path.join(config.results_dir, "gnn_tuning_per_seed.csv"), index=False)
    res = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(res.groupby(["test_set", "graph", "model", "setting"])[["r2", "mae", "rmse"]].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
