#!/usr/bin/env python
"""Build docs/decision_tree.html: a standalone, interactive picture of every decision in the MOFSemNet
experiments (what was tried, what each option scored, what has not been tried yet).

Numbers come from the aggregated server results (test R2, mean of 5 seeds). Options that appear in the
tree but have no result rows are marked "not tried yet".

Usage:
    python scripts/build_decision_tree.py [path/to/aggregated_results.csv]
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "outputs/results/server_cs2hubs_2026-11-19/aggregated_results.csv"
OUT = ROOT / "docs/decision_tree.html"

MODEL_LABEL = {"random_forest": "Random Forest", "mlp": "MLP", "knn": "k-NN", "gcn": "GCN",
               "graphsage": "GraphSAGE", "residual_gcn": "Residual GCN"}
LM_NAME = {"-": "", "chemberta": "ChemBERTa", "molt5": "MolT5"}
# (feature set, language model, label)
FEATURE_OPTIONS = [("fingerprint", "-", "Fingerprint only"),
                   ("embedding", "chemberta", "ChemBERTa embedding only"),
                   ("fingerprint+embedding", "chemberta", "Fingerprint + ChemBERTa"),
                   ("embedding", "molt5", "MolT5 embedding only"),
                   ("fingerprint+embedding", "molt5", "Fingerprint + MolT5")]
# (graph type, language model, label)
GRAPH_OPTIONS = [("cs2", "-", "Case Study 2 graph"), ("fingerprint", "-", "Fingerprint graph"),
                 ("semantic", "chemberta", "Semantic graph (ChemBERTa)"), ("hybrid", "chemberta", "Hybrid graph (ChemBERTa)"),
                 ("semantic", "molt5", "Semantic graph (MolT5)"), ("hybrid", "molt5", "Hybrid graph (MolT5)")]
GRAPH_NOTE = {"cs2": "0.9 x Tanimoto + 0.1 x same-metal", "fingerprint": "linker Tanimoto only",
              "semantic": "cosine of ChemBERTa embeddings", "hybrid": "0.5 x cosine + 0.5 x Tanimoto"}
PRUNE_LABEL = {"none": "No pruning (full graph)", "black_hole": "Black Hole", "random": "Random sampling",
               "k_center": "k-center sampling"}


def main(csv_path: Path) -> None:
    a = pd.read_csv(csv_path)
    a["sparsify"] = a["sparsify"].fillna("none")
    res = {}
    for r in a.itertuples():
        key = (r.block, r.model, r.node_features, r.graph_type, int(r.k), r.sparsify, round(float(r.retention), 2),
               r.language_model)
        res.setdefault(key, {})[r.test_set] = [round(r.r2_mean, 4), round(r.r2_std, 4), round(r.mae_mean, 3)]

    def leaf(label, key, lm="-", note=""):
        key = tuple(key) + (lm,)
        d = {"name": label}
        if key in res:
            d["res"] = res[key]
        else:
            d["status"] = "untried"
        if note:
            d["note"] = note
        return d

    def untried(name, note=""):
        d = {"name": name, "status": "untried"}
        if note:
            d["note"] = note
        return d

    # ---- Branch A: no graph ---------------------------------------------------------------
    a_feats = []
    for f, lm, label in FEATURE_OPTIONS:
        kids = [leaf(MODEL_LABEL[m], ("non_graph", m, f, "none", 0, "none", 1.0), lm) for m in ("random_forest", "mlp", "knn")]
        kids += [untried("XGBoost / LightGBM", "boosted trees: most likely accuracy gain"),
                 untried("Ensemble (RF + MLP + boosted)")]
        a_feats.append({"name": label, "q": "model", "children": kids})
    a_feats += [untried("Count fingerprints / RDKit descriptors", "richer linker features")]
    branch_a = {"name": "No graph (tabular models)", "q": "input features", "children": a_feats,
                "note": "Random Forest, MLP, k-NN on one row per MOF"}

    # ---- Branch B: GNN with no edges --------------------------------------------------------
    b_feats = []
    for f, lm, label in FEATURE_OPTIONS:
        kids = [leaf(MODEL_LABEL[m], ("no_edge", m, f, "none", 0, "none", 1.0), lm) for m in ("gcn", "graphsage", "residual_gcn")]
        kids.append(untried("GAT"))
        b_feats.append({"name": label, "q": "GNN model", "children": kids})
    branch_b = {"name": "GNN with NO edges", "q": "input features", "children": b_feats,
                "note": "Reference row: what the graph adds or removes"}

    # ---- Branch C: GNN with a graph -----------------------------------------------------------
    def models_for(block, g, k, sp, ret, lm):
        kids = [leaf(MODEL_LABEL[m], (block, m, "fingerprint", g, k, sp, ret), lm) for m in ("gcn", "graphsage", "residual_gcn")]
        kids.append(untried("GAT"))
        return kids

    graph_nodes = []
    for g, glm, glabel in GRAPH_OPTIONS:
        k_nodes = []
        for k in (3, 5, 10):
            prune_nodes = [{"name": PRUNE_LABEL["none"], "q": "GNN model",
                            "children": models_for("graph_full", g, k, "none", 1.0, glm)}]
            if k in (3, 10):
                for sp in ("black_hole", "random", "k_center"):
                    rets = []
                    for ret in (0.3, 0.5):
                        rets.append({"name": f"Keep {int(ret * 100)}% of MOFs", "q": "GNN model",
                                     "children": models_for("sparsify", g, k, sp, ret, glm)})
                    rets.append(untried("Keep 70% of MOFs"))
                    node = {"name": PRUNE_LABEL[sp], "q": "retention", "children": rets}
                    if sp == "black_hole":
                        node["children"].append(untried("Other gravity weights", "only equal weights 0.33/0.33/0.33 tried"))
                    prune_nodes.append(node)
            else:
                for sp in ("black_hole", "random", "k_center"):
                    prune_nodes.append(untried(PRUNE_LABEL[sp], "pruning only run at k=3 and k=10"))
            k_nodes.append({"name": f"k = {k} neighbours", "q": "pruning", "children": prune_nodes})
        k_nodes.append(untried("k = 2 or k = 20", "outside the tested range"))
        graph_nodes.append({"name": glabel, "q": "number of neighbours k", "note": GRAPH_NOTE[g],
                            "children": k_nodes})
    graph_nodes += [untried("Hybrid with other blend (0.25, 0.75)"),
                    untried("Linker-level molecular GNN", "atoms as nodes, not MOFs")]
    branch_c = {"name": "GNN with a graph", "q": "graph type", "children": graph_nodes,
                "note": "Edges join similar MOFs"}

    root = {"name": "MOFCSD dataset", "q": "does the model use a graph?",
            "note": "14,296 MOFs, target PLD in Å. 5 seeds (0-4).", "children": [branch_a, branch_b, branch_c]}

    settings = [
        {"group": "Test set", "items": [["Random 15% (2,144 MOFs)", "done"], ["Case Study 2 hubs (680 MOFs)", "done"]]},
        {"group": "Language model", "items": [["ChemBERTa", "done"], ["MolT5", "done"]]},
        {"group": "Seeds", "items": [["0-4 (5 seeds)", "done"], ["10 seeds", "todo"]]},
        {"group": "Feature scaling", "items": [["None (raw)", "done"], ["Standardised", "todo"]]},
        {"group": "GNN training", "items": [["hidden 64, lr 0.005, patience 20", "done"], ["Tuned values", "todo"]]},
        {"group": "Significance test", "items": [["Paired comparison vs seed spread", "done"], ["Formal test", "todo"]]},
    ]

    template = (ROOT / "scripts/decision_tree_template.html").read_text()
    html = template.replace("__TREE__", json.dumps(root)).replace("__SETTINGS__", json.dumps(settings))
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html) // 1024} KB)")
    public = ROOT / "frontend/public"
    if public.exists():  # the web console serves it at /decision_tree.html
        (public / "decision_tree.html").write_text(html)
        print(f"copied to {public / 'decision_tree.html'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV)
