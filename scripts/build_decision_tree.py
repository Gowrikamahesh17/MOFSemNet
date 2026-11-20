#!/usr/bin/env python
"""Build the data for the web console's decision-tree page (frontend/public/decision_tree_data.json; the page itself is
frontend/public/decision_tree.html): an interactive picture of every decision in the MOFSemNet
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
OUT = ROOT / "frontend/public/decision_tree_data.json"

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


def untried(name, note=""):
    d = {"name": name, "status": "untried"}
    if note:
        d["note"] = note
    return d


def step6_branch(results_dir: Path):
    """Step 6 (trustworthiness) branch from the step6_*.csv result files; None if they are missing."""
    def rd(name):
        f = results_dir / name
        return pd.read_csv(f) if f.exists() else None

    cols, shap_c, shap_m = rd("step6_column_groups_chemberta.csv"), rd("step6_shap_groups_chemberta.csv"), rd("step6_shap_groups_molt5.csv")
    conf_r, conf_h = rd("step6_confidence_chemberta.csv"), rd("step6_confidence_chemberta_hubs.csv")
    abl = {"chemberta": rd("step6_embedding_ablation_chemberta.csv"), "molt5": rd("step6_embedding_ablation_molt5.csv")}
    if cols is None or shap_c is None or conf_r is None or any(v is None for v in abl.values()):
        return None

    def r2leaf(name, row):
        return {"name": name, "res": {"random15": [round(row.r2_mean, 4), round(row.r2_std, 4), round(row.mae_mean, 3)]}}

    children = []
    # 6a SHAP
    shap_kids = []
    for label, d in (("Fingerprint-only forest", shap_c[shap_c.features == "fingerprint"]),
                     ("Fingerprint + ChemBERTa", shap_c[shap_c.features == "fingerprint+embedding"]),
                     ("Fingerprint + MolT5", None if shap_m is None else shap_m[shap_m.features == "fingerprint+embedding"])):
        if d is None or d.empty:
            continue
        short = {"pore geometry": "pore", "language-model embedding": "embed", "linker fingerprint": "linker", "metal": "metal"}
        ranked = list(d.sort_values("share", ascending=False).itertuples())
        top, rest = ranked[:2 if len(ranked) > 3 else 3], ranked[2 if len(ranked) > 3 else 3:]
        parts = " · ".join(f"{short.get(r.group, r.group)} {r.share:.0%}" for r in top)
        if rest:
            parts += f" · other {sum(r.share for r in rest):.0%}"
        shap_kids.append({"name": label, "txt": {"any": parts}, "note": "Share of the Random Forest's reliance (mean |SHAP|)"})
    children.append({"name": "SHAP: what does the forest read?", "q": "model inputs", "children": shap_kids,
                     "note": "Explained forest uses min_samples_leaf=5 (TreeSHAP is unreliable on fully grown trees)"})
    # column groups
    children.append({"name": "Which columns carry the signal?", "q": "column set",
                     "children": [r2leaf(r.columns.replace(" (2 columns)", ""), r) for r in cols.itertuples()],
                     "note": "Random Forest on each column set alone (random test set)"})
    # embedding ablation
    variants = {"none": "Real embeddings", "shuffle_rows": "Shuffled between linkers", "random_replacement": "Random vectors",
                "truncate_dims": "First 64 dimensions", "no_embedding": "No embedding"}
    abl_kids = []
    for lm, label in (("chemberta", "ChemBERTa"), ("molt5", "MolT5")):
        a = abl[lm]
        graph_kids = []
        for key in ("none", "shuffle_rows", "random_replacement", "truncate_dims"):
            sub = a[(a.experiment == "semantic_graph") & (a.ablation == key)]
            graph_kids.append({"name": variants[key], "q": "GNN model",
                               "children": [r2leaf(MODEL_LABEL[r.model], r) for r in sub.itertuples()]})
        rf_kids = [r2leaf(variants[key], r) for key in ("no_embedding", "none", "truncate_dims", "shuffle_rows", "random_replacement")
                   for r in a[(a.experiment == "rf_input") & (a.ablation == key)].itertuples()]
        abl_kids.append({"name": label, "q": "where the embedding is used", "children": [
            {"name": "Semantic graph (k=3)", "q": "embedding variant", "children": graph_kids,
             "note": "Real embeddings beat random or shuffled ones, so they carry chemistry"},
            {"name": "Random Forest input", "q": "embedding variant", "children": rf_kids,
             "note": "Fingerprint + (degraded) embedding"}]})
    children.append({"name": "Embedding ablation", "q": "language model", "children": abl_kids,
                     "note": "Do embeddings carry usable chemistry? (random test set)"})
    # conformal intervals
    conf_kids = []
    for alpha in (0.1, 0.2):
        kids = []
        for mode, mlabel in (("absolute", "Fixed width"), ("normalized", "Adaptive width")):
            txt = {}
            for key, df_ in (("random15", conf_r), ("cs2_hubs", conf_h)):
                if df_ is None:
                    continue
                r = df_[(df_.alpha == alpha) & (df_.interval == mode)].iloc[0]
                txt[key] = f"covers {r.coverage_mean:.1%}, width {r.width_mean:.1f} Å"
            kids.append({"name": mlabel, "txt": txt, "note": f"Target coverage {1 - alpha:.0%}; Random Forest on fingerprints"})
        conf_kids.append({"name": f"{1 - alpha:.0%} target coverage", "q": "interval type", "children": kids})
    children.append({"name": "Confidence intervals (conformal)", "q": "coverage target", "children": conf_kids,
                     "note": "Split conformal prediction; calibration = validation split"})
    children.append(untried("Map top fingerprint bits to chemical substructures", "SHAP gives columns, not substructures"))
    children.append(untried("Intervals for the GNN models"))
    return {"name": "Trustworthiness checks (Step 6)", "q": "analysis", "children": children,
            "note": "SHAP, embedding ablation, calibrated confidence"}


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
    branch_d = step6_branch(ROOT / "outputs/results")
    if branch_d:
        root["children"].append(branch_d)

    settings = [
        {"group": "Test set", "items": [["Random 15% (2,144 MOFs)", "done"], ["Case Study 2 hubs (680 MOFs)", "done"]]},
        {"group": "Language model", "items": [["ChemBERTa", "done"], ["MolT5", "done"]]},
        {"group": "Seeds", "items": [["0-4 (5 seeds)", "done"], ["10 seeds", "todo"]]},
        {"group": "Feature scaling", "items": [["None (raw)", "done"], ["Standardised", "todo"]]},
        {"group": "GNN training", "items": [["hidden 64, lr 0.005, patience 20", "done"], ["Tuned values", "todo"]]},
        {"group": "Significance test", "items": [["Paired comparison vs seed spread", "done"], ["Formal test", "todo"]]},
    ]

    OUT.write_text(json.dumps({"tree": root, "settings": settings}))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV)
