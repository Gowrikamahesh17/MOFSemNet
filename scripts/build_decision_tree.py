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
XLEAF = None
DEFAULT_CSV = ROOT / "outputs/results/server_molt5_2026-11-20/aggregated_results.csv"  # superset of the earlier server runs
EXTRA_CSVS = [ROOT / "outputs/results/server_extra_final_2026-11-20/aggregated_results.csv",   # GNN training fix, richer features, ensembles
              ROOT / "outputs/results/server_extra_2026-11-20/aggregated_results.csv"]         # boosted trees, no LCD/LFS, descriptors, new graphs (old GNN training)
OUT = ROOT / "frontend/public/decision_tree_data.json"

MODEL_LABEL = {"random_forest": "Random Forest", "mlp": "MLP", "knn": "k-NN", "gcn": "GCN",
               "graphsage": "GraphSAGE", "residual_gcn": "Residual GCN", "xgboost": "XGBoost", "lightgbm": "LightGBM",
               "xgboost_tuned": "XGBoost (tuned on validation)", "lightgbm_tuned": "LightGBM (tuned on validation)",
               "xgboost_smooth": "XGBoost + graph-smoothed features", "ensemble_avg": "Ensemble RF+XGBoost+GNN (equal weights)",
               "ensemble_val": "Ensemble RF+XGBoost+GNN (validation weights)"}
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


FEATURE_NAME = {"fp+pore+m1h": "Standard set: fingerprint + pore columns + metal one-hot",
                "fp+pore+m1h+ldesc": "+ RDKit linker descriptors", "fp+pore+m1h+mdesc": "+ metal descriptors",
                "fp+pore+mdesc": "Metal descriptors instead of the one-hot", "fp+pore+m1h+mdesc+ldesc": "+ metal and linker descriptors",
                "fpc+pore+m1h+mdesc+ldesc": "Count fingerprint instead of bits (+ both descriptors)",
                "fp+maccs+pore+m1h+mdesc+ldesc": "+ MACCS keys (+ both descriptors)",
                "fp+pore+m1h+mdesc+rdall": "All RDKit descriptors (+ metal descriptors)",
                "fpc+maccs+pore+m1h+mdesc+rdall": "Everything: counts, MACCS, all RDKit descriptors",
                "fp+m1h": "Standard set without LCD / LFS",
                "fp+m1h+ldesc": "+ RDKit linker descriptors", "fp+m1h+mdesc": "+ metal descriptors", "fp+mdesc": "Metal descriptors instead of the one-hot",
                "fp+m1h+mdesc+ldesc": "+ metal and linker descriptors", "fpc+m1h+mdesc+ldesc": "Count fingerprint (+ both descriptors)",
                "fp+maccs+m1h+mdesc+ldesc": "+ MACCS keys (+ both descriptors)", "fp+m1h+mdesc+rdall": "All RDKit descriptors (+ metal descriptors)",
                "fpc+maccs+m1h+mdesc+rdall": "Everything: counts, MACCS, all RDKit descriptors"}
WITH_PORE = ["fp+pore+m1h", "fp+pore+m1h+ldesc", "fp+pore+m1h+mdesc", "fp+pore+mdesc", "fp+pore+m1h+mdesc+ldesc", "fpc+pore+m1h+mdesc+ldesc",
             "fp+maccs+pore+m1h+mdesc+ldesc", "fp+pore+m1h+mdesc+rdall", "fpc+maccs+pore+m1h+mdesc+rdall"]
WITHOUT_PORE = ["fp+m1h", "fp+m1h+ldesc", "fp+m1h+mdesc", "fp+mdesc", "fp+m1h+mdesc+ldesc", "fpc+m1h+mdesc+ldesc",
                "fp+maccs+m1h+mdesc+ldesc", "fp+m1h+mdesc+rdall", "fpc+maccs+m1h+mdesc+rdall"]
EXTRA_GRAPH_LABEL = {("none", 0, "none", 1.0): "No edges", ("cs2", 3, "none", 1.0): "CS2 graph, k=3 (full)",
                     ("fingerprint", 3, "none", 1.0): "Fingerprint graph, k=3 (full)", ("cs2", 3, "black_hole", 0.3): "CS2 graph, k=3, Black Hole 30%",
                     ("dedup_cs2", 3, "none", 1.0): "De-duplicated CS2 graph (one MOF per neighbouring linker), k=3",
                     ("desc", 3, "none", 1.0): "Descriptor-space graph (RDKit + metal descriptors), k=3",
                     ("desc_dedup", 3, "none", 1.0): "De-duplicated descriptor graph, k=3"}


def load_extra(paths):
    frames = []
    for pth in paths:
        if Path(pth).exists():
            f = pd.read_csv(pth)
            for col, default in (("target", "raw"), ("train_setup", "v1")):
                f[col] = f[col].fillna(default) if col in f else default
            f["sparsify"] = f["sparsify"].fillna("none")
            f["retention"] = f["retention"].astype(float).round(2)
            frames.append(f)
    if not frames:
        return None
    keys = ["block", "model", "node_features", "graph_type", "k", "sparsify", "retention", "test_set", "target", "train_setup"]
    return pd.concat(frames).drop_duplicates(subset=keys, keep="first")  # earlier path wins


def make_helpers(X):
    def xres(**kw):
        d = X
        for key, val in kw.items():
            d = d[d[key] == val]
        out = {r.test_set: [round(r.r2_mean, 4), round(r.r2_std, 4), round(r.mae_mean, 3)] for r in d.itertuples()}
        return out or None

    def xleaf(name, note="", **kw):
        res = xres(**kw)
        d = {"name": name, "res": res} if res else {"name": name, "status": "untried"}
        if note:
            d["note"] = note
        return d

    def gnn_leaves(setup, target, feats, graph):
        gt, k, sp, ret = graph
        return [xleaf(MODEL_LABEL[m], block="extra_gnn", model=m, node_features=feats, graph_type=gt, k=k, sparsify=sp, retention=ret,
                      target=target, train_setup=setup) for m in ("gcn", "graphsage", "residual_gcn")]
    return xleaf, gnn_leaves


def extra_branches(X):
    """Branches for the extra experiments: boosted trees / richer features / ensembles, no LCD-LFS, GNN training fix and log target,
    other graphs and stacking. None if the extra result files are missing."""
    if X is None:
        return []
    xleaf, gnn_leaves = make_helpers(X)
    tab = lambda feats, models: [xleaf(MODEL_LABEL[m], block="extra_tab", model=m, node_features=feats, target="raw", train_setup="v1") for m in models]
    ens = lambda feats: [xleaf(MODEL_LABEL[m], block="extra_ens", model=m, node_features=feats, target="raw", train_setup="v2") for m in ("ensemble_avg", "ensemble_val")]

    def feature_branch(name, sets, note):
        kids = []
        for f in sets:
            models = ["random_forest", "mlp", "xgboost", "lightgbm", "xgboost_tuned", "lightgbm_tuned"]
            leaves = [l for l in tab(f, models) if l.get("res")] or tab(f, ["random_forest", "xgboost", "lightgbm"])
            leaves += [l for l in ens(f) if l.get("res")]
            kids.append({"name": FEATURE_NAME[f], "q": "model", "children": leaves})
        return {"name": name, "q": "input features", "children": kids, "note": note}

    out = [feature_branch("Boosted trees, richer features, ensembles", WITH_PORE,
                          "Non-graph models on richer inputs (target: raw PLD; seeds 0-4)")]
    nop = feature_branch("Without the pore columns (LCD, LFS removed)", WITHOUT_PORE, "R² falls to about 0.6; does the graph then help?")
    # GNNs without the pore columns (training fix)
    graphs = [("none", 0, "none", 1.0), ("cs2", 3, "none", 1.0), ("fingerprint", 3, "none", 1.0), ("cs2", 3, "black_hole", 0.3)]
    nop["children"].append({"name": "GNNs (training fix), plain features without LCD / LFS", "q": "graph",
                            "children": [{"name": EXTRA_GRAPH_LABEL[g], "q": "GNN model", "children": gnn_leaves("v2", "raw", "fp+m1h", g)} for g in graphs],
                            "note": "No graph variant beats the no-edge GNN"})
    out.append(nop)
    # training fix and log target
    fix = []
    for target, tlabel in (("raw", "Raw PLD target"), ("log1p", "log(1 + PLD) target (scored on the Å scale)")):
        feat_kids = []
        for feats, flabel in (("fp+pore+m1h", "With LCD / LFS"), ("fp+m1h", "Without LCD / LFS")):
            setups = []
            for setup, slabel in (("v2", "Training fix (standardised target + inputs)"), ("v1", "Old training (before the fix)")):
                setups.append({"name": slabel, "q": "graph", "children": [
                    {"name": EXTRA_GRAPH_LABEL[g], "q": "GNN model", "children": gnn_leaves(setup, target, feats, g)} for g in graphs]})
            feat_kids.append({"name": flabel, "q": "GNN training", "children": setups})
        fix.append({"name": tlabel, "q": "input columns", "children": feat_kids})
    out.append({"name": "GNN training fix and log target", "q": "target", "children": fix,
                "note": "Old GNN runs never converged (target scale); the fix lifts the no-edge GNN from 0.84 to 0.874"})
    # other graphs and stacking
    other = []
    for feats, flabel in (("fp+pore+m1h+ldesc", "With LCD / LFS (+ RDKit linker descriptors)"), ("fp+m1h+mdesc+ldesc", "Without LCD / LFS (+ both descriptor sets)")):
        gkids = [{"name": EXTRA_GRAPH_LABEL[g], "q": "GNN model", "children": gnn_leaves("v1", "raw", feats, g)}
                 for g in (("none", 0, "none", 1.0), ("cs2", 3, "none", 1.0), ("dedup_cs2", 3, "none", 1.0), ("desc", 3, "none", 1.0), ("desc_dedup", 3, "none", 1.0))]
        gkids.append({"name": "XGBoost + graph-smoothed neighbour features", "q": "graph", "children": [
            xleaf(lbl, block="extra_tab", model="xgboost_smooth", node_features=feats, graph_type=gt, k=3, sparsify="none", retention=1.0, target="raw", train_setup="v1")
            for gt, lbl in (("cs2", "Smoothed over the CS2 graph"), ("dedup_cs2", "Smoothed over the de-duplicated graph"))]})
        other.append({"name": flabel, "q": "graph", "children": gkids})
    other.append(untried("GNN predictions as XGBoost input", "needs out-of-fold predictions to avoid leakage"))
    out.append({"name": "Other graphs and stacking (old GNN training)", "q": "input columns", "children": other,
                "note": "De-duplicated and descriptor-space graphs; label-free graph-smoothed features"})
    return out


def untried(name, note=""):
    d = {"name": name, "status": "untried"}
    if note:
        d["note"] = note
    return d


def step6_extension(results_dir: Path):
    """Steps 6b-6d: SHAP for other models, GNN explainers, LIME vs SHAP, embedding dimensions, fragments."""
    def rd(name):
        f = results_dir / name
        return pd.read_csv(f) if f.exists() else None

    short = {"pore geometry": "pore", "linker fingerprint": "linker", "metal": "metal", "language-model embedding": "embedding"}
    out = []
    b = rd("step6b_method_groups.csv")
    if b is not None and "share_net" in b:
        kids = []
        for (method, model), d in b[b.method.str.startswith("SHAP")].groupby(["method", "model"], sort=False):
            net = " · ".join(f"{short[r.group]} {r.share_net:.0%}" for r in d.sort_values("share_net", ascending=False).itertuples())
            summed = " · ".join(f"{short[r.group]} {r.share:.0%}" for r in d.sort_values("share", ascending=False).itertuples())
            kids.append({"name": model, "txt": {"any": net}, "note": f"Net contribution per MOF. As a sum of per-column importances: {summed}"})
        out.append({"name": "SHAP for other models (Step 6b)", "q": "model", "children": kids,
                    "note": "Every model rests mostly on the pore columns once groups are counted by their net contribution"})
    c, nd = rd("step6c_gnn_graph_explainers_groups.csv"), rd("step6c_gnn_graph_shap_nodes.csv")
    if c is not None and nd is not None:
        gkids = []
        for graph in nd["graph"].drop_duplicates():
            mk = []
            for model in nd["model"].drop_duplicates():
                sub = nd[(nd.graph == graph) & (nd.model == model)]
                if sub.empty:
                    continue
                share_nb = sub.groupby("seed")["share_on_neighbours"].mean()
                shap_net = c[(c.graph == graph) & (c.model == model) & c.method.str.startswith("SHAP")].groupby("group")["share_net"].mean()
                own = " · ".join(f"{short[g]} {v:.0%}" for g, v in shap_net.sort_values(ascending=False).items())
                mk.append({"name": model, "txt": {"random15": f"R² {sub.groupby('seed').test_r2_of_model.mean().mean():.2f} · {share_nb.mean():.0%} of SHAP on neighbours (±{share_nb.std() * 100:.0f}) · own features: {own}"},
                           "note": "3 seeds, 30 test MOFs; PyG layers (GraphSAGE = SAGEConv)"})
            gkids.append({"name": graph, "q": "GNN model", "children": mk})
        out.append({"name": "GNNs with a graph: who do they listen to? (Step 6c)", "q": "graph", "children": gkids,
                    "note": "A plain GCN builds its answer from neighbours; pruning or a residual path reduces that"})
    lime_kids = []
    la, ga = rd("step6b_lime_vs_shap_agreement.csv"), rd("step6c_gnn_lime_vs_shap.csv")
    if la is not None:
        for model, d in la.groupby("model"):
            lime_kids.append({"name": f"LIME vs SHAP: {model}", "txt": {"random15": f"top-5 overlap {d.top5_overlap.mean():.0%} · local R² {d.lime_r2_local.mean():.2f}"},
                              "note": "100 MOFs, 15 features per explanation"})
    if ga is not None:
        d = ga.dropna(subset=["top5_overlap"])
        lime_kids.append({"name": "LIME vs SHAP: GNNs (all graphs)", "txt": {"random15": f"top-5 overlap {d.top5_overlap.mean():.0%} · local R² {d.lime_local_r2.mean():.2f} · LIME names the pore column first in {d.lime_top_is_pore.astype(float).mean():.0%}, SHAP in {d.shap_top_is_pore.astype(float).mean():.0%}"}})
    if lime_kids:
        out.append({"name": "LIME (Steps 6b, 6c)", "q": "model", "children": lime_kids, "note": "Unreliable here: rely on SHAP"})
    d6 = rd("step6d_embedding_shap_summary.csv")
    if d6 is not None:
        out.append({"name": "SHAP on the 768 embedding dimensions (Step 6d)", "q": "language model", "children": [
            {"name": r.language_model, "txt": {"random15": f"embedding {r.embedding_share_sum_of_columns:.0%} summed / {r.embedding_share_net:.0%} net · top-15 dims {r.top_dims_share_of_embedding_importance:.0%} · 12 RDKit descriptors explain R² {r.r2_net_embedding_shap_from_descriptors_cv:.2f} · Spearman with true PLD {r.spearman_net_embedding_shap_vs_true_pld:.2f}"}}
            for r in d6.itertuples()], "note": "Single dimensions track ring counts / polarity, the group as a whole follows pore size"})
    fr = rd("step6_shap_substructures_chemberta.csv")
    if fr is not None:
        out.append({"name": "Fragments behind the top fingerprint bits", "q": "bit", "children": [
            {"name": f"#{r.rank} bit {r.bit}: {r.fragment_smiles}", "txt": {"random15": f"{r.direction}" + (f" ({r.mean_shap_when_present:+.2f} Å)" if abs(r.mean_shap_when_present) >= 0.1 else "") + f" · {r.share_of_total_shap:.1%} of reliance · {r.n_distinct_fragments} fragments share the bit"}}
            for r in fr.head(6).itertuples()], "note": "Hashed bits mapped to the most common fragment"})
    return out


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
        net = ""
        if "net_share" in d and d["net_share"].notna().all():
            net = " Net contribution per MOF: " + " · ".join(f"{short.get(r.group, r.group)} {r.net_share:.0%}" for r in d.sort_values("net_share", ascending=False).itertuples()) + "."
        shap_kids.append({"name": label, "txt": {"any": parts}, "note": "Share of the Random Forest's reliance, as a sum of per-column mean |SHAP|." + net})
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
    children.extend(step6_extension(results_dir))
    children.append(untried("Intervals for the GNN models"))
    children.append(untried("Explainers on the Case Study 2 hub test set", "Steps 6b-6d use the random test set"))
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

    X_extra = load_extra(EXTRA_CSVS)
    global XLEAF
    XLEAF = make_helpers(X_extra)[0] if X_extra is not None else None

    # ---- Branch A: no graph ---------------------------------------------------------------
    a_feats = []
    for f, lm, label in FEATURE_OPTIONS:
        kids = [leaf(MODEL_LABEL[m], ("non_graph", m, f, "none", 0, "none", 1.0), lm) for m in ("random_forest", "mlp", "knn")]
        if f == "fingerprint" and XLEAF is not None:
            kids += [XLEAF(MODEL_LABEL[m], block="extra_tab", model=m, node_features="fp+pore+m1h", target="raw", train_setup="v1") for m in ("xgboost", "lightgbm")]
            kids += [XLEAF("Ensemble (see Boosted trees, richer features, ensembles)", note="small gain at best")]
        else:
            kids += [untried("XGBoost / LightGBM on this input")]
        a_feats.append({"name": label, "q": "model", "children": kids})
    a_feats += [untried("Embeddings as input to boosted trees")]
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

    extra = extra_branches(X_extra)
    root = {"name": "MOFCSD dataset", "q": "does the model use a graph?",
            "note": "14,296 MOFs, target PLD in Å. 5 seeds (0-4).", "children": [branch_a, branch_b, branch_c] + extra}
    branch_d = step6_branch(ROOT / "outputs/results")
    if branch_d:
        root["children"].append(branch_d)

    settings = [
        {"group": "Test set", "items": [["Random 15% (2,144 MOFs)", "done"], ["Case Study 2 hubs (680 MOFs)", "done"]]},
        {"group": "Language model", "items": [["ChemBERTa", "done"], ["MolT5", "done"]]},
        {"group": "Seeds", "items": [["0-4 (5 seeds)", "done"], ["10 seeds", "todo"]]},
        {"group": "Feature scaling", "items": [["None (raw)", "done"], ["Standardised target + inputs (GNN training fix)", "done"]]},
        {"group": "GNN training", "items": [["hidden 64, lr 0.005, patience 20", "done"], ["Longer training / lower learning rate (no gain)", "done"], ["Tuned hidden size / depth", "todo"]]},
        {"group": "Target", "items": [["Raw PLD", "done"], ["log(1 + PLD)", "done"]]},
        {"group": "Pore columns", "items": [["With LCD / LFS", "done"], ["Without LCD / LFS", "done"]]},
        {"group": "Significance test", "items": [["Paired comparison vs seed spread", "done"], ["Formal test", "todo"]]},
    ]

    OUT.write_text(json.dumps({"tree": root, "settings": settings}))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV)
