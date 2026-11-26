#!/usr/bin/env python
"""Predict PLD for every MOF with every saved model (no retraining) and write the predictions to CSV.

Uses the models trained and saved by MOFSemNet_server_save_models.ipynb (unpacked into outputs/models/mofsemnet_models/).
The feature, graph and model code is taken from that same notebook (its setup cells are executed), so the inputs match exactly.

Writes
  outputs/results/predictions_all_models.csv.gz   one row per MOF: split, true PLD and the prediction of all 286 models
  results/predictions/predictions_summary.csv     one row per MOF: true PLD and the headline models' predictions with 90% intervals
                                                  (headline models chosen by VALIDATION R2, never test)
  results/predictions/headline_models.csv         which models those are, with their saved test scores

Usage:  PYTHONPATH=. python scripts/export_model_predictions.py [--models-dir outputs/models/mofsemnet_models]
"""

import argparse
import json
import os
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_notebook_setup(notebook, data_path, emb_dir, scratch):
    """Executes the notebook's setup cells (imports, configuration, data, split, graph code, models) and returns their namespace."""
    nb = json.load(open(notebook))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"][:6]
    code = "\n\n".join(cells)
    code = "\n".join(l for l in code.splitlines() if not l.lstrip().startswith("!"))
    code = code.replace('DATA_PATH = "MOFCSD.csv"', f'DATA_PATH = {data_path!r}', 1)
    code = code.replace('EMB_PATHS = {"chemberta": "chemberta_embeddings.npz", "molt5": "molt5_embeddings.npz"}',
                        f'EMB_PATHS = {{"chemberta": {os.path.join(emb_dir, "chemberta_embeddings.npz")!r}, '
                        f'"molt5": {os.path.join(emb_dir, "molt5_embeddings.npz")!r}}}', 1)
    code = code.replace('OUT_DIR = "mofsemnet_models"', f'OUT_DIR = {scratch!r}', 1)
    ns = {"__name__": "nbsetup"}
    exec(compile(code, notebook, "exec"), ns)
    return ns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", default=os.path.join(ROOT, "outputs/models/mofsemnet_models"))
    ap.add_argument("--notebook", default=os.path.join(ROOT, "MOFSemNet_server_save_models.ipynb"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data/raw/MOFCSD.csv"))
    ap.add_argument("--emb-dir", default=os.path.join(ROOT, "data/embeddings"))
    args = ap.parse_args()
    t0 = time.time()

    with tempfile.TemporaryDirectory() as scratch:
        ns = load_notebook_setup(args.notebook, args.data, args.emb_dir, scratch)
    np, pd, torch, joblib = ns["np"], ns["pd"], ns["torch"], ns["joblib"]
    Net, r2_score = ns["Net"], ns["r2_score"]
    N, LABELS, df = ns["N"], ns["LABELS"], ns["df"]
    FEATURES, XS = ns["FEATURES"], ns["XS"]
    TRAIN, VAL, TEST = ns["TRAIN"], ns["VAL"], ns["TEST"]
    M = args.models_dir
    res = pd.read_csv(os.path.join(M, "results/model_results.csv"))
    saved_split = json.load(open(os.path.join(M, "features/split.json")))
    assert sorted(saved_split["test"]) == sorted(TEST.tolist()), "test split differs from the saved one"
    assert ns["DATA_SHA"][:16] == res["data_sha256"].iloc[0], "data file differs from the one the models were trained on"

    split = np.array(["train"] * N, dtype=object)
    split[VAL], split[TEST] = "validation", "test"
    out = pd.DataFrame({"refcode": df["refcode"].values, "linker_smiles": df["linker_smiles"].values, "metal": df["metal"].values,
                        "split": split, "true_pld": LABELS})
    graph_cache = {}
    max_gap = 0.0
    for r in res.itertuples():
        f = os.path.join(M, r.file)
        if r.family == "tree":
            X = FEATURES[r.mode]
            if f.endswith(".joblib"):
                pred = joblib.load(f).predict(X)
            elif f.endswith(".json"):
                import xgboost as xgb
                m = xgb.XGBRegressor(); m.load_model(f); pred = m.predict(X)
            elif f.endswith(".txt"):
                import lightgbm as lgb
                pred = lgb.Booster(model_file=f).predict(X)
            else:
                from catboost import CatBoostRegressor
                m = CatBoostRegressor(); m.load_model(f); pred = m.predict(X)
        else:
            ck = torch.load(f, map_location="cpu")
            net = Net(ck["model"], ck["d_in"], ck["settings"]); net.load_state_dict(ck["state_dict"]); net.eval()
            if ck["graph_file"]:
                if ck["graph_file"] not in graph_cache:
                    z = np.load(os.path.join(M, ck["graph_file"]))
                    graph_cache[ck["graph_file"]] = (torch.tensor(z["edge_index"], dtype=torch.long),
                                                     torch.tensor(z["edge_weight"]) if z["edge_weight"].size else None)
                ei, ew = graph_cache[ck["graph_file"]]
            else:
                ei, ew = torch.zeros((2, 0), dtype=torch.long), None
            with torch.no_grad():
                pred = net(torch.tensor(XS[ck["mode"]]), ei, ew).squeeze(-1).numpy() * ck["target_scaler"]["std"] + ck["target_scaler"]["mean"]
        out[r.model_id] = np.asarray(pred, dtype=np.float32)
        max_gap = max(max_gap, abs(r2_score(LABELS[TEST], pred[TEST]) - r.test_r2))
    print(f"predicted with {len(res)} models; largest difference to the saved test R2: {max_gap:.1e}")

    os.makedirs(os.path.join(ROOT, "outputs/results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results/predictions"), exist_ok=True)
    out.round(4).to_csv(os.path.join(ROOT, "outputs/results/predictions_all_models.csv.gz"), index=False, compression="gzip")

    # headline models, chosen by validation R2 (never test)
    picks = []
    for mode in ("with_pore", "chemistry_only"):
        r_ = res[res["mode"] == mode]
        for label, sel in (("best tree model", r_.family == "tree"), ("best GNN, no edges", (r_.family == "gnn") & (r_.graph == "none")),
                           ("best GNN with a graph", (r_.family == "gnn") & (r_.graph != "none"))):
            best = r_[sel].sort_values("val_r2", ascending=False).iloc[0]
            picks.append({"mode": mode, "role": label, **best[["model_id", "model", "graph", "k", "pruning", "val_r2", "test_r2", "test_mae", "test_rmse",
                                                                 "interval_halfwidth_90"]].to_dict()})
    picks = pd.DataFrame(picks)
    picks.to_csv(os.path.join(ROOT, "results/predictions/headline_models.csv"), index=False)
    summ = out[["refcode", "linker_smiles", "metal", "split", "true_pld"]].copy()
    for p in picks.itertuples():
        col = f"{p.mode}__{p.role.replace(' ', '_').replace(',', '')}"
        summ[col + "__pred"] = out[p.model_id].round(3)
        summ[col + "__low90"] = (out[p.model_id] - p.interval_halfwidth_90).clip(lower=0).round(3)
        summ[col + "__high90"] = (out[p.model_id] + p.interval_halfwidth_90).round(3)
    summ.to_csv(os.path.join(ROOT, "results/predictions/predictions_summary.csv"), index=False)
    test = summ[summ.split == "test"]
    cover = {c[:-6]: float(((test["true_pld"] >= test[c[:-6] + "__low90"]) & (test["true_pld"] <= test[c[:-6] + "__high90"])).mean())
             for c in summ.columns if c.endswith("__pred")}
    print(picks[["mode", "role", "model_id", "val_r2", "test_r2"]].to_string(index=False))
    print("90% interval coverage on the test MOFs:", {k: round(v, 3) for k, v in cover.items()})
    print(f"wrote outputs/results/predictions_all_models.csv.gz and results/predictions/ ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
