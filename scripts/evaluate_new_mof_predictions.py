#!/usr/bin/env python
"""How accurate are the saved graph GNNs when a MOF is treated as NEW (added to the graph as an extra node)?

The saved test scores of GNNs with a graph were measured with each test MOF in its original place in the graph. In the live demo a
MOF is added as a new node instead (joined to its k most similar MOFs). This script replays that for test MOFs: each one is fed to
src.demo.predictor exactly like a user's MOF, and the predictions are scored against the true PLD. Tree models and no-edge GNNs are
not affected (their predictions do not depend on the graph) and are not re-scored.

Usage:  PYTHONPATH=. python scripts/evaluate_new_mof_predictions.py [--n-test 400] [--mode with_pore]
Writes  results/predictions/new_mof_accuracy.csv
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.demo.predictor import ROOT, MOFPredictor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-test", type=int, default=400, help="test MOFs to replay (sampled with a fixed seed); 0 = all")
    ap.add_argument("--modes", nargs="+", default=["with_pore", "chemistry_only"])
    ap.add_argument("--graphs", nargs="+", default=["fingerprint", "semantic_chemberta", "semantic_molt5"])
    ap.add_argument("--k", type=int, nargs="+", default=[3])
    ap.add_argument("--prunings", nargs="+", default=["none", "black_hole_30"])
    args = ap.parse_args()
    P = MOFPredictor()
    allp = pd.read_csv(os.path.join(ROOT, "outputs/results/predictions_all_models.csv.gz"))
    raw = pd.read_csv(os.path.join(ROOT, "data/raw/MOFCSD.csv"), index_col=0)
    test = allp[allp.split == "test"]
    if args.n_test:
        test = test.sample(n=min(args.n_test, len(test)), random_state=0)
    rows, t0 = [], time.time()
    for mode in args.modes:
        for g in args.graphs:
            for k in args.k:
                for pr in args.prunings:
                    preds = {}
                    for r in test.itertuples():
                        x = raw.loc[r.refcode]
                        res = P.predict(r.linker_smiles, r.metal, mode, lcd=x["Largest Cavity Diameter"], lfs=x["Largest Free Sphere"],
                                        graph_type=g, k=k, pruning=pr)
                        for p in res.predictions:
                            if p.graph != "none":
                                preds.setdefault(p.model_id, []).append(p.pld)
                    y = test["true_pld"].values
                    for mid, pv in preds.items():
                        pv = np.array(pv)
                        stored = test[mid].values
                        rows.append({"model_id": mid, "mode": mode, "graph": g, "k": k, "pruning": pr, "n_test_mofs": len(y),
                                     "r2_as_new_mof": r2_score(y, pv), "mae_as_new_mof": mean_absolute_error(y, pv),
                                     "r2_in_original_graph_same_mofs": r2_score(y, stored), "mae_in_original_graph_same_mofs": mean_absolute_error(y, stored),
                                     "mean_abs_change_A": float(np.abs(pv - stored).mean())})
                    print(f"[{time.time() - t0:5.0f}s] {mode} {g} k={k} {pr}: " + ", ".join(
                        f"{r_['model_id'].split('__')[-1]} {r_['r2_as_new_mof']:.3f} (orig {r_['r2_in_original_graph_same_mofs']:.3f})" for r_ in rows[-len(preds):]), flush=True)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "results/predictions"), exist_ok=True)
    out.to_csv(os.path.join(ROOT, "results/predictions/new_mof_accuracy.csv"), index=False)
    print(out.groupby(["mode", "model_id"]).size().shape, "rows written to results/predictions/new_mof_accuracy.csv")


if __name__ == "__main__":
    main()
