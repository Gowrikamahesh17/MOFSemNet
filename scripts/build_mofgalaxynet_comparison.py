"""MOFSemNet vs. the regression MOFGalaxyNet baseline, from the server runs (same splits, same seeds, same metrics).

Definition used (agreed in the supervisor's e-mail: "MOFSemNet without the LLM and semantic components"):

  * MOFGalaxyNet-regression : GNN (GCN / GraphSAGE / residual GCN) with a Case Study 2 graph (0.9 x Tanimoto + 0.1 x
                              same metal; no language model), fingerprint node features, continuous PLD target.
  * MOFSemNet               : the same GNN, same k, same sparsification, same node features, but the graph is built
                              from LLM embeddings (semantic) or from fingerprints + embeddings (hybrid), ChemBERTa or MolT5.

Both are scored on identical splits (test set + seed) with MAE, RMSE and R2. Each row is traced to the notebook
run (run_id, per-run CSV, data hash, split state). Run:  python scripts/build_mofgalaxynet_comparison.py
Reads  results/server_runs/server_molt5_2026-11-20/per_run_results.csv  (the run that contains every other run's ids)
Writes results/mofgalaxynet_vs_mofsemnet/{per_seed,summary,paired,headline}.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/server_runs/server_molt5_2026-11-20/per_run_results.csv"
OUT = ROOT / "results/mofgalaxynet_vs_mofsemnet"
SCRIPT = "scripts/build_mofgalaxynet_comparison.py"
NOTEBOOK = "results/server_runs/server_molt5_2026-11-20/MOFSemNet_server_molt5.ipynb"
CONFIG = "results/server_runs/server_molt5_2026-11-20/run_metadata.json"
SPLIT_STATE = 42  # random15 test set; cs2_hubs is deterministic (Louvain seed 42)
MATCH = ["model", "graph_type_family", "k", "sparsify", "retention", "bh_weights", "test_set", "seed"]


def system_name(row) -> str:
    if row["graph_type"] == "none":
        return "no edges (reference)"
    if row["graph_type"] == "cs2":
        return "MOFGalaxyNet-regression (cs2 graph, no LLM)"
    if row["graph_type"] == "fingerprint":
        return "fingerprint graph (no LLM, no metal term)"
    return f"MOFSemNet ({row['graph_type']} graph, {row['language_model']})"


def main() -> None:
    per = pd.read_csv(SRC)
    g = per[per["block"].isin(["graph_full", "sparsify", "no_edge"]) & (per["node_features"] == "fingerprint")].copy()
    g["system"] = g.apply(system_name, axis=1)
    g["graph_type_family"] = g["graph_type"]
    g["source_per_run_csv"], g["script"], g["notebook"], g["config_file"] = str(SRC.relative_to(ROOT)), SCRIPT, NOTEBOOK, CONFIG
    g["split_random_state"] = np.where(g["test_set"] == "random15", str(SPLIT_STATE), "deterministic (Louvain seed 42)")
    cols = ["system", "test_set", "model", "graph_type", "language_model", "k", "sparsify", "retention", "bh_weights", "seed",
            "mae", "rmse", "r2", "run_id", "data_sha256", "split_random_state", "source_per_run_csv", "notebook", "config_file"]
    OUT.mkdir(parents=True, exist_ok=True)
    g[cols].sort_values(["test_set", "system", "model", "k", "sparsify", "retention", "seed"]).to_csv(OUT / "per_seed.csv", index=False)

    keys = ["system", "test_set", "model", "k", "sparsify", "retention", "bh_weights"]
    summary = (g.groupby(keys, dropna=False)
               .agg(n_seeds=("seed", "nunique"), mae_mean=("mae", "mean"), mae_std=("mae", "std"), rmse_mean=("rmse", "mean"),
                    rmse_std=("rmse", "std"), r2_mean=("r2", "mean"), r2_std=("r2", "std"))
               .reset_index())
    summary.to_csv(OUT / "summary.csv", index=False)

    # paired differences: MOFSemNet variant minus the MOFGalaxyNet-regression baseline, matched on model/k/sparsify/retention/test set/seed
    base = g[g["system"].str.startswith("MOFGalaxyNet")]
    rows = []
    for system, grp in g[g["system"].str.startswith("MOFSemNet")].groupby("system"):
        m = grp.merge(base, on=["model", "k", "sparsify", "retention", "bh_weights", "test_set", "seed"], suffixes=("", "_base"))
        m["d_r2"], m["d_mae"], m["d_rmse"] = m["r2"] - m["r2_base"], m["mae"] - m["mae_base"], m["rmse"] - m["rmse_base"]
        for key, d in m.groupby(["test_set", "model", "k", "sparsify", "retention"]):
            spread = max(d["r2"].std(), d["r2_base"].std())
            rows.append(dict(system=system, baseline="MOFGalaxyNet-regression (cs2 graph, no LLM)", test_set=key[0], model=key[1], k=key[2],
                             sparsify=key[3], retention=key[4], n_seeds=len(d), r2_semnet=d["r2"].mean(), r2_baseline=d["r2_base"].mean(),
                             mean_d_r2=d["d_r2"].mean(), std_d_r2=d["d_r2"].std(), mean_d_mae=d["d_mae"].mean(), mean_d_rmse=d["d_rmse"].mean(),
                             seed_spread=spread,
                             verdict="within seed noise" if abs(d["d_r2"].mean()) <= spread else ("MOFSemNet better" if d["d_r2"].mean() > 0 else "baseline better")))
    paired = pd.DataFrame(rows)
    paired.to_csv(OUT / "paired.csv", index=False)

    # headline: best configuration per system and test set (chosen by mean test R2 = a reporting choice, not a tuned model)
    head = summary.sort_values("r2_mean", ascending=False).groupby(["test_set", "system"], as_index=False).head(1)
    head.sort_values(["test_set", "r2_mean"], ascending=[True, False]).to_csv(OUT / "headline.csv", index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(g)} per-seed rows, {len(summary)} configurations, {len(paired)} paired comparisons")
    print(head.sort_values(["test_set", "r2_mean"], ascending=[True, False])[["test_set", "system", "model", "k", "sparsify", "retention", "r2_mean", "r2_std"]].round(4).to_string(index=False))
    print("\npaired verdicts:", paired["verdict"].value_counts().to_dict())


if __name__ == "__main__":
    main()
