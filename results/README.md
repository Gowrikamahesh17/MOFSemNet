# Results (CSV) — traceable and reproducible

Everything here is a copy of generated files from `outputs/` (which is git-ignored), plus the comparison tables.
`MANIFEST.csv` lists **every file** with: the script/notebook that produced it, the configuration, the data version,
the random seeds and the git commit. Re-create this folder with the commands at the bottom.

## Common facts
- Task: predict continuous Pore Limiting Diameter (PLD, Å) of 14,296 MOFs; metrics MAE, RMSE, R² on a held-out test set.
- Data: `data/raw/MOFCSD.csv`, SHA-256 prefix `e23a42e742099b4c` (stored in every per-run file as `data_sha256`).
- Seeds: 0–4 for every result. Each run seeds Python, NumPy and PyTorch with the seed; the validation split uses the same seed.
- Test sets (not comparable with each other): `random15` = random 15 % (2,144 MOFs, `split_random_state` 42);
  `cs2_hubs` = Case Study 2's high-gravity nodes per Louvain community (680 MOFs in the server notebook, 664 in the repo code; Louvain seed 42).
- Provenance columns inside the per-run CSVs: `run_id` (hash of the full configuration), `seed`, `test_set`, `data_sha256`, `device`, `finished_at`.

## Folders
| folder | what | produced by |
|---|---|---|
| `server_runs/server_standard_2026-11-19/` | first full server run, ChemBERTa, `random15` | notebook `MOFSemNet_server_experiments.ipynb` (profile `standard`) |
| `server_runs/server_cs2hubs_2026-11-19/` | same plus the `cs2_hubs` test set | same notebook, `CFG["test_sets"]` extended |
| `server_runs/server_molt5_2026-11-20/` | MolT5 + both test sets; **superset** (contains the earlier runs' `run_id`s) | `MOFSemNet_server_molt5.ipynb` |
| `server_runs/server_extra_2026-11-20/` | extra experiments (XGBoost/LightGBM, log target, no LCD/LFS, descriptors, new graphs, stacking), both test sets, 1,700 runs, GPU | `MOFSemNet_server_molt5_extra.ipynb` (profile `extra`) |
| `server_runs/server_extra_final_2026-11-20/` | GNN training fix (standardised target + inputs), log target, richer features for tree models, ensembles; 820 runs, GPU | `MOFSemNet_server_extra_v2.ipynb` (profile `extra_final`) |
| `step5/` | local Step 5 comparison and ablations (ChemBERTa) | `scripts/run_step5_comparison.py`, `scripts/run_step5b_ablations.py` |
| `step6/` | SHAP, column groups, embedding ablation, conformal intervals, fragment mapping of the top fingerprint bits (`step6_*`); SHAP for XGBoost and the GCN, GNNExplainer, LIME and their agreement (`step6b_*`) | `scripts/run_step6_explainability.py`, `scripts/run_step6b_explainers.py`, `scripts/run_step6c_gnn_explainers.py` (SHAP and LIME for GNNs with a graph, incl. Black Hole and MolT5 graphs: `step6c_*`), `scripts/run_step6d_embedding_shap.py` (embedding dimensions: `step6d_*`) |
| `mofgalaxynet_vs_mofsemnet/` | the comparison asked for by the supervisor (below) | `scripts/build_mofgalaxynet_comparison.py` |

Each `server_runs/*` folder holds `per_run_results.csv` (one row per configuration × seed), `aggregated_results.csv`
(mean/std over seeds), `graph_vs_no_graph.csv`, `run_metadata.json` (full config, package versions, GPU, data hash), the notebook used, and the notebook's own README.
Identical configurations appear in more than one server folder with the same `run_id` — that is the repeat-run reproducibility check
(784 of 990 repeated runs identical, largest per-run R² difference 0.008).

## MOFGalaxyNet (regression) vs MOFSemNet
- **Baseline = MOFGalaxyNet-regression:** GNN with the Case Study 2 graph (0.9 × Tanimoto + 0.1 same metal, no language model), fingerprint node features, continuous PLD target.
- **MOFSemNet:** identical GNN, k, pruning and node features, but the graph comes from LLM embeddings (`semantic`) or fingerprints + embeddings (`hybrid`), ChemBERTa or MolT5.
- Same data splits and seeds; paired per seed. Files: `per_seed.csv` (every run), `summary.csv` (mean ± std), `paired.csv` (MOFSemNet − baseline, with a "within seed noise" flag), `headline.csv` (best configuration per system and test set; selected by mean test R², i.e. a reporting choice, not a tuned model).
- Reading it: the semantic layer does **not** give a consistent gain. On `random15` GCN is worse with semantic graphs in 59 of 60 comparisons (mean ΔR² −0.05) while GraphSAGE is better on average (+0.05); on `cs2_hubs` GCN and GraphSAGE gain about +0.03 on average with mixed per-configuration verdicts. The no-edge GNN (~0.84) is as good as the best graph variant of either system, and the best non-graph model (Random Forest, 0.883 / 0.888) is higher than all of them. The residual GCN is a diagnostic, not part of the original two-GNN design, and shows no difference.
- Open point for the supervisor: whether the repo's CS2 graph is exactly the MOFGalaxyNet graph.

## Extra experiments (summary; tables in `server_runs/server_extra_2026-11-20/extra_*.csv`)
Seeds 0-4, both test sets; "within seed noise" = |mean paired difference| <= the larger per-seed std.
1. **XGBoost vs Random Forest** (plain features): random15 0.893 vs 0.883 (+0.010, above seed spread 0.003); cs2_hubs 0.877 vs 0.888 (RF better by 0.011). LightGBM never clearly beats RF. Tuned XGBoost: +0.011 / -0.005. No consistent winner.
2. **Without LCD/LFS** (R² falls to about 0.6): best non-graph XGBoost 0.613 (random15) / 0.649 (hubs); best graph GNN 0.594 / 0.588 (gap 0.019 / 0.062). The 0.019 gap is within 0.02 only on random15, and that GNN is the residual GCN whose no-edge version scores 0.590: the graph itself adds nothing. Every graph vs. no-edge paired comparison is zero or negative, with and without LCD/LFS.
3. **New descriptors:** RDKit linker descriptors help the tree models a little without LCD/LFS (about +0.01 to +0.03 R², mostly above seed spread) and not with LCD/LFS; metal descriptors do nothing.
4. **Log target:** no change for tree models; GNN runs trained on log1p collapsed (R² 0.57-0.72, GraphSAGE negative) — likely a training-setup issue (no feature scaling, fixed learning rate), not evidence about the target; do not cite.
5. **Graphs:** de-duplicated and descriptor-space graphs are not better than the CS2 graph and remain below the no-edge GNN. Graph-smoothed features give XGBoost no gain.

## GNN fix, richer features, ensembles (`server_extra_final_2026-11-20`; mean R², seeds 0-4)
- **GNN training fix:** the old GNN runs never converged (200 epochs always used; target scale). Standardising the target and inputs lifts the no-edge GNN from 0.84 to about 0.874 (random15, with LCD/LFS) and removes the log-target collapse (log 0.863). Residual GCN is 0.875.
- **Graphs still do not help GCN/GraphSAGE:** full CS2 graph 0.643-0.657 vs 0.872-0.874 without edges. Residual GCN with a graph equals its no-edge version (0.873 vs 0.875) except Black Hole 30% on the hub test set (+0.009, above seed spread; hub test set carries a selection-bias caveat) and negative on random15 (-0.008).
- **Gap to the best tree model** (with LCD/LFS) shrank from about 0.05 to 0.023-0.024 (best graph GNN 0.873/0.883 vs best non-graph 0.897/0.906); not within 0.02. Without LCD/LFS the gap is 0.047-0.057.
- **Richer features** (count fingerprints, MACCS, all RDKit descriptors): no reliable gain; differences are within seed noise (largest +0.004 for XGBoost); Random Forest gains up to +0.013 on random15 with LCD/LFS only.
- **Ensembles** (RF + XGBoost + no-edge GNN): equal to or up to +0.011-0.013 above the best single model in some settings (hubs with LCD/LFS 0.906; random15 without LCD/LFS 0.621 vs 0.609), but not above XGBoost on random15 with LCD/LFS (0.894 vs 0.897).

## Regenerate
```bash
python scripts/export_traceable_results.py          # copies outputs/results -> results/, writes MANIFEST.csv
python scripts/build_mofgalaxynet_comparison.py     # needs results/server_runs/server_molt5_2026-11-20/per_run_results.csv
python scripts/export_traceable_results.py          # run again so the manifest lists the comparison files
```
The server runs themselves are regenerated by opening the notebook named in the table on a machine with MOFCSD.csv and the embedding `.npz` files, setting the profile/overrides recorded in `run_metadata.json`, and Run All (delete `per_run_results.csv` first for a clean run). Step 5/6 CSVs: run the scripts above with the arguments in their `--help` and the log files in `outputs/logs/`.
