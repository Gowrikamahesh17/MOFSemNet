MOFSemNet server run — how these files were made and how to regenerate them
=============================================================================
Notebook      : the single notebook this folder came from (profile = extra_final)
Data          : MOFCSD.csv  (SHA-256 e23a42e742099b4c94ae55fb29f51678fa5d9f4a91923ca753143d60638451e8)
Seeds         : [0, 1, 2, 3, 4]  (every run seeds python, numpy and torch; splits use the same seed)
Test sets     : ['random15', 'cs2_hubs']  (random15 = 15% random MOFs, fixed with split_state=42;
                cs2_hubs = Case Study 2's high-gravity Louvain-community test set)
Target        : continuous Pore Limiting Diameter (Angstrom); metrics MAE, RMSE, R2 on the held-out test set.

Files
  per_run_results.csv     one row per (configuration, seed): all settings, metrics, timing, graph size
  aggregated_results.csv  mean and std over seeds per configuration
  graph_vs_no_graph.csv   each graph result compared with the same model run WITHOUT edges
  run_metadata.json       full configuration, package versions, GPU, data hash
  mofsemnet_results.xlsx  all of the above as sheets

To regenerate: put MOFCSD.csv (and the embeddings .npz files) next to the notebook, set PROFILE = "extra_final"
and the CFG overrides recorded in run_metadata.json, then Run All. Delete per_run_results.csv first
for a clean run (otherwise finished runs are skipped = resume).
