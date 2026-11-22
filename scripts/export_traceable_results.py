"""Copy the result CSVs that matter into the tracked `results/` folder and write results/MANIFEST.csv.

`outputs/` is git-ignored (generated artefacts), so nothing there can be pushed. This script gathers, per file, the
CSVs (never the .xlsx or .zip copies) and records for each one the script that produced it, the configuration, the
data version (SHA-256 prefix), the random seeds and the git commit. Run:  python scripts/export_traceable_results.py
Re-running overwrites the copies, so `results/` always mirrors `outputs/results/`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "outputs/results", ROOT / "results"
DATA_SHA = "e23a42e742099b4c"      # data/raw/MOFCSD.csv (also stored in every per-run file)
SEEDS = "0,1,2,3,4"
SERVER_NOTEBOOKS = {"server_standard_2026-11-19": "MOFSemNet_server_experiments.ipynb",
                    "server_cs2hubs_2026-11-19": "MOFSemNet_server_experiments.ipynb",
                    "server_molt5_2026-11-20": "MOFSemNet_server_molt5.ipynb",
                    "server_extra_2026-11-20": "MOFSemNet_server_molt5_extra.ipynb",
                    "server_extra_final_2026-11-20": "MOFSemNet_server_extra_v2.ipynb"}
STEP_SCRIPT = {"step5_": "scripts/run_step5_comparison.py", "step5b_": "scripts/run_step5b_ablations.py",
               "step6d_": "scripts/run_step6d_embedding_shap.py", "step6c_": "scripts/run_step6c_gnn_explainers.py", "step6b_": "scripts/run_step6b_explainers.py", "step6_": "scripts/run_step6_explainability.py", "cs2_hub_test_nodes": "scripts/run_step6_explainability.py"}


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    commit = git("rev-parse", "HEAD")
    dirty = "yes" if git("status", "--porcelain", "--", "scripts", "src", "MOFSemNet_server_molt5.ipynb") else "no"
    rows = []

    def record(rel: Path, script: str, config: str, seeds: str, note: str = "") -> None:
        n = len(pd.read_csv(DST / rel)) if rel.suffix == ".csv" else ""
        rows.append(dict(file=str(Path("results") / rel), n_rows=n, script_or_notebook=script, configuration=config,
                         data_version=f"MOFCSD.csv sha256 {DATA_SHA}…", seeds=seeds, git_commit=commit,
                         uncommitted_changes_to_code=dirty, note=note))

    for folder, notebook in SERVER_NOTEBOOKS.items():
        out = DST / "server_runs" / folder
        out.mkdir(parents=True, exist_ok=True)
        for name in ("per_run_results.csv", "aggregated_results.csv", "graph_vs_no_graph.csv", "run_metadata.json", "README.txt", notebook,
                     "extra_comparison_table.csv", "extra_paired_differences.csv", "extra_q2_graph_vs_nongraph.csv"):
            if (SRC / folder / name).exists():
                shutil.copy2(SRC / folder / name, out / name)
        meta = json.load(open(out / "run_metadata.json"))
        cfg = f"results/server_runs/{folder}/run_metadata.json (profile={meta['profile']}, test_sets={meta['config']['test_sets']}, language_models={meta['config']['language_models']})"
        for name in ("per_run_results.csv", "aggregated_results.csv", "graph_vs_no_graph.csv", "extra_comparison_table.csv",
                     "extra_paired_differences.csv", "extra_q2_graph_vs_nongraph.csv"):
            if not (out / name).exists():
                continue
            note = {"extra_comparison_table.csv": "model x features x target x test set, mean/std R2, MAE, RMSE (extra profile)",
                    "extra_paired_differences.csv": "paired per-seed R2 differences with within-seed-noise flag",
                    "extra_q2_graph_vs_nongraph.csv": "best graph GNN vs best non-graph model, with and without LCD/LFS",
                    "per_run_results.csv": "one row per configuration x seed; run_id is the key",
                    "aggregated_results.csv": "mean/std over seeds", "graph_vs_no_graph.csv": "each graph run minus the same model with no edges"}[name]
            record(Path("server_runs") / folder / name, f"results/server_runs/{folder}/{notebook}", cfg, ",".join(map(str, meta["seeds"])), note)

    for f in sorted(SRC.glob("*.csv")):
        script = next((s for prefix, s in STEP_SCRIPT.items() if f.name.startswith(prefix)), "")
        sub = "step5" if f.name.startswith(("step5_", "step5b_")) else "step6"
        (DST / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, DST / sub / f.name)
        record(Path(sub) / f.name, script, f"command-line arguments in {script}; model/test set in the file name (…_hubs = Case Study 2 hub test set)",
               SEEDS if "per_seed" in f.name or sub != "step6" else SEEDS, "")

    comp = DST / "mofgalaxynet_vs_mofsemnet"
    for f in sorted(comp.glob("*.csv")):
        record(Path("mofgalaxynet_vs_mofsemnet") / f.name, "scripts/build_mofgalaxynet_comparison.py",
               "reads results/server_runs/server_molt5_2026-11-20/per_run_results.csv", SEEDS, "regression MOFGalaxyNet baseline vs MOFSemNet")

    pd.DataFrame(rows).to_csv(DST / "MANIFEST.csv", index=False)
    print(f"copied results into {DST.relative_to(ROOT)}; MANIFEST.csv lists {len(rows)} files (commit {commit[:8]}, code uncommitted: {dirty})")


if __name__ == "__main__":
    main()
