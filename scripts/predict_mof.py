#!/usr/bin/env python
"""Predict the pore limiting diameter of a MOF from its linker and metal (optionally its pore geometry) with the saved models.

Examples
  PYTHONPATH=. python scripts/predict_mof.py --smiles "OC(=O)c1ccc(cc1)C(=O)O" --metal Zn
  PYTHONPATH=. python scripts/predict_mof.py --smiles "OC(=O)c1ccc2cc(ccc2c1)C(=O)O" --metal Zr --lcd 9.1 --lfs 8.4 \\
      --graph semantic_molt5 --k 5 --pruning black_hole_30
"""

import argparse

from src.demo.predictor import GRAPH_TYPES, PRUNINGS, MOFPredictor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smiles", required=True, help="linker SMILES")
    ap.add_argument("--metal", required=True, help="metal symbol, one of the 53 in the data (e.g. Zn, Cu, Zr)")
    ap.add_argument("--lcd", type=float, help="largest cavity diameter (A); with --lfs switches to the more accurate 'with pore geometry' mode")
    ap.add_argument("--lfs", type=float, help="largest free sphere (A)")
    ap.add_argument("--graph", default="semantic_chemberta", choices=GRAPH_TYPES)
    ap.add_argument("--k", type=int, default=3, choices=[3, 5, 10])
    ap.add_argument("--pruning", default="none", choices=PRUNINGS)
    ap.add_argument("--all-graphs", action="store_true", help="run every GNN of the grid, not just the chosen graph")
    a = ap.parse_args()
    mode = "with_pore" if a.lcd is not None and a.lfs is not None else "chemistry_only"
    r = MOFPredictor().predict(a.smiles, a.metal, mode, a.lcd, a.lfs, a.graph, a.k, a.pruning, all_graphs=a.all_graphs)
    print(f"\nLinker {r.smiles} with {r.metal}  |  mode: {r.mode}  |  "
          + (f"known linker, used in {r.mofs_with_this_linker} MOFs" if r.known_linker else "NEW linker (not in the data)"))
    if r.known_mofs_same_linker_and_metal:
        print("Known MOFs with this linker and metal (true PLD):", ", ".join(f"{c} {v} A" for c, v in r.known_mofs_same_linker_and_metal))
    best = r.predictions[0]
    print(f"\nPredicted PLD: {best.pld:.2f} A  (90% interval {best.low90:.2f}-{best.high90:.2f} A, best model: {best.model_id})\n")
    print(f"{'#':>2}  {'model':<62} {'PLD (A)':>8} {'90% interval':>15} {'test R2':>8}")
    for i, p in enumerate(r.predictions, 1):
        print(f"{i:>2}  {p.model_id:<62} {p.pld:8.2f} {p.low90:6.2f}-{p.high90:<7.2f} {p.test_r2:8.3f}")
    print("\nMost similar known MOFs:")
    for n in r.nearest_mofs:
        print(f"  {n['refcode']:<10} {n['metal']:<3} PLD {n['pld']:5.2f} A  similarity {n['similarity']:.3f} ({n['by']})  {n['linker_smiles']}")
    for note in r.notes:
        print("Note:", note)


if __name__ == "__main__":
    main()
