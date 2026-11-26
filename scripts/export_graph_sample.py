"""Export a real 140-MOF piece of the saved fingerprint graph for the Research Explorer's "The graphs" tab (hover details).

Reads the graphs saved by MOFSemNet_server_save_models.ipynb (outputs/models/mofsemnet_models/graphs/fingerprint_k{3,5,10}_{none,black_hole_30}.npz)
and data/raw/MOFCSD.csv. Picks a connected neighbourhood by breadth-first search on the k=5 graph, keeps the real edges among those MOFs for
k = 3, 5, 10, which MOFs Black Hole (keep 30%) keeps, and a fixed layout. Writes frontend/public/graph_sample.json,
and hover facts for the linkers of the fingerprint-vs-semantic panels to frontend/public/linker_info.json.

Usage:  PYTHONPATH=. python scripts/export_graph_sample.py [--n 140] [--seed-refcode ...]
"""
import argparse
import json
import os
from collections import deque

import networkx as nx
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = os.path.join(ROOT, "outputs/models/mofsemnet_models/graphs")


def load(k, pruning):
    z = np.load(os.path.join(G, f"fingerprint_k{k}_{pruning}.npz"))
    return z["edge_index"], z["edge_weight"], z["kept_nodes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=140)
    ap.add_argument("--seed-refcode", default=None, help="MOF to start from (default: a mid-degree MOF chosen with numpy seed 0)")
    a = ap.parse_args()
    mofs = pd.read_csv(os.path.join(ROOT, "outputs/models/mofsemnet_models/features/mofs.csv"))
    raw = pd.read_csv(os.path.join(ROOT, "data/raw/MOFCSD.csv")).set_index("refcode")
    e3, _, _ = load(5, "none")
    adj = [[] for _ in range(len(mofs))]
    for i, j in e3.T:
        adj[i].append(j)
    if a.seed_refcode:
        start = int(np.flatnonzero(mofs.refcode == a.seed_refcode)[0])
    else:   # a MOF whose linker is shared by only a few MOFs, so the sample is not one big block of identical-linker twins
        counts = mofs.linker_smiles.map(mofs.linker_smiles.value_counts())
        cand = np.flatnonzero((counts >= 2) & (counts <= 4))
        start = int(np.random.default_rng(0).choice(cand))
    seen, q = [start], deque([start])
    while q and len(seen) < a.n:
        for j in adj[q.popleft()]:
            if j not in seen and len(seen) < a.n:
                seen.append(int(j)); q.append(j)
    idx = {g: i for i, g in enumerate(seen)}
    out = {"source": "fingerprint graph (Tanimoto similarity of the linker's Morgan fingerprint), saved demo models", "k": {}, "nodes": []}
    for k in (3, 5, 10):
        res = {}
        for pr in ("none", "black_hole_30"):
            ei, ew, kept = load(k, pr)
            m = np.isin(ei[0], seen) & np.isin(ei[1], seen) & (ei[0] < ei[1])
            res[pr] = {"edges": [[idx[int(i)], idx[int(j)], round(float(w), 3)] for i, j, w in zip(ei[0][m], ei[1][m], ew[m])],
                       "kept": [idx[g] for g in seen if kept[g]], "degree": [int((ei[0] == g).sum()) for g in seen]}
        out["k"][str(k)] = {"edges": res["none"]["edges"], "degree": res["none"]["degree"], "bh_kept": res["black_hole_30"]["kept"],
                            "bh_edges": res["black_hole_30"]["edges"]}
    g = nx.Graph(); g.add_nodes_from(range(len(seen))); g.add_edges_from((i, j) for i, j, _ in out["k"]["10"]["edges"])
    pos = nx.spring_layout(g, seed=1, k=2.2 / np.sqrt(len(seen)), iterations=300)
    xy = np.array([pos[i] for i in range(len(seen))]); xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    for i, gi in enumerate(seen):
        r = mofs.iloc[gi]; rr = raw.loc[r.refcode]
        rr = rr.iloc[0] if isinstance(rr, pd.DataFrame) else rr
        out["nodes"].append({"refcode": r.refcode, "metal": r.metal, "linker": r.linker_smiles, "pld": round(float(r.pld), 2),
                             "lcd": round(float(rr["Largest Cavity Diameter"]), 2), "lfs": round(float(rr["Largest Free Sphere"]), 2),
                             "linker_mofs": int((mofs.linker_smiles == r.linker_smiles).sum()),
                             "x": round(30 + float(xy[i, 0]) * 940, 1), "y": round(25 + float(xy[i, 1]) * 330, 1)})
    p = os.path.join(ROOT, "frontend/public/graph_sample.json")
    json.dump(out, open(p, "w"), separators=(",", ":"))
    print(f"wrote {p}: {len(seen)} MOFs, edges k3/k5/k10 = {[len(out['k'][k]['edges']) for k in ('3', '5', '10')]}, "
          f"Black Hole keeps {[len(out['k'][k]['bh_kept']) for k in ('3', '5', '10')]}")
    export_linker_info(mofs)


def export_linker_info(mofs):
    """Hover facts for the linkers drawn in the fingerprint-vs-semantic panels (content/graph_comparison_<model>.json):
    formula, metals and PLD range of the MOFs that use each linker. Writes frontend/public/linker_info.json."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    smiles = set()
    for lm in ("chemberta", "molt5"):
        f = os.path.join(ROOT, f"content/graph_comparison_{lm}.json")
        if os.path.exists(f):
            for h in json.load(open(f))["neighbourhoods"]:
                smiles.update(n["smiles"] for n in h["nodes"])
    info = {}
    for s_ in sorted(smiles):
        sub = mofs[mofs.linker_smiles == s_]
        mol = Chem.MolFromSmiles(s_)
        info[s_] = {"formula": rdMolDescriptors.CalcMolFormula(mol) if mol else "",
                    "metals": [[m, int(c)] for m, c in sub.metal.value_counts().head(4).items()], "n_metals": int(sub.metal.nunique()),
                    "pld_min": round(float(sub.pld.min()), 2), "pld_max": round(float(sub.pld.max()), 2)}
    p = os.path.join(ROOT, "frontend/public/linker_info.json")
    json.dump(info, open(p, "w"), separators=(",", ":"))
    print(f"wrote {p}: {len(info)} linkers")


if __name__ == "__main__":
    main()
