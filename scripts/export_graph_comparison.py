#!/usr/bin/env python
"""Export a fingerprint-vs-semantic graph comparison for the web console
(content/graph_comparison_<model>.json, served at /api/graphs/comparison?model=<model>). Reads the cached Step 3
graphs; computes nothing new about the models, only describes how the two edge sets differ.

Many MOFs share one identical linker (the dataset has far fewer distinct linkers than MOFs), so
MOF-level edges between same-linker MOFs are mostly tie-breaks. The network view is therefore drawn
at the *linker* level: one node per distinct linker, an edge wherever any pair of MOFs with those two
different linkers is joined in the graph.

Usage (from the repo root):
    PYTHONPATH=. python scripts/export_graph_comparison.py --model chemberta
"""

import argparse
import collections
import json
import pickle
import random

import networkx as nx
import numpy as np

from src.baseline.data_ingestion import load_dataset
from src.common.config import LANGUAGE_MODEL_CHOICES, RuntimeConfig

OUT_PATH = "content/graph_comparison_{model}.json"
N_NEIGHBOURHOODS = 4
MAX_EDGES_DRAWN = 120  # skip neighbourhoods too dense to read
SEED = 1


def load_graphs(config: RuntimeConfig, model: str) -> dict:
    graphs = {}
    for graph_type in ("fingerprint", "semantic"):
        with open(f"{config.graphs_cache_dir}/{graph_type}_{model}.gpickle", "rb") as fh:
            graphs[graph_type] = pickle.load(fh)
    return graphs


def linker_edges(graph: nx.Graph, linkers) -> collections.Counter:
    weights: collections.Counter = collections.Counter()
    for a, b in graph.edges():
        if linkers[a] != linkers[b]:
            weights[tuple(sorted((linkers[a], linkers[b])))] += 1
    return weights


def neighbourhood(seed: str, union: nx.Graph) -> list:
    ball = [seed] + sorted(union[seed], key=lambda x: -union.degree(x))[:14]
    for n in list(ball[1:6]):
        ball += sorted((m for m in union[n] if m not in ball), key=lambda x: -union.degree(x))[:6]
    return ball


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=LANGUAGE_MODEL_CHOICES, default="chemberta")
    args = parser.parse_args()

    config = RuntimeConfig(language_model=args.model)
    df = load_dataset("large")
    linkers = df["linker_smiles"].values
    pld = df["pld_value"].values
    graphs = load_graphs(config, args.model)
    counts = collections.Counter(linkers)

    mof_edges = {t: {tuple(sorted(e)) for e in g.edges()} for t, g in graphs.items()}
    shared = mof_edges["fingerprint"] & mof_edges["semantic"]
    weights = {t: linker_edges(g, linkers) for t, g in graphs.items()}
    ef, es = set(weights["fingerprint"]), set(weights["semantic"])
    union = nx.Graph()
    union.add_edges_from(ef | es)

    rng = np.random.default_rng(0)
    random_pairs = np.abs(pld[rng.integers(0, len(pld), 100000)] - pld[rng.integers(0, len(pld), 100000)])
    stats = {
        "mofs": len(df),
        "distinct_linkers": len(counts),
        "mof_edges": {t: len(e) for t, e in mof_edges.items()},
        "mof_jaccard": len(shared) / len(mof_edges["fingerprint"] | mof_edges["semantic"]),
        "same_linker_edge_share": {
            t: float(np.mean([linkers[a] == linkers[b] for a, b in g.edges()])) for t, g in graphs.items()
        },
        "linker_pair_edges": {"fingerprint": len(ef), "semantic": len(es), "shared": len(ef & es)},
        "linker_pair_jaccard": len(ef & es) / len(ef | es),
        "components": {t: nx.number_connected_components(g) for t, g in graphs.items()},
        "mean_abs_pld_gap_between_neighbours": {
            t: float(np.mean([abs(pld[a] - pld[b]) for a, b in g.edges()])) for t, g in graphs.items()
        },
        "mean_abs_pld_gap_random_pair": float(random_pairs.mean()),
    }

    random.seed(SEED)
    candidates = [l for l, c in counts.items() if c >= 8 and l in union and union.degree(l) >= 8]
    picked = []
    for seed in random.sample(candidates, min(120, len(candidates))):
        ball = neighbourhood(seed, union)
        members = set(ball)
        fp = {e for e in ef if e[0] in members and e[1] in members}
        sem = {e for e in es if e[0] in members and e[1] in members}
        if 60 <= len(fp) + len(sem) <= MAX_EDGES_DRAWN * 2 and max(len(fp), len(sem)) <= MAX_EDGES_DRAWN:
            picked.append((len(fp & sem), seed, members, fp, sem))
    picked.sort(key=lambda p: -p[0])

    hoods = []
    used: set = set()
    for n_shared, seed, members, fp, sem in picked:
        if members & used:
            continue
        used |= members
        sub = nx.Graph()
        sub.add_edges_from(fp | sem)
        pos = nx.spring_layout(sub, seed=7, k=0.9 / np.sqrt(max(1, sub.number_of_nodes())), iterations=200)
        order = sorted(sub.nodes())
        xy = np.array([pos[n] for n in order])
        xy = (xy - xy.min(0)) / np.maximum(xy.max(0) - xy.min(0), 1e-9)
        index = {n: i for i, n in enumerate(order)}
        nodes = []
        for n, (x, y) in zip(order, xy):
            mask = linkers == n
            nodes.append({
                "smiles": n,
                "mofs": int(counts[n]),
                "mean_pld": round(float(pld[mask].mean()), 2),
                "x": round(float(x), 4),
                "y": round(float(y), 4),
            })
        def edge_list(edge_set, kind_weights):
            return [[index[a], index[b], kind_weights[(a, b)]] for a, b in sorted(edge_set)]
        hoods.append({
            "seed": index[seed],
            "nodes": nodes,
            "fingerprint": edge_list(fp, weights["fingerprint"]),
            "semantic": edge_list(sem, weights["semantic"]),
            "shared_edges": len(fp & sem),
        })
        if len(hoods) == N_NEIGHBOURHOODS:
            break

    payload = {"model": args.model, "k": 10, "stats": stats, "neighbourhoods": hoods}
    path = OUT_PATH.format(model=args.model)
    with open(path, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"wrote {path}: {len(hoods)} neighbourhoods")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
