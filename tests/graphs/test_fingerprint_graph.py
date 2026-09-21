import pandas as pd

from src.graphs.fingerprint_graph import build_fingerprint_graph

SAMPLE_SMILES = pd.Series([
    "c1ccccc1",       # benzene
    "c1ccccc1C",      # toluene (similar to benzene)
    "CCO",            # ethanol
    "CCCCCCCC",       # octane
    "c1ccc(cc1)O",    # phenol (similar to benzene)
])


def test_fingerprint_graph_node_count():
    graph = build_fingerprint_graph(SAMPLE_SMILES, k=2)
    assert graph.number_of_nodes() == len(SAMPLE_SMILES)


def test_fingerprint_graph_connects_similar_molecules():
    graph = build_fingerprint_graph(SAMPLE_SMILES, k=2)
    # Benzene, toluene, and phenol are all aromatic rings — expect at least one edge among them.
    aromatic_indices = {0, 1, 4}
    edges_among_aromatics = [
        (u, v) for u, v in graph.edges() if u in aromatic_indices and v in aromatic_indices
    ]
    assert len(edges_among_aromatics) > 0
