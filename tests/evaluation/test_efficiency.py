import time

import networkx as nx

from src.evaluation.efficiency import graph_efficiency_row, track_efficiency


def test_track_efficiency_measures_positive_elapsed_time():
    with track_efficiency() as stats:
        time.sleep(0.01)
    assert stats.elapsed_seconds > 0
    assert stats.peak_memory_mb > 0


def test_graph_efficiency_row():
    graph = nx.erdos_renyi_graph(10, 0.3, seed=0)
    row = graph_efficiency_row(graph)
    assert row["retained_nodes"] == 10
    assert row["retained_edges"] == graph.number_of_edges()
