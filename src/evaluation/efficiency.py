"""Step 5: training time + memory tracking, alongside retained nodes/edges, for the comparison
table. A context manager rather than scattered time.time()/psutil calls in each training script,
matching src/baseline/black_hole_sparsification.py's own inline timing/memory pattern but
reusable across every model in the Step 5 sweep (RF, MLP, GCN, GraphSAGE) instead of duplicated
per-script.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import psutil


@dataclass
class EfficiencyStats:
    elapsed_seconds: float
    peak_memory_mb: float


class track_efficiency:
    """Usage:
        with track_efficiency() as stats:
            train(...)
        stats.elapsed_seconds, stats.peak_memory_mb
    """

    def __enter__(self) -> EfficiencyStats:
        self._process = psutil.Process()
        self._start_time = time.time()
        self._mem_before = self._process.memory_info().rss / (1024 * 1024)
        self.stats = EfficiencyStats(elapsed_seconds=0.0, peak_memory_mb=self._mem_before)
        return self.stats

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        mem_after = self._process.memory_info().rss / (1024 * 1024)
        self.stats.elapsed_seconds = round(time.time() - self._start_time, 4)
        self.stats.peak_memory_mb = round(max(self._mem_before, mem_after), 2)
        return False


def graph_efficiency_row(graph) -> dict:
    """Retained nodes/edges — the structural half of Step 5's efficiency reporting, alongside
    EfficiencyStats' time/memory half."""
    return {
        "retained_nodes": graph.number_of_nodes(),
        "retained_edges": graph.number_of_edges(),
    }
