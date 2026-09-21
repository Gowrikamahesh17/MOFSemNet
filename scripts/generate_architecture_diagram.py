#!/usr/bin/env python
"""Generates docs/architecture.mmd (Mermaid source) from a plain Python list of pipeline
stages — a script rather than a hand-maintained diagram file, so the diagram never drifts from
the 6-step pipeline it's describing without someone noticing the mismatch (edit STAGES below,
re-run the script). Mermaid was chosen over a rendered-image diagramming tool because GitHub and
most Markdown viewers render ```mermaid code blocks natively, so README.md can embed it directly
with zero extra runtime dependency (no graphviz, no image file to keep in sync).

Usage:
    python scripts/generate_architecture_diagram.py
"""

import os

STAGES = [
    ("s1", "Step 1\nBaseline reproduction\n(RF / k-NN / GCN / GraphSAGE / GAT\non Tanimoto graph)"),
    ("s2", "Step 2\nLanguage embeddings\n(ChemBERTa / MolT5)"),
    ("s3a", "Fingerprint graph"),
    ("s3b", "Semantic graph"),
    ("s3c", "Hybrid graph"),
    ("s4", "Step 4\nSparsification\n(Black Hole vs. random vs. k-center\n@ 30% / 50% retention)"),
    ("s5", "Step 5\nModel comparison\n(RF / MLP vs. GCN / GraphSAGE,\n>=5 seeds, MAE/RMSE/R2 + efficiency)"),
    ("s6", "Step 6\nExplainability\n(SHAP, embedding ablation,\nconfidence calibration)"),
    ("report", "Thesis report\n(reports/)"),
]

MERMAID_TEMPLATE = """graph TD
    s1["{s1}"] --> s5
    s2["{s2}"] --> s3a["{s3a}"]
    s2 --> s3b["{s3b}"]
    s2 --> s3c["{s3c}"]
    s3a --> s4["{s4}"]
    s3b --> s4
    s3c --> s4
    s4 --> s5["{s5}"]
    s5 --> s6["{s6}"]
    s6 --> report["{report}"]
"""


def main() -> None:
    labels = dict(STAGES)
    mermaid_source = MERMAID_TEMPLATE.format(**labels)

    os.makedirs("docs", exist_ok=True)
    out_path = "docs/architecture.mmd"
    with open(out_path, "w") as fh:
        fh.write(mermaid_source)
    print(f"Wrote {out_path}")
    print("Embed it in README.md as a ```mermaid fenced code block, or render to SVG with:")
    print("  npx -y @mermaid-js/mermaid-cli -i docs/architecture.mmd -o docs/architecture.svg")


if __name__ == "__main__":
    main()
