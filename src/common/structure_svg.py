"""Small, theme-friendly RDKit structure drawings for the web console.

Strokes use `currentColor` so the drawing follows the page's text colour, and RDKit's verbose
header/class markup is stripped to keep each drawing a few KB.
"""

from __future__ import annotations

import re

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def structure_svg(smiles: str, width: int = 150, height: int = 110) -> str:
    """Returns an SVG string for `smiles`, or "" when it doesn't parse to a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = re.sub(r"<\?xml.*?\?>|<!--.*?-->|<rect.*?/>", "", svg, flags=re.S)
    svg = re.sub(r"\s+(xmlns:\w+|xml:space|baseProfile|version)=('[^']*'|\"[^\"]*\")", "", svg)
    svg = re.sub(r" class='[^']*'", "", svg)
    svg = svg.replace("stroke:#000000", "stroke:currentColor").replace("fill:#000000", "fill:currentColor")
    svg = svg.replace(
        "fill:none;fill-rule:evenodd;stroke:currentColor;stroke-width:2.0px;"
        "stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1",
        "fill:none;stroke:currentColor;stroke-width:2",
    )
    return re.sub(r"\s*\n\s*", " ", svg).strip()
