"""Pack the LaTeX report and its figures into one archive that compiles anywhere.

The .tex alone is not a deliverable. LaTeX treats a missing figure as a WARNING, not an
error, so a document uploaded without its PNGs still produces a PDF -- one with eleven
blank boxes and no indication that anything went wrong. This builds an archive whose
layout is flat, which is what Overleaf's "upload project" expects, and the document's
\\graphicspath covers the flat case as well as the repository one.

    python -m scripts.bundle_latex
    # -> reports/latex/PhaseI_report_bundle.zip, upload it whole to Overleaf
"""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle the LaTeX report with its figures.")
    parser.add_argument("--tex", default="reports/latex/PhaseI_report.tex", type=Path)
    parser.add_argument("--figdir", default="reports/figures", type=Path)
    parser.add_argument("--out", default="reports/latex/PhaseI_report_bundle.zip", type=Path)
    args = parser.parse_args()

    if not args.tex.exists():
        raise SystemExit(f"{args.tex} missing -- run python -m scripts.build_latex first")
    source = args.tex.read_text(encoding="utf-8")
    # Take the figure list from the document itself rather than from a glob, so a figure
    # the report does not use is not shipped and one it does use cannot be forgotten.
    wanted = sorted(set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", source)))
    missing = [n for n in wanted if not (args.figdir / n).exists()]
    if missing:
        raise SystemExit(f"the document references figures that do not exist: {missing}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.tex, args.tex.name)
        for name in wanted:
            archive.write(args.figdir / name, name)
        archive.writestr("README.txt", (
            "Phase I report -- LaTeX source and figures.\n\n"
            "Everything is flat: the .tex and all PNGs sit in one directory, which is the\n"
            "layout Overleaf's project upload expects. The document's \\graphicspath also\n"
            "covers ../figures/ and figures/, so it compiles unchanged inside the repository.\n\n"
            "To build:  pdflatex PhaseI_report.tex   (twice, so the table of contents fills in)\n\n"
            "If a figure is missing, LaTeX only warns and leaves a blank box, so check the\n"
            "log for 'not found' rather than trusting that a PDF appeared.\n"))

    size = args.out.stat().st_size / 1024**2
    print(f"wrote {args.out} ({size:.1f} MB): 1 tex + {len(wanted)} figures")
    for name in wanted:
        print(f"  {name}")


if __name__ == "__main__":
    main()
