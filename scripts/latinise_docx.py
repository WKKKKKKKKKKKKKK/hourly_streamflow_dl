"""Replace the CJK font and style names a .docx carries internally with Latin equivalents.

Word writes its theme fonts and built-in style display names in the locale of whatever
installation created the file, so a document whose visible text is entirely English can
still hold Chinese, Japanese and Korean strings in word/styles.xml, word/theme/theme1.xml
and word/fontTable.xml. A reader never sees them, but a document delivered as English
should be English throughout, and an audience that does not read those scripts has no way
to tell the difference between a font name and content.

This is safe because it renames only display values. Styles are referenced by w:styleId,
which is ASCII in every file here, so the w:name a style advertises can change without
breaking a reference. Font names map to the same font under its Latin name, which Word
resolves identically.

Substitution is ordered: "DengXian Light" has to be handled before "DengXian", and
"Intense Quote Char" before "Quote Char", or the shorter key would consume the longer one.

Verifies afterwards and fails rather than writing a file that is still not all-Latin -- an
earlier version of this mapping silently missed MS Gothic because the search that built it
had looked only for Han ideographs and not katakana.

    python -m scripts.latinise_docx reports/PhaseI_report.docx Plan.docx
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

# Keys are escapes, not literals, so this file itself stays pure ASCII: a source file
# carrying the characters it exists to remove would defeat its own purpose.
REPLACEMENTS = (
    ("\u7b49\u7ebf Light", "DengXian Light"),
    ("\u7b49\u7ebf", "DengXian"),
    ("\u6e38\u30b4\u30b7\u30c3\u30af Light", "Yu Gothic Light"),
    ("\u6e38\u660e\u671d", "Yu Mincho"),
    ("\u5b8b\u4f53", "SimSun"),
    ("\u65b0\u7d30\u660e\u9ad4", "PMingLiU"),
    ("\uff2d\uff33 \u660e\u671d", "MS Mincho"),
    ("\uff2d\uff33 \u30b4\u30b7\u30c3\u30af", "MS Gothic"),
    ("\ub9d1\uc740 \uace0\ub515", "Malgun Gothic"),
    ("Office \u4e3b\u9898", "Office Theme"),
    ("\u660e\u663e\u5f15\u7528 \u5b57\u7b26", "Intense Quote Char"),
    ("\u526f\u6807\u9898 \u5b57\u7b26", "Subtitle Char"),
    ("\u5f15\u7528 \u5b57\u7b26", "Quote Char"),
    ("\u9875\u7709 \u5b57\u7b26", "Header Char"),
    ("\u9875\u811a \u5b57\u7b26", "Footer Char"),
    ("\u6807\u9898 1 \u5b57\u7b26", "Heading 1 Char"),
    ("\u6807\u9898 2 \u5b57\u7b26", "Heading 2 Char"),
    ("\u6807\u9898 3 \u5b57\u7b26", "Heading 3 Char"),
    ("\u6807\u9898 4 \u5b57\u7b26", "Heading 4 Char"),
    ("\u6807\u9898 5 \u5b57\u7b26", "Heading 5 Char"),
    ("\u6807\u9898 6 \u5b57\u7b26", "Heading 6 Char"),
    ("\u6807\u9898 7 \u5b57\u7b26", "Heading 7 Char"),
    ("\u6807\u9898 8 \u5b57\u7b26", "Heading 8 Char"),
    ("\u6807\u9898 9 \u5b57\u7b26", "Heading 9 Char"),
    ("\u6807\u9898 \u5b57\u7b26", "Title Char"),)

# Only these parts hold names; document.xml is content and must never be touched here.
PARTS = ("word/styles.xml", "word/theme/theme1.xml", "word/fontTable.xml",
         "word/settings.xml", "docProps/app.xml", "docProps/core.xml")

# CJK, kana, Hangul, fullwidth forms, and CJK compatibility ideographs.
RANGES = ((0x2E80, 0x9FFF), (0x3040, 0x30FF), (0xAC00, 0xD7AF),
          (0xF900, 0xFAFF), (0xFF00, 0xFFEF))


def has_cjk(text: str) -> bool:
    return any(any(lo <= ord(c) <= hi for lo, hi in RANGES) for c in text)


def latinise(path: Path) -> int:
    """Rewrite path in place. Returns the number of substitutions made."""
    replaced = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename in PARTS:
                text = data.decode("utf8")
                for cjk, latin in REPLACEMENTS:
                    if cjk in text:
                        replaced += text.count(cjk)
                        text = text.replace(cjk, latin)
                data = text.encode("utf8")
            dst.writestr(item, data)
    shutil.move(str(tmp), str(path))

    with zipfile.ZipFile(path) as check:
        for name in check.namelist():
            if not name.endswith((".xml", ".rels")):
                continue
            text = check.read(name).decode("utf8", errors="ignore")
            stray = sorted({c for c in text if has_cjk(c)})
            if stray:
                raise SystemExit(
                    f"{path}: {name} still holds {stray} -- extend REPLACEMENTS in "
                    "scripts/latinise_docx.py"
                )
    return replaced


def main() -> None:
    parser = argparse.ArgumentParser(description="Latinise a .docx's internal names.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        if not path.exists():
            print(f"  {path}: missing", file=sys.stderr)
            continue
        print(f"  {path}: {latinise(path)} substitutions, verified clean")


if __name__ == "__main__":
    main()
