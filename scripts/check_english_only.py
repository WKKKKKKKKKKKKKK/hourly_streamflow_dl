"""Fail if any tracked file carries CJK text, in its name, its contents, or its .docx XML.

The deliverable requirement is that nothing in the repository is in Chinese. Verifying that
by hand is where this went wrong repeatedly, in three distinct ways worth recording:

* ``git ls-files`` octal-escapes non-ASCII paths unless ``core.quotePath=false``, so a
  Chinese FILENAME is invisible to a regex over the listing. One such file was tracked and
  pushed before this was noticed.
* Scanning file contents alone misses commit messages and the font and style names buried
  in a .docx's ``styles.xml``, ``theme1.xml``, ``fontTable.xml`` and ``numbering.xml``.
* Writing the ranges as literal characters made the checker's own source non-ASCII and, in
  one revision, silently wrong: it flagged U+F0B7, which is a Private Use Area glyph that
  Word emits for Symbol-font bullets in every document with a bulleted list, and is not
  Chinese at all. The ranges below are numeric for that reason, and the Private Use Area
  is deliberately not among them.

    python -m scripts.check_english_only          # exits non-zero on any finding
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

# Numeric, so this file stays ASCII and the ranges cannot be mistyped as glyphs.
CJK_RANGES = (
    (0x2E80, 0x2EFF),   # CJK radicals supplement
    (0x3000, 0x303F),   # CJK symbols and punctuation
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0x3100, 0x312F),   # Bopomofo
    (0x3130, 0x318F),   # Hangul compatibility jamo
    (0x3190, 0x319F),   # Kanbun
    (0x31C0, 0x31EF),   # CJK strokes
    (0x31F0, 0x31FF),   # Katakana phonetic extensions
    (0x3200, 0x32FF),   # Enclosed CJK letters and months
    (0x3300, 0x33FF),   # CJK compatibility
    (0x3400, 0x4DBF),   # CJK unified ideographs extension A
    (0x4E00, 0x9FFF),   # CJK unified ideographs
    (0xA000, 0xA4CF),   # Yi
    (0xAC00, 0xD7AF),   # Hangul syllables
    (0xF900, 0xFAFF),   # CJK compatibility ideographs
    (0xFE30, 0xFE4F),   # CJK compatibility forms
    (0xFF00, 0xFFEF),   # Halfwidth and fullwidth forms
    (0x20000, 0x2FA1F),  # CJK extensions B-F and compatibility supplement
)
PATTERN = re.compile("[" + "".join(f"\\U{lo:08x}-\\U{hi:08x}" for lo, hi in CJK_RANGES) + "]")

# Parts of a .docx that carry human-readable text or font and style names.
DOCX_SUFFIXES = (".xml", ".rels")
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".gz", ".nc", ".npz", ".pth",
                   ".pt", ".gpkg", ".ico", ".zip"}


def tracked_files(repo: Path) -> list[str]:
    """Tracked paths, unescaped. core.quotePath=false is the whole point of this helper."""
    out = subprocess.run(["git", "-c", "core.quotePath=false", "ls-files"],
                         cwd=repo, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.split("\n") if line]


def commit_messages(repo: Path) -> list[tuple[str, str]]:
    out = subprocess.run(["git", "log", "--format=%H%x1f%B%x1e"],
                         cwd=repo, capture_output=True, text=True, check=True)
    pairs = []
    for record in out.stdout.split("\x1e"):
        if "\x1f" not in record:
            continue
        sha, body = record.split("\x1f", 1)
        pairs.append((sha.strip(), body))
    return pairs


def scan(repo: Path, check_history: bool) -> list[str]:
    findings = []
    files = tracked_files(repo)
    for name in files:
        if PATTERN.search(name):
            findings.append(f"filename: {name}")
        path = repo / name
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        if suffix == ".docx":
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if not member.endswith(DOCX_SUFFIXES):
                        continue
                    text = archive.read(member).decode("utf-8", "ignore")
                    match = PATTERN.search(text)
                    if match:
                        start = max(0, match.start() - 50)
                        findings.append(f"{name} -> {member}: ...{text[start:match.end() + 20]}")
            continue
        if suffix in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        match = PATTERN.search(text)
        if match:
            line = text[:match.start()].count("\n") + 1
            findings.append(f"{name}:{line}: {match.group()!r} (U+{ord(match.group()):04X})")

    if check_history:
        for sha, body in commit_messages(repo):
            match = PATTERN.search(body)
            if match:
                findings.append(f"commit {sha[:9]}: {body.splitlines()[0][:70]}")
    return findings, len(files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail if any tracked file or commit message carries CJK text.")
    parser.add_argument("--repo", default=".", type=Path)
    parser.add_argument("--skip-history", action="store_true",
                        help="Skip commit messages. They were translated once and rewritten "
                             "with filter-branch; scanning them is cheap, so this is only "
                             "for a shallow clone where the history is absent.")
    args = parser.parse_args()

    findings, n_files = scan(args.repo.resolve(), not args.skip_history)
    print(f"checked {n_files} tracked files"
          + ("" if args.skip_history else " and every commit message"))
    if not findings:
        print("no CJK found")
        return
    print(f"CJK found in {len(findings)} place(s):")
    for item in findings:
        print(f"  {item}")
    sys.exit(1)


if __name__ == "__main__":
    main()
