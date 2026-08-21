"""Copy the summary-level result files into reports/evidence/ so the report is checkable.

outputs/ is gitignored -- it is 1.2 GB of checkpoints, per-sample CSVs and logs, and it does
not belong in version control. But that left every number in the report and in
RESULTS_phase1.md pointing at a file nobody outside this machine can open, so nothing could
be verified from the published branch alone.

These are the summary-level files only: the per-fold transfer summaries the main table reads,
the component decompositions, the split-effect pairings, the stratification, the significance
and degenerate summaries, the convergence and v3 checks, and the African summaries. About
240 KB in total. The per-sample and per-gauge tables they were computed FROM stay in
outputs/, as does anything large.

Paths are preserved relative to outputs/, so a citation like
outputs/v2_split_effect/summary_M1.json resolves to
reports/evidence/v2_split_effect/summary_M1.json without a lookup table.

    python -m scripts.collect_evidence
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Ordered so the manifest reads as a table of contents rather than a directory listing.
PATTERNS = (
    ("main results, per fold", "*/fold*/transfer/summary.json"),
    ("pretrain provenance, per fold", "*/fold*/pretrain/run_meta.json"),
    ("KGE component decomposition", "*/diagnostics*/kge_components_summary_target.csv"),
    ("per-gauge attribution verdict", "*/diagnostics*/verdict_target.json"),
    ("stratified gain", "v2_stratify/*.csv"),
    ("stratified gain summary", "v2_stratify/*.json"),
    ("latitude bands", "v2_stratify/maps/by_latitude_target.csv"),
    ("random vs blocked pairing", "v2_split_effect/summary_M*.json"),
    ("pairing, by agency", "v2_split_effect/by_agency_M*.csv"),
    ("pairing, recovery", "v2_split_effect/recovery_by_agency.csv"),
    ("significance with FDR", "*/significance/significance_summary.json"),
    ("within-day shape", "*/degenerate/degenerate_summary.json"),
    ("convergence and selection loss", "convergence_check/*"),
    ("v3 convergence check", "v3_check/*.csv"),
    ("v3 convergence check summary", "v3_check/summary.json"),
    ("Africa, in situ, per fold", "*africa_insitu_summary/by_fold.csv"),
    ("Africa, in situ, summaries", "*africa_insitu_summary/*summary*.json"),
    ("Africa, temperate transfer", "africa_runB_*/africa_comparison_*.csv"),
    ("Africa, temperate transfer summaries", "africa_runB_*/africa_summary_*.json"),
)

MAX_BYTES = 512 * 1024  # anything larger is a per-gauge table, not a summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect summary results for publication.")
    parser.add_argument("--src", default="outputs", type=Path)
    parser.add_argument("--out-dir", default="reports/evidence", type=Path)
    args = parser.parse_args()

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    manifest, total, skipped = [], 0, []
    for label, pattern in PATTERNS:
        hits = sorted(args.src.glob(pattern))
        copied = 0
        for src in hits:
            if not src.is_file():
                continue
            if src.stat().st_size > MAX_BYTES:
                skipped.append((src, src.stat().st_size))
                continue
            rel = src.relative_to(args.src)
            dst = args.out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            total += src.stat().st_size
            copied += 1
        manifest.append((label, pattern, copied))
        print(f"  {label:38s} {copied:3d} files")

    lines = ["# Evidence files", "",
             "Summary-level results copied from `outputs/`, which is gitignored, so every "
             "number in", "`reports/PhaseI_report.docx` and `RESULTS_phase1.md` can be "
             "checked from this branch alone.", "",
             "Paths mirror `outputs/`, so a citation of "
             "`outputs/v2_split_effect/summary_M1.json`", "resolves to "
             "`reports/evidence/v2_split_effect/summary_M1.json`.", "",
             "Regenerate with `python -m scripts.collect_evidence`.", "",
             "| Contents | Pattern | Files |", "|---|---|---|"]
    lines += [f"| {label} | `{pattern}` | {n} |" for label, pattern, n in manifest]
    per_gauge = sorted(
        (f for f in args.out_dir.rglob("*") if f.is_file() and f.stat().st_size > 64 * 1024),
        key=lambda f: -f.stat().st_size)
    lines += ["", f"Total {total / 1024:.0f} KB.", "",
              "Mostly summary-level. The exceptions, kept because they are the direct "
              "evidence for a", "headline claim rather than a rollup of it:", ""]
    lines += [f"- `{f.relative_to(args.out_dir)}` ({f.stat().st_size / 1024:.0f} KB)"
              for f in per_gauge] or ["- none"]
    lines += ["", "Everything else those were computed from stays in `outputs/`: the "
                  "per-sample tables run to", "hundreds of megabytes and are regenerable "
                  "from the checkpoints."]
    if skipped:
        lines += ["", "Deliberately not copied, over the "
                      f"{MAX_BYTES // 1024} KB summary threshold:", ""]
        lines += [f"- `{p.relative_to(args.src)}` ({s / 1024:.0f} KB)" for p, s in skipped[:12]]
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out_dir} ({total / 1024:.0f} KB, {len(skipped)} oversized skipped)")


if __name__ == "__main__":
    main()
