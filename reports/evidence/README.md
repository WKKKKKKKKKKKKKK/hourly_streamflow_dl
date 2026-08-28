# Evidence files

Summary-level results copied from `outputs/`, which is gitignored, so every number in
`reports/PhaseI_report.docx` and `RESULTS_phase1.md` can be checked from this branch alone.

Paths mirror `outputs/`, so a citation of `outputs/v2_split_effect/summary_M1.json`
resolves to `reports/evidence/v2_split_effect/summary_M1.json`.

Regenerate with `python -m scripts.collect_evidence`.

| Contents | Pattern | Files |
|---|---|---|
| main results, per fold | `*/fold*/transfer/summary.json` | 70 |
| pretrain provenance, per fold | `*/fold*/pretrain/run_meta.json` | 45 |
| KGE component decomposition | `*/diagnostics*/kge_components_summary_target.csv` | 10 |
| per-gauge attribution verdict | `*/diagnostics*/verdict_target.json` | 10 |
| stratified gain | `v2_stratify/*.csv` | 3 |
| stratified gain summary | `v2_stratify/*.json` | 1 |
| latitude bands | `v2_stratify/maps/by_latitude_target.csv` | 1 |
| random vs blocked pairing | `v2_split_effect/summary_M*.json` | 2 |
| pairing, by agency | `v2_split_effect/by_agency_M*.csv` | 2 |
| pairing, recovery | `v2_split_effect/recovery_by_agency.csv` | 1 |
| significance with FDR | `*/significance/significance_summary.json` | 4 |
| within-day shape | `*/degenerate/degenerate_summary.json` | 4 |
| convergence and selection loss | `convergence_check/*` | 3 |
| fold-to-fold dispersion by split | `split_dispersion/*` | 2 |
| v3 convergence check | `v3_check/*.csv` | 3 |
| v3 convergence check summary | `v3_check/summary.json` | 1 |
| Africa, in situ, per fold | `*africa_insitu_summary/by_fold.csv` | 2 |
| Africa, in situ, summaries | `*africa_insitu_summary/*summary*.json` | 4 |
| Africa, temperate transfer | `africa_runB_*/africa_comparison_*.csv` | 7 |
| Africa, temperate transfer summaries | `africa_runB_*/africa_summary_*.json` | 7 |
| Africa, hourly within-day shape per basin | `v2_africa_hourly/within_day_cv_per_basin.csv` | 1 |
| Africa, hourly within-day summary | `v2_africa_hourly/within_day_summary.json` | 1 |
| Africa, hourly run log | `v2_africa_hourly/hourly_series.log` | 1 |
| Africa, three-way daily per basin | `v2_africa_hourly/daily_three_way_per_basin.csv` | 1 |
| Africa, three-way daily summary | `v2_africa_hourly/daily_three_way_summary.json` | 1 |
| Africa, three-way daily log | `v2_africa_hourly/daily_three_way.log` | 1 |

Total 1009 KB.

Mostly summary-level. The exceptions, kept because they are the direct evidence for a
headline claim rather than a rollup of it:

- `v3_check/paired_gap_per_station.csv` (505 KB)
- `v2_africa_hourly/daily_three_way_per_basin.csv` (89 KB)

Everything else those were computed from stays in `outputs/`: the per-sample tables run to
hundreds of megabytes and are regenerable from the checkpoints.

Deliberately not copied, over the 512 KB summary threshold:

- `v2_stratify/gain_with_covariates_target.csv` (4226 KB)
