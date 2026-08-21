# Global MTS-LSTM for hourly streamflow — experiment plan

> **This is the plan as written before execution, kept for the record.** Several parts were
> followed exactly, several were deliberately changed, and a few turned out to be wrong.
> Where a decision here differs from what was actually run, `RESULTS_phase1.md` is
> authoritative and records the deviation and its reason. The larger departures:
>
> - **Temporal split.** Planned three periods (train 2000–2012 / val 2013–2015 / test
>   2016–2020); executed as a two-way 70/30 split per gauge on its own record, because the
>   prepared batches are already split that way and run B had to stay comparable with them.
>   The consequence is bounded at about 0.006 KGE and is documented as a limitation.
> - **Gauge count.** Planned 8,854 after quality control; 8,990 entered the folds and 9,181
>   are in the cache.
> - **The M2 symbolic prior** was not ported. Reasons are in `RESULTS_phase1.md`; the short
>   version is that its expressions were fitted on CAMELS-US attributes whose global
>   counterparts differ in scale by 2–400x, and it addresses daily-branch bias where the
>   defect actually diagnosed is an hourly variance ratio.
> - **Peak-timing analysis** (§5) was never run. It remains a gap, and §5 of this plan
>   understates how easy it is to do badly.
> - **A forget-gate initialisation** that this plan does not mention was later found missing
>   against Gauch et al.'s published method, and became the main difference between the v1
>   and v2 configurations.

---

## 1. What the experiment is

```
10,423 hourly gauges
   ├── source domain 80% (~8,338 gauges)  ── hourly observations available
   └── target domain 20% (~2,085 gauges)  ── hourly observations HIDDEN, only 24-h
                                             aggregates exposed; the real hourly series
                                             is used for final scoring only

Stage 1 (pretrain)   train MTS-LSTM on the source 80% with hourly observations
                     loss: NSE_h(H, Q_h^obs) + λ_reg·(D − mean_24h(H_seq))²
                     → source model M_src

Stage 2 (transfer)   fine-tune on the target 20% using daily aggregates only
                     loss: NSE_d(D, y_d) + w_agg·NSE_d(mean_24h(H_seq), y_d)
                     where y_d = y_h[t−24:t].mean(); the hourly y_t never enters the loss
                     freeze lstm_hourly; train lstm_daily / transfer_h / transfer_c / head_*
                     → M_transfer

Stage 3 (evaluate)   emit hourly predictions over the target 20%'s test period
                     score per-gauge KGE/NSE against the real hourly series, take the median
```

This loss and freezing strategy is already implemented and validated in
`transfer_daily_to_hourly_partial_ft_s2_random30.py` (30 gauges, test KGE 0.489). The global
version **changes the scale and the station split, not the method**.

### The controls that have to be there

A single M_transfer number proves nothing. At least two controls are needed:

| Label | What it is | What it answers |
|---|---|---|
| **M0** | M_src applied to the target 20% with no fine-tuning at all (zero-shot) | the baseline: how much did daily-aggregate supervision actually buy |
| **M1** | M_transfer, the product of stage 2 | the main result |
| **M2** (optional) | M1 plus the symbolic prior (following `..._symbolic_hybrid.py`; 0.489 → 0.506 on 30 gauges) | continues an existing line of work |

The headline conclusion is the **per-gauge paired difference M1 − M0**, tested with
Wilcoxon signed-rank plus BH-FDR (the script exists:
`run_s2_threeway_significance_tests.py`).

---

## 2. Data and splits

Data root `DB = /ibex/project/c2266/abbaa0a/data/gscad_database/processed/20250630`

Only the hourly archive (`DB/hourly/`) is needed. **The daily archive is not used at all** —
daily observations are the 24-hour mean of the hourly ones.

| Item | Content |
|---|---|
| target | `hourly/dataframes/hourly_q.nc`, `q(time=895608, stations=10423)`, 1922–2024, m³/s |
| forcing | `hourly/dataframes/` ERA5-HRES, 7 variables: P / Temp / SWd / LWd / Pres / RelHum / Wind (MSWEP also available for a product comparison) |
| static | `hourly/dataframes/static.csv`, 55 columns |
| gauge sources | CAMELSH (US) 5767, BOMAustralia 2059, LamaHCE 859, Japan 696, Germany 494, Czech 437, LamaHIce 111 |

### Quality control
- `years_q_valid >= 10` → **8,854 gauges usable** (from
  `EDA_global_hourly_runoff/tables/basin_features.csv`, already computed)
- Heavily regulated gauges (high `reservoir_impact_GRanD_v1_3`) are **not** dropped; they are
  flagged as a separate evaluation subset instead.

### Temporal split
A single window, **2000-01-01 → 2020-12-31** (where hourly Q is densest):
- train 2000–2012 / val 2013–2015 / test 2016–2020
- windows are built independently inside each split, and the first 365 days are burn-in
  (`t_min = lookback_daily × 24 = 8760`) producing no samples, so nothing leaks across splits

### Station split: 80/20 IS 5-fold, so run all five folds

**80/20 is the right ratio, but do not run only one fold.** 80/20 is exactly 5-fold CV:
rotate five times and **every gauge serves as a target gauge exactly once**, which yields
hourly KGE for all 8,854 gauges rather than for one fifth of them. That buys three things:

- no sampling luck in the global map — every gauge has a real evaluated value
- 5x the sample size for any analysis stratified by climate zone, continent or flashiness
- it closes off the "you got lucky with the split" objection

**The cost is low.** The hyper-parameter search runs **once, on fold 1**, and the other four
folds reuse those hyper-parameters — searching per fold would be both expensive and a source
of between-fold selection leakage. The five pretraining runs are independent, so a SLURM
array makes the wall-clock cost about that of one pretraining run.

Concretely (8,854 gauges after QC): about **1,771** target and **7,083** source gauges per
fold.

**Why not 90/10 or 50/50**

| Ratio | Source | Target | Assessment |
|---|---|---|---|
| 90/10 (10 folds) | 7,969 | 885 | doubles the pretraining runs; halves the daily data available per fold for fine-tuning, so M1 could be worse |
| **80/20 (5 folds)** | **7,083** | **1,771** | **recommended** |
| 70/30 (3.3 folds) | 6,198 | 2,656 | workable, but a non-integer fold count is awkward to organise |
| 50/50 (2 folds) | 4,427 | 4,427 | only two pretraining runs, but "half the world has no hourly data" is an unnatural premise |

The judgement behind this: **source-domain size is no longer the binding constraint at this
scale.** Regional LSTM performance saturates in the low hundreds of catchments (Kratzert's
PUB results), so 7,083 against 4,427 makes little difference. The ratio is therefore decided
by what the target domain is FOR, and target size affects two things: (a) statistical power,
where 30 gauges already sufficed in the earlier experiment and 1,771 is far more than
enough; (b) **the amount of daily data available for stage-2 fine-tuning**, which is a hidden
variable and means the per-fold target count must be reported. On balance, five folds is the
least trouble.

### The split method matters far more than the ratio

CAMELSH contributes 5,767 US gauges, so under purely random splitting a target gauge is
**almost guaranteed a hydrological neighbour in the source domain** (spatial
autocorrelation), and the result will be systematically optimistic. So run two splits; they
are two rungs of difficulty:

| Split | How | Question it answers |
|---|---|---|
| **CV-random** (primary) | stratified random 5-fold (by KGZ_major × source agency × `years_q_valid` quantile, so each fold matches on climate and region) | "**within a region**, can daily data substitute for hourly data" |
| **CV-blocked** (secondary) | spatially blocked 5-fold: the 8 clusters from `basin_embedding.csv`, or geographic k-means, so a target gauge has no neighbour in the source domain | "does it still work when **an entire region** has no hourly data" |

**The difference between the two is itself a headline result** — it quantifies how much of
the transfer skill comes from geographic proximity and how much from genuine
attribute-based generalisation. No code changes; only the split table changes.

Fix the seed and write the splits to `folds_random.csv` / `folds_blocked.csv` (columns
`station_id, fold`), shared by every experiment.

---

## 3. Code that has to change

The method is unchanged; the changes are all about **scale**. The existing code runs at 30
gauges and will fall over at 8,854.

### 3.1 Sample enumeration and data loading (the bulk of the work)

`TransferTargetDataset.__init__` and `MultiscaleLSTMDataset.__init__` currently enumerate
samples with a Python loop over time steps:

```python
for t in range(t_min, len(x)):
    x_h = x[t-168:t]; x_d_full = x[t-8760:t]
    if np.isnan(x_h).any() or np.isnan(x_d_full).any() or ...: continue
```

At 8,854 gauges × 184k time steps that is about **1.6×10⁹ iterations**, each slicing and
scanning 8,760 points for NaN. Required changes:

- **NetCDF → per-gauge chunked zarr / memmap cache** (`chunks=(1 station, 8760 h)`, float32).
  Size: 4 variables × 184k hours × 8,854 gauges × 4 B ≈ **26 GB**, on `/ibex/scratch`.
- **Vectorised validity test**: compute, for all windows at once, whether they contain NaN,
  using a cumulative sum of the validity mask and its difference — no per-step slicing.
  Store the result as an int32 `(station_idx, t_idx)` array on disk and compute it once.
- **stride = 24** (one sample per day, as in Gauch et al.), or the sample count reaches 10⁹.
- **Random subsampling per epoch plus gauge-balanced sampling** — CAMELSH is 55% of the
  archive and would otherwise dominate the loss.
- The conversion script can copy the h5py + timestamp-alignment + chunked-scatter pattern
  from `EDA_global_hourly_runoff/scripts/compute_features.py`, which is already known to
  handle 10,423 × 400k inside 40 GB.

### 3.2 Four bugs that must be fixed

1. **The forcing files' time axis is not monotonically increasing.**
   `ERA5_HRES_hourly_P_hourly.nc` reports first and last timestamps of 1979-09→2002-10 while
   holding 407,591 steps; intersecting timestamps with Q gives 403,247 hours
   (1979-01-01→2024-12-31). **Align on timestamps, never by position**, and reorder gauges by
   name, since the station order differs between files. (`compute_features.py`'s
   `get_indexer(common)` does this correctly — copy it.)

2. **`handle_extremes(max_streamflow=1000)`** (`Train.py:367`, hard-coded at L91 of the
   transfer script) sets anything above 1000 m³/s to NaN. Harmless across 100 US gauges; in
   the global set, catchments reach 26,000 km² and a single gauge's q95 is already 555 m³/s
   with q99 at 2,859. **That cap silently deletes a large number of real flood peaks — the
   very events of interest.** Convert to mm/h first (`q_mm_h = q_cms × 3.6 / area_km2`), then
   quality-control by per-basin quantile plus a physical bound.

3. **Unit normalisation.** q is m³/s and global catchment area spans four orders of
   magnitude, so it must become mm/h for cross-catchment training to converge. (`NSELoss`'s
   per-gauge std normalisation fixes the loss scale, not the target's units.)

4. **Model-selection leakage — must be fixed, and it also affects how the existing 30-gauge
   result should be described.** `transfer_daily_to_hourly_partial_ft_s2_random30.py`
   L418–428 selects the best epoch, and early-stops, on **`val_hourly_kge`**:

   ```python
   val_hourly = evaluate_hourly_per_station(model, loaders["val"], ...)
   val_kge = float(val_summary["median_kge"])          # target gauges' HOURLY KGE
   if val_kge > best_val_kge: best_state = ...          # used to pick the epoch
   ```

   The training loss uses daily aggregates only, which is correct, but **the epoch is chosen
   using the target gauges' hourly observations** — impossible under the premise that those
   gauges have no hourly data. It does not change the direction of the conclusion, but it
   makes the reported KGE optimistic, and it is the first thing a reviewer will check.

   **Fix:** early stopping and epoch selection move to a **daily-aggregate validation
   metric** (KGE of `mean_24h(H_seq)` against `y_d`, which the premise does allow); hourly
   KGE is computed once, at the end, on the test period.

   **And run a robustness check alongside:** record the test hourly KGE under both
   selection rules. If the difference is small, the existing 30-gauge results are unaffected
   and that can be stated outright in the write-up.

> Checked and confirmed **not** to need changes: the target does not leak into the inputs
> (`dyn.sel(dynamic_forcing=cfg.dynamic_vars)`); scalers are fitted on the training period
> only; KGE/NSE are computed in physical space after de-standardising; the temporal split is
> strictly ordered with no overlap. The 100-gauge experiment is correct on all four.
>
> Calendar-day alignment **is not an issue in this design** — the daily label
> `y_d = y[t−24:t].mean()` carries the same offset as the daily branch's input window, so it
> is self-consistent. (It would need handling only if `daily_q.nc` were read instead.)

### 3.3 Smaller changes

- `outputs["D"]` takes only the last day. It could instead compute a sequence-to-sequence
  daily loss over the whole `d_seq`, taking the daily supervision signal per sample from 1 to
  365 and improving sample efficiency considerably.
- The hourly archive has no PET variable; derive it from ERA5's Temp / SWd / RelHum / Wind
  (Priestley-Taylor or Hargreaves).
- Static attributes: choose 27–35 of the 55 columns, covering area, slope, elevation, KGZ
  one-hot, soil, land cover, snowfall fraction, reservoir impact, RGI glacier. **Be wary of
  the GDP / HDI / population columns** — the model would learn socio-economic proxies rather
  than hydrological mechanism.

---

## 4. Execution steps

| Step | Content | Output | Estimate |
|---|---|---|---|
| **S0** | quality control plus stratified 5-fold splits (random and blocked) | `folds_random.csv` / `folds_blocked.csv` | 1–2 days |
| **S1** | NetCDF → zarr cache, vectorised sample index, new Dataset | `data/build_cache.py`, `data/dataset.py` | 1–2 weeks |
| **S2** | 200-gauge smoke test through stages 1→2→3 | same order of magnitude as the 30-gauge result | 2–3 days |
| **S3** | 1,000 gauges, confirm convergence and that memory and I/O hold | fixes batch size, workers, look-back | 3–5 days |
| **S4** | hyper-parameter search on fold 1 to fix the architecture, then pretrain all five folds (SLURM array) | five `M_src` | 1–2 weeks including queueing |
| **S5** | stages 2 and 3 per fold: target transfer plus M0 / M1 / (M2) evaluation, pooled to all 8,854 gauges | per-gauge metric CSVs | 3–5 days |
| **S6** | significance testing, figures, writing | figures and tables | 1–2 weeks |

The hyper-parameter search reuses the existing SLURM array framework
(`tuning/submit_*.sbatch` + `make_grid.py` + `arrayrun_train.py` + a wandb sweep) with only
the data path changed. The space narrows from the 100-gauge optimum `idx2`:
`hidden_size_{d,h} ∈ {64,128,256}`, `lookback_hourly ∈ {168,336}`, `lookback_daily = 365`,
`dropout ∈ {0.2,0.4}`, `lr` keeping the piecewise schedule, `w_agg ∈ {0.25,0.5,1.0}`.

> The EDA supports `lookback_hourly = 168`: median best_lag is 17 h and P90 about 77 h. 336
> is there for slow-response and snowmelt catchments.

---

## 5. Evaluation and analysis

Headline metric: **median per-gauge hourly KGE / NSE over the target 20%'s test period**.

Scripts that transfer directly, needing only a new data path:

- `evaluate_s2_random30_alt_threeway.py` → the M0 / M1 / M2 three-way comparison
- `run_s2_threeway_significance_tests.py` → per-gauge paired Wilcoxon plus BH-FDR
- `plot_three_method_peak_lag_cdfs.py` → CDF of peak-timing error
- `evaluate_transfer_on_source_domain.py` and
  `analyze_source_domain_transfer_degradation.py` → source-domain forgetting (how far M_src
  degrades on the source 80% after stage-2 fine-tuning)

**Analyses specific to the global scale** — this is the incremental value of going global
rather than staying at 30 gauges:

1. **Spatial and climatic stratification of the gain.** `M1 − M0` against KGZ climate zone,
   continent, `area`, `years_q_valid`. Answers: in what kind of catchment is daily-aggregate
   supervision most useful.
2. **Gain against source-neighbour density.** Distance from each target gauge to its nearest
   source gauge, and the density of source gauges around it. This quantifies directly how
   much easier random splitting is, and predicts how much a whole-block hold-out will cost.
3. **Gain against EDA hydrological features.** Stratify by `best_lag`, `flashiness`,
   `max_lag_corr` from `basin_features.csv` and the 8 clusters in `basin_embedding.csv`.
   Gauges with low `max_lag_corr` (snowmelt- or storage-dominated) are predicted to gain
   least.
4. **Degenerate-solution diagnosis.** `loss_agg` constrains only the 24-hour mean and says
   nothing about the within-day distribution, so a degenerate solution exists: emit a flat
   line each day, perfect on the aggregate and meaningless hourly. Run the same
   `compute_features.py` over M1's predictions to get flashiness and Q95 event counts, and
   compare against the target gauges' **real observations**. The real hourly values are
   available here, so this can be measured directly rather than through a proxy.
5. **Global map** of hourly KGE across the target 20%.

---

## 6. Repository layout

```
global_mtslstm/
  PLAN.md
  configs/     exp_global.yaml (station split, time window, w_agg, hyper-parameters)
  data/
    select_stations.py    # S0 quality control plus the stratified 20% split
    build_cache.py        # S1 NetCDF → zarr
    dataset.py            # vectorised sample index plus the new Dataset
    units.py              # cms → mm/h, PET
  models/  Modelzoo.py  losses.py     # copied from MTSLSTM_100stations/code
  train/   pretrain_source.py  transfer_target.py  (tuning/ SLURM)
  eval/    analysis scripts ported from hourly_streamflow_dl
  outputs/
```

---

## 7. Principal risks

| Risk | Mitigation |
|---|---|
| random-read I/O over 8,854 gauges becomes the bottleneck | per-gauge chunked zarr plus per-gauge grouped sampling each epoch to cut cross-gauge random reads; 26 GB fits entirely in RAM on a large-memory node |
| gauge imbalance (CAMELSH 55%) | balanced sampling by gauge and region, plus the per-gauge normalised NSE loss |
| full-scale pretraining fails to converge | two-stage smoke test (S2, S3) fixes hyper-parameters first; start from the 100-gauge optimum rather than searching from scratch |
| the random 20% split is too easy and the conclusion is challenged | analysis #2 quantifies the neighbour effect up front; add a whole-block hold-out (no code changes needed) |
| stage-2 fine-tuning causes source-domain forgetting | quantify with the existing degradation scripts; if necessary, mix a small number of source hourly batches into stage 2 |

---

## 8. Open questions

1. **Is the 20% split random by gauge, or random plus a whole-block hold-out?** Suggested:
   finish the random version first, then the blocked hold-out as a second set (no code
   changes, only the split table).
2. **Include the symbolic prior (M2) this time?** Leaving it out makes the chain shorter and
   the result faster; including it continues an existing line of work.
3. **Scale for S4:** the full source 80% (~8,338 gauges, about 7,000 after QC) straight away,
   or a 2,000-gauge version first to confirm the conclusion is stable?
