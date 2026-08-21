# global_mtslstm — Phase I

Global-scale version of the `hourly_streamflow_dl / MTSLSTM_100stations` experiment,
following `Plan.docx` **Phase I (LSTM defender)** and the design notes in [PLAN.md](PLAN.md).

> This branch is self-contained and shares no history with `main`. `main` holds the
> 100-gauge experiment this work extends; the branch holds the global-scale Phase I project
> as its own tree, so nothing here overwrites or depends on `main`.

The premise: pretend 20% of the world's gauges have **no hourly** streamflow observations,
only 24-hour aggregates. Can daily-only supervision recover hourly skill at those gauges?

| Plan step | What runs | Output |
|---|---|---|
| Step 1 | `train.pretrain_source` — MTS-LSTM on the 80% source domain (hourly targets), then the same checkpoint scored on the 20% target domain | **M0**, zero-shot target hourly KGE/NSE |
| Step 2 | `train.transfer_target` — fine-tune on the target domain using **only** daily aggregates, early-stopped on daily KGE | **M1**, target hourly KGE/NSE |
| Step 3 | same script — M_src vs M_transfer on the source domain, paired per station | source-domain degradation |
| Step 4 | `scripts.aggregate_folds` — pool all 5 folds; `eval.africa` — the 5 models on African daily data vs prior work | every station scored once, Wilcoxon + BH-FDR; African daily KGE/NSE |
| Step 5 | `eval.africa` — same run, adding ERA5-Land daily runoff as the physical baseline | three-way comparison table |

**Status: all five steps have run at full scale and Phase I is complete.** Three
configurations were executed across 5 folds each (see below), 9,181 gauges are cached and
8,990 enter the folds, the African arm ran both as a cross-continent extrapolation test and
as an in-situ fine-tune, and every figure quoted anywhere in this repository is read from a
result file. The write-up is [reports/PhaseI_report.docx](reports/PhaseI_report.docx); the
full record, including the conclusions that were retracted along the way, is
[RESULTS_phase1.md](RESULTS_phase1.md).

Where to start, depending on what you want:

| You want | Read |
|---|---|
| the results and what they mean | `reports/PhaseI_report.docx` — 8 chapters plus an appendix |
| every number with its provenance and status | `RESULTS_phase1.md` |
| which config produced which output | `python -m scripts.inventory` |
| how to re-run any of it | the Reproducing section of `RESULTS_phase1.md` |
| the figures on their own | [reports/figures/](reports/figures/) — nine PNGs, regenerate with `python -m scripts.make_figures` |
| the file behind any number | [reports/evidence/](reports/evidence/) — mirrors `outputs/` paths, so a citation resolves directly |
| what was planned versus what was done | [PLAN.md](PLAN.md), whose header lists the deviations |

---

## Three configurations, and which one to read

Results are organised as v1 / v2 / v3. They differ in three settings and nothing else;
`python -m scripts.inventory` prints the full mapping from config to output directory.

| | hourly look-back | forget-gate init | epochs | patience | role |
|---|---|---|---|---|---|
| v1 | 72 | none | 30 | 6 | original setting |
| **v2** | **336** (168 for run A) | **3** | 30 | 6 | **primary result** |
| v3 | 336 | 3 | 50 | 10 | convergence check, kept out of the main table |

The forget-gate initialisation is part of Gauch et al.'s published method and was missing
from both this code and the 100-station reference; it was added for methodological
fidelity, not because it was shown to help on its own. It arrived together with the longer
look-back, so v1 -> v2 cannot be attributed to either alone.

**The v1 config files no longer reproduce v1.** `initial_forget_bias: 3` was added to them
when the forget gate was implemented, after those runs had finished, and no snapshot was
kept — re-running them today gives a 72-hour look-back with a v2 forget gate, a
combination that was never evaluated. The authority for what any run used is that run's
own `outputs/<run>/fold*/pretrain/run_meta.json`; `scripts.inventory` reads from there and
flags every config that has drifted.

Numbers, findings and their status live in `RESULTS_phase1.md`; the write-up is
`reports/PhaseI_report.docx`.

## Quick start

```bash
cd ~/global_mtslstm

# 1. one-off: index 7.1M batch files, build the 5-fold split, verify the chain
python -m scripts.build_index                       # ~10-20 min
python -m scripts.make_folds --require-both-splits
python -m scripts.smoke_test                        # ~1 min, safe on a login node

# 2. optional but recommended: per-station static attributes (lat/long/area/KGZ/...)
sbatch slurm/00_build_index.sbatch                  # or run the three above as one job
python -m scripts.build_station_table --workers 8   # ~55 GB read, run on a compute node

# 3. see which experiment is which, and what each run actually used
python -m scripts.inventory

# 4. train. NOTE: a pretrain is ~9 h but the scripts declare 4 h on purpose -- that is
#    the gpu4 partition boundary (95 nodes vs 46/31/58), and crossing it cost days of
#    queue time. So CHAIN the pretrain; a single job silently truncates the run.
CFG=configs/phase1_runB_v2.yaml
H=$(CONFIG=$CFG sbatch --parsable slurm/10_pretrain_runB.sbatch)
for i in 1 2; do H=$(CONFIG=$CFG sbatch --parsable --dependency=afterany:$H slurm/10_pretrain_runB.sbatch); done
CONFIG=$CFG sbatch --dependency=afterok:$H slurm/20_transfer_runB.sbatch

# 4. pool
python -m scripts.aggregate_folds
```

### Integration test (no GPU needed, ~5 min)

Runs the entire chain — pretrain → transfer → M0/M1 → degradation → pooling — on a
30-station fold table, so a broken assumption shows up before a GPU job is queued:

```bash
python -m scripts.make_folds --require-both-splits --sample-per-source 5 --out folds/folds_itest.csv
ITEST="--set folds.file=folds/folds_itest.csv train.epochs=2 train.batches_per_epoch=3 \
  train.val_max_stations=4 train.val_batches_per_station=1 transfer.epochs=2 \
  transfer.batches_per_epoch=3 transfer.holdout_batches_per_station=1 transfer.patience=2 \
  eval.max_batches_per_station=2 eval.min_samples_per_station=1 model.hidden_size_daily=16 \
  model.hidden_size_hourly=16 train.num_workers=2 transfer.num_workers=2 wandb.mode=disabled \
  output_root=outputs/_itest"
python -m train.pretrain_source  --fold 0 $ITEST
python -m train.transfer_target  --fold 0 --peek-hourly $ITEST
python -m scripts.aggregate_folds --set folds.file=folds/folds_itest.csv output_root=outputs/_itest \
                                  --out outputs/_itest/phase1_summary
```

Everything reads [`configs/phase1.yaml`](configs/phase1.yaml). Override any leaf inline:

```bash
python -m train.pretrain_source --fold 0 --set train.epochs=3 train.batches_per_epoch=200 \
                                             model.hidden_size_daily=64 wandb.mode=offline
```

---

## Data

`/ibex/project/c2266/abbaa0a/data/input_data/hourly_q_dl/6datasets_H_8760_512_..._9181_42`
— pre-batched, pre-standardized, **14 TB**.

* one batch = three files (`_x.pt`, `_y.pt`, `_metadata.pkl`), 512 samples
* `training/` 1,635,587 batches · `validation/` 748,755 batches
* `x = (x_dyn [512,1000,3], x_static [512,42])`, `y = [512,1]`
* dynamic features `pet, pcp, temp`; target `q_mm`, standardized with `scalers.json`
* split is **temporal + local**: each station's own record, first 70% → `training/`, last 30% → `validation/`, with a lookback-warmup gap and no overlap

### The 1000-step sequence and the two branches

Each sample stores **one** 1000-step sequence covering the previous 8760 hours,
power-law subsampled (`ts_func2`, γ=5.0): 1-hour spacing over the most recent **228**
positions, widening to ~29 hours at the far end. The exact offsets are recorded in
[`data/lookback_offsets.json`](data/lookback_offsets.json) (verified against `6sources.nc`;
identical for every sample and every station).

The two MTS-LSTM branches are a **split of that one sequence along time** —
the convention already used by the reference disk loader:

```
D (low-frequency, long range) = all 1000 steps
H (high-frequency, recent)    = the last lookback_hourly steps  (default 168)
frequency_factor = 1  ->  transfer index = len(D) - len(H) = 832
```

Because the tail is truly 1-hourly, `H_seq[:, -24:]` is exactly the last 24 hours and its
mean is the model's predicted daily aggregate — which is what Step 2 supervises.

> **Keep `lookback_hourly <= 228`.** Beyond that the hourly branch starts mixing spacings
> and the 24-step mean stops being a day. `scripts.smoke_test` warns if you cross it.

### Two batch filename forms

* `{source}__{id}_w{worker}_{batch}_*` — 512 consecutive hours from **one** station.
* `corrected_{n}_*` — 8,409 training + 4,216 validation batches with **no station in the
  name**, each mixing 2–3 stations (leftover per-station tails regrouped into full batches).
  `data/index.py` reads their metadata once and caches the membership; the dataset then
  masks out rows belonging to the other domain. Set `data.include_corrected: false` to drop
  them (~0.5% of samples).

### Daily targets — and why `min_daily_hours` matters

Rows inside a batch are **not** reliably consecutive hours: hours whose sample was dropped
during preparation are simply absent. Measured over 500 random training batches
(`python -m scripts.daily_coverage`):

| source | median observed hours per 24 h window |
|---|---|
| CAMELSH, Germany, Japan, LamaHCE, LamaHIce | 24 (complete) |
| BOMAustralia | 16 (min 3) |

So instead of demanding a gap-free day, each row carries a 24-slot occupancy mask and

```
y_daily[i]  = mean of the observed y in the 24 h ending at row i
prediction  = mean of H_seq[i, -24:] over exactly the same slots
```

Averaging both sides over identical slots keeps the target unbiased — a partial day is
compared like for like, not against a full-day prediction.

`transfer.min_daily_hours` (default **18**) drops rows with too few observed hours. It is a
real experimental knob, not a technicality: at a threshold of 1–2 the "daily aggregate" is
just an hourly observation wearing a hat, which would break the premise that the target
domain has no hourly data.

| threshold | rows kept | stations with any data |
|---|---|---|
| 12 | 92.8% | 99.8% |
| **18** | **89.0%** | **98.6%** |
| 24 (strict full day) | 84.6% | 94.6% |

Hourly observations of the target domain never enter the Step-2 loss — only this aggregate
does.

### Sizing: an epoch is a *sample* of the pool

The source pool is ~1.3M batch files (~8 TB). Reading it once per epoch is not viable, and
neither is a full evaluation pass (a 1,800-station domain has ~150k validation batches,
~930 GB). Three knobs control this, and **all three should be recalibrated once you see the
wall-clock of the first real run**:

**Measured**, V100-SXM2-32GB, `scripts/benchmark_gpu.py` (job 49948056):

```
train step  0.067 s/batch      eval step (forward only)  0.025 s/batch
IO alone    0.007 s/batch  ->  923 MiB/s: GPU-bound with 10x headroom
peak GPU memory  3.2 / 6.2 / 5.2 GiB at hidden 64 / 128 / 256, out of 32 GiB
```

Reading 6 MB batch files off Lustre is **not** the constraint — the opposite of what
the first guess assumed — so an epoch is sized by GPU time, not I/O:

| knob | value | meaning |
|---|---|---|
| `train.batches_per_epoch` | 20000 | 22.4 min/epoch; 30 epochs shows the model 307M of the 671M source samples (46%) |
| `train.val_max_stations` × `val_batches_per_station` | 0 (=all 7191) × 1 | **fixed** early-stopping set; 3.0 min/epoch, so every source station gets a vote |
| `eval.max_batches_per_station` | 12 | final reporting → ~6,100 samples/station |

That is 25.4 min/epoch → **12.7 h per fold**, against `--time=18:00:00`. The first,
uncalibrated guess would have used 2.7 h of a 24 h allocation and shown the model only 9%
of the data.

Why the validation set is fixed and station-*balanced* rather than a random draw: a batch
holds one station, so 500 random batches would cover 500 of 7,192 stations and a different
500 each epoch — the median-across-stations KGE would jump around for reasons unrelated to
the model, and early stopping would chase noise.

---

## Layout

```
configs/phase1*.yaml    13 experiment configs; `python -m scripts.inventory` maps each to its output
common/                 config loading, seeding, W&B, KGE/NSE, per-station accumulation
data/index.py           one-off scan of 7.1M files -> cached batch index
data/folds.py           stratified 5-fold station split (source 80% / target 20%)
data/dataset.py         batch-file dataset, D/H split, daily-target construction
data/lookback_offsets.json   hours-ago for each of the 1000 positions
models/mtslstm.py       sMTSLSTM (two branches + state transfer)
models/losses.py        basin-averaged NSE, MTS regulariser, daily-aggregate transfer loss
train/pretrain_source.py     Step 1
train/transfer_target.py     Steps 2 and 3
eval/evaluate.py        per-station metrics; also a standalone CLI
scripts/                build_index, make_folds, build_station_table, smoke_test, aggregate_folds
slurm/                  01_build_cache, 04_build_eval_index, 10_pretrain*, 20_transfer*,
                        30_diagnose, 31_by_hour, 32_degenerate, 33_significance,
                        40_basin_average, 41_africa, 42_africa_transfer, 43_africa_ensemble,
                        50_search
```

Outputs land in `outputs/<run>/fold{k}/{pretrain,transfer}/` — `best_model.pth`,
`training_history.csv`, `per_station_hourly_*.csv`, `summary.json`, `run_meta.json`
and a log. `<run>` is the config's `output_root`; `outputs/` is a symlink to
`/ibex/user/kongw0a/global_mtslstm_outputs` and is gitignored.

---

## Design decisions worth reviewing

**Folds.** 5-fold stratified by (source × record-length quintile). Rotating all five means
every station is a target station exactly once, so the pooled result covers all ~9,000
stations rather than a lucky 20%. CAMELSH is 57% of the stations, so an unstratified draw
would let fold composition drive the fold-to-fold spread.

**No separate test period.** The prepared data has only two temporal blocks. So:
Step 1 trains on the source domain's `training/` and early-stops on the source domain's
`validation/`; Step 2 fine-tunes on the target domain's `training/` and early-stops on a
15% held-out slice **of that same training period**. The target domain's `validation/`
period is touched exactly once, at final evaluation. No leakage in either direction.

**Early stopping on daily KGE (PLAN.md §3.2 #4).** Selecting the epoch by *hourly* KGE
would use observations the premise says don't exist. Step 2 selects on daily KGE only.
`--peek-hourly` logs what hourly test KGE *would* have been each epoch, purely for the
"does daily-based selection cost anything?" robustness check — it never touches the
optimizer or the stopper.

**Freezing.** Step 2 freezes `lstm_hourly` and fine-tunes `lstm_daily`, `transfer_h`,
`transfer_c`, `head_daily`, `head_hourly` — same as the reference
`set_trainable_parameters`. The smoke test asserts the frozen branch gets no gradient.

**Basin-averaged loss.** Squared error divided by `(stn_std + 0.1)²` with the per-station
std taken from each batch's metadata, so a 10,000 km² flashy basin and a small alpine one
weigh comparably.

**Units.** The prepared `q_mm` is already depth per hour, so PLAN.md §3.2 #3 (cms → mm/h)
and #2 (`handle_extremes(max=1000 m³/s)` silently deleting real flood peaks) **do not
apply** — neither conversion nor that clipping happens anywhere in this pipeline.

**Model.** `sMTSLSTM` carried over from `Modelzoo.py`; the only change is that the daily
branch now runs as prefix+remainder in a single pass instead of being executed twice for
the same result.

---

## Not done yet

* **Step 4 Africa / Step 5 ERA5-Land.** Scanned `/ibex/project/c2266/abbaa0a/` — findings:

  | thing | status |
  |---|---|
  | African **hourly** streamflow | **does not exist.** `processed/20250630/hourly/` holds exactly 10,423 stations (CAMELSH 5767, BOMAustralia 2059, LamaHCE 859, Japan 696, Germany 494, CzechRepublic 437, LamaHIce 111). Africa bounding box → 0 stations; the only southern-hemisphere stations are Australian. |
  | African **hourly forcing** | **not prepared.** Every hourly forcing file (`ERA5_HRES_hourly_{P,Temp,SWd,LWd,Pres,RelHum,Wind}`, `MSWEP_V316`) is dimensioned `stations: 10423` — basin-averaged for that set only. |
  | African **daily** streamflow | **yes.** `processed/20250630/daily/` has 1,577 stations in the Africa box: GRDC 826, GRDCCaravan 505, `restricted_ADHI` 246 (African Database of Hydrometric Indices). After the QC used by the continent-PUB setup: **294 basins** (GRDCCaravan 126, GRDC 119, ADHI 49). |
  | prepared African daily batches | pointer `input_data/hydrodeepai/.last_batches_path.pubfoldafrica` → `..._D_15872_38_pubfoldafrica` (16,166 − 15,872 = the 294 held-out basins), but **the batch directory itself is gone** — only the pointer survives. |
  | **ERA5-Land** | only `gscad_database/raw/HYSETS/HYSETS_2023_update_ERA5Land.nc` — daily, 14,425 HYSETS watersheds (**North America**), with `total_runoff` / `surface_runoff` in mm/d. No global or African ERA5-Land anywhere under `abbaa0a`. |

  **ERA5-Land is now being fetched** — see [Africa / ERA5-Land](#africa--era5-land) below.
  What remains blocked is the **model side of Step 4**: MTS-LSTM consumes an 8760-hour
  forcing window, and no hourly forcing exists for African basins.

* **Existing baselines found** (candidates for the "traditional LSTM" comparison — worth
  confirming with Ather what they actually are, since the run dirs contain
  `episode_rewards.csv` / `*_test_actions.csv`, i.e. these look like RL/dPL parameter
  regionalisation rather than a plain rainfall-runoff LSTM):
  * `results/regionalization/20250630/gscad_continent_lstm/47515076_*` — continent-holdout
    PUB, **daily**, 16,165 basins pooled, per-basin metrics with 18 scores each.
    Per-fold median KGE′: africa **0.277**, asia 0.513, europe 0.554, north_america 0.435,
    oceania 0.280, south_america 0.328; pooled 0.471.
  * `results/regionalization/hourly_20250630/dpl_hourly_nmul16/` — **hourly** PUB, 10 folds
    × ~650 basins, median KGE′ 0.5075 / KGE 0.476 / NSE 0.448 / corr 0.775. Same station
    universe as our hourly data, so this one is **directly comparable to the Phase I main
    result** and needs no new data.
* **CV-blocked split** (PLAN.md §2, spatially blocked folds). `data/folds.py` is written so
  a second fold table drops straight in — needs `station_static.csv` for the geographic
  clustering.
* **Hyperparameter search.** The reference SLURM array + W&B sweep pattern isn't ported yet;
  PLAN.md §4 wants it on fold 1 only.
* **M2 (symbolic prior).** Optional in PLAN.md; not started.

---

## Africa / ERA5-Land

Phase I Steps 4–5 evaluate on **African daily** streamflow, with ERA5-Land as the physical
baseline. Three scripts, run in order:

```bash
python -m scripts.build_africa_basins          # done: africa/africa_basins.{csv,gpkg} + tiles
python -m scripts.download_era5_land_africa --workers 4    # login node only (no internet on compute nodes)
python -m scripts.basin_average_era5_land      # -> per-basin mm/d
```

**The basins.** 294, taken from the continent-holdout PUB run that defines the African test
set: GRDCCaravan 126, GRDC 119, `restricted_ADHI` 49. Areas 19–9,839 km² (median 759 —
`max_area` is 10,000 in the QC config). Catchment polygons come from each station's own
`processed/20250630/daily/stations/<id>/boundary.shp`, which the gscad pipeline itself uses:
**294/294 resolved**, polygon area vs `static.csv` area at a median ratio of 0.998. Caravan's
`grdc_basin_shapes.shp` and GSHA's `boundaries.shp` are wired in as fallbacks but between
them miss all 49 ADHI basins, so they are never needed. Two polygons are flagged as suspect
(area ratio beyond 2×): `GRDC__1591800`, `GRDC__1547202`.

**The download.** Basin boxes are merged into **19 tiles / 45,494 ERA5-Land cells**; the full
Africa box would have been 33 TB. Only the **00:00 UTC** stamp is requested: ERA5-Land's
accumulated runoff resets at 00 UTC, so that value already is the previous day's total — 24×
smaller than pulling every hour. The one-day shift back onto the day it accumulates happens
in `basin_average_era5_land.py`, so what sits on disk is verbatim CDS output. Resumable via
`manifest.json`; re-running retries only what failed.

| variables × period | requests | volume |
|---|---|---|
| 3 vars, 1980–2024 | 171 | 8.4 GiB |
| `runoff` only, 1990–2024 | 76 | 2.2 GiB |

**Location.** `/ibex/user/kongw0a/era5_land_africa`, symlinked as
`~/era5_land_africa`. Not in `$HOME` proper: that filesystem is at 191 G against a 180 G
soft quota (200 G hard).

**Sanity check on the first tile** (t+02-01, southern DRC/Zambia, 2000): `ro` in metres,
366 daily steps all stamped 00:00; ×1000 → mm/d with median 0.13 and an annual total of
233 mm/yr, which is the right order for that region.

**Evaluation.** `eval/africa.py` loads each fold's checkpoint, predicts, aggregates its last
24 hourly outputs into a daily value, and scores all three references on identical
station-days: observed daily `q_mm`, ERA5-Land daily runoff, and the continent-PUB baseline
read from its per-basin result file. `--self-test` runs the whole path on synthetic forcing,
so it can be exercised before the download lands.

**Input packaging is verified, not assumed.** `scripts/verify_africa_inputs.py` rebuilds the
model input for stations that DO appear in the prepared batches, straight from `6sources.nc`
plus `static.csv`, and compares value by value against the batches themselves:

```
STATIC : max |error| over 6 stations x 42 features = 4.8e-07   (float32 rounding)
DYNAMIC: max |error| over 24 rebuilt (1000 x 3) sequences = 0  (exact)
```

That covers the lookback offsets, the dynamic and static standardization, the static column
order, and the Koeppen-Geiger code map — so for the African basins only the forcing data is
untested, not the code.

**Koeppen-Geiger needed reverse-engineering.** KGZ arrives as a string in `static.csv`
(`'C'`, `'Cfb'`) but as a numeric code in the prepared batches. The code turns out to be the
alphabetical index over the values present *in that dataset*, recovered by matching 700
batches against the hourly static table (one-to-one, reproduces every observed pair) and
recorded in [`data/kgz_codes.json`](data/kgz_codes.json). Consequence: the training
vocabulary is not the full Koeppen list, and **40 of the 294 African basins (13.6%) are Cwb
or Cwc, which have no code in it.** They are mapped to Cwa — same major class, same dry
winter, differing only in summer warmth. It is an approximation over 13.6% of the basins and
belongs in the writeup.

Separately verified on the 4,161 stations shared between the hourly and daily static tables:
41 of the 42 training features agree to the last digit, so taking African statics from the
daily table is consistent with training. The exception is `reservoir_impact_GRanD_v1_3`,
which differs on 17% of shared stations.

**Still open.** Running the MTS-LSTM over African basins needs hourly `pet`,
`pcp`, `temp` for them, and none existed — the processed hourly forcings are basin-averaged
for the 10,423 hourly stations only, and there are no global raw hourly grids under
`abbaa0a`. ERA5-Land is itself hourly at 0.1°, so the same tiling could produce that forcing
(~150 GB for 2000–2020 plus the Penman inputs, and days of CDS queue), but it swaps the
precipitation product: the model is trained on MSWEP V3.16, so an ERA5-Land-forced Africa
run would confound transfer skill with forcing differences. MSWEP V3 hourly for Africa would
have to come from gloh2o.org instead.

## W&B

Runs go to project `global_mtslstm_phase1`, grouped `phase1_pretrain` / `phase1_transfer`,
named `pretrain_fold{k}` / `transfer_fold{k}`. Log in once on a login node:

```bash
wandb login
```

Without credentials the job **falls back to offline mode** rather than dying; sync later with
`wandb sync wandb/offline-run-*`. Set `wandb.mode: disabled` in the config to turn it off.
