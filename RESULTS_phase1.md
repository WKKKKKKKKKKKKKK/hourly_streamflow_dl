# Phase I results — can daily-only supervision keep hourly skill?

Setup: 5-fold station split, 80% source / 20% target. Pretrain on the source domain
with hourly targets (**M0**), then fine-tune on the target domain using *only* 24-h
aggregates (**M1**). Report per-station hourly KGE on the untouched validation
period, median across stations. STEP 3 re-scores the source domain after the
fine-tune to price the forgetting.

Two data paths for the daily branch:

- **run A** (`configs/phase1.yaml`) — the prepared 14 TB batches. The "daily" branch
  is a power-law *subsample* of the past year: of 365 days, 8 carry 24 points, 176
  carry a single instantaneous hour, 7 carry none. `reg_window: 24` so the daily head
  is trained to mean a 24-h average even though `frequency_factor: 1` positions the
  state hand-off.
- **run B** (`configs/phase1_runB.yaml`) — windows rebuilt from `6sources.nc` with
  365 *genuine* daily means, `frequency_factor: 24`. What the 100-station reference
  does. Cache: `/ibex/user/kongw0a/hourly_cache`.

## Headline

| | M0 | M1 | ΔKGE | r | α | β | r is culprit |
|---|---|---|---|---|---|---|---|
| run A target | 0.4288 | 0.3982 | **−0.0212** | 0.788→0.775 | 0.723→0.708 | 0.890→0.914 | 9.6% |
| run A source (STEP 3) | 0.4429 | 0.4050 | −0.0256 | 0.791→0.779 | 0.731→0.713 | 0.892→0.914 | 10.1% |
| run B target | 0.5249 | 0.5758 | **+0.0264** | 0.790→0.798 | 0.861→0.956 | 1.016→0.977 | 7.7% |
| run B source (STEP 3) | 0.6435 | 0.5014 | **−0.1066** | 0.818→0.788 | 0.862→0.955 | 1.006→0.955 | 9.6% |

(Baseline with `reg_window` left equal to `frequency_factor: 1`: target M0 0.4212 →
M1 0.3828, Δ −0.0384. Superseded by run A.)

## What the KGE decomposition settles

`scripts/diagnose_kge.py` splits KGE into r (timing), α = std(sim)/std(obs)
(variance) and β = mean(sim)/mean(obs) (bias), then attributes each station's change
counterfactually: move ONE term to its M1 value, leave the other two at M0, and see
what that alone does. Whichever hurts most is that station's culprit.

**Daily-only supervision does not damage timing. It re-scales magnitude.** Across
all four cases above, r leads in only 7.7–10.1% of the stations that got worse, and
median Δr is −0.006 to +0.008. What moves is α and β. That answers the Phase I
question in the affirmative for the part that matters: hourly *dynamics* learned
from the source domain survive a fine-tune that never sees an hourly target. What
needs handling is calibration — exactly what a daily aggregate can inform.

Which direction the re-scaling helps depends on where the model started:

- run A starts under-dispersed *and* low (α 0.72, β 0.89). Fine-tuning adds volume
  (β→0.91) and costs a little variance (α→0.71). Net small loss.
- run B starts with near-perfect bias (β 1.02) and moderate under-dispersion
  (α 0.86). Fine-tuning largely fixes the under-dispersion (α→0.96) and pays in bias
  (β→0.98). Net gain.

The fine-tune applies essentially the **same** re-scaling everywhere — run B's target
lands at α 0.956 / β 0.977 and its source at α 0.955 / β 0.955, nearly coincident.
That is why run B's source domain, which was already well calibrated, loses 0.107:
the model is pushed onto the target's calibration and the source pays for it. Run B
gains more on the target than run A and pays an order of magnitude more on the
source. That trade is the Phase I result, not a bug.

## The bigger lever, which predates the transfer step

In run A, **76.6% of stations are already under-dispersed at M0** (median α 0.707)
and variance carries **59.4%** of the KGE deficit — versus 18.0% for timing and 22.6%
for bias. That ceiling exists before any fine-tuning and is an order of magnitude
larger than the −0.021 the transfer costs. Run B halves the gap on its own (α 0.861,
under-dispersed share 68.0%), which is most of where its +0.096 M0 advantage comes
from. Chasing α is worth more than chasing the transfer loss.

## Source replay: damping the re-scaling, not preventing forgetting

Plain daily-only fine-tuning cost run B 0.107 median KGE on the source domain
(STEP 3), with M1 landing at nearly the same alpha/beta on BOTH domains -- one
global re-scaling that the target gains from and the source only pays for. So mix
source batches, with their real hourly targets, back into the fine-tune
(`transfer.source_replay_ratio`). This is not leakage: the premise hides the TARGET
stations' hourly observations, and the source domain's hourly data is what Step 1
trained on. Same frozen modules, same target-daily early-stopping metric, same
pretrained weights (`PRETRAIN_ROOT`) -- the data mix is the only difference.

| ratio | target Δ | source Δ | target M1 | source M1 | station-weighted M1 |
|---|---|---|---|---|---|
| 0 (run B) | +0.0449 | −0.1064 | 0.5768 | 0.5006 | 0.5159 |
| **0.25** | **+0.0643** | −0.0668 | 0.5962 | 0.5544 | 0.5628 |
| 0.5 | +0.0578 | **−0.0502** | 0.5897 | 0.5746 | 0.5776 |

0.25 beats no-replay on BOTH domains, all 5 folds, both metrics. Going to 0.5 keeps
rescuing the source (5/5 folds) but gives target gain back (4/5 folds) -- the target
curve peaks near 0.25, the source improves monotonically. Station-weighted (20%
target + 80% source) still favours more replay, because the source is 80% of the
stations.

### What replay actually does

Not a general improvement. Grouping target stations by the ΔKGE that plain
fine-tuning gave them (all-hours paired, obs_std ≥ 1e-3):

| plain-transfer ΔKGE | stations | plain | replay 0.25 | replay's edge |
|---|---|---|---|---|
| < −0.1 (damaged) | 2254 | −0.232 | −0.090 | **+0.145** |
| −0.1 to 0 | 1648 | −0.046 | −0.020 | +0.026 |
| 0 to 0.05 | 956 | +0.025 | +0.010 | −0.015 |
| 0.05 to 0.15 | 1376 | +0.096 | +0.065 | −0.029 |
| > 0.15 (helped a lot) | 2609 | +0.359 | +0.304 | **−0.058** |

Replay rescues the damaged tail and gives up gain where fine-tuning was already
working. The net is positive only because the damaged group is large (44% of
stations). Stations badly hurt (ΔKGE < −0.1) fall from 25.5% to 18.3%.

The mechanism is visible in alpha. Plain fine-tuning's global re-scaling is too
aggressive for a substantial minority: it pushes **6.25%** of stations from
under-dispersed straight past alpha 1.2 into over-dispersion, wrecking their KGE.
Replay damps it -- median |Δalpha| 0.180 → 0.131, overshoot share 6.25% → **2.09%**,
median alpha 0.861 → 0.876 instead of 0.861 → 0.956. Damping costs accuracy where
the aggressive version happened to be right and saves much more where it was not.
That also explains the source domain: it was already well calibrated (alpha 0.862,
beta 1.006), so the aggressive re-scaling there is pure damage.

An earlier reading of the fold-0 numbers as "no trade-off, both sides win" was
wrong twice over: aggregate-level both domains do improve, but station-level there
IS a trade, and the mechanism is damping an over-aggressive re-scaling -- not
regularisation against overfitting the target training period.

### Picking a ratio

| goal | choice |
|---|---|
| best hourly forecasts at the daily-only stations (the Phase I question) | **0.25** |
| one model serving every station | **0.5** or higher -- the source is 80% of stations |
| reproduce the plain Phase I result | 0 |

## Random vs spatially blocked splits: how much skill is just proximity

CAMELSH alone contributes 5,767 US stations, so under a random split a target
station almost always keeps a hydrological neighbour in the source domain. Measured:
**48.1%** of target stations have a source-domain station within 10 km and **96.9%**
within 50 km, median nearest distance **10.4 km**. `folds/folds_blocked.csv`
(`scripts/make_folds_blocked.py`, 120 spatial blocks packed into 5 folds) pushes that
to a median **94.9 km**, with only 22.2% inside 50 km, while keeping fold sizes even
(1791-1801 target stations vs the random split's 1796-1800). Same run B config, no
replay, so the split is the only difference.

| split | target M0 | target M1 | ΔKGE | source M0 | source M1 |
|---|---|---|---|---|---|
| random | 0.5319 | 0.5768 | +0.0449 | 0.6434 | 0.5006 |
| **blocked** | **0.4040** | 0.4754 | **+0.0714** | 0.6405 | 0.3188 |

**Zero-shot skill falls 0.128** (all 5 folds, −0.091 to −0.191). About a quarter of
the random split's hourly KGE was coming from spatial proximity rather than from
generalising over basin attributes, which is what PLAN.md 2 predicted and why the
random-split numbers should not be read as regional-extrapolation performance.

Two further readings, one encouraging and one not:

- **Daily-only supervision helps MORE when there are no neighbours** (+0.0714 vs
  +0.0449). The method is most valuable in precisely the situation that motivates it
  — a whole region with no hourly gauges.
- **Forgetting is far worse under blocking**: the source domain falls 0.6405 → 0.3188
  (−0.32) against −0.14 for random. Source replay was tuned on the random split and
  should matter more here; that combination has not been run yet.

A caveat that comes with spatial blocking and cannot be removed: removing neighbours
also removes the region, so fold composition is uneven by construction (6 of 30
agency-by-fold cells empty; US share 40-73% across folds versus roughly even under
random). Averaging over 5 folds mitigates it but does not eliminate it. Block count
was chosen from that trade-off rather than picked: 60 blocks gives 140 km separation
but a 52% spread in US share, 240 blocks gives 64 km and a 17% spread.

## Train and test periods, stated precisely

Each station's own record is split **70% / 30% in time** — there is no single global
date, and the boundary's percentiles across stations are 2010-03-11 (25%),
**2015-07-26 (median)**, 2017-12-21 (75%). The first 365 days of each split are
burn-in and produce no samples. Training samples span 1981-01-09 to 2024-05-14
(51.7M), validation samples 1985-07-14 to 2024-12-31 (22.1M).

This is **two blocks, not the three in PLAN.md 2** (`train 2000-2012 / val 2013-2015
/ test 2016-2020`), because the prepared batches are cut 0.7/0.3 and run B reuses
that split so the two data paths stay comparable. Consequences, by number:

- **The main result is unaffected.** M0 and M1 are scored on the same untouched
  target validation period, so any period-sharing bias applies equally to both and
  cancels in ΔKGE. The transfer step early-stops on a holdout from the target
  TRAINING period using daily-aggregate KGE only, so the target validation period
  never informs epoch selection — the fix PLAN.md 3.2 #4 asks for.
- **Absolute M0/M1 are mildly optimistic.** Pretraining early-stops on the source
  domain's validation period, which shares a calendar span (though not stations)
  with target evaluation. Bounded empirically: across folds the best epoch beats the
  last epoch by only **+0.0063** (random) and **+0.0036** (blocked), with a
  0.014-0.017 spread over the final ten epochs. So perfect hindsight in epoch
  selection is worth ≤0.006 — one to two orders of magnitude below the effects being
  reported (0.128, 0.064, 0.107).
- **The source-domain STEP 3 number is the weak one**: selection and reporting use
  the same stations in the same period.

So report these as **held-out samples from the validation period, disjoint from the
early-stopping subset** — not as a temporally independent test period. Rebuilding a
three-way split would cost ~3 days of GPU across all runs, shrink the training data,
and change no comparative conclusion, so it is deliberately not done; if required
later, only the final configuration needs redoing.

Separately, the early-stopping set is too small and too narrow in time
(`val_batches_per_station: 1` = 512 samples/station, and on the prepared path those
come from a single batch file, i.e. one narrow window). That is why the selection
metric reads median KGE 0.085 where the final report on ~6,100 samples/station gives
0.433. It is metric noise rather than leakage, and by the bound above it costs at
most ~0.006, but any future run should widen it.

## Two evaluation traps found along the way

**Degenerate stations wreck the mean, not the median.** α and β divide by std(obs)
and mean(obs). The worst station here has obs_std 5e-7 mm/h — a near-constant record
— which sends α into the thousands and drags the target-domain *mean* ΔKGE to −30
while the median sits at −0.021. Only exactly-zero std is excluded upstream, so
`diagnose_kge.attribute()` filters `obs_std < 1e-3 mm/h` (3.5% of stations) and
reports the count. Report medians; if a mean is wanted, filter first.

**The cache path's "hourly" KGE was one hour of the day.** A target must sit at hour
23 for the last 24 hourly steps to form one calendar day, so all 73,808,280 samples
in `samples_stride24.npz` have `target_idx % 24 == 23`. run A's prepared batches
cover all 24 hours at ~4.2% each (verified over 20 batch files), so the two runs'
"hourly KGE" were not the same quantity — and daily-mean supervision is expected to
damage intra-day shape, which a once-daily snapshot cannot see.

`scripts/build_eval_index.py` rebuilds the index at stride 1 (54.8M samples, 4.12–
4.28% per hour) and `scripts/kge_by_hour.py` scores each hour separately. The result
**cleared the concern**: 23:00 is not a spike but sits inside the spread of other
hours (M0 0.519 at 23:00 vs 0.508 median, range [0.495, 0.525]; hour 21 is higher),
and r is flat around the clock (0.810–0.824). An out-of-distribution alignment would
have shown 23:00 spiking with the other 23 hours flat and uniformly worse. Re-scoring
run B on all hours moved M0 by only −0.007 (0.5319 → 0.5249) and made Δ slightly
*more* positive (+0.0214 → +0.0264). So stride-24 training did not lock the model to
one clock hour.

What the breakdown does show is a real diurnal cycle of amplitude ~0.03: lowest
05:00–11:00, highest around 21:00, with α falling in the afternoon (0.85 at 15:00–
19:00 vs 0.89 in the morning) and β dipping below 1 at the same time. Afternoon
convective peaks are systematically under-predicted, in both M0 and M1.

Coverage cost worth noting separately: 531,717,759 valid all-hour targets exist;
stride 24 uses 73,808,280 of them (13.9%), all at the same clock hour.

## Reproducing

```bash
# run B cache (once)
sbatch slurm/01_build_cache.sbatch
sbatch slurm/04_build_eval_index.sbatch          # all-hours evaluation index

# a run, per config
P=$(sbatch --parsable slurm/10_pretrain_runB.sbatch)
sbatch --dependency=afterok:$P slurm/20_transfer_runB.sbatch

# diagnostics (evaluation only, no retraining)
D=$(CONFIG=configs/phase1_runB.yaml INDEX=samples_evalhours.npz \
    OUT_ROOT=outputs/runB_truedaily/diagnostics_allhours sbatch --parsable slurm/30_diagnose.sbatch)
MERGE=1 INDEX=samples_evalhours.npz OUT_ROOT=outputs/runB_truedaily/diagnostics_allhours \
  CONFIG=configs/phase1_runB.yaml sbatch --dependency=afterany:$D --array=0 slurm/30_diagnose.sbatch
CONFIG=configs/phase1_runB.yaml sbatch slurm/31_by_hour.sbatch
```

## Open

- Early stopping selected on a badly biased metric. `val_batches_per_station: 1` is
  512 samples/station, and on the prepared path those come from ONE batch file —
  a narrow time window, so median KGE reads 0.085 where the final report on ~6,100
  samples/station gives 0.433. Left alone because the configs are frozen for
  comparability, but any future run should widen it.
- run B's daily branch uses calendar-day means, so the (daily-end, target) offset is
  always 24 h. The reference uses t-relative trailing means, which is alignment-
  invariant. The by-hour test says this costs little here, but it is a real
  difference from the reference.
- Steps 4–5 (Africa daily validation) wait on the ERA5-Land forcing download.
