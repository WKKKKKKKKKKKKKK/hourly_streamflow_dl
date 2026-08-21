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


> **Status note (2026-08-17).** Two configurations are reported throughout. **v1** is
> the original: `lookback_hourly: 72` and no `initial_forget_bias`. **v2** applies the
> fold-1 search result — `lookback_hourly: 336` (168 for run A, a data-layout limit) and
> `initial_forget_bias: 3`, the latter because it is part of Gauch et al.'s published
> method and was missing from both this code and the 100-station reference. Nothing else
> differs. v2 is the primary result; v1 is retained because several conclusions change
> in magnitude between them and the change is itself informative. run A's v2 completed
> 2026-08-20 and is quoted directly below; the run A vs run B comparison no longer
> depends on v1.
## Headline

**Every KGE, NSE, r, alpha and beta in this file is a per-station MEDIAN over target
stations, never a mean.** That is not a stylistic choice and the distinction is large
enough to invert the sign of a claim: for v2 run B the target-domain KGE median is +0.5322
at M0 and +0.6281 at M1, while the corresponding *means* are **-6.99** and **-7.88**, with
single stations reaching -10487 and -17484. Dropping the 140 numerically degenerate
stations (`obs_std < 1e-3`) pulls the means back to +0.07 and +0.34 -- still far below the
medians, because a tail down to -130 survives the filter. Quote these numbers as medians
or not at all; a reader who recomputes a mean from the per-station CSVs will get a negative
number and be right to.

| | M0 | M1 | ΔKGE | r | α | β | r is culprit |
|---|---|---|---|---|---|---|---|
| run A target | 0.4288 | 0.3982 | **−0.0212** | 0.788→0.775 | 0.723→0.708 | 0.890→0.914 | 9.6% |
| run A source (STEP 3) | 0.4429 | 0.4050 | −0.0256 | 0.791→0.779 | 0.731→0.713 | 0.892→0.914 | 10.1% |
| run B target | 0.5249 | 0.5758 | **+0.0264** | 0.790→0.798 | 0.861→0.956 | 1.016→0.977 | 7.7% |
| run B source (STEP 3) | 0.6435 | 0.5014 | **−0.1066** | 0.818→0.788 | 0.862→0.955 | 1.006→0.955 | 9.6% |
| **v2 run B target** | **0.5190** | **0.6210** | **+0.1020** | 0.797→0.812 | 0.813→0.851 | 1.025→0.996 | — |

### v2: what the search bought

Five folds, same folds and data, only `lookback_hourly` and `initial_forget_bias` changed
(difference of medians from the transfer logs):

| | M0 | M1 | ΔKGE | source Δ |
|---|---|---|---|---|
| v1 run B | 0.5319 | 0.5768 | +0.045 | −0.143 |
| **v2 run B** | 0.5318 | **0.6277** | **+0.096** | **−0.088** |
| v1 blocked | 0.4040 | 0.4754 | +0.071 | −0.322 |
| **v2 blocked** | **0.4249** | **0.6210** | **+0.196** | **−0.173** |
| v1 replay 0.25 | 0.5319 | 0.5962 | +0.064 | −0.089 |
| v2 replay 0.25 | 0.5318 | 0.6253 | +0.093 | −0.058 |

Three things fall out, and the second is the important one.

**Zero-shot barely moves (0.5319 → 0.5318) while M1 gains 0.051.** A longer hourly window
and an open forget gate do not make the pretrained model better; they make it more
*adaptable*. The entire gain is in what fine-tuning can then exploit.

**The blocked split gains nearly three times as much (+0.071 → +0.196), and its M1 comes
within 0.007 of the random split's** (0.6210 vs 0.6277). Under v1 the two differed by
0.101. So the penalty for removing hydrological neighbours is largely recoverable —
provided the hourly branch is long enough to use what the target basin's own daily record
says. That reframes the headline: proximity dominates *zero-shot* skill (M0 still differs
by 0.107), but after daily-only fine-tuning it barely matters.

**Source replay stops helping the target domain** (+0.096 → +0.093) while still protecting
the source (−0.088 → −0.058). Part of what replay bought under v1 was compensation for too
short a window; give the window and that part disappears.

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

### A caveat on this conclusion, not yet resolved

Gauch et al.'s own MTS-LSTM opens the LSTM forget gate at initialisation
(`initial_forget_bias: 3` in neuralhydrology). **Neither this implementation nor the
100-station reference it was built from does so** — the deviation is from the paper,
not between the two experiments here. Every result in this document was produced
without it.

That matters specifically for the under-dispersion claim. With a half-closed forget
gate, a 365-step daily branch lets the cell state decay before it reaches the
hand-off, so long-memory signals — snowpack, groundwater — never arrive at the hourly
branch. The symptom that predicts is exactly what the intra-day diagnostic measured:
M0's day-to-day variation far too small while its within-day jitter is 3.1x too large.

So part of what is called a ceiling here may be a missing component rather than an
inherent limit. The fold-1 search includes `g03_forgetbias_H72` and
`g04_forgetbias_H168`, which hold the hourly window fixed and change only the forget
gate, precisely to separate the two. Until those return, this section should be read
as "α is the dominant deficit **of the model as trained here**", not as a statement
about the architecture.

Nothing in the comparative results is affected: every run — baseline, run A, run B,
blocked, all replay ratios, Africa — used the same initialisation, so ΔKGE, the
random-vs-blocked gap, the replay sweep and the Africa gain all stand. What could
move is the absolute level, and the interpretation of α as a fundamental limit.

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

### The drop is not a composition artefact — paired, per station

Spatial blocking necessarily unbalances fold composition, so "M0 falls" invites the
objection that the two splits scored different mixes of catchments. The 5-fold design
settles it: every station serves as a target exactly once in EACH split, so the runs
pair station by station and composition is held fixed by construction.

Over the same **8,709** catchments (`scripts/paired_split_effect.py`):

| statistic | value |
|---|---|
| median KGE, random → blocked | 0.5396 → 0.4185 |
| difference of medians | −0.1212 |
| **paired median drop** | **−0.0780** (65.6% of stations worse, p = 6.9e-224) |

The two figures are different estimators and both are quoted elsewhere in this
document; the paired one is the stricter statistic and should be the headline.

**Every agency drops, without exception** — which a composition artefact could not
produce:

| agency | n | random | blocked | paired drop |
|---|---|---|---|---|
| BOMAustralia | 1,607 | 0.2281 | 0.1024 | −0.0746 |
| CAMELSH (US) | 5,047 | 0.5856 | 0.4794 | −0.0664 |
| Germany | 458 | 0.6146 | 0.5077 | −0.0627 |
| Japan | 690 | 0.5697 | 0.4017 | −0.1205 |
| LamaHCE | 834 | 0.5922 | 0.4504 | −0.1337 |
| LamaHIce (Iceland) | 73 | 0.3039 | 0.1018 | **−0.2457** |

**The spread across agencies is itself a result: reliance on spatial proximity scales
inversely with gauge density.** The US contributes 5,767 stations and loses least
(−0.066) because even after blocking there are still comparable American catchments in
the source domain. Iceland has 73 stations and is geographically isolated, so removing
its block leaves nothing similar to learn from — it loses **−0.246**, nearly four times
as much. Sparse networks are exactly where the random split is most optimistic, and
exactly where this method is meant to be useful.

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

## Africa: the same experiment on a continent with no training station

The 294 African basins have daily discharge and no hourly discharge — Phase I's
premise, occurring naturally. None appears anywhere in the training data, so this is
the only genuinely external test here. The model is driven by ERA5-Land hourly
forcing, its last 24 hourly outputs are averaged into a daily value, and that is
scored against observed daily `q_mm` over 1980–1995 (the forcing span; 282 of 294
basins have >365 observed days inside it, median 3,926).

| | KGE | r | α | β | KGE>0 |
|---|---|---|---|---|---|
| **M0 zero-shot** | −0.032 | 0.686 | **0.162** | 0.597 | 42.4% |
| **M1 daily-only fine-tune** | **+0.143** | 0.685 | **0.561** | 0.397 | 64.0% |
| replay 0.25 | +0.080 | 0.690 | 0.501 | 0.360 | 60.1% |
| blocked M0 / M1 | −0.066 / +0.078 | 0.678 / 0.675 | 0.144 / 0.492 | | 39.2% / 59.0% |
| ERA5-Land runoff | −0.334 | 0.403 | 1.595 | 1.107 | 24.7% |
| continent-PUB baseline | +0.279 | | | | |

**Paired ΔKGE = +0.165, 72.4% of basins improved (p = 3.7e-16)** — more than double
the global gain (+0.026 random, +0.071 blocked). Daily-only supervision is worth
*more*, not less, on a continent the model has never seen.

Three things line up with everything else in this document:

- **r is untouched again** (0.686 → 0.685). Across five cases — two data paths, two
  splits, three replay ratios, and now a different continent — daily-aggregate
  supervision never moves timing. That is the robust result.
- **α is the whole story and Africa is its extreme.** Zero-shot α = 0.162: the model
  reproduces 16% of the observed variability. Fine-tuning lifts it to 0.561, and that
  is where the +0.165 comes from. Compare run A 0.72, run B 0.86, Africa 0.16 — the
  further out of domain, the worse the under-dispersion.
- **Replay is slightly worse here (−0.016), which confirms the damping mechanism
  rather than contradicting it.** Replay damps the re-scaling; that helps where the
  model overshoots (6.25% of global stations) and costs where the model needs the
  full re-scaling (Africa, α far below 1).

M1 beats the physical baseline decisively (+0.143 vs −0.334) and loses to a model
trained specifically for continent holdout (+0.279), which is the expected place for
a model that has never seen an African basin.

The blocked-split models are slightly *worse* on Africa than the random-split ones
(+0.078 vs +0.143), which is counter-intuitive; the uneven fold composition that
spatial blocking forces (6 of 30 agency-by-fold cells empty) is the likeliest
explanation, but this is recorded rather than explained.

### Getting there took four wrong answers

Africa is the only part of this work that had never run on real data, and it showed:
a missing dependency, then tiles that cannot be assembled into a hypercube, then an
mm/h vs mm/d factor of 24, then — worst — feeding run B's checkpoints the *prepared*
1000-step subsample as their daily branch when they were trained on 365 genuine daily
means. Each produced a completed job and plausible-looking numbers. The fourth
yielded M0 = 0.001, M1 = −0.418 and an entire mechanistic story about "over-shooting
into over-dispersion" that was pure artefact — the corrected numbers above have the
opposite sign. `eval/africa.py` now dispatches on `cfg.data.source`, and
`predict_daily` refuses to report when the simulated/observed ratio is off by a
window length.

## Africa, run properly: fine-tuning on African daily observations

Everything else reported for Africa applies models fine-tuned on temperate target
stations. That tests extrapolation, not the method. Africa's 294 basins have daily
discharge and no hourly discharge — Phase I's premise occurring naturally — so the
protocol belongs there directly: pretrain on the global source domain, fine-tune on the
**African** training period using African daily observations only, score daily skill on
the African held-out period. Each basin's own record splits 70/30 in time.
`train/africa_transfer.py`, five folds, both configurations.

| | M0 | M1 | paired ΔKGE | improved | α | r |
|---|---|---|---|---|---|---|
| v1 (H=72) | −0.131 | +0.505 | +0.611 | 92.3% | 0.175→0.802 | 0.596→0.780 |
| **v2 (H=336)** | **+0.031** | **+0.499** | **+0.401** | **82.1%** | 0.448→0.830 | 0.679→0.784 |

**M1 is essentially identical between configurations (0.505 vs 0.499) while M0 rises by
0.162.** The apparent shrinkage of the gain is entirely a shift in the starting point,
not a loss of what daily observations contribute. Put the other way: **the ceiling
reached after fine-tuning is set by the African data itself and is indifferent to the
hourly window length**; the window only decides how far below that ceiling the zero-shot
model starts.

So the result stands with a corrected magnitude: **African daily observations are worth
+0.401 KGE (five folds, sd 0.060, range 0.299–0.457), improving 82.1% of basins**, still
far above the continent-holdout PUB baseline of +0.279 — from a model that has never seen
an African basin.

### The comparison that matters

Fine-tuning on temperate stations and applying the result to Africa is a different
operation from fine-tuning on Africa, and v2 separates them cleanly:

| | v1 ΔKGE | v2 ΔKGE | retained |
|---|---|---|---|
| temperate fine-tune, applied to Africa | +0.165 | **≈0** (0.1451 → 0.1432) | **0%** |
| **fine-tune on African daily observations** | +0.611 | **+0.401** | **66%** |

The temperate-transfer gain vanishes entirely once the window is long enough; the in-situ
gain keeps two thirds. **The v1 temperate-transfer "gain" was almost all compensation for
too short an hourly window. The in-situ gain is the model actually learning from African
data.** No single number in this document makes the case for in-situ adaptation as
directly as this contrast does.

### Timing: a bound, and a ceiling that looks physical

Elsewhere r barely moves — across two data paths, two splits and three replay ratios,
median Δr sits between −0.006 and +0.008. In situ on Africa it rises: +0.184 under v1,
**+0.105 under v2**. So the earlier claim needs a scope rather than a retraction:
**daily-aggregate supervision leaves timing alone when the model already has the region's
dynamics, and improves timing when it does not.** Zero-shot r on Africa is 0.60–0.68 —
the timing was never learned — and African daily observations carry enough to fix part of
it.

More striking, **M1's r converges to the same value under both configurations**: 0.777–0.782
(v1, sd 0.0016) and 0.777–0.790 (v2, sd 0.0060). Ten independent fine-tunes across two
different hyperparameter sets all land at r ≈ 0.78. That looks less like a property of the
model than a limit imposed by the forcing: ERA5-Land precipitation timing over these
basins is plausibly what caps it. Testing that would need a different forcing product,
which is outside Phase I.

## Blocked split, mechanistically

The blocked line previously had only M0/M1. Its full diagnostic suite (all-hours paired,
8,862 stations):

| | M0 | M1 | Δ |
|---|---|---|---|
| KGE | 0.4014 | 0.4775 | +0.0473 |
| r | 0.7589 | 0.7864 | +0.0139 |
| α | 0.8519 | 0.9493 | +0.0909 |
| β | 1.0325 | 0.8829 | −0.1413 |

Culprit shares among degraded stations: **r 4.2%**, α 23.1%, β 72.7% — timing is even
less implicated than under the random split (7.7%), so the "supervision re-scales, it
does not disturb timing" result holds under the harder split too.

Intra-day shape shows the same pattern as the random split and no degenerate solution:
observed flashiness 0.0285, M0 0.1697 (6.0× too jittery), M1 0.0747; mean 0.0483 vs M0
0.0280 (half the volume) and M1 0.0486 (1.01×). Fine-tuning fixes volume and halves the
excess jitter.

Significance: after BH correction 8,276 of 8,862 stations (93.4%) change significantly,
but again split in direction — 42.7% improved, 50.7% degraded on absolute error, while
pooled ΔKGE is +0.0421 (p = 4.4e-42). The calibration-not-accuracy caveat applies
identically here.


## Where the gain lands, and where it does not (PLAN.md 5.1–5.3)

`scripts/stratify_gain.py` cuts the per-station gain by 16 covariates.

**The gain does not depend on how close the nearest training station is.** This is
the opposite of what the blocked-split headline might suggest:

| nearest station in another fold | M0 | gain |
|---|---|---|
| random split, 2.5 km | 0.564 | +0.029 |
| random split, 31.8 km | 0.454 | +0.025 |
| blocked split, 31 km | 0.603 | +0.031 |
| blocked split, 94 km | 0.563 | +0.019 |
| blocked split, 211 km | 0.342 | +0.031 |

Proximity sets the *base* skill — M0 falls from 0.60 to 0.34 — and does nothing to
how much daily supervision adds. Neighbour distance ranks last of 16 covariates by
gain spread (0.011–0.012; Spearman −0.027 and −0.017). Together with the blocked
split gaining more (+0.071 vs +0.045) and Africa most of all (+0.165), the method
does not degrade in exactly the data-sparse setting that motivates it.

**Catchment area is the strongest single predictor** (Spearman −0.142, p = 4e-41):

| area | M0 | gain |
|---|---|---|
| < 87 km² | 0.429 | **+0.076** |
| 87–233 | 0.503 | +0.050 |
| 233–564 | 0.553 | +0.021 |
| 564–1537 | 0.590 | +0.008 |
| > 1537 km² | 0.599 | **−0.007** |

Small, fast-responding catchments are where zero-shot does worst (M0 0.43, α 0.758)
and where a daily total adds most. Large ones are already smooth, already at M0 0.60,
and gain nothing. By agency: Iceland +0.218, central Europe +0.050, Australia +0.045,
Germany +0.039, Japan +0.031, US +0.016. By Köppen zone the extremes are Cfc +0.235
and Dfc +0.158 against BSh −0.021 and Dfa −0.012; arid B zones start at M0 −0.073
(α 0.664) and daily aggregates do not rescue them.

**Two of PLAN.md 5.3's predictions do not hold.** Reservoir impact has *no* effect on
the gain (Spearman −0.006, p = 0.58) and `max_lag_corr` almost none (−0.022,
p = 0.039), where snow/storage-dominated catchments were expected to gain least.
Snow fraction is in fact positively associated (+0.081).

## No degenerate solution — but M0 is jittery, not smooth (PLAN.md 5.4)

`loss_agg` constrains only the 24-hour mean, so a constant-within-day output would
satisfy it perfectly while being useless hourly. Stride-24 sampling makes this
directly measurable: each sample's last 24 hourly outputs are one calendar day, and
consecutive days stitch into a continuous series. Over 8,432 stations (mm/h):

| | observed | M0 | M1 | M1/obs | M1/M0 |
|---|---|---|---|---|---|
| flashiness | 0.0285 | 0.1941 | 0.0572 | 2.01 | **0.30** |
| intra-day std | 0.0058 | 0.0181 | 0.0088 | 1.52 | **0.49** |
| intra-day range | 0.0187 | 0.0526 | 0.0285 | 1.53 | 0.54 |
| q95 events/yr | 11.94 | 19.32 | 21.37 | 1.79 | 1.11 |
| mean | 0.0483 | 0.0242 | **0.0490** | **1.01** | 2.03 |

Nothing flattens. The failure mode is the reverse: **M0 is jittery and dry** — 3.1×
too much intra-day variation, 6.8× too much flashiness, and only half the observed
volume. Fine-tuning fixes the volume almost exactly (1.01×) and halves the excess
jitter.

This resolves an apparent contradiction with α. Over the whole series α at M0 is
0.861, i.e. under-dispersed; within a day the model is over-dispersed by 3.1×. Both
hold at once because M0's *day-to-day* variation is far too small while its
*within-day* variation is too large — it fidgets without tracking events. Fine-tuning
improves both.

## Significance with FDR control, and a caveat that changes the claim (PLAN.md 5)

Each of 8,862 stations gets its own paired Wilcoxon on sample-level |error|, then
Benjamini-Hochberg across stations (at α = 0.05, ~443 stations would look significant
by chance).

- uncorrected p ≤ 0.05: 8,132 stations (91.8%)
- **after BH: 8,117 significant (91.6%)** — the effect is real, not multiplicity
- but the direction splits: **3,712 improved (41.9%) vs 4,405 degraded (49.7%)**
- median error change across all stations: **−0.00019 mm/h, i.e. slightly worse**
- pooled Wilcoxon on per-station KGE: median ΔKGE **+0.0214, p = 4.05e-33** (8,849
  pairs; 13 dropped for a non-finite KGE)

KGE and absolute error disagree often enough that quoting one alone misrepresents the
result: KGE improves at 54.6% of stations, absolute error at 46.5%, and the two agree
in direction at only 67.5% (Spearman +0.471). 1,798 stations gain KGE while losing
accuracy; 1,075 do the reverse.

**So the claim must be stated as calibration, not accuracy.** Daily-only supervision
makes the hydrograph better calibrated — right volume, better variance ratio, less
spurious intra-day jitter — while leaving point-wise error slightly worse at more
stations than it improves. That is mechanistically consistent: raising α toward 1
improves KGE, and absolute error is minimised by predicting closer to the conditional
median. The two metrics want opposite things.

## Global map (PLAN.md 5.5)

`scripts/global_map.py` writes four panels (M0, M1, gain, α at M0) over all 8,843
scored stations — the 5-fold design gives every station exactly one turn as a target,
so there is a real value at every gauge.

By latitude: >60° (Iceland) M0 0.283 gain **+0.218**; 45–60° 0.601/+0.046; 30–45°
0.582/+0.015; 0–30° 0.432/**−0.015** (the only negative band); −30–0° 0.171/+0.033;
<−30° 0.230/+0.055.

The map also makes the honest point that "global" describes the model, not the gauge
network: CAMELSH 5,059 | BOMAustralia 1,730 | LamaHCE 834 | Japan 690 | Germany 457 |
LamaHIce 73 — **no African, South American or mainland Asian stations at all**, which
is precisely why the Africa evaluation is not optional.

A note on the figure itself: the first version was misleading. Plotting in
value-sorted order puts extremes on top, and at this point density that repainted
whole regions in the tail colour — the gain panel read as mostly deep red when its
median is +0.026, and α as mostly deep purple when its median is 0.854. It now plots
in random order with spans from the 10–90% range and arrows marking clipped values.

## Was v2 trained long enough, and does the daily-only signal pick the right epoch?

Two questions that the numbers already on disk can answer without another GPU hour. Both
were prompted by noticing that the 30-epoch cap is **not** what stopped most v2 folds.

### The epoch cap was not binding; early stopping was, and it truncated the two splits unequally

All ten v2 pretrain folds wrote `DONE`, and every short fold stopped at exactly
`counter == patience` (6) -- these are legitimate early stops, not timeouts. Selection
tracks `val/median_kge`.

| run/fold | epochs | best@ | best val | slope over last 10 | counter at stop |
|---|---|---|---|---|---|
| v2_runB/fold0 | 30 | 25 | 0.6304 | +0.00059 | 5 |
| v2_runB/fold1 | 29 | 23 | 0.6328 | +0.00143 | 6 |
| v2_runB/fold2 | 30 | 27 | 0.6239 | +0.00048 | 3 |
| v2_runB/fold3 | 30 | 24 | 0.6301 | +0.00029 | 6 |
| v2_runB/fold4 | 30 | 28 | 0.6352 | +0.00146 | 2 |
| v2_blocked/fold0 | **20** | 14 | 0.6123 | **+0.00395** | 6 |
| v2_blocked/fold1 | 30 | 25 | 0.6398 | +0.00011 | 5 |
| v2_blocked/fold2 | 30 | 30 | 0.6296 | +0.00139 | 0 |
| v2_blocked/fold3 | 30 | 26 | 0.6323 | +0.00030 | 4 |
| v2_blocked/fold4 | **20** | 14 | 0.6243 | **+0.00257** | 6 |

The slope is positive in 10/10 folds (median +0.00099/epoch), so validation skill was
still creeping up everywhere -- but the per-epoch trend is about an eighth of the
epoch-to-epoch oscillation, so no single fold's stop is evidence of a plateau.

The asymmetry is the point. The two folds truncated earliest (blocked fold0 and fold4,
cut at epoch 20 with best@14) also carry the **steepest** remaining slopes, 3-13x the
runB folds'. So v2's early stopping cut the blocked configuration harder than the random
one. That makes the headline 0.007 gap (random M1 0.6277 vs blocked M1 0.6210) a
candidate artefact of unequal truncation rather than a cost of spatial blocking, and it
is why the v3 sensitivity check is worth running at all.

### Choosing the fine-tuning epoch from daily aggregates alone costs less than the noise

The whole premise is that the target domain has no hourly observations, so the transfer
stage must select its epoch on `holdout/daily_median_kge` -- a daily-aggregate signal.
`peek/target_hourly_median_kge` records the hidden hourly truth for diagnosis only (it
never touches selection). The gap between them is the price of the premise.

| config | median loss | mean | max | folds where selected != optimal |
|---|---|---|---|---|
| v2_runB (random) | +0.0053 | +0.0061 | +0.0150 | 3/5 |
| v2_blocked | +0.0000 | +0.0017 | +0.0070 | 2/5 |
| v2_replay025 | +0.0000 | +0.0027 | +0.0130 | 2/5 |
| all 15 | **+0.0000** | **+0.0035** | +0.0150 | 7/15 |

In 8 of 15 folds the daily-holdout criterion picks exactly the epoch an oracle with the
hidden hourly truth would pick. Pooled mean shortfall is +0.0035 KGE against an
epoch-to-epoch noise floor of 0.0078 (median |diff| of the peek series), so **the loss is
not measurable above the noise**. The apparent runB-vs-rest difference is not significant
(Kruskal-Wallis across the three runs p=0.567; the narrower one-sided runB-vs-rest
Mann-Whitney gives p=0.160, and neither supports a real difference) -- which also means
selection loss cannot explain the 0.007 gap, leaving unequal truncation as the one live
candidate.

This is a direct validation of the experimental premise, not a concession to it:
supervising and selecting with 24-h aggregates alone gives up nothing detectable
relative to seeing the hourly series.

Reproduce with `python -m scripts.convergence_check`; outputs land in
`outputs/convergence_check/` (`pretrain_truncation.csv`, `selection_loss.csv`,
`summary.json`).

### What v3 was changed to, and why it is not the single-variable run it started as

v3 was specified as "v2 with `train.epochs` 50 instead of 30". Given the table above that
design cannot answer the convergence question: with `patience` still 6 and counters
already at 2-6, folds terminate on noise rather than on the cap. v3 therefore now raises
`train.patience` 6 -> 10 as well, so the epoch cap is the binding constraint and the run
measures headroom instead of luck. It is reported as a **convergence check**, not as a
single-variable comparison against v2, and it does not enter the main results table.

v3 also resumes from v2's epoch-30 checkpoints rather than retraining from scratch. This
is exact, not an approximation: `lr_schedule` (`1:5e-4,12:1e-4,22:5e-5`) is keyed on
absolute epoch and carries no dependence on the total, and `apply_lr_schedule` reads only
the current epoch, so epochs 1-30 of a from-scratch v3 run are the same computation v2
already performed. `--resume` restores model, optimizer, `best`, `best_epoch`, `counter`
and history. One residual difference, stated because it is real: `epoch_subset`'s rng is
not checkpointed, so epochs 31+ draw a different same-distribution subset of training
batches than a single-shot 50-epoch run would -- equivalent to a seed change. v2 itself
has this property, since its sbatch always passes `--resume`. Provenance for every seeded
fold is in `outputs/v3_*/fold*/pretrain/SEEDED_FROM.txt`; the seeded configs were
diffed against the checkpoints' `run_meta.json` and differ only in `output_root`,
`train.epochs`, `train.patience` and `wandb.group`.


## v2, stratified: who the daily-only signal helps, and the one thing it makes worse

All three analyses below are evaluation-only and were run on v2's finished transfer
outputs; none of them needed the queued GPU jobs. Numbers are the random-split run
(`v2_runB`) unless stated. 8843 stations enter, 136 dropped as numerically degenerate.

Global picture: M0 0.527 -> M1 0.626, median gain **+0.062**, 68% of stations improve,
13% move by more than +-0.50. At M0 the model is **under**-dispersed (median alpha 0.808,
73% of stations below 1) -- the opposite sign to v1, where M0 over-dispersed by 6.8x, and
the direct consequence of the longer hourly look-back plus the forget-gate initialisation.
Fine-tuning pushes alpha up in every stratum, so the gain is largely a variance repair.

Map and latitude bands: `outputs/v2_stratify/maps/`.

### The gain is concentrated where zero-shot transfer was weakest

| stratum | n | M0 | M1 | gain | improved |
|---|---|---|---|---|---|
| LamaHIce (Iceland) | 73 | 0.2658 | 0.5937 | **+0.2475** | 89.0% |
| Japan | 690 | 0.5288 | 0.6219 | +0.0891 | 80.1% |
| LamaHCE | 834 | 0.6056 | 0.6982 | +0.0687 | 69.9% |
| Germany | 457 | 0.6172 | 0.6940 | +0.0672 | 67.2% |
| BOMAustralia | 1730 | 0.2482 | 0.3229 | +0.0574 | 60.4% |
| CAMELSH | 5059 | 0.5814 | 0.6696 | +0.0559 | 68.4% |

By latitude the same shape: >60 deg gains +0.2475, 0-30 deg +0.1138, 30-45 deg (where
most stations are) +0.0552.

Ranked by how much the gain varies across a covariate's strata, climate dominates:

| covariate | strata | gain range | spread |
|---|---|---|---|
| kgz_detailed | 15 | -0.0137 .. +0.3219 | **0.336** |
| source | 6 | +0.0559 .. +0.2475 | 0.192 |
| kgz_major | 5 | +0.0398 .. +0.1824 | 0.143 |
| area_km2 | 5 | +0.0222 .. +0.1119 | 0.090 |

Catchment area carries the strongest monotone trend (Spearman rho = -0.172, p = 7e-60):
the gain falls from +0.1119 in the smallest quintile (median 44 km2) to +0.0222 in the
largest (3200 km2), monotonically across all five bins. Small, fast catchments are where
the hourly branch has the most to learn and the daily aggregate still constrains it.

### Arid catchments are the exception, and the failure is in the tail

Hot-desert BWh (56 stations) is the only stratum with a negative median gain (-0.0137),
and its aggregate metrics get materially worse: median KGE -0.2012 at M0 -> **-0.4142** at
M1, with exactly 50.0% of stations improving. The two statistics disagree because they
measure different things -- the median *paired* gain is near zero while the *median of
each distribution* moves by -0.21 -- which means the typical arid station is unchanged and
a minority collapse. Arid B zone overall stays unusable either way (M1 median -0.0051).
So the honest claim is not "fine-tuning hurts drylands" but "fine-tuning leaves drylands
unfixed and destabilises a subset of them"; anything built on this should exclude or
special-case BWh rather than average over it.

### Daily-only fine-tuning recovers ~90% of the spatial-blocking penalty

Pairing the same 8709 stations across the random and blocked splits (every station is a
target exactly once in each, so composition is held fixed by construction):

| | paired median drop | stations worse | all agencies negative |
|---|---|---|---|
| M0 (zero-shot) | **-0.0594** | 63.7% | yes, 6/6 (p = 1.6e-168) |
| M1 (after daily-only fine-tune) | **-0.0061** | 52.2% | no -- Germany +0.0012 (p = 4.2e-06) |

**89.7% of the blocked-split penalty is recovered**, and afterwards whether a station
prefers the random or the blocked split is close to a coin flip (52.2%); the p-value is
tiny only because n = 8709. Note that `paired_split_effect`'s own verdict line reads
"the drop is NOT consistent across agencies, so composition cannot be ruled out" for M1 --
that heuristic just tests whether every agency is negative, and it fires because the drop
has shrunk until its *sign* is unstable. That instability is the result, not a warning.

Recovery per agency:

| agency | n | M0 drop | M1 drop | recovered |
|---|---|---|---|---|
| Germany | 458 | -0.0730 | +0.0012 | 102% |
| LamaHCE | 834 | -0.1213 | -0.0017 | 99% |
| CAMELSH | 5047 | -0.0501 | -0.0055 | 89% |
| BOMAustralia | 1607 | -0.0644 | -0.0076 | 88% |
| Japan | 690 | -0.0719 | -0.0106 | 85% |
| **LamaHIce** | 73 | -0.1515 | **-0.0942** | **38%** |

Five of six agencies recover 85-102%; Iceland recovers 38% and holds essentially the whole
residual. It is tempting to read this as recovery scaling with gauge density, and the v1
write-up said as much -- but across six agencies that relationship is not established
(Spearman rho = +0.257, p = 0.623). Iceland is one outlier, not a trend, and the claim
should be stated as such.

The mechanism is visible directly: under blocked splitting the gain *rises* with isolation,
from +0.0494 for stations ~62 km from the nearest other fold to +0.0881 at ~211 km
(rho = +0.047, p = 8e-06), whereas under random splitting it is flat to slightly negative
(rho = -0.035). Local daily observations substitute for spatial proximity, and they matter
most exactly where proximity is unavailable. Iceland is where even that substitution falls
short.

Note the two comparisons are different and both hold for Iceland: it gains the most from
fine-tuning in absolute terms (+0.2475) *and* keeps the largest residual blocked penalty
(-0.0942), because its zero-shot dependence on proximity was extreme enough that a large
gain still does not close the gap.

Reproduce with `python -m scripts.paired_split_effect --random-run v2_runB --blocked-run
v2_blocked --tag {M0,M1} --out-dir outputs/v2_split_effect`, `python -m
scripts.stratify_gain --run outputs/v2_runB/diagnostics_allhours --out-dir
outputs/v2_stratify` and `python -m scripts.global_map --run
outputs/v2_runB/diagnostics_allhours --out-dir outputs/v2_stratify/maps`. Recovery
fractions are in `outputs/v2_split_effect/recovery_by_agency.csv`.

`v2_blocked` and `v2_replay025` have no `diagnostics_allhours` yet, so the stratification
and map above are the random split only.

## Two v1 conclusions revisited under v2

Both analyses were already on disk for `v2_runB`; only the write-up was missing. Neither
needed a GPU. `v2_blocked` and `v2_replay025` still have no diagnostics, so these are the
random split.

### The intra-day jitter was a v1 artefact, and it is gone

v1 reported that M0 was "jittery, not smooth" -- the concern being over-dispersion rather
than the flattening the degenerate check was built to catch. v2 removes it outright.
Ratios to the observed median, over 8432 stations:

| ratio to observed | v1 runB M0 | v1 runB M1 | v2 runB M0 | v2 runB M1 |
|---|---|---|---|---|
| flashiness | **6.80x** | 2.00x | **0.95x** | **1.02x** |
| within-day std | 3.11x | 1.52x | 0.86x | 0.89x |
| within-day range | 2.82x | 1.53x | 0.88x | 0.91x |
| q95 events / year | 1.62x | 1.79x | 0.99x | 1.02x |
| mean flow | **0.50x** | 1.01x | 1.05x | 1.03x |

v1's zero-shot model was 6.8x too flashy, 3.1x too variable within the day, produced
1.6-1.8x too many high-flow events, and carried only half the observed mean; fine-tuning
pulled flashiness down to 2.0x but no further. v2 is calibrated before fine-tuning
(0.95x, mean 1.05x) and stays calibrated after it (1.02x). The longer hourly look-back
plus the forget-gate initialisation is what changed -- the same pair that turned M0's
alpha from 6.8x over-dispersed into 0.808 under-dispersed.

**A gap in the check itself, worth stating:** `degenerate_check` emits the same verdict
for all three runs -- "intra-day variability survives, so the daily-aggregate term is not
being satisfied by flattening the day". That is literally true every time, because the
test only looks for collapse. A model 6.8x too flashy passes it. The verdict string should
not be quoted as evidence that intra-day behaviour is *correct*; only the ratio table
supports that, and only for v2.

### KGE-vs-error divergence: the sign reverses, the magnitude gap does not

Per-station paired tests on sample-level absolute error (mm/h), BH-FDR at alpha 0.05,
8862 stations:

| | v1 runB | v1 blocked | v2 runB |
|---|---|---|---|
| significantly improved | 3712 (41.9%) | 3784 (42.7%) | **4712 (53.2%)** |
| significantly degraded | 4405 (**49.7%**) | 4492 (**50.7%**) | 3337 (37.7%) |
| median error reduction | **-0.0002** | **-0.0004** | **+0.0003** |
| pooled median dKGE | +0.0214 | +0.0421 | **+0.0576** |

Under v1 more stations significantly *degraded* than improved and the median error
reduction was negative: fine-tuning improved KGE while making point-wise error worse for
the majority. Under v2 that reverses -- 53.2% improved against 37.7% degraded, and the
median error reduction turns positive.

But the reversal should not be oversold, because the magnitudes are not comparable.
v2's median error reduction is **+0.00034 mm/h against an observed mean flow of 0.0483
mm/h -- 0.7% of it** -- while the pooled median KGE gain is +0.0576. So the honest v2
claim is that daily-aggregate fine-tuning **no longer damages point-wise accuracy**, not
that it improves it: the KGE gain comes from variance and bias calibration (see the alpha
column throughout), which KGE rewards and mean absolute error largely does not. The two
metrics still agree on only 67.9% of stations (Spearman 0.460), so a station-level claim
should say which metric it is made under.

Pooled Wilcoxon p = 1.5e-256 over 8849 pairs; 8049 of 8862 stations survive BH-FDR
against 443 expected by chance.

Sources: `outputs/v2_runB/degenerate/degenerate_summary.json`,
`outputs/v2_runB/significance/significance_summary.json`.

## Provenance: the config files are not the authority for v1

`configs/phase1.yaml`, `configs/phase1_runB.yaml`, `configs/phase1_runB_blocked.yaml` and
`configs/phase1_runB_replay.yaml` all now carry `initial_forget_bias: 3`, but **every v1
result in this file was produced without it** -- the field was added to those files when
the forget-gate initialisation was implemented, after the v1 runs had finished, and no
snapshot was kept. Re-running one of them today produces a v1 look-back (72 h) with a v2
forget gate: a configuration that was never evaluated and appears nowhere in this file.

Diffing each v1 config against the `run_meta.json` its own run wrote shows the drift is
exactly one field and nothing else:

| config | field | at runtime | in the file now |
|---|---|---|---|
| phase1.yaml | model.initial_forget_bias | `None` | `3` |
| phase1_runB.yaml | model.initial_forget_bias | `None` | `3` |
| phase1_runB_blocked.yaml | model.initial_forget_bias | `None` | `3` |

(`phase1_runB_replay.yaml` has no pretrain `run_meta.json` to compare, because replay
reuses run B's pretrained weights rather than training its own -- the same arrangement as
`v2_replay025` and `v3_replay025`.)

**The authority for what any run used is `outputs/<run>/fold*/pretrain/run_meta.json`,**
which stores the fully resolved config at launch. Every v1-vs-v2 comparison in this file
takes its provenance from there, not from the config files, and the same check is what
licensed seeding v3 from v2's checkpoints. A warning to this effect is now inline at the
`initial_forget_bias` line of all four configs.

The general lesson, since this will recur: a config file tracks the *current* intent of an
experiment, not the history of what was run. Anything that must stay reproducible needs
its parameters captured at launch into the output directory, which `run_meta.json` does --
and any comparison across versions should read from there.

## How the blocked split catches up: it is a timing repair

The 89.7% recovery established earlier says *that* daily-aggregate fine-tuning undoes
spatial blocking's penalty; the component decomposition says *how*. Paired per-station
medians over 8979 target stations, all-hours evaluation index:

| | r (timing) | alpha (variance) | beta (bias) | KGE |
|---|---|---|---|---|
| v2 random M0 -> M1 | 0.7970 -> 0.8122 (**+0.0054**) | 0.8133 -> 0.8508 (+0.0370) | 1.0246 -> 0.9960 | 0.5190 -> 0.6210 |
| v2 blocked M0 -> M1 | 0.7699 -> 0.8077 (**+0.0195**) | 0.8085 -> 0.8646 (+0.0607) | 1.0202 -> 0.9991 | 0.4185 -> 0.6165 |
| v2 replay M0 -> M1 | 0.7970 -> 0.8131 (+0.0060) | 0.8133 -> 0.8430 (+0.0257) | 1.0246 -> 1.0109 | 0.5190 -> 0.6182 |

Reading it as blocked-minus-random, which is what the 0.007 headline gap is made of:

| component | gap at M0 | gap at M1 | recovered |
|---|---|---|---|
| **r (timing)** | **-0.0271** | **-0.0045** | **83%** |
| alpha (variance) | -0.0048 | **+0.0138** | overshoots -- blocked ends up better |
| beta (bias) | -0.0044 | +0.0030 | overshoots slightly |

**Spatial blocking costs timing, almost nothing else.** At M0 the blocked model's r is
0.0271 below the random split's while its alpha and beta are within 0.005 -- removing a
target basin's neighbours degrades *when* the model thinks the water arrives, not how much
of it there is or how variable it is. Fine-tuning then recovers 83% of that timing deficit,
and pushes alpha past the random split's rather than merely matching it.

This is the part worth stating carefully, because it is counter-intuitive: **a 24-hour
aggregate contains no sub-daily timing information, yet supervising on it recovers most of
the sub-daily timing deficit.** The route is architectural rather than statistical. The
hourly branch does not read the daily labels; it inherits its initial hidden and cell state
from the daily branch through `transfer_h`/`transfer_c` at `transfer_index`. Fine-tuning
the daily branch on local daily observations gives a better catchment state -- storage and
wetness -- and a better state changes *when* the hourly branch releases water. Daily data
fixes hourly timing indirectly, through the state handoff that is the whole point of the
two-branch design. A single-branch hourly model given the same daily-only supervision has
no such channel.

It also explains why replay does not help under v2: replay's r gain (+0.0060) is
indistinguishable from plain run B's (+0.0054), and its alpha gain is smaller (+0.0257 vs
+0.0370). Mixing source batches back in protects the source domain but adds nothing to the
target's timing.

### A labelling bug found while reading this, fixed

`summarize_components` computed `frac_worse` as `(M1 - M0) < 0` for every component. That
is right for r, KGE and NSE, which are higher-is-better, but wrong for alpha and beta,
whose ideal is 1.0 -- and since the median beta sits *above* 1 (~1.02), a decrease there is
usually an improvement. The column therefore reported 52-55% of stations as "worse" on beta
while beta's median was moving toward 1.0, a flat contradiction. Corrected to score
two-sided components by whether `|value - 1|` grew:

| | alpha, as reported | alpha, corrected | beta, as reported | beta, corrected |
|---|---|---|---|---|
| v2 run B | 41.2% | **35.0%** | 55.0% | **40.9%** |
| v2 blocked | 41.1% | **30.0%** | 52.1% | **34.2%** |
| v2 replay | 42.8% | **35.7%** | 52.6% | **39.6%** |

The old rule overstated degradation everywhere it applied. It was never quoted in this file
or in the Word report -- `paired_split_effect`'s own `frac_worse` is computed on KGE, where
higher-is-better makes the sign test correct -- so no published number changes. The summary
CSVs for all eight diagnostic runs were regenerated from their unchanged per-station tables,
and each row now carries a `worse_criterion` column naming the rule applied, so the column
cannot be read under the wrong one again.

## run A under v2: the sampled daily branch still degrades

run A's v2 finished 2026-08-20, filling the one hole in the main table. It does not
change the v1 verdict:

| | M0 | M1 | ΔKGE | source Δ |
|---|---|---|---|---|
| run A v1 | 0.4275 | 0.3973 | **-0.0303** | -0.0387 |
| run A v2 | 0.4241 | 0.3965 | **-0.0276** | -0.0381 |
| run B v2 | 0.5318 | 0.6277 | +0.0959 | -0.0882 |

**Daily-aggregate fine-tuning makes run A worse, under both configurations.** The longer
hourly look-back and the forget-gate initialisation -- which took run B's gain from
+0.0449 to +0.0959 -- move run A by +0.0027, i.e. nothing. So the defect is in run A's
data path, not in the training configuration: feeding the daily branch a strided sample
of hourly values instead of true daily means gives the transfer step nothing usable to
calibrate against. This was the v1 conclusion and it survives the change that rescued
every other configuration.

## v3, so far: longer training improves the source model and not the target

v3 raises `train.epochs` 30 -> 50 and `train.patience` 6 -> 10 to test whether v2's
early stopping had truncated it, and whether it truncated the two splits unequally
(§4.8). The pretrain answer is yes on both counts; the target-domain answer is no.

Pretrain selection metric, gain over v2 per fold:

| | fold0 | fold1 | fold2 | fold3 | fold4 | mean |
|---|---|---|---|---|---|---|
| v3 run B | +0.0048 | 0 | +0.0093 | +0.0048 | 0 | **+0.0038** |
| v3 blocked | **+0.0142** | 0 | +0.0057 | +0.0027 | **+0.0105** | **+0.0066** |

The prediction from §4.8 holds exactly: the two folds that v2 truncated earliest --
blocked fold0 and fold4, both stopped at epoch 20 with best@14 and the steepest residual
slopes -- are the two that gained most, and blocked's mean gain is 1.7x run B's. Several
folds also ran the full 50 epochs with their best at or near epoch 50, so even 50 is
still a binding cap for them. v2 was therefore under-trained, and unequally so.

**None of it reaches the target domain.**

| | M1 under v2 | M1 under v3 | change |
|---|---|---|---|
| run B | 0.6277 | 0.6270 | **-0.0007** |
| replay 0.25 | 0.6252 | 0.6255 | +0.0003 |
| blocked | 0.6210 | pending | — |

A +0.0038 gain in the pretrain selection metric produces a -0.0007 change in target-domain
M1 -- nothing, in the direction of nothing. The mechanism is unremarkable once stated: the
pretrain metric is measured on the *source* domain, while M1 is what survives a fine-tune
on the *target* domain. The fine-tune re-adapts the model either way, so a slightly better
starting point does not have to produce a better end point.

The consequence for the headline is the part that still needs blocked's transfer. If
blocked behaves like run B -- pretrain gain not reaching M1 -- then the 0.007 random-vs-
blocked gap is **not** an artefact of unequal truncation, and v2's main table stands as
written. If instead blocked's larger pretrain gain does reach its M1, the gap narrows and
should be reported as a truncation artefact. Run B's result makes the first outcome more
likely, but blocked is the configuration that was actually truncated, so the question is
not settled by run B alone. `50660661` is queued.

## v3, settled: the 0.007 gap does not survive, and the pipeline's own noise is larger

v3's transfers finished 2026-08-21. Taking the paired per-station comparison that the
0.007 headline rests on -- blocked M1 minus random M1 over the same 8,709 catchments:

| | paired median gap | stations worse | p |
|---|---|---|---|
| under v2 (30 epochs) | **-0.0061** | 52.2% | 4.2e-06 |
| under v3 (50 epochs) | **-0.0015** | **50.5%** | 4.4e-02 |

The gap narrows by 0.0039 (paired over stations, p = 4.2e-03), and at 50.5% worse it is a
coin flip. So the answer to the question v3 was built for is: **the blocked-vs-random gap
does not survive longer training**, and it should not be reported as a robust cost of
spatial blocking.

### The finding that limits this one: the transfer stage is not reproducible to better than ~0.01

Found by chasing an inconsistency rather than by looking for it. Three folds carried
*bit-identical* pretrained weights between v2 and v3, because their early stopping had
already terminated and `best_model.pth` was never rewritten. Two of them reproduced
exactly; one did not:

| fold | weights | transfer epoch chosen | holdout daily KGE | M1 difference |
|---|---|---|---|---|
| run B fold1 | identical | 9 -> 9 | 0.719465 -> 0.719465 | **0.0000** |
| run B fold4 | identical | 9 -> 9 | 0.708492 -> 0.708492 | **0.0000** |
| blocked fold1 | **identical** | **12 -> 12** | 0.698512 -> 0.698**289** | **+0.0107** |

Same weights, same config, same seed, same selected epoch -- and M1 moves 0.0107. The
holdout metric differs only in its fourth decimal, so this is numeric non-determinism in
the fine-tune (kernel selection and non-deterministic reductions differ across hardware),
accumulating over 12 epochs into a target-domain difference an order of magnitude larger
than its cause. Two folds reproducing bit-exactly and one not is the signature of
hardware, not of a bug: jobs land on whatever node is free, which since 2026-08-19 can be
either a v100 or an a100.

**This is larger than the effect it was measuring.** 0.0107 at fold level against a 0.007
headline gap and a 0.0039 narrowing. Assuming per-fold noise near 0.01 and independence
across folds, a five-fold aggregate carries about 0.0045, so the narrowing is roughly
0.9 sigma.

It also exposes a real weakness in how the gap was tested. The paired per-station test has
8,709 replicates but **one run per configuration**, and run-level noise is shared across
every station in a run, so pairing stations cannot remove it. The tiny p-values
(4.2e-06, 4.2e-03) treat stations as independent replicates of a difference whose dominant
uncertainty is at the run level. They are not wrong about the stations; they are answering
a narrower question than the one being asked.

### What to report

1. The v2 main table stands as the primary result and v3 stays out of it, per its design
   (it changes two settings, so it is not a single-variable comparison).
2. The 0.007 random-vs-blocked M1 gap should be reported as **not distinguishable from
   zero**, with both reasons given: it narrows to -0.0015 under longer training, and it is
   smaller than the pipeline's run-to-run reproducibility.
3. The *zero-shot* blocked penalty is unaffected by any of this and remains solid:
   -0.0594 paired, 63.7% of stations worse, all six agencies negative, p = 1.6e-168. That
   is an order of magnitude above the noise discussed here. It is the M1 residual, not the
   M0 penalty, that dissolves.
4. Any future claim at the 0.01 level needs repeated runs per configuration, not more
   stations. Seeds vary nothing that matters here -- the variation is in the hardware --
   so the repeats must be actual reruns.

For completeness, the pretrain side of v3 did exactly what §4.8 predicted: blocked gained
+0.0066 on the selection metric against run B's +0.0038, with the two largest gains in the
two folds v2 truncated earliest (fold0 +0.0142, fold4 +0.0105). v2 was under-trained and
unequally so. That asymmetry is real; what it buys in the target domain is not separable
from noise.

## Reproducing

**Read this before copying the commands.** A pretrain run is ~9 h (30 epochs at ~17.9
min), but the sbatch scripts declare **4 h**, deliberately: that is the `gpu4` partition
boundary, and gpu4 carries 95 nodes against 46/31/58 for gpu24/gpu72/gpu. Declaring 5 h
loses access to the largest pool and measurably changed queue latency from minutes to
days. So a full pretrain must be **chained**, not submitted as one job. Submitting the
single-job form silently produces a truncated run, and because a wall-clock kill exits
non-zero, an `afterok` dependant then never starts.

```bash
# one-off
sbatch slurm/01_build_cache.sbatch               # 58 GB memmap from 6sources.nc
sbatch slurm/04_build_eval_index.sbatch          # all-hours evaluation index

# which config maps to which output, and what each run actually used
python -m scripts.inventory                      # also flags configs that have drifted

# a full run, per config. CONFIG selects the experiment; see scripts.inventory.
CFG=configs/phase1_runB_v2.yaml
H=$(CONFIG=$CFG sbatch --parsable slurm/10_pretrain_runB.sbatch)
for i in 1 2; do H=$(CONFIG=$CFG sbatch --parsable --dependency=afterany:$H slurm/10_pretrain_runB.sbatch); done
CONFIG=$CFG sbatch --dependency=afterok:$H slurm/20_transfer_runB.sbatch
```

`afterany` between chunks, not `afterok`: a chunk hitting the wall exits non-zero by
design and the next chunk is what handles it. `--resume` picks up the checkpoint; the
already-complete guard makes surplus chunks exit 0 in seconds, so over-provisioning the
chain is cheap. Point the final `afterok` at the last chunk.

For a variant that reuses another run's pretrained weights -- the replay configs do this
on purpose -- skip the pretrain chain and pass `PRETRAIN_ROOT`:

```bash
CONFIG=configs/phase1_runB_replay_v2.yaml PRETRAIN_ROOT=outputs/v2_runB \
  sbatch slurm/20_transfer_runB.sbatch
```

```bash
# diagnostics and analyses -- evaluation only, no retraining
CFG=configs/phase1_runB_v2.yaml; OUT=outputs/v2_runB/diagnostics_allhours
D=$(CONFIG=$CFG INDEX=samples_evalhours.npz OUT_ROOT=$OUT sbatch --parsable slurm/30_diagnose.sbatch)
MERGE=1 CONFIG=$CFG OUT_ROOT=$OUT sbatch --dependency=afterany:$D --array=0 slurm/30_diagnose.sbatch
CONFIG=$CFG sbatch slurm/32_degenerate.sbatch
CONFIG=$CFG sbatch slurm/33_significance.sbatch

# CPU only, from finished outputs
python -m scripts.stratify_gain       --run outputs/v2_runB/diagnostics_allhours --out-dir outputs/v2_stratify
python -m scripts.global_map          --run outputs/v2_runB/diagnostics_allhours --out-dir outputs/v2_stratify/maps
python -m scripts.paired_split_effect --random-run v2_runB --blocked-run v2_blocked --tag M0 --out-dir outputs/v2_split_effect
python -m scripts.paired_split_effect --random-run v2_runB --blocked-run v2_blocked --tag M1 --out-dir outputs/v2_split_effect
python -m scripts.convergence_check
python -m scripts.build_report --out reports/PhaseI_report.docx

# Africa (needs the rescaled forcing, not raw ERA5-Land)
PRESET=forcing sbatch slurm/40_basin_average.sbatch
python -m scripts.rescale_africa_forcing
CONFIG=configs/phase1_runB_v2.yaml KIND=transfer sbatch slurm/41_africa.sbatch
CONFIG=configs/phase1_runB_v2.yaml sbatch slurm/42_africa_transfer.sbatch      # in-situ
python -m scripts.africa_insitu_ensemble --insitu-glob outputs/v2_africa_insitu_fold
```

### What is and is not preserved

| | where | recreatable from |
|---|---|---|
| code, configs, sbatch | git, 81 commits, no remote | — (single copy) |
| results, weights, logs | `outputs/` -> `/ibex/user/kongw0a/global_mtslstm_outputs`, 1.2 GB, gitignored | rerunning everything (~hundreds of GPU-hours) |
| hourly cache, 58 GB | `/ibex/user/kongw0a/hourly_cache` | `scripts.build_hourly_cache` from `6sources.nc` |
| Africa forcing, 137 GB | `/ibex/user/kongw0a/era5_land_africa_forcing` | `scripts.download_era5_land_africa` + rescale |
| prepared batches, root data | `/ibex/project/c2266/.../hourly_q_dl/` | — (upstream) |

The derived 195 GB is cheap to rebuild; `outputs/` is not, and has no second copy. Note
the precedent: the 100-station experiment's prepared dataset has already been deleted from
`/ibex/project`, and its raw source under `/mnt/datawaha` is not mounted on this cluster
any more, so those results can no longer be regenerated here at all.

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
- **`initial_forget_bias` is unimplemented in every run reported here** (see the
  caveat under "The bigger lever"). Gauch et al. set it to 3; neither this code nor
  the 100-station reference does. Comparative conclusions are unaffected; absolute
  levels and the interpretation of α may be. The fold-1 search tests it directly.
- **M2 (symbolic prior) is not being ported, deliberately.** Its expression is fitted
  on CAMELS-US attributes whose global counterparts differ in scale by 2-400x, and one
  term (`PERMAVE`, average permeability) has no same-quantity column in the global
  table at all — `NSIDC_permafrost` is permafrost extent, a different variable, and
  `cos(PERMAVE^2)` is extremely scale-sensitive. Beyond portability, the method
  corrects **daily-branch bias**, while the deficit diagnosed throughout this work is
  **hourly variance ratio**; the two do not address the same thing. PLAN.md marks M2
  optional. The hyperparameter search IS running (fold 1, 26 variants). PLAN.md marks
  M2 optional; the search was to run on fold 1 only. Current hyperparameters are
  hand-set and frozen so the runs stay comparable, which is fine for every relative
  conclusion here but must be stated if absolute performance is quoted.
- ~~The Africa protocol applies temperate-fine-tuned models to African basins.~~ **Done**
  (see "Africa, run properly"): five folds of in-situ fine-tuning give paired ΔKGE
  +0.611, 92.3% of basins improved, beating the continent-holdout PUB baseline. The
  earlier temperate-transfer numbers remain valid as an extrapolation test and are
  reported as such.
- Blocked-split models underperform random-split ones on Africa (+0.078 vs +0.143).
  Recorded, not explained.
