# SymTorch + MTSLSTM Research Plan

## Working title

Symbolically distilled transferable priors for improving cross-basin generalization of MTSLSTM under limited target-domain hourly observations

## Core goal

Use `SymTorch` to extract interpretable symbolic relations from a target-domain daily model, then inject those relations as transferable priors into the `MTSLSTM` transfer-learning pipeline so that hourly prediction on the S2 Cfa-SE 30-station target set improves.

The final evaluation remains on hourly metrics:

- hourly KGE
- hourly NSE

## Current project status

We already have:

1. Source model
   Best 100-station pretrained `MTSLSTM` selected by validation KGE.

2. Target domain
   Random 30 stations from the S2 Cfa-SE region.

3. Existing target adaptation baseline
   Daily-supervised transfer learning that updates:
   - `lstm_daily`
   - `transfer_h`
   - `transfer_c`
   - `head_daily`
   - `head_hourly`

4. Current best transfer-learning setup
   Target domain uses daily supervision but final evaluation is hourly.

## Key research question

Can symbolically distilled interpretable relations serve as transferable priors that improve hourly cross-basin generalization beyond standard daily-supervised transfer learning?

## Main hypothesis

The target domain contains stable daily-scale hydrologic relations that can be distilled into symbolic expressions.

If these symbolic relations are injected into the `MTSLSTM` transfer process as soft priors, the model should:

- adapt more stably to the target basins
- reduce negative-station failures
- improve hourly prediction metrics
- provide interpretable physical insight into why transfer works

## Recommended method direction

### High-level idea

Do not directly symbolically replace the entire `MTSLSTM`.

Instead:

1. train a target-domain `dailyLSTM` teacher on the 30 stations
2. distill symbolic relations from that daily model or from its residual behavior
3. convert the symbolic relation into a prior loss
4. add that prior loss to the current `MTSLSTM` target-domain transfer learning

This is safer and more scientifically interpretable than directly forcing symbolic structure onto the full hourly model.

## Why this is better than re-training the 100-station source model from scratch

The original idea was:

1. train a `dailyLSTM` on the 30 stations
2. distill symbolic equations
3. use those equations as loss constraints when retraining the 100-station `MTSLSTM`

This is possible, but not the recommended first experiment because:

1. the symbolic relation is learned from the target region, so it is not necessarily a globally valid source-domain rule
2. injecting a target-specific equation into full 100-station source retraining may distort useful source-domain structure
3. it becomes harder to isolate whether improvement comes from:
   - target adaptation
   - symbolic priors
   - a changed source training procedure

Therefore the preferred first version is:

- keep the pretrained 100-station `MTSLSTM`
- perform target adaptation on the 30 stations
- use the symbolic model as a target-domain prior during fine-tuning

## Most stable first method

### Stage 1: Train a target-domain daily teacher

Train a simple `dailyLSTM` on the 30 stations using daily aggregated data.

Inputs can include:

- daily precipitation aggregates
- daily PET aggregates
- daily temperature aggregates
- static basin attributes
- optionally antecedent indices

Output:

- next-day or same-step daily streamflow target, aligned with the current daily branch setup

### Stage 2: Build interpretable feature set Z

Construct a compact hydrologically meaningful feature vector `Z`, for example:

- `P_1, P_3, P_7, P_30`
- `PET_1, PET_7, PET_30`
- `API_7, API_30`
- `Tair_1, Tair_7`
- `P - PET`
- `P / PET`
- `DOY_sin`, `DOY_cos`
- static attributes:
  - `aridity_index`
  - `BFI`
  - `slope`
  - `drainage_area`
  - `soil texture`

The goal is to give `SymTorch` features that are both learnable and interpretable.

### Stage 3: Distill a symbolic prior

The strongest first target is not the raw daily prediction itself, but the target-domain correction relative to the source model.

Recommended formulation:

`g_res(Z) ~= y_daily_target - y_daily_source`

This means the symbolic model learns the residual correction that the target region needs beyond the pretrained source model.

Why this is recommended:

- easier to learn than the full hydrologic mapping
- more clearly transferable as a prior
- less likely to fight the source model

### Stage 4: Inject symbolic prior into MTSLSTM transfer learning

Start from the current transfer loss:

`L = L_daily + lambda_agg * L_agg`

Then extend to:

`L = L_daily + lambda_agg * L_agg + lambda_sym * L_sym`

Recommended first symbolic term:

`L_sym = || D_pred - (D_src + g_res(Z)) ||^2`

Where:

- `D_pred` = current daily branch prediction from the fine-tuned `MTSLSTM`
- `D_src` = daily prediction from the frozen pretrained source `MTSLSTM`
- `g_res(Z)` = symbolic residual prior

This is a soft constraint, not a hard replacement.

## Why this should affect hourly output

In the current `MTSLSTM`:

- the daily branch generates hidden states
- those states are transferred through `transfer_h` and `transfer_c`
- the hourly branch uses those transferred states for initialization

So if the symbolic prior shapes the daily branch during target adaptation, its effect can propagate to hourly predictions through the existing daily-to-hourly transfer path.

This is why symbolic supervision on the daily side can still improve hourly KGE/NSE.

## Optional stronger version

After the first version works, a stronger but riskier variant is:

### Transfer-state symbolic prior

Instead of constraining only daily output, constrain the transferred hidden states:

- `h_H0`
- `c_H0`

Because those directly initialize the hourly branch, this may affect hourly skill more strongly.

However:

- those states are harder to interpret directly
- a low-dimensional projection such as PCA should probably be used first

Recommended advanced version:

1. compute low-rank projections of `h_H0` and `c_H0`
2. distill symbolic relations for those projections
3. regularize the transferred states toward the symbolic targets

This should be treated as a second-stage experiment, not the first one.

## Experimental design

### Models to compare

1. `A: Zero-shot source model`
   Best 100-station `MTSLSTM` evaluated directly on the 30 target stations.

2. `B: Daily-supervised transfer`
   Current transfer-learning baseline without symbolic prior.

3. `C1: Daily-supervised transfer + black-box prior`
   Use an MLP prior instead of a symbolic prior.

4. `C2: Daily-supervised transfer + symbolic prior`
   Main proposed method.

5. Optional `C3: Transfer-state symbolic prior`
   Advanced version if time allows.

### Why model C1 is important

It answers whether gains come from:

- any extra prior model

or specifically from:

- symbolic distilled priors

Without `C1`, reviewers can reasonably argue that the symbolic component is unnecessary.

## Main evaluation metrics

Primary:

- median hourly KGE on target test stations
- median hourly NSE on target test stations

Secondary:

- number of target stations with negative KGE
- number of target stations with negative NSE
- station-level improvement counts versus baseline transfer
- hydrograph peak timing or local peak lag
- robustness across repeated random 30-station draws

## Minimum publishable evidence

To make this direction convincing, the method should show at least some of the following:

1. better hourly test KGE/NSE than ordinary transfer learning
2. fewer negative-performance stations
3. lower variance across different random target station selections
4. symbolic relations that are hydrologically interpretable
5. symbolic prior performing at least comparably to the black-box prior

## Biggest risks

1. The symbolic model may overfit the 30-station target region.
2. The symbolic expression may be unstable across random splits.
3. The prior may improve daily behavior but not hourly behavior.
4. Complexity may increase without enough gain.

## Risk mitigation

1. Start with residual symbolic priors instead of direct-output symbolic priors.
2. Use a compact, interpretable `Z` feature set.
3. Compare against a black-box prior baseline.
4. Keep the symbolic prior as a soft regularizer, not a hard replacement.
5. Evaluate multiple random target station draws if computationally feasible.

## Practical value

If successful, this approach has practical value because it may:

- improve adaptation when hourly target observations are limited
- reduce failure cases in new basins
- provide physically interpretable relations for transfer
- improve user trust compared with purely black-box transfer

## Suggested implementation order

### Phase 1

Train a target-domain `dailyLSTM` teacher and verify daily performance.

### Phase 2

Build daily interpretable features `Z` and train a black-box prior model.

### Phase 3

Use `SymTorch` to distill a symbolic residual prior:

`g_res(Z)`

### Phase 4

Add `L_sym` to the current `MTSLSTM` transfer-learning pipeline.

### Phase 5

Run the four key model comparisons:

- A
- B
- C1
- C2

## Recommended next coding task

The most useful next step is:

1. define the daily interpretable feature set `Z`
2. build a `dailyLSTM` teacher on the 30 stations
3. compute source-model daily predictions on the same windows
4. construct residual targets:
   `y_daily_target - y_daily_source`
5. distill a symbolic residual model with `SymTorch`

This gives the minimum viable symbolic prior before modifying the existing `MTSLSTM` transfer code.
