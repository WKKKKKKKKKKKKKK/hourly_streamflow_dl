Additional check on the lowest-loss hall-of-fame equation

This note compares the default `model_selection="best"` equation against the lowest-loss equation in `equations_dim0.csv`.

Lowest-loss equation:

```text
((((((((-1.569884 + low_prec_freq) + (sin(PotEvap_lag60) - DRAIN_SQKM)) * -0.7946811) + Rainf_lag1) + ELEV_STD_M_BASIN) + (DRAIN_SQKM + DRAIN_SQKM)) * ((-1.9482533 - (STRAHLER_MAX - PotEvap_lag44)) * -0.05476038)) - 0.30561286) + -0.027748965
```

Its hall-of-fame properties are:
- complexity: `32`
- loss: `0.18593192`

Fidelity to the original dailyLSTM prediction:
- train: `RMSE(std)=0.437376`, `R2(std)=0.742301`
- val: `RMSE(std)=0.423254`, `R2(std)=0.720322`
- test: `RMSE(std)=0.430476`, `R2(std)=0.688540`

Hydrology metrics against observed daily streamflow:
- train: `median KGE=0.099833`, `median NSE=-0.003079`
- val: `median KGE=-0.239824`, `median NSE=-0.330000`
- test: `median KGE=-0.185430`, `median NSE=-0.383842`

Interpretation:
- The lower-loss equation preserves the dailyLSTM output better than the default `best` equation.
- Even so, direct MJO-style symbolic distillation on flattened windows still does not preserve useful basin-wise hydrologic skill well enough.
- This suggests the next step should be a more structured distillation target, such as a residual prior, rather than relying only on direct sequence-to-output symbolic fitting.
