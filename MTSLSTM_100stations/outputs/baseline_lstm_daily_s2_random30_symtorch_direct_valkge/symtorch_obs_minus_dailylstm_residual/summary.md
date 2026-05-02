Residual SymTorch distillation of the validation-KGE-best daily BaselineLSTM

Residual definition:
- residual = observed daily streamflow - dailyLSTM output
- corrected prediction = dailyLSTM output + symbolic residual

Workflow:
- Reuse the flattened daily input windows from the best validation-KGE dailyLSTM
- Use a DummyModel whose output is the observed-minus-dailyLSTM residual on the distillation sample
- Distill one symbolic equation for that residual

Julia executable: /home/kongw0a/miniconda3/envs/mtslstm_symtorch/julia_env/pyjuliapkg/install/bin/julia
Train distill sample size: 3810
Total flattened input dimension: 297
PySR select_k_features: 16

Best residual equation:
(Rainf_lag2 / exp(Tair_lag11)) * 0.024163548

Equation complexity: 9
Equation loss: 0.05518841

Selected features used by feature selection:
- PotEvap_lag89
- Rainf_lag85
- Tair_lag78
- PotEvap_lag70
- Tair_lag65
- Tair_lag64
- Tair_lag54
- PotEvap_lag43
- PotEvap_lag42
- Rainf_lag22
- Tair_lag11
- Rainf_lag9
- Rainf_lag2
- PotEvap_lag1
- DRAIN_SQKM
- STRAHLER_MAX

Residual fidelity against observed-minus-dailyLSTM target:
- train: RMSE(std)=0.238424, MAE(std)=0.100957, R2(std)=-0.039260, corr(std)=0.088808
- val: RMSE(std)=0.395051, MAE(std)=0.147482, R2(std)=-0.032483, corr(std)=0.018263
- test: RMSE(std)=0.379737, MAE(std)=0.156285, R2(std)=-0.034061, corr(std)=-0.000125

Hydrology metrics against observed daily streamflow:
- train: dailyLSTM median KGE=0.815785, dailyLSTM median NSE=0.842795, corrected median KGE=0.837248, corrected median NSE=0.754454
- val: dailyLSTM median KGE=0.662898, dailyLSTM median NSE=0.663122, corrected median KGE=0.605895, corrected median NSE=0.463162
- test: dailyLSTM median KGE=0.640016, dailyLSTM median NSE=0.577719, corrected median KGE=0.486775, corrected median NSE=0.217404
