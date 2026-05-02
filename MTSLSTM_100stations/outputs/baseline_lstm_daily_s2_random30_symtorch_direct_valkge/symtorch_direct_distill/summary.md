Direct SymTorch distillation of the validation-KGE-best daily BaselineLSTM

Workflow:
- Follow the MJO_prediction.ipynb logic
- Use flattened daily input windows directly
- Use a DummyModel whose output is the trained dailyLSTM prediction on the distillation sample
- Distill one symbolic equation for the dailyLSTM output

Julia executable: /home/kongw0a/miniconda3/envs/mtslstm_symtorch/julia_env/pyjuliapkg/install/bin/julia
Train distill sample size: 9000
Total flattened input dimension: 297
PySR select_k_features: 16

Best symbolic equation:
((PotEvap_lag44 - 3.4442365) * DRAIN_SQKM) * -0.19052482

Equation complexity: 7
Equation loss: 0.26043457

Selected features used by feature selection:
- PotEvap_lag60
- PotEvap_lag55
- PotEvap_lag54
- PotEvap_lag48
- PotEvap_lag46
- PotEvap_lag45
- PotEvap_lag44
- PotEvap_lag33
- PotEvap_lag29
- Rainf_lag2
- Rainf_lag1
- low_prec_freq
- DRAIN_SQKM
- ELEV_STD_M_BASIN
- SLOPE_PCT
- STRAHLER_MAX

Fidelity to dailyLSTM predictions:
- train: RMSE(std)=0.522171, MAE(std)=0.293338, R2(std)=0.632693, corr(std)=0.798538
- val: RMSE(std)=0.487157, MAE(std)=0.282343, R2(std)=0.629494, corr(std)=0.801373
- test: RMSE(std)=0.510874, MAE(std)=0.298488, R2(std)=0.561334, corr(std)=0.758626

Hydrology metrics against observed daily streamflow:
- train: LSTM median KGE=0.815785, LSTM median NSE=0.842795, Symbolic median KGE=-0.574743, Symbolic median NSE=-0.459029
- val: LSTM median KGE=0.662898, LSTM median NSE=0.663122, Symbolic median KGE=-0.926307, Symbolic median NSE=-1.099013
- test: LSTM median KGE=0.640016, LSTM median NSE=0.577719, Symbolic median KGE=-0.704476, Symbolic median NSE=-0.848180
