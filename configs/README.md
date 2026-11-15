# Configurations

One YAML per run, exactly as launched (each run also stores its resolved config in
`outputs/<root>/fold*/pretrain/run_meta.json`). The header comments of some files describe how
a config differs from its predecessor in the development repository; those predecessors are not
part of this repository and were not used for the manuscript. Every config here has
`data.window_alignment: figure`.

| Config | Output root | Role in the manuscript |
|---|---|---|
| `aligned/phase1_runB_v2.yaml` | `al_v2_runB` | random design, seed 42: M0 and M1 (Figs. 5-8); Africa M1 (`m1b` arm) |
| `aligned/phase1_runB_blocked_v2.yaml` | `al_v2_blocked` | blocked design, seed 42 |
| `aligned/phase1_runB_climate.yaml` | `al_v2_climate` | climate-zone design, seed 42 |
| `aligned/phase1_runB_longitudinal.yaml` | `al_v2_longitudinal` | longitudinal design, seed 42 |
| `aligned/phase1_runB_continental.yaml` | `al_v2_continental` | continental design, seed 42 |
| `variance/al_<design>_s43.yaml`, `_s44.yaml` | `var_al_<design>_s43`, `_s44` | seeds 43 and 44 (pretraining and fine-tuning repeated); the continental pair receives its seed at launch |
| `aligned/phase1_runB_replay_v2.yaml`, `phase1_runB_blocked_replay_v2.yaml`, `phase1_runB_<design>_replay025.yaml` | `al_v2_<design>_replay025` (random: `al_v2_replay025`) | M1 with source replay, ratio 0.25, seed 42 (Fig. C1) |
| `variance/al_<design>_replay025_s43.yaml`, `_s44.yaml` | `var_al_<design>_replay025_s4x` | replay, seeds 43 and 44 |
| `aligned/phase1_runB_daily_lr2e4.yaml`, `phase1_runB_daily_lr1e4.yaml`, `phase1_runB_blocked_lr2e4.yaml` | `al_v2_lrsweep_daily_lr2e4`, `..._lr1e4`, `al_v2lr2_blocked` | Step-2 learning rate (Table B2) |
| `aligned/phase1_runB_path_both_equal.yaml` (w_D 1, w_H 1), `phase1_runB_path_aggheavy.yaml` (1, 2), `phase1_runB_path_aggonly.yaml` (0, 1), `phase1_runB_daily0.yaml` (0, 0.5), `phase1_runB_path_stateonly_dailysel.yaml` (1, 0) | `al_v2_path_*`, `al_v2_path_daily0` | objective terms (Table B3); the adopted arm is `phase1_runB_v2` (1, 0.5) |
| `aligned/search/*.yaml` (26) | `al_search/<name>` | architecture search, fold 1 of the random design (Table B1) |
| `phase1_dailylstm_aligned_<design>.yaml` | `al_v2_dailylstm_<design>` | daily LSTM from scratch on the target gauges (Fig. D1) |
| `phase1_africa_daily_lstm_aligned.yaml` | `al_v2_africa_daily_lstm_fold<k>` | African daily LSTM (Figs. D1-D3) |
| `aligned/phase1_africa_scratch_agg.yaml` | `al_v2_africa_scratch_agg_fold<k>` | African M1 architecture trained from scratch; enters the common basin set of Appendix D |

`data.root` points at a prepared-dataset directory that no longer exists; see the README, "Data".
