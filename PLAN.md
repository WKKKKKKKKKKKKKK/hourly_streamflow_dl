# 全球 MTS-LSTM 小时径流实验计划

把 `hourly_streamflow_dl` 的美国实验扩展到全球。**核心实验一句话**：留出 20% 的站假装它们没有小时观测，只用这些站的**日聚合**观测去监督，最后在这 20% 站上输出小时预报，与被藏起来的真实小时观测比 KGE/NSE。

---

## 1. 实验定义

```
10,423 个小时站
   ├── 源域 80% (≈8,338 站)  ── 小时观测可用
   └── 目标域 20% (≈2,085 站) ── 小时观测「藏起来」，只暴露 24h 聚合值
                                  真实小时观测仅用于最终评估

阶段 1（预训练）  在源域 80% 上用小时观测训练 MTS-LSTM
                  损失：NSE_h(H, Q_h^obs) + λ_reg·(D − mean_24h(H_seq))²
                  → 得到源模型 M_src

阶段 2（迁移）    在目标域 20% 上微调，只用日聚合观测
                  损失：NSE_d(D, y_d) + w_agg·NSE_d(mean_24h(H_seq), y_d)
                  其中 y_d = y_h[t−24:t].mean()，小时观测 y_t 不进损失
                  冻结 lstm_hourly，训练 lstm_daily / transfer_h / transfer_c / head_*
                  → 得到 M_transfer

阶段 3（评估）    在目标域 20% 的 test 期输出小时预报
                  与真实小时观测算 per-station KGE/NSE，取中位数
```

**这套损失和冻结策略在 `transfer_daily_to_hourly_partial_ft_s2_random30.py` 里已经实现并验证过**（30 站：test KGE 0.489）。全球版**不改方法，只改规模和站点划分**。

### 必须有的对照

单独一个 M_transfer 的数字说明不了问题，至少要两个对照：

| 代号 | 说明 | 用途 |
|---|---|---|
| **M0** | M_src 直接套到目标域 20%，不做任何微调（zero-shot） | 基准。回答"日聚合监督到底带来了多少增益" |
| **M1** | M_transfer（阶段 2 的产物） | 主结果 |
| **M2**（可选） | M1 + 符号先验（沿用 `..._symbolic_hybrid.py`，30 站上 0.489 → 0.506） | 延续已有工作线 |

核心结论就是 **M1 − M0 的站级配对差异**，用 Wilcoxon signed-rank + BH-FDR 检验（脚本已有：`run_s2_threeway_significance_tests.py`）。

---

## 2. 数据与切分

数据根目录 `DB = /ibex/project/c2266/abbaa0a/data/gscad_database/processed/20250630`

只用小时库（`DB/hourly/`），**日库完全不需要** —— 日观测是从小时观测 24h 平均得来的。

| 项 | 内容 |
|---|---|
| 目标 | `hourly/dataframes/hourly_q.nc`，`q(time=895608, stations=10423)`，1922–2024，单位 m³/s |
| 强迫 | `hourly/dataframes/` ERA5-HRES 7 变量：P / Temp / SWd / LWd / Pres / RelHum / Wind（另有 MSWEP 可做产品对比） |
| 静态 | `hourly/dataframes/static.csv`，55 列 |
| 站点来源 | CAMELSH(US) 5767、BOMAustralia 2059、LamaHCE 859、Japan 696、Germany 494、Czech 437、LamaHIce 111 |

### 站点质控
- `years_q_valid ≥ 10` → **8,854 站可用**（数据来自 `EDA_global_hourly_runoff/tables/basin_features.csv`，已算好）
- 重度调蓄站（`reservoir_impact_GRanD_v1_3` 高）**不剔除**，单独标记成一个评估子集

### 时间切分
统一 **2000-01-01 → 2020-12-31**（小时 Q 密度最高的窗口）：
- train 2000–2012 / val 2013–2015 / test 2016–2020
- 每个 split 内独立建窗口，头 365 天是 burn-in（`t_min = lookback_daily×24 = 8760`），不产样本 → 无跨切分泄漏

### 站点划分：80/20 就是 5 折，把 5 折都跑完

**80/20 是对的，但不要只跑一折。** 80/20 恰好等于 5-fold CV：轮换 5 次，**每个站都恰好当一次目标站**，于是能拿到全部 8,854 站的小时 KGE，而不是只有其中 20%。这带来三件事：
- 全球地图无采样运气问题，每个站都有真实评估值
- 按气候带/大洲/flashiness 分层的分析样本量 ×5
- 堵住"你划分得巧"的质疑

**成本很低**：超参搜索**只在 fold 1 做一次**，其余 4 折复用同一组超参（否则每折都搜既贵、又引入折间选择泄漏）。5 次预训练彼此独立，用 SLURM array 并行跑，墙钟时间 ≈ 1 次预训练。

具体规模（质控后 8,854 站）：每折目标域 ≈ **1,771 站**，源域 ≈ **7,083 站**。

**为什么不是 90/10 或 50/50**

| 比例 | 源域 | 目标域 | 评价 |
|---|---|---|---|
| 90/10（10 折） | 7,969 | 885 | 预训练次数翻倍；每折微调用的日数据少一半，M1 反而可能更差 |
| **80/20（5 折）** | **7,083** | **1,771** | **推荐** |
| 70/30（3.3 折） | 6,198 | 2,656 | 也可以，但折数不是整数不好安排 |
| 50/50（2 折） | 4,427 | 4,427 | 只需 2 次预训练，但"一半的世界没有小时数据"这个设定不自然 |

关键判断：**源域大小在你这个量级已经不是约束**。区域 LSTM 的性能随流域数增长在几百个站就趋于饱和（Kratzert 的 PUB 系列结果），7,083 和 4,427 之间差别很小。所以比例主要由**目标域的用途**决定 —— 而目标域大小影响两件事：(a) 统计功效，30 站在你之前的实验里就已经够了，1,771 站远远够；(b) **阶段 2 微调可用的日数据量**，这是个隐藏变量，报告时要写清每折目标域站数。综合看 5 折最省事。

### 划分方式比比例重要得多

CAMELSH 有 5,767 个美国站，纯随机划分下**目标站在源域里几乎必然有水文近邻**（空间自相关），结果会系统性偏乐观。所以要跑两套划分，它们是难度的两个台阶：

| 划分 | 做法 | 回答的问题 |
|---|---|---|
| **CV-random**（主） | 分层随机 5 折（按 KGZ_major × 来源机构 × `years_q_valid` 分位数分层，保证每折气候/区域构成一致） | "**同一区域内**，日数据能否替代小时数据" |
| **CV-blocked**（次） | 空间分块 5 折：用 `basin_embedding.csv` 的 8 类聚类或地理 k-means 分块，保证目标站在源域中无近邻 | "**整个区域都没有小时数据**时还行不行" |

**两者的差值本身就是一个头条结果** —— 它量化了迁移技巧里有多少来自地理近邻、有多少来自真正基于流域属性的泛化。代码完全不用改，只换划分表。

固定 seed，5 折划分表落盘成 `folds_random.csv` / `folds_blocked.csv`（列：`station_id, fold`），所有实验共用。

---

## 3. 需要改的代码

方法不变，改动集中在**规模**。现有代码在 30 站能跑，在 8,854 站会崩。

### 3.1 样本枚举与数据加载（主要工程量）
现有 `TransferTargetDataset.__init__` / `MultiscaleLSTMDataset.__init__` 用 Python 逐时刻循环枚举样本：
```python
for t in range(t_min, len(x)):
    x_h = x[t-168:t]; x_d_full = x[t-8760:t]
    if np.isnan(x_h).any() or np.isnan(x_d_full).any() or ...: continue
```
8,854 站 × 18.4 万时刻 ≈ **1.6×10⁹ 次迭代**，且每次都要 slice + isnan 扫 8760 个点。必须改：
- **NetCDF → 按站分块的 zarr / memmap 缓存**（`chunks=(1 station, 8760 h)`，float32）。体量：4 变量 × 18.4 万小时 × 8,854 站 × 4 B ≈ **26 GB**，放 `/ibex/scratch`。
- **向量化有效性判断**：用有效掩码的累积和差分一次算出所有窗口是否含 NaN，不用逐时刻 slice。结果存成 `(station_idx, t_idx)` 的 int32 数组落盘，只算一次。
- **stride = 24**（每天一个样本，同 Gauch et al.），否则样本量到 10⁹ 级。
- **每 epoch 随机子采样 + 按站平衡采样**（CAMELSH 占 55%，不平衡会让美国站主导损失）。
- 转换脚本直接抄 `EDA_global_hourly_runoff/scripts/compute_features.py` 的 h5py + 时间戳对齐 + 分块 scatter 模式 —— 已验证能在 40 GB 内处理 10,423 × 40 万。

### 3.2 三个必须修的坑

1. **强迫文件 time 轴不是单调递增的**。`ERA5_HRES_hourly_P_hourly.nc` 首尾时间戳显示 1979-09→2002-10 但共 407,591 个时刻；与 Q 求时间戳交集才得到 403,247 h（1979-01-01→2024-12-31）。**必须按时间戳对齐，不能按位置切片**；各文件 stations 顺序也不同，要按站名重排。（`compute_features.py` 的 `get_indexer(common)` 做法是对的，照抄。）

2. **`handle_extremes(max_streamflow=1000)`**（`Train.py:367`，transfer 脚本 L91 硬编码）把 >1000 m³/s 置 NaN。100 个美国站影响有限，全球集里 area 到 26,000 km²、单站 q95 就有 555 m³/s、q99 到 2,859 m³/s，**这个上限会静默删掉大量真实洪峰 —— 恰好是最关心的极端事件**。改成先转 mm/h（`q_mm_h = q_cms × 3.6 / area_km2`），再按 per-basin 分位数 + 物理界质控。

3. **单位归一化**。q 是 m³/s，全球 area 跨 4 个数量级，必须转 mm/h 才能跨流域训练收敛。（`NSELoss` 的 per-station std 归一化只解决了损失尺度，没解决目标量纲。）

4. **模型选择泄漏（必须修，也影响 30 站已有结果的表述）**。`transfer_daily_to_hourly_partial_ft_s2_random30.py` L418–428 按 **`val_hourly_kge`** 选最优 epoch 并做 early stopping：
   ```python
   val_hourly = evaluate_hourly_per_station(model, loaders["val"], ...)
   val_kge = float(val_summary["median_kge"])          # 目标站的小时 KGE
   if val_kge > best_val_kge: best_state = ...          # 用它选 epoch
   ```
   训练损失只用日聚合观测（正确），但**选 epoch 用了目标站的小时观测** —— 在"这些站没有小时数据"的前提下做不到这件事。这不改变结论方向，但会让报告的 KGE 偏乐观，而且是审稿人一定会查的点。
   **改法**：早停/选 epoch 改用**日聚合的 val 指标**（`mean_24h(H_seq)` vs `y_d` 的 KGE，前提下可得），小时 KGE 只在最后 test 期算一次。
   **顺带做一个稳健性检查**：同时记录"按日指标选 epoch"和"按小时指标选 epoch"两种 test 小时 KGE。若差距很小，说明 30 站的已有结果不受影响，可以在文里直接交代掉。

> 已核对**不用**改的：目标未泄漏进输入（`dyn.sel(dynamic_forcing=cfg.dynamic_vars)`）；scaler 只在训练期拟合；KGE/NSE 在反标准化后的物理空间计算；时间切分严格时序无重叠。100 站实验这几处都是对的。
>
> 日历日对齐问题**在本设计下不存在** —— 日标签 `y_d = y[t−24:t].mean()` 与日分支输入窗口同偏移，自洽。（只有改读 `daily_q.nc` 时才需要处理。）

### 3.3 其它小改动
- `outputs["D"]` 只取最后一天。可改成对整条 `d_seq` 算 seq-to-seq 日损失，每样本的日监督信号从 1 个变 365 个，样本效率高很多。
- 小时库无 PET 变量，用 ERA5 的 Temp/SWd/RelHum/Wind 现算（Priestley-Taylor 或 Hargreaves）。
- 静态属性：从 55 列里选 27–35 列，含 area/slope/elevation/KGZ one-hot/soil/landcover/snowfall_fraction/reservoir_impact/RGI_glacier。**GDP/HDI/population 类慎用**（模型会学到社会经济代理变量而非水文机制）。

---

## 4. 执行步骤

| 步骤 | 内容 | 产出 | 估时 |
|---|---|---|---|
| **S0** | 站点质控 + 分层 5 折划分（random + blocked 两套） | `folds_random.csv` / `folds_blocked.csv` | 1–2 天 |
| **S1** | NetCDF → zarr 缓存 + 向量化样本索引 + 新 Dataset | `data/build_cache.py`、`data/dataset.py` | 1–2 周 |
| **S2** | 200 站冒烟测试，跑通阶段 1→2→3 全链路 | 与 30 站结果量级一致 | 2–3 天 |
| **S3** | 1,000 站中等规模，确认收敛与显存/IO 不炸 | 定下 batch/worker/lookback | 3–5 天 |
| **S4** | fold 1 上做超参搜索定架构，然后 5 折预训练（SLURM array 并行） | 5 个 `M_src` | 1–2 周（含排队） |
| **S5** | 5 折各自阶段 2+3：目标域迁移 + 评估 M0/M1/(M2)，汇总成全部 8,854 站的指标 | per-station 指标 CSV | 3–5 天 |
| **S6** | 显著性检验 + 出图 + 写作 | 图表 + 表格 | 1–2 周 |

超参搜索沿用现成的 SLURM array 框架（`tuning/submit_*.sbatch` + `make_grid.py` + `arrayrun_train.py` + wandb sweep），只换数据路径。搜索空间从 100 站最优 `idx2` 收缩：`hidden_size_{d,h} ∈ {64,128,256}`、`lookback_hourly ∈ {168,336}`、`lookback_daily = 365`、`dropout ∈ {0.2,0.4}`、`lr` 沿用分段 schedule、`w_agg ∈ {0.25,0.5,1.0}`。

> EDA 结果支持 `lookback_hourly = 168`：best_lag 中位数 17 h、P90 ≈ 77 h。336 用于慢响应/雪融流域。

---

## 5. 评估与分析

主指标：目标域 20% 站的 **test 期 per-station 小时 KGE / NSE 中位数**。

直接迁移的现成脚本（改数据路径即可）：
- `evaluate_s2_random30_alt_threeway.py` → M0/M1/M2 三方对比
- `run_s2_threeway_significance_tests.py` → 站级配对 Wilcoxon + BH-FDR
- `plot_three_method_peak_lag_cdfs.py` → 峰值时间误差 CDF
- `evaluate_transfer_on_source_domain.py` + `analyze_source_domain_transfer_degradation.py` → 源域遗忘（阶段 2 微调后 M_src 在源域 80% 上退化多少）

**全球尺度特有的分析**（这是"扩展到全球"相比 30 站的增量价值）：
1. **增益的空间/气候分层**：`M1 − M0` 的 KGE 增益 vs KGZ 气候带、大洲、`area`、`years_q_valid`。回答"日聚合监督在什么样的流域最有用"。
2. **增益 vs 源域近邻密度**：目标站到最近源域站的距离 / 周边源域站密度。这是"随机划分偏易"的直接量化，也预告了整块留出会掉多少。
3. **增益 vs EDA 水文特征**：用 `basin_features.csv` 的 `best_lag` / `flashiness` / `max_lag_corr` 和 `basin_embedding.csv` 的 8 类聚类分层。低 `max_lag_corr` 站（雪融/调蓄主导）预计增益最小。
4. **退化解诊断**：`loss_agg` 只约束 24h 均值，对日内分布无约束 —— 存在"每天输出一条直线"的退化解（日聚合完美、小时无意义）。用同一套 `compute_features.py` 对 M1 的预测算 flashiness / Q95 事件数，与目标站**真实观测**的对应值比。这里有真实小时值，所以能直接量化，不用代理指标。
5. **全球地图**：目标域 20% 站的小时 KGE 空间分布。

---

## 6. 仓库结构

```
global_mtslstm/
  PLAN.md
  configs/     exp_global.yaml（站点划分、时间窗、w_agg、超参）
  data/
    select_stations.py    # S0 质控 + 分层 20% 划分
    build_cache.py        # S1 NetCDF → zarr
    dataset.py            # 向量化样本索引 + 新 Dataset
    units.py              # cms → mm/h、PET
  models/  Modelzoo.py  losses.py     # 从 MTSLSTM_100stations/code 拷来
  train/   pretrain_source.py  transfer_target.py  (tuning/ SLURM)
  eval/    从 hourly_streamflow_dl 迁移的分析脚本
  outputs/
```

---

## 7. 主要风险

| 风险 | 对策 |
|---|---|
| 8,854 站随机读 I/O 成为瓶颈 | 按站分块 zarr + 每 epoch 按站分组采样减少跨站随机读；26 GB 可整体载入大内存节点 RAM |
| 站点不均衡（CAMELSH 55%） | 按站/按区域平衡采样 + per-station 归一化的 NSE 损失 |
| 全规模预训练不收敛 | S2/S3 两级冒烟测试先定超参；从 100 站最优配置出发而非从头搜 |
| 随机 20% 划分偏易，结论被质疑 | 分析 #2 主动量化近邻效应；补一组整块留出（代码不用改） |
| 阶段 2 微调导致源域遗忘 | 用已有的 degradation 脚本量化；必要时在阶段 2 混入少量源域小时 batch |

---

## 8. 待确认

1. **20% 是按站随机，还是按站随机 + 补一组整块留出？** 建议随机版先跑完，整块留出作为第二组（代码不用改，只换划分表）。
2. **符号先验（M2）这次要不要带上？** 不带的话链路更短更快出结果；带上则延续已有工作线。
3. **S4 的规模**：源域 80% ≈ 8,338 站（质控后约 7,000 站）全上，还是先做 2,000 站版本确认结论稳定再全上？
