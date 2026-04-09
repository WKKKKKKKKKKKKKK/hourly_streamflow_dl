# hourly_streamflow_dl
Deep learning for hourly streamflow forecasting

The multi-time-scale LSTM model is constructed following the methodology described in the paper (Gauch et al., 2021): https://hess.copernicus.org/articles/25/2045/2021/

### Dynamic data: 
- CAMELSH timeseries
  - BaselineLSTM & MTSLSTM comparison (15 stations)
  - MTSLSTM (100 stations)
- Forcings: Precipitation, Temperature, Potential evaporation
- Target: Streamflow

---

### Static attributes
### 15-Station Experiment
| Category | Variable | Description |
|----------|--------|------------|
| Climate | Mean precipitation | Long-term average precipitation |
| Climate | Mean potential evapotranspiration | Long-term average PET |
| Climate | Aridity index | Ratio of PET to precipitation |
| Climate | Fraction of snow precipitation | Fraction of precipitation falling as snow |
| Precipitation | High precipitation frequency | Frequency of high-intensity precipitation events |
| Precipitation | High precipitation duration | Duration of high-intensity precipitation events |
| Precipitation | Low precipitation frequency | Frequency of low-intensity precipitation events |
| Precipitation | Low precipitation duration | Duration of low-intensity precipitation events |
| Soil | Average silt content | Mean silt fraction |
| Soil | Average sand content | Mean sand fraction |
| Soil | Available water capacity | Soil water holding capacity |
| Soil | Permeability | Soil permeability |
| Soil | Bulk density | Soil bulk density |
| Land use | Forest percentage | Fraction of forest land |
| Land use | Cropland percentage | Fraction of cropland |
| Land use | Urban percentage | Fraction of urban land |
| Hydrology | Baseflow Index (BFI) | Ratio of baseflow to total streamflow |
| Hydrology | Annual water balance (WB5100_ANN_MM) | Long-term annual water balance |

---

### 100-Station Experiment
| Category | Variable | Description |
|----------|--------|------------|
| ID | STAID | Station identifier |
| Climate | p_mean | Mean precipitation |
| Climate | pet_mean | Mean potential evapotranspiration |
| Climate | aridity_index | Aridity index |
| Climate | p_seasonality | Precipitation seasonality |
| Climate | frac_snow | Fraction of snowfall |
| Precipitation | high_prec_freq | High precipitation frequency |
| Precipitation | high_prec_dur | High precipitation duration |
| Precipitation | low_prec_freq | Low precipitation frequency |
| Precipitation | low_prec_dur | Low precipitation duration |
| Topography | DRAIN_SQKM | Drainage area (km²) |
| Topography | ELEV_MEAN_M_BASIN | Mean elevation |
| Topography | ELEV_STD_M_BASIN | Elevation variability |
| Topography | SLOPE_PCT | Mean slope (%) |
| Hydrology | RRMEAN | Mean runoff ratio |
| Hydrology | RRMEDIAN | Median runoff ratio |
| Basin | BAS_COMPACTNESS | Basin compactness |
| River network | STREAMS_KM_SQ_KM | Stream density |
| River network | STRAHLER_MAX | Maximum Strahler order |
| Soil / Wetness | TOPWET | Topographic wetness index |
| Soil | CLAYAVE | Clay fraction |
| Soil | SANDAVE | Sand fraction |
| Soil | AWCAVE | Available water capacity |
| Soil | PERMAVE | Permeability |
| Soil | BDAVE | Bulk density |
| Land use | for_pc_use | Forest land fraction |
| Land use | crp_pc_use | Cropland fraction |
| Land use | urb_pc_use | Urban land fraction |

### Data Split

- Training: 1990-10-01 — 2003-09-30  
- Validation: 2003-10-01 — 2008-09-30  
- Test: 2008-10-01 — 2015-09-30  


### 100-Station Data Setup for Reproducibility

The 100-station NetCDF files are not stored directly in this GitHub repository because of file-size limits. Share the archive `data_100stations_share.tar.gz` separately (for example via Google Drive), then place the extracted files under a top-level `data/` directory in the repository root.

After extraction, the repository should contain:

```text
hourly_streamflow_dl/
├── data/
│   ├── selected_stn_data_100stations_west10_east90_proposal_boxes_part01of02.nc
│   ├── selected_stn_data_100stations_west10_east90_proposal_boxes_part02of02.nc
│   ├── selected_stn_data_100stations_west10_east90_proposal_boxes.csv
│   └── static_h_topo_priority27.csv
├── MTSLSTM_100stations/
└── ...
```

Notes:
- The two `part*.nc` files together form the full 100-station dynamic dataset. Both files are required.
- If extracting the archive creates another nested folder, move the files so that they end up directly inside `data/` as shown above.
- The updated `MTSLSTM_100stations` configuration and tuning script now look for these 100-station files in `repo_root/data/` by default, so no local absolute path edits are required if the files are placed there.
- These shared files are for the 100-station MTSLSTM experiments. The 15-station BaselineLSTM/MTSLSTM experiments use a different dataset and are not reproduced from this archive.


### Model Performance
### 15-Station Experiment
| Metric (hourly)               | MTS-LSTM | LSTM |
|----------------------|:--------:|:----:|
| Fitting Median KGE   | 0.926    |0.853 |
| Fitting Median NSE   | 0.912    |0.864 |
| Validate Median KGE  | 0.744    |0.704 |
| Validate Median NSE  | 0.621    |0.548 |
| Test Median KGE      | 0.628    |0.600 |
| Test Median NSE      | 0.565    |0.538 |

### 100-Station Experiment
| Metric (hourly)               | MTS-LSTM |
|----------------------|:--------:|
| Fitting Median KGE   | 0.921    |
| Fitting Median NSE   | 0.917    |
| Validate Median KGE  | 0.783    |
| Validate Median NSE  | 0.718    |
| Test Median KGE      | 0.721    |
| Test Median NSE      | 0.694    |
