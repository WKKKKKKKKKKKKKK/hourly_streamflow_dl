# hourly_streamflow_dl
Deep learning for hourly streamflow forecasting

The multi-time-scale LSTM model is constructed following the methodology described in the paper (Gauch et al., 2021): https://hess.copernicus.org/articles/25/2045/2021/

### Dynamic data: 
- CAMELSH timeseries (15 stations)
- Forcings: Precipitation, Temperature, Potential evaporation
- Target: Streamflow

### Static attributes
- Mean precipitation  
- Mean potential evapotranspiration  
- Aridity index  
- Fraction of snow precipitation  
- High precipitation frequency  
- High precipitation duration  
- Low precipitation frequency  
- Low precipitation duration  
- Average silt content  
- Average sand content  
- Average available water capacity  
- Average permeability  
- Average bulk density  
- Forest percentage of land use  
- Cropland percentage of land use  
- Urban percentage of land use  
- Average Baseflow Index  
- Annual water balance (WB5100_ANN_MM, mm)

### Data Split

- Training: 1990-10-01 — 2003-09-30  
- Validation: 2003-10-01 — 2008-09-30  
- Test: 2008-10-01 — 2015-09-30  

### Model Performance

| Metric (hourly)               | MTS-LSTM | LSTM |
|----------------------|:--------:|:----:|
| Fitting Median KGE   | 0.871    |0.890 |
| Fitting Median NSE   | 0.941    |0.897 |
| Validate Median KGE  | 0.523    |0.226 |
| Validate Median NSE  | 0.298    |0.229 |
| Test Median KGE      | 0.511    |0.160 |
| Test Median NSE      | 0.295    |0.195 |
