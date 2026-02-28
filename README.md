# hourly_streamflow_dl
Deep learning for hourly streamflow forecasting

### Dynamic data: 
- CAMELSH timeseries (15 stations)

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

| Metric                  | MTS-LSTM | LSTM |
|--------------------------|:---------:|:----:|
| Training time range      | 1990-10-01—2003-09-30 |      |
| Validation time range    | 2003-10-01—2008-09-30 |      |
| Test time range          | 2008-10-01—2015-09-30 |      |
| Fitting median KGE       | 0.871 |      |
| Fitting median NSE       | 0.941 |      |
| Validate Median KGE      | 0.523 |      |
| Validate Median NSE      | 0.298 |      |
| Test Median KGE          | 0.511 |      |
| Test Median NSE          | 0.295 |      |
