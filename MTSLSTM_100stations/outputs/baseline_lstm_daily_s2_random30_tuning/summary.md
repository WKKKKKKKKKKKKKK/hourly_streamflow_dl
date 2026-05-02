BaselineLSTM daily aggregation tuning on S2 random 30 stations

Training style mirrors the BaselineLSTM repo:
- ordinary LSTM with dynamic and static features concatenated at each time step
- NSE loss
- early stopping on validation loss
- hyperparameter selection by validation median NSE (same as the BaselineLSTM sweep metric)

Daily aggregation used here:
- Rainf: daily mean of hourly values
- Tair: daily mean of hourly values
- PotEvap: daily mean of hourly values
- Streamflow: daily mean of hourly values

Best run: idx4_lr0.0005_bs512_lb180_hs256_do0.4_lossnse_loss
Best val median NSE: 0.704676
Best val median KGE: 0.632653
Best test median NSE: 0.530864
Best test median KGE: 0.562697

All tuned runs:
- idx4_lr0.0005_bs512_lb180_hs256_do0.4_lossnse_loss: val NSE=0.704676, val KGE=0.632653, test NSE=0.530864, test KGE=0.562697
- idx15_lr0.0001_bs256_lb90_hs512_do0.2_lossnse_loss: val NSE=0.671829, val KGE=0.582138, test NSE=0.498782, test KGE=0.540982
- idx10_lr0.001_bs128_lb180_hs256_do0.2_lossnse_loss: val NSE=0.664135, val KGE=0.651911, test NSE=0.641921, test KGE=0.696133
- idx13_lr0.0005_bs128_lb180_hs512_do0.2_lossnse_loss: val NSE=0.663951, val KGE=0.609839, test NSE=0.613594, test KGE=0.661637
- idx9_lr0.001_bs128_lb90_hs128_do0.2_lossnse_loss: val NSE=0.663122, val KGE=0.662898, test NSE=0.577719, test KGE=0.640016
- idx3_lr0.0005_bs512_lb365_hs256_do0.4_lossnse_loss: val NSE=0.655626, val KGE=0.637511, test NSE=0.581436, test KGE=0.672530
- idx2_lr0.0005_bs256_lb180_hs256_do0.4_lossnse_loss: val NSE=0.648469, val KGE=0.659222, test NSE=0.630996, test KGE=0.672703
- idx1_lr0.0005_bs256_lb365_hs256_do0.4_lossnse_loss: val NSE=0.647569, val KGE=0.590467, test NSE=0.666872, test KGE=0.649839
- idx12_lr0.0005_bs128_lb90_hs128_do0.2_lossnse_loss: val NSE=0.641458, val KGE=0.560344, test NSE=0.574387, test KGE=0.628835
- idx8_lr0.0001_bs512_lb180_hs256_do0.4_lossnse_loss: val NSE=0.633245, val KGE=0.544860, test NSE=0.474971, test KGE=0.540158
- idx6_lr0.0001_bs256_lb180_hs256_do0.4_lossnse_loss: val NSE=0.630831, val KGE=0.537218, test NSE=0.535152, test KGE=0.579741
- idx5_lr0.0001_bs256_lb365_hs256_do0.4_lossnse_loss: val NSE=0.622223, val KGE=0.622520, test NSE=0.554760, test KGE=0.541674
- idx16_lr0.0001_bs256_lb180_hs512_do0.6_lossnse_loss: val NSE=0.618483, val KGE=0.590262, test NSE=0.539812, test KGE=0.540373
- idx11_lr0.001_bs128_lb365_hs256_do0.4_lossnse_loss: val NSE=0.618037, val KGE=0.598948, test NSE=0.610316, test KGE=0.610734
- idx7_lr0.0001_bs512_lb365_hs256_do0.4_lossnse_loss: val NSE=0.596396, val KGE=0.547931, test NSE=0.520857, test KGE=0.532670
- idx14_lr0.0005_bs128_lb365_hs512_do0.6_lossnse_loss: val NSE=0.588423, val KGE=0.579614, test NSE=0.564297, test KGE=0.543847
