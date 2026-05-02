BaselineLSTM daily aggregation tuning on S2 random 30 stations

This note selects the best run by validation median KGE instead of validation median NSE.

Best run by validation median KGE:
- tag: `idx9_lr0.001_bs128_lb90_hs128_do0.2_lossnse_loss`
- validation median KGE: `0.662898`
- validation median NSE: `0.663122`
- test median KGE: `0.640016`
- test median NSE: `0.577719`

Best configuration by validation median KGE:
- learning rate: `0.001`
- dropout: `0.2`
- hidden size: `128`
- batch size: `128`
- lookback days: `90`
- epochs run: `30`

Top 5 runs ranked by validation median KGE:
1. `idx9_lr0.001_bs128_lb90_hs128_do0.2_lossnse_loss`: val KGE=`0.662898`, val NSE=`0.663122`, test KGE=`0.640016`, test NSE=`0.577719`
2. `idx2_lr0.0005_bs256_lb180_hs256_do0.4_lossnse_loss`: val KGE=`0.659222`, val NSE=`0.648469`, test KGE=`0.672703`, test NSE=`0.630996`
3. `idx10_lr0.001_bs128_lb180_hs256_do0.2_lossnse_loss`: val KGE=`0.651911`, val NSE=`0.664135`, test KGE=`0.696133`, test NSE=`0.641921`
4. `idx3_lr0.0005_bs512_lb365_hs256_do0.4_lossnse_loss`: val KGE=`0.637511`, val NSE=`0.655626`, test KGE=`0.672530`, test NSE=`0.581436`
5. `idx4_lr0.0005_bs512_lb180_hs256_do0.4_lossnse_loss`: val KGE=`0.632653`, val NSE=`0.704676`, test KGE=`0.562697`, test NSE=`0.530864`
