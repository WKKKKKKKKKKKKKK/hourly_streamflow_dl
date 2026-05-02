Daily transfer learning on S2 random 30 stations

Source pretrained model: /home/kongw0a/hourly_streamflow_dl/MTSLSTM_100stations/training_runs/20260407_mtslstm_100stations_tuning_topo18_v100/idx2_bs128_do0.4_hs64_H168_D365/best_model.pth
Best transfer trial: lr=0.005, weight_decay=0.0001, best_epoch=13
Best validation median KGE: 0.184845
Best validation median NSE: -0.037005

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse
train,30,22,8,0.267977,0.189181
val,30,28,2,0.184845,-0.037005
test,30,28,2,0.206447,0.107114
