Source-domain transfer degradation analysis

Definitions:
- retained_close_kge: transfer KGE is within 0.1 of the original source model
- retained_usable: KGE >= 0.5 and NSE >= 0.5
- catastrophic_kge_lt_minus1: KGE < -1
- big_forgetting_delta_lt_minus05: transfer KGE - source KGE < -0.5

Test split medians:
- transfer: median KGE=-0.0587, median NSE=-0.3530, KGE<0 rate=0.420, KGE<-1 rate=0.280
- symbolic_transfer_sw0.05: median KGE=0.1793, median NSE=0.2157, KGE<0 rate=0.350, KGE<-1 rate=0.200
