Hybrid smooth-gated fusion of hydrology-informed symbolic log-residual corrections

Correction form:
log(1+q_corr) = log(1+q_lstm) + alpha * g_global(x) + beta * sigmoid((log(1+q_lstm) - (log_q75 + tau))/sharpness) * g_event(x)

Best alpha: 0.8
Best beta: 0.2
Best tau: 0.2
Best sharpness: 0.05

Global equation: (SLOPE_PCT + (rain_pet_logratio_90 - p_mean)) * ((0.5582391 - for_pc_use) * 0.06458572)
Event equation: -0.070106834 / cos(square(PERMAVE))

- val: baseline KGE=0.662899, baseline NSE=0.663122, hybrid KGE=0.740146, hybrid NSE=0.733259
- test: baseline KGE=0.640016, baseline NSE=0.577719, hybrid KGE=0.697128, hybrid NSE=0.631019
