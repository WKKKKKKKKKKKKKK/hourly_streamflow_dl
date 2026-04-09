import torch
import torch.nn as nn


class NSELoss(nn.Module):
    """Per-basin normalized MSE (NSE-style) loss (same as Baseline)."""

    needs_station_ids = True

    def __init__(self, station_std: dict, eps: float = 1e-6):
        super().__init__()
        self.station_std = station_std or {}
        self.eps = float(eps)
        self._default_std = 1.0

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, stations) -> torch.Tensor:
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        if isinstance(stations, torch.Tensor):
            stations = stations.tolist()

        station_to_indices = {}
        for i, stn in enumerate(stations):
            station_to_indices.setdefault(str(stn), []).append(i)

        losses = []
        for stn, idxs in station_to_indices.items():
            idx = torch.as_tensor(idxs, device=y_pred.device)
            err = y_pred.index_select(0, idx) - y_true.index_select(0, idx)
            mse = (err ** 2).mean()
            sigma = float(self.station_std.get(stn, self._default_std))
            denom = (sigma + self.eps) ** 2
            losses.append(mse / denom)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=y_pred.device)


class MTSNSERegularizedLoss(nn.Module):
    """NSE-style hourly loss + daily-hourly consistency regularization.

    Implements a practical variant of the loss shown in your figure.

    - Hourly term: per-basin normalized MSE for the supervised hourly target.
    - Regularization: (yhat_D - mean_last_k(yhat_H_seq))^2, averaged over batch.

    Notes:
    - We do not supervise the daily prediction directly (no daily targets in the
      current pipeline). The regularizer still ties daily and hourly predictions.
    """

    needs_station_ids = True

    def __init__(self, station_std: dict, frequency_factor: int = 24, reg_lambda: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.nse_hourly = NSELoss(station_std=station_std, eps=eps)
        self.frequency_factor = int(frequency_factor)
        self.reg_lambda = float(reg_lambda)

    def forward(self, outputs: dict, y_true_h: torch.Tensor, stations) -> torch.Tensor:
        # outputs expected from sMTSLSTM: {"H": (B,), "D": (B,), "H_seq": (B, T)}
        loss_h = self.nse_hourly(outputs['H'], y_true_h, stations)

        if self.reg_lambda <= 0:
            return 0.5 * loss_h

        h_seq = outputs.get('H_seq', None)
        d_pred = outputs.get('D', None)

        if h_seq is None or d_pred is None:
            return 0.5 * loss_h

        if h_seq.dim() != 2:
            return 0.5 * loss_h

        k = self.frequency_factor
        if h_seq.size(1) < k:
            return 0.5 * loss_h

        h_mean = h_seq[:, -k:].mean(dim=1)
        reg = ((d_pred.view(-1) - h_mean) ** 2).mean()

        return 0.5 * loss_h + self.reg_lambda * reg
