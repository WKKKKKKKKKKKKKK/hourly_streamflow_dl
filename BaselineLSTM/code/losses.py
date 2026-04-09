import torch
import torch.nn as nn


class NSELoss(nn.Module):
    """Per-basin normalized MSE (NSE-style) loss.

    For a batch with samples from multiple basins, we compute an MSE per basin,
    normalize it by (sigma_b + eps)^2 where sigma_b is the *standard deviation*
    of the observed discharge for that basin (computed on the training period),
    and then average over basins.

    This matches the denominator of the NSE definition (variance) while keeping
    the implementation numerically stable.
    """

    needs_station_ids = True

    def __init__(self, station_std: dict, eps: float = 1e-6):
        super().__init__()
        self.station_std = station_std or {}
        self.eps = float(eps)

        # Fallback (if station not found). 1.0 corresponds to no normalization.
        self._default_std = 1.0

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, stations) -> torch.Tensor:
        # Shapes: (B, 1) or (B,)
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # stations is typically a list[str]
        if isinstance(stations, torch.Tensor):
            stations = stations.tolist()

        # Group indices per station (small batches -> Python loop is fine)
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
