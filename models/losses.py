"""Losses for Phase I.

All of them are basin-averaged: the squared error of every sample is divided by
that station's streamflow variance, so a flashy 10,000 km2 basin and a small
alpine one contribute comparably. The per-station std travels with each batch
(``stn_std`` in the metadata) and is measured in the same standardized space the
model trains in.

Step 1 (pretrain, hourly targets available):
    loss = 0.5 * NSE_h(H, y_h) + reg_lambda * (D - mean_last_k(H_seq))^2

Step 2 (transfer, only daily aggregates available):
    loss = NSE_d(D, y_d) + w_agg * NSE_d(mean_24h(H_seq), y_d)
    y_d = mean(y_h[t-23 .. t]) -- hourly observations never enter the loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Standard stabiliser for basin-averaged NSE (Kratzert et al. 2019).
NSE_EPS = 0.1


def basin_nse(y_pred: torch.Tensor, y_true: torch.Tensor, stn_std: torch.Tensor, eps: float = NSE_EPS) -> torch.Tensor:
    """Mean of ``(pred - true)^2 / (sigma_basin + eps)^2`` over the finite rows."""
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    stn_std = stn_std.reshape(-1).to(y_pred.device, dtype=y_pred.dtype)

    mask = torch.isfinite(y_true)
    if not bool(mask.all()):
        y_pred, y_true, stn_std = y_pred[mask], y_true[mask], stn_std[mask]
    if y_true.numel() == 0:
        return y_pred.sum() * 0.0

    weights = 1.0 / (stn_std + eps) ** 2
    return torch.mean(weights * (y_pred - y_true) ** 2)


class MTSBasinNSELoss(nn.Module):
    """Step 1: hourly basin NSE plus a daily/hourly consistency regulariser."""

    def __init__(self, frequency_factor: int = 1, reg_lambda: float = 1.0, eps: float = NSE_EPS):
        super().__init__()
        self.frequency_factor = int(frequency_factor)
        self.reg_lambda = float(reg_lambda)
        self.eps = float(eps)

    def forward(self, outputs: dict, y_hourly: torch.Tensor, stn_std: torch.Tensor) -> dict[str, torch.Tensor]:
        loss_h = basin_nse(outputs["H"], y_hourly, stn_std, self.eps)
        total = 0.5 * loss_h
        parts = {"loss_hourly": loss_h.detach()}

        h_seq, d_pred = outputs.get("H_seq"), outputs.get("D")
        if self.reg_lambda > 0 and h_seq is not None and d_pred is not None and h_seq.dim() == 2:
            k = max(1, self.frequency_factor)
            if h_seq.size(1) >= k:
                reg = ((d_pred.reshape(-1) - h_seq[:, -k:].mean(dim=1)) ** 2).mean()
                total = total + self.reg_lambda * reg
                parts["loss_reg"] = reg.detach()

        parts["loss"] = total
        return parts


class DailyAggregateTransferLoss(nn.Module):
    """Step 2: supervise only the 24-hour aggregate.

    ``daily_window`` counts positions on the hourly branch. The prepared
    sequences are 1-hour spaced over their last 228 positions, so those
    positions really are the last 24 hours.

    ``daily_mask`` marks which of those 24 hours the station actually observed.
    The prediction is averaged over exactly the same slots as the target, so a
    partial day is compared like for like.
    """

    def __init__(self, daily_window: int = 24, agg_loss_weight: float = 0.5, eps: float = NSE_EPS):
        super().__init__()
        self.daily_window = int(daily_window)
        self.agg_loss_weight = float(agg_loss_weight)
        self.eps = float(eps)

    def forward(
        self,
        outputs: dict,
        y_daily: torch.Tensor,
        stn_std: torch.Tensor,
        daily_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        loss_daily = basin_nse(outputs["D"], y_daily, stn_std, self.eps)
        parts = {"loss_daily_branch": loss_daily.detach()}
        total = loss_daily

        h_seq = outputs.get("H_seq")
        if self.agg_loss_weight > 0 and h_seq is not None and h_seq.size(1) >= self.daily_window:
            pred_agg = daily_aggregate_prediction(outputs, self.daily_window, daily_mask)
            loss_agg = basin_nse(pred_agg, y_daily, stn_std, self.eps)
            total = total + self.agg_loss_weight * loss_agg
            parts["loss_hourly_agg"] = loss_agg.detach()

        parts["loss"] = total
        return parts


def daily_aggregate_prediction(
    outputs: dict,
    daily_window: int = 24,
    daily_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """The model's daily value: mean of its hourly predictions over the observed hours.

    Without a mask this is the plain mean of the last ``daily_window`` steps.
    With one it averages only the slots the station observed, matching how
    ``y_daily`` was built.
    """
    h_seq = outputs["H_seq"]
    window = min(int(daily_window), h_seq.size(1))
    tail = h_seq[:, -window:]
    if daily_mask is None:
        return tail.mean(dim=1)

    mask = daily_mask[:, -window:].to(tail.dtype)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return (tail * mask).sum(dim=1) / counts
