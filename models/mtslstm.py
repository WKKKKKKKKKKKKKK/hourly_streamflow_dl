"""sMTSLSTM -- two-branch MTS-LSTM (Gauch et al. 2021 style).

Carried over from MTSLSTM_100stations/code/Modelzoo.py with three changes:

* the daily branch runs ONCE and the transfer state is taken from the same
  pass (the original called ``lstm_daily`` a second time on the prefix, which
  doubled the cost of the longest branch for an identical result);
* ``build_model`` reads the config, so every entry point instantiates the same
  architecture;
* ``head_dropout`` (default on) applies the configured dropout to the LSTM
  outputs before each head. In the original, ``dropout`` was only handed to
  ``nn.LSTM``, which ignores it when ``num_layers == 1`` -- so the 100-station
  runs configured with ``dropout=0.4`` in fact trained with none. Set
  ``model.head_dropout: false`` to reproduce that behaviour exactly.

Interface is unchanged: ``forward(x_dict, x_static)`` with
``x_dict = {"D": (B, Td, F), "H": (B, Th, F)}`` returning
``{"D_seq", "D", "H_seq", "H"}``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class sMTSLSTM(nn.Module):
    def __init__(
        self,
        dyn_input_size: int,
        static_input_size: int,
        hidden_size_daily: int = 64,
        hidden_size_hourly: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        frequency_factor: int = 1,
        head_dropout: bool = True,
    ):
        super().__init__()
        self.hidden_size_daily = int(hidden_size_daily)
        self.hidden_size_hourly = int(hidden_size_hourly)
        self.num_layers = int(num_layers)
        self.frequency_factor = int(frequency_factor)

        lstm_kwargs = dict(
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )
        self.lstm_daily = nn.LSTM(dyn_input_size + static_input_size, self.hidden_size_daily, **lstm_kwargs)
        self.lstm_hourly = nn.LSTM(dyn_input_size + static_input_size, self.hidden_size_hourly, **lstm_kwargs)

        self.transfer_h = nn.Linear(self.hidden_size_daily, self.hidden_size_hourly)
        self.transfer_c = nn.Linear(self.hidden_size_daily, self.hidden_size_hourly)

        self.dropout = nn.Dropout(dropout) if head_dropout else nn.Identity()
        self.head_daily = nn.Linear(self.hidden_size_daily, 1)
        self.head_hourly = nn.Linear(self.hidden_size_hourly, 1)

    def forward(self, x_dict: dict[str, torch.Tensor], x_static: torch.Tensor) -> dict[str, torch.Tensor]:
        x_d, x_h = x_dict["D"], x_dict["H"]
        seq_len_d, seq_len_h = x_d.shape[1], x_h.shape[1]

        offset = seq_len_h // max(1, self.frequency_factor)
        transfer_index = seq_len_d - offset
        if offset <= 0:
            raise ValueError("hourly sequence too short for the frequency factor")
        if transfer_index <= 0:
            raise ValueError(
                f"daily sequence ({seq_len_d}) must be longer than the hourly window ({offset})"
            )

        # --- daily branch ---------------------------------------------------
        # Run it as prefix + remainder so the transfer state at
        # ``transfer_index`` and the full-length daily output both come out of a
        # single pass (the original ran the whole branch twice).
        x_daily = torch.cat([x_d, x_static.unsqueeze(1).expand(-1, seq_len_d, -1)], dim=2)
        out_head, (h_mid, c_mid) = self.lstm_daily(x_daily[:, :transfer_index, :])
        out_tail, _ = self.lstm_daily(x_daily[:, transfer_index:, :], (h_mid, c_mid))
        out_d = torch.cat([out_head, out_tail], dim=1)
        d_seq = self.head_daily(self.dropout(out_d)).squeeze(-1)

        h_transfer, c_transfer = h_mid[-1], c_mid[-1]

        h0 = self.transfer_h(h_transfer).unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()
        c0 = self.transfer_c(c_transfer).unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()

        # --- hourly branch --------------------------------------------------
        x_hourly = torch.cat([x_h, x_static.unsqueeze(1).expand(-1, seq_len_h, -1)], dim=2)
        out_h, _ = self.lstm_hourly(x_hourly, (h0, c0))
        h_seq = self.head_hourly(self.dropout(out_h)).squeeze(-1)

        return {"D_seq": d_seq, "D": d_seq[:, -1], "H_seq": h_seq, "H": h_seq[:, -1]}


def build_model(cfg, dyn_input_size: int, static_input_size: int) -> sMTSLSTM:
    model_cfg = cfg.model
    return sMTSLSTM(
        dyn_input_size=dyn_input_size,
        static_input_size=static_input_size,
        hidden_size_daily=int(model_cfg.hidden_size_daily),
        hidden_size_hourly=int(model_cfg.hidden_size_hourly),
        num_layers=int(model_cfg.num_layers),
        dropout=float(model_cfg.dropout),
        frequency_factor=int(model_cfg.frequency_factor),
        head_dropout=bool(model_cfg.get("head_dropout", True)),
    )


def set_trainable(model: sMTSLSTM, freeze_modules: list[str]) -> tuple[int, int]:
    """Freeze the named submodules, train everything else.

    Phase I Step 2 freezes ``lstm_hourly`` so the hourly dynamics learned from
    the source domain survive a fine-tune that only ever sees daily targets.
    Returns ``(n_trainable, n_frozen)`` parameter counts.
    """
    for param in model.parameters():
        param.requires_grad = True
    for name in freeze_modules or []:
        module = getattr(model, name, None)
        if module is None:
            raise AttributeError(f"model has no submodule {name!r}")
        for param in module.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen
