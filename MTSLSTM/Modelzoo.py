import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    LSTM receives both static and dynamic inputs at each time step
    """
    def __init__(
            self,
            input_size,
            hidden_size:int = 256,
            num_layers:int = 1,
            dropout:float = 0.4,
            output_size:int = 1
    ):
        
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):

        dynamic = x[0]
        static = x[1].unsqueeze(1).repeat(1, dynamic.shape[1], 1)

        x = torch.concat((dynamic, static), dim=2)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True, dtype=x.dtype).to(x.device)

        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True, dtype=x.dtype).to(x.device)
  
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]

        out = self.dropout(out)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

class sMTSLSTM(nn.Module):
    """
    Strict implementation of MTS-LSTM (Gauch et al. 2020 style)
    Two-branch version: Daily -> Hourly
    """

    def __init__(
        self,
        dyn_input_size,
        static_input_size,
        hidden_size_daily=64,
        hidden_size_hourly=64,
        num_layers=1,
        dropout=0.0,
        frequency_factor=24  # D -> H
    ):
        super().__init__()

        self.hidden_size_daily = hidden_size_daily
        self.hidden_size_hourly = hidden_size_hourly
        self.num_layers = num_layers
        self.frequency_factor = frequency_factor

        # ===============================
        # Daily LSTM
        # ===============================
        self.lstm_daily = nn.LSTM(
            input_size=dyn_input_size + static_input_size,
            hidden_size=hidden_size_daily,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ===============================
        # Hourly LSTM
        # ===============================
        self.lstm_hourly = nn.LSTM(
            input_size=dyn_input_size + static_input_size,
            hidden_size=hidden_size_hourly,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ===============================
        # State transfer layers
        # ===============================
        self.transfer_h = nn.Linear(hidden_size_daily, hidden_size_hourly)
        self.transfer_c = nn.Linear(hidden_size_daily, hidden_size_hourly)

        # ===============================
        # Prediction heads
        # ===============================
        self.head_daily = nn.Linear(hidden_size_daily, 1)
        self.head_hourly = nn.Linear(hidden_size_hourly, 1)

    # ==========================================================
    # Forward
    # ==========================================================
    def forward(self, x_dict, x_static):
        """
        x_dict:
            {
                "D": (batch, seq_len_D, dyn_features),
                "H": (batch, seq_len_H, dyn_features)
            }

        x_static:
            (batch, static_features)
        """

        outputs = {}

        # ======================================================
        # 1️⃣ DAILY BRANCH (full sequence)
        # ======================================================

        x_d = x_dict["D"]
        batch_size, seq_len_D, _ = x_d.shape

        # expand static to time dimension
        x_s_d = x_static.unsqueeze(1).repeat(1, seq_len_D, 1)

        x_daily = torch.cat([x_d, x_s_d], dim=2)

        out_D, (h_D, c_D) = self.lstm_daily(x_daily)

        # Daily prediction (use last timestep)
        daily_last = out_D[:, -1, :]
        outputs["D"] = self.head_daily(daily_last).squeeze(-1)

        # ======================================================
        # 2️⃣ Compute transfer index
        # ======================================================

        seq_len_H = x_dict["H"].shape[1]

        offset_days = seq_len_H // self.frequency_factor

        if offset_days <= 0:
            raise ValueError("Hourly sequence too short for frequency factor.")

        transfer_index = seq_len_D - offset_days

        if transfer_index <= 0:
            raise ValueError("Daily sequence too short for alignment.")

        # ======================================================
        # 3️⃣ Get hidden & cell at transfer point
        # ======================================================

        # Run daily LSTM only up to transfer point
        _, (h_mid, c_mid) = self.lstm_daily(
            x_daily[:, :transfer_index, :]
        )

        # take top layer state
        h_transfer = h_mid[-1]  # (batch, hidden_daily)
        c_transfer = c_mid[-1]

        # ======================================================
        # 4️⃣ Linear state transfer
        # ======================================================

        h_H0 = self.transfer_h(h_transfer)
        c_H0 = self.transfer_c(c_transfer)

        # expand to (num_layers, batch, hidden_hourly)
        h_H0 = h_H0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_H0 = c_H0.unsqueeze(0).repeat(self.num_layers, 1, 1)

        # ======================================================
        # 5️⃣ HOURLY BRANCH
        # ======================================================

        x_h = x_dict["H"]
        batch_size, seq_len_H, _ = x_h.shape

        x_s_h = x_static.unsqueeze(1).repeat(1, seq_len_H, 1)
        x_hourly = torch.cat([x_h, x_s_h], dim=2)

        out_H, _ = self.lstm_hourly(x_hourly, (h_H0, c_H0))

        hourly_last = out_H[:, -1, :]
        outputs["H"] = self.head_hourly(hourly_last).squeeze(-1)

        return outputs






