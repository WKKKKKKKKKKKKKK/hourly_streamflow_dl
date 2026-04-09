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
        # Forward propagate LSTM
        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.dropout(out)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out



class NaiveLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        fc_hidden=[64, 32],
        bidirectional=True,
        activation='relu'
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # 激活函数选择
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")

        # 全连接层
        fc_layers = []
        in_features = hidden_size * self.num_directions
        for h in fc_hidden:
            fc_layers.append(nn.Linear(in_features, h))
            fc_layers.append(self.activation)
            fc_layers.append(nn.Dropout(dropout))
            in_features = h
        fc_layers.append(nn.Linear(in_features, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)          # out: (batch, seq_len, hidden_size * num_directions)
        out = out[:, -1, :]            # 取最后一个时间步
        out = self.fc(out)
        return out.squeeze()


class MTSLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        fc_hidden=[64, 32],
        bidirectional=True,
        activation='relu',
        frequencies=["Y", "D"],  # 时间尺度顺序：低频 → 高频
        shared_lstm=False  # 是否共享 LSTM 参数
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.frequencies = frequencies
        self.shared_lstm = shared_lstm

        # LSTM 层，每个频率一个，或者共享
        self.lstms = nn.ModuleDict()
        for idx, freq in enumerate(frequencies):
            if shared_lstm and idx > 0:
                self.lstms[freq] = self.lstms[frequencies[0]]  # 共享 LSTM
            else:
                self.lstms[freq] = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                    bidirectional=bidirectional
                )

        # 状态投影层（低频 → 高频）
        self.transfer_h = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)
        self.transfer_c = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)

        # 激活函数
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")

        # 每个频率一个 FC head
        self.heads = nn.ModuleDict()
        for freq in frequencies:
            fc_layers = []
            in_features = hidden_size * self.num_directions
            for h in fc_hidden:
                fc_layers.append(nn.Linear(in_features, h))
                fc_layers.append(self.activation)
                fc_layers.append(nn.Dropout(dropout))
                in_features = h
            fc_layers.append(nn.Linear(in_features, 1))
            self.heads[freq] = nn.Sequential(*fc_layers)

    def forward(self, x_dict):
        """
        x_dict: dict
            key = frequency, value = tensor(batch, seq_len, input_size)
        """
        h_transfer, c_transfer = None, None
        outputs = {}

        for idx, freq in enumerate(self.frequencies):
            x = x_dict[freq]  # 当前频率的序列

            # 如果有低频状态传递，作为初始状态
            if h_transfer is not None:
                h0 = h_transfer.unsqueeze(0).repeat(self.num_layers * self.num_directions, 1, 1)
                c0 = c_transfer.unsqueeze(0).repeat(self.num_layers * self.num_directions, 1, 1)
                out, (h, c) = self.lstms[freq](x, (h0, c0))
            else:
                out, (h, c) = self.lstms[freq](x)

            # 更新低频 → 高频传递状态（线性投影）
            h_transfer = self.transfer_h(h[-1])  # 最后一层隐藏状态
            c_transfer = self.transfer_c(c[-1])

            # 取最后时间步输出
            out_last = out[:, -1, :]
            outputs[freq] = self.heads[freq](out_last).squeeze(-1)

        return outputs


class MTSLSTM1(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size=64,
            num_layers=2,
            dropout=0.4,
            fc_hidden=[64, 32],
            bidirectional=True,
            activation='relu',
            frequencies=["Y", "D"],  # 时间尺度顺序：低频 → 高频
            shared_lstm=False  # 是否共享 LSTM 参数
    ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1
            self.frequencies = frequencies
            self.shared_lstm = shared_lstm

            # LSTM 层，每个频率一个，或者共享
            self.lstms = nn.ModuleDict()
            for idx, freq in enumerate(frequencies):
                if shared_lstm and idx > 0:
                    self.lstms[freq] = self.lstms[frequencies[0]]  # 共享 LSTM
                else:
                    self.lstms[freq] = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=dropout if num_layers > 1 else 0.0,
                        bidirectional=bidirectional
                    )

            # 状态投影层（低频 → 高频）
            self.transfer_h = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)
            self.transfer_c = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)

            # 激活函数
            if activation.lower() == 'relu':
                self.activation = nn.ReLU()
            elif activation.lower() == 'gelu':
                self.activation = nn.GELU()
            else:
                raise ValueError("activation must be 'relu' or 'gelu'")
        # 每个频率一个 FC head
            self.heads = nn.ModuleDict()
            for freq in frequencies:
                fc_layers = []
                in_features = hidden_size * self.num_directions
                for h in fc_hidden:
                    fc_layers.append(nn.Linear(in_features, h))
                    fc_layers.append(self.activation)
                    fc_layers.append(nn.Dropout(dropout))
                    in_features = h
                fc_layers.append(nn.Linear(in_features, 1))
                self.heads[freq] = nn.Sequential(*fc_layers)

    def forward(self, x_dict):
        h_transfer, c_transfer = None, None
        outputs = {}

        for idx, freq in enumerate(self.frequencies):
            x = x_dict[freq]  # (batch, seq_len, input_size)

            if h_transfer is not None:
                h0 = h_transfer.unsqueeze(0).repeat(self.num_layers * self.num_directions, 1, 1)
                c0 = c_transfer.unsqueeze(0).repeat(self.num_layers * self.num_directions, 1, 1)
                out, (h, c) = self.lstms[freq](x, (h0, c0))
            else:
                out, (h, c) = self.lstms[freq](x)

            # 只用最后时间步更新状态
            h_transfer = self.transfer_h(h[-1])
            c_transfer = self.transfer_c(c[-1])

            # 只取最后时间步输出
            out_last = out[:, -1, :]
            outputs[freq] = self.heads[freq](out_last).squeeze(-1)

        return outputs
    

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


import torch
import torch.nn as nn


class sMTSLSTM_daily_hourly(nn.Module):
    """
    Multi-scale LSTM:
    Daily branch + Optional Hourly branch
    Compatible with:
    - stations with only daily
    - stations with daily + hourly
    """

    def __init__(
        self,
        dyn_input_size,
        static_input_size,
        hidden_size_daily=64,
        hidden_size_hourly=64,
        num_layers=1,
        dropout=0.0,
        frequency_factor=24
    ):
        super().__init__()

        self.num_layers = num_layers
        self.frequency_factor = frequency_factor

        # =========================
        # Daily LSTM
        # =========================
        self.lstm_daily = nn.LSTM(
            input_size=dyn_input_size + static_input_size,
            hidden_size=hidden_size_daily,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # =========================
        # Hourly LSTM
        # =========================
        self.lstm_hourly = nn.LSTM(
            input_size=dyn_input_size + static_input_size,
            hidden_size=hidden_size_hourly,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # =========================
        # State transfer
        # =========================
        self.transfer_h = nn.Linear(hidden_size_daily, hidden_size_hourly)
        self.transfer_c = nn.Linear(hidden_size_daily, hidden_size_hourly)

        # =========================
        # Output heads
        # =========================
        self.head_daily = nn.Linear(hidden_size_daily, 1)
        self.head_hourly = nn.Linear(hidden_size_hourly, 1)

    def forward(self, x_dict, x_static):

        outputs = {}

        # =====================================================
        # 1️⃣ DAILY BRANCH
        # =====================================================
        x_d = x_dict["D"]
        batch_size, seq_len_D, _ = x_d.shape

        # expand static
        x_s_d = x_static.unsqueeze(1).repeat(1, seq_len_D, 1)
        x_daily = torch.cat([x_d, x_s_d], dim=2)

        out_D, (h_D, c_D) = self.lstm_daily(x_daily)

        # daily prediction (last step)
        daily_last = out_D[:, -1, :]
        pred_D = self.head_daily(daily_last).squeeze(-1)

        outputs["D"] = pred_D

        # =====================================================
        # 2️⃣ If no hourly input → return
        # =====================================================
        if "H" not in x_dict or x_dict["H"] is None:
            outputs["H"] = None
            return outputs

        # =====================================================
        # 3️⃣ HOURLY BRANCH
        # =====================================================
        x_h = x_dict["H"]
        batch_size, seq_len_H, _ = x_h.shape

        x_s_h = x_static.unsqueeze(1).repeat(1, seq_len_H, 1)
        x_hourly = torch.cat([x_h, x_s_h], dim=2)

        # use last daily hidden state for transfer
        h_transfer = h_D[-1]
        c_transfer = c_D[-1]

        h_H0 = self.transfer_h(h_transfer)
        c_H0 = self.transfer_c(c_transfer)

        # expand to LSTM layers
        h_H0 = h_H0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_H0 = c_H0.unsqueeze(0).repeat(self.num_layers, 1, 1)

        out_H, _ = self.lstm_hourly(x_hourly, (h_H0, c_H0))

        hourly_last = out_H[:, -1, :]
        pred_H = self.head_hourly(hourly_last).squeeze(-1)

        outputs["H"] = pred_H

        return outputs



