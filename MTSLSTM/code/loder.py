
import logging
from torch.utils.data import ConcatDataset
from typing import Tuple
from typing import Union, Dict
import numpy as np

import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn

logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
import xarray as xr


def _normalize_station_id(station_id) -> str:
    station_id = str(station_id).strip()
    if station_id.endswith(".0"):
        whole, frac = station_id.rsplit(".", 1)
        if frac == "0":
            station_id = whole
    return station_id.lstrip("0") or "0"


def _resolve_static_index(static_df: pd.DataFrame, station_id) -> str:
    station_id = str(station_id).strip()
    if station_id in static_df.index:
        return station_id

    canonical = _normalize_station_id(station_id)
    if canonical in static_df.index:
        return canonical

    cache = static_df.attrs.get("_station_id_lookup")
    if cache is None:
        cache = {}
        ambiguous = set()
        for idx in static_df.index.astype(str):
            key = _normalize_station_id(idx)
            if key in cache and cache[key] != idx:
                ambiguous.add(key)
            else:
                cache[key] = idx
        for key in ambiguous:
            cache.pop(key, None)
        static_df.attrs["_station_id_lookup"] = cache

    match = cache.get(canonical)
    if match is None:
        raise KeyError(f"Static features missing for station {station_id!r} (canonical={canonical!r})")
    return match


def _get_static_row(static_df: pd.DataFrame, station_id) -> pd.Series:
    key = _resolve_static_index(static_df, station_id)
    row = static_df.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def generate_batches_LSTM(dataset, target, static_df=None, lookback=365*24,
                           batch_size=128, shuffle=True, start_date = "2010-01-01", end_date = "2015-12-31"):
    """
    生成 LSTM 训练 batch
    dataset: xr.Dataset, 每个站点是一个 data variable
    dynamic_vars: list[str], 动态特征名称，例如 ['Rainf', 'Tair', 'PotEvap']
    target_var: str, 径流变量
    static_df: pd.DataFrame, index 为站点名，列为静态特征
    lookback: int, LSTM 回溯步长
    batch_size: int, 每个 batch 大小
    shuffle: bool, 是否打乱顺序
    """

    stations = list(dataset.data_vars)  # 每个站点是一个 data variable
    
    x_dyn_list, x_static_list, y_list = [], [], []

    for stn in stations:
        # 直接取 DataArray
        ds_stn = dataset[stn]  # shape: (time, dynamic_forcing) 或 (time, features)
        target_y = target[stn]
        # 提取动态特征
        x_dyn = ds_stn.sel(time=slice(start_date, end_date))
        x_dyn = x_dyn.transpose('time','dynamic_forcing').values.astype('float32')

        # 提取目标
        #y = target_y.sel(time=slice(start_date, end_date)).astype('float32').values
        y = target_y.sel(time=slice(start_date, end_date)).astype('float32').values 
        

        # 静态特征
        if static_df is None:
            x_static = np.zeros((x_dyn.shape[0], 1), dtype='float32')
        else:
            static_values = _get_static_row(static_df, stn).values.astype("float32")
            x_static = np.repeat(static_values[np.newaxis, :], x_dyn.shape[0], axis=0).astype('float32')

        # 构建 LSTM 回溯样本
        n_samples = x_dyn.shape[0] - lookback + 1
        for i in range(n_samples):
            x_dyn_window = x_dyn[i:i+lookback]
            y_window = y[i+lookback-1]  # 取最后一步作为目标
            x_static_window = x_static[i+lookback-1]

            # 跳过 NaN
            if np.isnan(x_dyn_window).any() or np.isnan(y_window):
                continue

            x_dyn_list.append(x_dyn_window)
            x_static_list.append(x_static_window)
            y_list.append(y_window)

    # 转 numpy
    x_dyn_all = np.stack(x_dyn_list, axis=0)
    x_static_all = np.stack(x_static_list, axis=0)
    y_all = np.array(y_list, dtype='float32').reshape(-1,1)

    # 打乱顺序
    if shuffle:
        idx = np.arange(len(y_all))
        np.random.shuffle(idx)
        x_dyn_all = x_dyn_all[idx]
        x_static_all = x_static_all[idx]
        y_all = y_all[idx]

    # 按 batch 生成
    n_total = len(y_all)
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        yield x_dyn_all[start:end], x_static_all[start:end], y_all[start:end]


class LSTMDataset(Dataset):
    """
    Dataset does not pre-build a giant list of (station, index) tuples.

    It returns all possible windows (including those that may contain NaNs).
    NaN filtering can be handled in the training loop.
    """

    def __init__(self, dataset, target, static_df=None,
                 lookback=365*24,
                 start_date="2010-01-01",
                 end_date="2015-12-31"):

        self.lookback = lookback
        self.static_df = static_df
        self.x_data = {}
        self.y_data = {}

        self.stations = []
        self._cum_ends = []  # cumulative window counts

        total = 0
        stations = [str(s) for s in dataset.data_vars]

        for stn in stations:
            x = dataset[stn].sel(time=slice(start_date, end_date))
            y = target[stn].sel(time=slice(start_date, end_date))

            x = np.asarray(x.transpose("time", "dynamic_forcing").values, dtype="float32")
            y = np.asarray(y.values, dtype="float32")

            n_time = x.shape[0]
            n_windows = n_time - lookback + 1
            if n_windows <= 0:
                continue

            self.x_data[stn] = x
            self.y_data[stn] = y

            self.stations.append(stn)
            total += n_windows
            self._cum_ends.append(total)

    def __len__(self):
        return self._cum_ends[-1] if self._cum_ends else 0

    def __getitem__(self, idx):
        import bisect

        stn_pos = bisect.bisect_right(self._cum_ends, idx)
        prev_end = self._cum_ends[stn_pos - 1] if stn_pos > 0 else 0
        start_idx = idx - prev_end

        stn = self.stations[stn_pos]

        x_window = self.x_data[stn][start_idx:start_idx + self.lookback]
        y_target = self.y_data[stn][start_idx + self.lookback - 1]

        if self.static_df is None:
            x_static = np.zeros((1,), dtype="float32")
        else:
            x_static = np.asarray(_get_static_row(self.static_df, stn).values, dtype="float32")

        return (
            torch.from_numpy(x_window),
            torch.from_numpy(x_static),
            torch.tensor([y_target], dtype=torch.float32),
            stn
        )


def handle_extremes(
        dyn_ds: xr.Dataset,
        min_streamflow: float = 0.0,
        max_streamflow: float = 1000.0
        ):
    # replace -ve values with nans/consider them as missing values
    neg_vals = (dyn_ds.sel(dynamic_forcing='Streamflow') < min_streamflow).sum().to_array().sum().data
    outliers = (dyn_ds.sel(dynamic_forcing='Streamflow') > max_streamflow).sum().to_array().sum().data
    if  neg_vals> 0 or outliers > 0:
        if neg_vals>0: logger.info(f"Streamflow has {neg_vals} -ve vals, replacing them with NaN")
        if outliers>0: logger.info(f"Streamflow has {outliers} outliers replacing them with NaN")

        ds = dyn_ds.copy()
        for v in ds.data_vars:
            e_vals = ds[v].sel(dynamic_forcing='Streamflow')
            ds[v].loc[dict(dynamic_forcing='Streamflow')] = e_vals.where((e_vals >= min_streamflow) & (e_vals <= max_streamflow))
    
        return ds

    return dyn_ds

def calculate_scalers(
        dyn_tr:Union[Dict[str, xr.Dataset], xr.Dataset], 
        static_df:pd.DataFrame, 
        y_tr:Union[Dict[str, xr.Dataset], xr.Dataset]
        )->dict:

    dyn_scalers = {'mean': [], 'std': []}
    if isinstance(dyn_tr, dict):
        dyn_features = dyn_tr[list(dyn_tr.keys())[0]].dynamic_features.data
        for feature in dyn_features:
            feature_data = []
            for stn_da in dyn_tr.values():
                feature_data.append(stn_da.sel(dynamic_forcing=feature).data)
            dyn_scalers['mean'].append(np.nanmean(np.hstack(feature_data)))
            dyn_scalers['std'].append(np.nanstd(np.hstack(feature_data)))
    else:
        dyn_features = dyn_tr.dynamic_forcing.data
        for feature in dyn_features:
            feature = dyn_tr.sel(dynamic_forcing=feature).to_dataarray().data.flatten(order='F')
            dyn_scalers['mean'].append(np.nanmean(feature))
            dyn_scalers['std'].append(np.nanstd(feature))
    
    dyn_scalers = xr.Dataset(
        data_vars={
            key: ('dynamic_forcing', value) for key, value in dyn_scalers.items()
            },
            coords={
                'dynamic_forcing': dyn_features
                })

    x_st_mean = static_df.mean(skipna=True)
    x_st_std = static_df.std(skipna=True)

    if isinstance(y_tr, dict):
        y = np.hstack([y_tr[key].data for key in y_tr.keys()])
        y_mean = np.nanmean(y)
        y_std = np.nanstd(y)
    else:
        y = y_tr.to_pandas().drop(columns='dynamic_forcing').values.ravel(order='F')
        y_mean = np.nanmean(y, axis=0)
        y_std = np.nanstd(y, axis=0)

    scalers = {
        'x_dyn_mean': dyn_scalers['mean'],
        'x_dyn_std': dyn_scalers['std'],
        'x_st_mean': x_st_mean,
        'x_st_std': x_st_std,
        'y_mean': y_mean,
        'y_std': y_std,
    }

    return scalers


def standardize_data(
        dyn_data: Union[Dict[str, xr.Dataset], xr.Dataset],
        static_df: pd.DataFrame,
        y_data: Union[Dict[str, xr.Dataset], xr.Dataset],
        scalers: dict
    ):
    """
    使用 scalers 对数据进行标准化

    Parameters
    ----------
    dyn_data : xr.Dataset 或 dict[str, xr.Dataset]
        动态特征数据
    static_df : pd.DataFrame
        静态特征数据，index=站点名
    y_data : xr.Dataset 或 dict[str, xr.Dataset]
        目标数据
    scalers : dict
        calculate_scalers 输出的字典
        包含 x_dyn_mean/std, x_st_mean/std, y_mean/std

    Returns
    -------
    x_dyn_std : xr.Dataset 或 dict[str, xr.Dataset]
        标准化后的动态特征
    x_static_std : pd.DataFrame
        标准化后的静态特征
    y_std : xr.Dataset 或 dict[str, xr.Dataset]
        标准化后的目标
    """

    # -------- 动态特征 --------
    if isinstance(dyn_data, dict):
        x_dyn_std = {}
        for stn, ds in dyn_data.items():
            ds_std = ds.copy()
            for i, feature in enumerate(ds.dynamic_forcing.data):
                mean = scalers['x_dyn_mean'][i]
                std = scalers['x_dyn_std'][i]
                ds_std.loc[dict(dynamic_forcing=feature)] = (ds.loc[dict(dynamic_forcing=feature)] - mean) / std
            x_dyn_std[stn] = ds_std.astype('float32')
    else:
        x_dyn_std = dyn_data.copy()
        for i, feature in enumerate(dyn_data.dynamic_forcing.data):
            mean = scalers['x_dyn_mean'][i]
            std = scalers['x_dyn_std'][i]
            x_dyn_std.loc[dict(dynamic_forcing=feature)] = (dyn_data.loc[dict(dynamic_forcing=feature)] - mean) / std

    x_dyn_std = x_dyn_std.astype('float32')

    # -------- 静态特征 --------
    x_static_std = ((static_df - scalers['x_st_mean']) / scalers['x_st_std']).astype('float32')

    # -------- 目标 --------
    if isinstance(y_data, dict):
        y_std = {}
        for stn, ds in y_data.items():
            da = ds.copy()  # Dataset → DataArray，保留 coords
            y_std[stn] = ((da - scalers['y_mean']) / scalers['y_std']).astype('float32')
    else:
        y_std = ((y_data - scalers['y_mean']) / scalers['y_std']).astype('float32')

    return x_dyn_std, x_static_std, y_std




class MultiscaleLSTMDataset(Dataset):

    def __init__(self, dataset, target, static_df=None,
                 lookback_hourly=72,
                 lookback_daily=365,
                 frequency_factor=24,
                 start_date=None,
                 end_date=None):

        self.lookback_hourly = lookback_hourly
        self.lookback_daily = lookback_daily

        self.frequency_factor = frequency_factor
        self.static_df = static_df

        self.x_data = {}
        self.y_data = {}
        self.samples = []

        stations = [str(s) for s in dataset.data_vars]

        for stn in stations:

            x = dataset[stn].sel(time=slice(start_date, end_date))
            y = target[stn].sel(time=slice(start_date, end_date))

            x = np.asarray(x.transpose("time", "dynamic_forcing").values, dtype="float32")
            y = np.asarray(y.values, dtype="float32")

            self.x_data[stn] = x
            self.y_data[stn] = y

            T = x.shape[0]

            for t in range(lookback_hourly, T):

                # ---------- 高频窗口 ----------
                x_h = x[t - lookback_hourly : t]
                y_t = y[t]

                # 必须保证能构造低频
                if t < lookback_daily * self.frequency_factor:
                    continue

                # ---------- 低频窗口 ----------
                # 取过去 lookback_daily 天
                start_d = t - lookback_daily * self.frequency_factor
                x_d_full = x[start_d : t]

                # reshape 成 (days, 24, features)
                x_d = x_d_full.reshape(lookback_daily, self.frequency_factor, -1).mean(axis=1)

                # ---------- NaN 检查 ----------
                if (
                    np.isnan(x_h).any()
                    or np.isnan(x_d).any()
                    or np.isnan(y_t)
                ):
                    continue

                self.samples.append((stn, t))

        print("Total valid samples:", len(self.samples))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        stn, t = self.samples[idx]

        x = self.x_data[stn]
        y = self.y_data[stn]

        # 高频
        x_h = x[t - self.lookback_hourly : t]

        # 低频
        start_d = t - self.lookback_daily * self.frequency_factor
        x_d_full = x[start_d : t]
        x_d = x_d_full.reshape(self.lookback_daily, self.frequency_factor, -1).mean(axis=1)

        # 静态
        if self.static_df is None:
            x_s = torch.zeros((1,), dtype=torch.float32)
        else:
            x_s = torch.tensor(
                _get_static_row(self.static_df, stn).values.astype("float32")
            )

        y_t = y[t]

        return (
            {
                "H": torch.from_numpy(x_h),
                "D": torch.from_numpy(x_d),
                "S": x_s
            },
            torch.tensor([y_t], dtype=torch.float32),
            stn
        )



