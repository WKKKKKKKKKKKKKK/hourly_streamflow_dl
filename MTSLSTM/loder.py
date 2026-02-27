
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


logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
import xarray as xr




class LSTMDataset(Dataset):
    """
    Dataset don't filter out samples with NaN. 
    Instead, we will handle NaN in the training loop by skipping batches that contain any NaN values. 
    This way, we can keep all potential samples and only exclude those that are actually used in each batch.
    """
    def __init__(self, dataset, target, static_df=None,
                 lookback=365*24,
                 start_date="2010-01-01",
                 end_date="2015-12-31"):

        self.lookback = lookback
        self.static_df = static_df
        self.x_data = {}
        self.y_data = {}
        self.samples = []

        stations = [str(s) for s in dataset.data_vars]

        for stn in stations:
            x = dataset[stn].sel(time=slice(start_date, end_date))
            y = target[stn].sel(time=slice(start_date, end_date))

            x = x.transpose("time", "dynamic_forcing").values.astype("float32")
            y = y.values.astype("float32")

            self.x_data[stn] = x
            self.y_data[stn] = y

            n_time = x.shape[0]
            for i in range(n_time - lookback + 1):
                self.samples.append((stn, i))  # 不管 NaN

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stn, start_idx = self.samples[idx]

        x_window = self.x_data[stn][start_idx:start_idx+self.lookback]
        y_target = self.y_data[stn][start_idx+self.lookback-1]

        if self.static_df is None:
            x_static = np.zeros((1,), dtype="float32")
        else:
            x_static = self.static_df.loc[stn].values.astype("float32")

        return (
            torch.from_numpy(x_window),
            torch.from_numpy(x_static),
            torch.tensor([y_target], dtype=torch.float32),
            stn
        )



def handle_extremes(
        dyn_ds:xr.Dataset,
       
        ):

    # replace -ve values with nans/consider them as missing values
    neg_vals = (dyn_ds.sel(dynamic_forcing='Streamflow') < 0).sum().to_array().sum().data
    outliers = (dyn_ds.sel(dynamic_forcing='Streamflow') > 5000).sum().to_array().sum().data
    if  neg_vals> 0 or outliers > 0:
        if neg_vals>0: logger.info(f"Streamflow has {neg_vals} -ve vals, replacing them with NaN")
        if outliers>0: logger.info(f"Streamflow has {outliers} outliers replacing them with NaN")

        ds = dyn_ds.copy()
        for v in ds.data_vars:
            e_vals = ds[v].sel(dynamic_forcing='Streamflow')

            ds[v].loc[dict(dynamic_forcing='Streamflow')] = e_vals.where(e_vals >= 0)

            ds[v].loc[dict(dynamic_forcing='Streamflow')] = e_vals.where(e_vals <= 1000)
    
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
    

    # -------- dynamic --------
    if isinstance(dyn_data, dict):
        x_dyn_std = {}
        for stn, ds in dyn_data.items():
            ds_std = ds.copy()
            for i, feature in enumerate(ds.dynamic_forcing.data):
                mean = scalers['x_dyn_mean'][i]
                std = scalers['x_dyn_std'][i]
                ds_std.loc[dict(dynamic_forcing=feature)] = (ds.loc[dict(dynamic_forcing=feature)] - mean) / std
            x_dyn_std[stn] = ds_std
    else:
        x_dyn_std = dyn_data.copy()
        for i, feature in enumerate(dyn_data.dynamic_forcing.data):
            mean = scalers['x_dyn_mean'][i]
            std = scalers['x_dyn_std'][i]
            x_dyn_std.loc[dict(dynamic_forcing=feature)] = (dyn_data.loc[dict(dynamic_forcing=feature)] - mean) / std

    # -------- static --------
    x_static_std = (static_df - scalers['x_st_mean']) / scalers['x_st_std']

    # -------- target --------
    if isinstance(y_data, dict):
        y_std = {}
        for stn, ds in y_data.items():
            da = ds.copy()  # Dataset → DataArray
            y_std[stn] = (da - scalers['y_mean']) / scalers['y_std']
    else:
        y_std = (y_data - scalers['y_mean']) / scalers['y_std']

    return x_dyn_std, x_static_std, y_std




class MultiscaleLSTMDataset(Dataset):

    def __init__(self, dataset, target, static_df=None,
                 lookback_hourly=72,
                 lookback_daily=365,
                 start_date=None,
                 end_date=None):

        self.lookback_hourly = lookback_hourly
        self.lookback_daily = lookback_daily
        self.static_df = static_df

        self.x_data = {}
        self.y_data = {}
        self.samples = []

        stations = [str(s) for s in dataset.data_vars]

        for stn in stations:

            x = dataset[stn].sel(time=slice(start_date, end_date))
            y = target[stn].sel(time=slice(start_date, end_date))

            x = x.transpose("time", "dynamic_forcing").values.astype("float32")
            y = y.values.astype("float32")

            self.x_data[stn] = x
            self.y_data[stn] = y

            T = x.shape[0]

            for t in range(lookback_hourly, T):

                # ---------- high frequency window ----------
                x_h = x[t - lookback_hourly : t]
                y_t = y[t]

                # garantee that we have enough history for the daily window
                if t < lookback_daily * 24:
                    continue

                # ---------- low frequency window ----------
                # 
                start_d = t - lookback_daily * 24
                x_d_full = x[start_d : t]

                # reshape (days, 24, features)
                x_d = x_d_full.reshape(lookback_daily, 24, -1).mean(axis=1)

                # ---------- NaN check ----------
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

        # high frequency
        x_h = x[t - self.lookback_hourly : t]

        # low frequency
        start_d = t - self.lookback_daily * 24
        x_d_full = x[start_d : t]
        x_d = x_d_full.reshape(self.lookback_daily, 24, -1).mean(axis=1)

        # static
        if self.static_df is None:
            x_s = torch.zeros((1,), dtype=torch.float32)
        else:
            x_s = torch.tensor(
                self.static_df.loc[stn].values.astype("float32")
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
