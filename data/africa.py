"""African basins as model inputs (Plan.docx Phase I Steps 4-5).

The 294 African basins are not in the prepared hourly_q_dl batches at all, so
their inputs have to be assembled from scratch in exactly the form the trained
model expects:

  * dynamic: the fixed 1000-position subsample of the previous 8760 hours
    (``data/lookback_offsets.json``), standardized with the training
    ``scalers.json``;
  * static: the same columns in the same order, standardized with the same
    scalers, with Koeppen-Geiger strings converted through the code map
    recovered in ``data/kgz_codes.json``.

Targets are DAILY: for calendar day D the sample is placed at 23:00 of D, so the
last 24 hourly positions cover 00:00-23:00 of D and the mean of the model's last
24 hourly predictions is its prediction for that day. That is the same
aggregation Step 2 supervises.

``scripts/verify_africa_inputs.py`` checks this construction against the prepared
batches for stations that DO exist in them, so the packaging is validated without
any African data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from data.dataset import OneHotSpec, load_lookback_offsets, load_scalers

GSCAD = Path("/ibex/project/c2266/abbaa0a/data/gscad_database")
DAILY_STATIC = GSCAD / "processed/20250630/daily/dataframes/static.csv"
HOURLY_STATIC = GSCAD / "processed/20250630/hourly/dataframes/static.csv"
DAILY_DYNAMIC = Path(
    "/ibex/project/c2266/abbaa0a/data/input_data/hydrodeepai/"
    "MSWEP_V280_Past_penman_1095_10_19800101_20241231_16166_dynamic.nc"
)
DAILY_EPOCH = "1980-01-01"

# The three training feature names that the daily static table renamed.
STATIC_RENAME = {
    "GLiM_V01_Export_do_not_share.mat": "GLiM_V01_Export",
    "elevation_1KMmn_GMTEDmn_with_Antarctica_from_World_e-Atlas.mat": "elevation_1KMmn_GMTEDmn_World_e_Atlas",
    "Daly_PRISM_effective_terrain_height_dem_elevation.mat": "elevation_Daly_PRISM_effective_terrain_height_dem",
}
KGZ_COLUMNS = ("KGZ_major", "KGZ_detailed")
DYN_ORDER = ("pet", "pcp", "temp")
DAILY_WINDOW = 24


def load_kgz_codes() -> dict:
    with open(Path(__file__).resolve().parent / "kgz_codes.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_static_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, comment="#", low_memory=False)
    return frame.rename(columns={frame.columns[0]: "station_id"}).set_index("station_id")


def build_static_matrix(
    station_ids: list[str],
    feature_names: list[str],
    scalers: dict,
    static_path: Path = DAILY_STATIC,
    logger=None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Standardized static matrix ``(n_stations, len(feature_names))``, plus a report.

    Column names, order, and standardization all follow the training dataset. The
    report records, per station, which Koeppen-Geiger substitutions were applied
    and whether any feature had to be imputed.
    """
    table = read_static_table(static_path)
    codes = load_kgz_codes()
    substitutions = codes["africa_substitutions"]["KGZ_detailed"]

    missing_stations = [s for s in station_ids if s not in table.index]
    if missing_stations:
        raise KeyError(f"{len(missing_stations)} stations absent from {static_path}: {missing_stations[:5]}")
    rows = table.loc[station_ids]

    out = np.full((len(station_ids), len(feature_names)), np.nan, dtype=np.float64)
    substituted: dict[str, list[str]] = {}
    imputed: list[str] = []

    for j, name in enumerate(feature_names):
        # The hourly table carries all 42 training names verbatim; the daily table
        # renamed three of them. Prefer the original, fall back to the rename.
        column = name if name in rows.columns else STATIC_RENAME.get(name)
        if column is None or column not in rows.columns:
            raise KeyError(f"static feature {name!r} not in {static_path} (tried {STATIC_RENAME.get(name)!r})")
        raw = rows[column]

        if name in KGZ_COLUMNS:
            mapping = codes[name]
            values, subs = [], []
            for station, zone in zip(station_ids, raw.astype(str)):
                if zone not in mapping and name == "KGZ_detailed" and zone in substitutions:
                    subs.append(f"{station}:{zone}->{substitutions[zone]}")
                    zone = substitutions[zone]
                if zone not in mapping:
                    values.append(np.nan)
                    continue
                values.append(mapping[zone])
            if subs:
                substituted[name] = subs
            raw = pd.Series(values, index=rows.index, dtype="float64")
        else:
            raw = pd.to_numeric(raw, errors="coerce")

        column_values = raw.to_numpy(dtype=np.float64)
        if np.isnan(column_values).any():
            # Impute with the TRAINING mean, i.e. the standardized value 0, so a
            # missing attribute is neutral rather than an arbitrary number.
            n_bad = int(np.isnan(column_values).sum())
            imputed.append(f"{name}:{n_bad}")
            column_values = np.where(np.isnan(column_values), scalers["x_st_mean"][name], column_values)

        out[:, j] = (column_values - scalers["x_st_mean"][name]) / scalers["x_st_std"][name]

    if logger:
        if substituted:
            for name, subs in substituted.items():
                logger.warning("%s: %d Koeppen substitutions applied (e.g. %s)", name, len(subs), subs[:3])
        if imputed:
            logger.warning("features imputed with the training mean (feature:count): %s", imputed)
        else:
            logger.info("no static feature needed imputation")

    report = pd.DataFrame({"station_id": station_ids})
    for name in KGZ_COLUMNS:
        column = name if name in rows.columns else STATIC_RENAME.get(name, name)
        report[name] = rows[column].astype(str).to_numpy()
    report["kgz_substituted"] = [
        any(f"{s}:" in entry for entry in substituted.get("KGZ_detailed", [])) for s in station_ids
    ]
    return out.astype(np.float32), report


def apply_onehot(static_std: np.ndarray, static_keep: np.ndarray, specs: list[OneHotSpec]) -> np.ndarray:
    """Same continuous-slice + indicator-append layout the batch dataset produces."""
    tensor = torch.as_tensor(static_std)
    if not specs:
        return tensor.index_select(1, torch.as_tensor(static_keep, dtype=torch.long)).numpy()
    indicators = [spec.encode(tensor) for spec in specs]
    continuous = tensor.index_select(1, torch.as_tensor(static_keep, dtype=torch.long))
    return torch.cat([continuous] + indicators, dim=1).numpy()


def load_observed_daily(station_ids: list[str], logger=None) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Observed daily ``q_mm`` in mm/d, shape ``(n_stations, n_days)``."""
    import netCDF4 as nc

    handle = nc.Dataset(DAILY_DYNAMIC)
    all_stations = [str(x) for x in handle.variables["station"][:]]
    features = [str(x) for x in handle.variables["dynamic_features"][:]]
    q_index = features.index("q_mm")
    n_days = handle.dimensions["time"].size
    dates = pd.date_range(DAILY_EPOCH, periods=n_days, freq="D")

    lookup = {name: i for i, name in enumerate(all_stations)}
    missing = [s for s in station_ids if s not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} stations absent from the daily database: {missing[:5]}")

    out = np.full((len(station_ids), n_days), np.nan, dtype=np.float32)
    for k, station in enumerate(station_ids):
        out[k] = np.asarray(handle.variables["dyn"][lookup[station], :, q_index], dtype=np.float32)
    handle.close()
    if logger:
        finite = np.isfinite(out)
        logger.info("observed daily q_mm: %d station-days over %d stations (%.1f%% of the grid)",
                    int(finite.sum()), len(station_ids), 100 * finite.mean())
    return out, dates


def load_hourly_forcing(path: str | os.PathLike, station_ids: list[str], scalers: dict, logger=None):
    """Standardized hourly forcing ``(n_stations, n_hours, 3)`` in pet/pcp/temp order."""
    ds = xr.open_dataset(path)
    have = [str(s) for s in ds["station"].values]
    lookup = {name: i for i, name in enumerate(have)}
    missing = [s for s in station_ids if s not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} stations absent from {path}: {missing[:5]}")
    order = [lookup[s] for s in station_ids]
    times = pd.to_datetime(ds["time"].values)

    stacked = np.empty((len(station_ids), len(times), len(DYN_ORDER)), dtype=np.float32)
    for j, name in enumerate(DYN_ORDER):
        if name not in ds:
            raise KeyError(f"forcing file has no variable {name!r}; present: {list(ds.data_vars)}")
        values = np.asarray(ds[name].values, dtype=np.float32)[order]
        stacked[:, :, j] = (values - scalers["x_dyn_mean"][name]) / scalers["x_dyn_std"][name]
    ds.close()

    if logger:
        finite = np.isfinite(stacked)
        logger.info("hourly forcing: %d stations x %d hours (%s .. %s), %.2f%% finite",
                    stacked.shape[0], stacked.shape[1], times[0], times[-1], 100 * finite.mean())
    return stacked, times


class AfricaDailyDataset(Dataset):
    """One item is a chunk of daily samples, matching the batch-file convention.

    A sample is (basin, calendar day). Its dynamic input is the 1000-position
    subsample ending at 23:00 of that day, so the model's last 24 hourly outputs
    cover exactly that day.
    """

    def __init__(
        self,
        forcing: np.ndarray,
        forcing_times: pd.DatetimeIndex,
        static: np.ndarray,
        observed: np.ndarray,
        observed_dates: pd.DatetimeIndex,
        station_ids: list[str],
        lookback_hourly: int = 168,
        chunk_size: int = 512,
        logger=None,
    ):
        self.forcing = forcing
        self.static = static
        self.station_ids = list(station_ids)
        self.lookback_hourly = int(lookback_hourly)
        self.chunk_size = int(chunk_size)

        offsets = load_lookback_offsets()
        self.hours_ago = np.asarray(offsets["hours_ago"], dtype=np.int64)
        self.seq_len = int(offsets["seq_len"])
        max_lag = int(self.hours_ago.max())

        # Target hour = 23:00 of each day, and it must exist in the forcing axis.
        hour_pos = {ts: i for i, ts in enumerate(forcing_times)}
        target_hours = observed_dates + pd.Timedelta(hours=23)

        pairs = []
        for k in range(len(station_ids)):
            valid_days = np.flatnonzero(np.isfinite(observed[k]))
            for d in valid_days:
                it = hour_pos.get(target_hours[d])
                if it is None or it < max_lag:
                    continue
                pairs.append((k, d, it))
        if not pairs:
            raise ValueError(
                "no African sample has both an observation and 8760 h of preceding forcing -- "
                "the forcing period probably does not overlap the observations"
            )
        self.pairs = np.asarray(pairs, dtype=np.int64)

        # Drop samples whose window touches a gap in the forcing.
        window = self.pairs[:, 2][:, None] - self.hours_ago[None, :]
        ok = np.isfinite(self.forcing[self.pairs[:, 0][:, None], window, :]).all(axis=(1, 2))
        dropped = int((~ok).sum())
        self.pairs = self.pairs[ok]
        self.observed = observed
        self.observed_dates = observed_dates

        self.chunks = [
            self.pairs[i : i + self.chunk_size] for i in range(0, len(self.pairs), self.chunk_size)
        ]
        if logger:
            logger.info(
                "Africa daily samples: %d over %d basins (%d dropped for an incomplete forcing window) "
                "-> %d chunks of <=%d",
                len(self.pairs), len(set(self.pairs[:, 0].tolist())), dropped, len(self.chunks), self.chunk_size,
            )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        chunk = self.chunks[idx]
        basin, day, target = chunk[:, 0], chunk[:, 1], chunk[:, 2]
        window = target[:, None] - self.hours_ago[None, :]
        x_dyn = torch.from_numpy(self.forcing[basin[:, None], window, :])
        x_static = torch.from_numpy(self.static[basin])
        k_h = self.lookback_hourly if 0 < self.lookback_hourly <= x_dyn.shape[1] else x_dyn.shape[1]
        return {
            "x": {
                "D": x_dyn.contiguous(),
                "H": x_dyn[:, -k_h:, :].contiguous(),
                "S": x_static.contiguous(),
            },
            "stations": [self.station_ids[b] for b in basin],
            "dates": self.observed_dates[day],
            "y_daily_obs": torch.from_numpy(self.observed[basin, day].astype(np.float32)),
            "hours": torch.from_numpy(np.asarray(target, dtype=np.int64)),
            "stn_std": torch.from_numpy(self.station_y_std[basin]),
        }
        if self.with_daily:
            # The transfer loss wants the daily target standardised the same way the
            # model's outputs are, and a mask over the 24 hours it aggregates. Every
            # sample here sits at 23:00 with a verified-finite window, so the day is
            # complete and the mask is all-ones -- kept only so the loss signature
            # matches the global path.
            # The observations are mm/d; the model and its y scaler are mm/h. Divide by
            # the window before standardising, or the target is 24x too large and the
            # model learns to over-predict by that factor while the loss converges
            # perfectly well in its own wrong units. (This is the second time a mm/h vs
            # mm/d slip has reached a run on this path.)
            daily_mm_per_h = self.observed[basin, day].astype(np.float32) / DAILY_WINDOW
            y_daily = (daily_mm_per_h - self.y_mean) / self.y_std
            out["y_daily"] = torch.from_numpy(y_daily)
            out["daily_mask"] = torch.ones((n, 24), dtype=torch.bool)
            out["y"] = torch.from_numpy(y_daily)   # unused by the daily loss, kept for shape
        else:
            out["y_daily"] = None
            out["daily_mask"] = None
        return out

class AfricaWindowDataset(Dataset):
    """African daily samples shaped like the hourly-cache path (run B), not the prepared one.

    ``AfricaDailyDataset`` builds its daily branch from the prepared batches' 1000-step
    power-law subsample. Feeding that to a run B checkpoint is meaningless -- run B was
    trained with ``frequency_factor: 24`` and a daily branch of 365 GENUINE daily means,
    so the two inputs are different quantities of different lengths. Scoring run B
    checkpoints through the prepared-layout dataset produced Africa numbers that looked
    like a model failure and were really an input-structure mismatch.

    This mirrors ``data.hourly_windows.HourlyWindowDataset`` exactly:

        target hour t = 23:00 of the observed day, so the model's last 24 hourly
                        outputs cover that calendar day
        H = forcing[t - lookback_hourly : t]          raw hourly steps, excluding t
        D = daily[day_of(t) - lookback_daily : day_of(t)]   365 whole days before it

    Daily means are taken over ``h // 24`` blocks (00:00-23:00), the same blocking
    ``scripts.build_hourly_cache`` used, so the daily branch means the same thing here
    as it did in training.
    """

    def __init__(
        self,
        forcing: np.ndarray,
        forcing_times: pd.DatetimeIndex,
        static: np.ndarray,
        observed: np.ndarray,
        observed_dates: pd.DatetimeIndex,
        station_ids: list[str],
        lookback_hourly: int = 72,
        lookback_daily: int = 365,
        chunk_size: int = 512,
        logger=None,
        split: str | None = None,
        train_frac: float = 0.7,
        scalers: dict | None = None,
        with_daily: bool = False,
    ):
        """``split`` / ``with_daily`` turn this into a TRAINING set for the transfer step.

        Africa is Phase I's premise occurring naturally -- daily discharge, no hourly --
        so the strongest experiment fine-tunes on African daily observations themselves
        rather than applying a model fine-tuned on temperate target stations. That needs
        three things the evaluation path does not: a temporal split per basin, the daily
        target in standardised units, and the per-basin sigma the basin-NSE loss divides
        by. ``scalers`` is required whenever ``with_daily`` is set, because an unstandardised
        target would silently be off by the y scaler.
        """
        self.forcing = forcing
        self.static = static
        self.station_ids = list(station_ids)
        self.k_h = int(lookback_hourly)
        self.k_d = int(lookback_daily)
        self.chunk_size = int(chunk_size)

        # Block into whole 00:00-23:00 days, the same blocking build_hourly_cache used.
        # The series need not start at 00:00 -- the rescaling step drops the leading
        # hour that belongs to the day before the record -- so align to the first
        # midnight rather than assuming index 0 is one, which would shift every daily
        # mean by an hour.
        midnights = np.flatnonzero(forcing_times.hour.to_numpy() == 0)
        if midnights.size == 0:
            raise ValueError("forcing has no 00:00 stamp; cannot form whole days")
        self.day0 = int(midnights[0])
        n_days = (forcing.shape[1] - self.day0) // 24
        if n_days <= lookback_daily:
            raise ValueError(
                f"only {n_days} whole days of forcing after the first midnight, need "
                f"more than {lookback_daily}"
            )
        self.daily = forcing[:, self.day0 : self.day0 + n_days * 24, :3].reshape(
            forcing.shape[0], n_days, 24, 3
        ).mean(axis=2)

        hour_pos = {ts: i for i, ts in enumerate(forcing_times)}
        target_hours = observed_dates + pd.Timedelta(hours=23)

        pairs = []
        for k in range(len(station_ids)):
            for d in np.flatnonzero(np.isfinite(observed[k])):
                it = hour_pos.get(target_hours[d])
                if it is None:
                    continue
                day_end = (it - self.day0) // 24
                if it < self.k_h or day_end < self.k_d or day_end > self.daily.shape[1]:
                    continue
                pairs.append((k, d, it))
        if not pairs:
            raise ValueError(
                "no African sample has both an observation and enough preceding forcing "
                f"({self.k_d} days + {self.k_h} h) -- check the forcing/observation overlap"
            )
        pairs = np.asarray(pairs, dtype=np.int64)

        # Both branches must be finite: a NaN anywhere makes the whole prediction NaN.
        basin, _, target = pairs[:, 0], pairs[:, 1], pairs[:, 2]
        day_end = (target - self.day0) // 24
        h_ok = np.array([
            np.isfinite(forcing[b, t - self.k_h : t, :3]).all() for b, t in zip(basin, target)
        ])
        d_ok = np.array([
            np.isfinite(self.daily[b, e - self.k_d : e]).all() for b, e in zip(basin, day_end)
        ])
        keep = h_ok & d_ok
        dropped = int((~keep).sum())
        self.observed = observed
        self.observed_dates = observed_dates
        self.with_daily = bool(with_daily)
        self.y_mean = float(scalers["y_mean"]) if scalers else 0.0
        self.y_std = float(scalers["y_std"]) if scalers else 1.0
        if self.with_daily and scalers is None:
            raise ValueError("with_daily needs scalers -- an unstandardised daily target "
                             "would be off by the y scaler and the loss would be meaningless")
        pairs = pairs[keep]

        # Split each basin's OWN record in time, the same 70/30 convention the global
        # experiment uses. Splitting globally by date would give water-rich basins a
        # different train/test boundary than sparse ones.
        if split is not None:
            if split not in {"training", "validation"}:
                raise ValueError(f"split must be training|validation, got {split!r}")
            keep_split = np.zeros(len(pairs), dtype=bool)
            for b in np.unique(pairs[:, 0]):
                rows = np.flatnonzero(pairs[:, 0] == b)
                order = rows[np.argsort(pairs[rows, 2])]
                cut = int(len(order) * train_frac)
                chosen = order[:cut] if split == "training" else order[cut:]
                keep_split[chosen] = True
            pairs = pairs[keep_split]
            if len(pairs) == 0:
                raise ValueError(f"no African samples left in the {split} split")

        self.pairs = pairs
        # Per-basin sigma of the daily observations, for the basin-normalised loss.
        self.station_y_std = np.ones(len(station_ids), dtype=np.float32)
        for b in range(len(station_ids)):
            vals = observed[b][np.isfinite(observed[b])] / DAILY_WINDOW   # mm/d -> mm/h
            if vals.size > 1:
                self.station_y_std[b] = max(float(np.std(vals)) / max(self.y_std, 1e-9), 1e-3)

        self.chunks = [
            self.pairs[i : i + self.chunk_size]
            for i in range(0, len(self.pairs), self.chunk_size)
        ]
        if logger:
            logger.info(
                "Africa windows (run B layout): %d samples over %d basins | D=%d daily means, "
                "H=%d hourly | %d dropped for a non-finite window -> %d chunks of <=%d",
                len(self.pairs), len(set(self.pairs[:, 0].tolist())), self.k_d, self.k_h,
                dropped, len(self.chunks), self.chunk_size,
            )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        chunk = self.chunks[idx]
        basin, day, target = chunk[:, 0], chunk[:, 1], chunk[:, 2]
        n = len(basin)

        x_h = np.empty((n, self.k_h, 3), dtype=np.float32)
        x_d = np.empty((n, self.k_d, 3), dtype=np.float32)
        for i, (b, t) in enumerate(zip(basin, target)):
            x_h[i] = self.forcing[b, t - self.k_h : t, :3]
            end = (t - self.day0) // 24
            x_d[i] = self.daily[b, end - self.k_d : end]

        out = {
            "x": {
                "D": torch.from_numpy(x_d),
                "H": torch.from_numpy(x_h),
                "S": torch.from_numpy(self.static[basin]).contiguous(),
            },
            "stations": [self.station_ids[b] for b in basin],
            "dates": self.observed_dates[day],
            "y_daily_obs": torch.from_numpy(self.observed[basin, day].astype(np.float32)),
            "hours": torch.from_numpy(np.asarray(target, dtype=np.int64)),
            "stn_std": torch.from_numpy(self.station_y_std[basin]),
        }
        if self.with_daily:
            # The transfer loss wants the daily target standardised the same way the
            # model's outputs are, and a mask over the 24 hours it aggregates. Every
            # sample here sits at 23:00 with a verified-finite window, so the day is
            # complete and the mask is all-ones -- kept only so the loss signature
            # matches the global path.
            # The observations are mm/d; the model and its y scaler are mm/h. Divide by
            # the window before standardising, or the target is 24x too large and the
            # model learns to over-predict by that factor while the loss converges
            # perfectly well in its own wrong units. (This is the second time a mm/h vs
            # mm/d slip has reached a run on this path.)
            daily_mm_per_h = self.observed[basin, day].astype(np.float32) / DAILY_WINDOW
            y_daily = (daily_mm_per_h - self.y_mean) / self.y_std
            out["y_daily"] = torch.from_numpy(y_daily)
            out["daily_mask"] = torch.ones((n, 24), dtype=torch.bool)
            out["y"] = torch.from_numpy(y_daily)   # unused by the daily loss, kept for shape
        else:
            out["y_daily"] = None
            out["daily_mask"] = None
        return out
