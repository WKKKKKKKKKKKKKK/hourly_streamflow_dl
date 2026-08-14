"""Put the African hourly forcing on the same products the model was trained on.

Step 4 drives the model with ERA5-Land because nothing else offers hourly forcing
for these basins. But ERA5-Land's ``potential_evaporation`` is not the quantity the
model learned from: training used **Penman** PET, and over these 294 basins
ERA5-Land pev is **2.29x** larger (3978 vs 1737 mm/yr). Standardised with the
training scalers that lands PET at mean z = +2.54 with **30.8%** of hours beyond
z = 3 -- the model would be reading an input it never saw, and a poor Africa score
would say more about the forcing product than about transfer.

The training products themselves ARE available daily for exactly these basins
(``MSWEP_V280_Past_penman_..._dynamic.nc`` holds pet/pcp/temp for all 294). So keep
ERAA5-Land only for the sub-daily SHAPE and take the magnitude from the training
product:

  pet, pcp   fluxes that accumulate: scale each day's hourly values so their sum
             equals the daily total. A day ERA5-Land calls dry while the training
             product is wet gets the total spread evenly, since there is no shape
             to borrow.
  temp       an average, not a total: shift each day by a constant so the daily
             mean matches, which preserves the diurnal range instead of squashing it.

After rescaling, PET sits at z = +0.73 -- the same order as temperature's +0.64,
i.e. an ordinary Africa-is-hotter shift rather than an out-of-distribution input.

Day membership follows the ERA5 accumulation convention: a value stamped t covers
``(t-1h, t]``, so 01:00 of day D through 00:00 of day D+1 all belong to day D.

    python -m scripts.rescale_africa_forcing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

from data.africa import DAILY_DYNAMIC, DAILY_EPOCH

DEFAULT_IN = (
    "/ibex/user/kongw0a/era5_land_africa_forcing/era5_land_africa_hourly_forcing.nc"
)
ACCUMULATED = ("pet", "pcp")
AVERAGED = ("temp",)


def load_daily_products(station_ids: list[str], logger=None) -> tuple[dict[str, np.ndarray], pd.DatetimeIndex]:
    """Daily pet/pcp/temp for these basins, from the file the model was trained on."""
    handle = nc.Dataset(DAILY_DYNAMIC)
    features = [str(x) for x in handle.variables["dynamic_features"][:]]
    stations = [str(x) for x in handle.variables["station"][:]]
    lookup = {name: i for i, name in enumerate(stations)}
    missing = [s for s in station_ids if s not in lookup]
    if missing:
        raise SystemExit(f"{len(missing)} basins absent from {DAILY_DYNAMIC.name}: {missing[:5]}")
    rows = [lookup[s] for s in station_ids]

    times = pd.to_datetime(DAILY_EPOCH) + pd.to_timedelta(
        np.asarray(handle.variables["time"][:]), unit="D"
    )
    out = {}
    for name in ACCUMULATED + AVERAGED:
        if name not in features:
            raise SystemExit(f"{DAILY_DYNAMIC.name} has no {name!r}; present: {features}")
        layer = features.index(name)
        out[name] = np.asarray(handle.variables["dyn"][rows, :, layer], dtype=np.float32)
    handle.close()
    if logger:
        logger.info("daily training products: %s over %d basins x %d days",
                    list(out), len(station_ids), len(times))
    return out, times


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescale African hourly forcing onto the training products.")
    parser.add_argument("--in-file", default=DEFAULT_IN)
    parser.add_argument("--out", default=None, help="Defaults to <in>_penman.nc")
    parser.add_argument("--config", default="configs/phase1_runB.yaml")
    args = parser.parse_args()

    in_path = Path(args.in_file)
    out_path = Path(args.out) if args.out else in_path.with_name(
        in_path.stem + "_penman" + in_path.suffix
    )

    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("rescale")

    hourly = xr.open_dataset(in_path)
    station_ids = [str(s) for s in hourly["station"].values]
    times = pd.to_datetime(hourly["time"].values)
    logger.info("hourly forcing: %d basins x %d hours (%s .. %s)",
                len(station_ids), len(times), times[0], times[-1])

    daily, daily_times = load_daily_products(station_ids, logger)

    # A stamp t covers (t-1h, t], so subtract an hour before taking the date.
    owner = (times - pd.Timedelta(hours=1)).normalize()
    day_index = pd.DatetimeIndex(owner)
    slot = daily_times.get_indexer(day_index)
    if (slot < 0).any():
        # The 00:00 stamp of the first hour belongs to the day BEFORE the record
        # starts, so a handful of leading hours have no daily counterpart. Dropping
        # them is right; silently rescaling them against the wrong day is not.
        keep = slot >= 0
        dropped = day_index[~keep].unique()
        if len(dropped) > 2:
            raise SystemExit(
                f"{len(dropped)} hourly days absent from the daily file (e.g. "
                f"{list(dropped[:3])}) -- too many to be the record-boundary effect"
            )
        logger.warning("dropping %d hour(s) owned by %s, outside the daily record",
                       int((~keep).sum()), [str(d.date()) for d in dropped])
        hourly = hourly.isel(time=np.flatnonzero(keep))
        times = pd.to_datetime(hourly["time"].values)
        owner = (times - pd.Timedelta(hours=1)).normalize()
        day_index = pd.DatetimeIndex(owner)
        slot = daily_times.get_indexer(day_index)

    # Group columns by day once; every basin reuses the same grouping.
    order = np.argsort(slot, kind="stable")
    bounds = np.r_[0, np.flatnonzero(np.diff(slot[order])) + 1, slot.size]
    groups = [order[lo:hi] for lo, hi in zip(bounds[:-1], bounds[1:])]
    group_day = np.array([slot[g[0]] for g in groups])
    logger.info("%d whole days covered", len(groups))

    result = {}
    report = {}
    for name in ACCUMULATED + AVERAGED:
        values = np.asarray(hourly[name].values, dtype=np.float64).copy()
        target_all = daily[name]
        before = float(np.nanmean(values))
        n_uniform = 0

        for gi, cols in enumerate(groups):
            target = target_all[:, group_day[gi]].astype(np.float64)
            block = values[:, cols]
            if name in ACCUMULATED:
                # Sum of hourly mm/h over a day == the daily mm/d total.
                current = np.nansum(block, axis=1)
                scale = np.where(current > 0, target / np.where(current > 0, current, 1.0), 0.0)
                scaled = block * scale[:, None]
                # ERA5-Land says nothing happened but the training product says it
                # did: no shape to borrow, so spread the total evenly.
                flat = (current <= 0) & np.isfinite(target) & (target > 0)
                if flat.any():
                    scaled[flat] = (target[flat] / len(cols))[:, None]
                    n_uniform += int(flat.sum())
                values[:, cols] = scaled
            else:
                # Shift, not scale: keeps the diurnal amplitude intact.
                offset = target - np.nanmean(block, axis=1)
                values[:, cols] = block + offset[:, None]

        result[name] = values.astype(np.float32)
        report[name] = (before, float(np.nanmean(values)), n_uniform)
        logger.info("%-5s hourly mean %.4f -> %.4f%s", name, before, report[name][1],
                    f" | {n_uniform} basin-days spread uniformly" if n_uniform else "")

    ds = xr.Dataset(
        {name: (("station", "time"), values) for name, values in result.items()},
        coords={"station": station_ids, "time": times},
    )
    ds["n_cells"] = ("station", np.asarray(hourly["n_cells"].values))
    units = {"pcp": "mm h-1", "pet": "mm h-1", "temp": "degC"}
    for name in result:
        ds[name].attrs.update(units=units[name], source="ERA5-Land shape, training-product magnitude")
    ds.attrs.update(
        note=(
            "African hourly forcing with ERA5-Land sub-daily shape rescaled so each day "
            "matches the daily training products (Penman PET, MSWEP precipitation, temp) "
            f"from {DAILY_DYNAMIC.name}. pet/pcp scaled to the daily total; temp shifted "
            "to the daily mean. A stamp t covers (t-1h, t]."
        ),
    )
    ds.to_netcdf(out_path)
    logger.info("wrote %s (%.1f MiB)", out_path, out_path.stat().st_size / 1024**2)

    # The point of the exercise: where the inputs now sit in the training distribution.
    from common.config import load_config
    from data.dataset import load_scalers

    scalers = load_scalers(load_config(args.config, None).data.root)
    logger.info("standardised position after rescaling (was pet z=+2.54, 30.8%% beyond z=3):")
    for name in ("pet", "pcp", "temp"):
        flat = result[name].ravel()
        flat = flat[np.isfinite(flat)]
        z = (flat - scalers["x_dyn_mean"][name]) / scalers["x_dyn_std"][name]
        logger.info("  %-5s mean z=%+.2f | median z=%+.2f | beyond z=3: %.1f%%",
                    name, z.mean(), np.median(z), 100 * (z > 3).mean())


if __name__ == "__main__":
    main()
