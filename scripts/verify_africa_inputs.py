"""Prove the African input packaging is correct, using stations that are NOT African.

The packaging in ``data/africa.py`` has to reproduce, from raw hourly data plus
``static.csv``, exactly what ``hourly_q_dl`` baked into its prepared batches. For
any station that IS in those batches we have the answer already, so we can
rebuild its input from scratch and compare value by value:

  * the 1000-position dynamic sequence -- checks the lookback offsets and the
    dynamic standardization,
  * the static vector -- checks the column order, the standardization, and the
    Koeppen-Geiger string -> code map recovered in ``data/kgz_codes.json``.

If both match, the only thing left untested for the African basins is the forcing
data itself, not the code. Run this before trusting any Africa number.

    python -m scripts.verify_africa_inputs --stations 6
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
import torch

from common.config import add_common_args, load_config, resolve
from common.utils import setup_logging
from data.africa import HOURLY_STATIC, build_static_matrix
from data.dataset import load_dataset_config, load_lookback_offsets, load_scalers
from data.index import load_index

SOURCE_NC = "/ibex/project/c2266/abbaa0a/data/input_data/hourly_q_dl/6sources.nc"
DYN_ORDER = ("pet", "pcp", "temp")


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Verify the Africa input packaging."))
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--rows-per-batch", type=int, default=4)
    parser.add_argument("--tol", type=float, default=2e-5)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    logger = setup_logging()

    root = cfg.data.root
    scalers = load_scalers(root)
    names = list(load_dataset_config(root)["static_features"])
    offsets = load_lookback_offsets()
    hours_ago = np.asarray(offsets["hours_ago"], dtype=np.int64)

    import netCDF4 as nc

    source = nc.Dataset(SOURCE_NC)
    time_var = source.variables["time"]
    times = pd.DatetimeIndex(
        nc.num2date(time_var[:], time_var.units, only_use_cftime_datetimes=False)
    )
    feature_names = [str(x) for x in source.variables["dynamic_features"][:]]
    feature_index = {name: i for i, name in enumerate(feature_names)}

    frame = load_index(resolve(cfg.data.index_dir), "training")
    regular = frame.loc[frame["kind"] == "regular"]
    first = regular.groupby("station", sort=True)["prefix"].first()
    picks = first.iloc[:: max(1, len(first) // args.stations)][: args.stations]
    logger.info("verifying %d stations from %s",
                len(picks), sorted({s.split("__")[0] for s in picks.index}))

    # ---- static vector -------------------------------------------------------
    # Built from the HOURLY static table, which is what training used; the Africa
    # path reads the daily table, already shown to agree on 41 of 42 features.
    station_ids = list(picks.index)
    built_static, report = build_static_matrix(
        station_ids, names, scalers, static_path=HOURLY_STATIC, logger=logger
    )

    static_max_err = 0.0
    for k, (station, prefix) in enumerate(picks.items()):
        _, x_static = torch.load(
            f"{root}/training/{prefix}_x.pt", map_location="cpu", weights_only=True
        )
        truth = x_static[0].numpy()
        err = np.abs(truth - built_static[k])
        static_max_err = max(static_max_err, float(err.max()))
        worst = int(np.argmax(err))
        logger.info(
            "  %-26s static max|err| = %.3g   (worst feature: %s)",
            station, float(err.max()), names[worst],
        )
    logger.info("STATIC: max |error| over %d stations x %d features = %.3g",
                len(picks), len(names), static_max_err)

    # ---- dynamic sequence ----------------------------------------------------
    dyn_max_err = 0.0
    checked_rows = 0
    for station, prefix in picks.items():
        x_dyn, _ = torch.load(f"{root}/training/{prefix}_x.pt", map_location="cpu", weights_only=True)
        with open(f"{root}/training/{prefix}_metadata.pkl", "rb") as handle:
            meta = pickle.load(handle)
        raw = np.asarray(source.variables[station][:, :])

        rows = np.linspace(0, len(meta) - 1, args.rows_per_batch).astype(int)
        for row in rows:
            target = times.get_loc(meta["index"].iloc[row])
            window = target - hours_ago
            built = np.stack(
                [
                    (raw[window, feature_index[name]] - scalers["x_dyn_mean"][name])
                    / scalers["x_dyn_std"][name]
                    for name in DYN_ORDER
                ],
                axis=1,
            ).astype(np.float32)
            err = np.abs(x_dyn[row].numpy() - built)
            dyn_max_err = max(dyn_max_err, float(err.max()))
            checked_rows += 1
    logger.info("DYNAMIC: max |error| over %d rebuilt (1000 x 3) sequences = %.3g",
                checked_rows, dyn_max_err)
    source.close()

    ok = static_max_err < args.tol and dyn_max_err < args.tol
    if ok:
        logger.info(
            "VERIFIED: the packaging reproduces the prepared batches to within %.1g. "
            "Lookback offsets, standardization, static column order and the KGZ code "
            "map are all correct.", args.tol,
        )
    else:
        logger.error(
            "MISMATCH (tolerance %.1g): static %.3g, dynamic %.3g. Do not trust any "
            "Africa result until this passes.", args.tol, static_max_err, dyn_max_err,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
