"""Daily series of every African catchment, and hourly ones of a few, for the African flood examples.

The scored African tables keep one KGE per basin and member, not the series. This re-runs
the five members of M1 (outputs/al_v2_africa_m1b_fold*) and of the daily LSTM
(outputs/al_v2_africa_daily_lstm_fold*) on the validation period of all catchments and writes:

  daily_series_members.csv.gz  per basin, day and member: obs, M1 (24-h mean of its hourly
                               output) and daily LSTM (its daily head), mm/d -- so daily peak
                               metrics can be taken per member and then the median over members,
                               the convention of outputs/al_v2_africa_five_way
  daily_series.csv             per basin and day: obs, the members' mean of each arm, ERA5-Land
  hourly_series_M1.csv.gz      M1's members' mean hourly series (mm/h) for the candidate
                               catchments only (M1 KGE percentiles 10, 20, 25, 30, ..., 75, 80, 90)

Inference only.

Each member is checked against the scored table before anything is written: its per-basin
daily KGE recomputed here must equal per_basin_M1.csv of that member, otherwise these are not
the scored models on the scored windows.

Candidates are picked on M1's daily KGE (the median over members, outputs/al_v2_africa_five_way)
so the examples can span the outcome rather than be three good ones.

    python -m scripts.analysis.africa_example_series            # -> outputs/al_v2_africa_examples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from common.alignment import alignment_of_cfg, assert_same_alignment
from common.config import load_config, resolve
from common.utils import get_device, setup_logging
from data.africa import (AfricaWindowDataset, apply_onehot, build_static_matrix, load_hourly_forcing,
                         load_observed_daily)
from data.dataset import load_dataset_config, load_scalers, make_loader, resolve_static_spec
from common.metrics import kge_components
from models.mtslstm import build_model
from scripts.analysis.africa_daily_three_way import DEFAULT_ERA5, load_era5_daily

# ERA5-Land hourly forcing rescaled to the daily training products (scripts/prep/rescale_africa_forcing.py)
DEFAULT_FORCING = (
    "/ibex/user/kongw0a/era5_land_africa_forcing/"
    "era5_land_africa_hourly_forcing_penman.nc"
)

DAILY_WINDOW = 24
ARMS = {  # arm -> (config, member checkpoints, head)
    "M1": ("configs/aligned/phase1_runB_v2.yaml", "outputs/al_v2_africa_m1b_fold{k}", "agg"),
    "daily_lstm": ("configs/phase1_africa_daily_lstm_aligned.yaml", "outputs/al_v2_africa_daily_lstm_fold{k}", "daily"),
}
SELECT_FRAC = 0.15   # as both arms were trained (slurm/submit_3_search_baselines_africa.sh)


CANDIDATES = (0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90)


def pick(per_basin: Path, quantiles=CANDIDATES) -> list[str]:
    t = pd.read_csv(per_basin).dropna(subset=["M1"]).sort_values("M1").reset_index(drop=True)
    return [str(t.iloc[int(len(t) * q)]["station_id"]) for q in quantiles]


@torch.no_grad()
def run(model, loader, device, y_mean, y_std, head, keep_hourly: set[str]):
    model.eval()
    rows, hourly = [], []
    for batch in loader:
        x = {k: v.to(device) for k, v in batch["x"].items()}
        out = model({"D": x["D"], "H": x["H"]}, x["S"])
        tail = (out["H_seq"][:, -DAILY_WINDOW:].float().cpu().numpy() * y_std + y_mean)   # mm/h
        day = tail.mean(1) * DAILY_WINDOW if head == "agg" else \
            (out["D"].float().cpu().numpy() * y_std + y_mean) * DAILY_WINDOW            # mm/d
        dates = pd.to_datetime(pd.Series(batch["dates"]).values)
        rows.append(pd.DataFrame({"station_id": [str(s) for s in batch["stations"]], "date": dates,
                                  "sim": day, "obs": batch["y_daily_obs"].numpy()}))
        if head == "agg":
            for s, d, h in zip(batch["stations"], dates, tail):
                if str(s) not in keep_hourly:
                    continue
                hourly.append(pd.DataFrame({"station_id": str(s), "time": d + pd.to_timedelta(np.arange(24), "h"),
                                            "sim": h}))
    return pd.concat(rows, ignore_index=True), (pd.concat(hourly, ignore_index=True) if hourly else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stations", default=None, help="comma-separated ids for the hourly series; "
                    "default: the M1 KGE percentile candidates")
    ap.add_argument("--per-basin", default="outputs/al_v2_africa_five_way/per_basin_four_way.csv", type=Path)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--out-dir", default="outputs/al_v2_africa_examples", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.out_dir / "example_series.log")
    device = get_device()
    wanted = [s.strip() for s in args.stations.split(",")] if args.stations else pick(args.per_basin)
    logger.info("hourly series for: %s", ", ".join(wanted))

    basins = gpd.read_file(resolve("africa/africa_basins.gpkg"))
    ids = basins.station_id.astype(str).tolist()
    daily, hourly, members = {}, None, []
    for arm, (config, member, head) in ARMS.items():
        cfg = load_config(config, [])
        scalers = load_scalers(cfg.data.root)
        y_mean, y_std = float(scalers["y_mean"]), float(scalers["y_std"])
        dsc = load_dataset_config(cfg.data.root)
        keep, onehot, static_names = resolve_static_spec(cfg.data.root, cfg.data.get("static_exclude"),
                                                         cfg.data.get("onehot_static"))
        forcing, ftimes = load_hourly_forcing(DEFAULT_FORCING, ids, scalers, logger=logger)
        observed, odates = load_observed_daily(ids, logger=logger)
        full, _ = build_static_matrix(ids, list(dsc["static_features"]), scalers, logger=logger)
        static = apply_onehot(full, keep, onehot).astype(np.float32)
        ds = AfricaWindowDataset(forcing=forcing, forcing_times=ftimes, static=static, observed=observed,
                                 observed_dates=odates, station_ids=ids,
                                 lookback_hourly=int(cfg.data.lookback_hourly),
                                 lookback_daily=int(cfg.data.get("lookback_daily", 365)),
                                 chunk_size=int(cfg.data.get("chunk_size", 512)), split="validation",
                                 train_frac=0.7, select_frac=SELECT_FRAC, scalers=scalers, logger=logger,
                                 daily_includes_target_day=bool(cfg.data.get("daily_includes_target_day", False)),
                                 window_alignment=alignment_of_cfg(cfg))
        loader = make_loader(ds, num_workers=4)
        sims, hours = [], []
        for k in [int(f) for f in args.folds.split(",")]:
            d = Path(member.format(k=k))
            model = build_model(cfg, dyn_input_size=len(dsc["dyn_features"]),
                                static_input_size=len(static_names)).to(device)
            assert_same_alignment(d / "best_africa_model.pth", cfg, logger=logger)
            model.load_state_dict(torch.load(d / "best_africa_model.pth", map_location=device, weights_only=True))
            day, hr = run(model, loader, device, y_mean, y_std, head, set(wanted))
            # same models, same windows: per-basin KGE must equal the scored member table
            ref = pd.read_csv(d / "per_basin_M1.csv").set_index("station_id").kge
            gap, n = 0.0, 0
            for sid, g in day.groupby("station_id"):
                ok = np.isfinite(g.sim) & np.isfinite(g.obs)
                if sid not in ref.index or ok.sum() < 100:
                    continue
                kge = kge_components(g.obs[ok].to_numpy(float), g.sim[ok].to_numpy(float))[0]
                gap, n = max(gap, abs(kge - ref[sid])), n + 1
                if abs(kge - ref[sid]) > 1e-3:
                    raise SystemExit(f"{arm} fold {k} {sid}: KGE {kge:.4f} here vs {ref[sid]:.4f} scored")
            logger.info("%s fold %d reproduces the scored KGE of %d basins (max |diff| %.1e)", arm, k, n, gap)
            sims.append(day.set_index(["station_id", "date"]))
            members.append(day.assign(arm=arm, member=k))
            if hr is not None:
                hours.append(hr.set_index(["station_id", "time"]).sim)
        daily[arm] = pd.concat([s.sim for s in sims], axis=1).mean(axis=1)
        daily["obs"] = sims[0].obs
        if hours:
            hourly = pd.concat(hours, axis=1).mean(axis=1).rename("M1").reset_index()

    era5 = load_era5_daily(Path(DEFAULT_ERA5), logger=logger).set_index(["station_id", "date"]).era5_land
    out = pd.DataFrame(daily).join(era5, how="left").reset_index()
    out.to_csv(args.out_dir / "daily_series.csv", index=False)
    mem = pd.concat(members).pivot_table(index=["station_id", "date", "member"], columns="arm", values="sim")
    mem = mem.join(out.set_index(["station_id", "date"]).obs).reset_index()
    mem.to_csv(args.out_dir / "daily_series_members.csv.gz", index=False, compression="gzip")
    hourly.to_csv(args.out_dir / "hourly_series_M1.csv.gz", index=False, compression="gzip")
    logger.info("wrote %s (%d rows) and hourly_series_M1.csv.gz (%d rows)",
                args.out_dir / "daily_series.csv", len(out), len(hourly))


if __name__ == "__main__":
    main()
