"""Hourly KGE broken down by hour of day -- what the stride-24 index hides.

Run B trained and reported on targets at 23:00 only, because a target must sit at
hour 23 for the last 24 hourly steps to form one calendar day. Two different things
could make 23:00 look good, and a single pooled number cannot separate them:

  intra-day shape   Other hours are genuinely harder to predict -- the daily-mean
                    supervision has nothing to say about within-day timing, and
                    23:00 happens to be a mild target.
  alignment OOD     The daily branch always ended at the midnight 24 h before the
                    target, so the model only ever saw ONE (daily-end, target)
                    offset. At 10:00 the offset is 10 h, which it never trained on.

The signatures differ. Genuinely-harder hours vary smoothly around the clock --
a diurnal curve. An out-of-distribution alignment shows up as a spike AT 23:00
with the other 23 hours roughly flat and uniformly worse. This scores every hour
separately so the shape is visible.

    python -m scripts.kge_by_hour --config configs/phase1_runB.yaml --fold 0 \
        --set data.eval_sample_index=samples_evalhours.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.config import add_common_args, load_config, resolve
from common.metrics import compute_kge, compute_nse, kge_components
from common.utils import get_device, setup_logging
from data.dataset import load_dataset_config, load_scalers, make_loader, resolve_static_spec
from data.folds import domain_stations, load_folds
from data.sources import build_eval_set
from models.mtslstm import build_model


@torch.no_grad()
def collect_by_hour(model, loader, device, y_mean, y_std, logger=None):
    """(station, hour-of-day) -> pooled obs/sim, in physical units."""
    model.eval()
    obs_by: dict[tuple[str, int], list] = {}
    sim_by: dict[tuple[str, int], list] = {}
    n_batches = 0

    for batch in loader:
        stations = batch["stations"]
        if not stations:
            continue
        if "hours" not in batch or batch["hours"] is None:
            raise ValueError("the dataset did not return absolute target hours; "
                             "this breakdown needs the cache path (data.source=hourly_cache)")
        x = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
        sim = model({"D": x["D"], "H": x["H"]}, x["S"])["H"].float().cpu().numpy()
        obs = batch["y"].numpy()
        hod = (batch["hours"].numpy() % 24).astype(int)

        codes = np.asarray(stations, dtype=object)
        for station, hour, s, o in zip(codes, hod, sim, obs):
            key = (str(station), int(hour))
            sim_by.setdefault(key, []).append(s)
            obs_by.setdefault(key, []).append(o)

        n_batches += 1
        if logger and n_batches % 500 == 0:
            logger.info("  %d batches | %d (station, hour) cells", n_batches, len(obs_by))

    rows = []
    for key, sims in sim_by.items():
        station, hour = key
        sim = np.asarray(sims, dtype=np.float64) * y_std + y_mean
        obs = np.asarray(obs_by[key], dtype=np.float64) * y_std + y_mean
        kge, r, a, b = kge_components(obs, sim)
        rows.append({
            "station_id": station, "hour": hour, "samples": obs.size,
            "kge": kge, "kge_r": r, "kge_alpha": a, "kge_beta": b,
            "nse": compute_nse(obs, sim), "obs_std": float(np.nanstd(obs)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Hourly KGE by hour of day."))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--domain", default="target", choices=["target", "source"])
    parser.add_argument("--checkpoint", default="both", choices=["M0", "M1", "both"])
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--min-obs-std", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    run_dir = Path(args.run_dir) if args.run_dir else resolve(cfg.output_root)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "diagnostics" / "by_hour"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / f"kge_by_hour_fold{args.fold}_{args.domain}.log")
    device = get_device()

    folds = load_folds(resolve(cfg.folds.file))
    source_stations, target_stations = domain_stations(folds, args.fold)
    stations = target_stations if args.domain == "target" else source_stations

    scalers = load_scalers(cfg.data.root)
    dyn_size = len(load_dataset_config(cfg.data.root)["dyn_features"])
    _, _, static_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static")
    )
    logger.info("fold %d | %s domain, %d stations | index %s", args.fold, args.domain,
                len(stations), cfg.data.get("eval_sample_index", f"stride{cfg.data.get('stride')}"))

    dataset = build_eval_set(cfg, stations, "validation", logger=logger)
    loader = make_loader(dataset, num_workers=int(cfg.get_path("transfer.num_workers", 4)),
                         pin_memory=bool(cfg.get_path("train.pin_memory", False)) and device.type == "cuda")

    wanted = ["M0", "M1"] if args.checkpoint == "both" else [args.checkpoint]
    paths = {
        "M0": run_dir / f"fold{args.fold}" / "pretrain" / "best_model.pth",
        "M1": run_dir / f"fold{args.fold}" / "transfer" / "best_transfer_model.pth",
    }

    summaries = {}
    for tag in wanted:
        if not paths[tag].exists():
            raise FileNotFoundError(f"{paths[tag]} missing")
        model = build_model(cfg, dyn_input_size=dyn_size, static_input_size=len(static_names)).to(device)
        model.load_state_dict(torch.load(paths[tag], map_location=device, weights_only=True))
        logger.info("%s: scoring %s", tag, paths[tag])
        frame = collect_by_hour(model, loader, device,
                                float(scalers["y_mean"]), float(scalers["y_std"]), logger)
        frame["checkpoint"] = tag
        frame.to_csv(out_dir / f"by_hour_fold{args.fold}_{args.domain}_{tag}.csv", index=False)

        kept = frame.loc[frame["obs_std"] >= args.min_obs_std]
        table = kept.groupby("hour").agg(
            n_stations=("station_id", "nunique"),
            samples=("samples", "sum"),
            median_kge=("kge", "median"),
            median_r=("kge_r", "median"),
            median_alpha=("kge_alpha", "median"),
            median_beta=("kge_beta", "median"),
        ).reset_index()
        summaries[tag] = table
        logger.info("%s by hour:\n%s", tag,
                    table.to_string(index=False, float_format=lambda v: f"{v: .4f}"))

        others = table.loc[table["hour"] != 23, "median_kge"]
        at23 = table.loc[table["hour"] == 23, "median_kge"]
        if len(at23) and len(others):
            logger.info("%s: 23:00 KGE %.4f vs other hours %.4f (spread %.4f) -> gap %+.4f",
                        tag, at23.iloc[0], others.median(), others.max() - others.min(),
                        at23.iloc[0] - others.median())

    combined = pd.concat([t.assign(checkpoint=k) for k, t in summaries.items()], ignore_index=True)
    combined.to_csv(out_dir / f"by_hour_summary_fold{args.fold}_{args.domain}.csv", index=False)
    (out_dir / f"by_hour_fold{args.fold}_{args.domain}.json").write_text(
        json.dumps({"fold": args.fold, "domain": args.domain,
                    "index": cfg.data.get("eval_sample_index"),
                    "by_hour": combined.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
