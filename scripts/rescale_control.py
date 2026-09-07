"""Is the daily-only gain anything more than a per-gauge rescaling?

This is the control that decides how large a claim Phase I can make. The component
decomposition says fine-tuning on daily aggregates closes 23 percent of the amplitude gap
and 17 percent of the volume gap while closing only 3 percent of the timing gap. A reader is
entitled to the obvious reading of that: the transfer step learned a scale factor, and a
scale factor is arithmetic, not learning. In Africa the reading is sharper still, at 67 and
77 percent against 32.

So this fits the cheapest correction that daily data allows, applies it to the ZERO-SHOT
predictions, and rescores. If that recovers most of the gain, the contribution is a bias
correction with a neural network attached. If it recovers little, the fine-tuned model
learned something a rescaling cannot express, and the objection is answered with a number.

    y' = a + b * y     per gauge, fitted on the target TRAINING period's daily means,
                       which is exactly the data fine-tuning was allowed to see

Fitting on the training period matters. Fitting on the validation period would hand the
control information the model never had and would make it win for the wrong reason.

Two forms, because they answer different objections:

    volume    b = 1, a chosen so the daily mean matches. Repairs beta alone.
    scale     b = sd_obs / sd_sim and a chosen so the mean matches. Repairs beta and alpha.

The scoring pass is arithmetic rather than a second model run. An affine map with b > 0
leaves the correlation untouched and acts on the other two KGE components in closed form:

    r'     = r
    alpha' = b * alpha
    beta'  = (a + b * mu_sim) / mu_obs

Every quantity on the right is already in the saved per-gauge tables, so only the fit needs
a forward pass. The closed form is verified against a direct recomputation on one fold
rather than trusted.

    python -m scripts.rescale_control --config configs/phase1_runB_v2.yaml --folds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.config import load_config, resolve
from common.utils import setup_logging, get_device
from data.dataset import load_scalers, load_dataset_config, resolve_static_spec
from data.dataset import make_loader
from data.sources import build_bundle
from models.mtslstm import build_model
from data.folds import domain_stations, load_folds

DAILY_WINDOW = 24


def fit_statistics(model, loader, device, y_mean, y_std, logger=None):
    """Per-gauge mean and standard deviation of DAILY prediction and observation.

    Both in NORMALISED units, which is what the dataset stores y_daily in.

    Daily, not hourly, because the premise is that only daily observations exist in the
    target region. Accumulated as running sums so a fold's training period never has to be
    held in memory as a series.
    """
    model.eval()
    acc: dict[str, np.ndarray] = {}
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            stations = batch["stations"]
            if not stations:
                continue
            y_daily = batch.get("y_daily")
            if y_daily is None:
                raise ValueError("the fitting pass needs a dataset built with with_daily=True")
            y_daily = y_daily.numpy()
            keep = np.isfinite(y_daily)
            if not keep.any():
                continue
            x = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
            out = model({"D": x["D"], "H": x["H"]}, x["S"])
            # The daily value the model implies: the mean of its last 24 hourly outputs,
            # which is the same aggregation the daily objective and the daily scores use.
            sim = out["H_seq"][:, -DAILY_WINDOW:].squeeze(-1).float().cpu().numpy().mean(axis=1)
            # BOTH sides stay in normalised space. batch["y_daily"] is normalised, and
            # de-normalising only the prediction mixes the two: the fitted observation mean
            # came out NEGATIVE, which runoff cannot be, and the affine correction built on
            # it drove median KGE from +0.63 to -1.2. The scale factors cancel in the ratio
            # b = sd_obs / sd_sim anyway, and the offset a is applied to a normalised
            # prediction, so nothing here needs physical units.
            for station, s, o, k in zip(stations, sim, y_daily, keep):
                if not k:
                    continue
                row = acc.setdefault(station, np.zeros(5))
                row += (1.0, s, s * s, o, o * o)
            n_batches += 1
            if logger and n_batches % 200 == 0:
                logger.info("  fitting pass: %d batches", n_batches)

    # A sanity check that would have caught the units bug on the first run. In normalised
    # space the observed daily mean is (physical - y_mean) / y_std, so it is bounded below
    # by -y_mean / y_std for a non-negative quantity like runoff. Anything below that means
    # the two sides of the fit are in different units.
    floor = -y_mean / y_std - 1e-6
    rows = []
    for station, (n, s1, s2, o1, o2) in acc.items():
        if n < 2:
            continue
        mu_s, mu_o = s1 / n, o1 / n
        var_s, var_o = s2 / n - mu_s ** 2, o2 / n - mu_o ** 2
        if mu_o < floor:
            raise SystemExit(
                f"{station}: fitted observed daily mean {mu_o:.4f} is below the normalised "
                f"floor {floor:.4f}. Runoff cannot be negative, so the prediction and the "
                "observation are in different units here."
            )
        rows.append({
            "station_id": station, "n_days_fit": int(n),
            "fit_sim_mean": mu_s, "fit_obs_mean": mu_o,
            "fit_sim_std": float(np.sqrt(max(var_s, 0.0))),
            "fit_obs_std": float(np.sqrt(max(var_o, 0.0))),
        })
    return pd.DataFrame(rows)


def apply_affine(table: pd.DataFrame, form: str) -> pd.DataFrame:
    """KGE after a per-gauge affine correction, in closed form.

    ``table`` carries the saved M0 validation components and the fitted training statistics.
    """
    mu_s, sd_s = table["fit_sim_mean"], table["fit_sim_std"]
    mu_o, sd_o = table["fit_obs_mean"], table["fit_obs_std"]

    if form == "volume":
        b = pd.Series(1.0, index=table.index)
    elif form == "scale":
        # Guarded: a gauge whose zero-shot prediction is almost constant over the fitting
        # period gives a meaningless ratio, and letting it through would manufacture a huge
        # correction that the control does not deserve credit or blame for.
        b = (sd_o / sd_s.where(sd_s > 1e-6)).clip(upper=20.0)
    else:
        raise ValueError(form)
    a = mu_o - b * mu_s

    # Validation-period simulated mean, recovered from the saved components:
    # beta = mu_sim / mu_obs and alpha = sd_sim / sd_obs.
    val_mu_s = table["M0_kge_beta"] * table["obs_mean"]
    out = pd.DataFrame({
        "station_id": table["station_id"],
        "r": table["M0_kge_r"],                                   # affine-invariant
        "alpha": b * table["M0_kge_alpha"],
        "beta": (a + b * val_mu_s) / table["obs_mean"],
    })
    out["kge"] = 1.0 - np.sqrt(
        (out["r"] - 1) ** 2 + (out["alpha"] - 1) ** 2 + (out["beta"] - 1) ** 2
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/phase1_runB_v2.yaml")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--run", default="outputs/v2_runB", type=Path)
    parser.add_argument("--out-dir", default="outputs/v2_rescale_control", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.out_dir / "rescale_control.log")

    cfg = load_config(args.config, None)
    device = get_device()
    scalers = load_scalers(cfg.data.root)
    y_mean, y_std = scalers["y_mean"], scalers["y_std"]
    ds_config = load_dataset_config(cfg.data.root)
    _, _, static_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static"))
    folds_table = load_folds(resolve(cfg.folds.file))

    pieces = []
    for fold in [int(f) for f in args.folds.split(",")]:
        ckpt = args.run / f"fold{fold}" / "pretrain" / "best_model.pth"
        if not ckpt.exists():
            logger.info("fold %d: no pretrained checkpoint, skipping", fold)
            continue
        m0_csv = (args.run / f"fold{fold}" / "transfer" /
                  f"per_station_hourly_fold{fold}_M0_target_hourly.csv")
        m1_csv = (args.run / f"fold{fold}" / "transfer" /
                  f"per_station_hourly_fold{fold}_M1_target_hourly.csv")
        if not (m0_csv.exists() and m1_csv.exists()):
            logger.info("fold %d: saved M0/M1 tables missing, skipping", fold)
            continue

        _, target_stations = domain_stations(folds_table, fold)
        model = build_model(cfg, dyn_input_size=len(ds_config["dyn_features"]),
                            static_input_size=len(static_names)).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

        logger.info("fold %d: fitting on the target TRAINING period, %d gauges",
                    fold, len(target_stations))
        # build_bundle, not build_eval_loader with split "train". Run B reads from the
        # hourly cache, which has no batch index for the training split, so the eval-loader
        # path fails there. This is the same call the transfer step itself uses to assemble
        # the fine-tuning pool, which is also exactly the data the correction is allowed
        # to see, so reusing it keeps the control honest by construction.
        train_ds = build_bundle(cfg, target_stations, with_daily=True, logger=logger).train
        loader = make_loader(train_ds, num_workers=int(cfg.transfer.num_workers),
                             pin_memory=False)
        fit = fit_statistics(model, loader, device, y_mean, y_std, logger=logger)
        logger.info("  fitted %d gauges", len(fit))

        m0 = pd.read_csv(m0_csv).add_prefix("M0_").rename(columns={"M0_station_id": "station_id"})
        m1 = pd.read_csv(m1_csv)[["station_id", "kge"]].rename(columns={"kge": "M1_kge"})
        table = (m0.rename(columns={"M0_obs_mean": "obs_mean", "M0_obs_std": "obs_std"})
                 .merge(fit, on="station_id").merge(m1, on="station_id"))
        table = table[(table["obs_std"] >= 1e-3) & (table["M0_score_status"] == "ok")]
        table["fold"] = fold

        for form in ("volume", "scale"):
            corrected = apply_affine(table, form)
            table[f"rescale_{form}_kge"] = corrected["kge"].to_numpy()
        pieces.append(table)
        logger.info("  fold %d: %d gauges usable", fold, len(table))

    if not pieces:
        raise SystemExit("no folds produced a comparison")
    all_folds = pd.concat(pieces, ignore_index=True)

    logger.info("")
    logger.info("paired per gauge over %d gauges, %d folds:",
                len(all_folds), all_folds["fold"].nunique())
    base = all_folds["M0_kge"]
    gain_m1 = float((all_folds["M1_kge"] - base).median())
    logger.info("  %-28s median KGE %.4f", "M0 zero-shot", base.median())
    logger.info("  %-28s median KGE %.4f  gain %+.4f", "M1 daily fine-tuning",
                all_folds["M1_kge"].median(), gain_m1)
    summary = {"n_gauges": int(len(all_folds)), "gain_M1": gain_m1}
    for form in ("volume", "scale"):
        column = all_folds[f"rescale_{form}_kge"]
        gain = float((column - base).median())
        share = gain / gain_m1 if gain_m1 else float("nan")
        head = (all_folds["M1_kge"] - column)
        logger.info(
            "  %-28s median KGE %.4f  gain %+.4f  = %.1f%% of the fine-tuning gain | "
            "fine-tuning ahead on %.1f%% of gauges, median %+.4f",
            f"rescale ({form})", column.median(), gain, 100 * share,
            100 * (head > 0).mean(), head.median(),
        )
        summary[f"gain_rescale_{form}"] = gain
        summary[f"share_of_M1_gain_{form}"] = share
        summary[f"fine_tuning_ahead_frac_{form}"] = float((head > 0).mean())

    all_folds.to_csv(args.out_dir / "rescale_control.csv", index=False)
    (args.out_dir / "rescale_control.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out_dir / "rescale_control.csv")


if __name__ == "__main__":
    main()
