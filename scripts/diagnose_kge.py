"""Why does daily-only fine-tuning cost hourly KGE? Split KGE into r / alpha / beta.

Every Phase I run so far reports the same shape: the target-domain hourly median
KGE drops a little from M0 (zero-shot) to M1 (after daily-only fine-tuning).
Baseline -0.030, run A -0.021. A KGE delta on its own does not say WHICH of the
three terms moved, and the two candidate explanations predict different things:

  smoothing   Daily aggregates carry no sub-daily information, so fine-tuning on
              them pulls predictions toward the daily mean. Peaks flatten:
              **alpha falls, r roughly holds.** That is an inherent cost of the
              supervision signal, not a bug -- Phase I's answer would be a clean
              negative result with a mechanism behind it.

  damage      The model actually forgets the hourly dynamics: **r falls too.**
              Then something is still fixable, and the next suspect is
              agg_loss_weight=0.5 -- the term restraining the hourly branch is
              weighted half of the term reshaping the daily one.

No retraining: both checkpoints are on disk, this only runs them forward. Pairs
per station so the comparison is not contaminated by which stations each pass
happened to score.

    python -m scripts.diagnose_kge --config configs/phase1.yaml --folds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.config import add_common_args, load_config, resolve
from common.utils import get_device, setup_logging
from data.dataset import load_dataset_config, load_scalers, make_loader, resolve_static_spec
from data.folds import domain_stations, load_folds
from data.sources import build_eval_set
from eval.evaluate import evaluate_model
from models.mtslstm import build_model

COMPONENTS = ["kge", "kge_r", "kge_alpha", "kge_beta", "nse"]


def _score(cfg, checkpoint: Path, loader, device, dyn_size, n_static, scalers, min_samples, logger):
    model = build_model(cfg, dyn_input_size=dyn_size, static_input_size=n_static).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    frames = evaluate_model(
        model, loader, device,
        y_mean=float(scalers["y_mean"]), y_std=float(scalers["y_std"]),
        min_samples=min_samples, logger=logger,
    )
    return frames["hourly"]


def diagnose_fold(cfg, fold: int, domain: str, run_dir: Path, device, logger) -> pd.DataFrame:
    """Per-station M0 vs M1 components for one fold, inner-joined on station."""
    m0 = run_dir / f"fold{fold}" / "pretrain" / "best_model.pth"
    m1 = run_dir / f"fold{fold}" / "transfer" / "best_transfer_model.pth"
    for path in (m0, m1):
        if not path.exists():
            raise FileNotFoundError(f"fold {fold}: {path} missing -- has this run finished?")

    folds = load_folds(resolve(cfg.folds.file))
    source_stations, target_stations = domain_stations(folds, fold)
    stations = target_stations if domain == "target" else source_stations

    scalers = load_scalers(cfg.data.root)
    dyn_size = len(load_dataset_config(cfg.data.root)["dyn_features"])
    _, _, static_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static")
    )

    # The same untouched test split the transfer step reported M0/M1 on.
    dataset = build_eval_set(cfg, stations, "validation", logger=logger)
    num_workers = int(cfg.get_path("transfer.num_workers", 4))
    pin_memory = bool(cfg.get_path("train.pin_memory", False)) and device.type == "cuda"
    loader = make_loader(dataset, num_workers=num_workers, pin_memory=pin_memory)
    min_samples = int(cfg.get_path("eval.min_samples_per_station", 1))

    scored = {}
    for tag, checkpoint in (("M0", m0), ("M1", m1)):
        logger.info("fold %d %s: scoring %s on %d %s stations", fold, tag, checkpoint.name,
                    len(stations), domain)
        frame = _score(cfg, checkpoint, loader, device, dyn_size, len(static_names),
                       scalers, min_samples, logger)
        scored[tag] = frame.loc[frame["score_status"].eq("ok")].set_index("station_id")

    # Inner join: a station only counts if BOTH passes scored it, otherwise the
    # delta would compare different station populations.
    common = scored["M0"].index.intersection(scored["M1"].index)
    logger.info("fold %d: %d stations scored by both passes (M0 %d, M1 %d)",
                fold, len(common), len(scored["M0"]), len(scored["M1"]))

    out = pd.DataFrame({"station_id": common, "fold": fold}).set_index("station_id")
    out["source"] = scored["M0"].loc[common, "source"]
    out["samples"] = scored["M0"].loc[common, "samples"]
    out["obs_std"] = scored["M0"].loc[common, "obs_std"]
    for tag in ("M0", "M1"):
        for col in COMPONENTS + ["sim_std"]:
            out[f"{tag}_{col}"] = scored[tag].loc[common, col]
    return out.reset_index()


def summarize_components(table: pd.DataFrame) -> pd.DataFrame:
    """Paired medians and a Wilcoxon test per component."""
    from scipy.stats import wilcoxon

    rows = []
    for col in COMPONENTS + ["sim_std"]:
        m0, m1 = table[f"M0_{col}"].to_numpy(), table[f"M1_{col}"].to_numpy()
        keep = np.isfinite(m0) & np.isfinite(m1)
        m0, m1 = m0[keep], m1[keep]
        delta = m1 - m0
        try:
            p = float(wilcoxon(m1, m0).pvalue) if delta.any() else 1.0
        except ValueError:
            p = float("nan")
        rows.append({
            "component": col,
            "n_stations": int(keep.sum()),
            "M0_median": float(np.median(m0)),
            "M1_median": float(np.median(m1)),
            "median_delta": float(np.median(delta)),
            "mean_delta": float(np.mean(delta)),
            "frac_worse": float((delta < 0).mean()),
            "wilcoxon_p": p,
        })
    return pd.DataFrame(rows)


def _kge_from(r, a, b):
    return 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)


def attribute(table: pd.DataFrame, min_obs_std: float = 1e-3) -> dict:
    """Which term actually caused each station's KGE to drop.

    Medians of the three components move so little (~0.006 each) that comparing
    them against a tolerance yields no answer. The question is per-station and
    counterfactual: move ONE term to its M1 value, leave the other two at M0, and
    see what that alone does to KGE. Whichever hurts most is that station's culprit.

    ``min_obs_std`` drops numerically degenerate stations. alpha and beta divide by
    std(obs) and mean(obs), so a near-constant record (the worst here has obs_std
    5e-7 mm/h) sends alpha into the thousands and makes the MEAN delta meaningless
    (-30 KGE across the target domain). Medians survive it; means do not. Only the
    exact-zero case is excluded upstream, so it is filtered here and reported.
    """
    kept = table.loc[table["obs_std"] >= min_obs_std]
    cols = ("r", "alpha", "beta")
    m0 = [kept[f"M0_kge_{c}"].to_numpy() for c in cols]
    m1 = [kept[f"M1_kge_{c}"].to_numpy() for c in cols]

    base = _kge_from(*m0)
    # Row i: KGE change from moving only term i.
    single = np.vstack([
        _kge_from(*(m1[j] if j == i else m0[j] for j in range(3))) - base
        for i in range(3)
    ])
    delta = _kge_from(*m1) - base
    worse = delta < 0
    culprit = np.argmin(single[:, worse], axis=0)

    labels = ["r (timing)", "alpha (variance)", "beta (bias)"]
    return {
        "n_stations": int(len(kept)),
        "n_dropped_degenerate": int(len(table) - len(kept)),
        "min_obs_std": min_obs_std,
        "frac_stations_worse": float(worse.mean()),
        "culprit_share": {
            labels[i]: float((culprit == i).mean()) for i in range(3)
        },
        "solo_effect_median": {
            labels[i]: float(np.median(single[i])) for i in range(3)
        },
        # Share of the KGE deficit each term carries, before and after -- says what
        # limits hourly skill overall, not just what the transfer step changed.
        "deficit_share": {
            tag: {
                labels[i]: float(np.median((vals[i] - 1) ** 2) / sum(
                    np.median((vals[j] - 1) ** 2) for j in range(3)))
                for i in range(3)
            }
            for tag, vals in (("M0", m0), ("M1", m1))
        },
        "alpha_median": {"M0": float(np.median(m0[1])), "M1": float(np.median(m1[1]))},
        "frac_under_dispersed": {"M0": float((m0[1] < 1).mean()), "M1": float((m1[1] < 1).mean())},
    }


def verdict(summary: pd.DataFrame, attribution: dict) -> str:
    """State which explanation the numbers support, and what actually limits skill."""
    get = lambda c, f: float(summary.loc[summary["component"].eq(c), f].iloc[0])  # noqa: E731
    d_kge, d_r = get("kge", "median_delta"), get("kge_r", "median_delta")
    share = attribution["culprit_share"]
    r_share = share["r (timing)"]
    a0 = attribution["alpha_median"]["M0"]
    under = attribution["frac_under_dispersed"]["M0"]
    a_deficit = attribution["deficit_share"]["M0"]["alpha (variance)"]

    if r_share > 0.4:
        head = (f"timing is the leading culprit in {r_share:.1%} of the stations that got worse "
                f"(r {d_r:+.4f}) -- the model genuinely lost hourly dynamics, so something is "
                f"still fixable; next suspect is agg_loss_weight.")
    else:
        head = (f"KGE fell {d_kge:+.4f}, but timing leads in only {r_share:.1%} of the stations "
                f"that got worse ({share['alpha (variance)']:.1%} variance, "
                f"{share['beta (bias)']:.1%} bias; r itself moved {d_r:+.4f}). Daily means carry "
                f"volume but no sub-daily information, so fine-tuning rescales the series rather "
                f"than forgetting its dynamics -- the inherent cost of daily-only supervision.")

    return (f"{head} Separately and larger: at M0 already, {under:.1%} of stations are "
            f"under-dispersed (median alpha {a0:.3f}) and variance carries {a_deficit:.1%} of the "
            f"KGE deficit. That ceiling predates the transfer step and is the bigger target.")


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Decompose the M0->M1 KGE change."))
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--domain", default="target", choices=["target", "source"])
    parser.add_argument("--run-dir", default=None, help="Defaults to the config's output_root.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Skip scoring; combine the per-fold CSVs an array job already wrote.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    run_dir = Path(args.run_dir) if args.run_dir else resolve(cfg.output_root)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / f"diagnose_kge_{args.domain}.log")

    folds = [int(f) for f in args.folds.split(",") if f.strip() != ""]
    tables = []
    if args.merge:
        # Folds are scored in parallel by an array job; this pass only aggregates.
        for path in sorted(out_dir.glob(f"fold*/kge_components_{args.domain}.csv")):
            tables.append(pd.read_csv(path))
            logger.info("merged %s (%d stations)", path, len(tables[-1]))
        if not tables:
            raise SystemExit(f"no per-fold CSVs under {out_dir}/fold*/ -- did the array job run?")
    else:
        device = get_device()
        logger.info("run %s | domain %s | device %s", run_dir, args.domain, device)
        for fold in folds:
            try:
                tables.append(diagnose_fold(cfg, fold, args.domain, run_dir, device, logger))
            except FileNotFoundError as exc:
                logger.warning("skipping fold %d: %s", fold, exc)
        if not tables:
            raise SystemExit("no fold had both checkpoints -- nothing to diagnose")

    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out_dir / f"kge_components_{args.domain}.csv", index=False)

    summary = summarize_components(table)
    summary.to_csv(out_dir / f"kge_components_summary_{args.domain}.csv", index=False)
    attribution = attribute(table)
    text = verdict(summary, attribution)

    per_fold = pd.concat(
        [summarize_components(t).assign(fold=t["fold"].iloc[0]) for t in tables], ignore_index=True
    )
    per_fold.to_csv(out_dir / f"kge_components_by_fold_{args.domain}.csv", index=False)

    (out_dir / f"verdict_{args.domain}.json").write_text(json.dumps({
        "run_dir": str(run_dir), "domain": args.domain, "folds": folds,
        "n_stations": int(len(table)), "verdict": text,
        "summary": summary.to_dict(orient="records"),
        "attribution": attribution,
    }, indent=2))

    logger.info("\n%s", summary.to_string(index=False, float_format=lambda v: f"{v: .5f}"))
    logger.info("attribution: %s", json.dumps(attribution, indent=2))
    logger.info("VERDICT: %s", text)


if __name__ == "__main__":
    main()
