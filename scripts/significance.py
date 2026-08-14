"""Per-station paired significance for M1 vs M0, with FDR control (PLAN.md 5).

A single Wilcoxon test over per-station KGE says "the median station improved". It
does not say how many individual stations improved for real, and asking that question
of ~8,900 stations means ~8,900 tests: at alpha = 0.05 about 445 stations would look
significant by chance alone. So each station gets its own paired test on its own
sample-level errors, and the p-values then go through Benjamini-Hochberg.

Per station the test is a Wilcoxon signed-rank on |error| under M1 versus |error|
under M0, over the same evaluation samples in the same order -- paired by sample, not
merely by station, which is the strongest pairing available here.

Also reported, because significance and size are different questions: the median
error reduction per station, and the pooled test over per-station KGE for continuity
with the numbers quoted elsewhere.

    python -m scripts.significance --config configs/phase1_runB.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.config import add_common_args, load_config, resolve
from common.metrics import compute_kge
from common.utils import get_device, setup_logging
from data.dataset import load_dataset_config, load_scalers, make_loader, resolve_static_spec
from data.folds import domain_stations, load_folds
from data.sources import build_eval_set
from models.mtslstm import build_model

MIN_SAMPLES = 100


def benjamini_hochberg(p: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """(rejected, q-values). Step-up procedure controlling the false discovery rate."""
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downward, the standard correction.
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out_q = np.empty(n, dtype=np.float64)
    out_q[order] = q
    return out_q <= alpha, out_q


@torch.no_grad()
def predictions_by_station(model, loader, device, y_mean: float, y_std: float, logger=None):
    """station -> (sample keys, predictions, observations), all in physical units."""
    model.eval()
    keys: dict[str, list] = {}
    sim: dict[str, list] = {}
    obs: dict[str, list] = {}
    for batch in loader:
        stations = batch["stations"]
        if not stations:
            continue
        x = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
        pred = model({"D": x["D"], "H": x["H"]}, x["S"])["H"].float().cpu().numpy()
        truth = batch["y"].numpy()
        hours = batch["hours"].numpy() if batch.get("hours") is not None else np.arange(len(truth))
        for station, hour, s, o in zip(np.asarray(stations, dtype=object), hours, pred, truth):
            name = str(station)
            keys.setdefault(name, []).append(int(hour))
            sim.setdefault(name, []).append(s)
            obs.setdefault(name, []).append(o)
    out = {}
    for name in sim:
        k = np.asarray(keys[name])
        order = np.argsort(k, kind="stable")
        out[name] = (
            k[order],
            np.asarray(sim[name], dtype=np.float64)[order] * y_std + y_mean,
            np.asarray(obs[name], dtype=np.float64)[order] * y_std + y_mean,
        )
    if logger:
        logger.info("predictions for %d stations", len(out))
    return out


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Per-station paired tests with BH-FDR."))
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--pretrain-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    run_dir = Path(args.run_dir) if args.run_dir else resolve(cfg.output_root)
    pre_dir = Path(args.pretrain_dir) if args.pretrain_dir else run_dir
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "significance"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "significance.log")
    device = get_device()

    scalers = load_scalers(cfg.data.root)
    y_mean, y_std = float(scalers["y_mean"]), float(scalers["y_std"])
    dyn_size = len(load_dataset_config(cfg.data.root)["dyn_features"])
    _, _, static_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static")
    )
    folds = load_folds(resolve(cfg.folds.file))

    from scipy.stats import wilcoxon

    rows = []
    for fold in [int(f) for f in args.folds.split(",") if f.strip()]:
        m0 = pre_dir / f"fold{fold}" / "pretrain" / "best_model.pth"
        m1 = run_dir / f"fold{fold}" / "transfer" / "best_transfer_model.pth"
        if not (m0.exists() and m1.exists()):
            logger.warning("fold %d: checkpoint missing -- skipped", fold)
            continue
        _, target_stations = domain_stations(folds, fold)
        dataset = build_eval_set(cfg, target_stations, "validation", logger=logger)
        loader = make_loader(dataset, num_workers=int(cfg.get_path("transfer.num_workers", 4)),
                            pin_memory=device.type == "cuda")

        got = {}
        for tag, path in (("M0", m0), ("M1", m1)):
            model = build_model(cfg, dyn_input_size=dyn_size, static_input_size=len(static_names)).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            logger.info("fold %d %s: predicting", fold, tag)
            got[tag] = predictions_by_station(model, loader, device, y_mean, y_std, logger)

        for station in sorted(set(got["M0"]) & set(got["M1"])):
            k0, s0, o0 = got["M0"][station]
            k1, s1, o1 = got["M1"][station]
            # Pair by sample key, so the test compares the two models on identical rows.
            shared = np.intersect1d(k0, k1)
            if shared.size < MIN_SAMPLES:
                continue
            i0 = np.searchsorted(k0, shared)
            i1 = np.searchsorted(k1, shared)
            obs = o0[i0]
            if not np.allclose(obs, o1[i1], equal_nan=True):
                raise SystemExit(f"{station}: the two passes disagree about the observations")
            e0 = np.abs(s0[i0] - obs)
            e1 = np.abs(s1[i1] - obs)
            delta = e0 - e1                      # positive = M1 has the smaller error
            if not np.any(delta):
                continue
            try:
                p = float(wilcoxon(e1, e0).pvalue)
            except ValueError:
                continue
            rows.append({
                "station_id": station, "fold": fold, "n_samples": int(shared.size),
                "p_value": p,
                "median_error_reduction": float(np.median(delta)),
                "mean_abs_error_M0": float(np.mean(e0)),
                "mean_abs_error_M1": float(np.mean(e1)),
                "kge_M0": compute_kge(obs, s0[i0]),
                "kge_M1": compute_kge(obs, s1[i1]),
            })

    if not rows:
        raise SystemExit("no station produced a usable paired test")
    table = pd.DataFrame(rows)
    rejected, q = benjamini_hochberg(table["p_value"].to_numpy(), args.alpha)
    table["q_value"] = q
    table["significant"] = rejected
    table["direction"] = np.where(table["median_error_reduction"] > 0, "improved", "degraded")
    table.to_csv(out_dir / "per_station_tests.csv", index=False)

    n = len(table)
    sig = table.loc[table["significant"]]
    improved = sig.loc[sig["direction"].eq("improved")]
    degraded = sig.loc[sig["direction"].eq("degraded")]
    raw_sig = int((table["p_value"] <= args.alpha).sum())

    logger.info("%d stations tested | alpha = %.2f", n, args.alpha)
    logger.info("uncorrected p <= alpha: %d stations (%.1f%%) -- of which ~%.0f expected by chance",
                raw_sig, 100 * raw_sig / n, args.alpha * n)
    logger.info("after Benjamini-Hochberg: %d significant (%.1f%%) -> %d improved (%.1f%%), "
                "%d degraded (%.1f%%)",
                len(sig), 100 * len(sig) / n, len(improved), 100 * len(improved) / n,
                len(degraded), 100 * len(degraded) / n)
    logger.info("median error reduction: all %+.5f mm/h | significant-improved %+.5f | "
                "significant-degraded %+.5f",
                table["median_error_reduction"].median(),
                improved["median_error_reduction"].median() if len(improved) else float("nan"),
                degraded["median_error_reduction"].median() if len(degraded) else float("nan"))

    # Pooled test on per-station KGE, for continuity with the headline numbers.
    pooled = wilcoxon(table["kge_M1"], table["kge_M0"])
    logger.info("pooled Wilcoxon on per-station KGE: median ΔKGE %+.4f, p = %.3e",
                (table["kge_M1"] - table["kge_M0"]).median(), float(pooled.pvalue))

    (out_dir / "significance_summary.json").write_text(json.dumps({
        "n_stations": n, "alpha": args.alpha,
        "n_uncorrected_significant": raw_sig,
        "n_expected_by_chance": args.alpha * n,
        "n_significant_after_bh": int(len(sig)),
        "n_improved": int(len(improved)), "n_degraded": int(len(degraded)),
        "median_error_reduction": float(table["median_error_reduction"].median()),
        "pooled_median_delta_kge": float((table["kge_M1"] - table["kge_M0"]).median()),
        "pooled_wilcoxon_p": float(pooled.pvalue),
    }, indent=2))


if __name__ == "__main__":
    main()
