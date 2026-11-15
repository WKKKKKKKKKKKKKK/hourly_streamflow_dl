"""Score an existing transfer checkpoint on the target fold at the DAILY scale.

The target-domain evaluation in `train/transfer_target.py` writes hourly metrics only,
because hourly skill is what the study is about. Comparing M1 against an ordinary daily
LSTM needs a scale both models can be scored on, and a daily LSTM emits one value per
day, so the comparison has to happen at the daily scale -- exactly as it does for Africa,
where daily is all that is observed.

Nothing is retrained here. The 24 `best_transfer_model.pth` checkpoints already exist for
the five designs, and this reloads each one, rebuilds the same target validation set with
daily targets attached, and aggregates the hourly head over 24 hours the same way the
transfer loss does. `train/transfer_target.py` is not touched, so every hourly number
already reported stays bit-identical.

    python -m scripts.analysis.score_daily_target --config configs/phase1_runB_v2.yaml --fold 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common.config import add_common_args, load_config, resolve
from common.alignment import assert_same_alignment
from common.tracks import assert_same_track
from common.utils import get_device, setup_logging
from data.folds import domain_stations, load_folds
from data.sources import build_eval_set
from data.dataset import load_dataset_config, load_scalers, make_loader, resolve_static_spec
from eval.evaluate import evaluate_model, write_results
from models.mtslstm import build_model

DAILY_WINDOW = 24
DAILY_ROOT = Path("outputs/v2_design_daily")


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Defaults to <output_root>/fold{N}/transfer/best_transfer_model.pth")
    parser.add_argument("--tag", default="M1",
                        help="Name for the output files; use daily_lstm for the baseline arm")
    parser.add_argument("--out-dir", default=None)
    # Must match how the checkpoint was trained. An ordinary daily LSTM has no supervised
    # hourly branch, so its 24-hour mean is noise; its daily score has to read its own head.
    # The source domain has no daily scores anywhere else in the study: Step 3 evaluates
    # it hourly only. Scoring it daily is what makes the source and target maps
    # comparable at both timescales. A source pass is ~4x the work of a target pass.
    parser.add_argument("--domain", default="target", choices=("target", "source"),
                        help="which side of the fold to score (default: target)")
    parser.add_argument("--score-branch", default="agg", choices=("agg", "daily"),
                        help="head the daily metric reads (default: hourly aggregate)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    fold_dir = resolve(cfg.output_root) / f"fold{args.fold}" / "transfer"
    # Results go to a new root, never into the fold directory. The June outputs are
    # deliberately read-only, and this is new analysis rather than part of the run that
    # produced the checkpoint -- writing it alongside would both fail and blur what was
    # frozen. The checkpoint is only read.
    out_dir = Path(args.out_dir) if args.out_dir else (
        DAILY_ROOT / Path(str(cfg.output_root)).name / f"fold{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "score_daily.log")

    ckpt = Path(args.checkpoint) if args.checkpoint else fold_dir / "best_transfer_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    # --checkpoint is a free path, so this is exactly the route the track guard exists
    # for: scoring December weights and filing the table under a June output root would
    # not fail, it would silently mix the snapshots.
    assert_same_track(ckpt, out_dir, logger=logger,
                      cache=cfg.data.get("cache_dir"), dataset_root=cfg.data.root,
                      static_table=cfg.data.get("static_csv"),
                      fold_table=resolve(cfg.folds.file))
    assert_same_alignment(ckpt, cfg, logger=logger)

    device = get_device()
    folds = load_folds(resolve(cfg.folds.file))
    source_stations, target_stations = domain_stations(folds, args.fold)
    stations = source_stations if args.domain == "source" else target_stations
    logger.info("fold %d | domain %s | %d stations | %s",
                args.fold, args.domain, len(stations), ckpt)

    scalers = load_scalers(cfg.data.root)
    ds_config = load_dataset_config(cfg.data.root)
    _, _, static_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static"))

    # with_daily=True is the only difference from the hourly evaluation set: it attaches
    # the observed 24-h means the aggregated prediction is scored against.
    test_ds = build_eval_set(cfg, stations, "validation", with_daily=True, logger=logger)
    loader = make_loader(test_ds, num_workers=int(cfg.transfer.num_workers),
                         pin_memory=bool(cfg.get_path("train.pin_memory", False))
                         and device.type == "cuda")

    model = build_model(cfg, dyn_input_size=len(ds_config["dyn_features"]),
                        static_input_size=len(static_names)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    result = evaluate_model(model, loader, device, scalers["y_mean"], scalers["y_std"],
                            int(cfg.get_path("eval.min_samples_per_station", 1)),
                            daily_window=DAILY_WINDOW,
                            daily_branch=(args.score_branch == "daily"), logger=logger)
    summary = write_results(result, out_dir, tag=f"fold{args.fold}_{args.tag}_{args.domain}")
    for scale in ("hourly", "daily"):
        if scale in summary:
            logger.info("%s %s: median KGE %.4f | median NSE %.4f | %d stations",
                        args.tag, scale, summary[scale]["median_kge"],
                        summary[scale]["median_nse"], summary[scale]["n_stations"])


if __name__ == "__main__":
    main()
