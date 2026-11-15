"""Which snapshot a path belongs to, and a refusal when two of them are mixed.

The June and December experiments are meant to be independent. Every part of that is
enforced in a config file (checked by a script of the original development repository), with one
exception: the pretrained checkpoint. It arrives as a command-line path or an
environment variable, so no config check can see it, and mixing it is silent. A
December model started from June weights does not fail. It trains, it scores, it
writes a results table, and nothing in that table says the snapshots were crossed.

The guard therefore lives at the point every route converges on: the training entry
points, which know both the checkpoint and where results will be written. A guard in
one SLURM script is not enough, because three routes go round it -- the Africa
transfer script passes PRETRAINED rather than PRETRAIN_ROOT, the older transfer script
has no such variable at all, and python -m train.transfer_target --pretrained bypasses
SLURM entirely.
"""

from __future__ import annotations

import os
from pathlib import Path

# Markers that name the December track. Everything else is June, so a new December
# location has to be added here deliberately rather than being inferred.
DECEMBER_MARKERS = ("outputs/dec_", "/dec_track/", "hourly_cache_dec", "configs/dec/")
ALLOW_ENV = "ALLOW_CROSS_TRACK"


def track_of(path: str | os.PathLike) -> str:
    """``"december"`` if the path lies in the December track, else ``"june"``."""
    # Always resolve: a relative path such as "configs/dec/x.yaml" would otherwise miss
    # a marker written with a leading separator, and a path with ".." would classify by
    # its spelling rather than by where it lands.
    text = str(Path(path).expanduser().resolve())
    return "december" if any(m in text for m in DECEMBER_MARKERS) else "june"


def assert_same_track(checkpoint, out_dir, logger=None, **inputs) -> str:
    """Refuse to run one snapshot's data or weights under the other's output root.

    ``checkpoint`` and ``out_dir`` are the original pair. ``inputs`` takes any further
    named paths the run reads -- the cache, the dataset root, the fold table -- because
    weights are not the only thing that can cross: a June config pointed at the December
    cache would train on one snapshot and file the result under the other, and comparing
    only the checkpoint would not notice.

    Returns the shared track name. Set ALLOW_CROSS_TRACK=1 to proceed anyway, which is a
    real experiment (fine-tuning June weights on December data) but has to be an explicit
    choice rather than a forgotten environment variable.
    """
    out_track = track_of(out_dir)
    # ``checkpoint`` may be None: a from-scratch run loads no weights, so there is no
    # checkpoint track to cross. Every other named path is still checked, because the
    # cache and the fold table can cross whether or not weights are involved.
    named = {} if checkpoint is None else {"the pretrained weights": checkpoint}
    named.update({label.replace("_", " "): value
                  for label, value in inputs.items() if value is not None})

    crossed = {label: (value, track_of(value)) for label, value in named.items()
               if track_of(value) != out_track}
    if not crossed:
        if logger:
            logger.info("track check: results and %d input path(s) are all %s",
                        len(named), out_track)
        return out_track

    lines = [f"REFUSING: results would be written to a {out_track} location",
             f"    {out_dir}", "  but these belong to another snapshot:"]
    for label, (value, track) in sorted(crossed.items()):
        lines.append(f"    {label} is {track}: {value}")
    lines.append("  Crossing the two snapshots does not fail, it silently mixes them.")
    lines.append(f"  Point them at the {out_track} track, or set {ALLOW_ENV}=1 to mean it.")
    message = "\n".join(lines)

    if os.environ.get(ALLOW_ENV) != "1":
        raise SystemExit(message)
    if logger:
        logger.warning("%s=1 given; %s under a %s output root", ALLOW_ENV,
                       ", ".join(f"{k} is {t}" for k, (_, t) in sorted(crossed.items())),
                       out_track)
    return out_track
