"""sMTS-LSTM during the daily-only fine-tune: what carries information, what is frozen.

A simplified redraw of Gauch et al.'s MTS-LSTM schematic, reduced to the parts the
adaptation step turns on, and annotated with what this study actually freezes. The
numbers are read from the model itself rather than typed, so the figure cannot disagree
with the code.

The point of the figure is the asymmetry. Only the hourly LSTM is frozen, because its
weights encode the sub-daily dynamics learned from the source domain and a daily target
carries no information that could correct them. Everything the daily signal can
legitimately inform stays trainable: the daily LSTM, the two state-transfer layers, and
BOTH output heads -- including the hourly head, which is a per-timestep rescaling and is
exactly what a 24-hour total constrains.

    python -m scripts.figures.fig_architecture

Suggested caption:
    sMTS-LSTM during Step 2. The daily branch runs the full 365-day lookback; at day 351
    its hidden and cell states pass through two linear layers to initialise the hourly
    branch, which runs the final 336 hours. During the daily-only adaptation the hourly
    LSTM is frozen (hatched) and all other modules are trained. Both output heads are
    supervised by the same daily observation: the daily head directly, and the hourly
    head through the 24-hour mean of its output. No hourly observation enters the loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Liberation Sans", "Arial",
                        "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold",
    "mathtext.default": "it",
})

OUT = Path("reports/figures/fig26_architecture.png")
LINE, TEXT, MUTED = "#222222", "#111111", "#555555"
TRAIN_FILL, FROZEN_FILL = "#ffffff", "#c9d2da"
BASE = 8.6


def cell(ax, x, y, w, h, frozen=False, lw=0.9, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=FROZEN_FILL if frozen else TRAIN_FILL,
                                edgecolor=LINE, linewidth=lw,
                                hatch="//////" if frozen else None, zorder=3))


def plain(ax, x, y, w, h, frozen=False, lw=0.9):
    ax.add_patch(Rectangle((x, y), w, h,
                           facecolor=FROZEN_FILL if frozen else TRAIN_FILL,
                           edgecolor=LINE, linewidth=lw,
                           hatch="//////" if frozen else None, zorder=3))


def arrow(ax, p0, p1, lw=0.85, z=4, ls="-", colour=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6.5,
                                 linewidth=lw, color=colour or LINE, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=z))


def elbow(ax, pts, lw=0.85, z=4, colour=None):
    """Right-angled route with a single arrowhead on the last leg.

    Orthogonal edges read better here than straight lines: the branches sit on two
    horizontal bands, so a diagonal crosses the band boundary at an arbitrary angle and
    the eye has to work out which box it leaves and which it enters.
    """
    ax.plot([q[0] for q in pts[:-1]], [q[1] for q in pts[:-1]],
            color=colour or LINE, linewidth=lw, solid_capstyle="butt",
            solid_joinstyle="miter", zorder=z)
    arrow(ax, pts[-2], pts[-1], lw=lw, z=z, colour=colour)


def draw(ax, counts):
    cw, chh = 0.066, 0.092                       # LSTM cell size
    dy, hy = 0.235, 0.580                        # daily row, hourly row
    ROSE = "#8a5a2b"                             # the one accent, for the loss paths

    # --- daily branch, before the hand-off ---------------------------------
    pre = [0.032, 0.132]
    for x in pre:
        cell(ax, x, dy, cw, chh)
        arrow(ax, (x + cw / 2, dy - 0.070), (x + cw / 2, dy))
    ax.text(pre[0] + cw / 2, dy + chh + 0.016, "LSTM$^D$", fontsize=BASE - 0.5,
            ha="center", va="bottom")
    for x, lab in zip(pre, ("$x^D_1$", "$x^D_{351}$")):
        ax.text(x + cw / 2, dy - 0.092, lab, fontsize=BASE - 0.3, ha="center", va="top")
    ax.text((pre[0] + cw + pre[1]) / 2, dy + chh / 2, "…", fontsize=BASE + 1,
            ha="center", va="center")
    arrow(ax, (pre[0] + cw, dy + chh / 2), (pre[1], dy + chh / 2))

    # --- state transfer: two parallel routes, so the lines do not cross -----
    tx, tw = 0.238, 0.056
    post = [0.352, 0.500, 0.648]
    # The route with the LOWER endpoints takes the OUTER vertical. With the inner/outer
    # order the other way round, the upper route's horizontal leg crosses the lower
    # route's vertical leg; this way the two are nested and never meet.
    for lab, yy, frac, vin, vout in (("FC$_h$", 0.435, 0.70, 0.208, 0.310),
                                     ("FC$_c$", 0.333, 0.30, 0.230, 0.334)):
        plain(ax, tx, yy, tw, 0.056)
        ax.text(tx + tw / 2, yy + 0.028, lab, fontsize=BASE - 0.5, ha="center",
                va="center", zorder=4)
        y_from, y_to = dy + chh * frac, yy + 0.028
        elbow(ax, [(pre[1] + cw, y_from), (vin, y_from), (vin, y_to), (tx, y_to)])
        y_in = hy + chh * frac
        elbow(ax, [(tx + tw, y_to), (vout, y_to), (vout, y_in), (post[0], y_in)])
    ax.text(tx + tw / 2, 0.503, "state transfer", fontsize=BASE - 1.1, ha="center",
            color=MUTED)

    # --- both branches over the last 14 days -------------------------------
    for row, frozen, labels in ((dy, False, ("$x^D_{352}$", "$x^D_{364}$", "$x^D_{365}$")),
                                (hy, True, ("$x^H_{352,1}$", "$x^H_{364,24}$",
                                            "$x^H_{365,24}$"))):
        for k, x in enumerate(post):
            cell(ax, x, row, cw, chh, frozen=frozen)
            arrow(ax, (x + cw / 2, row - 0.070), (x + cw / 2, row))
            if k:
                ax.text((post[k - 1] + cw + x) / 2, row + chh / 2, "…",
                        fontsize=BASE + 1, ha="center", va="center")
                arrow(ax, (post[k - 1] + cw, row + chh / 2), (x, row + chh / 2))
            ax.text(x + cw / 2, row - 0.092, labels[k], fontsize=BASE - 0.3,
                    ha="center", va="top")
    arrow(ax, (pre[1] + cw, dy + chh / 2), (post[0], dy + chh / 2))
    ax.text(post[0] + cw / 2, hy + chh + 0.016, "LSTM$^H$", fontsize=BASE - 0.5,
            ha="center", va="bottom")

    # --- heads --------------------------------------------------------------
    hx, hw, hh = 0.748, 0.056, 0.072
    for row, lab, out in ((hy, "FC$^H$", "\u0177$^H$"), (dy, "FC$^D$", "\u0177$^D$")):
        plain(ax, hx, row + 0.010, hw, hh)
        ax.text(hx + hw / 2, row + 0.046, lab, fontsize=BASE - 0.5, ha="center",
                va="center", zorder=4)
        arrow(ax, (post[-1] + cw, row + chh / 2), (hx, row + 0.046))
        ax.text(hx + hw / 2, row + (0.098 if row == hy else -0.048), out,
                fontsize=BASE + 0.4, ha="center",
                va="bottom" if row == hy else "top")

    # --- the two loss terms, and the single target they share ---------------
    lx, lw_, lh = 0.862, 0.116, 0.130
    ax.add_patch(FancyBboxPatch((lx, 0.330), lw_, lh,
                                boxstyle="round,pad=0,rounding_size=0.010",
                                facecolor="white", edgecolor=ROSE, linewidth=1.0,
                                zorder=3))
    ax.text(lx + lw_ / 2, 0.330 + lh / 2, "observed\ndaily $y^D$", fontsize=BASE - 0.5,
            ha="center", va="center", color=ROSE, zorder=4, linespacing=1.5)
    cx = lx + lw_ / 2
    elbow(ax, [(cx, 0.330 + lh), (cx, hy + 0.046), (hx + hw, hy + 0.046)],
          colour=ROSE, lw=1.0)
    elbow(ax, [(cx, 0.330), (cx, dy + 0.046), (hx + hw, dy + 0.046)],
          colour=ROSE, lw=1.0)

    # The weights live in the loss statement top-left; on the arrows they collided with
    # the lines. What the arrows still have to say is that one path is aggregated first.
    # On the vertical leg, clear of the head box: the horizontal run passes over FC^H.
    # Left of the trunk: to its right the label runs off the page edge.
    ax.text(cx - 0.014, (hy + 0.046 + 0.330 + lh) / 2, "via the\n24-h mean",
            fontsize=BASE - 1.6, ha="right", va="center", color=ROSE, linespacing=1.5)


def legend(ax, counts):
    """Top-left, which the two branches leave empty; the bottom strip is for the inputs."""
    rows = [
        (False, "trained in Step 2 ({:,} parameters): the daily LSTM, both state-"
                "transfer\nlayers, and both output heads".format(counts["trainable"])),
        (True, "frozen in Step 2 ({:,} parameters): the hourly LSTM, whose weights hold\n"
               "the sub-daily dynamics a daily target cannot inform".format(counts["frozen"])),
    ]
    for i, (frozen, label) in enumerate(rows):
        y = 0.935 - i * 0.082
        plain(ax, 0.045, y - 0.030, 0.026, 0.032, frozen=frozen)
        ax.text(0.082, y - 0.014, label, fontsize=BASE - 0.9, va="center",
                linespacing=1.55)

    # NSE and the subscripts stay outside math mode: \mathrm sends mathtext to its
    # Computer Modern fallback, which puts a second font family in the PDF and breaks
    # the journal's one-family rule. Only the symbols themselves are math.
    # A pointer, not the algebra: the definition and the weights belong in the text,
    # where they can be stated once and cited, rather than ageing inside a figure.
    ax.text(0.045, 0.772,
            "Step 2 loss: both heads are supervised by the same daily observation, "
            "through the same function; no hourly observation enters.",
            fontsize=BASE - 0.7, va="top", color=TEXT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT, type=Path)
    ap.add_argument("--config", default="configs/aligned/phase1_runB_v2.yaml")
    args = ap.parse_args()

    from common.config import load_config
    from models.mtslstm import build_model, set_trainable
    cfg = load_config(args.config, None)
    # Inputs of both branches: 3 dynamic forcings (PET, precipitation, air temperature) and 25 static
    # attributes, as in every trained checkpoint (input weights of shape (512, 28)). The original
    # script passed 4 dynamic inputs (counting discharge, which is the target, not an input) and
    # printed 512 parameters too many per LSTM.
    model = build_model(cfg, dyn_input_size=3, static_input_size=25)
    trainable, frozen = set_trainable(model, list(cfg.transfer.freeze_modules))
    counts = {"trainable": trainable, "frozen": frozen}

    fig = plt.figure(figsize=(7.09, 3.45))
    ax = fig.add_axes([0.005, 0.005, 0.99, 0.99])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw(ax, counts)
    legend(ax, counts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=400)
    # Drop the PDF creation timestamp so two runs of this script are byte-identical:
    # a figure that changes checksum without changing content cannot be diffed, and a
    # reviewer asking "is this the same figure" deserves a one-command answer.
    fig.savefig(args.out.with_suffix(".pdf"), metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {args.out} and {args.out.with_suffix('.pdf')}")
    print(f"trainable {trainable:,} | frozen {frozen:,} "
          f"(freeze_modules = {list(cfg.transfer.freeze_modules)})")


if __name__ == "__main__":
    main()
