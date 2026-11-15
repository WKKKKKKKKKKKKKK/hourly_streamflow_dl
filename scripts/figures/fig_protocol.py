"""Protocol schematic: what is trained on what, and what is measured.

A print figure, not a slide: 180 mm width, thin strokes, one grey ramp
that survives greyscale printing, and no results numbers -- those belong to the results
figures, and a schematic that carries them goes stale the moment a run
is repeated.

Panel (a) is the fold structure, which is the part prose states badly: every gauge is a
target exactly once, so the reported metrics cover the whole network. Panel (b) is the
flow, and its point is the asymmetry -- hourly supervision enters only from the source,
the target contributes daily aggregates alone, and the arrow back to the source domain
is a measurement of what that costs rather than a check that passes by construction.

    python -m scripts.figures.fig_protocol

"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path("reports/figures/protocol.png")

# One sans-serif family, Arial or Helvetica. Neither is installed here;
# Nimbus Sans is URW's metrically identical Helvetica clone and Liberation Sans is the
# Arial equivalent, so either serves and the PDF is metric-correct.
# fonttype 42 embeds TrueType outlines instead of the default Type 3 fonts.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Liberation Sans", "Arial",
                        "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Without this the $M_0$/$M_1$ labels are set in matplotlib's own DejaVu, which puts
    # a second family in the file and breaks the "one font family" rule.
    "mathtext.fontset": "custom",
    "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic",
    "mathtext.bf": "Nimbus Sans:bold",
    "mathtext.cal": "Nimbus Sans:italic",
    "mathtext.default": "it",
})

# A single grey ramp plus black: prints identically in colour and in greyscale.
FILL_SOURCE, FILL_TARGET, FILL_BOX = "#e8ecf0", "#9aa7b2", "#ffffff"
LINE, TEXT, MUTED = "#222222", "#111111", "#555555"
BASE = 7.6          # pt; every other size is stated relative to this


def rect(ax, x, y, w, h, face=FILL_BOX, edge=LINE, lw=0.8, z=2, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                           linewidth=lw, zorder=z, linestyle=ls))


def arrow(ax, p0, p1, ls="-", lw=0.9, z=4, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=lw, color=LINE, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


def elbow(ax, pts, ls="-", lw=0.9, z=4):
    """A right-angled route with one arrowhead, for edges that must go round a box."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=LINE, linewidth=lw, linestyle=ls,
            solid_capstyle="butt", zorder=z)
    arrow(ax, pts[-2], pts[-1], ls=ls, lw=lw, z=z)


def bullets(ax, x_mark, x_text, y_top, items):
    """A bulleted list with a hanging indent, used by all four process boxes.

    One text line takes about 0.042 of the axes height at this size; 0.020 separates
    items. Returning nothing is deliberate -- the caller sizes the box, not the list.
    """
    y = y_top
    for item in items:
        ax.text(x_mark, y, "\u2022", fontsize=BASE - 1.2, ha="left", va="top",
                color=MUTED)
        ax.text(x_text, y, item, fontsize=BASE - 1.2, ha="left", va="top",
                color=MUTED, linespacing=1.5)
        y -= 0.042 * (item.count("\n") + 1) + 0.020


def panel_a(ax):
    """The grid's two axes are different things, and saying so is the whole job.

    Columns are the groups the network is partitioned into, once. Rows are the runs of
    the experiment, each holding out one of those groups. Five are drawn; the
    continental design has four, so the count is not stated as a property of the
    protocol. Without the column
    header a reader takes the five cells of a row for five different partitions, or the
    rows for five copies of one group -- both readings came up.
    """
    ax.text(0.0, 1.02, "(a)  Cross-validation (CV) structure", fontsize=BASE + 0.6,
            fontweight="bold", va="top")

    x0, y0, cw, ch, g = 0.115, 0.790, 0.148, 0.068, 0.012
    span = 5 * cw + 4 * 0.008
    # Column header: the groups are the SAME groups in every row.
    ax.annotate("", xy=(x0, y0 + ch + 0.048), xytext=(x0 + span, y0 + ch + 0.048),
                arrowprops=dict(arrowstyle="|-|,widthA=0.25,widthB=0.25",
                                linewidth=0.6, color=MUTED, shrinkA=0, shrinkB=0))
    ax.text(x0 + span / 2, y0 + ch + 0.060, "the same partition in every run",
            fontsize=BASE - 0.8, ha="center", va="bottom", color=MUTED)

    for row in range(5):
        y = y0 - row * (ch + g)
        ax.text(x0 - 0.016, y + ch / 2, f"run {row + 1}", fontsize=BASE - 0.7,
                ha="right", va="center", color=MUTED)
        for col in range(5):
            held = col == row
            rect(ax, x0 + col * (cw + 0.008), y, cw, ch,
                 face=FILL_TARGET if held else FILL_SOURCE, lw=0.5)

    ly = y0 - 5 * (ch + g) - 0.050
    for face, label in ((FILL_SOURCE, "source domain (80 %), hourly observations"),
                        (FILL_TARGET, "target domain (20 %), daily observations only")):
        rect(ax, 0.030, ly, 0.070, 0.046, face=face, lw=0.5)
        ax.text(0.118, ly + 0.023, label, fontsize=BASE - 0.6, va="center")
        ly -= 0.072


def panel_b(ax):
    ax.text(0.0, 1.02, "(b)  Transfer protocol, one run", fontsize=BASE + 0.6,
            fontweight="bold", va="top")

    # data row ---------------------------------------------------------------
    dy, dh = 0.860, 0.095
    rect(ax, 0.045, dy, 0.330, dh, face=FILL_SOURCE, lw=0.7)
    ax.text(0.210, dy + dh / 2, "source domain\nhourly observations",
            fontsize=BASE - 0.4, ha="center", va="center", linespacing=1.45)
    rect(ax, 0.560, dy, 0.330, dh, face=FILL_TARGET, lw=0.7)
    ax.text(0.725, dy + dh / 2, "target domain\ndaily aggregates only",
            fontsize=BASE - 0.4, ha="center", va="center", linespacing=1.45)

    # process row ------------------------------------------------------------
    # Raised from 0.480 when the evaluate box gained its three-line ERA5-Land bullet
    # (2026-10-20);
    # the room comes out of the gap above, which the replay route shares.
    py, ph = 0.578, 0.215
    rect(ax, 0.045, py, 0.330, ph, lw=0.9)
    ax.text(0.210, py + ph - 0.045, "Step 1  ·  pretrain", fontsize=BASE,
            fontweight="bold", ha="center", va="center")
    bullets(ax, 0.067, 0.085, py + ph - 0.082, [
        "hourly NSE objective",
        "all parameters trainable",
    ])

    rect(ax, 0.560, py, 0.330, ph, lw=0.9)
    ax.text(0.725, py + ph - 0.045, "Step 2  ·  fine-tune", fontsize=BASE,
            fontweight="bold", ha="center", va="center")
    bullets(ax, 0.582, 0.600, py + ph - 0.082, [
        "daily NSE objective",
        "hourly branch frozen",
    ])

    arrow(ax, (0.210, dy), (0.210, py + ph))
    arrow(ax, (0.725, dy), (0.725, py + ph))
    arrow(ax, (0.375, py + ph / 2), (0.560, py + ph / 2))
    ax.text(0.4675, py + ph / 2 + 0.024, "$M_0$", fontsize=BASE + 0.2, ha="center")

    # source replay ----------------------------------------------------------
    rep_y = 0.826
    elbow(ax, [(0.045, dy + dh / 2), (0.016, dy + dh / 2), (0.016, rep_y),
               (0.640, rep_y), (0.640, py + ph)], ls=(0, (2.6, 2.0)), lw=0.8)
    # The key for the dashed route goes in the figure;
    # the band beside the route is too narrow for the sentence, so it sits below instead.
    ax.plot([0.048, 0.098], [0.012, 0.012], color=LINE, linewidth=0.8,
            linestyle=(0, (2.6, 2.0)))
    ax.text(0.112, 0.012,
            "source replay (separate variant): hourly source batches mixed back into Step 2",
            fontsize=BASE - 1.0, va="center", color=MUTED)
    # outcome row ------------------------------------------------------------
    # Taller than the process row: the evaluate box carries two separate readings of
    # the same model, and squeezing them into 0.205 put the second one on the border.
    # The bottom sits at 0.068 rather than 0.098 because measuring the rendered text
    # put the last line's baseline at 0.0976 -- flush against the frame. This leaves
    # 0.030 of air below, against 0.025 above the title.
    # 0.468 = the earlier 0.322 plus one three-line bullet (3 x 0.042 + 0.020).
    oy, oh = 0.050, 0.468
    rect(ax, 0.045, oy, 0.330, oh, lw=0.9)
    ax.text(0.210, oy + oh - 0.042, "Step 3  ·  re-score", fontsize=BASE,
            fontweight="bold", ha="center", va="center")
    # Same scale and same wording pattern as the evaluate box. That this pass runs
    # twice, before and after Step 2, and reports the paired per-gauge change belongs
    # in the caption: on the figure it made the box read as a different kind of thing
    # from the one opposite.
    bullets(ax, 0.067, 0.085, oy + oh - 0.082, [
        "hourly KGE on the source\ndomain's test period",
    ])

    rect(ax, 0.560, oy, 0.330, oh, lw=0.9)
    ax.text(0.725, oy + oh - 0.042, "evaluate", fontsize=BASE, fontweight="bold",
            ha="center", va="center")
    # Two readings, bulleted because they are not the same measurement: the first has no
    # counterpart in any baseline, the second is the only scale on which a daily model
    # can be compared at all. Hanging indent, so wrapped lines align under the text and
    # not under the bullet.
    bullets(ax, 0.582, 0.600, oy + oh - 0.082, [
        "hourly KGE on the target's\ntest period",
        "daily KGE (aggregated from\nthe hourly output), compared\nto an ordinary daily LSTM",
        # ERA5-Land runoff is scored only on the African basins, never inside a CV fold,
        # so it gets its own bullet that says so rather than joining the line above.
        "outside the CV (Africa only):\ndaily KGE compared to\nERA5-Land runoff",
    ])
    # A dashed rule in the gap above that bullet, so the box reads as two parts: what
    # every cross-validation run reports, and the external African comparison. It sits
    # midway between the second item's last line (0.042 x 3 below its top) and the third
    # item's top, both derived from the spacing bullets() uses.
    y_third = oy + oh - 0.082 - (0.042 * 2 + 0.020) - (0.042 * 3 + 0.020)
    y_rule = y_third + 0.010
    ax.plot([0.572, 0.878], [y_rule, y_rule], color=MUTED, linewidth=0.6,
            linestyle=(0, (2.0, 1.6)))

    # Enters below Step 3's text rather than at the box's mid-height: the box grew
    # taller and its list is top-aligned, so the midpoint now falls inside the words.
    elbow(ax, [(0.560, py + 0.048), (0.4675, py + 0.048),
               (0.4675, oy + 0.075), (0.375, oy + 0.075)])
    ax.text(0.4560, (py + oy + oh) / 2, "$M_1$", fontsize=BASE + 0.2,
            ha="right", va="center")
    arrow(ax, (0.725, py), (0.725, oy + oh))
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT, type=Path)
    args = ap.parse_args()

    # 180 mm, the usual double-column limit.
    fig = plt.figure(figsize=(7.09, 3.55))
    a = fig.add_axes([0.012, 0.03, 0.315, 0.90]); a.axis("off")
    b = fig.add_axes([0.375, 0.03, 0.615, 0.90]); b.axis("off")
    for ax in (a, b):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    panel_a(a)
    panel_b(b)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=400)
    fig.savefig(args.out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
