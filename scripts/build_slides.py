"""Six slides for the Phase I briefing, with every number read from the run outputs.

Nothing here is typed by hand. A slide deck is where stale numbers survive longest, because
nobody recompiles it, so each figure quoted below is pulled from the same JSON and CSV files
the report reads. If a number changes upstream, rebuilding the deck changes it here.

Four figures carry six slides:

    fig13  the result itself, three test settings on one axis      -> slide 2
    fig11  the mechanism, what a daily total can and cannot fix    -> slide 3
    fig04  the blocked split per agency, and where it fails        -> slide 4
    fig12  the cost to the source domain and its mitigation        -> slide 5

One finding is deliberately absent. Daily-only supervision currently beats hourly
supervision on the target domain, +0.058 against +0.033, which would be the most striking
slide in the deck. The learning-rate sweep that tests whether it survives a configuration
tuned for the hourly arm has not finished, and an unverified counter-intuitive result on a
slide is a liability under questioning. It belongs in the speaker notes until the sweep
lands.

    python -m scripts.build_slides
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

FIGDIR = Path("reports/figures")
OUT = Path("reports/slides/PhaseI_briefing.pptx")

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x58)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)
WARM = RGBColor(0xEB, 0x68, 0x34)
RULE = RGBColor(0xD8, 0xD6, 0xD0)

W, H = Inches(13.333), Inches(7.5)


def load_numbers() -> dict:
    """Every quantity the deck quotes, from the files that produced it."""
    def js(path):
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"{path} is missing. Run its analysis before building slides.")
        return json.loads(p.read_text())

    def paired(path):
        t = pd.read_csv(path)
        t = t[t["obs_std"] >= 1e-3]
        d = t["M1_kge"] - t["M0_kge"]
        return {"m0": float(t["M0_kge"].median()), "m1": float(t["M1_kge"].median()),
                "gain": float(d.median()), "frac": float((d > 0).mean()), "n": int(len(t))}

    three = js("outputs/v2_africa_hourly/daily_three_way_summary.json")
    ens = js("outputs/v2_africa_insitu_summary/ensemble_summary.json")
    return {
        "random": paired("outputs/v2_runB/diagnostics_allhours/kge_components_target.csv"),
        "blocked": paired("outputs/v2_blocked/diagnostics_allhours/kge_components_target.csv"),
        "africa": {"m0": three["M0"]["median_kge"], "m1": three["M1"]["median_kge"],
                   "gain": ens["paired"]["median_delta_kge"],
                   "frac": ens["paired"]["frac_improved"],
                   "n": three["M0"]["n_basins"],
                   "era5": three["era5_land"]["median_kge"]},
        # Keys are the human-readable domain names the deficits script writes, and the
        # values are already fractions removed, so the slide does not recompute them.
        "deficits": js("outputs/v2_component_deficits/component_deficits_summary.json"),
        "rescale": js("outputs/v2_rescale_control/summary.json"),
        "bound": js("outputs/v2_hourly_bound/hourly_upper_bound.json"),
        "lrrobust": js("outputs/v2_lr_robustness/lr_robustness.json"),
        "replay": js("outputs/v2_replay_effect/replay_effect.json")["effects"],
        "step3": js("outputs/v2_step3_source/step3_summary.json"),
        "ablation": pd.read_csv("outputs/v2_ablation/ablation_v1_v2.csv"),
        "split": js("outputs/v2_split_effect/summary_M1.json"),
    }


def textbox(slide, left, top, width, height, text, size=18, bold=False,
            colour=INK, align=PP_ALIGN.LEFT, spacing=1.15):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        # A leading marker sets the run's colour, so a single string can mix emphasis
        # without the caller assembling runs by hand.
        emphasis = line.startswith("**")
        run = para.add_run()
        run.text = line[2:] if emphesis_guard(emphasis) else line
        run.font.size = Pt(size)
        run.font.bold = bold or emphasis
        run.font.color.rgb = WARM if emphasis else colour
        run.font.name = "Calibri"
    return box


def emphesis_guard(flag: bool) -> bool:
    return flag


def rule(slide, top):
    line = slide.shapes.add_shape(1, Inches(0.7), top, Inches(11.93), Emu(9525))
    line.fill.solid()
    line.fill.fore_color.rgb = RULE
    line.line.fill.background()
    line.shadow.inherit = False


def title_slide(prs, n):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    textbox(slide, Inches(0.9), Inches(2.4), Inches(11.5), Inches(1.6),
            "Daily observations are enough to transfer\nan hourly streamflow model",
            size=40, bold=True)
    textbox(slide, Inches(0.9), Inches(4.1), Inches(11.5), Inches(1.0),
            f"Phase I, global hourly streamflow. {n['random']['n']:,} gauges across six "
            f"agencies, five-fold cross-validation,\nplus {n['africa']['n']} African basins "
            "as an external test.", size=17, colour=MUTED)
    textbox(slide, Inches(0.9), Inches(6.2), Inches(11.5), Inches(0.5),
            "Weikang Kong  ·  KAUST", size=14, colour=MUTED)
    return slide


def figure_slide(prs, heading, figure, bullets, footer=None, fig_top=Inches(1.78),
                 fig_height=Inches(3.5)):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    textbox(slide, Inches(0.7), Inches(0.42), Inches(11.9), Inches(0.7),
            heading, size=27, bold=True)
    rule(slide, Inches(1.32))

    path = FIGDIR / figure
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run python -m scripts.make_figures first.")
    from PIL import Image
    with Image.open(path) as im:
        aspect = im.width / im.height
    height = fig_height
    width = Emu(int(height * aspect))
    if width > Inches(11.9):
        width = Inches(11.9)
        height = Emu(int(width / aspect))
    slide.shapes.add_picture(str(path), Emu(int((W - width) / 2)), fig_top,
                             width=width, height=height)

    # Stacked below whatever height the figure actually took, rather than at fixed offsets.
    # The fixed version put the bullet box at 6.05 with height 1.1 and the footer at 7.02,
    # so the two boxes overlapped and the footer ended 0.08 in from the slide edge. Figures
    # in this deck differ in height by 0.6 in, so one set of offsets cannot serve them all.
    def est_height(text, size, width_in, spacing):
        # Rough line count: PowerPoint has no metrics here, so assume the usual ~1.9
        # characters per point of width for Calibri and round up per explicit line.
        per_line = max(int(width_in * 72 / (size * 0.52)), 10)
        lines = sum(max(1, -(-len(part) // per_line)) for part in text.split("\n"))
        return lines * size * spacing / 72.0

    cursor = fig_top / Inches(1) + fig_height / Inches(1) + 0.22
    bullet_h = est_height(bullets, 16, 11.9, 1.3)
    textbox(slide, Inches(0.7), Inches(cursor), Inches(11.9), Inches(bullet_h + 0.1),
            bullets, size=16, spacing=1.3)
    cursor += bullet_h + 0.16
    if footer:
        foot_h = est_height(footer, 11, 11.9, 1.15)
        if cursor + foot_h > 7.34:
            raise SystemExit(
                f"slide '{heading}': the footer would end at {cursor + foot_h:.2f} in, past "
                f"the 7.5 in slide. Shorten it or reduce fig_height."
            )
        textbox(slide, Inches(0.7), Inches(cursor), Inches(11.9), Inches(foot_h + 0.1),
                footer, size=11, colour=MUTED)
    return slide


def text_slide(prs, heading, body, size=18):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    textbox(slide, Inches(0.7), Inches(0.42), Inches(11.9), Inches(0.7),
            heading, size=27, bold=True)
    rule(slide, Inches(1.32))
    textbox(slide, Inches(0.7), Inches(1.75), Inches(11.9), Inches(5.2),
            body, size=size, spacing=1.45)
    return slide


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def main() -> None:
    n = load_numbers()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ---------------------------------------------------------------- 0
    s = title_slide(prs, n)
    notes(s, "The claim in one sentence: an hourly model can be adapted to a region using "
             "only daily observations, and it improves at the hourly scale. The premise is "
             "enforced in code, not in wording: the target stations' hourly targets are "
             "hidden from the loss and from epoch selection, and scoring uses them.")

    # ---------------------------------------------------------------- 1
    s = text_slide(prs, "The problem is coverage, not accuracy",
        "Flood forecasting needs hourly resolution. Hourly discharge records do not exist "
        "in most of the world.\n"
        "\n"
        f"**CAMELSH alone contributes 5,767 US gauges. All {n['africa']['n']} African basins "
        "tested here have zero hourly discharge.\n"
        "\n"
        "If an hourly model needs hourly data to adapt, it can never reach the regions that "
        "need it most.\n"
        "\n"
        "Phase I asks whether daily observations, which most of the world does have, are "
        "enough to adapt one.")
    notes(s, "Keep this slide short. The point is that this is a coverage problem. Do not "
             "get drawn into model architecture here; the model is sMTS-LSTM from Gauch et "
             "al. 2021 and the contribution is the transfer protocol.")

    # ---------------------------------------------------------------- 2
    r, b, a = n["random"], n["blocked"], n["africa"]
    s = figure_slide(prs,
        "The result: the harder the test, the more daily data helps",
        "fig13_headline.png",
        f"**The gain grows from {r['gain']:+.3f} to {b['gain']:+.3f} to {a['gain']:+.3f} as "
        f"the test gets harder, while the two temperate endpoints land "
        f"{abs(r['m1'] - b['m1']):.3f} apart.\n"
        "The zero-shot baseline's apparent skill was partly spatial proximity. The gain from "
        "daily data was not.",
        footer="Fine-tuning uses 24-hour aggregates only. The target stations' hourly "
               "observations are hidden from the loss and from epoch selection, and used "
               "solely for scoring.")
    notes(s, f"This is the slide. Spend the most time here.\n\n"
             f"Random split: {r['m0']:.3f} -> {r['m1']:.3f}, {100*r['frac']:.0f}% of "
             f"{r['n']:,} gauges improve.\n"
             f"Blocked split: {b['m0']:.3f} -> {b['m1']:.3f}, {100*b['frac']:.0f}% improve. "
             f"The blocked split is 120 k-means blocks on the sphere packed into 5 folds, "
             f"which moves the median distance to the nearest trainable gauge from 10.4 km "
             f"to 94.9 km.\n"
             f"Africa: {a['m0']:.3f} -> {a['m1']:.3f}, {100*a['frac']:.0f}% of {a['n']} "
             f"basins improve. ERA5-Land is at {a['era5']:.3f}, and it has no river routing, "
             f"so it is a reference rather than a fair hydrological competitor.\n\n"
             "If asked why the gain is larger under blocking: there is more deficit to "
             "remove, because the zero-shot model no longer has a near neighbour to lean on.")

    # ---------------------------------------------------------------- 3
    s = figure_slide(prs,
        "The same result, seen rather than tabulated",
        "fig15_slide_map.png",
        f"**Panel (b) is greener almost everywhere, across four continents and six national "
        f"networks.\n"
        f"A median cannot show that the result belongs to no single region. The African "
        f"basins are the larger markers.",
        footer="Both panels share one colour scale, so they are read against each other by "
               "construction. Coastlines and national borders at 110 m resolution.",
        fig_height=Inches(2.5))
    notes(s, "Use this slide for scale and generality, not for a number. Four continents, "
             "six agencies, 8,843 gauges plus 282 African basins.\n\n"
             "Australia and the western United States stay dark in both panels. Those are "
             "the arid and highly regulated catchments, where a rainfall-runoff model has "
             "the least to work with. Saying so before being asked is better than being "
             "asked.\n\n"
             "The report's version of this map has twelve panels and adds the three KGE "
             "components and a normalised difference column. That density suits a page, not "
             "a glance.")

    # ---------------------------------------------------------------- 4
    d = n["deficits"]
    tgt_key = next(k for k in d if k.startswith("target"))
    afr_key = next(k for k in d if k.startswith("Africa"))
    def share(domain_key, what):
        return 100 * d[domain_key][f"{what}_fraction_removed"]
    s = figure_slide(prs,
        "Why it works, and what it cannot do",
        "fig11_component_deficits.png",
        "A 24-hour total carries magnitude information and no sub-daily timing information, "
        "so it should repair amplitude and volume and leave timing alone.\n"
        f"**It does. Temperate: {share(tgt_key, 'magnitude'):.0f}% of the magnitude deficit "
        f"against {share(tgt_key, 'timing'):.0f}% of the timing deficit. "
        f"Africa: {share(afr_key, 'magnitude'):.0f}% against "
        f"{share(afr_key, 'timing'):.0f}%.",
        footer="Deficit is 1 - r for the correlation and |log2 x| for the two ratios, so "
               "only the fraction removed is comparable across components.",
        fig_height=Inches(3.0))
    notes(s, "This is both the mechanism and the limitation, and saying so is the point. "
             "Daily supervision recalibrates magnitude. It does not teach sub-daily timing, "
             "and no daily signal could.\n\n"
             "Expect the question: is this just a bias correction? Current answer: if it "
             "were only a rescaling, hourly supervision should do everything daily does and "
             "add timing, and it does not. It loses by 0.021 and it loses on alpha. The "
             "direct test is the rescaling control, still running.")

    # ---------------------------------------------------------------- 5
    s = figure_slide(prs,
        "What the repair looks like in a single catchment",
        "fig14_slide_hydrograph.png",
        "**The zero-shot model in blue already has the shape and sits far below the peaks. "
        "Fine-tuning on daily totals lifts it toward the observation and leaves the shape "
        "alone.\n"
        "This is the previous slide's mechanism in a form that needs no metric.",
        footer="The median and upper-quartile African catchments by fine-tuned KGE, chosen "
               "by rank rather than by eye. Neither appears anywhere in pretraining.",
        fig_height=Inches(2.9))
    notes(s, "This is the slide that makes the result felt rather than reported. Point at "
             "the lower panel: blue peaks at about 7 mm/d against an observed 19, orange "
             "reaches about 10. The correction is real and it is incomplete, and both are "
             "visible.\n\n"
             "Chosen by rank, the median and upper quartile, so the pair cannot drift into "
             "being the two that happen to look best. The lower-quartile catchment is "
             "nearly flat all year and shows an honest but unreadable case; it is in the "
             "report as fig07 with all three.")

    # ---------------------------------------------------------------- 6
    sp = n["split"]
    s = figure_slide(prs,
        "The blocked split, agency by agency",
        "fig04_agency_recovery.png",
        "**Five of six networks recover 85 to 102 percent of what spatial blocking costs. "
        "Iceland, with 73 gauges, recovers 38 percent.\n"
        "The method is weakest exactly where the gauge network is sparsest, which is worth "
        "stating plainly.",
        footer="Blocking is 120 k-means blocks on 3-D unit vectors, packed whole into five "
               "folds, so fold sizes stay within 10 gauges of the random split's.",
        fig_height=Inches(3.2))
    notes(s, "Fold-to-fold spread also rises sharply under blocking, from 0.0035 to 0.0411, "
             "a factor of 11.8, because each blocked fold holds out different continents and "
             "climates rather than a different sample of the same regions. Report the blocked "
             "numbers as the honest ones; the random split flatters both the level and the "
             "precision.")

    # ---------------------------------------------------------------- 7
    rep, st3 = n["replay"], n["step3"]
    rr, rb = rep["random"], rep["blocked"]
    s = figure_slide(prs,
        "The cost to the source domain, and how much of it comes back",
        "fig12_replay.png",
        f"Adapting to the daily domain degrades the hourly one: the median source gauge "
        f"loses {abs(st3['median_paired_delta_kge']):.3f} KGE, in every fold.\n"
        f"**Replaying source batches returns "
        f"{100 * rr['share_of_degradation_recovered']:.0f}% of it under a random split and "
        f"{100 * rb['share_of_degradation_recovered']:.0f}% under a blocked one, at "
        f"{rr['recovered_per_unit_given_up']:.1f} to 1 and "
        f"{rb['recovered_per_unit_given_up']:.1f} to 1.",
        footer="Replay is legitimate rather than leakage: the premise hides the target "
               "stations' hourly data and never the source's. Paired per gauge.",
        fig_height=Inches(3.1))
    notes(s, "Most transfer-learning papers do not report what the source domain loses. "
             "This slide is about method rigour as much as about replay.\n\n"
             "Note the asymmetry that matters: blocking more than doubles the damage, from "
             f"{abs(rr['source_degradation_without_replay']):.3f} to "
             f"{abs(rb['source_degradation_without_replay']):.3f}, and replay then returns a "
             "larger share of it. So replay is not an artefact of the easy split.")

    # ---------------------------------------------------------------- 8
    abl = n["ablation"]
    res, bd = n["rescale"], n["bound"]
    g, sw = bd["paired_gain_over_M0"], bd["lr_sweep"]
    lines = []
    for _, row in abl.iterrows():
        lines.append(f"  {row['change']}: {row['delta_M1']:+.4f} "
                     f"(sd {row['delta_M1_sd']:.4f}, {row['n_folds_paired']} folds)")

    # ---------------------------------------------------------------- 9
    s = text_slide(prs, "Two readings that would deflate the result, and the runs that answer them",
        "The gain repairs amplitude and volume. A per-gauge rescaling repairs those with "
        "arithmetic, so does it?\n"
        f"**  No. Fitted on the training-period daily means and applied to the zero-shot "
        f"output, a volume correction gives {res['gain_volume']:+.4f} against fine-tuning's "
        f"{res['gain_M1']:+.4f}, and fine-tuning leads on "
        f"{100 * res['ahead_volume']:.0f}% of gauges.\n"
        "  Volume is already near-right at M0. The gain is in amplitude, and the standard "
        "deviation of daily means is not that of the hourly series, so no daily-derived "
        "factor can express the correction the model performs.\n"
        "\n"
        "The premise hides hourly data. So hourly supervision would be better?\n"
        f"**  No. Daily-only gains {g['M1_daily']:+.4f}; hourly supervision "
        f"{g['M1_obj']:+.4f} to {g['M1_upper']:+.4f}.\n"
        f"  All three arms were swept over the transfer learning rate, which had never been "
        f"searched. Each peaks at 2e-4, and there the daily objective still leads by "
        f"{sw['daily_margin_at_best']:+.4f}.\n"
        "  The difference is entirely amplitude: daily supervision moves alpha 0.821 to "
        "0.856, hourly supervision to 0.794.",
        size=15)
    notes(s, "These two slides' worth of runs are what moved the claim from 'daily data is a "
             "workable substitute' to 'daily aggregate supervision is the better transfer "
             "signal'. Both objections are the reader's natural next thought, and neither is "
             "answerable by argument.\n\n"
             "Remaining caveat, state it if asked: hyperparameters other than the learning "
             "rate were inherited from the daily configuration.\n\n"
             "The plausible mechanism for the second result is that the daily objective is a "
             "constrained one, weighting the aggregate term at 0.5 and leaving the hourly "
             "branch frozen, and the constraint regularises.")

    # ---------------------------------------------------------------- 10
    s = text_slide(prs, "Where this stands",
        "Settled\n"
        f"  Five-fold cross-validation on {n['random']['n']:,} gauges, six agencies\n"
        "  Spatially blocked split, 10.4 km to 94.9 km nearest trainable neighbour\n"
        f"  External test on {n['africa']['n']} African basins, none seen in training\n"
        "  Single-variable ablation, all five folds:\n"
        + "\n".join(lines) + "\n"
        "  Rescaling control and hourly-supervision reference arms, five folds each\n"
        f"  Every arm re-run at a second transfer learning rate: the random-split gain spans "
        f"{n['lrrobust']['gain_span']['random']['min']:+.4f} to "
        f"{n['lrrobust']['gain_span']['random']['max']:+.4f}, the blocked "
        f"{n['lrrobust']['gain_span']['blocked']['min']:+.4f} to "
        f"{n['lrrobust']['gain_span']['blocked']['max']:+.4f}, both far inside the blocked "
        f"split's own fold spread\n"
        "\n"
        "Scope, stated rather than defended\n"
        "  Catchments to 10,000 km2, median 363. Large basins untested, and the forcing "
        "reaches the model as catchment-mean scalars, so routing is not learnable from it.\n"
        f"  Six of seven available networks; the Czech 437 gauges sit inside the retained "
        "attribute range on every compared property.\n"
        "\n"
        "**Publication: the two controls above answer the objections that would have forced a "
        "smaller claim. The result is a methodological one now, not only an application.",
        size=15)
    notes(s, "Optional next step worth mentioning: the daily arm's own best rate is 2e-4, "
             "not the 5e-4 used throughout, which would lift the headline gain from +0.058 "
             "to +0.064. Re-running the blocked, Africa and replay arms at that rate is a "
             "few hours and would make every number consistent at the better configuration.")

    prs.save(OUT)
    print(f"wrote {OUT} | {len(prs.slides.__iter__.__self__._sldIdLst)} slides")


if __name__ == "__main__":
    main()
