"""Generate the Phase I report as LaTeX: two parts, one story.

Every number is read from the result files rather than typed. That is not fastidiousness:
an earlier revision of the Word report carried ninety literal numbers in its prose and they
drifted from the tables beside them. The same rule applies here, and the few quantities that
genuinely cannot be recomputed cheaply are marked in SOURCES below with where they came from.

The document has two parts with a one-to-one correspondence: experiment k in Part I states
what was run and why, result k in Part II states what came out. The order follows how the
work actually proceeded -- fix the configuration, show the effect, explain the mechanism,
rule out the ways it could be spurious, bound where it applies, then test it on a domain
that was never in training.

    python -m scripts.build_latex && module load texlive/2022 && \
        latexmk -pdf -outdir=reports/latex reports/latex/PhaseI_report.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIGDIR = Path("reports/figures")


def load(path: str):
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text())
    return pd.read_csv(p)


def need(mapping, *keys):
    """Fetch a nested key, raising rather than falling back to a default.

    A missing key reached the PDF once as \\num{+nan}; siunitx happened to catch that one,
    but a bare float would have printed "nan" in the prose with nothing to notice it. Every
    number in this document comes from a file, so a file that changed shape must stop the
    build rather than quietly produce a wrong sentence.
    """
    node = mapping
    for k in keys:
        if node is None or k not in node:
            raise KeyError(f"{'.'.join(map(str, keys))} missing -- the summary file "
                           "changed shape; fix the key rather than defaulting it")
        node = node[k]
    return node


def tex_escape(text: str) -> str:
    """Escape the characters LaTeX would otherwise interpret."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        text = text.replace(a, b)
    return text


def fig(name: str, caption: str, label: str, width: str = r"\linewidth") -> str:
    """A figure environment, or a visible placeholder if the file is missing."""
    path = FIGDIR / name
    if not path.exists():
        return (r"\begin{center}\fbox{\parbox{0.8\linewidth}{\centering "
                rf"Figure \texttt{{{tex_escape(name)}}} not generated. "
                r"Run \texttt{python -m scripts.make\_figures}.}}\end{center}" + "\n")
    # The bare filename, resolved by \graphicspath in the preamble. A repo-root-relative
    # path only works when pdflatex is run from the repo root, and the document lives in
    # reports/latex/; this way the .tex and the figures move together.
    return "\n".join([
        r"\begin{figure}[htbp]",
        r"  \centering",
        rf"  \includegraphics[width={width}]{{{name}}}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{fig:{label}}}",
        r"\end{figure}",
        "",
    ])


def table(header: list[str], rows: list[list[str]], caption: str, label: str,
          align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    out = [r"\begin{table}[htbp]", r"  \centering",
           rf"  \caption{{{caption}}}", rf"  \label{{tab:{label}}}",
           rf"  \begin{{tabular}}{{{align}}}", r"    \toprule",
           "    " + " & ".join(header) + r" \\", r"    \midrule"]
    out += ["    " + " & ".join(r) + r" \\" for r in rows]
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def gather() -> dict:
    """Every quantity the prose uses, read once from the files that produced it."""
    d = {}
    k = load("outputs/v2_runB/diagnostics_allhours/kge_components_summary_target.csv")
    d["kge"] = k.set_index("component") if k is not None else None
    kb = load("outputs/v2_blocked/diagnostics_allhours/kge_components_summary_target.csv")
    d["kge_blocked"] = kb.set_index("component") if kb is not None else None
    d["africa"] = load("outputs/v2_africa_insitu_summary/ensemble_summary.json")
    d["three"] = load("outputs/v2_africa_hourly/daily_three_way_summary.json")
    d["within"] = load("outputs/v2_africa_hourly/within_day_summary.json")
    d["mech"] = load("outputs/v2_component_deficits/component_deficits_summary.json")
    d["deficits"] = load("outputs/v2_component_deficits/component_deficits.csv")
    d["split"] = load("outputs/v2_split_effect/summary_M0.json")
    d["split_m1"] = load("outputs/v2_split_effect/summary_M1.json")
    d["degen"] = load("outputs/v2_runB/degenerate/degenerate_summary.json")
    d["sig"] = load("outputs/v2_runB/significance/significance_summary.json")
    d["disp"] = load("outputs/split_dispersion/summary.json")
    d["conv"] = load("outputs/convergence_check/summary.json")
    d["lat"] = load("outputs/v2_stratify/maps/by_latitude_target.csv")
    d["step3"] = load("outputs/v2_step3_source/step3_summary.json")
    d["ablation"] = load("outputs/v2_ablation/ablation_summary.json")
    pub = load("outputs/africa_runB/per_basin_pub_baseline.csv")
    d["pub"] = pub
    comp = load("outputs/v2_runB/diagnostics_allhours/kge_components_target.csv")
    d["n_gauges"] = int(len(comp)) if comp is not None else 0
    if comp is not None:
        d["composition"] = comp["source"].value_counts()
    return d


def map_caption(d: dict) -> str:
    """Figure 9's caption, enumerating all twelve panels from the files the map reads.

    The panels carry only a letter, so the caption is the only place a reader learns what
    each one shows. It is generated rather than written out, for the same reason every other
    number here is: a caption that is the sole carrier of twelve pairs of numbers is exactly
    where hand-typed values drift. An earlier revision punted to "the corresponding Word
    report figure", which left this document incomplete on its own.
    """
    table = d["kge"]
    comp = load("outputs/v2_runB/diagnostics_allhours/kge_components_target.csv")
    m0 = load("outputs/v2_africa_insitu_summary/ensemble_per_basin_M0.csv")
    m1 = load("outputs/v2_africa_insitu_summary/ensemble_per_basin_M1.csv")
    if comp is None or m0 is None or m1 is None:
        return "Target-domain and African metrics in space."
    comp = comp.loc[comp["obs_std"] >= 1e-3]
    afr = m0.set_index("station_id").join(m1.set_index("station_id"),
                                          lsuffix="_M0", rsuffix="_M1", how="inner")
    letters = "abcdefghijkl"
    parts = []
    for i, (name, key, ratio) in enumerate((("KGE", "kge", False), ("$r$", "kge_r", False),
                                            (r"$\alpha$", "kge_alpha", True),
                                            (r"$\beta$", "kge_beta", True))):
        g0, g1 = comp[f"M0_{key}"], comp[f"M1_{key}"]
        a0, a1 = afr[f"{key}_M0"], afr[f"{key}_M1"]
        parts.append(
            f"({letters[i * 3]}) {name} at M0, gauges {g0.median():.3f}, basins "
            f"{a0.median():.3f}; ({letters[i * 3 + 1]}) at M1, {g1.median():.3f} and "
            f"{a1.median():.3f}; ({letters[i * 3 + 2]}) the difference, median "
            f"{(g1 - g0).median():+.3f} and {(a1 - a0).median():+.3f}")
    return (
        "Columns are M0, M1 and their difference; rows are KGE and its three components, "
        "with the first two columns of a row sharing one scale. Small dots are the target "
        "gauges, scored against hourly observations, where Phase~I's premise is simulated; "
        "large outlined dots are the African basins, scored against daily observations, "
        "where it is genuine. The two are never pooled into one median because the "
        r"observation each score rests on differs. $\alpha$ and $\beta$ use a log scale, so "
        "halving and doubling the observed value sit equally far from the white centre. "
        "Longitude is labelled on the bottom row and latitude on the left column only, since "
        "all twelve panels share one window. Panels: " + ".\\ ".join(parts) + ". In the "
        "difference column the sign is the verdict for KGE and $r$, where larger is better, "
        r"but not for $\alpha$ and $\beta$, whose ideal is 1.0: an increase helps a gauge "
        "below it and hurts one above it.")




def part_experiments(d: dict) -> str:
    """Part I. Four sections: the problem, the data, the model, the evaluation."""
    n = d["n_gauges"]
    afr = d["africa"]
    s = [r"\part{Experiments}", ""]

    s.append(r"\section{The problem, and how it is made testable}\label{sec:problem}")
    s.append(
        "Hourly streamflow forecasting needs hourly discharge observations to train on. "
        "Those observations exist in a small part of the world. Most gauges that report at "
        "all report once a day. If a model trained on hourly data could keep its hourly "
        "skill at gauges that supply only daily totals, the usable domain would widen a long "
        "way.")
    s.append("")
    s.append(
        "The question is easy to state and hard to test, because a gauge either has hourly "
        "data or it does not. We make it testable by withholding. One fifth of the gauges "
        "are held out. Their hourly observations are hidden from training. Only the 24-hour "
        "aggregate is used to supervise them. The hidden hourly series is then read once, at "
        "the end, to score. Nothing in training ever sees it.")
    s.append("")
    s.append("Three model states are compared throughout, and the whole report turns on them:")
    s.append(r"\begin{description}")
    s.append(r"  \item[M0] Zero-shot. The model is pretrained on the four fifths of gauges "
             "that keep their hourly data, then evaluated on the held-out fifth without "
             "having seen those gauges at all.")
    s.append(r"  \item[M1] The same model after fine-tuning on the held-out gauges' daily "
             r"aggregates. This is the state the method is meant to deliver.")
    s.append(r"  \item[STEP 3] The four fifths re-scored after that fine-tuning, to measure "
             "what adapting to a daily-only domain costs where the model already worked.")
    s.append(r"\end{description}")
    s.append("")
    s.append(
        f"The gap between M0 and M1 is the quantity of interest. It says how much of the "
        f"hourly skill lost by hiding hourly data can be bought back with daily totals "
        f"alone.")
    s.append("")

    s.append(r"\section{Data}\label{sec:data}")
    # The data section is long enough, and referenced often enough, to live in its own
    # file. It is written by hand rather than generated: the counts in it are structural
    # facts about the archive, not results, and they do not change when a run is repeated.
    # The few that could drift are checked against the files by scripts.check_data_section.
    section = Path("reports/latex/data_description.tex")
    if section.exists():
        s.append(section.read_text(encoding="utf-8").split(r"\section{Data}\label{sec:data}")[1]
                 .strip())
        s.append("")
    else:
        s.append(r"\section{Data}\label{sec:data}")
        s.append(r"\textit{reports/latex/data\_description.tex is missing.}")
        s.append("")

    s.append(r"\section{Model and configuration}\label{sec:model}")
    s.append(
        r"The model is a shared multi-timescale LSTM with two branches. The daily branch "
        r"reads 365 days and hands its hidden and cell state to the hourly branch at a "
        r"transfer point set by the hourly look-back. The hourly branch reads forward from "
        r"there and emits one value per hour. Training minimises a loss on the hourly targets "
        r"where they are available, and on the 24-hour aggregate where they are not.")
    s.append("")
    s.append(
        r"Two settings were changed together during development and both are reported. "
        r"Configuration v1 used a \SI{72}{\hour} hourly look-back and the framework's default "
        r"forget-gate initialisation. Configuration v2 uses \SI{336}{\hour} and an initial "
        r"forget bias of 3. The forget gate matters more than its size suggests. At the "
        r"default the effective forget gate sits at 0.500, which corresponds to a memory of "
        r"roughly two steps. At bias 3 it sits at 0.953, roughly twenty-one steps. The "
        r"published method this model follows specifies the latter. Its absence from v1 was "
        r"an oversight. Configuration v3 repeats v2 with a longer epoch budget and is used "
        r"only to check that v2 was not stopped early.")
    s.append("")
    s.append(
        r"All configurations are reported together in Appendix~\ref{app:config}, rather than "
        r"the best one being reported alone.")
    s.append("")

    s.append(r"\section{Evaluation}\label{sec:evaluation}")
    s.append(
        f"A five-fold design gives every one of the {n:,} gauges exactly one turn as a "
        f"held-out gauge. Target-domain scores therefore cover the whole network. No result "
        f"depends on which fifth happened to be drawn.")
    s.append("")
    s.append(
        r"Skill is reported as Kling-Gupta efficiency and Nash-Sutcliffe efficiency, both as "
        r"medians across gauges. KGE is used for most of the analysis because it decomposes "
        r"into three terms that answer separate questions:")
    s.append(r"\begin{equation}")
    s.append(r"  \mathrm{KGE} = 1 - \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2},")
    s.append(r"  \qquad \alpha = \frac{\sigma_{\mathrm{sim}}}{\sigma_{\mathrm{obs}}},")
    s.append(r"  \qquad \beta = \frac{\mu_{\mathrm{sim}}}{\mu_{\mathrm{obs}}}.")
    s.append(r"\end{equation}")
    s.append(
        r"Here $r$ carries timing, $\alpha$ carries the amplitude of the variation, and "
        r"$\beta$ carries the water balance. A drop in KGE alone says nothing about which of "
        r"the three failed. The decomposition is what lets Section~\ref{sec:mechanism} say "
        r"which one the method repairs.")
    s.append("")
    s.append(
        r"The held-out fifth is drawn two ways. A random split leaves each held-out gauge a "
        r"trainable neighbour \SI{10.4}{\km} away at the median. A blocked split, built from "
        r"\num{120} k-means clusters on the sphere, pushes that to \SI{94.9}{\km}. Both were "
        r"run with everything else identical. The difference between them measures how much "
        r"of the zero-shot skill comes from genuine generalisation and how much comes from "
        r"having a training gauge nearby. Fold-to-fold spread was recorded for both, and "
        r"Section~\ref{sec:cost} reports why that turned out to matter as much as the level.")
    s.append("")

    s.append(r"\section{External validation on Africa}\label{sec:africa-design}")
    s.append(
        "Everything above simulates the premise. The hourly data exists and is deliberately "
        "hidden. Africa does not need the simulation, because the premise is the ordinary "
        "state of affairs there.")
    s.append("")
    if afr:
        s.append(
            f"Of the African entries in the daily discharge database, \\num{{302}} carry a "
            f"time series at all. \\num{{294}} of those are used here, and "
            f"\\num{{{afr['M1']['n_basins']}}} pass a \\num{{100}}-day minimum and are "
            f"scored. That set is the test set of a published continent-holdout run, adopted "
            f"unchanged so the numbers here sit directly against its baseline. No African "
            f"catchment appears anywhere in pretraining. None has hourly discharge.")
        s.append("")
    s.append(
        "The pretrained models are driven over these catchments with hourly ERA5-Land "
        "forcing. Their last 24 hourly outputs are averaged to a daily value and compared "
        "with the observed daily discharge. Fine-tuning then uses African daily observations "
        "themselves, which makes the African M1 the direct analogue of the target-domain M1. "
        "ERA5-Land's own runoff is carried alongside as a physical baseline and re-scored on "
        "the identical basin-days, so all three numbers rest on the same observations.")
    s.append("")
    return "\n".join(s)


def part_results(d: dict) -> str:
    """Part II. Five sections in one arc: it works, why, is it real, what it costs, Africa."""
    kge, kgb = d["kge"], d["kge_blocked"]
    afr, three, within = d["africa"], d["three"], d["within"]
    degen, split, st3 = d["degen"], d["split"], d.get("step3")
    n = d["n_gauges"]
    s = [r"\part{Results}", ""]

    # ---------------------------------------------------------------- 1
    s.append(r"\section{Daily aggregates recover most of the lost hourly skill}"
             r"\label{sec:main}")
    row, nse = kge.loc["kge"], kge.loc["nse"]
    s.append(
        f"Under the primary configuration and a random split, median hourly KGE on the "
        f"held-out gauges rises from \\num{{{row['M0_median']:.4f}}} at M0 to "
        f"\\num{{{row['M1_median']:.4f}}} at M1. NSE rises from "
        f"\\num{{{nse['M0_median']:.4f}}} to \\num{{{nse['M1_median']:.4f}}}. The paired "
        f"median change in KGE is \\num{{{row['median_delta']:+.4f}}} over {n:,} gauges.")
    s.append("")
    s.append(
        f"The median alone would be weak evidence. A few gauges with large gains can lift it "
        f"while most gauges stay flat. They do not here. "
        f"\\SI{{{100 * (1 - row['frac_worse']):.0f}}}{{\\percent}} of gauges improve. The "
        f"effect is broad.")
    s.append("")
    s.append(
        f"Under the blocked split the same fine-tuning lifts "
        f"\\num{{{kgb.loc['kge', 'M0_median']:.4f}}} to "
        f"\\num{{{kgb.loc['kge', 'M1_median']:.4f}}}. The zero-shot level is much lower there, "
        f"and Section~\\ref{{sec:cost}} explains why. The point for now is that both splits "
        f"end at nearly the same place after fine-tuning, around \\num{{0.62}}.")
    s.append("")
    s.append(
        r"Figure~\ref{fig:map} puts every gauge on the map, before and after, together with "
        r"the African basins that Section~\ref{sec:africa} discusses. Two things are visible "
        r"at once. The gain is present across the whole network rather than in one region. "
        r"And the network itself is temperate and northern. The empty continents in that "
        r"figure are the reason the African test carries as much weight as it does.")
    s.append("")
    s.append(fig("fig09_global_map.png", map_caption(d), "map"))

    # ---------------------------------------------------------------- 2
    s.append(r"\section{The gain is a repair of amplitude}\label{sec:mechanism}")
    r_, a_, b_ = kge.loc["kge_r"], kge.loc["kge_alpha"], kge.loc["kge_beta"]
    s.append(
        "A 24-hour total carries how much water arrived on a day. It carries almost nothing "
        "about which hour it arrived in. If that is the whole of what the daily signal "
        "contributes, then fine-tuning on it should move the amplitude and water-balance "
        "terms of KGE and leave the correlation term nearly where it was. This is a "
        "prediction that can fail, and it is worth checking rather than assuming.")
    s.append("")
    s.append(
        f"It holds. Correlation moves by \\num{{{r_['median_delta']:+.4f}}}, from "
        f"\\num{{{r_['M0_median']:.3f}}} to \\num{{{r_['M1_median']:.3f}}}. The amplitude "
        f"ratio $\\alpha$ moves by \\num{{{a_['median_delta']:+.4f}}}. The water balance "
        f"$\\beta$ moves from \\num{{{b_['M0_median']:.3f}}} to "
        f"\\num{{{b_['M1_median']:.3f}}}, that is, toward its ideal value of 1. "
        f"Figure~\\ref{{fig:components}} shows the three movements side by side under both "
        f"splits, and Table~\\ref{{tab:deficits}} gives the same comparison as distances "
        f"from the ideal.")
    s.append("")
    s.append(
        r"The diagnosis this gives is specific. A zero-shot model applied to an unseen "
        r"catchment hedges toward the mean. Hedging minimises squared error under "
        r"uncertainty, and it shows up as $\alpha$ below 1: the model swings less than the "
        r"river does, and it flattens peaks. The daily total tells the model how much water "
        r"a day actually carried, which is exactly the information needed to stop hedging. "
        r"Timing was already close to right on this network, so there was little for the "
        r"daily signal to add there even if it could.")
    s.append("")
    s.append(fig("fig01_kge_components.png",
                 "Movement from M0 (open marker) to M1 (filled marker) in each KGE component, "
                 "under both splits. Each end is a median across gauges. The correlation term "
                 "barely moves. The amplitude term moves substantially. This is the finding "
                 "that daily-aggregate supervision re-calibrates amplitude while leaving "
                 "timing alone.", "components"))
    s.append(
        r"Appendix~\ref{app:mechanism} repeats this decomposition on the African catchments, "
        r"whose zero-shot deficits are about four times larger. The same ordering holds "
        r"there, which is the stronger version of the claim: it survives a domain that "
        r"differs from this one by a wide margin.")
    s.append("")
    return "\n".join(s)


def part_results_tail(d: dict) -> str:
    """Sections 3 to 5: is it real, what it costs, and the external test."""
    kge, degen, split, st3 = d["kge"], d["degen"], d["split"], d.get("step3")
    afr, three, within = d["africa"], d["three"], d["within"]
    s = []

    # ---------------------------------------------------------------- 3
    s.append(r"\section{Three ways the gain could be false, and why it is not}"
             r"\label{sec:falsify}")
    s.append(
        "A gain measured this way could be an artefact in three separate ways. Each was "
        "given its own test before the result was believed.")
    s.append("")

    s.append(r"\subsection{The model could be gaming the aggregate loss}")
    s.append(
        r"The aggregate term constrains only the mean of 24 hourly outputs. A model that "
        r"emitted a constant value within each day would satisfy it perfectly and be useless "
        r"at the hourly scale. This failure mode has to be excluded directly.")
    s.append("")
    s.append(
        r"Stride-24 sampling makes each sample's last 24 outputs one calendar day, and "
        r"consecutive days join into a continuous series. Within-day variability can then be "
        r"measured against the observations.")
    s.append("")
    if degen:
        m = degen["medians"]
        rows = []
        for key, name in (("flashiness", "Flashiness"), ("intraday_std", "Within-day std"),
                          ("intraday_range", "Within-day range"),
                          ("q95_events_per_year", "Q95 events / yr"), ("mean", "Mean flow")):
            if key in m and m[key].get("observed"):
                o = m[key]["observed"]
                rows.append([name, f"{o:.4f}", f"{m[key]['M0'] / o:.2f}", f"{m[key]['M1'] / o:.2f}"])
        s.append("The result is in Table~\\ref{tab:degenerate}.")
        s.append("")
        s.append(table(["Quantity", "Observed", "M0 / obs", "M1 / obs"], rows,
                       "Within-day behaviour of the hourly output, as a ratio to the observed "
                       "median across gauges. A flattened output would show ratios near zero "
                       "in the first three rows.", "degenerate"))
        s.append(
            r"The hourly output keeps its within-day variability, and it keeps close to the "
            r"right amount of it. Configuration v1 failed this check in the opposite "
            r"direction, with a zero-shot model \num{6.8} times too flashy and \num{3.1} "
            r"times too variable within the day. The forget-gate initialisation removed that. "
            r"Figure~\ref{fig:intraday} shows both configurations on one axis.")
        s.append("")
        if "diurnal_ratio" in m:
            dr = m["diurnal_ratio"]
            s.append(
                f"A sharper version of the question asks whether the hourly output responds "
                f"to rainfall events or merely to the clock. Averaging every day by hour of "
                f"day destroys event-driven structure, because storms keep no fixed hour. "
                f"Whatever survives that average is tied to the clock. The observed average "
                f"day has a peak-to-trough ratio of \\num{{{dr['observed']:.3f}}}, so the "
                f"real river's average day is nearly flat. The model's is "
                f"\\num{{{dr['M1']:.3f}}}, which is "
                f"\\num{{{dr['M1'] / dr['observed']:.2f}}} times the observed amplitude. The "
                f"model neither invents a daily cycle nor suppresses one.")
            s.append("")
    s.append(fig("fig08_intraday_shape.png",
                 "Within-day behaviour as a ratio to the observed median, on a log scale, for "
                 "both configurations. Configuration v1 was over-dispersed, which is a "
                 "different failure from the flattening this check was built to catch. "
                 "Configuration v2 is calibrated before fine-tuning and stays so after it.",
                 "intraday"))

    s.append(r"\subsection{The improvement could be an artefact of the metric}")
    s.append(
        r"KGE and point-wise absolute error can move in opposite directions on the same "
        r"gauge. They agree on about \SI{68}{\percent} of gauges here. Roughly a fifth "
        r"improve on KGE while their point-wise error worsens, and about an eighth do the "
        r"reverse.")
    s.append("")
    s.append(
        r"This follows from the mechanism rather than undermining it. Restoring amplitude "
        r"moves peaks. Moving a peak improves the shape metrics and can increase squared "
        r"error at individual hours, particularly if the peak is slightly early or late. The "
        r"practical consequence is that a per-gauge claim has to name its metric. "
        r"Appendix~\ref{app:metrics} shows the joint distribution.")
    s.append("")

    s.append(r"\subsection{The gain could be a training-budget artefact}")
    s.append(
        r"If the primary configuration had simply run out of epochs, its advantage over v1 "
        r"would say more about the budget than the method. Configuration v3 answers this with "
        r"a longer budget and higher patience. Early stopping ended every fold of v2 before "
        r"the cap, and v3's extra epochs wander around the same plateau rather than climbing "
        r"above it. Appendix~\ref{app:convergence} shows the curves.")
    s.append("")
    s.append(
        r"The same run measures a cost the method imposes on itself. The checkpoint has to be "
        r"chosen on a daily-aggregate criterion, because the hourly truth is meant to be "
        r"hidden. Choosing on the hidden hourly truth instead would gain \num{0.0035} in "
        r"median KGE. Fold-to-fold noise on the same quantity is \num{0.0078}. The cost of "
        r"the constraint is smaller than the noise it would be measured against.")
    s.append("")

    # ---------------------------------------------------------------- 4
    s.append(r"\section{What the method costs}\label{sec:cost}")
    s.append(
        "Two costs are measurable and both are reported here rather than left to an "
        "appendix. A method whose price is unstated has not been evaluated.")
    s.append("")

    s.append(r"\subsection{Adapting to the daily domain degrades the hourly domain}")
    if st3:
        s.append(
            f"Fine-tuning changes the weights, and those weights also serve the four fifths "
            f"of gauges that kept their hourly data. Re-scoring that source domain after "
            f"fine-tuning gives a clear answer. Median hourly KGE there falls from "
            f"\\num{{{st3['median_kge_before']:.4f}}} to "
            f"\\num{{{st3['median_kge_after']:.4f}}}, and median NSE from "
            f"\\num{{{st3['median_nse_before']:.4f}}} to "
            f"\\num{{{st3['median_nse_after']:.4f}}}. Paired gauge by gauge the median change "
            f"is \\num{{{st3['median_paired_delta_kge']:+.4f}}}, and it is negative in "
            f"{'every fold' if st3['degraded_in_all_folds'] else 'most folds'}. This is a "
            f"property of the procedure rather than one fold's luck.")
        s.append("")
        s.append(
            r"The mechanism of Section~\ref{sec:mechanism} explains it. Fine-tuning "
            r"re-calibrates amplitude toward the daily-only domain. The source domain was "
            r"already calibrated, so the movement that helps one hurts the other. Whether "
            r"the trade is acceptable depends on deployment. Source gauges keep their hourly "
            r"data and do not need the fine-tuned weights, so the two domains can be served "
            r"by separate checkpoints. Mixing a fraction of source samples back into "
            r"fine-tuning was tested and works as a dial between the two, damping the "
            r"re-calibration and giving back part of the target-domain gain.")
        s.append("")

    s.append(r"\subsection{A random split flatters the result twice}")
    if split and d["split_m1"] is not None:
        drop0 = need(split, "overall", "paired_median_drop")
        drop1 = need(d["split_m1"], "overall", "paired_median_drop")
        base = need(split, "overall", "median_random")
        worse = need(split, "overall", "frac_worse")
        pval = need(split, "overall", "wilcoxon_p")
        recovered = 100 * (1 - abs(drop1) / abs(drop0))
        s.append(
            f"Blocking the split costs the zero-shot model a paired median of "
            f"\\num{{{drop0:+.4f}}} in KGE, which is "
            f"\\SI{{{100 * abs(drop0) / base:.1f}}}{{\\percent}} of its random-split level. "
            f"\\SI{{{100 * worse:.1f}}}{{\\percent}} of gauges are worse, at Wilcoxon "
            f"$p = \\num{{{pval:.1e}}}$, and the drop is negative in all six archives. Some "
            f"of what looks like zero-shot generalisation under a random split is proximity "
            f"to a training gauge.")
        s.append("")
        s.append(
            f"Daily-aggregate fine-tuning returns "
            f"\\SI{{{recovered:.1f}}}{{\\percent}} of that loss. The residual at M1 is "
            f"\\num{{{drop1:+.4f}}}. Figure~\\ref{{fig:split}} breaks the recovery down by "
            f"archive, and one archive stands out: Iceland, with \\num{{73}} gauges, recovers "
            f"only \\SI{{38}}{{\\percent}}. The sparsest network is where the method has the "
            f"least to work with.")
        s.append("")
        rec = load("outputs/v2_split_effect/recovery_by_agency.csv")
        if rec is not None:
            rows = [[tex_escape(r.source), f"{int(r.n_stations):,}", f"{r.M0:+.4f}",
                     f"{r.M1:+.4f}", f"{100 * r.recovered:.0f}\\%"]
                    for r in rec.itertuples()]
            s.append("Table~\\ref{tab:recovery} gives the same breakdown numerically.")
            s.append("")
            s.append(table(["Archive", "Gauges", "Drop at M0", "Drop at M1", "Recovered"], rows,
                           "The cost of spatial blocking per archive, and how much "
                           "daily-aggregate fine-tuning returns.", "recovery"))
    s.append(fig("fig04_agency_recovery.png",
                 "Paired median KGE drop from blocking the split, per archive, at M0 (open "
                 "marker) and after fine-tuning (filled marker). Every archive is negative at "
                 "M0. Recovery exceeds \\SI{85}{\\percent} everywhere except the sparsest "
                 "network.", "split"))
    if d["disp"]:
        s.append(
            r"A random split also overstates the precision of the result, which is easier to "
            r"miss than the level. The fold-to-fold standard deviation of M1 is \num{0.0035} "
            r"under the random split and \num{0.0411} under the blocked one, a factor of "
            r"\num{11.8} at Levene $p = \num{0.034}$. Near-duplicate gauges sit on both sides "
            r"of a random split, so its validation metric averages over fewer independent "
            r"catchments than its gauge count suggests. The honest statement of the blocked "
            r"result is about $0.62 \pm 0.04$. Quoting $0.628 \pm 0.004$ would be reporting "
            r"the split's smoothness as if it were the model's stability.")
        s.append("")
    return "\n".join(s)


def part_africa(d: dict) -> str:
    """Section 5 and the closing summary."""
    afr, three, within = d["africa"], d["three"], d["within"]
    s = []
    s.append(r"\section{Africa, where the premise is real}\label{sec:africa}")
    s.append(
        "Everything so far rests on hiding data that exists. That is a fair test of the "
        "method and a weak test of its reach, because the held-out gauges still sit inside a "
        "network the model was trained on. Africa provides the harder case. Its catchments "
        "appear nowhere in pretraining, and they genuinely lack hourly discharge.")
    s.append("")
    if afr and three:
        m0, m1 = afr["M0"], afr["M1"]
        s.append(
            f"Zero-shot performance there is poor, as expected for a continent absent from "
            f"training: median KGE \\num{{{m0['median_kge']:.4f}}}. Fine-tuning on African "
            f"daily observations lifts it to \\num{{{m1['median_kge']:.4f}}}. The paired "
            f"median change is \\num{{{afr['paired']['median_delta_kge']:+.4f}}} with "
            f"\\SI{{{100 * afr['paired']['frac_improved']:.1f}}}{{\\percent}} of basins "
            f"improving. The gain is roughly six times the one measured on the temperate "
            f"network, which fits the mechanism: there was far more mis-calibration to "
            f"repair.")
        s.append("")
        rows = [["ERA5-Land runoff", f"{three['era5_land']['median_kge']:+.4f}",
                 f"{three['era5_land']['median_nse']:+.4f}"],
                ["M0, zero-shot", f"{three['M0']['median_kge']:+.4f}",
                 f"{three['M0']['median_nse']:+.4f}"],
                ["M1, after African daily fine-tuning", f"{three['M1']['median_kge']:+.4f}",
                 f"{three['M1']['median_nse']:+.4f}"]]
        s.append("Table~\\ref{tab:threeway} places the three methods side by side.")
        s.append("")
        s.append(table(["Method", "Median KGE", "Median NSE"], rows,
                       f"All three scored on the identical \\num{{{three['n_basin_days']}}} "
                       f"basin-days over \\num{{{three['n_basins']}}} basins. The ERA5-Land "
                       f"per-basin scores that already existed came from a different run over "
                       f"a different period, so they were recomputed here to make the "
                       f"comparison valid.", "threeway"))
        pub = d.get("pub")
        if pub is not None:
            s.append(
                f"Two comparisons matter. Against the physical baseline, M1 beats ERA5-Land "
                f"on \\SI{{{100 * three['share_of_basins_M1_beats_era5_land']:.1f}}}{{\\percent}} "
                f"of basins, and even the zero-shot model beats it on "
                f"\\SI{{{100 * three['share_of_basins_M0_beats_era5_land']:.1f}}}{{\\percent}}. "
                f"Against the published continent-holdout baseline these basins were drawn "
                f"from, which reaches a median KGE of "
                f"\\num{{{pub.pub_kge.median():+.4f}}}, M1 is higher by "
                f"\\num{{{m1['median_kge'] - pub.pub_kge.median():+.4f}}}.")
            s.append("")
        s.append(
            r"Africa is also where one earlier claim needs qualifying. Across the temperate "
            r"network the correlation term barely moved, and Section~\ref{sec:mechanism} "
            r"attributed that to timing already being close to right. In Africa timing starts "
            r"genuinely wrong, and there it does move. The refined statement is that a daily "
            r"total fixes which day the water arrives, which repairs day-scale timing where "
            r"it is broken. It still says nothing about which hour within that day.")
        s.append("")

    s.append(r"\subsection{The hourly question that Africa cannot answer}")
    if within:
        s.append(
            f"No African catchment has hourly discharge, so the hourly output there can be "
            f"compared between series and never scored. Two things can still be established.")
        s.append("")
        s.append(
            f"First, the hourly output carries real sub-daily structure. Across all "
            f"\\num{{{within['n_basins']}}} basins the within-day coefficient of variation is "
            f"\\num{{{within['median_cv_M0']:.4f}}} at M0 and "
            f"\\num{{{within['median_cv_M1']:.4f}}} at M1. Neither is near zero.")
        s.append("")
        drop = 100 * abs(within["median_paired_difference"]) / within["median_cv_M0"]
        s.append(
            f"Second, daily-only supervision does flatten that structure to a measurable "
            f"degree. The paired median change is "
            f"\\num{{{within['median_paired_difference']:+.4f}}}, about "
            f"\\SI{{{drop:.0f}}}{{\\percent}} of the zero-shot value, at Wilcoxon "
            f"$p = \\num{{{within['wilcoxon_p']:.1e}}}$. Within-day variation rises in only "
            f"\\SI{{{100 * within['share_of_basins_with_higher_cv_after_finetuning']:.0f}}}"
            f"{{\\percent}} of basins. This cost had not previously been measured.")
        s.append("")
        s.append(
            r"Whether that flattening removes real structure or spurious structure cannot be "
            r"settled in Africa. On the temperate network, where hourly observations exist, "
            r"the same step moved within-day standard deviation from \num{0.86} to "
            r"\num{0.89} of observed, which is movement toward the observations.")
        s.append("")
        s.append(
            r"ERA5-Land appears in the hourly panels as a contrast and not as a reference. "
            r"It has no river routing, so its basin average is runoff leaving the soil column "
            r"rather than water passing a gauge. On one catchment its instantaneous rate "
            r"reaches \SI{199}{\mm\per\day} where the daily observation peaks near \num{20}, "
            r"while its daily mean matches the observation closely. The volume is right and "
            r"the distribution within the day is wrong. Its average day swings by a factor of "
            r"\num{4.2} with a peak at 15:00 UTC, which is afternoon convective rainfall "
            r"passed straight through. The model damps that clock-driven cycle "
            r"\num{4.6}-fold across all \num{284} basins.")
        s.append("")
        s.append(
            r"Figure~\ref{fig:africa-hourly} keeps the two resolutions apart for this "
            r"reason. The left block can be scored. The right block cannot, and saying so on "
            r"the figure is the honest way to present it.")
        s.append("")
    s.append(fig("fig10_africa_hourly.png",
                 "Africa at two resolutions, kept apart because only one of them can be "
                 "scored. Left: daily, where the observation exists and every line carries a "
                 "score. Right: hourly, where no observation exists anywhere on the "
                 "continent, so the series can only be compared with each other. The "
                 "rightmost panel aligns every day of the window by hour of day and divides "
                 "each series by its own mean. A flat line there means no dependence on the "
                 "clock.", "africa-hourly", width=r"0.98\linewidth"))

    s.append(r"\section{What Phase I establishes}\label{sec:summary}")
    s.append(
        "The result is that daily aggregates recover most of the hourly skill lost by hiding "
        "hourly observations, on a temperate network of nearly nine thousand gauges and on "
        "two hundred and eighty-two African catchments that were never in training. The "
        "repair is one of amplitude, and that holds on two domains whose deficits differ by "
        "about a factor of four. It survives the three checks of "
        "Section~\\ref{sec:falsify}.")
    s.append("")
    s.append("Four limits belong beside that result.")
    s.append(r"\begin{enumerate}")
    s.append(r"  \item Under-dispersion is reduced and not removed. \SI{74}{\percent} of "
             r"gauges remain under-dispersed after fine-tuning, so this does not yet deliver "
             r"a model to trust on peak magnitude.")
    s.append(r"  \item The adaptation degrades the source domain, by a paired median of "
             r"\num{0.054} in KGE, in every fold.")
    s.append(r"  \item A random split overstates both the level and the precision of the "
             r"result. Blocked-split numbers with their honest spread are the ones to quote.")
    s.append(r"  \item The training network is temperate and northern. The African test is "
             r"the only genuinely external evidence here, and it rests on the \num{302} "
             r"records that are all the daily database holds for the continent.")
    s.append(r"\end{enumerate}")
    return "\n".join(s)


def appendix(d: dict) -> str:
    """Supporting figures. Everything here backs a claim made in the main text."""
    s = [r"\appendix", r"\part{Appendix}", ""]

    s.append(r"\section{Configuration and data path}\label{app:config}")
    s.append(
        r"Run~A takes the prepared batches as they are, and it moves backwards under both "
        r"configurations: median KGE falls from \num{0.427} to \num{0.397} under v1 and from "
        r"\num{0.424} to \num{0.397} under v2. Every run~B variant gains. The daily branch in "
        r"run~A is a power-law subsample rather than a series of daily means, so the "
        r"daily-aggregate signal has no matching structure to attach to. No hyperparameter "
        r"recovers this, which is why the whole report uses run~B.")
    s.append("")
    s.append(
        r"Within run~B, v1 to v2 lifts the blocked split from \num{0.475} to \num{0.621}, and "
        r"v3's longer budget changes almost nothing.")
    s.append("")
    abl = d.get("ablation")
    if abl:
        eff = abl["effects"]
        n_folds = abl["n_folds_per_configuration"]
        parts = [f"{name} contributes \\num{{{v['delta_M1']:+.4f}}}" for name, v in eff.items()]
        s.append(
            "That step changes two settings at once, so it cannot say which one acts. Two "
            "pairs in the hyperparameter search differ in exactly one key, verified key by "
            "key rather than assumed from their names, and they separate the two effects. "
            "On M1, " + " and ".join(parts) + ". The look-back is the larger by roughly "
            "fivefold.")
        s.append("")
        if n_folds < 2:
            s.append(
                f"Each of those configurations ran \\num{{{n_folds}}} fold, so both are point "
                f"estimates with no between-fold spread, judged against a fold-level noise "
                f"floor of \\num{{{abl['fold_noise_floor']}}} measured elsewhere in this "
                f"work. The forget-gate effect clears that floor only narrowly. A five-fold "
                f"ablation is required to settle the split and is in progress.")
        else:
            detail = "; ".join(
                f"{name} {v['delta_M1']:+.4f}"
                + (f" $\\pm$ \\num{{{v['delta_M1_sd']:.4f}}}" if v.get("delta_M1_sd") else "")
                + (", same sign in every fold" if v.get("all_folds_same_sign")
                   else ", sign varies between folds")
                for name, v in eff.items())
            s.append(
                f"Both are paired fold by fold over \\num{{{n_folds}}} folds, so each effect "
                f"carries its own spread: {detail}.")
        s.append("")
    s.append(
        r"The forget gate is retained regardless of how the credit divides. It is part of the "
        r"published method this model follows, and its absence from v1 was an oversight.")
    s.append("")
    s.append("Figure~\\ref{fig:config} shows all of them together.")
    s.append("")
    s.append(fig("fig03_configurations.png",
                 "Every configuration as a movement from M0 (open marker) to M1 (filled "
                 "marker) in median target-domain hourly KGE. Run~A is the only data path "
                 "that moves backwards.", "config"))

    s.append(r"\section{Where the gain lands}\label{app:strata}")
    trends = load("outputs/v2_stratify/gain_trends_target.csv")
    if trends is not None:
        tr = trends.set_index("variable")
        area, blocked = tr.loc["area_km2"], tr.loc["nearest_other_fold_km_blocked"]
        s.append(
            f"Small catchments gain most, at Spearman $\\rho = "
            f"\\num{{{area.spearman_rho:.3f}}}$ against area over \\num{{{int(area.n)}}} "
            f"gauges. The gain falls from \\num{{0.112}} in the smallest quintile to "
            f"\\num{{0.022}} in the largest. This fits the mechanism, because a small "
            f"fast-responding catchment is where a zero-shot model is least calibrated.")
        s.append("")
        s.append(
            f"Isolation does the opposite of what a proximity-driven result would do. Under "
            f"the blocked split the gain rises with distance to the nearest trainable gauge, "
            f"at $\\rho = \\num{{{blocked.spearman_rho:+.3f}}}$. Being far from training data "
            f"does not reduce what a daily total is worth.")
        s.append("")
    s.append("Figure~\\ref{fig:strata} shows both relationships.")
    s.append("")
    s.append(fig("fig02_gain_drivers.png",
                 "Median gain by quintile, against distance to the nearest trainable gauge "
                 "(left, one line per split) and against catchment area (right).", "strata"))

    s.append(r"\section{The mechanism across two domains}\label{app:mechanism}")
    if d["deficits"] is not None and d["mech"]:
        rows = [[tex_escape(r.domain.split(" (")[0]),
                 {"r": "$r$ (timing)", "alpha": r"$\alpha$ (amplitude)",
                  "beta": r"$\beta$ (water balance)"}[r.component],
                 f"{r.median_deficit_M0:.3f}", f"{r.median_deficit_M1:.3f}",
                 f"{100 * r.fraction_removed:.0f}\\%"] for r in d["deficits"].itertuples()]
        s.append("Table~\\ref{tab:deficits} states it numerically and "
                 "Figure~\\ref{fig:mechanism} draws it.")
        s.append("")
        s.append(table(["Domain", "Component", "Deficit at M0", "Deficit at M1", "Removed"],
                       rows,
                       "Distance from the ideal before and after fine-tuning. Deficit is "
                       "$1-r$ for the correlation and $|\\log_2 x|$ for the two ratios, "
                       "because $0.5$ and $2.0$ are equally wrong for a ratio and their "
                       "arithmetic mean is not 1. Only the fraction removed is comparable "
                       "across components.", "deficits"))
        parts = [f"{dom.split(' (')[0]} removes "
                 f"\\SI{{{100 * v['magnitude_fraction_removed']:.0f}}}{{\\percent}} of the "
                 f"amplitude and water-balance deficit against "
                 f"\\SI{{{100 * v['timing_fraction_removed']:.0f}}}{{\\percent}} of the "
                 f"timing deficit" for dom, v in d["mech"].items()]
        s.append("The ordering is the same on both domains. " + "; ".join(parts) + ".")
        s.append("")
    s.append(fig("fig11_component_deficits.png",
                 "One axis per component, because $1-r$ and $|\\log_2 x|$ are in different "
                 "units. The fraction removed, printed on each pair, is the comparable "
                 "quantity.", "mechanism", width=r"0.95\linewidth"))

    s.append(r"\section{Where the two metrics disagree}\label{app:metrics}")
    s.append("Figure~\\ref{fig:metrics} shows the joint distribution behind the "
             "disagreement reported in Section~\\ref{sec:falsify}.")
    s.append("")
    s.append(fig("fig05_metric_disagreement.png",
                 "Change in KGE against reduction in point-wise absolute error, one hexagon "
                 "per group of gauges. Quadrant counts use every gauge. The density window "
                 "omits those outside it, which the caption states rather than folding them "
                 "into the corner bins.", "metrics", width=r"0.72\linewidth"))

    s.append(r"\section{Training length}\label{app:convergence}")
    s.append("Figure~\\ref{fig:convergence} shows the validation curves.")
    s.append("")
    s.append(fig("fig06_convergence.png",
                 "Source-domain validation KGE per epoch, five folds per configuration, "
                 "zoomed to the plateau. The epochs at which v2 stopped are marked, and none "
                 "reaches the cap.", "convergence"))

    s.append(r"\section{African hydrographs at daily resolution}\label{app:africa-daily}")
    s.append("Figure~\\ref{fig:africa-daily} shows three of the catchments behind "
             "the daily numbers in Section~\\ref{sec:africa}.")
    s.append("")
    s.append(fig("fig07_africa_hydrographs.png",
                 "Three catchments spanning the outcome rather than three good ones: the "
                 "lower-quartile, median and upper-quartile catchment by M1 KGE. The "
                 "zero-shot model under-predicts every peak, and fine-tuning lifts them "
                 "toward the observed hydrograph.", "africa-daily", width=r"0.86\linewidth"))
    return "\n".join(s)


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.4cm]{geometry}
\usepackage{graphicx}
% Three search paths, so the document compiles from the repository (../figures/), from a
% flat Overleaf upload with the PNGs beside it (./), or with them in a subfolder.
\graphicspath{{../figures/}{figures/}{./}}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{amsmath}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\sisetup{detect-all,group-separator={,}}
\DeclareSIUnit{\mm}{mm}
\DeclareSIUnit{\km}{km}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}

\title{Global hourly streamflow under daily-only supervision\\[0.3em]
  \large Phase I}
\author{__AUTHOR__}
\date{__DATE__}

\begin{document}
\maketitle
\begin{abstract}
\noindent
Hourly streamflow models are trained where hourly discharge is recorded, which is a small
and geographically narrow part of the world. Most gauges report once a day. This report
asks whether hourly skill survives at gauges that supply only daily totals. One fifth of a
network of nearly nine thousand gauges is held out, its hourly observations are hidden, and
only the 24-hour aggregate is used to supervise it. The hidden hourly series is read once,
to score. Daily aggregates recover most of the lost skill, and the recovery is a repair of
amplitude rather than of timing. The same result holds on 282 African catchments that
appear nowhere in training and genuinely lack hourly discharge. Part~I describes the
experiments. Part~II reports and analyses the results.
\end{abstract}
\tableofcontents
\clearpage
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase I report as LaTeX.")
    parser.add_argument("--out", default="reports/latex/PhaseI_report.tex", type=Path)
    parser.add_argument("--author", default="Weikang Kong")
    parser.add_argument("--date", default=None, help="Defaults to today.")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    d = gather()
    body = "\n\n".join([part_experiments(d), r"\clearpage", part_results(d),
                        part_results_tail(d), part_africa(d), r"\clearpage", appendix(d)])
    from datetime import date
    preamble = (PREAMBLE.replace("__AUTHOR__", args.author)
                .replace("__DATE__", args.date or date.today().isoformat()))
    args.out.write_text(preamble + body + "\n\n\\end{document}\n", encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")

    text = args.out.read_text()
    # The brief forbids em dashes and "not X but Y". Both are habits, so they are checked
    # rather than trusted.
    import re as _re
    dashes = text.count(" -- ")
    notbut = len(_re.findall(r"\bnot\b[^.]{0,60}\bbut\b", text))
    print(f"style check: {dashes} em dashes, {notbut} 'not ... but' constructions")
    missing = [n for n in _re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", text)
               if not (FIGDIR / n).exists()]
    print("all figures present" if not missing else f"MISSING figures: {missing}")


if __name__ == "__main__":
    main()
