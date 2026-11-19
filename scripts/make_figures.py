"""Publication figures and tables for the M0 confirmation and the M1 study.

Standalone on purpose: it reads only JSON/NPZ result files and imports neither
torch nor fedxmag, so it can run under any interpreter with numpy and
matplotlib (the project interpreter's BLAS build cannot render figures).

Usage::

    python scripts/make_figures.py m0 --m0-dir output/m0/xiiotid_stage1_v3b_confirmation
    python scripts/make_figures.py m1 --m1-dir output/m1/xiiotid_stage1_m1 \
        --m1-protocol configs/m1_protocol.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Reference palette (dataviz skill): first three categorical slots validate
# all-pairs in light mode; gray is recessive context, never a series identity.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CONTROL_GRAY = "#b4b2a9"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#d03b3b", "#f0efec", "#2a78d6"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

FAMILIES = ("node_timing", "node_distribution", "edge_timing", "edge_distribution")
FAMILY_LABEL = {
    "node_timing": "Node timing",
    "node_distribution": "Node distribution",
    "edge_timing": "Edge timing",
    "edge_distribution": "Edge distribution",
}
CONTROLS = ("local", "set_context", "shuffled_graph")
PREDICTOR_LABEL = {
    "constant": "Constant",
    "local": "Local",
    "set_context": "Set-context",
    "shuffled_graph": "Shuffled\ngraph",
    "graph": "Graph",
}
CONTROL_LABEL = {"local": "Local", "set_context": "Set-context", "shuffled_graph": "Shuffled graph"}

METHOD_LABEL = {
    "handcrafted_structural_lw": "Graph descriptors (MAG-ED-style) + LW",
    "raw_attribute_classical": "Raw attribute stats + empirical Σ",
    "raw_attribute_lw": "Raw attribute stats + LW",
    "handcrafted_combined_lw": "Descriptors + attribute stats + LW",
    "m0_confirmed_graph": "M0 Graph (confirmed)",
    "m0_confirmed_set_context": "M0 Set-context (confirmed)",
    "m0_confirmed_shuffled_graph": "M0 Shuffled graph (confirmed)",
    "m0_recon": "M0 recon-only (M1 schedule)",
    "m1_empirical": "M1, empirical Σ",
    "m1_fixed": "M1, fixed α = 0.5",
    "m1_lw": "M1, Ledoit–Wolf (primary)",
    "m1_oas": "M1, OAS",
    "m1_lw_no_cov_reg": "M1, LW without R_cov",
    "m1_lw_in_sample": "M1, LW in-sample energy",
    "ns_only_lw": "Mahalanobis only (no recon)",
}
METHOD_ORDER = tuple(METHOD_LABEL)


FAMILY_COLORS = (
    ("Handcrafted descriptors", ORANGE),
    ("M0 reconstruction representations", AQUA),
    ("M1 normal-support arms", BLUE),
)


def method_color(name: str) -> str:
    if name.startswith("handcrafted") or name.startswith("raw_attribute"):
        return ORANGE
    if name.startswith("m0_"):
        return AQUA
    return BLUE


def family_legend(fig: plt.Figure, names: list[str], y: float = -0.02) -> None:
    from matplotlib.lines import Line2D

    present = {method_color(name) for name in names}
    handles = [
        Line2D([], [], marker="o", linestyle="", color=color, markersize=5, label=label)
        for label, color in FAMILY_COLORS
        if color in present
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, y))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 7,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{suffix}")
    plt.close(fig)


def write_table(directory: Path, name: str, header: list[str], rows: list[list[str]],
                caption: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    markdown = [f"<!-- {caption} -->", "| " + " | ".join(header) + " |",
                "| " + " | ".join("---" for _ in header) + " |"]
    markdown += ["| " + " | ".join(row) + " |" for row in rows]
    (directory / f"{name}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    def tex(text: str) -> str:
        replacements = {"%": r"\%", "_": r"\_", "&": r"\&", "#": r"\#", "±": r"$\pm$",
                        "Σ": r"$\Sigma$", "α": r"$\alpha$", "κ": r"$\kappa$", "–": "--",
                        "≥": r"$\geq$", "≤": r"$\leq$", "✓": r"\checkmark", "✗": r"$\times$",
                        "→": r"$\rightarrow$", "Δ": r"$\Delta$"}
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.replace("**", "")

    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             rf"\caption{{{tex(caption)}}}", rf"\label{{tab:{name}}}",
             r"\begin{tabular}{" + "l" + "r" * (len(header) - 1) + "}", r"\toprule",
             " & ".join(tex(item) for item in header) + r" \\", r"\midrule"]
    lines += [" & ".join(tex(item) for item in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (directory / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


# --------------------------------------------------------------------------- M0


def m0_outputs(m0_dir: Path, figures: Path, tables: Path) -> None:
    report = json.loads((m0_dir / "confirmation_report.json").read_text(encoding="utf-8"))
    decision = report["decision"]["families"]
    seeds = [str(seed) for seed in report["model_seeds"]]
    secondary = report["secondary_descriptive"]
    per_seed = {
        predictor: [run["aggregate"] for run in runs]
        for predictor, runs in report["per_seed_runs"].items()
    }
    constant = report["constant"]["aggregate"]

    # Figure 1: family-wise control ladder.
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.25), sharey=True)
    order = ("constant", "local", "set_context", "shuffled_graph", "graph")
    for axis, family in zip(axes, FAMILIES, strict=True):
        means, colors = [], []
        for position, predictor in enumerate(order):
            if predictor == "constant":
                values = [float(constant[f"{family}_mse_mean"])]
            else:
                values = [float(item[f"{family}_mse_mean"]) for item in per_seed[predictor]]
            means.append(float(np.mean(values)))
            colors.append(BLUE if predictor == "graph" else CONTROL_GRAY)
            if len(values) > 1:
                jitter = np.linspace(-0.16, 0.16, len(values))
                axis.scatter(position + jitter, values, s=7, color=INK_2, zorder=3,
                             linewidths=0)
        axis.bar(range(len(order)), means, width=0.62, color=colors, zorder=2)
        axis.axhline(1.0, color=INK_2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        status = "pass" if decision[family]["pass"] else "FAIL"
        axis.set_title(f"{FAMILY_LABEL[family]} · {status}", color=INK, loc="left")
        axis.set_xticks(range(len(order)), ["Const.", "Local", "Set", "Shuffled", "Graph"],
                        rotation=40, ha="right", rotation_mode="anchor", fontsize=6.4)
        axis.grid(axis="x", visible=False)
        graph = means[-1]
        axis.annotate(f"{graph:.3f}", (len(order) - 1, graph), xytext=(0, 2),
                      textcoords="offset points", ha="center", va="bottom", color=INK,
                      fontsize=6.5)
    axes[0].set_ylabel("Masked MSE (standardized)")
    axes[0].annotate("MSE gate = 1.0", (0.0, 1.0), xytext=(1, 2), textcoords="offset points",
                     fontsize=6, color=INK_2, va="bottom")
    fig.tight_layout(w_pad=0.6)
    save(fig, figures, "fig_m0_family_ladder")

    # Figure 2: gains with session-cluster bootstrap intervals.
    bootstrap = secondary["session_cluster_bootstrap"]
    fig, axis = plt.subplots(figsize=(3.4, 3.0))
    labels, row = [], 0
    for family in FAMILIES:
        for control in CONTROLS:
            gain = decision[family][f"graph_gain_over_{control}"]
            interval = bootstrap[family][f"graph_gain_over_{control}"]
            axis.plot([100 * interval["lower"], 100 * interval["upper"]], [row, row],
                      color=BLUE, linewidth=1.6, solid_capstyle="round", zorder=2)
            axis.scatter([100 * gain], [row], s=18, color=BLUE, zorder=3,
                         edgecolors="white", linewidths=0.8)
            labels.append(f"{FAMILY_LABEL[family]} vs {CONTROL_LABEL[control]}")
            row += 1
        row += 0.6
    positions = []
    current = 0.0
    for _family in FAMILIES:
        positions.extend([current, current + 1, current + 2])
        current += 3.6
    axis.set_yticks(positions, labels, fontsize=6.3)
    axis.invert_yaxis()
    axis.axvline(5.0, color=INK_2, linewidth=0.8, linestyle=(0, (3, 2)))
    axis.axvline(0.0, color=AXIS, linewidth=0.8)
    axis.set_ylim(positions[-1] + 0.7, -1.2)
    axis.annotate("5% gate", (5.0, -0.9), xytext=(2, 0), textcoords="offset points",
                  fontsize=6, color=INK_2, va="center")
    axis.set_xlabel(
        "Graph MSE reduction vs control (%)\n"
        "point: gate statistic; bar: 95% date-cluster bootstrap"
    )
    axis.grid(axis="y", visible=False)
    save(fig, figures, "fig_m0_gain_intervals")

    # Figure 3: paired per-(seed, mask) margins from the per-graph errors.
    errors = np.load(m0_dir / "confirmation_per_graph_errors.npz")
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1), sharey=False)
    for column, (axis, family) in enumerate(zip(axes, FAMILIES, strict=True)):
        graph = np.nanmean(errors["graph"][..., column], axis=-1)
        for position, control in enumerate(CONTROLS):
            reference = np.nanmean(errors[control][..., column], axis=-1)
            margins = 100 * ((reference - graph) / reference).ravel()
            jitter = np.random.default_rng(position).uniform(-0.18, 0.18, margins.size)
            axis.scatter(position + jitter, margins, s=3, color=BLUE, alpha=0.45,
                         linewidths=0, zorder=3)
            axis.hlines(np.median(margins), position - 0.28, position + 0.28, color=INK,
                        linewidth=1.2, zorder=4)
        axis.axhline(5.0, color=INK_2, linewidth=0.8, linestyle=(0, (3, 2)))
        axis.axhline(0.0, color=AXIS, linewidth=0.8)
        wins = secondary["per_mask_wins"][family]
        axis.set_xticks(
            range(3),
            [f"{name}\n{wins[control]['wins']}/{wins[control]['comparisons']}"
             for name, control in zip(("Local", "Set", "Shuffled"), CONTROLS, strict=True)],
            fontsize=6.4,
        )
        axis.set_title(FAMILY_LABEL[family], loc="left", color=INK)
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("Paired MSE reduction (%)\nper (seed, mask); Graph wins below")
    fig.tight_layout(w_pad=0.6)
    save(fig, figures, "fig_m0_paired_margins")

    # Figure 4: degree-stratum gains (diverging, centered at zero).
    strata = secondary["degree_strata"]
    keys = sorted(strata["graph"], key=lambda key: (key.split("|")[0], key.split("=")[-1]))
    matrix = np.array(
        [
            [
                (strata[control][key]["mse_mean"] - strata["graph"][key]["mse_mean"])
                / strata[control][key]["mse_mean"]
                if key in strata[control]
                else np.nan
                for control in CONTROLS
            ]
            for key in keys
        ]
    )
    fig, axis = plt.subplots(figsize=(3.4, 0.18 * len(keys) + 0.8))
    limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    image = axis.imshow(100 * matrix, cmap=DIVERGING, vmin=-100 * limit, vmax=100 * limit,
                        aspect="auto")
    for (row_index, column_index), value in np.ndenumerate(matrix):
        if np.isfinite(value):
            text = f"{100 * value:.1f}" if abs(value) < 0.01 else f"{100 * value:.0f}"
            axis.text(column_index, row_index, text, ha="center", va="center",
                      fontsize=5.8, color=INK)
    names = [
        f"{key.split('|')[0].replace('_', ' ')} · deg {key.split('=')[-1]} "
        f"(n≈{strata['graph'][key]['scalar_count_mean']:.0f})"
        for key in keys
    ]
    axis.set_yticks(range(len(keys)), names, fontsize=5.8)
    axis.set_xticks(range(3), ["Local", "Set-context", "Shuffled"])
    axis.grid(False)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.05, pad=0.02)
    colorbar.set_label("Graph MSE reduction (%)", color=INK_2)
    colorbar.outline.set_visible(False)
    save(fig, figures, "fig_m0_degree_strata")

    # Tables.
    rows = []
    for family in FAMILIES:
        entry = decision[family]
        cells = [FAMILY_LABEL[family], f"{entry['constant_mse']:.3f}"]
        for predictor in (*CONTROLS, "graph"):
            values = [float(item[f"{family}_mse_mean"]) for item in per_seed[predictor]]
            cells.append(f"{np.mean(values):.3f} ± {np.std(values):.3f}")
        for control in CONTROLS:
            interval = bootstrap[family][f"graph_gain_over_{control}"]
            cells.append(
                f"{pct(entry[f'graph_gain_over_{control}'])} "
                f"[{pct(interval['lower'])}, {pct(interval['upper'])}]"
            )
        cells.append("✓ pass" if entry["pass"] else "✗ fail")
        rows.append(cells)
    write_table(
        tables,
        "table_m0_confirmation",
        ["Family", "Constant", "Local", "Set-context", "Shuffled", "Graph",
         "Gain vs Local", "Gain vs Set", "Gain vs Shuffled", "Decision"],
        rows,
        f"Independent v3b M0 confirmation on calibration-normal "
        f"({report['confirmation_normal_graphs']} windows, "
        f"{report['confirmation_sessions']} capture dates). "
        "MSE: mean ± s.d. over 5 model seeds of the 25-mask mean. Gains: frozen gate "
        "statistic [95% date-cluster bootstrap]. Overall decision: "
        f"{report['status'].upper()}.",
    )
    seed_rows = []
    for seed in seeds:
        gains = secondary["per_seed_gains"][seed]
        seed_rows.append(
            [seed]
            + [
                f"{pct(gains[family]['graph_gain_over_set_context'])} / "
                f"{pct(gains[family]['graph_gain_over_shuffled_graph'])}"
                for family in FAMILIES
            ]
        )
    write_table(
        tables,
        "table_m0_per_seed",
        ["Seed", *(FAMILY_LABEL[family] for family in FAMILIES)],
        seed_rows,
        "Per-seed Graph gain over Set-context / Shuffled graph on the confirmation partition "
        "(descriptive; the decision uses the five-seed mean).",
    )
    training_rows = []
    for seed in seeds:
        summary = report["training_summary"][seed]
        training_rows.append(
            [seed]
            + [
                f"{summary[name]['best_epoch']}/{summary[name]['epochs_run']}"
                for name in ("local", "set_context", "shuffled_graph", "graph")
            ]
        )
    write_table(
        tables,
        "table_m0_training",
        ["Seed", "Local", "Set-context", "Shuffled", "Graph"],
        training_rows,
        "Selected epoch / epochs run (select-normal early stopping, patience 15, cap 100).",
    )
    loso = secondary["leave_one_seed_out"]
    print(json.dumps({"status": report["status"],
                      "leave_one_seed_out": {k: v["status"] for k, v in loso.items()}}))


# --------------------------------------------------------------------------- M1


def _ordered(names: list[str]) -> list[str]:
    return [name for name in METHOD_ORDER if name in names] + [
        name for name in names if name not in METHOD_ORDER
    ]


def _label(name: str) -> str:
    return METHOD_LABEL.get(name, name)


def _dot_rows(axis: plt.Axes, names: list[str], values: dict[str, list[float]],
              log: bool = False) -> None:
    for row, name in enumerate(names):
        data = np.asarray(values.get(name, []), dtype=float)
        data = data[np.isfinite(data)]
        if not data.size:
            continue
        color = method_color(name)
        axis.scatter(data, np.full(data.size, row), s=9, color=color, alpha=0.55,
                     linewidths=0, zorder=3)
        axis.scatter([np.mean(data)], [row], s=26, color=color, zorder=4,
                     edgecolors="white", linewidths=0.8, marker="D")
    if log:
        axis.set_xscale("log")
    axis.set_yticks(range(len(names)), [_label(name) for name in names])
    axis.set_ylim(len(names) - 0.5, -0.5)
    axis.grid(axis="y", visible=False)


def _roc(normal: np.ndarray, attack: np.ndarray, grid: np.ndarray) -> np.ndarray:
    thresholds = np.sort(np.unique(np.concatenate((normal, attack))))[::-1]
    fpr = np.array([0.0] + [float(np.mean(normal >= t)) for t in thresholds] + [1.0])
    tpr = np.array([0.0] + [float(np.mean(attack >= t)) for t in thresholds] + [1.0])
    return np.interp(grid, fpr, tpr)


def m1_outputs(m1_dir: Path, protocol_path: Path, figures: Path, tables: Path) -> None:
    import yaml  # local import: only the M1 tables need the protocol

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    geometry = json.loads((m1_dir / "geometry_report.json").read_text(encoding="utf-8"))
    test_path = m1_dir / "test_report.json"
    test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else None
    primary = protocol["protocol"]["primary_arm"]
    methods = _ordered(list(geometry["methods"]))
    embedding_methods = [name for name in methods
                         if geometry["methods"][name]["family"] != "handcrafted"]

    # Table: pre-declared ablation matrix.
    defaults = protocol["objective_defaults"]
    matrix_rows = []
    for name, spec in protocol["baselines"]["descriptor_methods"].items():
        matrix_rows.append([_label(name), spec["descriptors"], "–", "–", "–", "–",
                            spec["estimator"].replace("_", "-")])
    for name in protocol["baselines"]["m0_confirmed_embeddings"]:
        matrix_rows.append([_label(f"m0_confirmed_{name}"), f"{name} encoder", "✓", "–", "–",
                            "–", "ledoit-wolf"])
    for arm, spec in protocol["arms"].items():
        settings = {**defaults, **(spec.get("overrides") or {})}
        uses_ns = settings["lambda_m"] > 0
        matrix_rows.append([
            _label(arm),
            "graph encoder",
            "✓" if settings["reconstruction_weight"] > 0 else "–",
            settings["estimator"].replace("_", "-") if uses_ns else "–",
            ("cross-fitted" if settings["energy_mode"] == "cross_fitted" else "in-sample")
            if uses_ns else "–",
            "✓" if settings["lambda_c"] > 0 or settings["lambda_s"] > 0 else "–",
            protocol["scoring"]["score_time_estimator"].replace("_", "-"),
        ])
    write_table(
        tables,
        "table_m1_ablation_matrix",
        ["Configuration", "Representation", "Recon", "L_NS estimator", "Energy", "R_cov",
         "Score-time Σ"],
        matrix_rows,
        "Pre-registered baseline and ablation matrix. Every configuration is scored by a "
        "Mahalanobis detector fitted on train-normal and thresholded on calibration-normal.",
    )

    # Geometry figure: covariance health and held-out stability.
    def member_values(name: str, getter: Any) -> list[float]:
        values = []
        for member in geometry["methods"][name]["members"].values():
            try:
                values.append(float(getter(member)))
            except (KeyError, TypeError):
                continue
        return values

    panels = (
        ("Effective rank\nof train Σ (raw)",
         lambda m: m["covariance"]["raw_effective_rank_entropy"], False),
        ("Condition number\nof train Σ (raw, log)",
         lambda m: m["covariance"]["raw_condition_number"], True),
        ("Held-out / train\nmean energy",
         lambda m: m["heldout_to_fit_mean_energy_ratio"], False),
        ("Select-normal FPR at\ncalibrated 5% threshold",
         lambda m: m["selection_fpr_at_calibrated"]["0.05"], False),
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 0.2 * len(embedding_methods) + 1.1),
                             sharey=True)
    for index, (axis, (title, getter, log)) in enumerate(zip(axes, panels, strict=True)):
        _dot_rows(axis, embedding_methods,
                  {name: member_values(name, getter) for name in embedding_methods}, log=log)
        axis.set_title(title, loc="left", color=INK, fontsize=7)
        if index:
            axis.tick_params(axis="y", length=0)
    axes[2].axvline(1.0, color=INK_2, linewidth=0.8, linestyle=(0, (3, 2)))
    axes[3].axvline(0.05, color=INK_2, linewidth=0.8, linestyle=(0, (3, 2)))
    fig.tight_layout(w_pad=0.4)
    family_legend(fig, embedding_methods)
    save(fig, figures, "fig_m1_geometry")

    # Representation change relative to reconstruction alone.
    change = geometry["representation_change_cka_vs_m0_recon"]
    arms = [name for name in _ordered(list(change)) if name in protocol["arms"]]
    fig, axis = plt.subplots(figsize=(3.4, 0.2 * len(arms) + 0.9))
    _dot_rows(axis, arms, {arm: list(change[arm].values()) for arm in arms})
    baseline = geometry["methods"]["m0_recon"].get("seed_stability", {}).get(
        "pairwise_selection_cka_mean"
    )
    if baseline is not None:
        axis.axvline(baseline, color=INK_2, linewidth=0.8, linestyle=(0, (3, 2)))
        axis.annotate("M0 recon-only\nseed-to-seed CKA", (baseline, -0.5), xytext=(-3, -2),
                      textcoords="offset points", fontsize=6, color=INK_2, ha="right",
                      va="top")
    axis.set_xlim(0, 1.02)
    axis.set_xlabel("Linear CKA with M0 recon-only, same seed (select-normal)")
    save(fig, figures, "fig_m1_representation_change")

    # Training trajectories.
    highlight = [("m1_lw", BLUE), ("m1_lw_no_cov_reg", ORANGE), ("m0_recon", AQUA),
                 ("ns_only_lw", YELLOW)]
    seed = str(protocol["training"]["seeds"][0])
    panels = (
        ("Shrinkage α (LW, train batches)", lambda h: h["training"].get("alpha")),
        ("Effective rank, clean select Σ",
         lambda h: h["validation"]["clean_select_geometry"].get("effective_rank_entropy")),
        ("Cross-fitted energy / d (train)", lambda h: h["training"].get("energy_over_d")),
        ("Select worst-family recon MSE", lambda h: h["validation"]["worst_family_mse"]),
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
    for axis, (title, getter) in zip(axes, panels, strict=True):
        for arm in protocol["arms"]:
            path = m1_dir / f"m1_{arm}_seed_{seed}.json"
            if not path.exists():
                continue
            history = json.loads(path.read_text(encoding="utf-8"))["history"]
            values = np.array([getter(item) if getter(item) is not None else np.nan
                               for item in history], dtype=float)
            color = dict(highlight).get(arm)
            axis.plot(np.arange(1, len(values) + 1), values,
                      color=color or CONTROL_GRAY, linewidth=1.6 if color else 0.8,
                      zorder=3 if color else 2, label=_label(arm) if color else None)
        axis.set_title(title, loc="left", color=INK, fontsize=7)
        axis.set_xlabel("Epoch")
    axes[2].set_yscale("log")
    from matplotlib.lines import Line2D

    handles, labels = axes[1].get_legend_handles_labels()
    handles.append(Line2D([], [], color=CONTROL_GRAY, linewidth=0.8))
    labels.append("Other arms (context)")
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08),
               fontsize=6.3)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6)
    save(fig, figures, "fig_m1_trajectories")

    # Geometry table.
    geometry_rows = []
    for name in methods:
        entry = geometry["methods"][name]
        members = list(entry["members"].values())

        def mean_of(getter: Any, members: list[Any] = members) -> float:
            return float(np.mean([getter(member) for member in members]))

        geometry_rows.append([
            _label(name),
            f"{mean_of(lambda m: m['covariance']['raw_effective_rank_entropy']):.1f}"
            f"/{members[0]['representation_dim']}",
            f"{mean_of(lambda m: m['covariance']['raw_condition_number']):.3g}",
            f"{mean_of(lambda m: m['covariance']['alpha']):.3f}",
            f"{mean_of(lambda m: m['split_stability']['mean_spearman']):.3f}",
            f"{mean_of(lambda m: m['split_stability']['q99_over_d_cv']):.3f}",
            f"{mean_of(lambda m: m['heldout_to_fit_mean_energy_ratio']):.2f}",
            f"{pct(mean_of(lambda m: m['selection_fpr_at_calibrated']['0.01']))} / "
            f"{pct(mean_of(lambda m: m['selection_fpr_at_calibrated']['0.05']))}",
        ])
    write_table(
        tables,
        "table_m1_geometry",
        ["Method", "r_eff / d", "κ (raw)", "Score α", "Split ρ", "q99 CV",
         "Held-out/train E", "Select FPR @1%/5%"],
        geometry_rows,
        "Normal-only geometry diagnostics (means over seeds). r_eff: entropy effective rank "
        "of the train-normal embedding covariance; split ρ: Spearman agreement of "
        "calibration energies under two disjoint train halves; select FPR: realized "
        "select-normal false-positive rate at calibration-derived thresholds.",
    )

    recon = geometry["selection_reconstruction_at_best_epoch"]
    recon_rows = []
    for arm in [name for name in _ordered(list(recon))]:
        seeds = list(recon[arm].values())
        recon_rows.append(
            [_label(arm)]
            + [f"{np.mean([item[f'{family}_mse'] for item in seeds]):.3f}"
               for family in FAMILIES]
            + [f"{np.mean([item['best_epoch'] for item in seeds]):.0f}"]
        )
    write_table(
        tables,
        "table_m1_reconstruction",
        ["Arm", *(FAMILY_LABEL[family] for family in FAMILIES), "Best epoch"],
        recon_rows,
        "Select-normal masked reconstruction MSE at the selected checkpoint (mean over seeds).",
    )

    if test is None:
        print("geometry outputs written; test report not present")
        return

    # Detection forest: AUROC and calibrated TPR with paired bootstrap intervals.
    intervals = test["paired_cluster_bootstrap"]["intervals"]
    test_methods = _ordered(list(test["methods"]))
    metrics = (("auroc", "AUROC"), ("tpr@0.01", "TPR at calibrated 1% FPR"),
               ("tpr@0.05", "TPR at calibrated 5% FPR"))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 0.2 * len(test_methods) + 0.9), sharey=True)
    for index, (axis, (metric, title)) in enumerate(zip(axes, metrics, strict=True)):
        for row, name in enumerate(test_methods):
            aggregate = test["methods"][name]["aggregate_over_members"][metric]["mean"]
            interval = intervals[name][metric]
            color = method_color(name)
            axis.plot([interval["lower"], interval["upper"]], [row, row], color=color,
                      linewidth=1.6, solid_capstyle="round", zorder=2, alpha=0.8)
            axis.scatter([aggregate], [row], s=22, color=color, zorder=3,
                         edgecolors="white", linewidths=0.8,
                         marker="D" if name == primary else "o")
        axis.set_title(title, loc="left", color=INK, fontsize=7)
        axis.set_yticks(range(len(test_methods)), [_label(name) for name in test_methods])
        axis.set_ylim(len(test_methods) - 0.5, -0.5)
        axis.grid(axis="y", visible=False)
        if index:
            axis.tick_params(axis="y", length=0)
    axes[0].axvline(0.5, color=AXIS, linewidth=0.8)
    fig.tight_layout(w_pad=0.4)
    family_legend(fig, test_methods)
    save(fig, figures, "fig_m1_detection_forest")

    # ROC curves (vertical average across seeds), linear and log-FPR.
    scores = np.load(m1_dir / "test_scores.npz")
    grid = np.concatenate(([0.0], np.logspace(-3, 0, 200)))
    curves = [("m1_lw", BLUE), ("m0_confirmed_graph", AQUA),
              ("handcrafted_combined_lw", ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for name, color in curves:
        members = list(test["methods"].get(name, {}).get("members", {}))
        if not members:
            continue
        tprs = np.mean(
            [_roc(scores[f"{name}#{member}#normal"], scores[f"{name}#{member}#attack"], grid)
             for member in members],
            axis=0,
        )
        auc = test["methods"][name]["aggregate_over_members"]["auroc"]["mean"]
        for axis in axes:
            axis.plot(grid, tprs, color=color, linewidth=1.8,
                      label=f"{_label(name)} (AUROC {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color=AXIS, linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlim(1e-3, 1)
    for target in (0.01, 0.05, 0.10):
        axes[1].axvline(target, color=INK_2, linewidth=0.6, linestyle=(0, (3, 2)))
    for axis in axes:
        axis.set_xlabel("False-positive rate on test-normal")
        axis.set_ylim(0, 1.01)
    axes[0].set_ylabel("True-positive rate on test-attack")
    axes[1].set_title("Low-FPR region (log scale)", loc="left", color=INK, fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.06),
               fontsize=6.3)
    save(fig, figures, "fig_m1_roc")

    # Per-category TPR heatmap at the calibrated 5% threshold.
    focus = [name for name in ("handcrafted_combined_lw", "raw_attribute_classical",
                               "m0_confirmed_graph", "m0_recon", "m1_lw", "m1_oas")
             if name in test["methods"]]
    categories = sorted({category for name in focus
                         for member in test["methods"][name]["members"].values()
                         for category in member["per_category_tpr"]})
    heat = np.array([
        [np.mean([member["per_category_tpr"].get(category, {}).get("tpr", np.nan)
                  for member in test["methods"][name]["members"].values()])
         for name in focus]
        for category in categories
    ])
    windows = {category: next(
        member["per_category_tpr"][category]["windows"]
        for member in test["methods"][focus[0]]["members"].values()
        if category in member["per_category_tpr"]) for category in categories}
    fig, axis = plt.subplots(figsize=(4.6, 0.24 * len(categories) + 1.2))
    image = axis.imshow(heat, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for (row_index, column_index), value in np.ndenumerate(heat):
        if np.isfinite(value):
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center",
                      fontsize=5.8, color="white" if value > 0.55 else INK)
    axis.set_yticks(range(len(categories)),
                    [f"{category} (n={windows[category]})" for category in categories])
    axis.set_xticks(range(len(focus)), [_label(name) for name in focus], rotation=30,
                    ha="right")
    axis.grid(False)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
    colorbar.set_label("TPR at calibrated 5% FPR", color=INK_2)
    colorbar.outline.set_visible(False)
    save(fig, figures, "fig_m1_category_tpr")

    # Detection table.
    detection_rows = []
    for name in test_methods:
        aggregate = test["methods"][name]["aggregate_over_members"]
        interval = intervals[name]["auroc"]
        detection_rows.append([
            _label(name),
            f"{aggregate['auroc']['mean']:.3f} ± {aggregate['auroc']['std']:.3f} "
            f"[{interval['lower']:.3f}, {interval['upper']:.3f}]",
            f"{aggregate['auprc']['mean']:.3f}",
            *(f"{pct(aggregate[f'tpr@{t}']['mean'])} ({pct(aggregate[f'fpr@{t}']['mean'])})"
              for t in ("0.01", "0.05", "0.1")),
            f"{aggregate['f1@0.05']['mean']:.3f}",
        ])
    write_table(
        tables,
        "table_m1_detection",
        ["Method", "AUROC ± s.d. [95% CI]", "AUPRC", "TPR (FPR) @1%", "TPR (FPR) @5%",
         "TPR (FPR) @10%", "F1 @5%"],
        detection_rows,
        f"Sealed test ({test['counts']['test_normal']} normal, "
        f"{test['counts']['test_attack']} attack windows). Thresholds: split-conformal on "
        "calibration-normal for target FPR; realized test-normal FPR in parentheses. "
        "CI: paired capture-date cluster bootstrap.",
    )
    difference_rows = []
    for key, entry in test["paired_cluster_bootstrap"]["differences"].items():
        left, right = key.split("__minus__")
        difference_rows.append([
            f"{_label(left)} − {_label(right)}",
            *(f"{entry[metric]['mean_difference']:+.3f} "
              f"[{entry[metric]['lower']:+.3f}, {entry[metric]['upper']:+.3f}]"
              for metric in ("auroc", "tpr@0.01", "tpr@0.05")),
        ])
    write_table(
        tables,
        "table_m1_paired_differences",
        ["Comparison", "ΔAUROC [95% CI]", "ΔTPR@1% [95% CI]", "ΔTPR@5% [95% CI]"],
        difference_rows,
        "Paired capture-date cluster bootstrap differences "
        f"({test['paired_cluster_bootstrap']['resamples_used']} resamples; seed metrics "
        "averaged within each replicate).",
    )
    replay_rows = []
    for name in test_methods:
        members = test["methods"][name]["members"]
        variants = next(iter(members.values())).get("shrinkage_replay")
        if not variants:
            continue
        for variant in variants:
            values = [member["shrinkage_replay"][variant] for member in members.values()
                      if variant in member.get("shrinkage_replay", {})]
            replay_rows.append([
                _label(name), variant.replace("_", " "),
                f"{np.mean([item['alpha'] for item in values]):.3f}",
                f"{np.mean([item['auroc'] for item in values]):.3f}",
                f"{pct(np.mean([item['at_calibrated_fpr']['0.05']['fpr'] for item in values]))}",
                f"{pct(np.mean([item['at_calibrated_fpr']['0.05']['tpr'] for item in values]))}",
            ])
    write_table(
        tables,
        "table_m1_shrinkage_replay",
        ["Method", "Score-time α", "α value", "AUROC", "Test FPR @5%", "TPR @5%"],
        replay_rows,
        "Counterfactual score-time shrinkage replay on the same frozen checkpoints.",
    )
    print("m1 outputs written")


def theory_outputs(simulation: Path, figures: Path, tables: Path) -> None:
    data = json.loads(simulation.read_text(encoding="utf-8"))
    series = (("ledoit_wolf", "Ledoit–Wolf", BLUE, "o"), ("fixed", "Fixed α = 0.5", ORANGE, "s"),
              ("oas", "OAS", AQUA, "^"), ("empirical", "Empirical (floor 1e-6)", YELLOW, "D"))
    sizes = sorted({item["reference_size"] for item in data["results"]})
    fig, axes = plt.subplots(1, len(sizes), figsize=(7.0, 2.3), sharey=False)
    for axis, size in zip(np.atleast_1d(axes), sizes, strict=True):
        rows = sorted((item for item in data["results"] if item["reference_size"] == size),
                      key=lambda item: item["effective_rank"])
        ranks = [item["effective_rank"] for item in rows]
        for key, label, color, marker in series:
            values = [item["energy_over_d"][key]["mean"] for item in rows]
            axis.plot(ranks, values, color=color, linewidth=1.6, marker=marker, markersize=3.5,
                      label=label)
        if all("empirical_in_sample" in item["energy_over_d"] for item in rows):
            axis.plot(ranks, [item["energy_over_d"]["empirical_in_sample"]["mean"]
                              for item in rows],
                      color=INK_2, linewidth=0.9, linestyle=(0, (1, 1.5)),
                      label="Empirical, in-sample")
        hotelling = data["hotelling_expectation_over_d"].get(str(size))
        if hotelling is not None:
            axis.axhline(hotelling, color=INK_2, linewidth=0.8, linestyle=(0, (3, 2)))
            axis.annotate("Hotelling constant", (ranks[-1], hotelling), xytext=(0, 3),
                          textcoords="offset points", ha="right", fontsize=6, color=INK_2)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
        axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if max(item["energy_over_d"]["empirical"]["mean"] for item in rows) < 10:
            ticks = [0.1, 0.2, 0.5, 1.0, 2.0]
            axis.set_yticks(ticks, [f"{tick:g}" for tick in ticks])
            axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axis.set_title(f"Reference size n_R = {size} (d = {data['dimension']})", loc="left",
                       color=INK, fontsize=7)
    fig.supxlabel("Effective rank of the true normal spectrum (LW and OAS overlap)",
                  fontsize=7.5, color=INK_2, y=0.06)
    np.atleast_1d(axes)[0].set_ylabel("Expected cross-fitted energy / d")
    handles, labels = np.atleast_1d(axes)[-1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.8)
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.03),
               fontsize=6.3)
    save(fig, figures, "fig_theory_ns_incentive")
    rows = []
    for size in sizes:
        subset = sorted((item for item in data["results"] if item["reference_size"] == size),
                        key=lambda item: -item["effective_rank"])
        for item in (subset[0], subset[len(subset) // 2], subset[-1]):
            rows.append([str(size), f"{item['effective_rank']:.1f}",
                         *(f"{item['energy_over_d'][key]['mean']:.3g}"
                           for key, *_ in series)])
    write_table(
        tables,
        "table_theory_ns_incentive",
        ["n_R", "True r_eff", "LW", "Fixed α=0.5", "OAS", "Empirical"],
        rows,
        f"Expected symmetric cross-fitted energy / d for Gaussian embeddings "
        f"(d = {data['dimension']}, {data['replicates']} replicates per cell). Shrinkage "
        "estimators reward lower effective rank; the empirical estimator is flat at the "
        "Hotelling constant until eigenvalues fall below the numerical floor.",
    )


def posthoc_outputs(m1_dir: Path, figures: Path, tables: Path) -> None:
    """POST HOC, EXPLORATORY figure: graph-size confounding in the sealed test."""

    diagnostics = json.loads((m1_dir / "posthoc_size_diagnostics.json").read_text("utf-8"))
    sizes = np.load(m1_dir / "posthoc_sizes.npz")
    scores = np.load(m1_dir / "test_scores.npz")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    panels = (("handcrafted_combined_lw", "deterministic", "(a) Descriptors + attributes, LW"),
              ("m1_lw", "17", "(b) M1 Ledoit–Wolf, seed 17"))
    for axis, (method, member, title) in zip(axes[:2], panels, strict=True):
        for role, color, label, alpha in (("attack", ORANGE, "test-attack", 0.30),
                                          ("normal", AQUA, "test-normal", 0.75)):
            nodes = sizes[f"test_{role}#nodes"]
            jitter = np.random.default_rng(1).uniform(-0.25, 0.25, nodes.size)
            axis.scatter(nodes + jitter, scores[f"{method}#{member}#{role}"], s=4,
                         color=color, alpha=alpha, linewidths=0, label=label)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
        axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axis.set_xlabel("Nodes in window")
        axis.set_title(title, loc="left", color=INK, fontsize=7)
    axes[0].set_ylabel("Mahalanobis energy")
    axes[0].legend(loc="upper right", markerscale=2.5, fontsize=6.2)
    names = [name for name in METHOD_ORDER if name in diagnostics["methods"]]
    for row, name in enumerate(names):
        entry = diagnostics["methods"][name]
        overall, stratified = entry["auroc_mean"], entry["size_stratified_auroc_mean"]
        color = method_color(name)
        axes[2].plot([overall, stratified], [row, row], color=color, linewidth=1.2, zorder=2)
        axes[2].scatter([overall], [row], s=16, facecolors="white", edgecolors=color,
                        linewidths=1.2, zorder=3)
        axes[2].scatter([stratified], [row], s=18, color=color, zorder=3)
    axes[2].axvline(0.5, color=AXIS, linewidth=0.8)
    axes[2].set_yticks(range(len(names)), [_label(name) for name in names], fontsize=6.2)
    axes[2].set_ylim(len(names) - 0.5, -0.5)
    axes[2].yaxis.tick_right()
    axes[2].grid(axis="y", visible=False)
    axes[2].set_xlabel("AUROC")
    axes[2].set_title("(c) ○ overall  ● within size quartiles", loc="left", color=INK,
                      fontsize=7)
    fig.tight_layout(w_pad=1.2)
    save(fig, figures, "fig_posthoc_size_confound")
    rows = [
        [_label(name),
         f"{diagnostics['methods'][name]['auroc_mean']:.3f}",
         f"{diagnostics['methods'][name]['size_stratified_auroc_mean']:.3f}",
         f"{diagnostics['methods'][name]['spearman_energy_vs_log_nodes_normal_mean']:+.2f}",
         f"{diagnostics['methods'][name]['spearman_energy_vs_log_nodes_attack_mean']:+.2f}"]
        for name in names
    ]
    write_table(
        tables,
        "table_posthoc_size_confound",
        ["Method", "AUROC", "Size-stratified AUROC", "ρ(E, log nodes) normal",
         "ρ(E, log nodes) attack"],
        rows,
        "POST HOC (exploratory, computed after the sealed test). Size-stratified AUROC is "
        "the normal×attack-weighted mean of AUROC within node-count quartiles. Size alone: "
        f"AUROC {diagnostics['size_only_auroc']['nodes']:.3f} (nodes), "
        f"{diagnostics['size_only_auroc']['edges']:.3f} (edges).",
    )


DIAGNOSIS_ROWS = (
    ("POOLED:handcrafted_combined_lw", "Descriptors + attributes (pooled, LW)", ORANGE),
    ("POOLED:m0_confirmed_graph", "M0 Graph embedding (pooled)", AQUA),
    ("POOLED:m1_lw", "M1 LW embedding (pooled)", BLUE),
    ("raw_features|node|top3_mean", "Raw node features (entity top-3)", CONTROL_GRAY),
    ("m0_confirmed_set_context|node|top3_mean", "M0 Set-context states (entity)", AQUA),
    ("m0_confirmed_shuffled_graph|node|top3_mean", "M0 Shuffled-graph states (entity)", AQUA),
    ("m0_confirmed_graph|node|top3_mean", "M0 Graph states (entity)", AQUA),
    ("m0_recon|node|top3_mean", "M0 recon-only states (entity)", AQUA),
    ("m1_lw|node|top3_mean", "M1 LW states (entity)", BLUE),
    ("entity_m1_lw|node|top3_mean", "Entity-level M1 states (entity)", BLUE),
)


def diagnosis_outputs(diagnosis: Path, figures: Path, tables: Path) -> None:
    """EXPLORATORY post-test diagnosis figure and table."""

    raw = json.loads((diagnosis / "raw_record_diagnosis.json").read_text("utf-8"))
    windows = json.loads((diagnosis / "test_band_windows.json").read_text("utf-8"))
    exploration = json.loads((diagnosis / "entity_scoring_exploration.json").read_text("utf-8"))
    headroom = json.loads((diagnosis / "flow_headroom.json").read_text("utf-8"))
    table = exploration["table"]
    fig = plt.figure(figsize=(7.0, 4.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.55, wspace=0.45)

    axis = fig.add_subplot(grid[0, 0])
    fractions = np.sort([w["attack_fraction"] for w in windows if w["attack_records"]])
    axis.plot(fractions, np.arange(1, fractions.size + 1) / fractions.size, color=BLUE,
              linewidth=1.8)
    axis.axvline(0.5, color=INK_2, linewidth=0.8, linestyle=(0, (3, 2)))
    median = float(np.median(fractions))
    axis.annotate(f"median {median:.2f}", (median, 0.5), xytext=(6, -4),
                  textcoords="offset points", fontsize=6.2, color=INK)
    axis.set_xscale("log")
    axis.set_xlabel("Attack-record fraction in an attack window")
    axis.set_ylabel("Cumulative share of windows")
    axis.set_title("(a) Label dilution", loc="left", color=INK, fontsize=7)

    axis = fig.add_subplot(grid[0, 1])
    groups = (("used_by_v3b_schema", "v3b inputs\nbytes, pkts"),
              ("flow_unused", "Flow\nports, flags"),
              ("host_telemetry", "Host\ntelemetry"),
              ("host_log_activity", "Host\nlogs"))
    record = raw["record_level_univariate_auroc"]
    for position, (group, _) in enumerate(groups):
        values = [2 * abs(v["auroc"] - 0.5) for v in record.values() if v["group"] == group]
        jitter = np.random.default_rng(position).uniform(-0.15, 0.15, len(values))
        axis.scatter(position + jitter, values, s=8, color=BLUE, alpha=0.6, linewidths=0)
        axis.hlines(max(values), position - 0.25, position + 0.25, color=INK, linewidth=1.2)
    axis.set_xticks(range(len(groups)), [label for _, label in groups], fontsize=6.0)
    axis.set_ylabel("Record-level separation\n2·|AUROC − 0.5|")
    axis.set_ylim(0, 1)
    axis.grid(axis="x", visible=False)
    axis.set_title("(b) Where record-level attack signal lives", loc="left", color=INK, fontsize=7)

    axis = fig.add_subplot(grid[1, :])
    rows = [row for row in DIAGNOSIS_ROWS if row[0] in table]
    for index, (key, _, color) in enumerate(rows):
        entry = table[key]
        axis.scatter([entry["auroc"]], [index], s=26, color=color, zorder=3,
                     edgecolors="white", linewidths=0.8, label="overall" if index == 0 else None)
        axis.scatter([entry["within_date_auroc"]], [index], s=24, facecolors="white",
                     edgecolors=color, linewidths=1.2, zorder=3,
                     label="within capture date" if index == 0 else None)
        axis.plot([entry["within_date_auroc"], entry["auroc"]], [index, index], color=color,
                  linewidth=1.0, alpha=0.6, zorder=2)
    flow_row = len(rows)
    axis.scatter([headroom["auroc_top3"]], [flow_row], s=26, color=CONTROL_GRAY, zorder=3)
    axis.scatter([headroom["within_date_auroc_top3"]], [flow_row], s=24, facecolors="white",
                 edgecolors=CONTROL_GRAY, linewidths=1.2, zorder=3)
    axis.plot([headroom["auroc_top3"], headroom["within_date_auroc_top3"]],
              [flow_row, flow_row], color=CONTROL_GRAY, linewidth=1.0)
    labels = [label for _, label, _ in rows] + ["Discarded flow fields, no learning (headroom)"]
    axis.set_yticks(range(len(labels)), labels, fontsize=6.3)
    axis.set_ylim(len(labels) - 0.5, -0.5)
    axis.axvline(0.5, color=AXIS, linewidth=0.8)
    axis.set_xlabel("Test AUROC (exploratory; ● overall, ○ within capture date)")
    axis.grid(axis="y", visible=False)
    axis.set_title("(c) Pooled graph scoring vs entity-level scoring", loc="left", color=INK,
                   fontsize=7)
    save(fig, figures, "fig_diagnosis_detection")

    header = ["Scoring", "AUROC", "Size-stratified", "Within date", "Attack fraction < 0.1",
              "TPR (FPR) @5%"]
    body = [
        [label, f"{table[key]['auroc']:.3f}", f"{table[key]['size_stratified_auroc']:.3f}",
         f"{table[key]['within_date_auroc']:.3f}", f"{table[key]['auroc[fraction<0.1]']:.3f}",
         f"{pct(table[key]['tpr@0.05'])} ({pct(table[key]['fpr@0.05'])})"]
        for key, label, _ in rows
    ]
    body.append(["Discarded flow fields, no learning (headroom)",
                 f"{headroom['auroc_top3']:.3f}", "–",
                 f"{headroom['within_date_auroc_top3']:.3f}",
                 f"{headroom['auroc_top3_fraction<0.1']:.3f}", "–"])
    write_table(
        tables, "table_diagnosis_scoring", header, body,
        "EXPLORATORY (post-test, consumed X-IIoTID test partition). Entity scoring: train-normal "
        "Ledoit-Wolf Mahalanobis on clean-view node states, graph score = mean of the top-3 "
        "node energies; calibration-normal conformal thresholds.",
    )


CONFIRMATION_ROWS = (
    ("POOLED:m0_confirmed_graph", "M0 Graph embedding, pooled (M1 scoring)"),
    ("POOLED:m1_lw", "M1 LW embedding, pooled"),
    ("POOLED:handcrafted_structural_lw", "Graph descriptors (MAG-ED-style), pooled"),
    ("POOLED:handcrafted_combined_lw", "Descriptors + attributes, pooled"),
    ("POOLED:raw_attribute_classical", "Raw attribute stats, pooled, empirical Σ"),
    ("raw_features|node|top3_mean", "Raw node features, entity"),
    ("m0_confirmed_set_context|node|top3_mean", "M0 Set-context states, entity"),
    ("m0_confirmed_shuffled_graph|node|top3_mean", "M0 Shuffled-graph states, entity"),
    ("m0_confirmed_graph|node|top3_mean", "M0 Graph states, entity (primary)"),
    ("m0_confirmed_graph|edge|top3_mean", "M0 Graph edge states, entity"),
    ("m0_recon|node|top3_mean", "M0 recon-only states, entity"),
    ("m1_lw|node|top3_mean", "M1 LW states, entity"),
    ("entity_m1_lw|node|top3_mean", "Entity-level M1 states, entity"),
)


def confirmation_color(key: str) -> str:
    if key.startswith("POOLED:handcrafted") or key.startswith("POOLED:raw"):
        return ORANGE
    if key.startswith("raw_features"):
        return CONTROL_GRAY
    if key.startswith("m1_") or key.startswith("entity_m1") or key == "POOLED:m1_lw":
        return BLUE
    return AQUA


def confirmation_outputs(report_path: Path, figures: Path, tables: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    table, intervals = report["table"], report["bootstrap"]["intervals"]
    rows = [(key, label) for key, label in CONFIRMATION_ROWS if key in table]
    primary = "m0_confirmed_graph|node|top3_mean"
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 0.22 * len(rows) + 1.0), sharey=True,
                             gridspec_kw={"width_ratios": [1.2, 1, 1]})
    panels = (("auroc", "AUROC (● all, ○ within date)"), ("tpr@0.01", "TPR at calibrated 1% FPR"),
              ("tpr@0.05", "TPR at calibrated 5% FPR"))
    for index, (axis, (metric, title)) in enumerate(zip(axes, panels, strict=True)):
        for row, (key, _) in enumerate(rows):
            color = confirmation_color(key)
            interval = intervals[key][metric]
            axis.plot([interval["lower"], interval["upper"]], [row, row], color=color,
                      linewidth=1.6, alpha=0.8, solid_capstyle="round", zorder=2)
            axis.scatter([table[key][metric]], [row], s=24, color=color, zorder=3,
                         edgecolors="white", linewidths=0.8, marker="D" if key == primary else "o")
            within = table[key]["within_date_auroc"]
            if metric == "auroc" and within == within:
                axis.scatter([within], [row], s=20, facecolors="white",
                             edgecolors=color, linewidths=1.1, zorder=3)
        axis.set_title(title, loc="left", color=INK, fontsize=7)
        axis.set_yticks(range(len(rows)), [label for _, label in rows], fontsize=6.3)
        axis.set_ylim(len(rows) - 0.5, -0.5)
        axis.grid(axis="y", visible=False)
        if index:
            axis.tick_params(axis="y", length=0)
    axes[0].axvline(0.5, color=AXIS, linewidth=0.8)
    fig.tight_layout(w_pad=0.4)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", linestyle="", color=c, markersize=5, label=label)
               for label, c in (("Handcrafted, pooled", ORANGE), ("M0 encoders", AQUA),
                                ("M1 encoders", BLUE), ("Raw entity features", CONTROL_GRAY))]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    save(fig, figures, "fig_confirmation_entity_scoring")

    body = []
    for key, label in rows:
        entry = table[key]
        body.append([
            label,
            f"{entry['auroc']:.3f} [{intervals[key]['auroc']['lower']:.3f}, "
            f"{intervals[key]['auroc']['upper']:.3f}]",
            f"{entry['size_stratified_auroc']:.3f}",
            f"{entry['within_date_auroc']:.3f}",
            f"{entry['auroc[fraction<0.1]']:.3f}",
            *(f"{pct(entry[f'tpr@{t}'])} ({pct(entry[f'fpr@{t}'])})"
              for t in ("0.01", "0.05", "0.1")),
            "–" if entry["localization_top1"] != entry["localization_top1"]
            else pct(entry["localization_top1"]),
        ])
    write_table(
        tables, "table_confirmation_entity_scoring",
        ["Scoring", "AUROC [95% CI]", "Size-strat.", "Within date", "Fraction<0.1",
         "TPR (FPR) @1%", "TPR (FPR) @5%", "TPR (FPR) @10%", "Top-1 node is attack"],
        body,
        f"Pre-registered audit-band confirmation ({report['counts']['confirm_normal']} normal, "
        f"{report['counts']['confirm_attack']} attack windows never scored before). Thresholds "
        "split-conformal on calibration-normal; date-cluster bootstrap intervals.",
    )
    hypothesis_rows = [
        [name.split("_", 1)[0] + " " + name.split("_", 1)[1].replace("_", " "),
         f"{entry['mean_difference']:+.3f} [{entry['lower']:+.3f}, {entry['upper']:+.3f}]",
         entry["expected"], "✓ supported" if entry["supported"] else "✗ not supported"]
        for name, entry in report["hypotheses"].items()
    ]
    write_table(tables, "table_confirmation_hypotheses",
                ["Hypothesis", "ΔAUROC [95% CI]", "Pre-declared", "Outcome"], hypothesis_rows,
                "Pre-registered hypotheses on the audit band.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("m0", "m1", "theory", "posthoc", "diagnosis",
                                          "confirmation"))
    parser.add_argument("--simulation", type=Path,
                        default=PROJECT_ROOT / "output" / "m1" / "simulation_ns_incentive.json")
    parser.add_argument("--m0-dir", type=Path,
                        default=PROJECT_ROOT / "output" / "m0" / "xiiotid_stage1_v3b_confirmation")
    parser.add_argument("--m1-dir", type=Path,
                        default=PROJECT_ROOT / "output" / "m1" / "xiiotid_stage1_m1")
    parser.add_argument("--m1-protocol", type=Path,
                        default=PROJECT_ROOT / "configs" / "m1_protocol.yaml")
    parser.add_argument("--figures", type=Path, default=PROJECT_ROOT / "output" / "figures")
    parser.add_argument("--tables", type=Path, default=PROJECT_ROOT / "output" / "tables")
    args = parser.parse_args()
    style()
    if args.stage == "m0":
        m0_outputs(args.m0_dir, args.figures / "m0", args.tables)
    elif args.stage == "theory":
        theory_outputs(args.simulation, args.figures / "theory", args.tables)
    elif args.stage == "posthoc":
        posthoc_outputs(args.m1_dir, args.figures / "posthoc", args.tables)
    elif args.stage == "confirmation":
        confirmation_outputs(
            PROJECT_ROOT / "output" / "entity_confirmation" / "confirmation_report.json",
            args.figures / "confirmation", args.tables,
        )
    elif args.stage == "diagnosis":
        diagnosis_outputs(PROJECT_ROOT / "output" / "diagnosis", args.figures / "diagnosis",
                          args.tables)
    else:
        m1_outputs(args.m1_dir, args.m1_protocol, args.figures / "m1", args.tables)


if __name__ == "__main__":
    main()
