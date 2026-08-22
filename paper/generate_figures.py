"""Generate publication figures from the reproducible BONSAI artifacts.

The script deliberately reads benchmark JSON files rather than embedding
hand-transcribed values.  It can therefore be rerun after a new benchmark
without silently changing the plotted numbers.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = [
    ("new_bonsai_diagonal", "Diagonal"),
    ("new_bonsai_tgrsc", "TGRSC"),
    ("new_bonsai_atgtr", "ATGTR"),
    ("new_bonsai_atgfr", "ATGFR"),
]
VARIANT_ORDER = [
    ("hierarchical_plus_vib", "Diagonal"),
    ("hierarchical_tgrsc", "TGRSC"),
    ("hierarchical_atgtr", "ATGTR"),
    ("hierarchical_atgfr", "ATGFR"),
]
VARIANT_LABEL = dict(VARIANT_ORDER)
COLORS = {
    "Diagonal": "#4C78A8",
    "TGRSC": "#F58518",
    "ATGTR": "#54A24B",
    "ATGFR": "#B279A2",
}


def load_json(relative: str):
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(f"missing benchmark artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(rows: list[dict], field: str) -> tuple[float, float]:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def configure() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def figure1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 3.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")
    stages = [
        (0.2, 1.5, 1.8, 1.0, "Incoming task\n(x, y)", "#DCEAF7"),
        (2.6, 1.5, 1.8, 1.0, "VIB encoder\nqϕ(z|x)", "#E8E4F3"),
        (5.0, 1.5, 2.1, 1.0, "Task repository\nprototype · OT · TDA", "#FCE8D5"),
        (7.8, 1.5, 1.8, 1.0, "Hierarchical\nRiemannian route", "#E2F0D9"),
        (10.2, 1.5, 2.1, 1.0, "Shared adapter +\nclassifier", "#F4DDEB"),
    ]
    for x, y, w, h, label, color in stages:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.1, edgecolor="#36454F", facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", weight="bold")
    for left, right in [(2.0, 2.6), (4.4, 5.0), (7.1, 7.8), (9.6, 10.2)]:
        ax.add_patch(
            FancyArrowPatch(
                (left, 2.0), (right, 2.0), arrowstyle="-|>", mutation_scale=14,
                linewidth=1.2, color="#36454F",
            )
        )
    ax.text(6.0, 3.35, "Cached task graph: OT/TDA similarity + local SPD metric", ha="center", weight="bold")
    ax.annotate(
        "ATGFR: bounded training coreset, old logits/features, drift thermostat",
        xy=(6.0, 1.5), xytext=(6.0, 0.55), ha="center",
        arrowprops={"arrowstyle": "-|>", "color": "#B279A2", "linewidth": 1.2},
        color="#6B3D68",
    )
    ax.set_title("BONSAI modular continual-learning pipeline", pad=10, weight="bold")
    save(fig, "figure1_architecture_overview")


def figure2_new_modular() -> None:
    artifact = "results/new_version_image_comparison_5seed.json"
    if not (ROOT / artifact).exists():
        artifact = "results/new_version_image_comparison.json"
    rows = load_json(artifact)
    labels = [label for _, label in METHOD_ORDER]
    grouped = {
        label: [row for row in rows if row["method"] == key]
        for key, label in METHOD_ORDER
    }
    metrics = [
        ("task_aware_average_accuracy", "Task-aware accuracy", (0, 0.85)),
        ("forgetting", "Final forgetting", (-0.45, 0.7)),
        ("task_free_average_accuracy", "Task-free accuracy", (0, 0.85)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), sharex=True)
    x = np.arange(len(labels))
    for ax, (field, title, ylim) in zip(axes, metrics):
        means, stds = zip(*(mean_std(grouped[label], field) for label in labels))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=[COLORS[label] for label in labels], alpha=0.9)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(title, weight="bold")
        ax.set_ylim(*ylim)
        ax.set_xticks(x, labels)
        ax.grid(axis="y")
        for bar, value in zip(bars, means):
            va = "bottom" if value >= 0 else "top"
            offset = 0.02 if value >= 0 else -0.02
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.3f}", ha="center", va=va, fontsize=8)
    axes[0].set_ylabel("Mean over seeds")
    axes[1].set_ylabel("Best pre-final − final")
    fig.suptitle("New modular BONSAI on the structured image stream", weight="bold")
    fig.tight_layout()
    save(fig, "figure2_new_modular_comparison")


def figure3_robustness() -> None:
    payload = load_json("results/architecture_metrics_atgfr.json")
    rows = payload["continual_robustness_matrix"]
    cells = sorted({(int(row["num_tasks"]), int(row["input_dim"])) for row in rows})
    cell_labels = [f"{tasks} tasks\n{features} features" for tasks, features in cells]
    variants = [label for _, label in VARIANT_ORDER]
    by_cell = defaultdict(dict)
    for row in rows:
        key = (int(row["num_tasks"]), int(row["input_dim"]))
        if row["variant"] not in VARIANT_LABEL:
            continue
        label = VARIANT_LABEL[row["variant"]]
        by_cell[key][label] = row
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.0), gridspec_kw={"width_ratios": [1.45, 1]})
    x = np.arange(len(cells))
    width = 0.18
    for index, label in enumerate(variants):
        means = []
        stds = []
        for cell in cells:
            cell_rows = [
                r for r in rows
                if (int(r["num_tasks"]), int(r["input_dim"])) == cell
                and VARIANT_LABEL.get(r["variant"]) == label
            ]
            mean, std = mean_std(cell_rows, "task_aware_average_accuracy")
            means.append(mean)
            stds.append(std)
        axes[0].bar(x + (index - 1.5) * width, means, width, yerr=stds, capsize=2, label=label, color=COLORS[label])
    axes[0].set_xticks(x, cell_labels)
    axes[0].set_ylim(0, 0.42)
    axes[0].set_ylabel("Task-aware accuracy")
    axes[0].set_title("Across all task/feature cells", weight="bold")
    axes[0].legend(ncol=2, frameon=False)
    for index, label in enumerate(variants):
        values = [
            by_cell[cell][label]["consolidation_memory_fraction"]
            for cell in cells
        ]
        axes[1].plot(x, values, marker="o", linewidth=1.8, label=label, color=COLORS[label])
    axes[1].set_xticks(x, cell_labels)
    axes[1].set_ylabel("Consolidation memory / parameters")
    axes[1].set_title("Memory trade-off", weight="bold")
    axes[1].legend(frameon=False)
    fig.suptitle("Predeclared robustness grid: 2/8 tasks × 8/128 features", weight="bold")
    fig.tight_layout()
    save(fig, "figure3_robustness_grid")


def figure4_scaling() -> None:
    payload = load_json("results/architecture_metrics_atgfr.json")
    rows = payload["scaling"]
    tasks = np.asarray([row["task_count"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    axes[0].plot(tasks, [row["candidate_reduction"] for row in rows], marker="o", label="Candidate reduction")
    axes[0].plot(tasks, [row["routing_accuracy"] for row in rows], marker="s", label="Routing accuracy")
    axes[0].set_xlabel("Repository tasks")
    axes[0].set_ylabel("Fraction / accuracy")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Retrieval behavior", weight="bold")
    axes[0].legend(frameon=False)
    axes[1].plot(tasks, [row["routing_latency_ms"] for row in rows], marker="o", label="Route latency (ms)")
    axes[1].plot(tasks, [row["repository_build_ms"] for row in rows], marker="s", label="Build time (ms)")
    axes[1].set_xlabel("Repository tasks")
    axes[1].set_ylabel("Milliseconds")
    axes[1].set_title("Measured CPU cost", weight="bold")
    axes[1].legend(frameon=False)
    fig.suptitle("Repository scaling diagnostics (single seeded latent-cloud run)", weight="bold")
    fig.tight_layout()
    save(fig, "figure4_scaling")


def figure5_training_ablation() -> None:
    payload = load_json("results/architecture_metrics_atgfr.json")
    rows = payload["training_ablation"]
    labels = [label for _, label in VARIANT_ORDER]
    grouped = {
        label: [row for row in rows if row["variant"] == key]
        for key, label in VARIANT_ORDER
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharex=True)
    x = np.arange(len(labels))
    for ax, field, title, ylabel in [
        (axes[0], "task_aware_average_accuracy", "Matched training accuracy", "Accuracy"),
        (axes[1], "interference_drop_percentage_points", "Interference diagnostic", "Drop in percentage points"),
    ]:
        means, stds = zip(*(mean_std(grouped[label], field) for label in labels))
        ax.bar(x, means, yerr=stds, capsize=3, color=[COLORS[label] for label in labels])
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, weight="bold")
    fig.suptitle("Three-seed matched continual-learning ablation", weight="bold")
    fig.tight_layout()
    save(fig, "figure5_training_ablation")


def figure6_current_baselines() -> None:
    baseline_rows = load_json("results/current_backbone_baselines.json")
    primary_rows = load_json("results/new_version_image_comparison_5seed.json")
    baseline_order = [
        ("current_ewc", "EWC\ncurrent core"),
        ("current_si", "SI\ncurrent core"),
        ("current_packnet", "PackNet\ncurrent core"),
        ("current_pnn", "PNN\ncurrent core"),
    ]
    bonsai_order = [
        ("new_bonsai_diagonal", "BONSAI\ndiagonal"),
        ("new_bonsai_tgrsc", "BONSAI\nTGRSC"),
        ("new_bonsai_atgtr", "BONSAI\nATGTR"),
        ("new_bonsai_atgfr", "BONSAI\nATGFR"),
    ]
    order = baseline_order + bonsai_order
    baseline_grouped = {key: [row for row in baseline_rows if row["method"] == key] for key, _ in baseline_order}
    primary_grouped = {key: [row for row in primary_rows if row["method"] == key] for key, _ in bonsai_order}
    accuracy = []
    accuracy_std = []
    forgetting = []
    forgetting_std = []
    for key, _ in baseline_order:
        mean, std = mean_std(baseline_grouped[key], "task_aware_average_accuracy")
        accuracy.append(mean)
        accuracy_std.append(std)
        mean, std = mean_std(baseline_grouped[key], "forgetting")
        forgetting.append(mean)
        forgetting_std.append(std)
    for key, _ in bonsai_order:
        mean, std = mean_std(primary_grouped[key], "task_aware_average_accuracy")
        accuracy.append(mean)
        accuracy_std.append(std)
        mean, std = mean_std(primary_grouped[key], "forgetting")
        forgetting.append(mean)
        forgetting_std.append(std)
    labels = [label for _, label in order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.8), sharex=True)
    colors = ["#4C78A8", "#F58518", "#9D755D", "#79706E", COLORS["Diagonal"], COLORS["TGRSC"], COLORS["ATGTR"], COLORS["ATGFR"]]
    axes[0].bar(x, accuracy, yerr=accuracy_std, capsize=3, color=colors)
    axes[0].set_title("Task-aware accuracy", weight="bold")
    axes[0].set_ylabel("Mean over five seeds")
    axes[0].set_ylim(0, 1.12)
    axes[1].bar(x, forgetting, yerr=forgetting_std, capsize=3, color=colors)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Peak-before-final forgetting", weight="bold")
    axes[1].set_ylabel("Best pre-final - final")
    axes[1].set_ylim(-0.15, 1.1)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y")
    fig.suptitle("Current-core baselines versus modular BONSAI", weight="bold")
    fig.tight_layout()
    save(fig, "figure6_current_baselines")


def figure7_accuracy_memory_frontier() -> None:
    artifact = "results/new_version_image_comparison_5seed.json"
    if not (ROOT / artifact).exists():
        artifact = "results/new_version_image_comparison.json"
    rows = load_json(artifact)
    labels = [label for _, label in METHOD_ORDER]
    grouped = {
        label: [row for row in rows if row["method"] == key]
        for key, label in METHOD_ORDER
    }
    means = {label: mean_std(grouped[label], "task_aware_average_accuracy")[0] for label in labels}
    # The JSON stores total consolidation plus replay state / model parameters.
    memory = {label: mean_std(grouped[label], "consolidation_memory_fraction")[0] for label in labels}
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    for label in labels:
        xs = [row["consolidation_memory_fraction"] for row in grouped[label]]
        ys = [row["task_aware_average_accuracy"] for row in grouped[label]]
        ax.scatter(xs, ys, s=34, alpha=0.65, color=COLORS[label])
        ax.scatter([memory[label]], [means[label]], s=110, marker="X", color=COLORS[label], edgecolor="black", linewidth=0.5, label=label)
    ax.set_xlabel("Consolidation and replay memory / model parameters")
    ax.set_ylabel("Final task-aware accuracy")
    ax.set_title("Accuracy-memory frontier on the new modular stream", weight="bold")
    ax.legend(frameon=False)
    ax.grid(True, axis="both")
    fig.tight_layout()
    save(fig, "figure7_accuracy_memory_frontier")


def figure8_real_replay() -> None:
    payload = load_json("results/real_replay_cifar100_review.json")
    rows = payload["records"]
    labels = ["ER", "DER++", "ER-ACE", "ATGFR"]
    grouped = {label: [row for row in rows if row["method"] == label] for label in labels}
    colors = [COLORS["Diagonal"], COLORS["TGRSC"], "#9D755D", COLORS["ATGFR"]]
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.8))
    x = np.arange(len(labels))
    for ax, field, title, ylabel in [
        (axes[0], "average_accuracy", "CIFAR-100 accuracy", "Final global accuracy"),
        (axes[1], "forgetting", "CIFAR-100 forgetting", "Best pre-final - final"),
        (axes[2], "wall_time_seconds", "Wall-clock cost", "Seconds per run"),
    ]:
        means, stds = zip(*(mean_std(grouped[label], field) for label in labels))
        ax.bar(x, means, yerr=stds, capsize=3, color=colors)
        ax.set_xticks(x, labels)
        ax.set_title(title, weight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        if field == "forgetting":
            ax.axhline(0, color="#333333", linewidth=0.8)
    fig.suptitle("Matched real-image replay study: Split-CIFAR-100", weight="bold")
    fig.tight_layout()
    save(fig, "figure8_real_replay")


def figure9_component_ablation() -> None:
    rows = load_json("results/atgfr_component_ablation.json")
    labels = [
        "labels\nonly", "labels +\nlogits", "labels +\nfeatures", "fixed\nfull",
        "no OT", "no H0", "Euclidean\nrelation", "full\nATGFR",
    ]
    variants = [
        "labels_only", "labels_logits", "labels_features", "fixed_full",
        "no_ot", "no_tda", "euclidean_relation", "full_atgfr",
    ]
    grouped = {variant: [row for row in rows if row["variant"] == variant] for variant in variants}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8), sharex=True)
    x = np.arange(len(labels))
    for ax, field, title, ylabel in [
        (axes[0], "task_aware_average_accuracy", "Component accuracy", "Task-aware accuracy"),
        (axes[1], "forgetting", "Component forgetting", "Best pre-final - final"),
    ]:
        means, stds = zip(*(mean_std(grouped[variant], field) for variant in variants))
        ax.bar(x, means, yerr=stds, capsize=2, color="#B279A2")
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, weight="bold")
        ax.grid(axis="y")
    fig.suptitle("Predeclared ATGFR component ablation (three seeds)", weight="bold")
    fig.tight_layout()
    save(fig, "figure9_component_ablation")


def figure10_order_sensitivity() -> None:
    payload = load_json("results/real_replay_cifar100_review.json")
    rows = payload["records"]
    labels = ["ER", "DER++", "ER-ACE", "ATGFR"]
    colors = [COLORS["Diagonal"], COLORS["TGRSC"], "#9D755D", COLORS["ATGFR"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharex=True)
    orders = sorted({int(row["order_id"]) for row in rows})
    for label, color in zip(labels, colors):
        values = []
        errors = []
        for order_id in orders:
            subset = [row for row in rows if row["method"] == label and int(row["order_id"]) == order_id]
            mean, std = mean_std(subset, "average_accuracy")
            values.append(mean)
            errors.append(std)
        axes[0].errorbar(orders, values, yerr=errors, marker="o", capsize=3, label=label, color=color)
        values = []
        errors = []
        for order_id in orders:
            subset = [row for row in rows if row["method"] == label and int(row["order_id"]) == order_id]
            mean, std = mean_std(subset, "forgetting")
            values.append(mean)
            errors.append(std)
        axes[1].errorbar(orders, values, yerr=errors, marker="o", capsize=3, label=label, color=color)
    axes[0].set_title("Accuracy by task order", weight="bold")
    axes[1].set_title("Forgetting by task order", weight="bold")
    axes[0].set_ylabel("Final global accuracy")
    axes[1].set_ylabel("Best pre-final - final")
    axes[0].set_xlabel("Predeclared order index (0 = identity)")
    axes[1].set_xlabel("Predeclared order index (0 = identity)")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Task-order sensitivity on Split-CIFAR-100", weight="bold")
    fig.tight_layout()
    save(fig, "figure10_order_sensitivity")


def main() -> None:
    configure()
    figure1_architecture()
    figure2_new_modular()
    figure3_robustness()
    figure4_scaling()
    figure5_training_ablation()
    figure6_current_baselines()
    figure7_accuracy_memory_frontier()
    figure8_real_replay()
    figure9_component_ablation()
    figure10_order_sensitivity()
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
