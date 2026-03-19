import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
 
# -----------------------------
# Font
# -----------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
 
# -----------------------------
# Models & Categories
# -----------------------------
models = ["DUET-LNP", "LiON", "LANTERN", "Baseline*"]
categories = ["Transfection", "Toxicity"]
 
x = np.array([0, 0.82])  # slightly tighter gap between groups
n = len(models)
width = 0.18
offsets = np.linspace(-(n - 1) / 2 * width, (n - 1) / 2 * width, n)
 
# -----------------------------
# Data: [[transfection, toxicity], ...] per model
# -----------------------------
pearson_vals = [
    [0.5,  0.75],   # DUET-LNP
    [0.45, 0.72],   # LiON
    [0.40, 0.70],   # LANTERN
    [0.38, 0.65],   # Baseline*
]
pearson_errs = [
    [0.02, 0.03],
    [0.03, 0.02],
    [0.02, 0.06],
    [0.03, 0.04],
]
 
spearman_vals = [
    [0.44, 0.61],
    [0.46, 0.58],
    [0.43, 0.58],
    [0.41, 0.53],
]
spearman_errs = [
    [0.02, 0.03],
    [0.02, 0.02],
    [0.03, 0.03],
    [0.03, 0.04],
]
 
mse_vals = [
    [0.98,  0.006],
    [1.20,  0.006],
    [1.1,  0.007],
    [1.16,  0.007],
]
mse_errs = [
    [0.03,  0.0003],
    [0.1,  0.0001],
    [0.05,  0.0002],
    [0.04,  0.0003],
]
 
r2_vals = [
    [0.18, 0.55],
    [0.14, 0.48],
    [0.15, 0.50],
    [0.11, 0.45],
]
r2_errs = [
    [0.03, 0.04],
    [0.03, 0.04],
    [0.04, 0.05],
    [0.05, 0.05],
]
 
# -----------------------------
# Colors per model (bottom → top gradient)
# -----------------------------
edge_color = "#1B3F8B"
model_colors = [
    ("#1B3F8B", "#2C6BED"),   # DUET-LNP  — navy/blue (unchanged)
    ("#2F7F8F", "#9CCED6"),   # LiON      — light cyan-blue
    ("#5A6FCC", "#AAB8F5"),   # LANTERN   — light periwinkle
    ("#8CACED", "#D7E1F5"),   # Baseline* — lightest
]
 
# -----------------------------
# Gradient Helper
# -----------------------------
def apply_gradient(ax, bars, color_bottom, color_top):
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    cmap = LinearSegmentedColormap.from_list("grad", [color_bottom, color_top])
    for bar in bars:
        x_bar, y_bar = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        im = ax.imshow(
            gradient,
            extent=[x_bar, x_bar + w, y_bar, y_bar + h],
            aspect="auto",
            cmap=cmap,
            origin="lower",
            zorder=bar.get_zorder() - 0.1,
        )
        im.set_clip_path(bar)
        bar.set_facecolor((0, 0, 0, 0))
 
# -----------------------------
# Plot Function
# -----------------------------
def create_chart(all_vals, all_errs, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
 
    for i, (vals, errs, (c_bot, c_top)) in enumerate(
        zip(all_vals, all_errs, model_colors)
    ):
        bars = ax.bar(
            x + offsets[i], vals, width,
            yerr=errs, capsize=5,
            edgecolor=edge_color, linewidth=1.4,
            error_kw=dict(lw=1.8, capthick=1.8, zorder=3),
        )
        apply_gradient(ax, bars, c_bot, c_top)
 
    # -------------------------
    # Y-axis limits
    # -------------------------
    max_val = max(
        v + e
        for vals, errs in zip(all_vals, all_errs)
        for v, e in zip(vals, errs)
    )
    ax.set_ylim(0, max_val * 1.45)
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
 
    # -------------------------
    # Axes labels / ticks
    # -------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=22)
 
    # -------------------------
    # Legend — all 4 models, DUET-LNP bold
    # -------------------------
    legend_elements = [
        Patch(facecolor=c_top, edgecolor=edge_color, label=model)
        for model, (_, c_top) in zip(models, model_colors)
    ]
    leg = ax.legend(
        handles=legend_elements,
        fontsize=14,
        loc="upper right",
        ncol=2,
        fancybox=True,
        framealpha=0.85,
        facecolor="white",
        edgecolor="#CCCCCC",
        borderpad=0.8,
    )
    for text in leg.get_texts():
        if text.get_text() == "DUET-LNP":
            text.set_fontweight("bold")
 
    # -------------------------
    # Value labels
    # -------------------------
    for i, (vals, errs) in enumerate(zip(all_vals, all_errs)):
        for j, (v, e) in enumerate(zip(vals, errs)):
            ax.text(
                x[j] + offsets[i],
                v + e + max_val * 0.022,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=10,
            )
 
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
 
    plt.tight_layout()
    plt.savefig(filename, dpi=600, transparent=True)
    plt.close()
    print(f"Saved {filename}")
 
def create_noise_reduction_chart(before_vals, after_vals, metric_names, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(metric_names))
    width = 0.32

    # -----------------------------
    # Colors (BEFORE vs AFTER)
    # -----------------------------
    before_color = ("#8CACED", "#D7E1F5")  # light (before)
    after_color  = ("#1B3F8B", "#2C6BED")  # strong blue (after)

    # -----------------------------
    # Bars
    # -----------------------------
    bars_before = ax.bar(
        x - width/2, before_vals, width,
        edgecolor=edge_color, linewidth=1.4,
    )
    bars_after = ax.bar(
        x + width/2, after_vals, width,
        edgecolor=edge_color, linewidth=1.4,
    )

    # Apply gradients (assuming this is your custom function)
    apply_gradient(ax, bars_before, before_color[0], before_color[1])
    apply_gradient(ax, bars_after, after_color[0], after_color[1])

    # -----------------------------
    # Axes scaling & Margins (THE FIX)
    # -----------------------------
    max_val = max(max(before_vals), max(after_vals))
    ax.set_ylim(0, max_val * 1.4)
    
    # Create a dynamic buffer based on the bar width to pad the left and right sides
    x_buffer = width * 2 
    ax.set_xlim(x[0] - x_buffer, x[-1] + x_buffer)

    # -----------------------------
    # Labels
    # -----------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=20)

    # -----------------------------
    # Correct Legend
    # -----------------------------
    legend_elements = [
        Patch(facecolor=before_color[1], edgecolor=edge_color, label="Before Noise Reduction"),
        Patch(facecolor=after_color[1], edgecolor=edge_color, label="After Noise Reduction"),
    ]

    ax.legend(
        handles=legend_elements,
        fontsize=13,
        loc="upper left",
        fancybox=True,
        framealpha=0.85,
        facecolor="white",
        edgecolor="#CCCCCC",
    )

    # -----------------------------
    # Value labels
    # -----------------------------
    for bars in [bars_before, bars_after]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h + max_val * 0.02,
                f"{h:.2f}",
                ha="center", va="bottom", fontsize=10
            )

    # -----------------------------
    # Styling
    # -----------------------------
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=600, transparent=True)
    plt.close()

    print(f"Saved {filename}")

# -----------------------------
# Generate Charts
# -----------------------------
# create_chart(pearson_vals,  pearson_errs,  "Pearson Correlation",  "Pearson r",  "pearson_chart.png")
# create_chart(spearman_vals, spearman_errs, "Spearman Correlation", "Spearman ρ", "spearman_chart.png")
# create_chart(mse_vals,      mse_errs,      "Mean Squared Error",   "MSE",        "mse_chart.png")
# create_chart(r2_vals,       r2_errs,       "R² Score",             "R²",         "r2_chart.png")
 
metrics = ["Pearson R", "Spearman", "R²"]

before = [0.43, 0.45, 0.12]   # BEFORE noise reduction (example)
after  = [0.50, 0.44, 0.18]   # AFTER noise reduction (example)

create_noise_reduction_chart(
    before,
    after,
    metrics,
    "Noise Reduction on Model Performance",
    "Score",
    "noise_reduction_chart.png"
)
print("All charts generated.")