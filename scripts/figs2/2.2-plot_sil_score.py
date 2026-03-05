import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


# =========================
# Configuration
# =========================
INPUT_PKL = "all_res_ari_sil_icc.pkl"
OUTPUT_FIG = "figs1/fig1_sil_site5.pdf"

# Display order of methods in the final figure
# NOTE: this is a manual ordering for visualization/legend consistency
METHOD_ORDER = [2, 0, 1, 3]

# Legend labels corresponding to METHOD_ORDER above
METHOD_NAMES = ["GaussianHMM", "EiDA", "LEiDA", "NeuroLex"]

# Color palette for the displayed methods
COLOR_LIST = ["#3CB7CC", "#6AC4B0", "#C4937C", "#CC5A5A"]

# Candidate site labels; only the first `num_sites` will be used
SITE_NAMES = [
    "NKIRS_Siemens",
    "HBN-Site-RU",
    "SRPBS-OPEN-KUT",
    "SRPBS_1600-HUH",
    "ds000030",
    "ABIDE-NYU",
]


# =========================
# Load results
# =========================
with open(INPUT_PKL, "rb") as f:
    all_res = pickle.load(f)

# Select top-5 indices based on descending order of `all_res[-1]['len']`
# NOTE: this is a result selection rule and should be documented in the README
use_max = np.argsort(all_res[-1]["len"])[::-1][:5]


# =========================
# Collect plotting data
# =========================
final_all_res = []
for idx in METHOD_ORDER:
    res = all_res[idx]
    sub_score = []
    for ff in use_max:
        sub_score.append(res["all_sil"][ff])
    final_all_res.append(sub_score)

num_methods = len(final_all_res)
num_sites = len(final_all_res[0])

# Keep only as many colors as needed
color_list = COLOR_LIST[:num_methods]


# =========================
# Determine y-range
# =========================
all_scores = np.concatenate([np.concatenate(method_scores) for method_scores in final_all_res])
ymin, ymax = np.percentile(all_scores, [1, 99])
y_margin = 0.03 * (ymax - ymin)
ymin -= y_margin
ymax += y_margin


# =========================
# Create figure layout
# =========================
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.4)

fig = plt.figure(figsize=(9, 3.5))
gs = gridspec.GridSpec(1, 2, width_ratios=[12, 1], wspace=0.05)

ax_main = fig.add_subplot(gs[0])
ax_kde = fig.add_subplot(gs[1], sharey=ax_main)

positions = np.arange(num_sites)
offset = 0.18


# =========================
# Main panel: violin + box + scatter + mean
# =========================
for m in range(num_methods):
    data = final_all_res[m]
    x_base = positions + (m - (num_methods - 1) / 2) * offset

    # Violin plot
    parts = ax_main.violinplot(
        data,
        positions=x_base,
        widths=0.15,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(color_list[m])
        pc.set_edgecolor("none")
        pc.set_alpha(0.10)

    # Boxplot
    bp = ax_main.boxplot(
        [np.asarray(v) for v in data],
        positions=x_base,
        widths=0.09,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        manage_ticks=False,
        zorder=5,
    )
    for box in bp["boxes"]:
        box.set(facecolor=(1, 1, 1, 0), edgecolor="grey", linewidth=1.3)
    for whisker in bp["whiskers"]:
        whisker.set(color="grey", linewidth=1.1, alpha=0.6)
    for cap in bp["caps"]:
        cap.set(color="grey", linewidth=1.1, alpha=0.6)
    for median in bp["medians"]:
        median.set(color="grey", linewidth=1.2, alpha=0.6)

    for s_idx, vals in enumerate(data):
        vals = np.asarray(vals)
        jitter = np.random.uniform(-0.04, 0.04, size=len(vals))
        ax_main.scatter(
            np.ones_like(vals) * x_base[s_idx] + jitter,
            vals,
            s=8,
            alpha=0.75,
            color=color_list[m],
            edgecolors="none",
            zorder=4,
        )

    # Mean marker
    for s_idx, vals in enumerate(data):
        vals = np.asarray(vals)
        mu = float(np.mean(vals))
        x = float(x_base[s_idx])
        ax_main.scatter(
            [x],
            [mu],
            s=42,
            facecolors="white",
            edgecolors="black",
            linewidths=1.2,
            zorder=6,
        )


ax_main.set_ylim(0, ymax)
ax_main.set_xlim(-0.5, 4.5)

ax_main.set_ylabel("Silhouette Score", fontsize=18)
ax_main.set_xticks(positions)
ax_main.set_xticklabels(
    [SITE_NAMES[i] for i in range(num_sites)],
    rotation=30,
    fontsize=14,
)
ax_main.tick_params(axis="y", labelsize=14)

for spine in ["top", "right"]:
    ax_main.spines[spine].set_visible(False)
ax_main.grid(axis="y", alpha=0.15)

handles = [
    ax_main.scatter([], [], color=color_list[m], label=f"Method {m + 1}")
    for m in range(num_methods)
]
ax_main.legend(
    handles,
    METHOD_NAMES,
    frameon=False,
    ncol=4,
    fontsize=16,
    loc=(-0.1, 1),
)


for m in range(num_methods):
    vals = np.concatenate(final_all_res[m])
    sns.kdeplot(
        y=vals,
        ax=ax_kde,
        lw=2,
        color=color_list[m],
        fill=False,
        clip=(ymin, ymax),
    )
    med = np.median(vals)
    ax_kde.axhline(med, ls="--", lw=1.3, color=color_list[m])

ax_kde.set_xticks([])
ax_kde.set_xlabel("")
ax_kde.tick_params(axis="y", left=False, labelleft=False)
ax_kde.set_xlim(0, 10)

for spine in ["top", "right", "bottom"]:
    ax_kde.spines[spine].set_visible(False)

ax_kde.spines["right"].set_linewidth(1.0)
ax_kde.spines["right"].set_alpha(0.4)


os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, bbox_inches="tight")
plt.show()