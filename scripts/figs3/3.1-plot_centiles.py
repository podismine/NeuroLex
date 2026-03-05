import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rpy2.robjects as ro

from matplotlib.colors import LinearSegmentedColormap
from scipy.special import expit

from gamlss_python.gamlss_main import Gamlss
from utils.utils_plot import plot_centiles


# ============================================================
# Global settings
# ============================================================
warnings.filterwarnings("ignore")

NUM_TOKEN = 12
INPUT_CSV = "gather_tokens.csv"
MODEL_DIR = "model/gamlss_model"
OUTPUT_DIR = "results/figs/figs3/centiles"

CENTILES = [5, 10, 25, 50, 75, 90, 95]

# Fixed token-to-BST display mapping used in the manuscript
MAP_NAME = {
    0: 2, 1: 1, 2: 9, 3: 10,
    4: 8, 5: 7, 6: 6, 7: 3,
    8: 5, 9: 11, 10: 0, 11: 4,
}

# Colormap for centile bands
CMAP = LinearSegmentedColormap.from_list(
    "teal_light_smooth",
    ["#E9FBFB", "#D1F5F5", "#A5E4E4", "#74C9C9", "#43A8A8"],
    N=256
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================
def make_x_piecewise(xmin=5, xmax=88):
    """
    Create the plotting x-grid used for lifespan centile visualization.
    """
    x1 = np.arange(xmin, min(25, xmax) + 1e-9, 1.0)
    x2 = np.arange(max(25, xmin), xmax + 1e-9, 1.0)
    x = np.unique(np.concatenate([x1, x2]))
    x = x[(x >= xmin) & (x <= xmax)]
    return x


def resample_centiles(ret, x_new):
    """
    Resample centile curves onto a new x-grid using 1D linear interpolation.
    """
    x_old = np.asarray(ret["x"], dtype=float)
    y_old = np.asarray(ret["centile"], dtype=float)  # shape: (N, C)

    order = np.argsort(x_old)
    x_old = x_old[order]
    y_old = y_old[order]

    uniq_x, uniq_idx = np.unique(x_old, return_index=True)
    x_old = uniq_x
    y_old = y_old[uniq_idx]

    x_new = np.asarray(x_new, dtype=float)
    x_new = x_new[(x_new >= x_old.min()) & (x_new <= x_old.max())]

    y_new = np.empty((x_new.shape[0], y_old.shape[1]), dtype=float)
    for j in range(y_old.shape[1]):
        y_new[:, j] = np.interp(x_new, x_old, y_old[:, j])

    out = dict(ret)
    out["x"] = x_new
    out["centile"] = y_new
    return out


def set_child_emphasis_axis(ax, use_symlog=True):
    """
    Apply the custom lifespan x-axis scaling and tick settings.
    """
    if use_symlog:
        ax.set_xscale("symlog", linthresh=12, linscale=0.4)

    ticks = [5, 10, 15, 20, 25, 45, 85]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])


def add_reference_vlines(ax, xs=(25, 45)):
    """
    Draw dotted vertical reference lines at selected ages.
    """
    for xx in xs:
        ax.axvline(
            x=xx,
            linestyle=":",
            linewidth=2.0,
            color="black",
            alpha=0.8,
            zorder=4
        )


def transform_if_needed(values, fea):
    """
    Apply inverse-logit transform for log-scale FO outputs.
    """
    return expit(values) if "log" in fea else values


def get_ylabel(fea):
    """
    Return axis label for the current feature type.
    """
    return "FO" if "log" in fea else "DT"


# ============================================================
# Load data
# ============================================================
xdf = pd.read_csv(INPUT_CSV)
hc_df = xdf[xdf["group"].isin(["train"])].reset_index(drop=True)
test_hc_df = xdf[xdf["group"].isin(["test"])].reset_index(drop=True)  # kept for parity with original script


# ============================================================
# Main plotting loop
# ============================================================
for fea in ["log_pro", "dwell"]:
    for token in range(NUM_TOKEN):
        bst_name = MAP_NAME[token] + 1

        # ------------------------------------------------------------
        # Load model
        # ------------------------------------------------------------
        model_path = f"{MODEL_DIR}/model_{fea}_{token}.rds"
        model = Gamlss.load_model(model_path)

        # Legacy bridge line: preserves original behavior for R-side state
        ro.r(f"Gamlss_{fea}_{token} <- loaded_model")

        # ------------------------------------------------------------
        # Compute overall and sex-stratified centiles
        # ------------------------------------------------------------
        ret_sex = plot_centiles(
            model,
            hc_df,
            hc_df,
            site_correction=True,
            plot_all_data=True,
            variables_to_split={"sex": [0, 1]},
            show_fig=False,
            centiles=CENTILES
        )

        ret = plot_centiles(
            model,
            hc_df,
            hc_df,
            site_correction=True,
            plot_all_data=True,
            show_fig=False,
            centiles=CENTILES
        )

        # ------------------------------------------------------------
        # Figure 1: overall centile bands + median
        # ------------------------------------------------------------
        plt.clf()
        plt.figure(figsize=(7, 5))

        x_new = make_x_piecewise(
            xmin=float(ret["x"].min()),
            xmax=float(ret["x"].max())
        )
        ret_plot = resample_centiles(ret, x_new)

        x = ret_plot["x"]
        centile = ret_plot["centile"]

        centile_len = centile.shape[1]
        mid_idx = centile_len // 2
        colors = [CMAP(v) for v in np.linspace(0.25, 0.75, mid_idx)]

        for i in range(mid_idx):
            lower = transform_if_needed(centile[:, i], fea)
            upper = transform_if_needed(centile[:, -(i + 1)], fea)

            rel = i / (mid_idx - 1) if mid_idx > 1 else 1.0
            alpha = 0.4 + 0.2 * rel

            plt.fill_between(
                x,
                lower,
                upper,
                color=colors[i],
                alpha=alpha,
                linewidth=0,
                zorder=1
            )

        plt.plot(
            x,
            transform_if_needed(centile[:, mid_idx], fea),
            color="black",
            linewidth=3.2,
            alpha=0.95,
            zorder=3
        )

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.8)
        ax.spines["bottom"].set_linewidth(1.8)

        set_child_emphasis_axis(ax, use_symlog=True)
        add_reference_vlines(ax, xs=(25, 45))

        plt.ylabel(get_ylabel(fea), labelpad=8, fontsize=26)

        y_min = transform_if_needed(centile[:, 0], fea).min()
        y_max = transform_if_needed(centile[:, -1], fea).max()
        plt.ylim(y_min - 0.03, y_max + 0.03)

        plt.xlim(4.9, 88)
        plt.title(f"BST {bst_name}", pad=10)

        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/fig3_nm_centiles_{fea}_{bst_name}.pdf",
            bbox_inches="tight"
        )

        # ------------------------------------------------------------
        # Figure 2: sex-specific median curves
        # Assumption preserved from original code:
        #   ret_sex[1] -> Female
        #   ret_sex[0] -> Male
        # ------------------------------------------------------------
        plt.clf()
        plt.figure(figsize=(7, 5))

        sex_f = resample_centiles(ret_sex[1], x_new)
        sex_m = resample_centiles(ret_sex[0], x_new)

        y_f = transform_if_needed(sex_f["centile"][:, mid_idx], fea)
        y_m = transform_if_needed(sex_m["centile"][:, mid_idx], fea)

        plt.plot(x_new, y_f, color="#D75F5F", linewidth=3.5, label="Female")
        plt.plot(x_new, y_m, color="#43A8A8", linewidth=3.5, label="Male")

        ax = plt.gca()
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        set_child_emphasis_axis(ax, use_symlog=True)
        add_reference_vlines(ax, xs=(25, 45))

        plt.title(f"BST {bst_name}", pad=10)
        plt.xlim(4.9, 88)
        plt.ylim(y_min - 0.03, y_max + 0.03)

        leg = plt.legend(title="Sex               ", fontsize=18, title_fontsize=18)
        leg.get_frame().set_facecolor("none")
        leg.get_frame().set_edgecolor("none")

        plt.ylabel(get_ylabel(fea), labelpad=8, fontsize=26)

        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/fig3_nm_sex_{fea}_{bst_name}.pdf",
            bbox_inches="tight"
        )

        plt.close("all")