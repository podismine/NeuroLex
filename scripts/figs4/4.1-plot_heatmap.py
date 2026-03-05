import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox
from statsmodels.stats.multitest import multipletests

# ============================================================
# Configuration
# ============================================================
INPUT_CSV = "Disease_stats.csv"
OUTPUT_PDF = "results/figs/figs3/fig3_heatmap_FO.pdf"

# Fixed feature display order
FEATURES = [f"log_pro_BST_{i}" for i in range(12)]

# Fixed group display order; only groups present in data will be kept
ALL_GROUPS = ["ASD", "ADHD", "MDD", "SCZ", "ANX"]

COLOR_THEMES = {
    1: {
        "group_colors": {
            "ASD": "#3B5F8A",
            "ADHD": "#E68B3A",
            "MDD": "#707070",
            "SCZ": "#A65E60",
            "ANX": "#6997B6",
        },
        "heatmap_colors": ["#43A8A8", "white", "#D75F5F"],  # negative - neutral - positive
        "group_label_color": "black",
    }
}

THEME = COLOR_THEMES[1]


# ============================================================
# Utilities
# ============================================================
def add_feature_type_column(df):
    """
    Add a coarse feature-type label based on feature name prefix.
    """
    df = df.copy()
    df["fea_type"] = np.where(
        df["feature"].str.startswith("log_pro"),
        "log_pro",
        np.where(df["feature"].str.startswith("dwell"), "dwell", "other")
    )
    return df


def fdr_group(x):
    """
    Apply FDR correction within each (group, fea_type) subset.

    Original behavior preserved:
    - method='fdr_tsbh'
    """
    p = x["pval"].to_numpy()
    valid = ~np.isnan(p)

    q = np.full_like(p, np.nan, dtype=float)
    rej = np.zeros_like(valid, dtype=bool)

    if valid.sum() > 0:
        reject, qvals, _, _ = multipletests(p[valid], method="fdr_tsbh")
        q[valid] = qvals
        rej[valid] = reject

    x = x.copy()
    x["pval_fdr"] = q
    x["reject_fdr"] = rej
    return x


def make_three_color_cmap(colors3):
    """
    Create a three-anchor linear colormap.
    """
    nodes = [0.0, 0.5, 1.0]
    return LinearSegmentedColormap.from_list(
        "custom_cmap",
        list(zip(nodes, colors3))
    )


def encode_marker(pfdr, p):
    """
    Encode significance marker.

    Original logic preserved:
    - '*' if FDR-corrected p < 0.05
    - blank otherwise
    """
    if pd.notna(pfdr):
        if pfdr < 0.05:
            return "*"
    return ""


def create_correlation_circos(
    df,
    df_markers,
    targets,
    theme,
    method="Effect size",
    scheme_id=None,
    symmetric=False,
):
    """
    Create a fan-shaped polar heatmap with significance markers.
    """
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 18,
    })

    # ------------------------------------------------------------
    # Colormap and normalization
    # ------------------------------------------------------------
    heat = theme.get("heatmap_colors", "RdBu_r")
    cmap = make_three_color_cmap(heat) if isinstance(heat, (list, tuple)) else plt.get_cmap(heat)

    if symmetric:
        # Original behavior preserved: fixed symmetric range for cross-figure comparability
        norm = plt.Normalize(vmin=-0.25, vmax=0.25)
    else:
        vmin = float(np.nanmin(df.values))
        vmax = float(np.nanmax(df.values))
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax):
            vmax = 0.0
        if vmin == vmax:
            eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-3
            vmin -= eps
            vmax += eps
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # ------------------------------------------------------------
    # Polar layout geometry
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 14), subplot_kw=dict(projection="polar"))

    n_features = len(df)
    total_fan_angle_deg = 150
    total_fan_angle_rad = np.deg2rad(total_fan_angle_deg)
    start_angle = (np.pi / 2) - (total_fan_angle_rad / 2)
    end_angle = (np.pi / 2) + (total_fan_angle_rad / 2)

    theta = np.linspace(start_angle, end_angle, n_features)[::-1]
    single_bar_span = total_fan_angle_rad / max(n_features, 1)
    width = single_bar_span * 0.95

    base = 5.0
    ring_thickness = 0.9
    radii = np.arange(base, base + len(targets))

    # ------------------------------------------------------------
    # Draw rings and significance markers
    # ------------------------------------------------------------
    for i, target_name in enumerate(targets):
        r0 = radii[i]
        values = df[target_name].values

        for j, val in enumerate(values):
            angle = theta[j]

            ax.bar(
                angle,
                ring_thickness,
                width=width,
                bottom=r0,
                color=cmap(norm(val)),
                edgecolor="white",
                linewidth=0.5,
                align="center",
            )

            mark = "" if df_markers is None else str(df_markers.iloc[j, i] or "")
            if mark != "":
                mark = f"\n{mark}"

            rot = np.rad2deg(angle) - 90

            if mark.strip():
                ax.text(
                    angle,
                    r0 + ring_thickness / 2 - 0.15,
                    mark.strip(),
                    rotation=rot,
                    rotation_mode="anchor",
                    ha="center",
                    va="center",
                    fontsize=16,
                )

    # ------------------------------------------------------------
    # Feature labels
    # ------------------------------------------------------------
    label_radius = radii[-1] + 1.2
    for i, feature_name in enumerate(df.index):
        angle = theta[i]
        rot = np.rad2deg(angle) - 90

        name_str = str(feature_name)
        try:
            idx = int(name_str.split("_")[-1]) + 1
            bst_label = f"BST {idx}"
        except ValueError:
            bst_label = name_str

        ax.text(
            angle,
            label_radius,
            bst_label,
            rotation=rot,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=20,
        )

    # ------------------------------------------------------------
    # Group labels
    # NOTE:
    # These small position adjustments are manual layout tweaks preserved
    # from the original script for visual spacing only.
    # ------------------------------------------------------------
    group_colors = theme.get("group_colors", {})
    fallback_color = theme.get("group_label_color", "black")

    side_angle = end_angle + 0.10
    for i, g in enumerate(targets):
        if g == "ASD":
            r0 = radii[i] + ring_thickness / 2 - 0.2
        elif g == "ADHD":
            side_angle = side_angle - 0.01
            r0 = radii[i] + ring_thickness / 2
        elif g == "MDD":
            side_angle = side_angle - 0.01
            r0 = radii[i] + ring_thickness / 2
        else:
            r0 = radii[i] + ring_thickness / 2

        ax.text(
            side_angle + 0.07,
            r0 + 0.6,
            g,
            ha="left",
            va="center",
            color=group_colors.get(g, fallback_color),
            fontsize=18,
            fontweight="bold",
        )

    # ------------------------------------------------------------
    # Axis styling
    # ------------------------------------------------------------
    ax.set_ylim(0, label_radius + 0.8)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    # ------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------
    cax = fig.add_axes([0.36, 0.52, (0.5 - 0.36) * 2, 0.01])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(f"{method}", size=22, labelpad=-60)
    cbar.ax.tick_params(labelsize=20)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig, ax


# ============================================================
# Load data and preprocess
# ============================================================
df = pd.read_csv(INPUT_CSV)
df = add_feature_type_column(df)

df_in = (
    df.groupby(["group", "fea_type"], group_keys=False)
      .apply(fdr_group)
)

# Keep only groups that truly exist in the data
valid_groups = df_in["group"].unique().tolist()
targets = [g for g in ALL_GROUPS if g in valid_groups]

# ============================================================
# Build plotting matrices
# ============================================================
df_plot = (
    df_in.pivot_table(
        index="feature",
        columns="group",
        values="effect_size",
        aggfunc="first"
    )
    .reindex(index=FEATURES, columns=targets)
    .astype(float)
)

p_fdr = (
    df_in.pivot_table(
        index="feature",
        columns="group",
        values="pval_fdr",
        aggfunc="first"
    )
    .reindex(index=FEATURES, columns=targets)
)

p_raw = (
    df_in.pivot_table(
        index="feature",
        columns="group",
        values="pval",
        aggfunc="first"
    )
    .reindex(index=FEATURES, columns=targets)
)

df_markers = pd.DataFrame(index=FEATURES, columns=targets, data="")
for feature in FEATURES:
    for group in targets:
        df_markers.loc[feature, group] = encode_marker(
            p_fdr.loc[feature, group],
            p_raw.loc[feature, group]
        )

# ============================================================
# Plot and save
# ============================================================
fig, ax = create_correlation_circos(
    df=df_plot,
    df_markers=df_markers,
    targets=targets,
    theme=THEME,
    method="Effect size",
    scheme_id="demo",
    symmetric=True,
)

# Manual crop preserved from the original script
bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
x0, y0, width, height = bbox.bounds
new_bbox = Bbox.from_bounds(
    x0 + 2,
    y0 + height / 2 - 0.1,
    width - 4,
    height / 2 - 0.8
)

plt.savefig(OUTPUT_PDF, bbox_inches=new_bbox)
plt.show()