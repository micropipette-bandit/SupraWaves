import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# 1) Load the data
# -----------------------------------------------------------------------------
# Expected Excel structure:
#   - Column "Exp"   : experiment identifier (e.g. 1, 2, 3)
#   - Column "Light" : illumination condition ("BF", "PC", "UV")
#   - Column "Period": temporal period of oscillations
#   - Column "Lambda": spatial period of oscillations
#
# Only the listed columns are imported; everything else in the sheet is ignored.
df = pd.read_excel(
    r"C:/Users/Pietro/Desktop/PIV-UV/PIV-comparison.xlsx",
    sheet_name="Sheet1",
    usecols=['Exp', 'Light', 'Period', 'Lambda']
)

# -----------------------------------------------------------------------------
# 2) Harmonise labels and define PC → UV ordering
# -----------------------------------------------------------------------------
# For the analysis, "BF" is treated as "PC" (phase contrast). This recoding
# ensures that PC and UV comparisons are consistent across experiments.
df['Light'] = df['Light'].replace({'BF': 'PC'})

# Convert "Light" to a categorical variable with a fixed order: PC first, UV second.
# This ordering is used later when building PC–UV pairs and boxplots.
df['Light'] = pd.Categorical(
    df['Light'],
    categories=['PC', 'UV'],
    ordered=True
)

# -----------------------------------------------------------------------------
# 3) Summary statistics (mean ± SEM) per experiment and light condition
# -----------------------------------------------------------------------------
def sem(x: pd.Series) -> float:
    """Standard error of the mean for a pandas Series."""
    return x.std(ddof=1) / np.sqrt(x.count())

# Group by experiment and light condition, then compute mean and SEM
summary = (
    df
    .groupby(['Exp', 'Light'])
    .agg(
        Period_mean=('Period', 'mean'),
        Period_sem=('Period', sem),
        Lambda_mean=('Lambda', 'mean'),
        Lambda_sem=('Lambda', sem)
    )
    .reset_index()
)

print("=== Summary: mean ± SEM by Experiment & Lighting ===")
for _, row in summary.iterrows():
    print(
        f"Exp {row.Exp} [{row.Light}]: "
        f"Period = {row.Period_mean:.2f} ± {row.Period_sem:.2f} h;  "
        f"Lambda = {row.Lambda_mean:.1f} ± {row.Lambda_sem:.1f} µm"
    )

# -----------------------------------------------------------------------------
# 4) Build PC–UV pairs for paired statistics
# -----------------------------------------------------------------------------
# The goal is to form paired observations (PC_i, UV_i) across conditions,
# respecting both the experiment identity and the pairing within each experiment.
#
# collect_pairs('Period') returns two arrays:
#   - all PC values pooled across experiments
#   - all corresponding UV values, in the same order
#
# Any NaN pair (PC or UV missing) is skipped.

def collect_pairs(var: str):
    """
    Collect PC–UV pairs for a given variable.

    Parameters
    ----------
    var : {"Period", "Lambda"}
        Name of the column to pair.

    Returns
    -------
    a_list, b_list : np.ndarray, np.ndarray
        Arrays of PC and UV values matched pairwise across all experiments.
    """
    a_list, b_list = [], []
    for exp in df['Exp'].unique():
        sub = df[df['Exp'] == exp]

        # Values for this experiment and condition
        pc = sub[sub['Light'] == 'PC'][var].values
        uv = sub[sub['Light'] == 'UV'][var].values

        # Paired design requires equal number of PC and UV entries
        if len(pc) != len(uv):
            raise ValueError(f"Exp {exp}: {len(pc)} PC vs {len(uv)} UV entries")

        for v1, v2 in zip(pc, uv):
            if np.isfinite(v1) and np.isfinite(v2):
                a_list.append(v1)
                b_list.append(v2)

    return np.array(a_list), np.array(b_list)

# Construct paired datasets for the two variables of interest
period_pc, period_uv = collect_pairs('Period')
lambda_pc, lambda_uv = collect_pairs('Lambda')

# -----------------------------------------------------------------------------
# 5) Paired t-tests (PC vs UV)
# -----------------------------------------------------------------------------
# Paired t-test is used here because each PC value is matched to a UV value
# from the same experiment and position. If fewer than 2 pairs exist, the test
# is not meaningful and NaN is returned.

if len(period_pc) >= 2:
    t_p, p_p = ttest_rel(period_pc, period_uv)
else:
    t_p = p_p = np.nan

if len(lambda_pc) >= 2:
    t_l, p_l = ttest_rel(lambda_pc, lambda_uv)
else:
    t_l = p_l = np.nan

print("\n=== Paired t-test results ===")
print(f"Period:   t = {t_p:.3f}, p = {p_p:.4f}")
print(f"Lambda:   t = {t_l:.3f}, p = {p_l:.4f}")

# -----------------------------------------------------------------------------
# 6) Visualisation: boxplots + paired overlays for PC vs UV
# -----------------------------------------------------------------------------
# For each variable (Period, Lambda), a two-panel figure is created:
#   - boxplot of pooled PC and UV values,
#   - overlaid lines connecting paired values within each experiment,
#   - points coloured/marked by experiment.
#
# PC values are plotted at x=1, UV values at x=2.

fig, axes = plt.subplots(1, 2, figsize=(10, 6))

# Different markers and grey levels are used to distinguish experiments visually.
markers = ['o', 's', 'D']
gray_shades = ['dimgray', 'gray', 'lightgray']

# Configuration for the two panels
plot_specs = [
    ('Period', period_pc, period_uv, 'Time period (h)',  p_p, 'Time period: PC vs UV'),
    ('Lambda', lambda_pc, lambda_uv, 'Space period (µm)',     p_l, 'Space period: PC vs UV')
]

for ax, (var, A, B, ylabel, pval, title) in zip(axes, plot_specs):
    # Boxplot for pooled PC and UV values
    ax.boxplot(
        [A, B],
        positions=[1, 2],
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        medianprops=dict(color='black', linewidth=2),
        boxprops=dict(facecolor='lightgray', edgecolor='gray',
                      linewidth=1.5, alpha=0.6),
        zorder=1
    )

    # Overlay paired values, grouped by experiment
    for i, exp in enumerate(df['Exp'].unique()):
        sub = df[df['Exp'] == exp]
        v_pc = sub[sub['Light'] == 'PC'][var].values
        v_uv = sub[sub['Light'] == 'UV'][var].values
        color = gray_shades[i % len(gray_shades)]

        for x_val, y_val in zip(v_pc, v_uv):
            if not (np.isfinite(x_val) and np.isfinite(y_val)):
                continue

            # Line joining PC and UV for a single pair
            ax.plot([1, 2], [x_val, y_val],
                    color=color, alpha=0.5, linewidth=1.2, zorder=2)

            # Individual points for PC and UV
            ax.scatter(1, x_val,
                       marker=markers[i % len(markers)], s=80,
                       facecolor=color, edgecolors='none',
                       alpha=0.5, zorder=3)
            ax.scatter(2, y_val,
                       marker=markers[i % len(markers)], s=80,
                       facecolor=color, edgecolors='none',
                       alpha=0.5, zorder=3)

    # Axes formatting
    ax.grid(False)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['PC', 'UV'], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16)

    # Annotate p-value above the boxplots, if available
    if np.isfinite(pval) and len(A) > 0 and len(B) > 0:
        y_min = min(A.min(), B.min())
        y_max = max(A.max(), B.max())
        y_text = y_max + 0.05 * (y_max - y_min)
        ax.text(1.5, y_text, f"p = {pval:.3g}",
                ha='center', va='bottom', fontsize=12)
    else:
        ax.text(1.5, 0.95, "p = n/a",
                ha='center', transform=ax.transAxes, fontsize=12)

# -----------------------------------------------------------------------------
# 7) Legend: experiments as separate markers (no box edges)
# -----------------------------------------------------------------------------
exp_list = df['Exp'].unique()
handles, labels = [], []
for i, exp in enumerate(exp_list):
    # Dummy Line2D used only for the legend entry of each experiment
    h = Line2D(
        [0], [0],
        marker=markers[i % len(markers)],
        color='none',
        markerfacecolor=gray_shades[i % len(gray_shades)],
        markeredgecolor='none',
        markersize=8,
        alpha=0.5
    )
    handles.append(h)
    labels.append(f'Exp {exp}')

axes[0].legend(
    handles, labels,
    title='Experiment',
    loc='upper center',
    ncol=len(handles),
    bbox_to_anchor=(0.5, -0.15),
    fontsize=12
)

plt.tight_layout()
plt.show()
