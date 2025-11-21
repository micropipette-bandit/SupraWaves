import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # kept for possible legend use

# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw data
#    - "area" and "velocity" are paired measurements for the same tissues.
#    - Units for the y-axis are spatial periods in µm.
# ─────────────────────────────────────────────────────────────────────────────

area = np.array([
    203, 403, 239.75, 174, 387, 848, 265.5, 292.6666667, 509.5, 418, 232,
    206.6, 389.6666667, 249.2, 295.25, 411.3333333, 497, 542, 489.5, 295.25,
    393.3333333, 205.6666667, 271.75, 194.5
])

velocity = np.array([
    654, 380.6666667, 815.3333333, 785.3333333, 286.6666667, 320.6666667,
    789.3333333, 430.6666667, 620.6666667, 234, 445.3333333, 360.6666667,
    210.6666667, 620, 192, 365.3333333, 459.3333333, 429.3333333, 0,
    545.3333333, 0, 676.6666667, 474, 210
])

# Zeros in velocity are interpreted as "no periodicity detected" and are
# excluded from the paired analysis (converted to NaN).
velocity = np.where(velocity == 0, np.nan, velocity)

# Build a DataFrame to keep track of tissue indices and drop incomplete pairs.
df = pd.DataFrame({
    "Tissue": np.arange(1, 25),
    "Area": area,
    "Velocity": velocity
}).dropna()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Paired statistical tests
#
# Both tests compare Area vs Velocity for the same tissues:
#   - Paired t-test  : assumes approximate normality of the differences.
#   - Wilcoxon test  : non-parametric alternative (signed-rank test).
# ─────────────────────────────────────────────────────────────────────────────

t_stat, t_p = stats.ttest_rel(df["Area"], df["Velocity"])
w_stat, w_p = stats.wilcoxon(df["Area"], df["Velocity"])

print("Paired comparisons (Area vs Velocity)")
print(f"  n pairs  : {len(df)}")
print(f"  t-test   : t = {t_stat:.3f}, p = {t_p:.4f}")
print(f"  Wilcoxon : W = {w_stat:.1f}, p = {w_p:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Paired boxplot with connecting lines
#
# Area and Velocity are shown as two boxplots, with each tissue represented
# by a line connecting its Area and Velocity values. This keeps the pairing
# visible in the figure.
# ─────────────────────────────────────────────────────────────────────────────

gray_shades = ['#d0d0d0', '#a8a8a8', '#787878']  # cycling greys for individual pairs
markers = ['o']                                  # same marker for all points

fig, ax = plt.subplots(figsize=(5, 4))

A = df["Area"].values
B = df["Velocity"].values

# Boxplot configuration: no outliers shown, grey boxes, black whiskers/median.
ax.boxplot(
    [A, B],
    positions=[1, 2],
    widths=0.6,
    patch_artist=True,
    showfliers=False,
    whiskerprops=dict(color='black', linewidth=1.5),
    capprops=dict(color='black', linewidth=1.5),
    medianprops=dict(color='black', linewidth=2),
    boxprops=dict(
        facecolor='lightgray',
        edgecolor='gray',
        linewidth=1.5,
        alpha=0.6
    ),
    zorder=1
)

# Overlay one line + two points per tissue to show the pairing explicitly.
for i, (x_val, y_val) in enumerate(zip(A, B)):
    color = gray_shades[i % len(gray_shades)]
    marker = markers[i % len(markers)]

    # Line from Area (x=1) to Velocity (x=2)
    ax.plot(
        [1, 2],
        [x_val, y_val],
        color=color,
        alpha=0.5,
        linewidth=1.2,
        zorder=2
    )

    # Area point (x=1)
    ax.scatter(
        1, x_val,
        marker=marker,
        s=80,
        facecolor=color,
        edgecolors='none',
        alpha=0.5,
        zorder=3
    )

    # Velocity point (x=2)
    ax.scatter(
        2, y_val,
        marker=marker,
        s=80,
        facecolor=color,
        edgecolors='none',
        alpha=0.5,
        zorder=3
    )

# Axis labels and title
ax.grid(False)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Area', 'Velocity'], fontsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylabel("Spatial period (µm)", fontsize=16)
ax.set_title("Paired comparison: Area vs Velocity", fontsize=16)

plt.tight_layout()
plt.show()
