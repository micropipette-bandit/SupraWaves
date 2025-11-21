import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Per-sample GC content plot from FastQC "per_sequence_gc_content" export
#
# This script:
#   * reads a TSV file where each column is a sample and each row is a %GC bin,
#   * optionally renames the long sample names to shorter labels,
#   * plots all per-sequence GC distributions on the same axes.
#
# The expected input is compatible with the "fastqc_per_sequence_gc_content_plot.tsv"
# file produced by MultiQC or a similar aggregation of FastQC outputs.
# -----------------------------------------------------------------------------

# Path to the TSV file containing the per-sequence GC content distributions
tsv_path = r"C:\Users\Pietro\Desktop\Trans-QC\fastqc_per_sequence_gc_content_plot.tsv"

# -----------------------------------------------------------------------------
# 1) Load TSV file and separate GC bins from sample columns
# -----------------------------------------------------------------------------
# The first column is assumed to be "% GC" (GC bin),
# the remaining columns are one curve per sample.
df = pd.read_csv(tsv_path, sep='\t')

# GC percentage bins (x-axis)
gc_bins = df['% GC']

# All sample columns (y-values for each curve)
data = df.drop(columns=['% GC'])

# -----------------------------------------------------------------------------
# 2) Optional: rename columns to compact sample names
# -----------------------------------------------------------------------------
# This dictionary maps original, verbose sample identifiers to shorter labels
# that are easier to read in figures. Only matching keys will be renamed;
# columns not present in the map keep their original names.
rename_map = {
    'Border-Photoactivation-1_R1': 'TB1_R1',
    'Border-Photoactivation-1_R2': 'TB1_R2',
    'Border-Photoactivation-2_R1': 'TB2_R1',
    'Border-Photoactivation-2_R2': 'TB2_R2',
    'Border-Photoactivation-3_R1': 'TB3_R1',
    'Border-Photoactivation-3_R2': 'TB3_R2',

    'Central-Photoactivation-1_R1': 'TC1_R1',
    'Central-Photoactivation-1_R2': 'TC1_R2',
    'Central-Photoactivation-2_R1': 'TC2_R1',
    'Central-Photoactivation-2_R2': 'TC2_R2',
    'Central-Photoactivation-3_R1': 'TC3_R1',
    'Central-Photoactivation-3_R2': 'TC3_R2',

    'Entirely-Photoactivated-1_R1': 'C+1_R1',
    'Entirely-Photoactivated-1_R2': 'C+1_R2',
    'Entirely-Photoactivated-2_R1': 'C+2_R1',
    'Entirely-Photoactivated-2_R2': 'C+2_R2',
    'Entirely-Photoactivated-3_R1': 'C+3_R1',
    'Entirely-Photoactivated-3_R2': 'C+3_R2',

    'Control-1_R1': 'C-1_R1',
    'Control-1_R2': 'C-1_R2',
    'Control-2_R1': 'C-2_R1',
    'Control-2_R2': 'C-2_R2',
    'Control-3_R1': 'C-3_R1',
    'Control-3_R2': 'C-3_R2'
}

data = data.rename(columns=rename_map)

# -----------------------------------------------------------------------------
# 3) Plot all GC content distributions on a single figure
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))

# One line per sample / library
for col in data.columns:
    ax.plot(
        gc_bins,
        data[col],
        linewidth=1,
        alpha=0.7,
        label=col
    )

# -----------------------------------------------------------------------------
# 4) Axis formatting and labels
# -----------------------------------------------------------------------------
ax.grid(False)
ax.set_xlim(0, 100)

ax.set_xlabel('GC content (%)', fontsize=14)
ax.set_ylabel('Proportion of sequences', fontsize=14)
ax.set_title(
    'Per-sequence GC content distribution for all samples (R1 and R2)',
    fontsize=16
)
ax.tick_params(axis='both', labelsize=12)

# If the number of samples is small enough, the legend can be displayed.
# For larger panels, it is often clearer to omit it or build it manually.
# ax.legend(fontsize=8, ncol=2,
#           bbox_to_anchor=(1.05, 1),
#           loc='upper left',
#           frameon=False)

plt.tight_layout()
plt.show()
