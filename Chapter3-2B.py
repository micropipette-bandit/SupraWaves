import pandas as pd
import matplotlib.pyplot as plt

# ─── Update this to your TSV file path ──────────────────────────────────────────
tsv_path = r"C:\Users\Pietro\Desktop\Trans-QC\fastqc_per_base_sequence_quality_plot.tsv"
# ────────────────────────────────────────────────────────────────────────────────

# Load the TSV
df = pd.read_csv(tsv_path, sep='\t')

# Drop the position column (adjust name if yours differs)
for col in ['Position (bp)', 'Position']:
    if col in df.columns:
        df = df.drop(columns=[col])
        break

# Rename each of the 24 read‑set columns (R1/R2 for each of the 12 samples)
df.columns = [
    'TC1_R1', 'TC1_R2',
    'TC2_R1', 'TC2_R2',
    'TC3_R1', 'TC3_R2',
    'TB1_R1', 'TB1_R2',
    'TB2_R1', 'TB2_R2',
    'TB3_R1', 'TB3_R2',
    'C+1_R1', 'C+1_R2',
    'C+2_R1', 'C+2_R2',
    'C+3_R1', 'C+3_R2',
    'C-1_R1', 'C-1_R2',
    'C-2_R1', 'C-2_R2',
    'C-3_R1', 'C-3_R2'
]

# Prepare data
samples = df.columns.tolist()
data = [df[s].values for s in samples]

# Plot
fig, ax = plt.subplots(figsize=(len(samples)*0.35, 4))

ax.boxplot(
    data,
    positions=range(1, len(samples)+1),
    widths=0.6,
    patch_artist=True,
    showfliers=False,
    whiskerprops=dict(color='black', linewidth=1.5),
    capprops=dict(color='black', linewidth=1.5),
    medianprops=dict(color='black', linewidth=2),
    boxprops=dict(facecolor='lightgray', edgecolor='gray', linewidth=1.5, alpha=0.6),
    zorder=1
)

ax.grid(False)
ax.set_xticks(range(1, len(samples)+1))
ax.set_xticklabels(samples, fontsize=8, rotation=90, ha='center')
ax.tick_params(axis='y', labelsize=12)
ax.set_ylabel("Phred Quality Score", fontsize=14)
ax.set_title("Per-base Phred quality score distribution per sample (R1 & R2)", fontsize=16)

plt.tight_layout()
plt.show()
