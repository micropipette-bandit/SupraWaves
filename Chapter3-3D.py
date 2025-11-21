#!/usr/bin/env python3
"""
B&W clustered heatmap of mean nuclear area vs. time, with solid-black dendrogram.

Two modes:

1) Cached mode (recommended for sharing / publication)
   ---------------------------------------------------
   If a file named 'nuclear_area_matrix.csv' is present in the same directory
   as this script, it is loaded and used as the input matrix. No access to
   the original '*-sorted.csv' files is required.

   The CSV is expected to have:
     - the time axis as its index (e.g. "Time (min)"),
     - one column per sample (mean nuclear area per timepoint).

2) Raw-data mode (for internal use / cache generation)
   ----------------------------------------------------
   If 'nuclear_area_matrix.csv' is not found, the script:
     - scans `base_dir` for '*-sorted.csv',
     - builds time series of mean nuclear area per file,
     - aligns them on a common time axis (interpolation + ffill/bfill),
     - drops constant/NaN series,
     - saves the resulting matrix to 'nuclear_area_matrix.csv' next to
       this script,
     - performs hierarchical clustering and draws the dendrogram + heatmap.
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

# ─────────────────────────────────────────────────────────────────────────────
# 1. Settings for raw-data mode
# ─────────────────────────────────────────────────────────────────────────────
# These are only used when the cache file is NOT found.
base_dir          = r"C:\Users\Genesis\Desktop\Nuclei-2"
file_glob         = "*-sorted.csv"
minutes_per_frame = 10
time_unit_label   = "min"

# Colormap: grayscale with black = high area, white = low area.
cmap = "gray_r"

# Name of the cached matrix file (stored next to this script).
CACHE_FILENAME = "nuclear_area_matrix.csv"


def script_dir() -> str:
    """
    Return the directory containing this script.

    Falls back to the current working directory in environments where
    __file__ is not defined.
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def load_or_build_matrix():
    """
    Load the nuclear area matrix from cache if available.

    If 'nuclear_area_matrix.csv' exists in the same directory as this script,
    it is read and returned as a DataFrame.

    Otherwise, the function:
      - scans `base_dir` for '*-sorted.csv',
      - builds mean AREA vs time for each file,
      - aligns all series on a common time axis,
      - filters out degenerate series,
      - saves the result to 'nuclear_area_matrix.csv',
      - returns the resulting DataFrame.

    Returns
    -------
    df_all : pandas.DataFrame
        Rows are timepoints, columns are samples. Values are mean nuclear
        area at each timepoint.
    """
    here = script_dir()
    cache_path = os.path.join(here, CACHE_FILENAME)

    # Cached path: use it if it exists.
    if os.path.exists(cache_path):
        print(f"Loading cached matrix from: {cache_path}")
        df_all = pd.read_csv(cache_path, index_col=0)
        # Keep index name consistent with plotting labels if present.
        if df_all.index.name is None:
            df_all.index.name = f"Time ({time_unit_label})"
        return df_all

    # No cache present: build matrix from raw '*-sorted.csv' files.
    print("No cached matrix found, building from raw CSV files...")
    print(f"Scanning directory: {base_dir}")

    paths = sorted(glob.glob(os.path.join(base_dir, file_glob)))
    if not paths:
        raise FileNotFoundError(f"No files matching '{file_glob}' in {base_dir}")

    series = {}
    all_t = set()

    for p in paths:
        # Label: strip extension and '-sorted' suffix.
        label = os.path.splitext(os.path.basename(p))[0].replace("-sorted", "")
        df = pd.read_csv(p)

        # Require at least FRAME and AREA columns.
        if not {"FRAME", "AREA"}.issubset(df.columns):
            continue

        # Build time axis in the chosen unit (default: minutes).
        df["TIME"] = df["FRAME"] * minutes_per_frame

        # Mean AREA per timepoint.
        ts = (
            df.groupby("TIME")["AREA"]
            .mean()
            .sort_index()
        )

        series[label] = ts
        all_t.update(ts.index)

    if not series:
        raise RuntimeError("No usable time series found (missing FRAME/AREA columns?).")

    # Common time axis covering the full range of all series.
    # The number of points is equal to the number of distinct time values found.
    common = np.linspace(min(all_t), max(all_t), len(all_t))
    common.sort()
    idx = pd.Index(common, name=f"Time ({time_unit_label})")

    # Reindex each series on the common time axis and interpolate missing values.
    df_all = pd.DataFrame(
        {
            lbl: s.reindex(idx).interpolate().ffill().bfill()
            for lbl, s in series.items()
        }
    )

    # Drop series that are constant or contain NaNs after alignment.
    keep = [
        c for c in df_all
        if df_all[c].std() > 0 and not df_all[c].isna().any()
    ]
    df_all = df_all[keep]

    if df_all.empty:
        raise RuntimeError("All aligned series were constant or contained NaNs.")

    # Save to CSV next to the script so that future runs do not need raw CSVs.
    df_all.to_csv(cache_path)
    print(f"Saved aligned nuclear area matrix to: {cache_path}")

    return df_all


def main():
    # Load or construct the (time × sample) area matrix.
    df_all = load_or_build_matrix()

    # ─────────────────────────────────────────────────────────────────────
    # 2. Hierarchical clustering across samples
    # ─────────────────────────────────────────────────────────────────────
    # Distance matrix between samples (columns) using correlation distance.
    D = pdist(df_all.values.T, metric="correlation")

    # Average-linkage clustering.
    Z = linkage(D, method="average")

    # Obtain the order of leaves (sample ordering along y-axis).
    ddata = dendrogram(Z, no_plot=True)
    leaves = ddata["leaves"]
    ordered_labels = [df_all.columns[i] for i in leaves]

    # Reorder columns according to the dendrogram leaves.
    clustered = df_all[ordered_labels]

    # ─────────────────────────────────────────────────────────────────────
    # 3. Plot: dendrogram (left) + heatmap (right)
    # ─────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(5, 4))

    # Dendrogram axes on the left
    ax1 = fig.add_axes([0.05, 0.1, 0.09, 0.8])
    dendrogram(
        Z,
        labels=ordered_labels,
        orientation="left",
        link_color_func=lambda k: "k",  # all branches in black
        color_threshold=None,
        ax=ax1
    )

    # Remove all spines and ticks on the dendrogram for a clean look.
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Heatmap axes on the right
    ax2 = fig.add_axes([0.25, 0.1, 0.7, 0.8])
    im = ax2.imshow(
        clustered.values.T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[
            clustered.index.min(),
            clustered.index.max(),
            0,
            clustered.shape[1],
        ],
    )

    # One row label per sample, centered within each heatmap row.
    ax2.set_yticks(np.arange(clustered.shape[1]) + 0.5)
    ax2.set_yticklabels(ordered_labels, fontsize=10)

    ax2.set_xlabel(f"Time ({time_unit_label})", fontsize=14)
    ax2.set_ylabel("Samples (clustered)", fontsize=14)

    # Increase tick label size
    ax2.tick_params(axis="x", labelsize=12)
    ax2.tick_params(axis="y", labelsize=10)

    # Colorbar: label on the right-hand side, in µm².
    cbar = fig.colorbar(
        im,
        ax=ax2,
        label="Area (µm²)",
        fraction=0.046,
        pad=0.04
    )
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.tick_right()
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("area µm²", fontsize=14)

    plt.show()


if __name__ == "__main__":
    main()
