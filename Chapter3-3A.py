#!/usr/bin/env python3
"""
AREA autocorrelation peak picker with JSON export
-------------------------------------------------

Two usage modes:

1) JSON-only mode
   ------------------------------------------------
   If a file named 'manual_peaks.json' is present in the same directory as
   this script, it is loaded and the script:
     * prints per-file peak positions and spacings (Δt, Δx),
     * prints global mean ± SEM for temporal and spatial spacings.

   In this mode, no CSV files are required and no interactive clicking occurs.
   This is the mode that can be used by readers to reproduce the reported
   summary numbers directly from the JSON file.

2) Interactive mode (for generating the JSON)
   ------------------------------------------------
   If 'manual_peaks.json' is not found, the script:
     * scans a folder for '*-sorted.csv' files,
     * builds AREA kymographs (time × position),
     * computes temporal and spatial autocorrelations,
     * applies light smoothing (temporal) and heavy smoothing (spatial),
     * opens an interactive plot where peaks on the smoothed curve are
       selected by mouse clicks,
     * saves all picked peaks and derived statistics to 'manual_peaks.json'
       in the same directory as this script,
     * prints per-file and global statistics.

The JSON produced in mode (2) is exactly what is consumed in mode (1).
"""

import os
import glob
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, median_filter

# ─── SMOOTHING PARAMETERS ─────────────────────────────────────────────────────
# Temporal autocorrelation is only lightly smoothed; spatial autocorrelation
# is smoothed much more aggressively.
TEMP_ROLL, TEMP_SIG = 5,   1     # temporal AC  → rolling window + Gaussian
SPAT_ROLL, SPAT_SIG = 301, 35    # spatial  AC  → median + Gaussian
# ──────────────────────────────────────────────────────────────────────────────


def script_dir() -> str:
    """
    Return the directory where this script lives.

    Falls back to the current working directory if __file__ is not defined
    (e.g. in some interactive environments).
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def normalize(ac: np.ndarray) -> np.ndarray:
    """
    Normalize an autocorrelation array so that R(0) = 1.

    Parameters
    ----------
    ac : np.ndarray
        Raw autocorrelation, such that ac[0] is the zero-lag value.

    Returns
    -------
    np.ndarray
        Normalized autocorrelation.
    """
    ac = ac.astype(float)
    return ac / ac[0] if ac[0] != 0 else ac


def pick_peaks(lags, raw, sm, xlabel, title, is_spatial=False):
    """
    Plot raw and smoothed autocorrelation and let the user click on peaks.

    Parameters
    ----------
    lags : array-like
        Lag values on the x-axis (time or position).
    raw : array-like
        Raw autocorrelation values.
    sm : array-like
        Smoothed autocorrelation values.
    xlabel : str
        Label for the x-axis.
    title : str
        Plot title.
    is_spatial : bool
        If True, the x-axis is interpreted as spatial (µm) and tick marks
        are spaced accordingly.

    Returns
    -------
    list of float
        Sorted list of clicked peak positions (in the same units as `lags`).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lags, raw, color="gray", alpha=0.4, linewidth=2, label="Raw")
    ax.plot(lags, sm,  color="black", linewidth=2, label="Smoothed")

    ax.set_xlim(0, lags.max())
    ax.set_ylim(np.min(raw[1:]), np.max(raw[1:]))

    if is_spatial:
        max_x = lags.max()
        ax.set_xticks([0, max_x * 0.25, max_x * 0.5, max_x * 0.75, max_x])

    ax.set_title(title, fontsize=28)
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel("Autocorrelation", fontsize=22)
    ax.tick_params(labelsize=18)
    ax.legend(fontsize=18)
    plt.tight_layout()

    # User clicks on the peaks of interest; press Enter to finish.
    pts = plt.ginput(n=-1, timeout=-1, show_clicks=True)
    peaks = np.sort([p[0] for p in pts])

    # Report the picked peaks in the terminal.
    unit = "µm" if is_spatial else "h"
    print(f"\n{title} — picked peaks:")
    for pk in peaks:
        print(f"  • {pk:.3f} {unit}")

    # Mark the peaks on the smoothed curve for visual feedback.
    ax.scatter(
        peaks,
        np.interp(peaks, lags, sm),
        color="red",
        s=50,
        zorder=5,
        label="Peaks"
    )
    plt.draw()
    plt.pause(0.5)
    plt.close(fig)
    return peaks.tolist()


def sem(a):
    """
    Standard error of the mean (SEM) for a 1D array.

    Returns 0.0 when fewer than two finite values are present.
    """
    a = np.array(a, float)
    a = a[np.isfinite(a)]
    return np.nanstd(a, ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0


def summarize_from_json(json_path: str):
    """
    Load manual peaks from a JSON file and print per-file and global statistics.

    This is the mode intended for publication: given a 'manual_peaks.json'
    file, it reproduces all summary numbers without re-processing the raw data.
    """
    with open(json_path, "r") as jf:
        all_picks = json.load(jf)

    print(f"Using manual peaks from: {json_path}")

    global_dts = []
    global_dxs = []

    # Per-file report
    for name in sorted(all_picks):
        rec = all_picks[name]

        t_peaks = rec.get("temporal_peaks_h", [])
        t_deltas = rec.get("temporal_deltas_h", [])
        mt_mean = rec.get("mean_temporal_h", float("nan"))
        mt_sem = rec.get("sem_temporal_h", float("nan"))

        x_peaks = rec.get("spatial_peaks_um", [])
        x_deltas = rec.get("spatial_deltas_um", [])
        mx_mean = rec.get("mean_spatial_um", float("nan"))
        mx_sem = rec.get("sem_spatial_um", float("nan"))

        print(f"\n=== {name} ===")
        print(" temporal peaks (h):", t_peaks)
        print(
            "   deltas →", t_deltas,
            f"mean ± SEM = {mt_mean:.3f} ± {mt_sem:.3f} h"
        )
        print(" spatial peaks (µm):", x_peaks)
        print(
            "   deltas →", x_deltas,
            f"mean ± SEM = {mx_mean:.1f} ± {mx_sem:.1f} µm"
        )

        global_dts.extend(t_deltas)
        global_dxs.extend(x_deltas)

    # Global (all-files) summary
    if global_dts:
        g_mean_t = float(np.nanmean(global_dts))
        g_sem_t = sem(global_dts)
        print(
            f"\nOverall temporal: mean = {g_mean_t:.3f} h  "
            f"± {g_sem_t:.3f} h"
        )

    if global_dxs:
        g_mean_x = float(np.nanmean(global_dxs))
        g_sem_x = sem(global_dxs)
        print(
            f"Overall spatial : mean = {g_mean_x:.1f} µm "
            f"± {g_sem_x:.1f} µm"
        )


def main():
    # Directory where this script resides (used for manual_peaks.json)
    here = script_dir()
    json_path = os.path.join(here, "manual_peaks.json")

    # If manual_peaks.json is present, run in JSON-only mode and exit.
    if os.path.exists(json_path):
        summarize_from_json(json_path)
        return

    # If no JSON is present, fall back to interactive mode using CSV files.
    # Update 'base' to the directory containing your '*-sorted.csv' files.
    base = r"C:\Users\Pietro\Desktop\Nuclei-2"
    files = sorted(glob.glob(os.path.join(base, "*-sorted.csv")))

    if not files:
        print("No 'manual_peaks.json' found and no '*-sorted.csv' files in:", base)
        sys.exit(1)

    plt.ion()
    all_picks = {}
    global_dts = []
    global_dxs = []

    for fp in files:
        name = os.path.basename(fp)[:-4]

        df = pd.read_csv(fp)
        if not {"FRAME", "AREA"}.issubset(df.columns):
            continue

        # Identify the column that encodes spatial position
        pos_col = next((c for c in df if "POSITION" in c.upper()), None)
        if pos_col is None:
            continue

        # ------------------------------------------------------------------
        # Build AREA kymograph: time × position grid
        # ------------------------------------------------------------------
        # TIME_H is computed assuming 10 minutes per frame.
        df["TIME_H"] = df["FRAME"] * (10 / 60.0)

        kymo = (
            df.pivot_table(
                index="TIME_H",
                columns=pos_col,
                values="AREA",
                aggfunc="mean"
            )
            .sort_index()
            .sort_index(axis=1)
        )

        times = kymo.index.values
        poss = kymo.columns.values

        dt = np.diff(times).mean()  # time step in hours
        dx = np.diff(poss).mean()   # spatial step in µm

        # Subtract the global mean and replace NaN with 0 for correlation
        mat = np.nan_to_num(kymo.values - np.nanmean(kymo.values))

        # ------------------------------------------------------------------
        # Temporal autocorrelation of AREA (averaged over positions)
        # ------------------------------------------------------------------
        ms = mat.mean(axis=1)
        ac_t = normalize(
            np.correlate(ms, ms, mode="full")[len(ms) - 1:]
        )

        # Light smoothing: rolling mean + small Gaussian filter.
        ac_t_sm = gaussian_filter1d(
            pd.Series(ac_t)
            .rolling(TEMP_ROLL, center=True, min_periods=1)
            .mean()
            .to_numpy(),
            sigma=TEMP_SIG
        )

        lag_t = np.arange(ac_t.size) * dt

        peaks_t = pick_peaks(
            lag_t,
            ac_t,
            ac_t_sm,
            xlabel="Time lag (h)",
            title=f"{name} – Temporal AC",
            is_spatial=False
        )

        # ------------------------------------------------------------------
        # Spatial autocorrelation of AREA (averaged over time)
        # ------------------------------------------------------------------
        mt = mat.mean(axis=0)
        ac_x = normalize(
            np.correlate(mt, mt, mode="full")[len(mt) - 1:]
        )

        # Heavy smoothing: median filter (to remove small-scale noise)
        # followed by a wide Gaussian.
        ac_x_sm = median_filter(ac_x, size=SPAT_ROLL, mode="nearest")
        ac_x_sm = gaussian_filter1d(ac_x_sm, sigma=SPAT_SIG)

        lag_x = np.arange(ac_x.size) * dx

        peaks_x = pick_peaks(
            lag_x,
            ac_x,
            ac_x_sm,
            xlabel="Position lag (µm)",
            title=f"{name} – Spatial AC",
            is_spatial=True
        )

        # ------------------------------------------------------------------
        # Peak spacings and summary statistics for this file
        # ------------------------------------------------------------------
        dts = np.diff(peaks_t).tolist() if len(peaks_t) > 1 else []
        dxs = np.diff(peaks_x).tolist() if len(peaks_x) > 1 else []

        mt_mean, mt_sem = (np.nanmean(dts), sem(dts)) if dts else (np.nan, np.nan)
        mx_mean, mx_sem = (np.nanmean(dxs), sem(dxs)) if dxs else (np.nan, np.nan)

        all_picks[name] = {
            "temporal_peaks_h": peaks_t,
            "temporal_deltas_h": dts,
            "mean_temporal_h": mt_mean,
            "sem_temporal_h": mt_sem,
            "spatial_peaks_um": peaks_x,
            "spatial_deltas_um": dxs,
            "mean_spatial_um": mx_mean,
            "sem_spatial_um": mx_sem
        }

        global_dts.extend(dts)
        global_dxs.extend(dxs)

    # ----------------------------------------------------------------------
    # Save the manual peaks to 'manual_peaks.json' next to this script
    # ----------------------------------------------------------------------
    with open(json_path, "w") as jf:
        json.dump(all_picks, jf, indent=2)
    print("\nSaved manual peaks to", json_path)

    # Print the same per-file and global summary as in JSON-only mode
    summarize_from_json(json_path)


if __name__ == "__main__":
    main()
