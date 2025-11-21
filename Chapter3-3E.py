"""
Peak spacing analysis for MT2A intensity profiles along bands.

Two usage modes
===============

1) JSON-only mode (recommended for sharing / publication)
   ------------------------------------------------------
   If a file named 'manual_peaks_MT2A.json' is present in the same
   directory as this script, the script:

     - loads the manually picked peak positions for each band,
     - computes peak-to-peak spacings (Δx),
     - reports per-file mean spacing and uncertainty,
     - reports overall mean spacing ± SEM,
     - propagates manual picking and instrumental uncertainties,
     - plots the histogram of all spacings.

   In this mode, the original image-derived CSV files are not required.

2) Raw-data mode (for generating the JSON)
   ---------------------------------------
   If 'manual_peaks_MT2A.json' is not found, the script:

     - scans a set of base directories for band subfolders,
     - loads:
         Nuclei.csv          (to define spatial bins along POSITION_X),
         MT2A.csv            (with a 'Mean' column),
     - builds MT2A intensity vs. bin position for each band,
     - applies light 1D Gaussian smoothing,
     - opens an interactive plot (dark theme) where the user clicks
       on the MT2A peaks,
     - saves the selected peaks to 'manual_peaks_MT2A.json' in the
       script directory,
     - computes and reports the same spacing statistics as in JSON-only mode.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# Helper functions
# =============================================================================

def script_dir() -> str:
    """
    Return the directory where this script is located.

    Falls back to the current working directory when __file__ is not defined.
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def process_band(band_dir: str):
    """
    Process a single band folder to obtain MT2A intensity as a function
    of position along the band.

    The following files are used:
      - Nuclei.csv
          Required columns: 'POSITION_X', 'AREA'
          Used to define spatial bins along POSITION_X.

      - MT2A.csv
          Required columns: first column = bin index, one column 'Mean'
          Intensity is aggregated per bin.

    For each band:
      1) Nuclei are binned along POSITION_X (40 equal-width bins).
      2) Bin centers (in µm) are computed.
      3) MT2A intensity is associated with bins via 'Bin'.

    Parameters
    ----------
    band_dir : str
        Path to a band directory containing Nuclei.csv and MT2A.csv.

    Returns
    -------
    dict or None
        Dictionary with keys:
            'band'   : band label (folder name),
            'rna_df' : DataFrame with columns ['Bin_center_um', 'Mean']
        or None if the band is not usable.
    """
    nuclei_file = os.path.join(band_dir, "Nuclei.csv")
    if not os.path.exists(nuclei_file):
        print(f"Skipping {band_dir}: Missing Nuclei.csv")
        return None

    try:
        nuclei_df = pd.read_csv(nuclei_file, encoding="ISO-8859-1")
    except Exception as e:
        print(f"Error reading {nuclei_file}: {e}")
        return None

    # Column name cleanup and checks
    nuclei_df.columns = nuclei_df.columns.str.strip()
    if 'POSITION_X' not in nuclei_df.columns or 'AREA' not in nuclei_df.columns:
        print(f"Skipping {band_dir}: Required columns missing in Nuclei.csv")
        return None

    # Convert relevant columns to numeric
    nuclei_df['POSITION_X'] = pd.to_numeric(
        nuclei_df['POSITION_X'], errors='coerce'
    )
    nuclei_df['AREA'] = pd.to_numeric(
        nuclei_df['AREA'], errors='coerce'
    )
    nuclei_df = nuclei_df.dropna(subset=['POSITION_X', 'AREA'])

    # Bin nuclei along the POSITION_X axis into 40 equal-width bins
    bin_series, bin_edges = pd.cut(
        nuclei_df['POSITION_X'],
        bins=40,
        labels=False,
        retbins=True
    )
    nuclei_df['Bin'] = bin_series

    # Bin centers in physical units (µm)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_coords = pd.DataFrame(
        {'Bin': np.arange(len(bin_centers)), 'Bin_center_um': bin_centers}
    )

    def load_csv(file_path: str) -> pd.DataFrame:
        """
        Load a CSV file with 'Bin' as first column and 'Mean' as value column.

        Returns a DataFrame with columns ['Bin', 'Mean'], or an empty
        DataFrame if the file is missing or malformed.
        """
        if not os.path.exists(file_path):
            return pd.DataFrame()
        try:
            df_local = pd.read_csv(file_path, encoding="ISO-8859-1")
        except Exception as e_local:
            print(f"Error reading {file_path}: {e_local}")
            return pd.DataFrame()

        df_local.columns = df_local.columns.str.strip()
        # Treat the first column as 'Bin'
        df_local = df_local.rename(columns={df_local.columns[0]: 'Bin'})
        df_local['Bin'] = pd.to_numeric(df_local['Bin'], errors='coerce')

        if 'Mean' not in df_local.columns:
            print(f"{file_path} missing 'Mean' column.")
            return pd.DataFrame()

        return df_local[['Bin', 'Mean']].dropna()

    # Load MT2A intensity data
    rna_df = load_csv(os.path.join(band_dir, "MT2A.csv"))

    if rna_df.empty:
        return None

    # Attach physical bin centers to MT2A mean intensity
    rna_merged = pd.merge(rna_df, bin_coords, on='Bin', how='left')
    df_rna = rna_merged[['Bin_center_um', 'Mean']].dropna()

    return {'band': os.path.basename(band_dir), 'rna_df': df_rna}


def sem(a):
    """
    Standard error of the mean for a 1D array-like input.

    Returns 0.0 if fewer than two finite values are present.
    """
    a = np.array(a, float)
    a = a[np.isfinite(a)]
    return np.nanstd(a, ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0


# =============================================================================
# JSON-only mode: load existing peaks and report statistics
# =============================================================================

def summarize_from_json(json_path: str):
    """
    Load manually picked peaks from JSON and compute spacing statistics.

    Parameters
    ----------
    json_path : str
        Path to 'manual_peaks_MT2A.json'.

    The JSON must map band labels to a list of peak positions in µm:
        {
          "band_01": [x1, x2, ...],
          "band_02": [...],
          ...
        }
    """
    with open(json_path) as f:
        stored_peaks = json.load(f)

    all_diffs = []
    band_means = []

    # Loop over all bands in the JSON file
    for band_label in sorted(stored_peaks.keys()):
        peaks = np.array(stored_peaks[band_label], dtype=float)
        peaks = np.sort(peaks)

        if peaks.size >= 2:
            diffs = np.diff(peaks)
            all_diffs.extend(diffs)
            band_means.append(float(np.mean(diffs)))

    if not band_means:
        print("No band with at least two peaks found in JSON.")
        return

    # Summary across bands
    n_files = len(band_means)
    overall_mean = float(np.mean(band_means))
    sd_files = np.std(band_means, ddof=1) if n_files > 1 else 0.0
    sem_files = sd_files / np.sqrt(n_files) if n_files > 1 else 0.0

    print(f"Peaks loaded from: {json_path}")
    print(f"Number of files considered: {n_files}")
    print(
        f"Mean period across files: "
        f"{overall_mean:.2f} µm ± {sem_files:.2f} µm (SEM)"
    )

    # Additional uncertainty components (manual and instrumental)
    sigma_manual = 20.0  # µm (manual picking jitter)
    sigma_instr = 10.0   # µm (instrumental)
    sigma_total = np.sqrt(sem_files**2 + sigma_manual**2 + sigma_instr**2)

    print(f"Manual picking error: ±{sigma_manual:.1f} µm")
    print(f"Instrumental error: ±{sigma_instr:.1f} µm")
    print(f"Combined total uncertainty: ±{sigma_total:.2f} µm")

    # Histogram of all peak-to-peak spacings (dark background)
    plt.style.use("dark_background")
    plt.figure(figsize=(4, 4), facecolor="black")
    plt.hist(
        all_diffs,
        bins=15,
        edgecolor='white',
        color='white',
        alpha=0.9
    )
    plt.title('Distribution', fontsize=35, color='white')
    plt.xlim(150, 700)
    plt.xlabel('Period (µm)', fontsize=30, color='white')
    plt.ylabel('Frequency', fontsize=30, color='white')
    plt.xticks([200, 400, 600], fontsize=20, color='white')
    plt.yticks([0, 20, 40], fontsize=20, color='white')
    plt.tight_layout()
    plt.show()
    plt.style.use("default")


# =============================================================================
# Main script
# =============================================================================

def main():
    here = script_dir()
    peaks_file = os.path.join(here, "manual_peaks_MT2A.json")

    # If a JSON with manually picked peaks exists, use it directly
    if os.path.exists(peaks_file):
        print(f"Using stored peaks from: {peaks_file}")
        summarize_from_json(peaks_file)
        return

    # -------------------------------------------------------------------------
    # No JSON present: process raw band folders and enter interactive mode
    # -------------------------------------------------------------------------
    base_dirs = [
        r"C:/Users/Genesis/Desktop/MT2A/1/",
        r"C:/Users/Genesis/Desktop/MT2A/2/",
        r"C:/Users/Genesis/Desktop/YAP-MT2A/1/",
        r"C:/Users/Genesis/Desktop/YAP-MT2A/Hoechst",
        r"C:/Users/Genesis/Desktop/YAP-MT2A/DAPI"
    ]

    band_dirs = []
    for bd in base_dirs:
        if os.path.isdir(bd):
            for d in os.listdir(bd):
                full = os.path.join(bd, d)
                if os.path.isdir(full):
                    band_dirs.append(full)
        else:
            print(f"Base directory {bd} does not exist.")

    # Process all band folders and keep those with valid MT2A data
    results = []
    for bd in band_dirs:
        res = process_band(bd)
        if res and not res['rna_df'].empty:
            results.append(res)
            print(f"Processed band: {res['band']}")
        else:
            print(f"Skipped band: {os.path.basename(bd)}")

    if not results:
        raise RuntimeError("No valid MT2A data found in any band folders.")

    print(f"Total bands analyzed: {len(results)}")

    # Interactive peak picking
    gaussian_sigma = 1.0
    stored_peaks = {}
    all_diffs = []
    band_means = []

    for band in results:
        df_rna = band['rna_df']
        if df_rna.empty:
            continue

        # Sort by position and prepare raw and smoothed profiles
        df_rna = df_rna.sort_values('Bin_center_um')
        x = df_rna['Bin_center_um'].values
        y_raw = df_rna['Mean'].values
        y_smooth = gaussian_filter1d(y_raw, sigma=gaussian_sigma)

        key = band['band']

        # Dark-background interactive plot for peak selection
        plt.style.use("dark_background")
        plt.figure(figsize=(12, 5), facecolor="black")
        plt.plot(x, y_raw, color='white', alpha=0.4, linewidth=4, label='Raw')
        plt.plot(x, y_smooth, color='white', linewidth=4, label='Smooth')
        plt.title(f"Select peaks: {key}", fontsize=16, color='white')
        plt.xlabel("Position (µm)", color='white')
        plt.ylabel("Intensity", color='white')
        plt.legend()

        # Optional bounds for visual consistency
        plt.xlim(60, 2000)
        plt.xticks([500, 1000, 1500, 2000], color='white')
        plt.yticks(color='white')
        plt.tight_layout()
        plt.show(block=False)

        # User clicks all visible peaks; right-click or Enter to finish
        pts = plt.ginput(n=-1, timeout=0)
        plt.close()
        plt.style.use("default")

        peaks = np.sort([p[0] for p in pts])
        stored_peaks[key] = peaks.tolist()

        if peaks.size >= 2:
            diffs = np.diff(peaks)
            all_diffs.extend(diffs)
            band_means.append(float(np.mean(diffs)))

    # Save all manually selected peaks to JSON in the script directory
    with open(peaks_file, 'w') as f:
        json.dump(stored_peaks, f, indent=4)
    print(f"Saved peaks to {peaks_file}")

    # If at least one band has valid spacing data, summarize it
    if band_means:
        n_files = len(band_means)
        overall_mean = float(np.mean(band_means))
        sd_files = np.std(band_means, ddof=1) if n_files > 1 else 0.0
        sem_files = sd_files / np.sqrt(n_files) if n_files > 1 else 0.0

        print(f"Number of files considered: {n_files}")
        print(
            f"Mean period across files: "
            f"{overall_mean:.2f} µm ± {sem_files:.2f} µm (SEM)"
        )

        sigma_manual = 20.0  # µm (manual picking)
        sigma_instr = 10.0   # µm (instrumental)
        sigma_total = np.sqrt(sem_files**2 + sigma_manual**2 + sigma_instr**2)

        print(f"Manual picking error: ±{sigma_manual:.1f} µm")
        print(f"Instrumental error: ±{sigma_instr:.1f} µm")
        print(f"Combined total uncertainty: ±{sigma_total:.2f} µm")

        # Histogram of all peak-to-peak spacings
        plt.style.use("dark_background")
        plt.figure(figsize=(4, 4), facecolor="black")
        plt.hist(
            all_diffs,
            bins=15,
            edgecolor='white',
            color='white',
            alpha=0.9
        )
        plt.title('Distribution', fontsize=35, color='white')
        plt.xlim(150, 700)
        plt.xlabel('Period (µm)', fontsize=30, color='white')
        plt.ylabel('Frequency', fontsize=30, color='white')
        plt.xticks([200, 400, 600], fontsize=20, color='white')
        plt.yticks([0, 20, 40], fontsize=20, color='white')
        plt.tight_layout()
        plt.show()
        plt.style.use("default")
    else:
        print("No peak-to-peak data to summarize.")


if __name__ == "__main__":
    main()
