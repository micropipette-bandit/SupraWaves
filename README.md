# SupraWaves – Chapter 3 analysis code

This repository contains the analysis scripts used for **Chapter 3** of the PhD work on *Self-sustained velocity waves and pattern emergence in tissues*.

The scripts are written in Python and are organised by section:

- **PIV analysis of tissue-level standing waves**
- **RNA-seq / sequencing quality control panels**
- **Nuclear area dynamics & autocorrelation analyses**
- **UV illumination, velocity vs area comparison, and MT2A spatial patterning**

The code is designed so that readers can either:
1. Reproduce the **summary statistics and figures** from shared cache / JSON / CSV files, or  
2. Point the scripts to their **own raw data** (MATLAB `.mat`, `*-sorted.csv`, etc.) and regenerate all intermediate products.

---

## 1. Requirements

Tested with:

- **Python** ≥ 3.9  
- Python packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `h5py`
  - `mat73` *(optional, for MATLAB v7.3 files in `Chapter3-1.py`)*

Install via:

```bash
pip install numpy pandas matplotlib scipy h5py mat73
```

or using your preferred environment manager.

---

## 2. Repository structure

Main scripts included:

- `Chapter3-1.py`  
- `Chapter3-2A.py`  
- `Chapter3-2B.py`  
- `Chapter3-3A.py`  
- `Chapter3-3B.py`  
- `Chapter3-3C.py`  
- `Chapter3-3D.py`  
- `Chapter3-3E.py`  

Each script is a standalone analysis for a specific part of Chapter 3. Path variables (e.g. `base`, `base_dir`, file names) are defined near the top of each file and should be edited to match your system if you want to run them on raw data.

---

## 3. Script overview & usage

### 3.1 `Chapter3-1.py` – PIV analysis of confined epithelial monolayers

**Purpose**

High-level analysis of **standing waves from PIV data** in banded monolayers. It:

- Scans band folders containing MATLAB `.mat` files with a `Line` struct (fields `PIVPeaks`, `PIV`, etc.).
- Extracts:
  - Spatial and temporal periods (`s`, `t`)
  - Phase velocity (`v = ds/dt`)
  - RMS velocity and peak |Vx|.
- Pools metrics across bands and generates:
  - Histograms of spatial/temporal periods, velocities, RMS, peak |Vx|.
  - Summary statistics per band and globally.
  - Metrics vs band length (Lx).
  - Standing-wave occurrence plot (with Wilson confidence intervals).
  - A table-only figure with counts per band length.

**Reproducible (cache-only) mode**

The script is designed around a cache file:

- Cache file: `Sample-data.pkl`, stored next to the script.
- Main entry point: `analyse(base, cache_file=None, use_cache_only=True)`.

For pure reproduction of the reported numbers:

1. Place `Chapter3-1.py` and the provided `Sample-data.pkl` in the same directory.
2. Run:

   ```bash
   python Chapter3-1.py
   ```

   By default it calls:

   ```python
   base = r"C:/Users/Genesis/Desktop/PIV-DATA"
   analyse(base)
   ```

   with `use_cache_only=True`, so **raw `.mat` files are not touched**. All statistics and figures are generated from the cache.

**Raw-data mode (optional)**

If you want to re-analyse your own `.mat` files:

1. Edit the `base` path in `main()` to your PIV directory.
2. Call `analyse(base, use_cache_only=False)` once to build a new cache.
3. Re-run with `use_cache_only=True` for stable, reproducible output.

---

### 3.2 `Chapter3-2A.py` – Per-sequence GC content QC panel

**Purpose**

Plot **per-sequence GC content distributions** for all libraries from a MultiQC export.

- Input: a TSV file such as `fastqc_per_sequence_gc_content-plot.tsv` from MultiQC.
- The script:
  - Loads GC bins and per-sample distributions.
  - Renames columns to experimental labels (e.g. `TB1_R1`, `TC2_R2`, `C+3_R1`, etc.).
  - Plots all GC content curves on a single panel.

**Usage**

1. Update the path at the top:

   ```python
   tsv_path = r".../fastqc_per_sequence_gc_content-plot.tsv"
   ```

2. Run:

   ```bash
   python Chapter3-2A.py
   ```

This produces the GC content QC figure for all R1/R2 libraries.

---

### 3.3 `Chapter3-2B.py` – Per-base Phred quality score QC panel

**Purpose**

Boxplot of **per-base Phred quality scores** for each read set (R1 and R2 for all samples).

- Input: `fastqc_per_base_sequence_quality_plot.tsv` (MultiQC export).
- The script:
  - Drops the position column.
  - Renames 24 columns as `TC1_R1`, `TC1_R2`, …, `C-3_R2`.
  - Creates a compact boxplot figure with one box per read-set.

**Usage**

1. Update:

   ```python
   tsv_path = r".../fastqc_per_base_sequence_quality_plot.tsv"
   ```

2. Run:

   ```bash
   python Chapter3-2B.py
   ```

This reproduces the per-base quality QC panel.

---

### 3.4 `Chapter3-3A.py` – AREA kymograph autocorrelation & peak picker

**Purpose**

Analyse **spatiotemporal periodicity of nuclear area (AREA)** using kymographs and autocorrelations. Two modes are provided.

**1) JSON-only mode (publication / reproducibility)**

If a file called `manual_peaks.json` is present in the same directory:

- The script:
  - Loads the pre-picked peaks (temporal & spatial).
  - Prints per-file peak positions and spacings (Δt, Δx).
  - Prints global mean ± SEM for temporal and spatial spacings.

This mode **does not require any CSVs** and is what readers should use to reproduce the reported numbers.

**2) Interactive mode (for generating the JSON)**

If `manual_peaks.json` is missing:

- The script:
  - Scans a `base` directory for `*-sorted.csv` files.
  - Builds AREA kymographs (time × position).
  - Computes temporal & spatial autocorrelations with light/heavy smoothing.
  - Opens interactive plots where the user clicks on autocorrelation peaks.
  - Saves all picks and derived statistics to `manual_peaks.json`.
  - Prints per-file and global summaries.

**Usage**

- To **reproduce** published stats:

  ```bash
  # In a folder containing Chapter3-3A.py and manual_peaks.json
  python Chapter3-3A.py
  ```

- To generate `manual_peaks.json` from your own data:
  1. Edit `base` in `main()` to point to your `*-sorted.csv` files.
  2. Run `python Chapter3-3A.py` and click the peaks when prompted.

---

### 3.5 `Chapter3-3B.py` – PIV periods under phase-contrast vs UV illumination

**Purpose**

Compare **temporal and spatial periods** (Period, Λ) of SupraWaves under different illumination conditions (phase contrast / brightfield vs UV).

The script:

- Loads an Excel file (`PIV-comparison.xlsx`) with columns:
  - `Exp` (experiment ID)
  - `Light` (`BF`, `PC`, `UV`)
  - `Period` (temporal period)
  - `Lambda` (spatial period).
- Recodes `BF` → `PC` for analysis.
- Computes per-experiment mean ± SEM for each condition.
- Builds **paired PC–UV datasets** for both Period and Λ.
- Runs paired tests and generates publication-style boxplots.

**Usage**

1. Update the path at the top:

   ```python
   df = pd.read_excel(
       r".../PIV-comparison.xlsx",
       sheet_name="Sheet1",
       usecols=['Exp', 'Light', 'Period', 'Lambda']
   )
   ```

2. Run:

   ```bash
   python Chapter3-3B.py
   ```

This prints summary statistics and produces comparison plots for PC vs UV.

---

### 3.6 `Chapter3-3C.py` – Paired comparison: AREA-based vs velocity-based spatial periods

**Purpose**

Paired statistical comparison between **spatial periods estimated from nuclear area** and **spatial periods from PIV velocity fields**.

The script:

- Defines arrays `area` and `velocity` (spatial periods in µm).
- Interprets `0` in `velocity` as “no periodicity detected” (converted to `NaN` and dropped).
- Builds a paired DataFrame and runs:
  - Paired t-test
  - Wilcoxon signed-rank test.
- Plots a **paired boxplot** with connecting lines for each tissue.

**Usage**

The current version uses hard-coded arrays (as used in the manuscript). To reproduce:

```bash
python Chapter3-3C.py
```

To adapt to your own data, replace the `area = [...]` and `velocity = [...]` arrays at the top.

---

### 3.7 `Chapter3-3D.py` – Clustered heatmap of mean nuclear area vs time

**Purpose**

Clustered **black-and-white heatmap** of mean nuclear area as a function of time, with a dendrogram showing hierarchical clustering of samples. Two modes are provided.

**1) Cached mode (recommended for reproduction)**

If `nuclear_area_matrix.csv` is present next to the script:

- The file is read into a DataFrame where:
  - Index = time axis (e.g. `"Time (min)"`).
  - Columns = samples.
- The script performs hierarchical clustering (correlation distance, average linkage) and renders:
  - Left: dendrogram with all branches in black.
  - Right: grayscale heatmap (black = high area, white = low area).

**2) Raw-data mode (cache generation)**

If the cache is missing:

- The script:
  - Scans `base_dir` for `*-sorted.csv`.
  - Builds mean nuclear AREA vs time per file.
  - Aligns series on a common time axis (interpolation + ffill/bfill).
  - Drops degenerate series.
  - Saves `nuclear_area_matrix.csv` next to the script.
  - Runs the same clustering & plotting as above.

**Usage**

- To **reproduce** the panel from a shared `nuclear_area_matrix.csv`:

  ```bash
  python Chapter3-3D.py
  ```

- To regenerate from raw CSV files:
  1. Edit `base_dir`, `file_glob`, and `minutes_per_frame` at the top.
  2. Delete `nuclear_area_matrix.csv` (if present).
  3. Run `python Chapter3-3D.py` once to build the cache.

---

### 3.8 `Chapter3-3E.py` – MT2A peak spacing along bands

**Purpose**

Quantify **spatial periodicity of MT2A intensity** along bands and propagate manual & instrumental uncertainties. Two modes are provided.

**1) JSON-only mode (recommended for sharing / publication)**

If `manual_peaks_MT2A.json` is present next to the script:

- Loads manually picked peak positions per band.
- Computes peak-to-peak spacings (Δx) for each band.
- Reports:
  - Per-file mean spacing ± SEM.
  - Overall mean spacing across bands ± SEM.
  - Additional uncertainty components:
    - Manual picking error (e.g. ±20 µm).
    - Instrumental error (e.g. ±10 µm).
  - Combined total uncertainty.
- Plots a histogram of all peak-to-peak spacings (dark background).

**2) Raw-data interactive mode (for generating the JSON)**

If `manual_peaks_MT2A.json` is not found:

- The script:
  - Scans a list of base directories for band subfolders.
  - For each band, uses:
    - `Nuclei.csv` (`POSITION_X`, `AREA`) to define 40 spatial bins.
    - `MT2A.csv` with `Bin` + `Mean` intensity.
  - Builds MT2A intensity vs positional bin center (µm).
  - Applies light Gaussian smoothing.
  - Opens an interactive dark-themed plot where the user clicks all visible MT2A peaks.
  - Stores peaks for each band in `manual_peaks_MT2A.json`.
  - Computes and prints the same spacing and uncertainty statistics.
  - Displays a histogram of all spacings.

**Usage**

- To **reproduce** spacing statistics and the histogram from the JSON:

  ```bash
  python Chapter3-3E.py
  ```

- To generate `manual_peaks_MT2A.json` from raw band folders:
  1. Edit `base_dirs` at the bottom of the script to point to your band directories.
  2. Ensure each band folder contains `Nuclei.csv` and `MT2A.csv`.
  3. Run `python Chapter3-3E.py` and click all MT2A peaks when prompted.

---

## 4. Data & paths

The scripts contain **absolute paths** corresponding to the original analysis environment (e.g. `C:/Users/...`). For portability:

- When using **provided cache/JSON/CSV** files:
  - Place these files in the same directory as their corresponding script.
  - Leave `base` / `base_dir` unused (cache/JSON modes do not access raw data).

- When adapting to **your own datasets**:
  - Edit path variables at the top of each script:
    - `base`, `base_dir`
    - File paths like `tsv_path`, Excel file paths, etc.
  - Keep the expected column names / formats (`FRAME`, `AREA`, `POSITION_X`, `Mean`, etc.) consistent.

---

## 5. Reproducibility philosophy

Several scripts implement a two-step workflow:

1. **Interactive / raw-data mode**  
   Used only by the authors to build:
   - `Sample-data.pkl`
   - `nuclear_area_matrix.csv`
   - `manual_peaks.json`
   - `manual_peaks_MT2A.json`.

2. **Cache / JSON-only mode**  
   Intended for **readers and reviewers**, who can:
   - Download the script + its cache/JSON file.
   - Run the script without any raw microscopy / PIV / sequencing files.
   - Reproduce all **reported summary statistics and plots**.

This makes the analysis auditable without requiring large imaging datasets.

---

## 6. Citation

If you use this code in a publication, please cite the associated thesis (details to be completed once the manuscript is publicly available), and reference this repository as the source of the Chapter 3 analysis scripts.

A generic placeholder citation:

> G. Marquez-Vivas, *Self-sustained velocity waves and pattern emergence in tissues*, PhD thesis, Université Grenoble Alpes, 2025.

---

## 7. License

```text
MIT License

Copyright (c) 2025 Genesis Marquez-Vivas

Permission is hereby granted, free of charge, to any person obtaining a copy
...
```

---

## 8. Contact

For questions about the code or data, please contact:

- **Name:** Génésis Marquez-Vivas  
- **Affiliation:** Laboratoire Interdisciplinaire de Physique
- **Role:** PhD candidate (SupraWaves project)
